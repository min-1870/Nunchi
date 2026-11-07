"""Export a stream tagger checkpoint to ONNX for the app, plus a JSON with everything outside the graph.

  graph : ids, pos, wpos, widx (B, T) int64 -> emissions (B, T, 2) float, one en/ko score pair per key
          (embeddings, the attention mask built from widx, Transformer, output layer)
  app   : key ids, trimming to MAX_T, cut() to `left` words, picking the judged word's keys,
          CRF Viterbi on them (viterbi() below is the reference), thresholds

The model's context comes from models/registry.json: a decider (TX: right = 1) judges the last word typed;
a reviser (RV: right = null, the whole window) judges the word before it, once the last word is typed; a TR
model (whole window, every word labelled) does both, so it is checked both ways and its JSON says
"revises": true.

check compares graph + viterbi() against stream_tagger.decide() on held-out sentences, and golden writes
<name>_golden.json: streams as the app builds them, with decide()'s output, for the app's own tests.

    .venv/Scripts/python -m nunchi.export_onnx <file name, e.g. tr4_51k or tx4_long> [sentences per source]
"""
import json
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from nunchi.sentence_tagger import MAX_KEYS, load_sentences, to_sentence
from nunchi.stream_tagger import DIGIT0, MAX_T, POS, SPACE, StreamTagger, cut, decide
from nunchi.stream_tagger import encode as encode_digits
from nunchi.tagger import CHARS, EN, KO
from nunchi.train_fasttext import ACT_T, MODELS

PAD_W = 10_000  # widx of padding keys: never visible
DEFAULT_THRESHOLD = 0.99  # with 0.999 for a single key, set in the app
# Per checkpoint (models/registry.json names), from the model session's dev evaluation:
#   split   : split threshold used in eval_sentence for that model
#   targets : wrong corrections per 1,000 words -> threshold; empty until swept for that model
TUNED = {
    "tx4": {"split": 0.7, "targets": {}},  # split as swept for TX3, kept by eval_sentence; no sweep yet
    # at 0.99 the model session saw RV1 fix 25 TX4 errors on dev and break none (0.9: 70 fixed, 10 broken)
    "rv1": {"split": 0.7, "targets": {}, "act": 0.99},
}
# longer runs of the same models, not in the registry yet: same context and thresholds (not re-swept)
TUNED["tx4_long"] = TUNED["tx4"]
TUNED["rv1_long"] = TUNED["rv1"]
# one model deciding and revising: the app's thresholds for both until swept for it (the model session's own
# revision rule is >= 0.95 three words after; the app revises one word after, strictly)
TUNED["tr4_51k"] = {"split": 0.7, "targets": {}, "act": 0.99, "revise_act": 0.99}
# TR5n-102K, tuned with the app rule (experiments/logs/app_tune.py, 2026-11-05): 550 settings on dev, chosen with a
# broken word (right as typed, changed by the app) counted as 2-3 missed ones; checked once on test. Dev / test wrong
# 2782 / 2814 at 0.99 / 0.99 / 0.7 -> 1550 / 1587 here; broken 122 / 86 -> 236 / 228. Costs: split words (fused,
# English + particle) and key collisions a few %p worse, every switching cell and the main sets better.
# early_act: switching in the middle of a word (the app's key()), a prefix, not yet the word (experiments/logs/
# early_tune.py): at 0.99 instead of act 0.85, 57% fewer wrong mid-word switches (좀ㅅ while typing what), fewer
# broken words and fewer wrong at Space.
TUNED["tr5n_102k"] = {"split": 0.9, "targets": {}, "act": 0.85, "revise_act": 0.8, "early_act": 0.99}

DIGITS = False  # set by load(): encode digit keys with their own ids (TR2 on)


def encode(batch, targets=None):
    return encode_digits(batch, targets, digits=DIGITS)


class Graph(nn.Module):
    """StreamTagger.forward_words up to the per-key emissions."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, ids, pos, wpos, widx):
        m = self.m
        T = ids.size(1)
        x = m.tok(ids) + m.pos(pos) + m.wpos(wpos)
        see = widx.unsqueeze(1) < PAD_W
        if m.right is not None:
            see = see & (widx.unsqueeze(1) <= widx.unsqueeze(2) + m.right)
        see = see | torch.eye(T, dtype=torch.bool, device=ids.device)
        mask = (~see).repeat_interleave(m.heads, 0)
        return m.out(m.norm(m.enc(x, mask=mask)))


def lse(a, axis):
    mx = a.max(axis, keepdims=True)
    return (mx + np.log(np.exp(a - mx).sum(axis, keepdims=True))).squeeze(axis)


def viterbi(em, trans, start, end):
    """em (L, 2) of one word -> (path, P(path), P(all ko), P(all en)); same as Tagger.decode_em."""
    score, back = start + em[0], []
    alpha = start + em[0]
    for t in range(1, len(em)):
        cand = score[:, None] + trans
        back.append(cand.argmax(0))
        score = cand.max(0) + em[t]
        alpha = lse(alpha[:, None] + trans + em[t][None, :], 0)
    score = score + end
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path = path[::-1]
    logz = lse(alpha + end, 0)

    def path_score(p):
        return start[p[0]] + em[0, p[0]] + sum(trans[p[t - 1], p[t]] + em[t, p[t]] for t in range(1, len(p))) + end[p[-1]]

    n = len(em)
    return path, np.exp(score.max() - logz), np.exp(path_score([KO] * n) - logz), np.exp(path_score([EN] * n) - logz)


def load(name):
    """The model in models/<name>.pt, built as its registry entry says (found by file name)."""
    global DIGITS
    from nunchi.models import REGISTRY
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    s = next((e for e in entries if e.get("file") == f"{name}.pt"), None)
    if s is None:  # a variant not registered under its file (tx4_long): the base model's context
        s = dict(next(e for e in entries if e["name"].startswith(name.split("_")[0].upper())), file=f"{name}.pt")
    size = dict(zip(("d", "heads", "layers", "ff"), s["size"])) if s.get("size") else {}
    model = StreamTagger(left=s["left"], right=s["right"], digits=s.get("digits", False), **size)
    model.load_state_dict(torch.load(MODELS / s["file"], map_location="cpu"))
    model.tr = s["name"].startswith("TR")
    DIGITS = model.digits
    return model.eval()


def judged_from_end(model):
    """0: the last word (a decider, or a TR model deciding); 1: the word before it (a reviser)."""
    return 1 if model.right is None and not model.tr else 0


def backs(model):
    """Which words from the end the app judges with this model."""
    return [0, 1] if model.tr else [judged_from_end(model)]


def export(name):
    model = load(name)
    left = model.left
    # two rows: torch.export fixes a dimension of size 1 to 1
    enc = encode([[(k, [0] * len(k)) for k in s] for s in [["dkssud", "so", "rkwk"], ["MBCsms", "good"]]])
    args = tuple(enc[:4])
    onnx_path = MODELS / f"{name}.onnx"
    B, T = torch.export.Dim("batch"), torch.export.Dim("keys", max=MAX_T)
    torch.onnx.export(Graph(model), args, onnx_path, dynamo=True, external_data=False,
                      input_names=["ids", "pos", "wpos", "widx"], output_names=["emissions"],
                      dynamic_shapes=({0: B, 1: T},) * 4)
    meta = {
        "model": name,
        "inputs": "ids, pos, wpos, widx: int64 (batch, keys); output emissions: float32 (batch, keys, 2)",
        "labels": {"0": "en", "1": "ko"},
        "vocab": {"pad": 0, "other": 1, "chars": CHARS, "first_char_id": 2, "space": SPACE,
                  "digit0": DIGIT0 if model.digits else None},
        "max_keys_per_word": MAX_KEYS,
        "max_stream_keys": MAX_T,
        "pos_table": POS,
        "padding_widx": PAD_W,
        "left_words": left,
        "right_words": model.right,
        "judged_from_end": judged_from_end(model),
        "revises": model.tr,
        "revise_act_threshold": TUNED[name].get("revise_act"),
        "lookahead": 0,
        "temperature": 1.0,
        "act_threshold": TUNED[name].get("act", DEFAULT_THRESHOLD),
        "early_act_threshold": TUNED[name].get("early_act"),
        "split_threshold": TUNED[name]["split"],
        "targets": TUNED[name]["targets"],
        "crf": {k: getattr(model, k).detach().tolist() for k in ("trans", "start", "end")},
    }
    (MODELS / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and {name}.json", flush=True)


def held_out(per_source, seed):
    """Held-out sentences as [[keys]], per_source from each sentence source."""
    rnd = random.Random(seed)
    sents = []
    for source, (_, valid) in load_sentences().items():
        sents += [[k for k, _ in to_sentence(line)] for line in rnd.sample(valid, min(per_source, len(valid)))]
    return sents


EDGE = [  # streams the app must encode the same way; the last word is the one judged
    ["dkssud"], ["MBCsms"], ["T"], ["gg"], ["rkwk", "so"], ["QWERTY"],
    ["dkssudgktpdy" * 3],  # longer than max_keys_per_word
    ["dhsmf", "skfTlrk", "wjdakf", "whgdk", "tksckrdmf", "gkfo", "rjtrhwjr", "rkxdk", "go"],  # cut to left_words
    ["dkssudgktpdydkss", "rjtrhwjrgkwlakfk", "tksckrdmfgkfjrkwk", "whgdkdkssudgktpd"],  # past max_stream_keys
]


def golden(name, n=400, seed=1):
    """Write <name>_golden.json: per case the words typed so far, the stream the model sees (after cut and
    trimming) and decide()'s path and probabilities for the judged word (see judged_from_end)."""
    model = load(name)
    left = model.left
    cases = []
    for back in backs(model):
        rnd = random.Random(seed)
        sents = [s for s in held_out(200, seed) if len(s) > back]
        streams = [s[:rnd.randrange(back, len(s)) + 1] for s in rnd.sample(sents, n)]
        streams += [e for e in EDGE + [["dkssud", "i", "think"], ["i", "think"]] if len(e) > back]
        res = decide(model, streams, [len(s) - 1 - back for s in streams])
        for s, (path, pb, pko, pen) in zip(streams, res):
            ids, pos, wpos, widx = (x[0].tolist() for x in encode([[(k, [0] * len(k)) for k in cut(s, back, left)]])[:4])
            cases.append({"words": s, "back": back, "ids": ids, "pos": pos, "wpos": wpos, "widx": widx,
                          "path": path, "p_best": pb, "p_ko": pko, "p_en": pen})
    out = MODELS / f"{name}_golden.json"
    out.write_text(json.dumps(cases), encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)", flush=True)

    # the app's 두벌식 composer against demo.compose: every word of the sentences, both as typed and as
    # its keys would show in Korean mode
    from nunchi.demo import compose
    keys = sorted({k for s in sents for k in s if k.isascii() and k.isalpha()})
    out = MODELS / "compose_golden.json"
    out.write_text(json.dumps([[k, compose(k)] for k in keys], ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(keys)} words)", flush=True)


def check(name, per_source=300, seed=0):
    model = load(name)
    for back in backs(model):
        print(f"judging the word {back} from the end", flush=True)
        check_back(model, name, back, per_source, seed)


def check_back(model, name, back, per_source, seed):
    import onnxruntime as ort

    left = model.left
    sess = ort.InferenceSession(str(MODELS / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    crf = [getattr(model, k).detach().numpy().astype(np.float64) for k in ("trans", "start", "end")]
    sents = held_out(per_source, seed)
    pre, tgt = [], []
    for s in sents:
        for i in range(len(s) - back):
            if back:
                pre.append(s[:i + 2]); tgt.append(i)  # revised once the next word is typed
            else:
                for j in (i + 1, i + 2)[:1 + (i + 1 < len(s))]:  # now, and with one word of lookahead
                    pre.append(s[:j]); tgt.append(i)
    print(f"{len(sents)} sentences, {len(pre)} decisions", flush=True)

    t0 = time.time()
    ref = decide(model, pre, tgt)
    print(f"torch: {time.time() - t0:.0f}s", flush=True)

    t0, same, worst, flips = time.time(), 0, 0.0, 0
    for i in range(0, len(pre), 256):
        chunk = pre[i:i + 256]
        t_end = [len(s) - 1 - t for s, t in zip(chunk, tgt[i:i + 256])]
        chunk = [cut(s, e, left) for s, e in zip(chunk, t_end)]
        ids, pos, wpos, widx, kidx, _, cm, wm = encode([[(k, [0] * len(k)) for k in s] for s in chunk], t_end)
        em = sess.run(None, {"ids": ids.numpy(), "pos": pos.numpy(), "wpos": wpos.numpy(), "widx": widx.numpy()})[0]
        for b in range(len(chunk)):
            w = int(wm[b].nonzero()[0, 0])
            n = int(cm[b, w].sum())
            got = viterbi(em[b, kidx[b, w, :n].numpy()].astype(np.float64), *crf)
            want = ref[i + b]
            same += got[0] == want[0]
            worst = max(worst, *(abs(g - x) for g, x in zip(got[1:], want[1:])))
            flips += (got[1] >= ACT_T) != (want[1] >= ACT_T)
    print(f"onnx : {time.time() - t0:.0f}s", flush=True)
    print(f"same path {same}/{len(pre)}, max prob diff {worst:.2e}, act-threshold flips {flips}", flush=True)

    # one call as the app makes it: batch 1, a full 64-key stream
    s = max((cut(p, 1, left) for p in pre), key=lambda p: sum(len(k) + 1 for k in p))
    feed = dict(zip(["ids", "pos", "wpos", "widx"], (x.numpy() for x in encode([[(k, [0] * len(k)) for k in s]])[:4])))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    one = ort.InferenceSession(str(MODELS / f"{name}.onnx"), opts, providers=["CPUExecutionProvider"])
    for _ in range(20):
        one.run(None, feed)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        one.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"latency, {feed['ids'].shape[1]} keys, 1 thread: median {times[150]:.2f} ms, p99 {times[296]:.2f} ms", flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    export(name)
    golden(name)
    check(name, *[int(a) for a in sys.argv[2:]])
