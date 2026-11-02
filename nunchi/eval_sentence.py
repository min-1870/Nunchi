"""Shared decision helpers for the stream / sentence taggers, and a small evaluation on our own sets.

Each word is decided online from the words typed so far:
  now    : no word after it (what an IME that commits at Space has)
  hybrid : now when its best path is >= the act threshold, else after one more word
A word whose best path changes language inside (GitHub에) is split when that path is >= SPLIT_T;
otherwise the whole word goes to the likelier of all-ko / all-en.

    .venv-rocm/Scripts/python -m nunchi.eval_sentence      (NUNCHI_MODEL=stream for the stream tagger)
"""
import os
import random
from collections import Counter

from nunchi.context import build, real_mixed
from nunchi.demo import compose
from nunchi.dubeolsik import to_keys
from nunchi.make_noise import with_typos
from nunchi.tagger import KO
from nunchi.train_fasttext import ACT_T

if os.environ.get("NUNCHI_MODEL") == "stream":  # the one-line Transformer
    from nunchi.stream_tagger import decide, load_model
else:
    from nunchi.sentence_tagger import decide, load_model

# Defaults for models without their own values in models/registry.json (nunchi.models.load sets
# model.split_threshold / model.act_threshold). Swept on the dev sets for TX3: 0.9 -> 0.7 splits 38 more
# English+Korean words right and 1 more single-language word wrongly (4 more with typos)
SPLIT_T = 0.7


def run_sentences(model, sents):
    """sents: [[keys]] -> per sentence, per word: {'now': r, 'lag': r, 'hybrid': r}, r = decide() result."""
    pre, tgt, where = [], [], []
    for si, s in enumerate(sents):
        for i in range(len(s)):
            pre.append(s[:i + 1]); tgt.append(i); where.append((si, i, "now"))
            if i + 1 < len(s):
                pre.append(s[:i + 2]); tgt.append(i); where.append((si, i, "lag"))
    res = decide(model, pre, tgt)
    out = [[{} for _ in s] for s in sents]
    for (si, i, kind), r in zip(where, res):
        out[si][i][kind] = r
    act_t = getattr(model, "act_threshold", None) or ACT_T  # the model's own, if the registry has one
    for s in out:
        for w in s:
            w.setdefault("lag", w["now"])
            w["hybrid"] = w["now"] if w["now"][1] >= act_t else w["lag"]
    return out


def verdict(r, split_t=None):
    """-> ('split', parts) or ('ko'|'en', None). split_t: the model's own threshold (model.split_threshold)."""
    path, pb, pko, pen = r
    if len(set(path)) > 1 and pb >= (split_t or SPLIT_T):
        return "split", path
    return ("ko" if pko > pen else "en"), None


def text_of(keys, r):
    """The text a decision types for these keys."""
    lang, path = verdict(r)
    if lang == "split":
        path = list(path) + [path[-1]] * (len(keys) - len(path))  # words longer than MAX_KEYS: path is cut
        parts, start = [], 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or path[i] != path[start]:
                k = keys[start:i]
                parts.append(compose(k) if path[start] == KO else k)
                start = i
        return "".join(parts)
    return compose(keys) if lang == "ko" else keys


def main():
    model = load_model()
    sets = build()["test"]
    sets["real"] = real_mixed()[0]
    keyed = {kind: [[(lab, to_keys(w)) for lab, w in s] for s in sents] for kind, sents in sets.items()}
    rnd = random.Random(7)
    for kind in ("ko", "en", "real"):
        keyed[kind + "+typo"] = with_typos(keyed[kind], rnd)
    print(f"{'set':<11}{'words':>7} | {'now':>6} {'hybrid':>6} {'lag':>6}  (wrong words)")
    for kind, sents in keyed.items():
        res = run_sentences(model, [[k for _, k in s] for s in sents])
        c = Counter()
        for s, r in zip(sents, res):
            for (lab, _), w in zip(s, r):
                c["n"] += 1
                for pol in ("now", "hybrid", "lag"):
                    c[pol] += verdict(w[pol])[0] != lab
        print(f"{kind:<11}{c['n']:>7} | {c['now']:>6} {c['hybrid']:>6} {c['lag']:>6}")


if __name__ == "__main__":
    main()
