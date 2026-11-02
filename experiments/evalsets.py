"""Evaluation for the context-length experiment (stage 2 of the plan).

Sets (every word decided online, hybrid policy: act now if confident, else after one more word):
  nsmc_dev / nsmc_test : NSMC real mixed reviews, split in half by sentence hash
  web_dev / web_test   : mC4 validation mixed sentences, split the same way
  guards               : synthetic test sets (pure ko, pure en, ko + English, en + Korean) and English +
                         particle words (path accuracy) -- must not get clearly worse
Primary metric: mean of the nsmc and web error rates (wrong words / words) on the dev halves.
Paired bootstrap over sentences (each set resampled on its own) gives the 95% range of A - B.
"""
import hashlib
import json
import random
import sys
import time

import torch

import nunchi.eval_sentence as es
import nunchi.stream_tagger as st
from nunchi.context import build, real_mixed
from nunchi.dubeolsik import to_keys
from nunchi.tagger import device, mixed_rows
from nunchi.web_mixed import web_mixed

LOG = None


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def half(sents, which):
    return [s for s in sents
            if int(hashlib.md5(" ".join(w for _, w in s).encode()).hexdigest(), 16) % 2 == which]


_TRAIN_TEXT = None


def in_training(sent):
    """True if the sentence (its words, Latin+Hangul tokens dropped) is also a training sentence."""
    global _TRAIN_TEXT
    if _TRAIN_TEXT is None:
        from nunchi.sentence_data import MIXED
        _TRAIN_TEXT = set()
        # web50: the bigger mC4 pool the long models train on; *_v2: the TR2 pools (numbers and Hangul+Latin
        # kept), compared with the tokens the eval sets keep (Hangul and Latin only)
        for name in ("nsmc", "kcbert", "web", "web50", "nsmc_v2", "kcbert_v2", "web50_v2", "nsmc_v3", "kcbert_v3", "web50_v3"):
            if not (ROOT_SENT / f"{name}.txt").exists():
                continue
            with open(ROOT_SENT / f"{name}.txt", encoding="utf-8") as f:
                _TRAIN_TEXT.update(" ".join(t for t in line.split() if HANGUL_RE.match(t) or LATIN_RE.match(t))
                                   for line in f)
    return " ".join(w for _, w in sent) in _TRAIN_TEXT


def load_sets(split, drop_leaked=False):
    which = 0 if split == "dev" else 1
    sets = {"nsmc": half(real_mixed()[0], which), "web": half(web_mixed(), which)}
    if drop_leaked:
        sets = {k: [s for s in v if not in_training(s)] for k, v in sets.items()}
    if split == "dev":
        sets.update({f"guard_{k}": v for k, v in build()["test"].items()})
    return {k: [[(lab, to_keys(w)) for lab, w in s] for s in v] for k, v in sets.items()}


# ---- sets whose words carry per-key labels (English + particle words must come out split exactly)
from nunchi.sentence_data import OUT as ROOT_SENT  # noqa: E402
from nunchi.sentence_data import HANGUL as HANGUL_RE, LATIN as LATIN_RE, MIXED as MIXED_RE  # noqa: E402

KEYED = {"particle": "web_particle_eval.txt", "trans_en": "trans_en_eval.txt", "trans_ko": "trans_ko_eval.txt"}


_WEB_KEYS = None


def load_keyed(split, drop_leaked=False):
    """drop_leaked: leave out English+particle sentences that also sit in the mC4 training pool (the same
    sentence repeats across sites); the translation sets come from held-out sentences already."""
    global _WEB_KEYS
    if drop_leaked and _WEB_KEYS is None:
        from nunchi.sentence_tagger import word
        _WEB_KEYS = set()
        for name in ("web.txt", "web50.txt", "web50_v2.txt", "web50_v3.txt"):
            if not (ROOT_SENT / name).exists():
                continue
            with open(ROOT_SENT / name, encoding="utf-8") as f:
                _WEB_KEYS.update("\t".join(word(t)[0] for t in line.split()
                                           if HANGUL_RE.match(t) or LATIN_RE.match(t) or MIXED_RE.match(t))
                                 for line in f)
    which = 0 if split == "dev" else 1
    out = {}
    for name, fname in KEYED.items():
        rows = []
        for line in open(ROOT_SENT / fname, encoding="utf-8"):
            line = line.rstrip("\n")
            if line and int(hashlib.md5(line.encode()).hexdigest(), 16) % 2 == which:
                row = [(k, [int(c) for c in y]) for k, y in (w.rsplit(":", 1) for w in line.split("\t"))]
                if drop_leaked and name == "particle" and "\t".join(k for k, _ in row) in _WEB_KEYS:
                    continue
                rows.append(row)
        out[name] = rows
    return out


def evaluate_keyed(model, sets):
    """-> {set: {'mixed': [n, wrong], 'plain': [n, wrong], 'switch': [n, wrong]}} (hybrid policy)."""
    es.decide = st.decide
    out = {}
    for name, sents in sets.items():
        rs = es.run_sentences(model, [[k for k, _ in s] for s in sents])
        c = {"mixed": [0, 0], "plain": [0, 0], "switch": [0, 0]}
        for s, r in zip(sents, rs):
            for i, ((k, y), w) in enumerate(zip(s, r)):
                v = es.verdict(w["hybrid"])
                if len(set(y)) > 1:
                    kind, ok = "mixed", v[0] == "split" and v[1] == y
                else:
                    kind, ok = "plain", v[0] == ("ko" if y[0] == 1 else "en")
                c[kind][0] += 1
                c[kind][1] += not ok
                if i > 0 and s[i - 1][1][-1] != y[0]:
                    c["switch"][0] += 1
                    c["switch"][1] += not ok
        out[name] = c
    return out


def load(path, left):
    m = st.StreamTagger(left=left)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device()).eval()


def evaluate(model, sets):
    """-> {set: [[per sentence: list of (label, predicted, is_switch)]]} plus particle accuracy."""
    es.decide = st.decide
    out = {}
    for name, sents in sets.items():
        t = time.time()
        rs = es.run_sentences(model, [[k for _, k in s] for s in sents])
        out[name] = [[(lab, es.verdict(w["hybrid"])[0], i > 0 and s[i - 1][0] != lab)
                      for i, ((lab, _), w) in enumerate(zip(s, r))] for s, r in zip(sents, rs)]
        say(f"    {name}: {sum(map(len, sents)):,} words in {time.time() - t:.0f} s")
    rows = [[(k, y)] for k, y, _ in mixed_rows()[0]["valid"]]
    out["particle_acc"] = st.path_acc(model, rows, next(model.parameters()).device)
    return out


def summary(res):
    s = {}
    for name, sents in res.items():
        if name == "particle_acc":
            s[name] = res[name]
            continue
        words = [w for x in sents for w in x]
        sw_en = [w for w in words if w[2] and w[0] == "en"]
        sw_ko = [w for w in words if w[2] and w[0] == "ko"]
        s[name] = {"wrong": sum(l != p for l, p, _ in words), "n": len(words),
                   "ko_en_wrong": sum(l != p for l, p, _ in sw_en), "ko_en_n": len(sw_en),
                   "en_ko_wrong": sum(l != p for l, p, _ in sw_ko), "en_ko_n": len(sw_ko)}
    return s


def primary(res, idx=None):
    rates = []
    for name in ("nsmc", "web"):
        sents = res[name] if idx is None else [res[name][i] for i in idx[name]]
        rates.append(sum(l != p for x in sents for l, p, _ in x) / sum(map(len, sents)))
    return sum(rates) / 2


def bootstrap(a, b, n=2000, seed=0):
    """95% range of primary(a) - primary(b), resampling sentences (paired: same sentences for both)."""
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = {k: [rnd.randrange(len(a[k])) for _ in a[k]] for k in ("nsmc", "web")}
        diffs.append(primary(a, idx) - primary(b, idx))
    diffs.sort()
    return primary(a) - primary(b), diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def table(named):
    """named: {model name: summary} -> lines."""
    sets = [k for k in next(iter(named.values())) if k != "particle_acc"]
    lines = [f"{'model':<30}" + "".join(f"{k:>16}" for k in sets) + f"{'particle':>10}"]
    for name, s in named.items():
        lines.append(f"{name:<30}" + "".join(f"{s[k]['wrong']:>7}/{s[k]['n']:<8}" for k in sets) +
                     f"{s['particle_acc']:>10.4f}")
    lines.append("switch points (wrong/n):")
    for name, s in named.items():
        lines.append(f"  {name:<28}" + "  ".join(
            f"{k} ko->en {s[k]['ko_en_wrong']}/{s[k]['ko_en_n']}, en->ko {s[k]['en_ko_wrong']}/{s[k]['en_ko_n']}"
            for k in ("nsmc", "web")))
    return lines


def save(res, path):
    json.dump(res, open(path, "w"))


if __name__ == "__main__":
    LOG = sys.argv[1]
