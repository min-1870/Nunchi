"""split TR3-51K's errors two ways, then rank by how often each split occurs in real typing.
  1. by what would fix them: direction (English->Korean / Korean->English / split word), fixed by its own
     revision 3 words later (no gate) or not, both readings real words (en zipf >= 3 and Korean >= 3) or not
  2. by the data: a small decision tree (gini, own code) on word features predicts "wrong at Space"; its
     leaves are the kinds. The same features on raw KcBERT / NSMC text give each leaf's frequency per 1,000
     words, and frequency x leaf error rate = expected wrong words per 1,000 typed words.
Eval words: dev halves with numbers in place, real mixed sentences + ambiguous-key targets."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from wordfreq import zipf_frequency

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import natural_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.ambiguous_eval import OUT_V3  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.sentence_data import SPLIT, ko_keys, ko_zipf, korean_lines, wrong_mode  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repository
LOG = S / "logs" / "error_tree_log.txt"
FEATS = ["truth (0 EN, 1 KO, 2 mixed)", "letters", "case (0 lower, 1 Cap, 2 CAPS, -1 Korean)", "English zipf",
         "Korean zipf", "previous (0 start, 1 KO, 2 EN)", "next (0 end, 1 KO, 2 EN)"]
_z = {}


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def zipfs(k):
    if k not in _z:
        _z[k] = (zipf_frequency(k.lower(), "en"), ko_zipf(compose(k)))
    return _z[k]


def feats(k, y, prev, nxt):
    truth = 2 if len(set(y)) > 1 else (1 if y[0] == KO else 0)
    case = -1 if truth == 1 else (2 if k.isupper() and len(k) > 1 else 1 if k[:1].isupper() else 0)
    en, ko = zipfs(k)
    lang = lambda w: 0 if w is None else (1 if w[1][-1] == KO else 2)
    return [truth, sum(c.isalpha() for c in k), case, en, ko, lang(prev), 0 if nxt is None else (1 if nxt[1][0] == KO else 2)]


# ---- eval words: features, wrong at Space, fixed by own revision 3 words later
dev_ = device()
m = st.StreamTagger(left=None, right=None, digits=True)
m.load_state_dict(torch.load(ROOT / "models" / "tr3_51k.pt", map_location="cpu"))
m = m.to(dev_).eval()
amb = []
for line in OUT_V3.open(encoding="utf-8"):
    r = json.loads(line)
    w = [(k, [int(c) for c in y]) for k, y in r["w"]]
    if int(hashlib.md5("\t".join(k for k, _ in w).encode()).hexdigest(), 16) % 2 == 0:
        amb.append((w, r["t"]))
X, Y, tags = [], [], Counter()
for data in (natural_sets.load("dev"), amb):
    where = [(si, i) for si, (_, idx) in enumerate(data) for i in idx if not data[si][0][i][0].isdigit()]
    keys = [[k for k, _ in s] for s, _ in data]
    r0 = st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])
    r3 = st.decide(m, [keys[si][:min(len(keys[si]), i + 4)] for si, i in where], [i for _, i in where])
    for (si, i), a, b in zip(where, r0, r3):
        s = data[si][0]
        k, y = s[i]
        x = feats(k, y, s[i - 1] if i else None, s[i + 1] if i + 1 < len(s) else None)
        bad = text_of(k, a) != expected(k, y)
        X.append(x); Y.append(bad)
        if bad:
            d = ["English typed, became Korean", "Korean typed, became English", "mixed word split wrong"][x[0]]
            later = "fixed 3 words later" if text_of(k, b) == expected(k, y) else "not fixed later"
            both = "both readings real words" if x[3] >= 3 and x[4] >= 3 else "one reading only"
            tags[(d, later, both)] += 1
X, Y = np.array(X, float), np.array(Y, float)
n_err = int(Y.sum())
say(f"=== 1. TR3-51K errors by what would fix them ({n_err} errors of {len(Y)} eval words, dev)")
for (d, later, both), c in sorted(tags.items(), key=lambda t: -t[1]):
    say(f"  {d:<32} {later:<22} {both:<26} {c:>4} ({c / n_err:.0%})")


# ---- 2. decision tree
def gini(y):
    p = y.mean() if len(y) else 0
    return 2 * p * (1 - p)


def grow(idx, depth, rule, leaves, min_leaf=40, max_depth=4):
    y = Y[idx]
    best = None
    if depth < max_depth and y.sum() >= 5:
        for f in range(X.shape[1]):
            vals = np.unique(X[idx, f])
            cuts = vals if len(vals) < 20 else np.quantile(X[idx, f], np.linspace(0.05, 0.95, 19))
            for c in np.unique(cuts):
                left = idx[X[idx, f] <= c]
                right = idx[X[idx, f] > c]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                g = (len(left) * gini(Y[left]) + len(right) * gini(Y[right])) / len(idx)
                if best is None or g < best[0]:
                    best = (g, f, c, left, right)
    if best is None or best[0] >= gini(y) - 1e-4:
        leaves.append((rule, idx))
        return
    _, f, c, left, right = best
    grow(left, depth + 1, rule + [(f, "<=", c)], leaves)
    grow(right, depth + 1, rule + [(f, ">", c)], leaves)


leaves = []
# the eval sets hold far more English than typing does: split by truth first, so each leaf's rate is the rate
# of one truth only and multiplying by raw-text frequency is fair
for t in (0, 1, 2):
    grow(np.where(X[:, 0] == t)[0], 1, [(0, "<=", t), (0, ">", t - 1)], leaves)


def match(rule, x):
    return all((x[f] <= c) if op == "<=" else (x[f] > c) for f, op, c in rule)


def show(rule):
    truth = {0: "English", 1: "Korean", 2: "mixed word"}[int(rule[0][2])]
    rest = " & ".join(f"{FEATS[f].split(' (')[0]} {op} {c:g}" for f, op, c in rule[2:])
    return f"[{truth}] {rest or 'all'}"


# ---- raw-text frequency of each leaf
kk = ko_keys()
raw = {}
for src, n_max in (("nsmc", 150_000), ("kcbert", 300_000)):
    cnt, words = Counter(), 0
    for li, text in enumerate(korean_lines(src)):
        if li >= n_max:
            break
        toks = [t for t in filter(None, SPLIT.split(text)) if re.fullmatch(r"[가-힣ㄱ-ㅣA-Za-z]+", t)]
        if not any(re.match(r"[가-힣]", t) for t in toks):
            continue
        sent = []
        for t in toks:
            if re.fullmatch(r"[가-힣ㄱ-ㅣ]+", t):
                k = to_keys(t); sent.append((k, [KO] * len(k)))
            elif re.fullmatch(r"[A-Za-z]+", t):
                sent.append((t, [KO if wrong_mode(t, kk) else 0] * len(t)))  # wrong-mode Latin: Korean meant
            else:
                mm = re.match(r"^([A-Za-z]+)([가-힣]+)$", t) or re.match(r"^([가-힣]+)([A-Za-z]+)$", t)
                if not mm:
                    continue
                a, b = mm.group(1), mm.group(2)
                ka = a if a.isascii() else to_keys(a)
                kb = b if b.isascii() else to_keys(b)
                sent.append((ka + kb, [0 if a.isascii() else KO] * len(ka) + [0 if b.isascii() else KO] * len(kb)))
        for i, (k, y) in enumerate(sent):
            x = feats(k, y, sent[i - 1] if i else None, sent[i + 1] if i + 1 < len(sent) else None)
            words += 1
            for li2, (rule, _) in enumerate(leaves):
                if match(rule, x):
                    cnt[li2] += 1
                    break
    raw[src] = (cnt, words)
    say(f"  raw {src}: {words:,} words")

say("=== 2. decision-tree kinds, ranked by expected wrong words per 1,000 typed words (NSMC / KcBERT)")
rows = []
for li, (rule, idx) in enumerate(leaves):
    rate = Y[idx].mean()
    fr = [1000 * raw[s][0][li] / raw[s][1] for s in ("nsmc", "kcbert")]
    rows.append((max(f * rate for f in fr), rule, idx, rate, fr))
for e, rule, idx, rate, fr in sorted(rows, key=lambda r: -r[0]):
    say(f"  expected {fr[0] * rate:6.3f} / {fr[1] * rate:6.3f}   per 1k {fr[0]:7.2f} / {fr[1]:7.2f}   "
        f"error rate {rate:6.2%} ({int(Y[idx].sum())}/{len(idx)})   {show(rule)}")
say("done")
