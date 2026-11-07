"""bias curve on ambiguous English vs its guards (2026-11-04): move the Korean / English decision bias of a model
and count errors on amb_en (English that reads as Korean) against jamo1 + amb_ko (Korean that reads as English),
after the revision rule (d 3, T 0.95, the same at 2 and 5; the bias applies at every lookahead, as policy_sweep).
A change that only moves a model along its curve leans; it learned something only if it sits below the curve.
    python -u experiments/logs/bias_curve_amb.py NAME ...   -> prints one curve per model"""
import math
import sys

import numpy as np

sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.models import load  # noqa: E402

CELLS = ("amb_en", "jamo1", "amb_ko")
BIASES = [-3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3]
its = [x for x in se.items("dev") if x[2][0] in CELLS]
keys = [[k for k, _ in s] for s, _, _ in its]
exp = [expected(*s[i]) for s, i, _ in its]
cell = np.array([g[0] for *_, g in its])
c_ko = np.array([compose(keys[j][i]) == exp[j] for j, (_, i, _) in enumerate(its)])
c_en = np.array([keys[j][i] == exp[j] for j, (_, i, _) in enumerate(its)])
n_en, n_guard = (cell == "amb_en").sum(), (cell != "amb_en").sum()

for name in sys.argv[1:]:
    m = load(name)
    out = {}
    for d in (0, 2, 3, 5):
        res = st.decide(m, [k[:min(len(k), i + 1 + d)] for k, (_, i, _) in zip(keys, its)], [i for _, i, _ in its])
        pb = np.array([r[1] for r in res]); lg = np.array([math.log(max(r[2], 1e-9)) - math.log(max(r[3], 1e-9)) for r in res])
        multi = np.array([len(set(r[0])) > 1 for r in res])
        c_split = np.array([multi[j] and text_of(keys[j][its[j][1]], (r[0], 1.0, r[2], r[3])) == exp[j] for j, r in enumerate(res)])
        out[d] = (pb, lg, multi, c_split)
    print(f"{name}: bias  amb_en wrong  guard wrong (jamo1 + amb_ko)", flush=True)
    for b in BIASES:
        kind = {d: np.where(out[d][2] & (out[d][0] >= 0.7), 2, np.where(out[d][1] > b, 0, 1)) for d in out}
        right = {d: np.choose(kind[d], [c_ko, c_en, out[d][3]]) for d in out}
        take = (kind[3] != kind[0]) & (out[3][0] >= 0.95) & (kind[2] == kind[3]) & (kind[5] == kind[3])
        ok = np.where(take, right[3], right[0])
        e_en, e_g = (~ok & (cell == "amb_en")).sum(), (~ok & (cell != "amb_en")).sum()
        print(f"  {b:+5.2f}  {100 * e_en / n_en:6.2f}% ({e_en})  {100 * e_g / n_guard:6.2f}% ({e_g})", flush=True)
