"""decision policy on every cell system_eval scores, for one model (2026-11-04): the split threshold (split a word at the
key where its best path changes language when the path is at least this sure) and the Korean / English bias, with the
revision rule in use (d 3, T 0.95, the same at 2 and 5). --norm: the keys go to the model with an upper-case vowel
after a double consonant made lower case (까 typed RK -> Rk: Shift held), the text still from the keys as typed.
    python -u experiments/logs/policy_cells.py NAME [--norm]"""
import math
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.models import load  # noqa: E402

NORM = "--norm" in sys.argv
name = [a for a in sys.argv[1:] if not a.startswith("--")][0]
SHIFT = re.compile(r"(?<=[QWERT])[HJKLYUINMB]")
its = se.items("dev")
keys = [[k for k, _ in s] for s, _, _ in its]
fed = [[SHIFT.sub(lambda m: m.group(0).lower(), k) for k in ks] for ks in keys] if NORM else keys
tgt = [i for _, i, _ in its]
exp = [expected(*s[i]) for s, i, _ in its]
word = [keys[j][tgt[j]] for j in range(len(its))]


def cell(g):
    if g[0] == "switch":
        return g[1]
    if g[0] == "main":
        return "main KO" if g[1] in ("nsmc", "kcbert", "web") else "main EN"
    return {"end_extra": "KO>EN end", "num_en_set": "number EN", "collide": "collisions"}.get(g[0], g[0])


cells = np.array([cell(g) for *_, g in its])
m = load(name)
out = {}
for d in (0, 2, 3, 5):
    res = st.decide(m, [k[:min(len(k), i + 1 + d)] for k, i in zip(fed, tgt)], tgt)
    pb = np.array([r[1] for r in res])
    lg = np.array([math.log(max(r[2], 1e-9)) - math.log(max(r[3], 1e-9)) for r in res])
    multi = np.array([len(set(r[0])) > 1 for r in res])
    c_split = np.array([multi[j] and text_of(word[j], (r[0], 1.0, r[2], r[3])) == exp[j] for j, r in enumerate(res)])
    out[d] = (pb, lg, multi, c_split)
    print(f"lookahead {d} done", flush=True)
c_ko = np.array([compose(w) == e for w, e in zip(word, exp)])
c_en = np.array([w == e for w, e in zip(word, exp)])


def wrong(bias, split):
    kind = {d: np.where(out[d][2] & (out[d][0] >= split), 2, np.where(out[d][1] > bias, 0, 1)) for d in out}
    right = {d: np.choose(kind[d], [c_ko, c_en, out[d][3]]) for d in out}
    take = (kind[3] != kind[0]) & (out[3][0] >= 0.95) & (kind[2] == kind[3]) & (kind[5] == kind[3])
    return ~np.where(take, right[3], right[0])


names = sorted(set(cells), key=lambda c: (c.startswith("main"), c))
n = {c: (cells == c).sum() for c in names}
cur = wrong(0.0, 0.7)
rate = lambda w: {c: 100 * (w & (cells == c)).sum() / n[c] for c in names}  # noqa: E731
base = rate(cur)
print(f"\n{name}{' (input normalised)' if NORM else ''}: current policy (bias 0, split 0.7), per cell, and the change of others")
settings = [(0.0, s) for s in (0.5, 0.6, 0.8, 0.9)] + [(b, 0.7) for b in (-0.5, -0.25, 0.25, 0.5)]
print(f"{'cell':<16}{'n':>7}{'now':>8}" + "".join(f"{f'b{b:+.2f}/s{s:.1f}':>12}" for b, s in settings))
rows = {st_: rate(wrong(*st_)) for st_ in settings}
for c in names:
    print(f"{c:<16}{n[c]:>7}{base[c]:>7.2f}%" + "".join(f"{rows[s_][c] - base[c]:>+12.2f}" for s_ in settings))
tot = cur.sum()
print(f"{'all words':<16}{len(cur):>7}{tot:>8}" + "".join(f"{int(wrong(*s_).sum()) - int(tot):>+12}" for s_ in settings))
