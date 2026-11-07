"""decision policy for one trained model, on the words system_eval.py scores (dev): the Korean / English bias and
the split threshold at Space, and the revision rule (take the answer d words later when it is T sure and agrees
with the answers at the words listed in "stable"). The model is run once per lookahead d = 0..5 and its outputs
cached; every setting is then scored on the CPU.

    python -u experiments/policy_sweep.py cache NAME   -> experiments/logs/policy/NAME.npz  (registry name)
    python -u experiments/policy_sweep.py sweep NAME   -> prints the best settings, logs/policy/NAME_sweep.json

Cells and units as the results table: switching-set cells (mean over sources), KO>EN end with the extra points,
main-set sources, key collisions, ambiguous English; changes vs TR3-10K (mean of its two seeds, at Space) in units
of each cell's noise (max(2 sqrt(1.5) x binomial se, 2 x seed difference)); objective = mean over the 16 cells
(ambiguous English shown, not counted). Current rule: bias 0, split 0.7, d 3, T 0.95, stable at 2 and 5."""
import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
OUT = S / "logs" / "policy"
LOGS = S / "logs"
D = 6  # lookaheads 0..5
CATS = ["KO>EN start", "KO>EN middle", "KO>EN end", "KO>EN after", "EN>KO start", "EN>KO middle", "EN>KO end",
        "EN>KO after", "number, KO base", "number, EN base"]
SRCS = ["nsmc", "kcbert", "web", "imdb", "reddit"]
CUR = {"bias": 0.0, "split": 0.7, "d": 3, "T": 0.95, "stable": "cur"}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cache(name):
    import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)
    import nunchi.stream_tagger as st
    from nunchi.demo import compose
    from nunchi.models import load
    from system_eval import items
    from textcheck import expected, text_of

    its = items()
    keys = [[k for k, _ in s] for s, _, _ in its]
    tgt = [i for _, i, _ in its]
    exp = [expected(*s[i]) for s, i, _ in its]
    m = load(name)
    n = len(its)
    pb, lg, multi, c_split = (np.zeros((D, n), np.float32), np.zeros((D, n), np.float32),
                              np.zeros((D, n), bool), np.zeros((D, n), bool))
    t0, done, step = time.time(), 0, 8192
    for d in range(D):
        for a in range(0, n, step):
            res = st.decide(m, [k[:min(len(k), i + 1 + d)] for k, i in zip(keys[a:a + step], tgt[a:a + step])],
                            tgt[a:a + step])
            for j, r in enumerate(res, a):
                path, p, pko, pen = r
                pb[d, j], lg[d, j] = p, math.log(max(pko, 1e-9)) - math.log(max(pen, 1e-9))
                multi[d, j] = len(set(path)) > 1
                # right if split along its path (forced past any threshold; text_of splits at pb >= 0.7)
                c_split[d, j] = multi[d, j] and text_of(keys[j][tgt[j]], (path, 1.0, pko, pen)) == exp[j]
            done += len(res)
            rate = done / (time.time() - t0)
            say(f"lookahead {d}: {min(a + step, n):,}/{n:,}  ({100 * done / (D * n):.0f}% of all, ETA "
                f"{(D * n - done) / rate / 60:.1f} min)")
    c_ko = np.array([compose(keys[j][tgt[j]]) == exp[j] for j in range(n)])
    c_en = np.array([keys[j][tgt[j]] == exp[j] for j in range(n)])
    groups = sorted({" | ".join(g) for *_, g in its})
    gid = np.array([groups.index(" | ".join(g)) for *_, g in its])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", pb=pb, lg=lg, multi=multi, c_split=c_split, c_ko=c_ko, c_en=c_en,
                        gid=gid, groups=np.array(groups))
    say(f"cached {n:,} words x {D} lookaheads -> {OUT / f'{name}.npz'}")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def baseline():
    """-> (rates of the TR3-10K seed mean at Space, noise per cell), the units of the results table"""
    K = LOGS / "knobs"
    pts, ee, coll, amb = (read(K / "_points.json"), read(LOGS / "end_extra_scores_all.json"),
                          read(LOGS / "collision_all.json"), read(LOGS / "amb_en_scores.json"))
    cells = {c: [k for k, v in pts["switch_cells"].items() if v >= 20 and k.split(" | ")[0] == c] for c in CATS}

    def rates(nm):
        r = read(K / f"{nm}.json")
        v = {c: 100 * np.mean([r["switch_cells"].get(k, 0) / pts["switch_cells"][k] for k in cells[c]]) for c in CATS}
        v["KO>EN end"] = ee[nm]["rate"]
        v.update({s: 100 * r["main"].get(s, 0) / pts["main"][s] for s in SRCS})
        v["collide"], v["amb_en"] = 100 * coll[nm] / coll["_n"], 100 * amb[nm] / amb["_n"]
        return v

    def ns(c):
        if c == "KO>EN end":
            return list(ee["points"].values())
        if c in CATS:
            return [pts["switch_cells"][k] for k in cells[c]]
        return [pts["main"][c]] if c in SRCS else [{"collide": coll["_n"], "amb_en": amb["_n"]}[c]]

    b0, b1 = rates("TR3-10K"), rates("TR3-10K-s1")
    base = {c: (b0[c] + b1[c]) / 2 for c in b0}
    th = {}
    for c in base:
        q, n = base[c] / 100, ns(c)
        se = 100 * math.sqrt(sum(q * (1 - q) / x for x in n)) / len(n)
        th[c] = max(2 * math.sqrt(1.5) * se, 2 * abs(b0[c] - b1[c]))
    return base, th


def stable_set(d, kind):
    if kind == "none":
        return []
    if kind == "cur":  # as the rule in use: one word before and two after (2 and 5 for d = 3)
        return sorted({x for x in (d - 1, min(d + 2, D - 1)) if 1 <= x != d})
    return list(range(d + 1, D))  # "after": every later lookahead


def sweep(name):
    z = np.load(OUT / f"{name}.npz")
    pb, lg, multi, c_split, c_ko, c_en, gid = (z[k] for k in ("pb", "lg", "multi", "c_split", "c_ko", "c_en", "gid"))
    groups = list(z["groups"])
    npts = np.bincount(gid, minlength=len(groups))
    base, th = baseline()
    EQ = CATS + SRCS + ["collide"]
    # cell -> [(group index of each source part, ...)] as the table pools them
    sw = {c: [i for i, g in enumerate(groups) if g.startswith(f"switch | {c} | ") and npts[i] >= 20] for c in CATS}
    end_src = sorted({g.split(" | ")[1] for g in groups if g.startswith("end_extra | ")})
    end_parts = [[groups.index(f"switch | KO>EN end | {s}"), groups.index(f"end_extra | {s}")] for s in end_src]

    def cells(wrong):
        v = {c: 100 * np.mean([wrong[i] / npts[i] for i in sw[c]]) for c in CATS}
        v["KO>EN end"] = 100 * np.mean([wrong[p].sum() / npts[p].sum() for p in end_parts])
        for s in SRCS:
            i = groups.index(f"main | {s}")
            v[s] = 100 * wrong[i] / npts[i]
        for c in ("collide", "amb_en"):
            i = groups.index(c)
            v[c] = 100 * wrong[i] / npts[i]
        return v

    def score(ok):
        v = cells(np.bincount(gid, weights=~ok, minlength=len(groups)))
        return v, float(np.mean([(v[c] - base[c]) / th[c] for c in EQ]))

    def kind(d, b, t):  # 0 ko, 1 en, 2 split
        return np.where(multi[d] & (pb[d] >= t), 2, np.where(lg[d] > b, 0, 1))

    def right(d, k):
        return np.choose(k, [c_ko, c_en, c_split[d]])

    results = []
    biases = [-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2]
    splits = [0.5, 0.6, 0.7, 0.8, 0.9]
    revs = [None] + [(d, T, s) for d in range(1, D) for T in (0.8, 0.9, 0.95, 0.98, 0.99)
                     for s in ("none", "cur", "after")]
    t0 = time.time()
    for b, t in itertools.product(biases, splits):
        k = [kind(d, b, t) for d in range(D)]
        ok0 = right(0, k[0])
        v0, a0 = score(ok0)
        seen = set()
        for rv in revs:
            if rv is None:
                ok, key = ok0, None
            else:
                d, T, s = rv
                st_ = stable_set(d, s)
                key = (d, T, tuple(st_))
                if key in seen:
                    continue
                seen.add(key)
                take = (k[d] != k[0]) & (pb[d] >= T)
                for x in st_:
                    take &= k[x] == k[d]
                ok = np.where(take, right(d, k[d]), ok0)
            v, a = score(ok)
            results.append({"bias": b, "split": t, "d": rv[0] if rv else 0, "T": rv[1] if rv else None,
                            "stable": list(key[2]) if key else None, "avg": round(a, 3), "avg_space": round(a0, 3),
                            "cells": {c: round(x - base[c], 2) for c, x in v.items()},
                            "space": {c: round(x - base[c], 2) for c, x in v0.items()}})
        say(f"bias {b:+.2f} split {t:.1f}: {len(results):,} settings ({time.time() - t0:.0f} s)")
    cur = next(r for r in results if r["bias"] == CUR["bias"] and r["split"] == CUR["split"] and r["d"] == CUR["d"]
               and r["T"] == CUR["T"] and r["stable"] == stable_set(CUR["d"], CUR["stable"]))
    worse = lambda r: [c for c in EQ if r["cells"][c] - cur["cells"][c] > th[c]]  # noqa: E731
    ok = sorted((r for r in results if not worse(r)), key=lambda r: r["avg"])
    say(f"{len(results):,} settings; current rule: avg {cur['avg']:+.3f} (at Space {cur['avg_space']:+.3f}); "
        f"{len(ok):,} have no cell worse than it beyond noise")
    print(f"{'bias':>6}{'split':>6}{'d':>3}{'T':>6}  {'stable':<8}{'avg':>8}{'Space':>8}  " +
          "".join(f"{c.replace('KO>EN', 'ke').replace('EN>KO', 'ek').replace('number, ', 'n')[:9]:>10}" for c in EQ + ["amb_en"]))
    for r in [cur] + ok[:15]:
        print(f"{r['bias']:>+6.2f}{r['split']:>6.1f}{r['d']:>3}{(r['T'] or 0):>6.2f}  {str(r['stable']):<8}"
              f"{r['avg']:>+8.3f}{r['avg_space']:>+8.3f}  " + "".join(f"{r['cells'][c]:>+10.2f}" for c in EQ + ["amb_en"]))
    (OUT / f"{name}_sweep.json").write_text(json.dumps({"current": cur, "noise": th, "results": results}),
                                            encoding="utf-8")


if __name__ == "__main__":
    {"cache": cache, "sweep": sweep}[sys.argv[1]](sys.argv[2])
