"""regression over data / loss knobs (RegMix-style, on 10k runs): sample knob settings, train them, fit each
switching-set cell's error rate (and each main-set source's error count) as a function of the knobs, pick
the setting with the lowest predicted objective that passes the main-set guard.

    python -u experiments/mix_search.py plan N [SEED]   -> experiments/configs/knobs_mix.json (N runs, Latin hypercube)
    python -u experiments/mix_search.py fit              -> fit on every scored run with known knobs, print the pick
    python -u experiments/mix_search.py round N          -> the same, and write configs/knobs_roundN.json (pick, 2 seeds)

Knobs (on the TR3-10K recipe): en_mk (share of en_marked.txt), amb / switch / code (line multipliers in
NUNCHI_SHORT_EN_MODE), amb_ko (NUNCHI_AMB_KO_X), entry (NUNCHI_ENTRY_W, loss weight of words entering English).
Features: each knob scaled to 0..1 over its range (multipliers on log2), plus its square. Objective: the
switching set's per-category rate, averaged over sources (cells with at least MIN_N points) and then over
categories, as switch_eval.py prints it. Guard: every main-set source predicted no worse than the baseline
by more than its count noise (2 sqrt(1.5 x base)). Leave-one-out error of the objective is printed: if it is
not well below the spread between runs, the fit is not to be trusted."""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).parent
DIR = S / "logs" / "knobs"
CFG = S / "configs" / "knobs_mix.json"
MIN_N = 20
RANGE = {"en_mk": (0.0, 0.12), "amb": (1, 4), "switch": (1, 3), "code": (1, 4), "amb_ko": (1, 4), "entry": (1, 3)}
LOG2 = {"amb", "switch", "code", "amb_ko"}
# knobs drawn by plan() and searched by fit(); the others only describe earlier runs (code: two tries, number-word
# lines x4 and the v2 web pool, moved no error kind and raised jamo-in-English, so it stays at 1)
SAMPLED = ("en_mk", "amb", "switch", "amb_ko", "entry")
# knobs of runs trained before this script (docs/EXPERIMENTS.md, log 7); unlisted knobs are at 1 / 0
KNOWN = {"TR3-10K": {},
         "TR3-10K-s1": {},
         "A-amb-10K": {"amb": 2},
         "A-amb4-10K": {"amb": 4},
         "A-switch-10K": {"switch": 2},
         "A-amb-switch-10K": {"amb": 2, "switch": 2},
         "R3-1-10K": {"amb": 4, "switch": 2},
         "R3-2-10K": {"amb": 4, "switch": 2, "amb_ko": 4},
         "R3-3-10K": {"amb_ko": 4},
         "R3-2s-10K": {"amb": 3, "switch": 2, "amb_ko": 4},
         "R3-2c-10K": {"amb": 4, "switch": 2, "code": 2, "amb_ko": 4},
         "C-10K": {"entry": 3},
         "K1a-10K": {"en_mk": 0.05},
         "K1b-10K": {"en_mk": 0.1},
         "K2a-10K": {"code": 4}}


def scaled(k, v):
    lo, hi = RANGE[k]
    return (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) if k in LOG2 else (v - lo) / (hi - lo)


def unscaled(k, u):
    lo, hi = RANGE[k]
    return 2 ** (math.log2(lo) + u * (math.log2(hi) - math.log2(lo))) if k in LOG2 else lo + u * (hi - lo)


def env_of(knobs):
    kn = {k: knobs.get(k, RANGE[k][0]) for k in RANGE}
    modes = "+".join(f"{m}:{kn[m]:.2f}" for m in ("amb", "switch", "code") if kn[m] > 1.001)
    env = {"NUNCHI_EN_MK": f"{kn['en_mk']:.3f}", "NUNCHI_AMB_KO_X": f"{kn['amb_ko']:.2f}",
           "NUNCHI_ENTRY_W": f"{kn['entry']:.2f}"}
    if modes:
        env["NUNCHI_SHORT_EN_MODE"] = modes
    return env


def plan(n, seed):
    rnd = random.Random(seed)
    cols = {k: rnd.sample([(i + rnd.random()) / n for i in range(n)], n) if k in SAMPLED else [0.0] * n
            for k in RANGE}
    base = json.loads((S / "configs" / "r3_2_branches.json").read_text(encoding="utf-8"))["base"]
    runs, known = [], {}
    for i in range(n):
        kn = {k: round(unscaled(k, cols[k][i]), 3) for k in RANGE}
        runs.append([f"M{i:02d}-10K", f"m{i:02d}_10k.pt", env_of(kn)])
        known[f"M{i:02d}-10K"] = kn
    CFG.write_text(json.dumps({"steps": 10000, "base": base, "runs": runs, "evals": [], "knobs": known},
                              indent=1), encoding="utf-8")
    print(f"{n} runs -> {CFG}")


def features(kn, quad=True):
    u = [scaled(k, kn.get(k, RANGE[k][0])) for k in RANGE]
    return np.array([1.0] + u + ([x * x for x in u] if quad else []))


def all_known():
    """knobs of every run: the ones listed above and the "knobs" of every configs/knobs_*.json"""
    known = dict(KNOWN)
    for cfg in sorted((S / "configs").glob("knobs_*.json")):
        known.update(json.loads(cfg.read_text(encoding="utf-8")).get("knobs", {}))
    return known


def fit(write_round=None):
    """Objective: quadratic or linear in the knobs, whichever predicts left-out runs better. Guard: main-set
    errors per source, linear in the knobs (the quadratic fit under-predicted the Korean cost of the first
    pick: kcbert 21 -> 33 observed, inside the limit predicted), must stay within the baseline plus one count
    noise unit (sqrt(1.5 x base)), half the usual limit, to leave room for the fit's own error.
    write_round=n: write configs/knobs_round{n}.json with the pick at seeds 0 and 1."""
    known = all_known()
    pts = json.loads((DIR / "_points.json").read_text(encoding="utf-8"))
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in DIR.glob("*.json")
            if not p.stem.startswith("_") and p.stem in known}
    names = sorted(runs)
    cells = sorted(c for c, v in pts["switch_cells"].items() if v >= MIN_N)
    cats = sorted({c.split(" | ")[0] for c in cells})
    srcs = sorted(pts["main"])
    Y_cell = np.array([[runs[nm]["switch_cells"].get(c, 0) / pts["switch_cells"][c] for c in cells] for nm in names])
    Y_main = np.array([[runs[nm]["main"].get(s, 0) for s in srcs] for nm in names], dtype=float)

    def objective(rates):  # rates: (..., cells) -> (...)
        per_cat = [rates[..., [i for i, c in enumerate(cells) if c.startswith(cat + " | ")]].mean(-1) for cat in cats]
        return 100 * np.stack(per_cat, -1).mean(-1)

    def solve(Xa, Ya):
        lam = 1e-2 * np.eye(Xa.shape[1])
        lam[0, 0] = 0
        return np.linalg.solve(Xa.T @ Xa + lam, Xa.T @ Ya)

    obs = objective(Y_cell)
    loo = {}
    for quad in (False, True):
        X = np.stack([features(known[nm], quad) for nm in names])
        err = []
        for i in range(len(names)):
            m = np.arange(len(names)) != i
            err.append(objective(X[i] @ solve(X[m], Y_cell[m])) - obs[i])
        loo[quad] = np.array(err)
    quad = bool(np.std(loo[True]) < np.std(loo[False]))
    print(f"{len(names)} runs; objective (mean switching rate %) observed {obs.min():.2f}..{obs.max():.2f} "
          f"(sd {obs.std():.2f}); leave-one-out error sd: linear {np.std(loo[False]):.2f}, quadratic "
          f"{np.std(loo[True]):.2f} -> {'quadratic' if quad else 'linear'}")
    for nm, o, e in sorted(zip(names, obs, loo[quad]), key=lambda r: r[1])[:8]:
        print(f"  {nm:<14}{o:>7.2f}  (loo pred {o + e:.2f})  {known[nm]}")
    X = np.stack([features(known[nm], quad) for nm in names])
    W_cell = solve(X, Y_cell)
    W_main = solve(np.stack([features(known[nm], False) for nm in names]), Y_main)
    base_main = np.array([(runs["TR3-10K"]["main"].get(s, 0) + runs["TR3-10K-s1"]["main"].get(s, 0)) / 2 for s in srcs])
    limit = base_main + np.sqrt(1.5 * np.maximum(base_main, 1))
    rnd = np.random.default_rng(0)
    U = rnd.random((200_000, len(RANGE))) * np.array([k in SAMPLED for k in RANGE])
    ones = np.ones((len(U), 1))
    F = np.concatenate([ones, U] + ([U * U] if quad else []), 1)
    pred, pred_main = objective(F @ W_cell), np.concatenate([ones, U], 1) @ W_main
    ok = (pred_main <= limit).all(1)
    best = np.argmin(np.where(ok, pred, np.inf))
    kn = {k: round(unscaled(k, U[best, j]), 3) for j, k in enumerate(RANGE)}
    s0 = objective(features({}, quad) @ W_cell)
    print(f"pick: {kn}\n  predicted objective {pred[best]:.2f} (TR3-10K fitted {s0:.2f}, observed "
          f"{(obs[names.index('TR3-10K')] + obs[names.index('TR3-10K-s1')]) / 2:.2f}); main set predicted "
          f"{dict(zip(srcs, pred_main[best].round(1)))} limit {dict(zip(srcs, limit.round(1)))}; "
          f"{ok.mean():.0%} of samples pass the guard")
    print(f"  env: {env_of(kn)}")
    if write_round:
        base = json.loads((S / "configs" / "r3_2_branches.json").read_text(encoding="utf-8"))["base"]
        tag = f"R{write_round}-10K"
        runs_out = [[tag, f"r{write_round}_10k.pt", env_of(kn)],
                    [f"{tag}-s1", f"r{write_round}_10k_s1.pt", {**env_of(kn), "NUNCHI_SEED": "1"}]]
        out = S / "configs" / f"knobs_round{write_round}.json"
        out.write_text(json.dumps({"steps": 10000, "base": base, "runs": runs_out, "evals": [],
                                   "knobs": {tag: kn, f"{tag}-s1": kn}}, indent=1), encoding="utf-8")
        print(f"  -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif sys.argv[1] == "round":
        fit(int(sys.argv[2]))
    else:
        fit()
