"""sensitivity table from knob_eval.py's JSONs: per error kind (switching set) and per source (main set), each
run's wrong count minus the mean of the two baseline seeds. A change is marked * when it is larger than both
twice the count noise of a difference with that mean (2 sqrt(1.5 x base), Poisson) and twice the seed
difference.
    python -u experiments/knob_table.py [RUN ...]      (default: every JSON there, baseline TR3-10K and TR3-10K-s1)"""
import json
import math
import sys
from pathlib import Path

S = Path(__file__).parent
DIR = S / "logs" / "knobs"
B0, B1 = "TR3-10K", "TR3-10K-s1"


def load(name):
    return json.loads((DIR / f"{name}.json").read_text(encoding="utf-8"))


def main():
    names = sys.argv[1:] or sorted(p.stem for p in DIR.glob("*.json") if not p.stem.startswith("_"))
    names = [B0, B1] + [n for n in names if n not in (B0, B1)]
    runs = {n: load(n) for n in names}
    rows = []
    kinds = sorted({k for r in runs.values() for k in r["kinds"]},
                   key=lambda k: -(runs[B0]["kinds"].get(k, 0) + runs[B1]["kinds"].get(k, 0)))
    rows += [("kind", k) for k in kinds]
    rows += [("main", s) for s in sorted(runs[B0]["main"])]
    others = names[2:]
    print(f"{'':<62}{'base':>5}{'s1':>5}{'seed':>5}" + "".join(f"{n[:9]:>10}" for n in others))
    moved = {n: [] for n in others}
    for part, key in rows:
        v = {n: runs[n][{"kind": "kinds", "main": "main"}[part]].get(key, 0) for n in names}
        base = (v[B0] + v[B1]) / 2
        seed = abs(v[B0] - v[B1])
        thr = max(2 * math.sqrt(1.5 * max(base, 1)), 2 * seed)
        cells = []
        for n in others:
            d = v[n] - base
            mark = "*" if abs(d) > thr else " "
            if mark == "*":
                moved[n].append((key, d))
            cells.append(f"{d:>+8.0f}{mark} ")
        label = ("main set, " if part == "main" else "") + key
        print(f"{label[:61]:<62}{v[B0]:>5}{v[B1]:>5}{seed:>5}" + "".join(cells))
    tot = {n: sum(runs[n]["kinds"].values()) for n in names}
    base = (tot[B0] + tot[B1]) / 2
    print(f"{'switching errors, total':<62}{tot[B0]:>5}{tot[B1]:>5}{abs(tot[B0] - tot[B1]):>5}"
          + "".join(f"{tot[n] - base:>+8.0f}  " for n in others))


if __name__ == "__main__":
    main()
