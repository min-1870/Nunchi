"""overnight 51k+ runs (log 24) against TR4-51K, per cell, final error on dev (cells as in scr_table.py).
51k runs have one seed each (until TR5-51K-s1), so the seed term of the noise is the larger of: TR4-10K's two seeds,
and TR5-51K's two seeds once both are scored. A change counts when it is more than
max(2 x that seed difference, 2 x the binomial se of a difference of two single runs).
Cells TR4-51K was never scored on (typos, acronyms, fused) show the run alone.
    python experiments/logs/tr5_table.py [NAME ...]   (default: every TR5 run that has a score file)"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scr_table import SY, cells  # noqa: E402


def main():
    names = sys.argv[1:] or [n for n in ("TR5-51K", "TR5n-51K", "TR5n-102K", "TR5-51K-s1", "TR5n-102K-s1", "TR5n-204K", "TR5n-204K-s1") if (SY / f"{n}.json").exists()]
    base = cells("TR4-51K")
    seeds = [(cells("TR4-10K"), cells("TR4-10K-s1"))]
    for a in ("TR5-51K", "TR5n-102K"):
        if (SY / f"{a}-s1.json").exists() and (SY / f"{a}.json").exists():
            seeds.append((cells(a), cells(f"{a}-s1")))
    reg = {m["name"]: m for m in json.loads(Path("models/registry.json").read_text(encoding="utf-8"))["models"]}
    for name in names:
        r = cells(name)
        print(f"\n{name}: {reg.get(name, {}).get('what', '')}")
        print(f"  {'cell':<30}{'TR4-51K':>9}{name:>11}{'change':>9}{'noise':>8}")
        tot = {"better": 0, "WORSE": 0}
        for c in r:
            if r[c][0] is None:
                continue
            if base[c][0] is None:
                print(f"  {c:<30}{'-':>9}{r[c][0]:11.2f}")
                continue
            sd = max(abs(a[c][0] - b[c][0]) for a, b in seeds if a[c][0] is not None and b[c][0] is not None)
            q = max(base[c][0], 0.05) / 100
            se = 100 * math.sqrt(2 * q * (1 - q) / base[c][1])
            noise = max(2 * sd, 2 * se)
            d = r[c][0] - base[c][0]
            mark = "better" if d < -noise else "WORSE" if d > noise else ""
            if mark:
                tot[mark] += 1
            print(f"  {c:<30}{base[c][0]:9.2f}{r[c][0]:11.2f}{d:+9.2f}{noise:8.2f} {mark}")
        print(f"  -> better {tot['better']}, worse {tot['WORSE']}")


if __name__ == "__main__":
    main()
