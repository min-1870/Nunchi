"""overnight screen (log 20): each run's two seeds against TR4-10K's two seeds, per cell, final error on dev.
A change counts when the seed means differ by more than max(2 x the larger seed difference, 2 x the binomial se
of a difference of two-seed means). Cells: the switching cells (mean over sources), KO>EN end with its extra
points, main sets (pooled), key collisions (mean over keys), number words in English, ambiguous English and its
guards, typos (position, Shift held), upper-case acronyms.
    python experiments/logs/scr_table.py [NAME ...] [--base=RUN-10K]   (default: every scr-NN with both seeds scored,
    against TR4-10K)"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SY = Path("experiments/logs/systems")
SW = ["KO>EN start", "KO>EN middle", "KO>EN after", "EN>KO start", "EN>KO middle", "EN>KO end", "EN>KO after",
      "number, KO base"]


def cells(name):
    s = json.loads((SY / f"{name}.json").read_text(encoding="utf-8"))
    w, p = s["wrong"], s["points"]
    g = lambda k: w.get(k, 0)  # noqa: E731
    out = {}

    def mean_of(keys):
        keys = [k for k in keys if p.get(k, 0) >= 20]
        return (100 * np.mean([g(k) / p[k] for k in keys]), sum(p[k] for k in keys)) if keys else (None, 0)

    for c in SW:
        out[c] = mean_of([k for k in p if k.startswith(f"switch | {c} | ")])
    srcs = sorted({k.split(" | ")[1] for k in p if k.startswith("end_extra | ")})
    out["KO>EN end"] = (100 * np.mean([(g(f"switch | KO>EN end | {s_}") + g(f"end_extra | {s_}")) /
                                       (p[f"switch | KO>EN end | {s_}"] + p[f"end_extra | {s_}"]) for s_ in srcs]),
                        sum(p[f"switch | KO>EN end | {s_}"] + p[f"end_extra | {s_}"] for s_ in srcs))
    for lab, pre in (("main KO (nsmc kcbert web)", ("main | nsmc", "main | kcbert", "main | web")),
                     ("main EN (imdb reddit)", ("main | imdb", "main | reddit"))):
        ks = [k for k in p if k in pre]
        out[lab] = (100 * sum(g(k) for k in ks) / sum(p[k] for k in ks), sum(p[k] for k in ks))
    for lab, pre in (("key collisions", "collide | "), ("number, EN (real)", "num_en_set | "),
                     ("typo, position", "typo_pos | "), ("typo, Shift held", "typo_shift | "),
                     ("single jamo", "jamo1 | "), ("Korean that reads as English", "amb_ko | "),
                     ("upper-case acronyms", "acro | "), ("fused words (no space)", "fused | "),
                     ("one word, Korean", "word1 | ko1"), ("one word, English", "word1 | en1"),
                     ("one word, English + particle", "word1 | mix1")):
        out[lab] = mean_of([k for k in p if k.startswith(pre)])
    out["ambiguous English"] = (100 * g("amb_en") / p["amb_en"], p["amb_en"]) if "amb_en" in p else (None, 0)
    return out


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted({f.stem[:-4] for f in SY.glob("scr-*-10K.json")})
    base = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--base=")), "TR4-10K")  # e.g. wrd-00-10K
    b0, b1 = cells(base), cells(f"{base}-s1")
    for name in names:
        if not (SY / f"{name}-10K-s1.json").exists():
            continue
        r0, r1 = cells(f"{name}-10K"), cells(f"{name}-10K-s1")
        reg = {m["name"]: m for m in json.loads(Path("models/registry.json").read_text(encoding="utf-8"))["models"]}
        print(f"\n{name}: {reg.get(name + '-10K', {}).get('what', '')}")
        print(f"  {'cell':<30}{base:>16}{name:>16}{'change':>9}{'noise':>8}")
        for c in b0:
            if b0[c][0] is None or r0[c][0] is None or b1[c][0] is None or r1[c][0] is None:
                continue
            bm, rm = (b0[c][0] + b1[c][0]) / 2, (r0[c][0] + r1[c][0]) / 2
            n = b0[c][1]
            q = max(bm, 0.05) / 100
            se = 100 * math.sqrt(2 * q * (1 - q) / (2 * n))
            noise = max(2 * max(abs(b0[c][0] - b1[c][0]), abs(r0[c][0] - r1[c][0])), 2 * se)
            d = rm - bm
            mark = " better" if d < -noise else " WORSE" if d > noise else ""
            print(f"  {c:<30}{b0[c][0]:7.2f}/{b1[c][0]:<7.2f} {r0[c][0]:7.2f}/{r1[c][0]:<7.2f}{d:+8.2f}{noise:8.2f}{mark}")


if __name__ == "__main__":
    main()
