"""slices inside the known weak cells (log 25): are a cell's errors concentrated on a few keys (fixable with data for
those keys) or spread over many (the floor for what the model sees)? And which word shapes fail beyond the cell's
rate? Reads the dev error dumps of experiments/logs/error_dump.py (one per seed) and counts every scored dev word
of system_eval.items as the denominator; no model runs.
    python experiments/logs/slice_errors.py ERRORS.jsonl [ERRORS.jsonl ...] > OUT.txt
Per cell: words, errors per seed; concentration (share of the errors in the top 10 / 20 keys against the share of
words those keys hold; distinct keys with an error); the keys with the most errors (both seeds); and the rate per
shape slice (expected language, length, upper case, digit, how the keys read in the other language, error
direction), with a 2 binomial se band against the cell's own rate."""
import json
import math
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected  # noqa: E402

from nunchi.demo import compose  # noqa: E402

CELLS = [("single jamo", "jamo1"), ("Korean that reads as English", "amb_ko"), ("ambiguous English", "amb_en"),
         ("fused words (no space)", "fused"), ("one word, English + particle", "word1 | mix1"),
         ("one word, English", "word1 | en1"), ("key collisions", "collide"), ("EN>KO end", "switch | EN>KO end"),
         ("KO>EN end", "switch | KO>EN end"), ("KO>EN start", "switch | KO>EN start"),
         ("EN>KO middle", "switch | EN>KO middle"), ("upper-case acronyms", "acro"), ("typo, position", "typo_pos")]
HANGUL = re.compile(r"[가-힣ㄱ-ㅣ]")
LATIN = re.compile(r"[A-Za-z]")
SYL = re.compile(r"^[가-힣]+$")
JAMO = re.compile(r"[ㄱ-ㅣ]")


def lang(text):
    h, a = bool(HANGUL.search(text)), bool(LATIN.search(text))
    return "mixed" if h and a else "ko" if h else "en" if a else "other"


def shape(keys, exp):
    """slices of one word, from its keys and the expected text"""
    letters = re.sub(r"[^A-Za-z]", "", keys)
    reading = compose(letters) if letters else ""
    el = lang(exp)
    other = ("Korean reading: whole syllables" if SYL.match(reading) else "Korean reading: has bare jamo"
             if JAMO.search(reading) else "Korean reading: none") if el != "ko" else \
        ("English reading: lower case" if letters.islower() else "English reading: has upper case")
    n = len(keys)
    return {"expected": el, "length": "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+",
            "upper case": "yes" if re.search(r"[A-Z]", keys) else "no",
            "digit": "yes" if re.search(r"\d", keys) else "no", "other reading": other}


def main():
    dumps = sys.argv[1:]
    its = se.items("dev")
    words = defaultdict(list)  # cell -> [(key, shape)]
    index = {}
    for s, i, g in its:
        gs = " | ".join(g)
        cell = next((c for c, pre in CELLS if gs == pre or gs.startswith(pre + " | ") or gs.startswith(pre)), None)
        if cell is None:
            continue
        k, y = s[i]
        e = expected(k, y)
        words[cell].append((k, shape(k, e)))
        index[(gs, tuple(expected(*w) for w in s), i)] = cell
    errs = []  # per seed: cell -> [(key, shape, direction)]
    for d in dumps:
        per = defaultdict(list)
        miss = 0
        for line in open(d, encoding="utf-8"):
            r = json.loads(line)
            cell = index.get((r["g"], tuple(r["sent"]), r["i"]))
            if cell is None:
                miss += r["g"].split(" | ")[0] in {pre.split(" | ")[0] for _, pre in CELLS}
                continue
            sh = shape(r["keys"], r["exp"])
            sh["error"] = f"{lang(r['exp'])} -> {lang(r['out'])}"
            per[cell].append((r["keys"], sh))
        errs.append(per)
        print(f"{d}: {sum(map(len, per.values()))} errors in these cells" + (f" ({miss} not matched)" if miss else ""))
    for cell, _ in CELLS:
        ws = words[cell]
        if not ws:
            continue
        N = len(ws)
        nk = Counter(k for k, _ in ws)
        ek = [Counter(k for k, _ in e[cell]) for e in errs]
        E = [len(e[cell]) for e in errs]
        print(f"\n=== {cell}: {N:,} words, {len(nk):,} distinct keys; errors " +
              " / ".join(f"{x} ({100 * x / N:.2f}%)" for x in E))
        both = sum(ek, Counter())
        for top in (10, 20):
            keys = [k for k, _ in both.most_common(top)]
            share_e = sum(both[k] for k in keys) / max(1, sum(both.values()))
            share_w = sum(nk[k] for k in keys) / N
            print(f"  top {top} keys: {100 * share_e:.0f}% of errors, {100 * share_w:.1f}% of words")
        print(f"  keys with an error: " + " / ".join(str(len(c)) for c in ek) +
              f"; in both seeds: {len(set.intersection(*(set(c) for c in ek)))}")
        print("  most errors (key: errors per seed / words, expected):")
        exp_of = {}
        for k, sh in ws:
            exp_of.setdefault(k, sh["expected"])
        print("    " + "; ".join(f"{k}: {'/'.join(str(c[k]) for c in ek)} of {nk[k]} {exp_of[k]}"
                                 for k, _ in both.most_common(15)))
        rate = sum(E) / (len(E) * N)
        for dim in ("expected", "length", "upper case", "digit", "other reading", "error"):
            if dim == "error":
                c = Counter(sh["error"] for e in errs for _, sh in e[cell])
                print(f"  error direction: " + ", ".join(f"{v} {100 * n / max(1, sum(E)):.0f}%" for v, n in c.most_common()))
                continue
            vals = Counter(sh[dim] for _, sh in ws)
            if len(vals) < 2:
                continue
            parts = []
            for v, n in vals.most_common():
                er = sum(1 for e in errs for _, sh in e[cell] if sh[dim] == v) / len(E)
                r = er / n
                band = 2 * math.sqrt(rate * (1 - rate) / n) if n else 0
                flag = " HIGH" if r - rate > band else " low" if rate - r > band else ""
                parts.append(f"{v}: {100 * r:.1f}% of {n}{flag}")
            print(f"  {dim}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
