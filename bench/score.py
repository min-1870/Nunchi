"""Score an input method's output on the Nunchi benchmark (SPEC.md). Python 3.8+, no other dependencies.

    python score.py nunchi_bench.jsonl predictions.jsonl [--json scores.json]

predictions: one line per benchmark line, either JSON ({"id": "nb000123", "text": "what the field holds"}) or
"id<TAB>text". A word is right when the predicted word equals the expected one exactly (after Unicode NFC); the
text is split on whitespace, and a line whose word count differs from the expected one counts every scored word in
it as wrong. Lines with no prediction count as wrong too. Both are reported."""
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict


def load_predictions(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                r = json.loads(line)
                out[r["id"]] = r["text"]
            else:
                i, _, text = line.partition("\t")
                out[i] = text
    return out


def words(text):
    return unicodedata.normalize("NFC", text).split()


def cells_of(row):
    """-> [(cell, word index)] for the line's scored words: cell names as in SPEC.md"""
    if row.get("cats"):
        return [(f"{row['set']} | {row['src']} | {c}", i) for i, c in zip(row["pts"], row["cats"])]
    if row["set"] == "collide":
        return [(f"collide | {row['key']}", i) for i in row["pts"]]
    return [(f"{row['set']} | {row['src']}", i) for i in row["pts"]]


def score(bench_path, pred_path):
    pred = load_predictions(pred_path)
    wrong, n = Counter(), Counter()
    missing = misaligned = 0
    with open(bench_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref = words(row["text"])
            got = pred.get(row["id"])
            if got is None:
                missing += 1
            hyp = words(got) if got is not None else None
            if hyp is not None and len(hyp) != len(ref):
                misaligned += 1
                hyp = None
            for cell, i in cells_of(row):
                n[cell] += 1
                wrong[cell] += hyp is None or hyp[i] != ref[i]
    return wrong, n, missing, misaligned


def rate(w, n):
    p = w / n if n else 0.0
    return p, 2 * math.sqrt(p * (1 - p) / n) if n else 0.0


def summaries(wrong, n):
    """pooled rows for groups of cells (a cell's own row stays the result; these are for reading)"""
    out = defaultdict(lambda: [0, 0])
    keys = [c for c in n if c.startswith("collide | ")]
    for c in n:
        parts = c.split(" | ")
        if parts[0] == "switch":
            base = "Korean base" if parts[2].startswith("KO") or parts[2].endswith("KO base") else "English base"
            out[f"switch, {base} (all points)"][0] += wrong[c]
            out[f"switch, {base} (all points)"][1] += n[c]
    if keys:
        macro = sum(wrong[k] / n[k] for k in keys) / len(keys)
        out[f"collide, mean over {len(keys)} keys"] = [macro, None]
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    if len(args) < 2:
        sys.exit(__doc__)
    json_out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if json_out in args:
        args.remove(json_out)
    wrong, n, missing, misaligned = score(args[0], args[1])
    print(f"{'cell':<46}{'wrong / points':>18}{'rate':>9}{'± 2 SE':>9}")
    for c in sorted(n):
        p, two_se = rate(wrong[c], n[c])
        print(f"{c:<46}{f'{wrong[c]} / {n[c]}':>18}{100 * p:>8.2f}%{100 * two_se:>8.2f}")
    print()
    for name, (w, m) in summaries(wrong, n).items():
        if m is None:
            print(f"{name:<46}{'':>18}{100 * w:>8.2f}%")
        else:
            p, two_se = rate(w, m)
            print(f"{name:<46}{f'{w} / {m}':>18}{100 * p:>8.2f}%{100 * two_se:>8.2f}")
    print(f"\nlines without a prediction: {missing}; lines whose word count differs: {misaligned}")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"cells": {c: {"wrong": wrong[c], "points": n[c]} for c in sorted(n)},
                       "missing_lines": missing, "misaligned_lines": misaligned}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
