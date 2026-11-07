"""words that hold both languages with no space between (절충trade, zonna비싼데, km리뷰, 채널A, MBC는): 262 of the main
set's dev words, about 11% wrong for TR4-10K, against 0.1% for the rest. Held-out KcBERT comments (after line 2.4M,
as amb_en_sets) and mC4 validation lines, cleaned and cut as source_sets.held_out_lines, Korean-base lines only,
minus every eval-set line; every word whose keys carry both labels (digits aside) is a point, scored as the others:
the typed text must split at the right key. Dev / test halves by hash of the line.

    python -u experiments/fused_sets.py   -> data/eval/fused_set.jsonl  {"split", "src", "w", "pts": [idx]}"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import source_sets  # noqa: E402
from amb_en_sets import korean_base, raw  # noqa: E402

from nunchi.sentence_tagger import to_sentence  # noqa: E402

OUT = S.parent / "data" / "eval" / "fused_set.jsonl"
WANT = {"kcbert": 2000, "web": 1000}  # points per source, both halves
DIGIT = re.compile(r"\d")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "amb_en_set.jsonl", "num_en_set.jsonl", "collision_set.jsonl",
              "jamo1_set.jsonl", "amb_ko_set.jsonl", "typo_set.jsonl", "acro_set.jsonl"):
        p = OUT.parent / f
        if p.exists():
            used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in p.open(encoding="utf-8"))
    say(f"{len(used):,} eval-set lines excluded")
    rows, per = [], Counter()
    for src, want in WANT.items():
        got = lines = 0
        for line in source_sets.iter_held_out(src, texts=raw(src)):
            if line in used:
                continue
            sent = to_sentence(line)
            pts = [i for i, (k, y) in enumerate(sent) if len(set(y)) > 1 and not DIGIT.search(k)]
            if not pts or not korean_base(sent):
                continue
            used.add(line)
            rows.append({"split": source_sets._half(line), "src": src,
                         "w": [[k, "".join(map(str, y))] for k, y in sent], "pts": pts})
            per.update(sent[i][0] for i in pts)
            got += len(pts)
            lines += 1
            if lines % 500 == 0:
                say(f"{src}: {got:,}/{want:,} points ({100 * got / want:.0f}%)")
            if got >= want:
                break
        say(f"{src}: {got:,} points in {lines:,} lines")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    halves = Counter(r["split"] for r in rows for _ in r["pts"])
    say(f"{OUT}: {len(rows):,} lines, {dict(halves)}; " + ", ".join(f"{w} {n}" for w, n in per.most_common(20)))


if __name__ == "__main__":
    main()
