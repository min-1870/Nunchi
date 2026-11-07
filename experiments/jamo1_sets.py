"""single jamo in Korean text (ㅋ, ㅠ, ㄴ, ㅗ): the other side of ambiguous English. Their keys are one Latin letter
(z, b, s, h), so pushing short English in Korean text toward English can turn them into Latin; the main set holds
only 142 of them on dev. Held-out KcBERT comments (after line 2.4M, as amb_en_sets) and mC4 validation lines,
cleaned and cut as source_sets.held_out_lines, Korean-base lines only, minus every eval-set line; every one-jamo
Hangul word is a point. Dev / test halves by hash of the line.

    python -u experiments/jamo1_sets.py   -> data/eval/jamo1_set.jsonl  {"split", "src", "w", "pts": [idx]}"""
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

from nunchi.demo import compose  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

OUT = S.parent / "data" / "eval" / "jamo1_set.jsonl"
WANT = {"kcbert": 5000, "web": 1000}  # points per source, both halves
JAMO1 = re.compile(r"^[ㄱ-ㅣ]$")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "amb_en_set.jsonl", "num_en_set.jsonl", "collision_set.jsonl"):
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
            pts = [i for i, (k, y) in enumerate(sent) if set(y) == {1} and JAMO1.match(compose(k))]
            if not pts or not korean_base(sent):
                continue
            used.add(line)
            rows.append({"split": source_sets._half(line), "src": src,
                         "w": [[k, "".join(map(str, y))] for k, y in sent], "pts": pts})
            per.update(compose(sent[i][0]) for i in pts)
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
