"""Korean words whose keys are a common English word (해 = go, 내 = so, 새 = to, 소 = th): the other side of ambiguous
English, by the same rule (amb_en_eval.amb: the Hangul reading 200+ times in KcBERT + NSMC) but with the English
reading a common word (zipf 4.5+, EN_ZIPF), on Korean-labelled words (jamo-only words are jamo1_sets.py). Pushing short English in Korean text toward
English can turn these into Latin. Held-out KcBERT comments (after line 2.4M, as amb_en_sets) and mC4 validation lines,
cleaned and cut as source_sets.held_out_lines, Korean-base lines only, minus every eval-set line; every such
word is a point. Dev / test halves by hash of the line.

    python -u experiments/amb_ko_sets.py   -> data/eval/amb_ko_set.jsonl  {"split", "src", "w", "pts": [idx]}"""
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
from amb_en_eval import amb  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402
from amb_en_sets import korean_base, raw  # noqa: E402

from nunchi.demo import compose  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

OUT = S.parent / "data" / "eval" / "amb_ko_set.jsonl"
WANT = {"kcbert": 5000, "web": 1000}  # points per source, both halves
SYL = re.compile(r"^[가-힣]+$")
# amb's zipf 3 lets in keys that are English only as abbreviations (다 = ek, 이 = dl, 그 = rm), which made up most
# of the words and are no real risk; 4.5 keeps the readings that are common English words (go, so, to, do, en, rid)
EN_ZIPF = 4.5


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "amb_en_set.jsonl", "num_en_set.jsonl", "collision_set.jsonl",
              "jamo1_set.jsonl"):
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
            pts = [i for i, (k, y) in enumerate(sent) if set(y) == {1} and SYL.match(compose(k)) and amb(k)
                   and zipf_frequency(k.lower(), "en") >= EN_ZIPF]
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
    say(f"{OUT}: {len(rows):,} lines, {dict(halves)}; " + ", ".join(f"{w} {n}" for w, n in per.most_common(25)))


if __name__ == "__main__":
    main()
