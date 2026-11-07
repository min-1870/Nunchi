"""more number words in English text: words with digits and letters (5km, gpt4, 3rd, z68) in real English lines.
The switching set's "number, EN base" cell has 241 dev points from synthetic lines (IMDB / Reddit with Korean
put in) and its errors are one or two, too few to see anything. Here: WikiText-103 and Reddit lines (the big
dumps under data/raw/), cut as en_marked.collision_pool cuts them (20 words around the first number word),
minus every line in any data/sent/en_marked*.txt (en_marked trains on lines from this same text) and in the eval
sets. Every number word of a line is a point; WANT points per source, both halves; dev / test by hash of the
line, as the other sets.

    python -u experiments/num_en_sets.py   -> data/eval/num_en_set.jsonl  {"split", "src", "w", "pts": [idx]}"""
import json
import random
import re
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (before pyarrow: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import source_sets  # noqa: E402

from nunchi.sentence_data import OUT as SENT, SPLIT, TOKEN_V2  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

EVAL = SENT.parent / "eval"
OUT = EVAL / "num_en_set.jsonl"
WANT = 3000  # points per source, both halves
NUM = re.compile(r"(?=.*\d)(?=.*[A-Za-z])")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import pyarrow.parquet as pq
    used = set()
    for f in sorted(SENT.glob("en_marked*.txt")):
        used.update(line.rstrip("\n") for line in f.open(encoding="utf-8"))
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "collision_eval.jsonl", "collision_test.jsonl"):
        used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in (EVAL / f).open(encoding="utf-8"))
    say(f"{len(used):,} lines excluded")
    raw = SENT.parent / "raw"
    sources = {"wikitext": [(p, "text") for p in sorted((raw / "wikitext103").glob("*.parquet"))],
               "reddit": [(p, "body") for p in sorted((raw / "reddit_big").glob("*.parquet"))]}
    rows = []
    for src, files in sources.items():
        got = 0
        files = files[:]
        random.Random(1).shuffle(files)  # not only the first file's topics
        for f, col in files:
            for text in pq.read_table(f, columns=[col]).column(col).to_pylist():
                for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
                    toks = [t for t in filter(None, SPLIT.split(re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", sent)))
                            if TOKEN_V2.match(t)]
                    if len(toks) < 3:
                        continue
                    hit = next((i for i, t in enumerate(toks) if NUM.match(t)), None)
                    if hit is None:
                        continue
                    lo = max(0, min(hit - 10, len(toks) - 20))
                    line = " ".join(toks[lo:lo + 20])
                    if line in used:
                        continue
                    used.add(line)
                    s = to_sentence(line)
                    pts = [i for i, (k, _) in enumerate(s) if NUM.match(k)]
                    if not pts:
                        continue
                    rows.append({"split": source_sets._half(line), "src": src,
                                 "w": [[k, "".join(map(str, y))] for k, y in s], "pts": pts})
                    got += len(pts)
                    if got >= WANT:
                        break
                if got >= WANT:
                    break
            say(f"{src}: {got:,}/{WANT:,} points ({100 * got / WANT:.0f}%) after {f.name}")
            if got >= WANT:
                break
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    halves = Counter((r["split"], r["src"]) for r in rows for _ in r["pts"])
    shapes = Counter(re.sub(r"\d+", "9", r["w"][i][0].lower()) for r in rows for i in r["pts"])
    say(f"{OUT}: {len(rows):,} lines; points {dict(halves)}; shapes: "
        + ", ".join(f"{w} {n}" for w, n in shapes.most_common(25)))


if __name__ == "__main__":
    main()
