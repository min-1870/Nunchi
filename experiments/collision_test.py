"""test half of the key-collision check: collision_eval.jsonl (built with en_marked v2) has been read while choosing
recipes, so it serves as dev. Lines for the test half come from the big English text (WikiText-103, Reddit dumps),
cut as en_marked.collision_pool cuts them, minus every line in any data/sent/en_marked*.txt (en_marked v3 trains
on lines from this same text), in the eval sets or in the dev check. Per colliding key, TIMES lines for every word
of that key in the dev check (same key mix, more words), as many as there are.

    python -u experiments/collision_test.py   -> data/eval/collision_test.jsonl"""
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch  # noqa: F401  (before pyarrow: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S.parent))
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.en_marked import korean_edges, markers  # noqa: E402
from nunchi.sentence_data import OUT as SENT, SPLIT, TOKEN_V2  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

EVAL = SENT.parent / "eval"
DEV = EVAL / "collision_eval.jsonl"
OUT = EVAL / "collision_test.jsonl"
TIMES = 3


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import pyarrow.parquet as pq
    keys = {to_keys(w) for w, *_ in markers(*korean_edges())}
    base = lambda t: t if t in keys else re.sub(r"\d+$", "", t) if re.search(r"\d$", t) else None  # noqa: E731
    dev_lines = [json.loads(line)["w"] for line in DEV.open(encoding="utf-8")]
    want = Counter(base(k) for w in dev_lines for k, _ in w if base(k) in keys)
    used = {" ".join(k for k, _ in w) for w in dev_lines}
    for f in sorted(SENT.glob("en_marked*.txt")):
        used.update(line.rstrip("\n") for line in f.open(encoding="utf-8"))
    for f in ("source_sets.jsonl", "switch_sets.jsonl"):
        used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in (EVAL / f).open(encoding="utf-8"))
    say(f"dev check: {sum(want.values())} words, {len(want)} keys; {len(used):,} lines excluded")
    raw = SENT.parent / "raw"
    files = [(p, "text") for p in sorted((raw / "wikitext103").glob("*.parquet"))] + \
            [(p, "body") for p in sorted((raw / "reddit_big").glob("*.parquet"))]
    pool, seen = defaultdict(list), set()
    for n, (f, col) in enumerate(files, 1):
        for text in pq.read_table(f, columns=[col]).column(col).to_pylist():
            for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
                toks = [t for t in filter(None, SPLIT.split(re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", sent)))
                        if TOKEN_V2.match(t)]
                if len(toks) < 3:
                    continue
                for i, t in enumerate(toks):  # the window en_marked.collision_pool cuts around the first hit
                    k = base(t)
                    if k in keys:
                        lo = max(0, min(i - 10, len(toks) - 20))
                        line = " ".join(toks[lo:lo + 20])
                        if line not in seen and line not in used and k in want:
                            seen.add(line)
                            pool[k].append(line)
                        break
        say(f"{n}/{len(files)} {f.parent.name}/{f.name}: {sum(map(len, pool.values())):,} candidate lines")
    rnd = random.Random(1)
    out = []
    for k, n in want.most_common():
        have = pool.get(k, [])
        rnd.shuffle(have)
        out += have[:TIMES * n]
        say(f"  {k}: dev {n} words, candidates {len(have):,}, taken {min(len(have), TIMES * n)}")
    rnd.shuffle(out)
    with OUT.open("w", encoding="utf-8") as fh:
        for line in out:
            fh.write(json.dumps({"split": "test", "w": [[k, "".join(map(str, y))] for k, y in to_sentence(line)]},
                                ensure_ascii=False) + "\n")
    got = Counter(base(t) for line in out for t in line.split() if base(t) in keys)
    say(f"{OUT}: {len(out)} lines, {sum(got.values())} colliding words; " + ", ".join(f"{k} {v}" for k, v in got.most_common()))


if __name__ == "__main__":
    main()
