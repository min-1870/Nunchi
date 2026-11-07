"""key collisions, per key: English words in real English lines whose keys equal a Korean marker's (WW = ㅉㅉ, mm =
ㅡㅡ, z = ㅋ, h = ㅗ, gg = ㅎㅎ), a digit suffix allowed (WW2) as in collision_eval.

The first sets were lopsided: dev (collision_eval.jsonl) held 186 words, 68 of them WW, which models rarely get
wrong, while the keys they do get wrong had 13-21 words each; the test half (collision_test.py) came mostly from
WikiText, where h / mm / z follow a number as units (65 km h, 300 mm), far easier than the casual dev lines. Here
both halves come from the same casual text (the Reddit dumps under data/raw/reddit_big), up to PER_KEY words per
key and half, so that each key can be scored on its own and the cell is the mean over keys (macro average), not
a pool led by the most common key. Lines: cut as en_marked.collision_pool cuts them (20 words around the first
hit), minus every line in any data/sent/en_marked*.txt (en_marked trains on lines from this same text) and in
the eval sets. Keys with fewer than MIN_KEY words in a half are reported and left out of that half.

    python -u experiments/collision_sets.py   -> data/eval/collision_set.jsonl  {"split", "key", "w", "pts": [idx]}"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch  # noqa: F401  (before pyarrow: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import source_sets  # noqa: E402

from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.en_marked import korean_edges, markers  # noqa: E402
from nunchi.sentence_data import OUT as SENT, SPLIT, TOKEN_V2  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

EVAL = SENT.parent / "eval"
OUT = EVAL / "collision_set.jsonl"
PER_KEY = 60
MIN_KEY = 15


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import pyarrow.parquet as pq
    keys = {to_keys(w) for w, *_ in markers(*korean_edges())}
    base = lambda t: t if t in keys else re.sub(r"\d+$", "", t) if re.search(r"\d$", t) else None  # noqa: E731
    used = set()
    for f in sorted(SENT.glob("en_marked*.txt")):
        used.update(line.rstrip("\n") for line in f.open(encoding="utf-8"))
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "collision_eval.jsonl", "collision_test.jsonl",
              "amb_en_set.jsonl", "num_en_set.jsonl"):
        if (EVAL / f).exists():
            used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in (EVAL / f).open(encoding="utf-8"))
    say(f"{len(keys)} marker keys; {len(used):,} lines excluded")
    got = defaultdict(Counter)  # key -> {half: words}
    rows = []
    files = sorted((SENT.parent / "raw" / "reddit_big").glob("*.parquet"))
    for n, f in enumerate(files, 1):
        for text in pq.read_table(f, columns=["body"]).column("body").to_pylist():
            for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
                toks = [t for t in filter(None, SPLIT.split(re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", sent)))
                        if TOKEN_V2.match(t)]
                if len(toks) < 3:
                    continue
                hit = next((i for i, t in enumerate(toks) if base(t) in keys), None)
                if hit is None:
                    continue
                lo = max(0, min(hit - 10, len(toks) - 20))
                line = " ".join(toks[lo:lo + 20])
                k, half = base(toks[hit]), source_sets._half(line)
                if line in used or got[k][half] >= PER_KEY:
                    continue
                used.add(line)
                s = to_sentence(line)
                pts = [i for i, (t, y) in enumerate(s) if base(t) == k and set(y) == {0}]
                if not pts:
                    continue
                rows.append({"split": half, "key": k, "w": [[t, "".join(map(str, y))] for t, y in s], "pts": pts})
                got[k][half] += len(pts)
        say(f"{n}/{len(files)} {f.name}: {sum(sum(c.values()) for c in got.values()):,} words, "
            f"{sum(1 for c in got.values() if min(c['dev'], c['test']) >= MIN_KEY)} keys with {MIN_KEY}+ in both halves")
    keep = {k for k, c in got.items() if min(c["dev"], c["test"]) >= MIN_KEY}
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            if r["key"] in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    say(f"{OUT}: {len(keep)} keys kept (dev / test words): " +
        ", ".join(f"{k} {got[k]['dev']}/{got[k]['test']}" for k in sorted(keep, key=lambda k: -sum(got[k].values()))))
    short = {k: c for k, c in got.items() if k not in keep}
    say("too few (dev / test): " + ", ".join(f"{k} {c['dev']}/{c['test']}" for k, c in
                                              sorted(short.items(), key=lambda x: -sum(x[1].values()))))
    say("no line at all: " + " ".join(sorted(keys - set(got))))


if __name__ == "__main__":
    main()
