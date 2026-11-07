"""one-word input (a search box, a one-word reply: 대박, ㅇㅇ, lol, thanks): the guard for the single-word training
samples (NUNCHI_WORDS, log 24). Sentence training data needs 2+ words (sentence_data.MIN_W), so the single-word
samples are the only place a model sees a word with nothing before or after it, and no eval cell measured that.
Three sources, every point is a whole line of one word:
  ko1  one-word KcBERT comments, held out (lines 2.4M+, as amb_en_sets)
  en1  one-word Reddit comments (reddit_raw.txt, then the subreddit dumps in data/raw/reddit_big: whole comments)
  mix1 English + Korean particle (apk의, LG가) taken out of held-out KcBERT text and typed alone
ko1 keeps words of at most 5 syllables (longer one-token comments are sentences typed without spaces).
Cleaned as the training lines are (sentence_data.clean, v3) with the word minimum lowered to 1; at most 3 copies of
the same word (ㅋㅋㅋ would fill the set). Dev / test halves by hash of the word.

    python -u experiments/word1_sets.py   -> data/eval/word1_set.jsonl  {"split", "src", "w", "pts": [0]}"""
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import source_sets  # noqa: E402
from amb_en_sets import raw  # noqa: E402

import nunchi.sentence_data as sd  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

OUT = S.parent / "data" / "eval" / "word1_set.jsonl"
WANT = {"ko1": 4000, "en1": 4000, "mix1": 2000}  # points, both halves
CAP = 3


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    kk = sd.ko_keys()
    sd.MIN_W = 1
    rows, seen = [], Counter()

    def add(src, tok):
        if seen[tok] >= CAP:
            return False
        sent = to_sentence(tok)
        if len(sent) != 1:
            return False
        seen[tok] += 1
        rows.append({"split": source_sets._half(tok), "src": src,
                     "w": [[k, "".join(map(str, y))] for k, y in sent], "pts": [0]})
        return True

    got = Counter()
    for text in raw("kcbert"):
        toks = sd.clean(text, kk, 3)
        if not toks:
            continue
        if got["ko1"] < WANT["ko1"] and len(toks) == 1 and sd.HANGUL.match(toks[0]) and len(toks[0]) <= 5:
            got["ko1"] += add("ko1", toks[0])
        if got["mix1"] < WANT["mix1"]:
            for t in toks:
                if sd.MIXED.match(t) and got["mix1"] < WANT["mix1"]:
                    got["mix1"] += add("mix1", t)
        if got["ko1"] >= WANT["ko1"] and got["mix1"] >= WANT["mix1"]:
            break
    say(f"kcbert done: {dict(got)}")
    import pyarrow.parquet as pq  # after torch: pyarrow first breaks the ROCm DLL load
    big = (text for f in sorted((S.parent / "data" / "raw" / "reddit_big").glob("*.parquet"))
           for text in pq.read_table(f, columns=["body"]).column("body").to_pylist() if text)
    for line in itertools.chain(source_sets.raw_lines("reddit"), big):
        toks = sd.clean(source_sets.APOS.sub("", line), kk, 3)
        if toks and len(toks) == 1 and sd.LATIN.match(toks[0]):
            got["en1"] += add("en1", toks[0])
            if got["en1"] >= WANT["en1"]:
                break
    say(f"points: {dict(got)}")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    halves = Counter((r["src"], r["split"]) for r in rows)
    say(f"{OUT}: {len(rows):,} lines, {dict(halves)}")
    for src in WANT:
        say(f"  {src}: " + ", ".join(r["w"][0][0] for r in rows if r["src"] == src)[:300])


if __name__ == "__main__":
    main()
