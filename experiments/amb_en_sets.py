"""more ambiguous English for amb_en_eval: real Korean sentences holding an English word that is also a common
Korean word (so = 내, to = 새, a = ㅁ, go = 해; the rule in amb_en_eval.amb). The switching set holds 606 such
words on dev, too few to see a change under ~2.6 %p. Held-out text only:
  kcbert  news comments after the first 2.4M lines of the dump (training reads the first 2M, the main set
          2M-2.4M)
  web     the mC4 ko validation file (training reads train shards 0-49)
lines cleaned and cut as source_sets.held_out_lines does (lines in a training pool dropped), minus every line of
the main and switching sets, and only Korean-base lines (more Korean words than English: the switching set's
real lines are Korean-base too; English-base lines and web markup with a stray N are not this case); every
ambiguous English word in a line is a point. Words as they come (no cap per
word), so the mix is what Korean text holds. Dev / test halves by hash of the line, as the other sets.

    python -u experiments/amb_en_sets.py   -> data/eval/amb_en_set.jsonl  {"split", "src", "w", "pts": [idx]}"""
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
from amb_en_eval import amb  # noqa: E402

from nunchi.sentence_tagger import to_sentence  # noqa: E402

OUT = S.parent / "data" / "eval" / "amb_en_set.jsonl"
WANT = {"kcbert": 4000, "web": 1000}  # ambiguous words per source, both halves
KC_FROM, KC_TO = 2_400_000, 12_000_000


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def korean_base(sent):
    """more Korean words than English ones (digit words aside): English words in Korean text, not English text"""
    ys = [set(y) for k, y in sent if not any(c.isdigit() for c in k)]
    return sum(y == {1} for y in ys) > sum(y == {0} for y in ys)


def raw(src):
    if src == "web":
        yield from source_sets.raw_lines("web")
        return
    import nunchi.sentence_data as sd
    sd.SENT_KCBERT = KC_TO
    for i, text in enumerate(itertools.islice(sd.korean_lines("kcbert"), KC_FROM, None)):
        if i % 500_000 == 0:
            say(f"  kcbert line {KC_FROM + i:,}")
        yield text


def main():
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl"):
        for line in (OUT.parent / f).open(encoding="utf-8"):
            used.add(" ".join(k for k, _ in json.loads(line)["w"]))
    say(f"{len(used):,} eval-set lines excluded")
    rows, per_word = [], Counter()
    for src, want in WANT.items():
        got = lines = 0
        for line in source_sets.iter_held_out(src, texts=raw(src)):
            if line in used:
                continue
            sent = to_sentence(line)
            pts = [i for i, (k, y) in enumerate(sent) if set(y) == {0} and amb(k)]
            if not pts or not korean_base(sent):
                continue
            used.add(line)
            rows.append({"split": source_sets._half(line), "src": src,
                         "w": [[k, "".join(map(str, y))] for k, y in sent], "pts": pts})
            per_word.update(sent[i][0].lower() for i in pts)
            got += len(pts)
            lines += 1
            if lines % 200 == 0:
                say(f"{src}: {got:,}/{want:,} words in {lines:,} lines ({100 * got / want:.0f}%)")
            if got >= want:
                break
        say(f"{src}: {got:,} words in {lines:,} lines")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    halves = Counter(r["split"] for r in rows for _ in r["pts"])
    say(f"{OUT}: {len(rows):,} lines, {sum(halves.values()):,} words ({dict(halves)}); most common: "
        + ", ".join(f"{w} {n}" for w, n in per_word.most_common(25)))


if __name__ == "__main__":
    main()
