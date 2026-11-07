"""typos at the moment the language is decided: the main set's sentences with typos put in, scored on the typo'd
words only. Text people post has few typos (they fix them first), but at Space the word is not fixed yet, and a typo
that flips the language turns the whole word into the other script. Training puts in the first kind (15% of
words, nunchi.tagger.typo); there was no cell for either kind since the sets of 2026-11-03.

  pos    one keyboard typo (a neighbour key, a dropped / inserted key, two keys swapped: nunchi.tagger.typo, labels
         kept aligned) in P_POS of the words of 2+ keys, every source; the right output is the typo'd keys in the
         word's language
  shift  Shift still held after a double consonant: the vowel key after Q W E R T comes out upper case (까 Rk ->
         RK, 싸 Tk -> TK); o / p are left out (Shift gives ㅒ / ㅖ there, another vowel). Korean words of the
         Korean sources, every word that has the pattern; composes to the same Hangul, so the right output is the
         word as meant

Sentences: the main set's (source_sets.jsonl), both halves as they are; the context words stay as written.

    python -u experiments/typo_sets.py   -> data/eval/typo_set.jsonl  {"split", "src", "kind", "w", "pts": [idx]}"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import source_sets  # noqa: E402

from nunchi.tagger import typo  # noqa: E402

OUT = S.parent / "data" / "eval" / "typo_set.jsonl"
P_POS = 0.15
KO_SRC = ("nsmc", "kcbert", "web")
SHIFT = re.compile(r"(?<=[QWERT])[hjklyuinmb]")


def main():
    rnd = random.Random(7)
    rows, cnt = [], Counter()
    for split in ("dev", "test"):
        for src, sents in source_sets.load(split).items():
            for s in sents:
                # position typos
                w, pts = [], []
                for i, (k, y) in enumerate(s):
                    if len(k) >= 2 and rnd.random() < P_POS:
                        k2, y2 = typo(k, y, rnd)
                        if k2 != k:
                            w.append((k2, y2)); pts.append(i)
                            continue
                    w.append((k, y))
                if pts:
                    rows.append({"split": split, "src": src, "kind": "pos", "w": [[k, "".join(map(str, y))] for k, y in w],
                                 "pts": pts})
                    cnt[(split, "pos")] += len(pts)
                # Shift held into the vowel
                if src in KO_SRC:
                    w, pts = [], []
                    for i, (k, y) in enumerate(s):
                        k2 = SHIFT.sub(lambda m: m.group(0).upper(), k) if set(y) == {1} else k
                        if k2 != k:
                            pts.append(i)
                        w.append((k2, y))
                    if pts:
                        rows.append({"split": split, "src": src, "kind": "shift",
                                     "w": [[k, "".join(map(str, y))] for k, y in w], "pts": pts})
                        cnt[(split, "shift")] += len(pts)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{OUT}: {len(rows):,} lines; points {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
