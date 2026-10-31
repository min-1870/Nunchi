"""Build fastText train/valid/test files: one word's raw key sequence per line.

ko: Hangul eojeol from NSMC reviews -> 두벌식 keys
en: top English words from wordfreq (lowercase a-z only)
Split is by word type, so valid/test keys are never seen in train.

Outputs per split:
  {split}.txt        __label__ko dkssud
  {split}_shape.txt  __label__ko dkssud _S_full   (decoded-shape token added)
  {split}.tsv        label, keys, word, freq, shape (for reports)
"""
import random
import re
from collections import Counter
from pathlib import Path

import wordfreq

from nunchi.dubeolsik import shape, to_keys

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nsmc_train.txt"
OUT = ROOT / "data" / "ft"
N_PER_LANG = 50_000
SEED = 0

HANGUL_RUN = re.compile(r"[가-힣ㄱ-ㅣ]+")
EN_WORD = re.compile(r"^[a-z]+$")


def korean_words() -> list[tuple[str, float]]:
    counts = Counter()
    with RAW.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                counts.update(HANGUL_RUN.findall(parts[1]))
    return [(w, float(c)) for w, c in counts.most_common(N_PER_LANG)]


def english_words() -> list[tuple[str, float]]:
    words = [w for w in wordfreq.top_n_list("en", 200_000) if EN_WORD.match(w)]
    return [(w, wordfreq.word_frequency(w, "en")) for w in words[:N_PER_LANG]]


def main():
    ko = {to_keys(w): (w, c) for w, c in korean_words()}
    en = {w: (w, c) for w, c in english_words()}
    ambiguous = sorted(ko.keys() & en.keys())
    print("ambiguous sample:", [f"{k}={ko[k][0]}" for k in ambiguous[:20]])
    for k in ambiguous:
        del ko[k], en[k]
    print(f"ko={len(ko)} en={len(en)} ambiguous(dropped)={len(ambiguous)}")

    rows = [("ko", k, w, c) for k, (w, c) in ko.items()] + [("en", k, w, c) for k, (w, c) in en.items()]
    random.Random(SEED).shuffle(rows)
    n = len(rows)
    splits = {"train": rows[: int(n * 0.8)], "valid": rows[int(n * 0.8): int(n * 0.9)], "test": rows[int(n * 0.9):]}

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        shapes = [shape(k) for _, k, _, _ in part]
        with (OUT / f"{name}.txt").open("w", encoding="utf-8") as f:
            f.writelines(f"__label__{lab} {k}\n" for lab, k, _, _ in part)
        with (OUT / f"{name}_shape.txt").open("w", encoding="utf-8") as f:
            f.writelines(f"__label__{lab} {k} _S_{s}\n" for (lab, k, _, _), s in zip(part, shapes))
        with (OUT / f"{name}.tsv").open("w", encoding="utf-8") as f:
            f.writelines(f"{lab}\t{k}\t{w}\t{c}\t{s}\n" for (lab, k, w, c), s in zip(part, shapes))
        print(name, len(part), Counter(f"{r[0]}:{s}" for r, s in zip(part, shapes)))


if __name__ == "__main__":
    main()
