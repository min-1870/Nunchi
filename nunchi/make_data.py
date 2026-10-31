"""Build fastText train/valid/test files: one word's raw key sequence per line.

ko: Hangul words from NSMC train reviews, KcBERT news comments and Tatoeba Korean (count_corpus),
    each source weighted equally -> 두벌식 keys
en: top English words from wordfreq (lowercase), plus cased forms seen in real text:
    Tatoeba sentences (The, Tom, I) and English inside the Korean text (TV, OST, CG, KBS).
    Shift changes the keys, so "The" and "the" are separate rows.
Split is by word type, so valid/test keys are never seen in train.

Outputs per split:
  {split}.txt        __label__ko dkssud
  {split}_shape.txt  __label__ko dkssud _S_full   (decoded-shape token added)
  {split}.tsv        label, keys, word, freq, shape (for reports)
"""
import bz2
import random
import re
from collections import Counter
from pathlib import Path

import wordfreq

from nunchi.dubeolsik import shape, to_keys

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nsmc_train.txt"
TATOEBA = ROOT / "data" / "raw" / "eng_sentences.tsv.bz2"
OUT = ROOT / "data" / "ft"
N_PER_LANG = 100_000
N_EN = 50_000  # wordfreq past this is mostly noise (zl, apsb)
SEED = 0
# Korean text sources counted by count_corpus (FrequencyWords stays an outside eval set, never trained on)
KO_SOURCES = ("nsmc", "kcbert", "tatoeba")

HANGUL_RUN = re.compile(r"[가-힣ㄱ-ㅣ]+")
EN_WORD = re.compile(r"^[a-z]+$")
LATIN_RUN = re.compile(r"[A-Za-z]+")
ACRONYM = re.compile(r"^[A-Z]{2,5}$")
MASK = re.compile(r"^O+$")  # NSMC censors words as OOO
MIN_TATOEBA, MIN_NSMC = 5, 3
AMBIG_RATIO = 10


def korean_words() -> list[tuple[str, float]]:
    """Top Korean words over the sources, each source weighted equally (relative frequency summed),
    so news comments do not drown out reviews and sentences."""
    from nunchi.count_corpus import load

    combined = Counter()
    for source in KO_SOURCES:
        counts = load(source, "hangul")
        total = sum(counts.values())
        for w, c in counts.items():
            combined[w] += c / total
    return combined.most_common(N_PER_LANG)


def english_words() -> list[tuple[str, float]]:
    from nunchi.count_corpus import load

    vocab = [w for w in wordfreq.top_n_list("en", 200_000) if EN_WORD.match(w)]
    known = set(vocab)
    out = {w: wordfreq.word_frequency(w, "en") for w in vocab[:N_EN]}

    tatoeba = Counter()
    with bz2.open(TATOEBA, "rt", encoding="utf-8") as f:
        for line in f:
            tatoeba.update(LATIN_RUN.findall(line.rstrip("\n").split("\t")[-1]))
    # English inside Korean text (reviews, comments): TV, OST, CG, KBS
    nsmc = Counter()
    for source in KO_SOURCES:
        nsmc.update({w: c for w, c in load(source, "latin").items() if not MASK.match(w)})

    # Korean text also holds Korean typed in the wrong mode (dl = 이, gks = 한); never call those English
    ko_keys = {to_keys(w) for w, _ in korean_words()}
    nsmc = Counter({w: c for w, c in nsmc.items() if w not in ko_keys and w.lower() not in ko_keys})

    # cased forms of real English words; acronyms (EBS, CSI) only need to recur in reviews
    by_lower = Counter()
    for w, c in tatoeba.items():
        by_lower[w.lower()] += c
    for counts, min_count in ((tatoeba, MIN_TATOEBA), (nsmc, MIN_NSMC)):
        for w, c in counts.items():
            if w in out or c < min_count or not (w.lower() in known or (counts is nsmc and ACRONYM.match(w))):
                continue
            base = wordfreq.word_frequency(w.lower(), "en") or 1e-7
            share = tatoeba[w] / by_lower[w.lower()] if by_lower[w.lower()] else 1.0
            out[w] = base * max(share, 0.01)
    return sorted(out.items(), key=lambda x: -x[1])


def main():
    ko = {to_keys(w): (w, c) for w, c in korean_words()}
    en = {w: (w, c) for w, c in english_words()}
    # A key both languages produce goes to the one where it is far more common (dl: 이 >> "dl");
    # close calls (so = 내 / "so") keep a row in both languages, so the model learns ~50/50 and
    # context decides, instead of the key never being seen.
    ko_total, en_total = sum(c for _, c in ko.values()), sum(c for _, c in en.values())
    both, kept = [], []
    for k in sorted(ko.keys() & en.keys()):
        ratio = (ko[k][1] / ko_total) / (en[k][1] / en_total)
        if ratio >= AMBIG_RATIO:
            del en[k]
        elif ratio <= 1 / AMBIG_RATIO:
            del ko[k]
        else:
            both.append(f"{k}={ko[k][0]}/{en[k][0]}")
            continue
        kept.append(f"{k}={'ko' if k in ko else 'en'}")
    print("collisions given to one side:", kept[:30])
    print("collisions kept in both:", both[:30])
    print(f"ko={len(ko)} en={len(en)} both={len(both)} one side={len(kept)}")

    rows = [("ko", k, w, c) for k, (w, c) in ko.items()] + [("en", k, w, c) for k, (w, c) in en.items()]
    # case variants (the / The / THE) share a split, so test never sees a recased train word
    groups = sorted({k.lower() for _, k, _, _ in rows})
    random.Random(SEED).shuffle(groups)
    n = len(groups)
    split_of = {g: "train" if i < n * 0.8 else "valid" if i < n * 0.9 else "test" for i, g in enumerate(groups)}
    splits = {name: [r for r in rows if split_of[r[1].lower()] == name] for name in ("train", "valid", "test")}
    for part in splits.values():
        random.Random(SEED).shuffle(part)

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
