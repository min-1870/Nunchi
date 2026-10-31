"""Token counts from the Korean text sources, written once and reused by make_data / the tagger.

Sources (each counted separately, so one can be dropped later, e.g. for licensing):
  nsmc    : data/raw/nsmc_train.txt (movie reviews; nsmc_test stays held out)
  kcbert  : data/raw/kcbert/kcbert-train.tar.gz{aa,ab,ac} (news comments, CC BY-SA 4.0), streamed,
            first KCBERT_LINES lines
  tatoeba : data/raw/kor_sentences.tsv.bz2 (Korean sentences, CC BY 2.0 FR)

Per source, three tables of whitespace tokens with punctuation stripped from the ends:
  hangul : only Hangul (syllables or jamo)           -> Korean words
  latin  : only Latin letters, as written (case kept) -> English inside Korean text
  mixed  : Latin then Hangul (SF는, GitHub에)         -> English + particle, for the key tagger
Output: data/counts/{source}_{table}.tsv  (token<TAB>count, count >= MIN_COUNT)

    .venv/Scripts/python -m nunchi.count_corpus [source ...]
"""
import bz2
import re
import subprocess
import sys
from collections import Counter

from nunchi.train_fasttext import ROOT

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "counts"
KCBERT_LINES = 30_000_000
MIN_COUNT = 2
EDGE = re.compile(r"^[^\w가-힣ㄱ-ㅣ]+|[^\w가-힣ㄱ-ㅣ]+$")
HANGUL = re.compile(r"^[가-힣ㄱ-ㅣ]+$")
LATIN = re.compile(r"^[A-Za-z]+$")
MIXED = re.compile(r"^[A-Za-z]+[가-힣]+$")


def lines(source):
    if source == "nsmc":
        with (RAW / "nsmc_train.txt").open(encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    yield parts[1]
    elif source == "tatoeba":
        with bz2.open(RAW / "kor_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n").split("\t")[-1]
    elif source == "kcbert":
        parts = " ".join(str(p) for p in sorted((RAW / "kcbert").glob("kcbert-train.tar.gz*")))
        proc = subprocess.Popen(f"cat {parts} | tar -xzOf -", shell=True, stdout=subprocess.PIPE)
        for i, raw in enumerate(proc.stdout):
            if i >= KCBERT_LINES:
                break
            yield raw.decode("utf-8", "replace").rstrip("\n")
        proc.kill()
    else:
        raise ValueError(source)


def count(source):
    tables = {"hangul": Counter(), "latin": Counter(), "mixed": Counter()}
    for n, text in enumerate(lines(source)):
        for tok in text.split():
            tok = EDGE.sub("", tok)
            if HANGUL.match(tok):
                tables["hangul"][tok] += 1
            elif LATIN.match(tok):
                tables["latin"][tok] += 1
            elif MIXED.match(tok):
                tables["mixed"][tok] += 1
        if n and n % 5_000_000 == 0:
            print(f"  {source}: {n:,} lines", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, c in tables.items():
        rows = [(t, k) for t, k in c.most_common() if k >= MIN_COUNT]
        with (OUT / f"{source}_{name}.tsv").open("w", encoding="utf-8") as f:
            f.writelines(f"{t}\t{k}\n" for t, k in rows)
        print(f"{source}_{name}: {len(rows):,} types, {sum(k for _, k in rows):,} tokens", flush=True)


def load(source, table):
    path = OUT / f"{source}_{table}.tsv"
    with path.open(encoding="utf-8") as f:
        return Counter({t: int(k) for t, k in (line.rstrip("\n").split("\t") for line in f)})


if __name__ == "__main__":
    for s in sys.argv[1:] or ["nsmc", "tatoeba", "kcbert"]:
        count(s)
