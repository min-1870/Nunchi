"""English casual typed text, the counterpart of the Korean comment / review sources (KcBERT, NSMC):
  imdb   : IMDB movie reviews (stanfordnlp/imdb, train + test), split into sentences  <-> NSMC
  reddit : Reddit comments (google-research-datasets/go_emotions, simplified, all splits)  <-> KcBERT
Raw text as typed (typos, case, numbers kept); only markup and anonymisation masks are undone.
    python -m nunchi.en_casual        # data/raw/{imdb,goemotions}/*.parquet -> data/sent/{imdb,reddit}_raw.txt
Download (parquet from the Hugging Face API):
    https://huggingface.co/api/datasets/stanfordnlp/imdb/parquet/plain_text/{train,test}/0.parquet
    https://huggingface.co/api/datasets/google-research-datasets/go_emotions/parquet/simplified/{train,validation,test}/0.parquet"""
import hashlib
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "sent"
NAMES = ["James", "Mary", "John", "Sarah", "Mike", "Emma", "David", "Anna", "Chris", "Kate", "Tom", "Lisa"]
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def texts(name, splits):
    for s in splits:
        yield from pq.read_table(RAW / name / f"{s}.parquet").column("text").to_pylist()


def one_line(t):
    return re.sub(r"\s+", " ", t).strip()


def imdb():
    for t in texts("imdb", ["train", "test"]):
        t = re.sub(r"<br\s*/?>", " ", t)
        yield from (s for s in map(one_line, SENT_END.split(t)) if s)


def reddit():
    for t in texts("goemotions", ["train", "validation", "test"]):
        if re.search(r"\[(?!NAME\])[A-Z]+\]", t):  # rare masks ([RELIGION] ...): no natural stand-in
            continue
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)  # the same stand-in names on every rebuild
        t = re.sub(r"\[NAME\]", lambda m, i=iter(range(99)): NAMES[(h + next(i)) % len(NAMES)], t)
        if t := one_line(t):
            yield t


def main():
    for name, gen in [("imdb", imdb), ("reddit", reddit)]:
        lines = list(dict.fromkeys(gen()))
        (OUT / f"{name}_raw.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(lines):,} lines, {sum(len(x.split()) for x in lines):,} words", flush=True)


if __name__ == "__main__":
    main()
