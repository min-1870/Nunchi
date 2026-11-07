"""the main eval set: the same number of words from each source of typed text, sentences as they come (Korean
only, English only or mixed, numbers and typos in place), so no source outweighs another by its size.
Weights between sources are applied when reporting, not by the set.

  source  text                          held out from training by
  nsmc    Korean movie reviews          NSMC test file (training reads nsmc_train)
  kcbert  Korean news comments          lines after the first 2M of the dump (training reads the first 2M)
  web     Korean web pages (mC4)        mC4 ko validation file (training reads train shards)
  imdb    English movie reviews         not used in training (data/sent/imdb_raw.txt, nunchi/en_casual.py)
  reddit  English Reddit comments       not used in training (data/sent/reddit_raw.txt)

Tokens as the training pools keep them (sentence_data.clean, v3): Hangul, Latin and digit tokens; a line with
Latin that is Korean typed in English mode is dropped, as in training. Lines longer than 20 words are cut
into consecutive pieces of at most 20 (not dropped, which would leave only short sentences). Lines equal to
a training pool line are dropped. Every word is scored. Dev / test halves by hash of the line.
Cleanup (held_out_lines), found by reading the errors of the first build:
  - an apostrophe inside a word is removed, not split on (don't -> dont, as in the Tatoeba pools; split, the
    halves "don t" became words of their own)
  - a line is dropped if it holds a token longer than LONG characters (weather tables and hashes in web text:
    강우량1일8일15일...0mm50mm..., one "word" of hundreds of keys)
  - a line is dropped if it holds lower-case Latin that is no English word (zipf < 1.5) but whose keys compose
    into whole syllables of a Korean token we have seen (gusl = 혀니, wldn = 지우): Korean typed in English
    mode, which the text gives as English, so the label would be wrong. clean() drops only the ones in its
    word list.

    python -u experiments/source_sets.py     -> data/eval/source_sets.jsonl  {"split", "src", "w": [[keys, labels]]}
    load(split) -> {src: [sentence [(keys, labels)]]}
"""
import gzip
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parent  # the repository
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "eval" / "source_sets.jsonl"
WORDS = 40_000  # per source, both halves
CHUNK = 20
LONG = 30
APOS = re.compile(r"(?<=[A-Za-z])['’](?=[A-Za-z])")
SYLLABLES = re.compile(r"^[가-힣]+$")


def _half(line):
    return "dev" if int(hashlib.md5(line.encode()).hexdigest(), 16) % 2 == 0 else "test"


def raw_lines(src):
    from nunchi.sentence_data import RAW, SENT_KCBERT, korean_lines
    from nunchi.web_mixed import RAW as MC4_VALID, SENT_END, SPAM

    if src == "nsmc":
        with (RAW / "nsmc_test.txt").open(encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    yield parts[1]
    elif src == "kcbert":
        import nunchi.sentence_data as sd
        sd.SENT_KCBERT = SENT_KCBERT + 400_000
        yield from itertools.islice(korean_lines("kcbert"), SENT_KCBERT, None)
    elif src == "web":
        with gzip.open(MC4_VALID, "rt", encoding="utf-8") as f:
            for line in f:
                for text in SENT_END.split(json.loads(line)["text"]):
                    if not SPAM.search(text):
                        yield text
    else:
        with (ROOT / "data" / "sent" / f"{src}_raw.txt").open(encoding="utf-8") as f:
            yield from (line.rstrip("\n") for line in f)


_KK, _TRAIN = None, None


def wrong_mode(tok):
    """lower-case Latin that is no English word but composes into a Korean token we have seen"""
    from wordfreq import zipf_frequency
    from nunchi.demo import compose
    from nunchi.sentence_data import ko_zipf
    if not (tok.isalpha() and tok.isascii() and tok.islower() and len(tok) >= 2):
        return False
    ko = compose(tok)
    return bool(SYLLABLES.match(ko)) and ko_zipf(ko) > 0 and zipf_frequency(tok, "en") < 1.5


def held_out_lines(src, stats=None):
    """-> the source's held-out lines as clean token lines (cut to CHUNK words), in raw order. stats: a Counter
    of why lines were dropped."""
    return list(iter_held_out(src, stats))


def iter_held_out(src, stats=None, texts=None):
    """held_out_lines one line at a time; texts: raw text to read instead of raw_lines(src)"""
    global _KK, _TRAIN
    import nunchi.sentence_data as sd
    sd.MAX_W = 10 ** 9  # cut below instead of dropping long lines
    if _KK is None:
        _KK, _TRAIN = sd.ko_keys(), set()
        for name in ("nsmc_v3", "kcbert_v3", "web50_v3"):
            with (sd.OUT / f"{name}.txt").open(encoding="utf-8") as f:
                _TRAIN.update(line.rstrip("\n") for line in f)
    stats = stats if stats is not None else {}
    seen = set()
    for text in (raw_lines(src) if texts is None else texts):
        toks = sd.clean(APOS.sub("", text), _KK, 3)
        if not toks:
            stats["training filter"] = stats.get("training filter", 0) + 1
            continue
        for i in range(0, len(toks), CHUNK):
            piece = toks[i:i + CHUNK]
            line = " ".join(piece)
            if sum(not t.isdigit() for t in piece) < sd.MIN_W or line in _TRAIN or line in seen:
                continue
            why = "long token" if any(len(t) > LONG for t in piece) else \
                "Korean typed in English mode" if any(wrong_mode(t) for t in piece) else None
            if why:
                stats[why] = stats.get(why, 0) + 1
                continue
            seen.add(line)
            yield line


def build():
    from nunchi.sentence_tagger import to_sentence

    rows = []
    for src in ("nsmc", "kcbert", "web", "imdb", "reddit"):
        stats = {}
        lines = held_out_lines(src, stats)
        random.Random(0).shuffle(lines)
        n = 0
        for line in lines:
            if n >= WORDS:
                break
            rows.append({"split": _half(line), "src": src,
                         "w": [[k, "".join(map(str, y))] for k, y in to_sentence(line)]})
            n += len(line.split())
        print(f"  {src}: {n:,} words in {sum(r['src'] == src for r in rows):,} lines (of {len(lines):,} usable); "
              f"dropped: {stats}", flush=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"-> {OUT}", flush=True)


def load(split):
    out = {}
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out.setdefault(r["src"], []).append([(k, [int(c) for c in y]) for k, y in r["w"]])
    return out


if __name__ == "__main__":
    build()
