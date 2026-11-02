"""Mixed Korean/English eval set from the web (mC4 Korean, validation split only: training may later use
the train split, so the two never overlap). The NSMC set is movie reviews; this one covers whatever the
web covers (shopping, tech, news, hobbies), so a model that only learned review vocabulary shows it here.

Kept: sentences of 3-20 words, Hangul words mostly, at least one Latin word. Dropped:
  - Korean typed in the wrong mode (same rule as sentence_data.clean)
  - Latin that is not an English word or an acronym: mC4 Korean has machine-translated spam whose
    English is cut into pieces ("Batte Ries"), and those pieces are not something anyone types
  - Latin + Hangul tokens (SF는): the word-level eval has no label for them, as in real_mixed()
  - spam (escort ads, gambling) and keyword lists: no particle or ending on any Korean word, or the
    same word over and over, is a tag list or a product title, not something typed as a sentence
Diversity: a Latin word appears in at most PER_WORD sentences and a site gives at most PER_SITE, so a
few brand names or one shop cannot fill the set.

`train` builds the training pool (data/sent/web.txt) the same way from the first TRAIN_SHARDS shards of
the train split, with looser rules: Latin + Hangul tokens stay (SF는 is exactly what the tagger must
learn), one Latin word per sentence may be outside the English word list (brand names, romanised
Korean), and the caps are per-word TRAIN_PER_WORD / per-site TRAIN_PER_SITE.

    .venv-rocm/Scripts/python -m nunchi.web_mixed          # eval set
    .venv-rocm/Scripts/python -m nunchi.web_mixed train    # training pool
"""
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from urllib.parse import urlparse

from wordfreq import zipf_frequency

from nunchi.sentence_data import HANGUL, LATIN, LATIN_RUN, MASK, MIXED, SPLIT, clean, ko_keys, wrong_mode
from nunchi.train_fasttext import ROOT

RAW = ROOT / "data" / "raw" / "mc4" / "c4-ko-validation.json.gz"
# v1 dropped every sentence with Latin whose keys spell a Korean word (so, go, B, T ...), the same
# over-wide filter as the training data had; v2 uses sentence_data.wrong_mode(). v1 kept for comparison.
OUT_V1 = ROOT / "data" / "sent" / "web_mixed.txt"
OUT = ROOT / "data" / "sent" / "web_mixed_v2.txt"
N = 2000
PER_WORD, PER_SITE = 3, 5
SHARD_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ko.tfrecord-{:05d}-of-01024.json.gz"
TRAIN_SHARDS = 10
TRAIN_OUT = ROOT / "data" / "sent" / "web.txt"
TRAIN_PER_WORD, TRAIN_PER_SITE = 200, 30
SENT_END = re.compile(r"(?<=[.!?])\s+|\n")
SPAM = re.compile(r"출장|콜걸|카지노|먹튀|토토|바카라|슬롯|조건만남|오피|안마|카톡")
# last syllable of a Korean word that carries a particle or an ending (책을, 좋다, 해요, 보고 ...)
ENDING = set("은는이가을를에의로도와과서다요고며면해지만게며데까니라")


def english_like(t):
    return zipf_frequency(t.lower(), "en") >= 2.5 or (t.isupper() and 2 <= len(t) <= 6)


def sentence(text, kk):
    if SPAM.search(text):
        return None
    words = list(filter(None, SPLIT.split(text)))
    out = []
    for t in words:
        if HANGUL.match(t):
            out.append(("ko", t))
        elif MIXED.match(t):
            return None
        elif LATIN.match(t):
            if MASK.match(t) or wrong_mode(t, kk) or not english_like(t):
                return None
            out.append(("en", t))
        # digits and symbols are skipped, as in real_mixed()
    n_en = sum(lab == "en" for lab, _ in out)
    if not 3 <= len(out) <= 20 or n_en == 0 or n_en * 2 > len(out):
        return None
    if len({w for _, w in out}) < 0.8 * len(out):
        return None
    if sum(lab == "ko" and w[-1] in ENDING for lab, w in out) < 2:
        return None
    return out


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_sentence(text, kk, v2=False):
    """-> raw tokens for the training pool (same format as data/sent/*.txt), or None.
    v2: numbers and Hangul+Latin tokens stay (sentence_data.clean); Latin next to digits (gpt4, 5km) counts
    as English-like, a name or a unit."""
    if SPAM.search(text):
        return None
    toks = clean(text, kk, v2)  # Hangul / Latin / Latin+Hangul (v2: + digits), wrong-mode dropped, 2-20 words
    if not toks or len(toks) < 3:
        return None
    if v2:
        lat_toks = [t for t in toks if LATIN_RUN.search(t)]
        latin = [r for t in lat_toks if not any(c.isdigit() for c in t) for r in LATIN_RUN.findall(t)]
        if not lat_toks or len(lat_toks) * 2 > len(toks) or sum(not english_like(t) for t in latin) > 1:
            return None
    else:
        latin = [m.group(1) if (m := MIXED.match(t)) else t for t in toks if not HANGUL.match(t)]
        if not latin or len(latin) * 2 > len(toks) or sum(not english_like(t) for t in latin) > 1:
            return None
    if len(set(toks)) < 0.8 * len(toks):
        return None
    if sum(HANGUL.match(t) is not None and t[-1] in ENDING or MIXED.match(t) is not None or
           (v2 and LATIN_RUN.search(t) is not None and HANGUL.search(t) is not None) for t in toks) < 2:
        return None
    return toks


def _scan_shard(args):
    """one train shard -> [(site, tokens)] (a worker process: shards are scanned in parallel)."""
    i, v2 = args
    kk = ko_keys()
    out = []
    with gzip.open(RAW.parent / f"c4-ko-train-{i:05d}.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            site = urlparse(doc["url"]).netloc
            for text in SENT_END.split(doc["text"]):
                s = train_sentence(text, kk, v2)
                if s:
                    out.append((site, s))
    return out


def build_train(shards=TRAIN_SHARDS, out_path=TRAIN_OUT, v2=False):
    """shards: the first this many train shards; the per-word / per-site caps grow with them, so a bigger
    pool keeps the same spread (`train 50 web50.txt` builds a five times bigger one next to web.txt).
    v2: train_sentence(v2=True) (`train 50 web50_v2.txt v2`)."""
    from multiprocessing import Pool

    cap_word, cap_site = TRAIN_PER_WORD * shards // TRAIN_SHARDS, TRAIN_PER_SITE * shards // TRAIN_SHARDS
    raw = RAW.parent
    for i in range(shards):
        path = raw / f"c4-ko-train-{i:05d}.json.gz"
        if path.exists():
            continue
        say(f"downloading shard {i + 1}/{shards}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SHARD_URL.format(i), tmp)
        tmp.rename(path)
    cands = []
    t0 = time.time()
    with Pool(8) as pool:
        for n, part in enumerate(pool.imap(_scan_shard, [(i, v2) for i in range(shards)]), 1):
            cands += part
            el = time.time() - t0
            say(f"  scanned shard {n}/{shards} ({n / shards:.0%})  {len(cands):,} sentences so far  "
                f"ETA {el / n * (shards - n):.0f} s")
    rnd = random.Random(0)
    rnd.shuffle(cands)
    seen, per_word, per_site, out = set(), Counter(), Counter(), []
    for site, s in cands:
        key = " ".join(s)
        # caps count English words, not letters inside numbers (the k of 4k, the s of s10 used up the
        # cap of k and s and pushed out sentences where s / a / k stand alone: v2 had 41% fewer of those)
        latin = {r.lower() for t in s if not any(c.isdigit() for c in t) for r in LATIN_RUN.findall(t)}
        # v3: a line whose only Latin sits in codes (5km, 3D) holds no English word; v2 kept them all (12% of
        # the pool), which thinned the English the web share is there for. Keep 3 in 10 of them.
        if v2 == 3 and not latin and int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 >= 3:
            continue
        if key in seen or per_site[site] >= cap_site or any(per_word[w] >= cap_word for w in latin):
            continue
        seen.add(key)
        per_site[site] += 1
        per_word.update(latin)
        out.append(key)
    tmp = out_path.with_suffix(".part")  # a reader waiting for the file never sees half of it
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    say(f"training pool: {len(out):,} sentences (of {len(cands):,}), {len(per_site):,} sites, "
        f"{len(per_word):,} distinct English words -> {out_path}")


def build():
    say("loading the Korean word list (wrong-mode filter)")
    kk = ko_keys()
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    say(f"{total:,} web documents to scan")
    cands = []
    t0 = time.time()
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i % 1000 == 0 or i == total:
                rate = i / (time.time() - t0)
                say(f"  {i:,}/{total:,} docs ({i / total:.0%})  {rate:,.0f} docs/s  "
                    f"{len(cands):,} sentences so far  ETA {(total - i) / rate:.0f} s")
            doc = json.loads(line)
            site = urlparse(doc["url"]).netloc
            for text in SENT_END.split(doc["text"]):
                s = sentence(text, kk)
                if s:
                    cands.append((site, s))
    rnd = random.Random(0)
    rnd.shuffle(cands)
    seen, per_word, per_site, out = set(), Counter(), Counter(), []
    for site, s in cands:
        key = " ".join(w for _, w in s)
        latin = {w.lower() for lab, w in s if lab == "en"}
        if key in seen or per_site[site] >= PER_SITE or any(per_word[w] >= PER_WORD for w in latin):
            continue
        seen.add(key)
        per_site[site] += 1
        per_word.update(latin)
        out.append(s)
        if len(out) == N:
            break
    say(f"candidates {len(cands):,}, kept {len(out):,}, sites {len(per_site):,}, English words {len(per_word):,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in out:
            f.write(" ".join(f"{lab}:{w}" for lab, w in s) + "\n")


PARTICLE_OUT = ROOT / "data" / "sent" / "web_particle_eval.txt"


def build_particle_eval(n=3000):
    """English + Korean particle inside real sentences (SF는, 미팅을 -> meeting을 is not here: real only),
    mC4 validation with the training filters, as keys:labels per word (the word sets above drop these)."""
    from nunchi.sentence_tagger import word

    kk = ko_keys()
    sents = []
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in f:
            for text in SENT_END.split(json.loads(line)["text"]):
                toks = train_sentence(text, kk)
                if toks and any(MIXED.match(t) for t in toks):
                    sents.append(" ".join(toks))
    sents = list(dict.fromkeys(sents))
    random.Random(0).shuffle(sents)
    rows = ["\t".join(f"{k}:{''.join(map(str, y))}" for k, y in map(word, s.split())) for s in sents[:n]]
    PARTICLE_OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    say(f"English+particle eval: {len(rows):,} sentences (of {len(sents):,}) -> {PARTICLE_OUT}")


CODE_OUT = ROOT / "data" / "sent" / "web_code_eval.txt"


def build_code_eval(n=3000):
    """v2 eval: real sentences (mC4 validation, v2 training filters) holding a token with digits next to
    letters (gpt4, 5km, 시즌2, 3개) or Hangul+Latin (채널A), as keys:labels per word like the particle set.
    Labels as sentence_tagger.to_sentence gives them (digits follow their letters, else the word before)."""
    from nunchi.sentence_tagger import to_sentence

    kk = ko_keys()
    sents = []
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in f:
            for text in SENT_END.split(json.loads(line)["text"]):
                toks = train_sentence(text, kk, v2=True)
                if toks and any(re.search(r"[0-9]", t) and re.search(r"[A-Za-z가-힣]", t) or
                                re.match(r"^[가-힣]+[A-Za-z]", t) for t in toks):
                    sents.append(" ".join(toks))
    sents = list(dict.fromkeys(sents))
    random.Random(0).shuffle(sents)
    rows = ["\t".join(f"{k}:{''.join(map(str, y))}" for k, y in to_sentence(s)) for s in sents[:n]]
    CODE_OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    say(f"digits / Hangul+Latin eval: {len(rows):,} sentences (of {len(sents):,}) -> {CODE_OUT}")


def web_mixed(path=OUT):
    """Same shape as context.real_mixed()[0]: [[(lab, word)]]."""
    with path.open(encoding="utf-8") as f:
        return [[tuple(t.split(":", 1)) for t in line.split()] for line in f]


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:  # train [shards] [output file name in data/sent] [v2]
        a = sys.argv[2:]
        build_train(int(a[0]) if a else TRAIN_SHARDS, TRAIN_OUT.parent / a[1] if len(a) > 1 else TRAIN_OUT,
                    {"v2": 2, "v3": 3}.get(a[2] if len(a) > 2 else "", False))
    else:
        {"particle": build_particle_eval, "code": build_code_eval}.get(sys.argv[1] if sys.argv[1:] else "", build)()
