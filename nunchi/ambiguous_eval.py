"""Eval set for revising a word once more words are typed: real sentences, never trained on, holding a word
whose keys are common in both languages (so = 내, go = 해, did = 양, b = ㅠ). Both truths are kept:
  en targets : the word was written in Latin (채널 go 해서, so what) -> a revision should make it English
  ko targets : the word was written in Hangul (해 보니까, 내 생각엔) -> a revision must leave it Korean
Ambiguous keys: English zipf >= 3 (wordfreq) and the Hangul they type has zipf >= 3 in our Korean text, within
2 of each other; from Latin in Korean text and from common English words.
Sources (held out from training): NSMC test reviews, mC4 Korean validation, the 1% valid split of KcBERT,
NSMC train, Tatoeba ko / en; sentences that also appear in training text are dropped, and so is Latin that
was most likely Korean typed in the wrong mode (sentence_data.clean).
Output: data/sent/ambiguous_eval.jsonl, one sentence per line: {"src", "w": [[keys, labels]], "t": [target
word indices]}; at most PER_KEY sentences per (keys, truth).

    .venv-rocm/Scripts/python -m nunchi.ambiguous_eval
    .venv-rocm/Scripts/python -m nunchi.ambiguous_eval v2   # bigger

v2 adds mC4 train shards V2_SHARDS (training uses shards 0-49 at most, web50.txt) and allows PER_KEY_V2
sentences per (keys, truth), so a few words' difference between two models is not lost in noise.
"""
import gzip
import json
import random
import sys
import time
import urllib.request
from collections import Counter

from wordfreq import top_n_list, zipf_frequency

from nunchi.count_corpus import load as load_counts
from nunchi.demo import compose
from nunchi.dubeolsik import to_keys
from nunchi.sentence_data import HANGUL, LATIN, OUT as SENT, RAW as RAW_DIR, clean, ko_keys, ko_zipf
from nunchi.sentence_tagger import load_sentences, to_sentence, word
from nunchi.web_mixed import RAW as MC4_VALID, SENT_END, SHARD_URL

OUT = SENT / "ambiguous_eval.jsonl"
PER_KEY = 40
OUT_V2 = SENT / "ambiguous_eval_v2.jsonl"
# v3: the same sources and selection as v2, with numbers left in the sentences (sentence_data.clean v3) as
# TR2 on train with them: the v1/v2 sets removed every token holding a digit, gluing its neighbours together
OUT_V3 = SENT / "ambiguous_eval_v3.jsonl"
PER_KEY_V2 = 150
V2_SHARDS = list(range(50, 60))


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ambiguous_keys():
    lat = load_counts("kcbert", "latin") + load_counts("nsmc", "latin")
    words = [w.lower() for w, _ in lat.most_common(4000)] + top_n_list("en", 5000)
    out = set()
    for k in words:
        if k.isascii() and k.isalpha() and len(k) <= 4 and k not in out:
            en, ko = zipf_frequency(k, "en"), ko_zipf(compose(k))
            if en >= 3 and ko >= 3 and abs(en - ko) <= 2:
                out.add(k)
    return out


def sources(shards=()):
    with (RAW_DIR / "nsmc_test.txt").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                yield "nsmc_test", parts[1]
    with gzip.open(MC4_VALID, "rt", encoding="utf-8") as f:
        for line in f:
            for text in SENT_END.split(json.loads(line)["text"]):
                yield "mc4_valid", text
    for n, i in enumerate(shards, 1):
        path = MC4_VALID.parent / f"c4-ko-train-{i:05d}.json.gz"
        if not path.exists():
            say(f"downloading mC4 train shard {i} ({n}/{len(shards)})")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(SHARD_URL.format(i), tmp)
            tmp.rename(path)
        say(f"scanning mC4 train shard {i} ({n}/{len(shards)})")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                for text in SENT_END.split(json.loads(line)["text"]):
                    yield "mc4_train", text
    data = load_sentences()
    for src in ("nsmc", "kcbert", "tatoeba", "tatoeba_en"):
        for line in data[src][1]:
            yield f"{src}_valid", line


def main(out=OUT, per_key=PER_KEY, shards=(), version=False):
    amb = ambiguous_keys()
    say(f"{len(amb)} ambiguous keys, e.g. {sorted(amb)[:40]}")
    kk = ko_keys()
    train = set()
    for name in ("nsmc", "kcbert", "web", "web50", "tatoeba", "tatoeba_en", "nsmc_v3", "kcbert_v3", "web50_v3", "tatoeba_v3"):
        if not (SENT / f"{name}.txt").exists():
            continue
        with (SENT / f"{name}.txt").open(encoding="utf-8") as f:
            train.update(line.rstrip("\n") for line in f)
    rows, per, seen = [], Counter(), set()
    n_src = Counter()
    for n, (src, text) in enumerate(sources(shards), 1):
        if n % 500_000 == 0:
            say(f"  {n:,} texts read, {len(rows):,} sentences kept")
        toks = clean(text, kk, version)
        if not toks or len(toks) < 3:
            continue
        line = " ".join(toks)
        if line in train or line in seen:
            continue
        targets = []
        for i, t in enumerate(toks):
            if LATIN.match(t) and t.lower() in amb:
                key = (t.lower(), "en")
            elif HANGUL.match(t) and to_keys(t) in amb:
                key = (to_keys(t), "ko")
            else:
                continue
            if per[key] < per_key:
                targets.append((i, key))
        if not targets:
            continue
        seen.add(line)
        for _, key in targets:
            per[key] += 1
        n_src[src] += 1
        rows.append({"src": src, "w": [[k, "".join(map(str, y))] for k, y in to_sentence(" ".join(toks))],
                     "t": [i for i, _ in targets]})
    random.Random(0).shuffle(rows)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    truth = Counter(k[1] for k in per.elements())
    say(f"{len(rows):,} sentences, targets: {truth['en']:,} English, {truth['ko']:,} Korean; by source {dict(n_src)}")
    say(f"most used keys: {per.most_common(20)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["v3"]:
        main(OUT_V3, PER_KEY_V2, V2_SHARDS, 3)
    else:
        main(OUT_V2, PER_KEY_V2, V2_SHARDS) if sys.argv[1:2] == ["v2"] else main()
