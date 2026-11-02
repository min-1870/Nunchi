"""Sentence training data for the sentence tagger: real text, each word labelled by its script.

Sources (written once to data/sent/{source}.txt, one sentence per line, tokens space-separated):
  nsmc     : nsmc_train reviews (nsmc_test stays held out: ko / real eval sets)
  kcbert   : KcBERT news comments, SENT_KCBERT lines from the start
  tatoeba  : Tatoeba Korean
  tatoeba_en : Tatoeba English, minus every sentence context.build() uses for valid/test
Tokens kept: Hangul, Latin (case kept), Latin+Hangul (SF는). A sentence is dropped when it holds
anything else that would break a word into pieces we cannot label (digits etc. are just removed, like
the real eval set), or Latin that was most likely Korean typed in the wrong mode (dl = 이), see
wrong_mode(). An earlier rule dropped any Latin whose keys spell a known Korean word; that also threw
out so (내), go (해), work (재가), B급, and every single letter but a few, so the model never saw short
English inside Korean text and turned it into Korean.

    .venv/Scripts/python -m nunchi.sentence_data
"""
import bz2
import math
import random
import re
import subprocess

from wordfreq import zipf_frequency

from nunchi.context import SEED as EVAL_SEED, en_sentences, ko_sentences
from nunchi.demo import compose
from nunchi.train_fasttext import ROOT, load

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sent"
SENT_KCBERT = 2_000_000
MIN_W, MAX_W = 2, 20
SPLIT = re.compile(r"[\s.,!?~…\"'()\[\]/:;*^+=\-]+")
HANGUL = re.compile(r"^[가-힣ㄱ-ㅣ]+$")
LATIN = re.compile(r"^[A-Za-z]+$")
MIXED = re.compile(r"^([A-Za-z]+)([가-힣]+)$")
MASK = re.compile(r"^O+$")
# v2 (TR2 on): any token of Hangul, Latin and digits stays (3개, 2019, gpt4, 채널A, 시즌2): v1 kept only
# Hangul, Latin and Latin+Hangul and silently removed the rest, so the models never saw a digit key, numbers
# vanished from the middle of sentences (별 5 개 -> 별 개), and Hangul+Latin (한미FTA) never appeared
TOKEN_V2 = re.compile(r"^[가-힣ㄱ-ㅣA-Za-z0-9]+$")
LATIN_RUN = re.compile(r"[A-Za-z]+")


def ko_keys():
    return {r["keys"].lower() for s in ("train", "valid", "test") for r in load(s) if r["lab"] == "ko"}


_KO = None  # short Hangul token counts in the Korean corpora, and their total


def ko_zipf(hangul):
    """Zipf frequency (log10 per billion tokens) of a Hangul token in our Korean text (KcBERT + NSMC)."""
    global _KO
    if _KO is None:
        cnt, total = {}, 0
        for src in ("kcbert", "nsmc"):
            with (ROOT / "data" / "counts" / f"{src}_hangul.tsv").open(encoding="utf-8") as f:
                for line in f:
                    w, c = line.rstrip("\n").split("\t")
                    total += int(c)
                    if len(w) <= 6:
                        cnt[w] = cnt.get(w, 0) + int(c)
        _KO = cnt, total
    c = _KO[0].get(hangul, 0)
    return math.log10(c / _KO[1] * 1e9) if c else 0.0


def wrong_mode(latin, kk):
    """Was this Latin most likely Korean typed with the keyboard in English mode?
    Only if its keys spell a Korean word, it has no capital that Shift cannot type in Korean (Shift
    gives ㅃㅉㄸㄲㅆㅒㅖ on QWERTOP only, so B / M / S are deliberate), and the Korean word is more
    frequent in Korean text than the Latin word is in English (dl = 이 yes; so = 내, work = 재가 no)."""
    if latin.lower() not in kk or any(c.isupper() and c not in "QWERTOP" for c in latin):
        return False
    return ko_zipf(compose(latin)) > zipf_frequency(latin.lower(), "en")


def clean(text, kk, v2=False):
    """Korean-text line -> kept tokens, or None if the sentence should be dropped.
    v2: numbers stay (see TOKEN_V2); v2=3 (v3) also leaves pure numbers out of the MIN_W..MAX_W word count,
    since v2 counting them dropped long spec-like web lines and with them 41% of lone 1-letter English."""
    out = []
    for t in filter(None, SPLIT.split(text)):
        if HANGUL.match(t):
            out.append(t)
        elif v2 and TOKEN_V2.match(t):
            for latin in LATIN_RUN.findall(t):
                # wrong-mode Latin: only in tokens without digits (gpt4 is a name, not 헷 typed in English)
                if MASK.match(latin) or (not any(c.isdigit() for c in t) and wrong_mode(latin, kk)):
                    return None
            out.append(t)
        elif LATIN.match(t) or (m := MIXED.match(t)):
            latin = t if LATIN.match(t) else m.group(1)
            if MASK.match(latin) or wrong_mode(latin, kk):
                return None
            out.append(t)
    n = sum(not t.isdigit() for t in out) if v2 == 3 else len(out)
    return out if MIN_W <= n <= MAX_W else None


def korean_lines(source):
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
            if i >= SENT_KCBERT:
                break
            yield raw.decode("utf-8", "replace").rstrip("\n")
        proc.kill()


def held_out_english():
    """The English sentences context.build() draws its valid/test sets from (en, en_ko)."""
    rnd = random.Random(EVAL_SEED)
    ko, en = ko_sentences(), en_sentences()
    rnd.shuffle(ko)
    rnd.shuffle(en)
    return {" ".join(w for _, w in s) for s in en[:8 * 2000]}


def main(v2=False):
    """v2 (2 or 3): writes {source}_v{n}.txt with clean(v2=n) next to the v1 files (which the eval leak
    filters read)."""
    OUT.mkdir(parents=True, exist_ok=True)
    kk = ko_keys()
    suffix = f"_v{int(v2)}" if v2 else ""
    for source in ("nsmc", "tatoeba", "kcbert"):
        n = kept = 0
        with (OUT / f"{source}{suffix}.txt").open("w", encoding="utf-8") as f:
            for text in korean_lines(source):
                n += 1
                if n % 200_000 == 0:
                    print(f"  {source}: {n:,} lines read, {kept:,} kept", flush=True)
                toks = clean(text, kk, v2)
                if toks:
                    kept += 1
                    f.write(" ".join(toks) + "\n")
        print(f"{source}: kept {kept:,} / {n:,}", flush=True)
    if v2:  # tatoeba_en has no Korean to clean: the v1 file serves both
        return
    held = held_out_english()
    kept = 0
    with (OUT / "tatoeba_en.txt").open("w", encoding="utf-8") as f:
        for s in en_sentences():
            line = " ".join(w for _, w in s)
            if line not in held and len(s) >= MIN_W:
                kept += 1
                f.write(line + "\n")
    print(f"tatoeba_en: kept {kept:,} (held out {len(held):,})", flush=True)


if __name__ == "__main__":
    import sys
    main({"v2": 2, "v3": 3}.get(sys.argv[1] if sys.argv[1:] else "", False))
