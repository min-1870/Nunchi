"""Code-switched sentences that keep their meaning: a Korean loanword written in English instead
(미팅 끝나고 -> meeting 끝나고, 캠핑이 -> camping이), the way people actually switch.

Pairs come from Wiktionary's Korean entries borrowed from English (kaikki.org extract of the English
Wiktionary, CC BY-SA 4.0 / GFDL; Ylonen, "Wiktextract", LREC 2022). Kept only when both sides are in
real use: the Hangul word in our Korean text, and the English word written in Latin inside Korean text
(KcBERT / NSMC Latin tables and the mC4 pool); and only for words with no other Wiktionary entry, so
homographs whose Korean reading is also a native or Sino-Korean word (차이 -> chai, 아이 -> i) are out.

Sentences: real Korean sentences (data/sent nsmc / kcbert / tatoeba) holding a pair's Hangul word,
alone or followed by an ending seen after English in real text (미팅에서); every such word is written
in English. At most PER_PAIR sentences per pair, so a few common words cannot fill the set.
Output: data/sent/loan_pairs.tsv (pair, uses) and data/sent/loan.txt (sentences, raw-token format).

    .venv-rocm/Scripts/python -m nunchi.loanwords
"""
import json
import random
import re
import time
from collections import Counter

from nunchi.count_corpus import load as load_counts
from nunchi.sentence_data import OUT as SENT
from nunchi.tagger import mixed_rows
from nunchi.train_fasttext import ROOT

DUMP = ROOT / "data" / "raw" / "wiktionary" / "kaikki-korean.jsonl"
HANGUL = re.compile(r"^[가-힣]+$")
LATIN = re.compile(r"^[A-Za-z]+$")
MIN_KO, MIN_EN = 5, 3  # uses in our Korean text: of the Hangul word, of the English word in Latin
MIN_ENDING = 50  # an ending must follow English at least this often in real text
PER_PAIR = 300


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wiktionary_pairs():
    """(Hangul word, English word) for words whose every Wiktionary entry is a borrowing from English:
    a word that also has a Sino-Korean or native entry is a homograph (차이 差異 / chai, 아이 / i)."""
    pairs, other = set(), set()
    with DUMP.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            w = d["word"]
            if not HANGUL.match(w):
                continue
            en = [t["args"].get("3", "") for t in d.get("etymology_templates") or []
                  if t.get("name") in ("bor", "bor+") and t.get("args", {}).get("2") == "en"]
            if not en:
                other.add(w)
            elif d.get("pos") != "name" and len(w) >= 2:  # names: rarely switched, often homographs
                pairs.update((w, e) for e in en if LATIN.match(e))
    return {(w, e) for w, e in pairs if w not in other}


def real_use_pairs(verbose=False):
    """{Hangul loanword: English word} in real use on both sides (see the module doc)."""
    pairs = wiktionary_pairs()
    ko = load_counts("kcbert", "hangul") + load_counts("nsmc", "hangul")
    en = Counter()
    for src in ("kcbert", "nsmc"):
        en.update({w.lower(): c for w, c in load_counts(src, "latin").items()})
    for line in (SENT / "web.txt").open(encoding="utf-8"):
        en.update(t.lower() for t in line.split() if LATIN.match(t))
    kept = {k: e for k, e in pairs if ko[k] >= MIN_KO and en[e.lower()] >= MIN_EN}
    if verbose:
        dropped = sorted((k, e) for k, e in pairs if (k, e) not in kept.items() and ko[k] >= MIN_KO)
        say(f"Wiktionary: {len(pairs):,} Hangul word <- English word pairs; in real use on both sides: "
            f"{len(kept):,} (dropped e.g. {dropped[:12]})")
    return kept


def main():
    kept = real_use_pairs(verbose=True)

    _, endings = mixed_rows()
    endings = {e for e, c in endings.items() if c >= MIN_ENDING}
    say(f"{len(endings)} endings seen after English at least {MIN_ENDING} times")

    def switch(tok):
        for n in range(len(tok), 1, -1):  # longest loanword first: 데이터베이스 before 데이터
            head, rest = tok[:n], tok[n:]
            if head in kept and (not rest or rest in endings):
                return kept[head] + rest, head
        return None

    rnd = random.Random(0)
    found = []
    for src in ("nsmc", "kcbert", "tatoeba"):
        n = 0
        for line in (SENT / f"{src}.txt").open(encoding="utf-8"):
            toks = line.split()
            sw = [switch(t) if HANGUL.match(t) else None for t in toks]
            if any(sw):
                found.append(([s[0] if s else t for t, s in zip(toks, sw)], {s[1] for s in sw if s}))
                n += 1
        say(f"  {src}: {n:,} sentences hold a loanword")
    rnd.shuffle(found)
    uses, out = Counter(), []
    for toks, heads in found:
        if any(uses[h] >= PER_PAIR for h in heads):
            continue
        uses.update(heads)
        out.append(" ".join(toks))
    (SENT / "loan.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    with (SENT / "loan_pairs.tsv").open("w", encoding="utf-8") as f:
        f.writelines(f"{k}\t{kept[k]}\t{uses[k]}\n" for k in sorted(kept, key=lambda k: -uses[k]))
    say(f"loan.txt: {len(out):,} sentences from {sum(1 for k in kept if uses[k]):,} pairs; "
        f"most used: {[(k, kept[k], uses[k]) for k in sorted(kept, key=lambda k: -uses[k])[:15]]}")


if __name__ == "__main__":
    main()
