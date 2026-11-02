"""Code-switched sentences by translation: a word swapped for its translation, so the switch sits where
the meaning fits (the weather -> the 날씨, 솔직히 별로 -> honestly 별로), not at a random spot.

  en base : Tatoeba English with 1-2 content words in Korean (nouns mostly; verbs and adjectives in
            their dictionary form, I 먹다 dinner: awkward, but the meaning still lines up with the rest)
  ko base : real Korean sentences with one noun (with its ending: 날씨가 -> weather가) or adverb in English;
            Korean verbs are conjugated in text (먹었는데), so they cannot be matched to a dictionary form

Pairs: the first sense of each Korean Wiktionary entry (kaikki.org extract, CC BY-SA 4.0 / GFDL), one-word
English glosses only ("to eat, drink" -> eat), plus loanwords with the English word they come from
(미팅 -> meeting, loanwords.real_use_pairs()): a loanword is one more translation, not a separate case.
English side: common words (wordfreq). Korean side: words in our Korean text. An English word with
several Korean ones (meeting -> 회의, 미팅; eat -> 먹다, 드시다) takes one in proportion to its use in our
text (verbs counted by their stem).

Output (data/sent): trans_en.txt, trans_ko.txt for training; trans_en_eval.txt, trans_ko_eval.txt from
sentences held out of training (the sentence tagger's valid split), as "keys<TAB>labels" per word.

    .venv-rocm/Scripts/python -m nunchi.translations
"""
import json
import random
import re
import time
from collections import Counter, defaultdict

from wordfreq import zipf_frequency

from nunchi.count_corpus import load as load_counts
from nunchi.sentence_data import OUT as SENT
from nunchi.sentence_tagger import load_sentences, word
from nunchi.tagger import mixed_rows
from nunchi.train_fasttext import ROOT

DUMP = ROOT / "data" / "raw" / "wiktionary" / "kaikki-korean.jsonl"
HANGUL = re.compile(r"^[가-힣]+$")
LATIN = re.compile(r"^[A-Za-z]+$")
POS = {"noun": 0.6, "adv": 0.15, "adj": 0.15, "verb": 0.1}  # en base: which word gets swapped
STOP = set("the a an and or but if of to in on at by for with from as is are was were be been being am do "
           "does did have has had i you he she it we they me him her us them my your his its our their this "
           "that these those there here not no yes so very can could will would shall should may might must "
           "what who whom which when where why how all any some".split())
MIN_EN_ZIPF, MIN_KO = 3.0, 50
PER_WORD = 400
N_EN, N_KO, N_EVAL = 120_000, 100_000, 2000  # ko base now also holds what loan.txt had (48k)
MIN_ENDING = 50


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def one_word_glosses(gloss):
    """'to eat, drink, have (food)' -> ['eat', 'drink', 'have']; 'a meeting' -> ['meeting']"""
    gloss = re.sub(r"\([^)]*\)", "", gloss)
    out = []
    for part in re.split(r"[;,]", gloss):
        p = re.sub(r"^\s*(to be|to|a|an|the)\s+", "", part.strip().lower()).strip()
        if LATIN.match(p) and p not in STOP and zipf_frequency(p, "en") >= MIN_EN_ZIPF:
            out.append(p)
    return out


def dictionary():
    """-> en2ko {english: [(korean, pos, use)]}, ko2en {korean: (english, pos)} for nouns / adverbs."""
    from nunchi.loanwords import real_use_pairs

    first = {}  # (korean, pos) -> glosses of its first sense, from its first entry
    borrowed = set()
    with DUMP.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            w, pos = d["word"], d.get("pos")
            if not HANGUL.match(w) or pos not in POS:
                continue
            if any(t.get("name") in ("bor", "bor+") and t.get("args", {}).get("2") == "en"
                   for t in d.get("etymology_templates") or []):
                borrowed.add(w)  # its gloss is its own source word: the loanword pair below covers it
                continue
            senses = d.get("senses") or []
            if (w, pos) not in first and senses and senses[0].get("glosses"):
                first[w, pos] = one_word_glosses(senses[0]["glosses"][0])
    for w, e in real_use_pairs().items():
        # same English-side rules as the glosses: Wiktionary has 마이웨이 <- my, which would turn "my" into 마이웨이
        if e.lower() not in STOP and len(e) >= 3 and zipf_frequency(e.lower(), "en") >= MIN_EN_ZIPF:
            first[w, "noun"] = [e]
    ko = load_counts("kcbert", "hangul") + load_counts("nsmc", "hangul")
    stems = {w[:-1] for (w, pos) in first if pos in ("verb", "adj") and w.endswith("다")}
    stem_count = Counter()
    for tok, c in ko.items():  # a verb's use: tokens that start with its stem (먹 -> 먹었는데)
        for n in (1, 2, 3):
            if tok[:n] in stems:
                stem_count[tok[:n]] += c

    def use(w, pos):
        return stem_count[w[:-1]] if pos in ("verb", "adj") else ko[w]

    en2ko, ko2en = defaultdict(list), {}
    for (w, pos), glosses in first.items():
        if not glosses or use(w, pos) < MIN_KO:
            continue
        for g in glosses:
            en2ko[g.lower()].append((w, pos, use(w, pos)))
        if pos in ("noun", "adv"):
            ko2en[w] = (glosses[0], pos)
    return dict(en2ko), ko2en


def labelled(tokens):
    return "\t".join(f"{k}:{''.join(map(str, y))}" for k, y in map(word, tokens))


def main():
    en2ko, ko2en = dictionary()
    loans = set(__import__("nunchi.loanwords", fromlist=["x"]).real_use_pairs())
    say(f"dictionary: {len(en2ko):,} English words -> Korean ({sum(map(len, en2ko.values())):,} pairs, "
        f"{sum(k in loans for c in en2ko.values() for k, _, _ in c):,} of them loanwords), "
        f"{len(ko2en):,} Korean noun/adverb -> English")
    for e in ("meeting", "design", "weather", "eat", "camera", "game", "later"):
        say(f"  {e}: {[(k, u) for k, _, u in sorted(en2ko.get(e, []), key=lambda x: -x[2])][:5]}")
    rnd = random.Random(0)
    data = load_sentences()

    # ---- English base
    def en_base(lines, n, cap):
        uses, out = Counter(), []
        lines = list(lines)
        rnd.shuffle(lines)
        for line in lines:
            toks = line.split()
            # each candidate word draws its Korean in proportion to use (meeting -> 회의 or 미팅)
            cands = []
            for i, t in enumerate(toks):
                if t.lower() in en2ko:
                    ko_w, pos, _ = rnd.choices(en2ko[t.lower()], [u for *_, u in en2ko[t.lower()]])[0]
                    if uses[ko_w] < cap:
                        cands.append((i, ko_w, pos))
            if not cands:
                continue
            k = min(len(cands), rnd.choice([1, 1, 1, 2]))
            picked = {}
            while len(picked) < k:
                i, ko_w, pos = rnd.choices(cands, [POS[p] for *_, p in cands])[0]
                picked[i] = ko_w
            for i, ko_w in picked.items():
                uses[ko_w] += 1
                toks[i] = ko_w
            out.append(toks)
            if len(out) == n:
                break
        return out

    tr = en_base(data["tatoeba_en"][0], N_EN, PER_WORD)
    ev = en_base(data["tatoeba_en"][1], N_EVAL, 10**9)
    (SENT / "trans_en.txt").write_text("\n".join(" ".join(s) for s in tr) + "\n", encoding="utf-8")
    (SENT / "trans_en_eval.txt").write_text("\n".join(labelled(s) for s in ev) + "\n", encoding="utf-8")
    say(f"en base: {len(tr):,} train, {len(ev):,} eval; e.g. {[' '.join(s) for s in tr[:6]]}")

    # ---- Korean base
    _, endings = mixed_rows()
    endings = {e for e, c in endings.items() if c >= MIN_ENDING}

    def swap(tok):
        # two syllables at least: one-syllable heads match inside names and conjugations (안철수, 막고)
        for n in range(len(tok), 1, -1):
            head, rest = tok[:n], tok[n:]
            if head in ko2en and (not rest or (ko2en[head][1] == "noun" and rest in endings)):
                return ko2en[head][0] + rest, head
        return None

    def ko_base(lines, n, cap):
        uses, out = Counter(), []
        lines = list(lines)
        rnd.shuffle(lines)
        for line in lines:
            toks = line.split()
            cands = [(i, s) for i, t in enumerate(toks) if HANGUL.match(t) and (s := swap(t)) and uses[s[1]] < cap]
            if not cands:
                continue
            i, (new, head) = rnd.choice(cands)
            uses[head] += 1
            toks[i] = new
            out.append(toks)
            if len(out) == n:
                break
        return out

    ko_train = [l for s in ("nsmc", "kcbert", "tatoeba") for l in data[s][0]]
    ko_valid = [l for s in ("nsmc", "kcbert", "tatoeba") for l in data[s][1]]
    tr = ko_base(ko_train, N_KO, PER_WORD)
    ev = ko_base(ko_valid, N_EVAL, 10**9)
    (SENT / "trans_ko.txt").write_text("\n".join(" ".join(s) for s in tr) + "\n", encoding="utf-8")
    (SENT / "trans_ko_eval.txt").write_text("\n".join(labelled(s) for s in ev) + "\n", encoding="utf-8")
    say(f"ko base: {len(tr):,} train, {len(ev):,} eval; e.g. {[' '.join(s) for s in tr[:6]]}")


if __name__ == "__main__":
    main()
