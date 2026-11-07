"""the switching eval set: words where the language changes, scored on their own (like PIER in code-switched
speech recognition: error rate at the points of interest only). The main set (source_sets.py) holds few of
them, since typed text is mostly one language.

Points, each with its direction, its place in the sentence and what follows:
  base -> other  a word in the other language than the sentence's base (the majority of its words);
                 place: start / middle / end (end: no next word to revise it with)
  after          the base-language word right after it (the switch back)
  number         a word with digits and letters (3개, 5km, gpt4), base Korean / English

Sources, the same number of sentences from each (results are averaged over sources, not pooled):
  real       ko_casual (NSMC test has only ~700 mixed lines: all of them, the rest from KcBERT), web:
             held-out lines (as source_sets.raw_lines) that hold both languages; nearly all Korean base with
             English words, a few English-base web lines
  synthetic  imdb, reddit: English lines with Korean put in, since no source we have holds English text with
             Korean in it. Half the lines get a word swapped for its translation (translations.dictionary(),
             nouns / adverbs as they are, adjectives / verbs in the form most often ending a sentence in our
             Korean comments: 귀엽다 -> 귀엽네), half get an interjection or discourse word put in (Wiktionary
             intj / adv, or jamo only like ㅋㅋ, seen at a sentence edge in at least 35% of their uses in KcBERT +
             NSMC), drawn by count and placed where they occur (start / middle / end as in that text). Korean
             words in trans_en.txt (training) are not used. The place (start / middle / end) is drawn evenly.

    python -u experiments/switch_sets.py   -> data/eval/switch_sets.jsonl
        {"split", "src", "kind": real|synthetic, "w": [[keys, labels]], "pts": [[idx, category]]}
    load(split) -> [(src, sentence [(keys, labels)], [(idx, category)])]
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parent  # the repository
sys.path.insert(0, str(S))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "eval" / "switch_sets.jsonl"
LINES = 4000  # per source, both halves
EDGE = 0.35
HANGUL = re.compile(r"^[가-힣ㄱ-ㅣ]+$")
JAMO = re.compile(r"^[ㄱ-ㅣ]{2,}$")
PLACES = ("start", "middle", "end")


def lang(y):
    return "KO" if y[0] == 1 else "EN"


def points(sent):
    """-> [(idx, category)] for a labelled sentence, or [] if it holds one language only."""
    words = [(i, k, y) for i, (k, y) in enumerate(sent)]
    letters = [(i, lang(y)) for i, k, y in words if not any(c.isdigit() for c in k) and len(set(y)) == 1]
    cnt = Counter(lg for _, lg in letters)
    if len(cnt) < 2:
        return []
    base = "KO" if cnt["KO"] >= cnt["EN"] else "EN"
    other = "EN" if base == "KO" else "KO"
    out, last = [], len(sent) - 1
    for i, k, y in words:
        if any(c.isdigit() for c in k):
            if any(c.isalpha() for c in k):
                out.append((i, f"number, {base} base"))
        elif len(set(y)) == 1 and lang(y) == other:
            place = "start" if i == 0 else "end" if i == last else "middle"
            out.append((i, f"{base}>{other} {place}"))
            if i < last and len(set(sent[i + 1][1])) == 1 and lang(sent[i + 1][1]) == base \
                    and not any(c.isdigit() for c in sent[i + 1][0]):
                out.append((i + 1, f"{base}>{other} after"))
    return out


def korean_edges():
    """counts in KcBERT + NSMC lines: total, first word, last word."""
    from nunchi.sentence_data import OUT as SENT
    tot, first, last = Counter(), Counter(), Counter()
    for name in ("nsmc_v3", "kcbert_v3"):
        with (SENT / f"{name}.txt").open(encoding="utf-8") as f:
            for line in f:
                t = line.split()
                if len(t) < 3:
                    continue
                tot.update(t)
                first[t[0]] += 1
                last[t[-1]] += 1
    return tot, first, last


def markers(tot, first, last):
    """interjections / discourse words: [(word, count, share at start, share at end)]"""
    from nunchi.translations import DUMP
    pos = defaultdict(set)
    with DUMP.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pos[d["word"]].add(d.get("pos"))
    out = []
    for w, c in tot.items():
        if c < 300 or not HANGUL.match(w):
            continue
        if not (JAMO.match(w) or pos.get(w, set()) & {"intj", "adv"}):
            continue
        fs, ls = first[w] / c, last[w] / c
        if fs + ls >= EDGE:
            out.append((w, c, fs, ls))
    return sorted(out, key=lambda r: -r[1])


def build():
    import source_sets
    import nunchi.sentence_data as sd
    from nunchi.sentence_tagger import to_sentence
    from nunchi.translations import dictionary

    rnd = random.Random(0)
    lines_of = source_sets.held_out_lines

    rows = []
    mixed = {}
    for src in ("nsmc", "kcbert", "web"):
        mixed[src] = [s for s in (to_sentence(line) for line in lines_of(src)) if points(s)]
        rnd.shuffle(mixed[src])
        print(f"  {src}: {len(mixed[src]):,} lines with both languages", flush=True)
    # NSMC test holds too few mixed lines on its own: Korean casual = all of them, the rest from KcBERT
    casual = mixed["nsmc"][:LINES] + mixed["kcbert"][:max(LINES - len(mixed["nsmc"]), 0)]
    rows += [{"src": "ko_casual", "kind": "real", "w": s} for s in casual]
    rows += [{"src": "web", "kind": "real", "w": s} for s in mixed["web"][:LINES]]

    tot, first, last = korean_edges()
    mk = markers(tot, first, last)
    print(f"  markers ({len(mk)}): " + " ".join(f"{w}({c})" for w, c, *_ in mk), flush=True)
    seen_tr = {t for line in (sd.OUT / "trans_en.txt").open(encoding="utf-8") for t in line.split() if HANGUL.match(t)}
    en2ko, _ = dictionary()

    # adjectives / verbs: stem + one of the endings that most often close a sentence across all of them (a
    # word's own most frequent final form can be 상상하지마라: a negation, not the meaning swapped in)
    stems = {w[:-1] for c in en2ko.values() for w, p, _ in c if p in ("adj", "verb") and w.endswith("다")}
    end_cnt = Counter()
    for t, c in last.items():
        for n in (1, 2, 3):
            if t[:n] in stems and 1 <= len(t) - n <= 2:
                end_cnt[t[n:]] += c
    endings = {e for e, _ in end_cnt.most_common(12)}
    print(f"  endings: {[e for e, _ in end_cnt.most_common(12)]}", flush=True)

    def form(ko_w, p):
        if p not in ("adj", "verb"):
            return ko_w
        stem = ko_w[:-1]
        cands = [(last[stem + e], stem + e) for e in endings if last[stem + e]]
        return max(cands)[1] if cands else None

    def put_marker(toks, place):
        weight = {"start": lambda fs, ls: fs, "end": lambda fs, ls: ls, "middle": lambda fs, ls: 1 - fs - ls}[place]
        w = rnd.choices(mk, [c * max(weight(fs, ls), 0) for _, c, fs, ls in mk])[0][0]
        i = 0 if place == "start" else len(toks) if place == "end" else rnd.randrange(1, len(toks))
        return toks[:i] + [w] + toks[i:]

    def swap(toks, place):
        idx = [0] if place == "start" else [len(toks) - 1] if place == "end" else list(range(1, len(toks) - 1))
        cands = []
        for i in idx:
            for ko_w, p, use in en2ko.get(toks[i].lower(), []):
                f = form(ko_w, p)
                if f and f not in seen_tr:
                    cands.append((i, f, use))
        if not cands:
            return None
        i, f, _ = rnd.choices(cands, [u for *_, u in cands])[0]
        return toks[:i] + [f] + toks[i + 1:]

    for src in ("imdb", "reddit"):
        lines = lines_of(src)
        rnd.shuffle(lines)
        made, kinds = [], Counter()
        for line in lines:
            if len(made) == LINES:
                break
            toks = line.split()
            if len(toks) < 3:
                continue
            place = PLACES[len(made) % 3]
            new = swap(toks, place) if len(made) % 2 == 0 else put_marker(toks, place)
            if new is None:
                continue
            s = to_sentence(" ".join(new))
            if points(s):
                made.append(s)
                kinds[place] += 1
        rows += [{"src": src, "kind": "synthetic", "w": s} for s in made]
        print(f"  {src}: {len(made):,} synthetic lines ({dict(kinds)}); e.g. "
              + " | ".join(" ".join(source_sets_text(s)) for s in made[:4]), flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            s = r["w"]
            line = " ".join(k for k, _ in s)
            f.write(json.dumps({"split": source_sets._half(line), "src": r["src"], "kind": r["kind"],
                                "w": [[k, "".join(map(str, y))] for k, y in s], "pts": points(s)},
                               ensure_ascii=False) + "\n")
    print(f"-> {OUT}", flush=True)


def source_sets_text(s):
    from nunchi.demo import compose
    return [compose(k) if y[0] == 1 else k for k, y in s]


def load(split):
    out = []
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out.append((r["src"], [(k, [int(c) for c in y]) for k, y in r["w"]], [tuple(p) for p in r["pts"]]))
    return out


if __name__ == "__main__":
    build()
