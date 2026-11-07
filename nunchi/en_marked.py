"""English sentences with Korean put in the way people who type both put it, for training: no source we have
holds English text with Korean in it (in 4.6M training lines, 46 English-base lines hold ㅋㅋ / ㅠㅠ), and
trans_en (translations.py) only swaps dictionary words, so the model reads ㅋㅋ typed after English as "zz".

Half the lines get an interjection or discourse word (Wiktionary intj / adv, or jamo only like ㅋㅋ, seen at a
sentence edge in at least 35% of their uses in KcBERT + NSMC) put in, drawn by count and placed where it
occurs in that text; half get a word swapped for its translation (translations.dictionary(): nouns / adverbs
as they are, adjectives / verbs as stem + one of the endings that most often close a sentence in our Korean
comments: 웃기다 -> 웃기네). The place (start / middle / end) is drawn evenly.
Base: Tatoeba English, training split (the switching eval set builds its synthetic part the same way from
IMDB / Reddit, which no training data uses).

    python -m nunchi.en_marked      -> data/sent/en_marked.txt (one line per sentence, as the other pools)
    python -m nunchi.en_marked v3 [RATIO] -> data/sent/en_marked_v3.txt (RATIO English lines per marked line for each
                                       colliding key, default 1; another ratio writes en_marked_v3_r<percent>.txt)
    python -m nunchi.en_marked v2   -> data/sent/en_marked_v2.txt: the same plus English lines where a marker's keys
                                       are English (WW2 = ㅉㅉ2, gg = ㅎㅎ), and data/eval/collision_eval.jsonl

Key collisions (v2): ㅉㅉ is typed WW, ㅎㅎ gg, ㅡㅡ mm. After en_marked taught that Korean markers turn up in
English text, the model read WW2 as ㅉㅉ2. English text that uses those keys as English (real lines from Tatoeba,
IMDB and Reddit, 448 in all, minus the lines the eval sets use) is split by hash: half goes into training,
repeated to COLLIDE_SHARE of the file, half becomes a small check set (collision_eval.jsonl).
"""
import json
import random
import re
from collections import Counter, defaultdict

from nunchi.sentence_data import OUT as SENT

N = 120_000
EDGE = 0.35
HANGUL = re.compile(r"^[가-힣ㄱ-ㅣ]+$")
JAMO = re.compile(r"^[ㄱ-ㅣ]{2,}$")
PLACES = ("start", "middle", "end")


def korean_edges():
    """counts in KcBERT + NSMC lines: total, as first word, as last word."""
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


class Inserter:
    """puts Korean into English token lists: marker(toks, place) or swap(toks, place) -> new toks or None"""

    def __init__(self, rnd, exclude=frozenset()):
        from nunchi.translations import dictionary
        self.rnd, self.exclude = rnd, exclude
        tot, first, self.last = korean_edges()
        self.mk = markers(tot, first, self.last)
        self.en2ko, _ = dictionary()
        stems = {w[:-1] for c in self.en2ko.values() for w, p, _ in c if p in ("adj", "verb") and w.endswith("다")}
        end_cnt = Counter()
        for t, c in self.last.items():
            for n in (1, 2, 3):
                if t[:n] in stems and 1 <= len(t) - n <= 2:
                    end_cnt[t[n:]] += c
        self.endings = [e for e, _ in end_cnt.most_common(12)]

    def form(self, ko_w, p):
        if p not in ("adj", "verb"):
            return ko_w
        stem = ko_w[:-1]
        cands = [(self.last[stem + e], stem + e) for e in self.endings if self.last[stem + e]]
        return max(cands)[1] if cands else None

    def marker(self, toks, place):
        weight = {"start": lambda fs, ls: fs, "end": lambda fs, ls: ls, "middle": lambda fs, ls: 1 - fs - ls}[place]
        w = self.rnd.choices(self.mk, [c * max(weight(fs, ls), 0) for _, c, fs, ls in self.mk])[0][0]
        i = 0 if place == "start" else len(toks) if place == "end" else self.rnd.randrange(1, len(toks))
        return toks[:i] + [w] + toks[i:]

    def swap(self, toks, place):
        idx = [0] if place == "start" else [len(toks) - 1] if place == "end" else list(range(1, len(toks) - 1))
        cands = []
        for i in idx:
            for ko_w, p, use in self.en2ko.get(toks[i].lower(), []):
                f = self.form(ko_w, p)
                if f and f not in self.exclude:
                    cands.append((i, f, use))
        if not cands:
            return None
        i, f, _ = self.rnd.choices(cands, [u for *_, u in cands])[0]
        return toks[:i] + [f] + toks[i + 1:]

    def lines(self, lines, n):
        """-> up to n token lists from English lines, places even, marker / swap alternating"""
        out = []
        for line in lines:
            if len(out) == n:
                break
            toks = line.split()
            if len(toks) < 3:
                continue
            place = PLACES[len(out) % 3]
            new = self.swap(toks, place) if len(out) % 2 == 0 else self.marker(toks, place)
            if new:
                out.append(new)
        return out


def collision_lines(mk):
    """English lines (clean tokens) holding a token whose keys are a marker's (exact case; digits may follow:
    WW2), minus the lines the eval sets use -> (train, check)"""
    import hashlib
    from nunchi.dubeolsik import to_keys
    from nunchi.sentence_data import SPLIT, TOKEN_V2
    eval_dir = SENT.parent / "eval"
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl"):
        with (eval_dir / f).open(encoding="utf-8") as fh:
            used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in fh)
    keys = {to_keys(w) for w, *_ in mk}
    hit = lambda t: t in keys or (re.search(r"\d$", t) and re.sub(r"\d+$", "", t) in keys)
    # tokens as sentence_data.clean keeps them, but without its wrong-mode filter: in English text WW is WW, and
    # that filter (made for Korean text) drops exactly these lines as "Korean typed in English mode"
    seen, out = set(), ([], [])
    for src in ("tatoeba_en", "imdb_raw", "reddit_raw"):
        with (SENT / f"{src}.txt").open(encoding="utf-8") as fh:
            for raw in fh:
                text = re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", raw)
                toks = [t for t in filter(None, SPLIT.split(text)) if TOKEN_V2.match(t)]
                if len(toks) < 3 or not any(hit(t) for t in toks):
                    continue
                line = " ".join(toks[:20])
                if line in used or line in seen:
                    continue
                seen.add(line)
                out[int(hashlib.md5(line.encode()).hexdigest(), 16) % 2].append(line)
    return out


COLLIDE_SHARE = 0.03
MAX_REPEAT = 5


def big_english_lines():
    """sentences of the bigger English text (data/raw): WikiText-103 and four Reddit subreddit dumps"""
    import pyarrow.parquet as pq  # after torch: pyarrow first breaks the ROCm DLL load
    raw = SENT.parent / "raw"
    for f, col in [(p, "text") for p in sorted((raw / "wikitext103").glob("*.parquet"))] + \
                  [(p, "body") for p in sorted((raw / "reddit_big").glob("*.parquet"))]:
        for text in pq.read_table(f, columns=[col]).column(col).to_pylist():
            if text:
                yield from re.split(r"(?<=[.!?])\s+|\n", text)


def collision_pool(mk, check):
    """key -> English lines (20 words around the hit, tokens as clean keeps them, no wrong-mode filter) from the
    big English text, minus the check set's lines"""
    from nunchi.dubeolsik import to_keys
    from nunchi.sentence_data import SPLIT, TOKEN_V2
    keys = {to_keys(w) for w, *_ in mk}
    base = lambda t: t if t in keys else re.sub(r"\d+$", "", t) if re.search(r"\d$", t) else None
    pool, seen = defaultdict(list), set(check)
    for sent in big_english_lines():
        toks = [t for t in filter(None, SPLIT.split(re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", sent)))
                if TOKEN_V2.match(t)]
        if len(toks) < 3:
            continue
        for i, t in enumerate(toks):
            k = base(t)
            if k in keys:
                lo = max(0, min(i - 10, len(toks) - 20))
                line = " ".join(toks[lo:lo + 20])
                if line not in seen:
                    seen.add(line)
                    pool[k].append(line)
                break
    return pool


def main(v2=False, v3=False, ratio=1.0):
    from nunchi.sentence_tagger import load_sentences, to_sentence
    rnd = random.Random(0)
    ins = Inserter(rnd)
    lines = list(load_sentences()["tatoeba_en"][0])
    rnd.shuffle(lines)
    out = [" ".join(t) for t in ins.lines(lines, N)]
    name = "en_marked.txt"
    if v3:
        # per colliding key, as many English lines using it as marked lines holding the marker (1:1), real lines
        # from the training half of the v2 lines and the big English text, repeated at most MAX_REPEAT times
        from nunchi.dubeolsik import to_keys
        train, check = collision_lines(ins.mk)
        pool = collision_pool(ins.mk, check)
        for line in train:
            for t in line.split():
                k = t if t in pool else re.sub(r"\d+$", "", t)
                if k in pool or k in {to_keys(w) for w, *_ in ins.mk}:
                    pool[k].append(line)
                    break
        key_of = {w: to_keys(w) for w, *_ in ins.mk}
        marked = Counter(key_of[t] for line in out for t in line.split() if t in key_of)
        added = []
        for k, need in sorted(marked.items(), key=lambda kv: -kv[1]):
            have = pool.get(k, [])
            if not have:
                continue
            take = min(round(need * ratio), len(have) * MAX_REPEAT)
            rnd.shuffle(have)
            added += [have[i % len(have)] for i in range(take)]
            print(f"  {k}: marked {need}, English lines {len(have)}, added {take}", flush=True)
        out += added
        rnd.shuffle(out)
        name = "en_marked_v3.txt" if ratio == 1.0 else f"en_marked_v3_r{int(ratio * 100)}.txt"
        print(f"collisions v3: {len(added)} English lines added", flush=True)
    elif v2:
        train, check = collision_lines(ins.mk)
        reps = max(1, round(COLLIDE_SHARE * N / max(len(train), 1)))
        out += train * reps
        rnd.shuffle(out)
        name = "en_marked_v2.txt"
        with (SENT.parent / "eval" / "collision_eval.jsonl").open("w", encoding="utf-8") as f:
            for line in check:
                f.write(json.dumps({"w": [[k, "".join(map(str, y))] for k, y in to_sentence(line)]},
                                   ensure_ascii=False) + "\n")
        print(f"collisions: {len(train)} train lines x{reps}, {len(check)} check lines", flush=True)
    (SENT / name).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{name}: {len(out):,} lines; markers {len(ins.mk)}, endings {ins.endings}; e.g. "
          + " | ".join(out[:8]), flush=True)


if __name__ == "__main__":
    import sys
    main(v2=sys.argv[1:] == ["v2"], v3=sys.argv[1:2] == ["v3"],
         ratio=float(sys.argv[2]) if sys.argv[1:2] == ["v3"] and len(sys.argv) > 2 else 1.0)
