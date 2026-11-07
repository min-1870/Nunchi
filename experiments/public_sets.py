"""the public benchmark set: the cells of the internal eval sets (docs/EVALUATION.md), built by the same rules from
text that may be published and that no internal set or training pool holds. The internal test halves stay private
(they keep judging models); this set is what a public page reports and what other input methods can be run on.

  src         text, licence                                             cells (internal source it stands in for)
  nsmc        NSMC test reviews, CC0                                    main
  kcbert      KcBERT news comments, CC BY-SA 4.0, dump lines KC_FROM+   main, switching (ko_casual), KO>EN end,
              (internal sets read lines up to 12M, training the 2M)     amb_en, jamo1, amb_ko, acro, fused, ko1, mix1
  se_post     Stack Exchange questions / answers, CC BY-SA, 2024-03     main (imdb), switching synthetic (imdb)
              dump (archive.org item "stackexchange"), SITES
  se_comment  comments on the same sites                                main (reddit), switching synthetic (reddit)
  se          posts and comments                                        number words (reddit), key collisions (reddit)
  wikitext    WikiText-103, CC BY-SA                                    number words
Left out: mC4 web text (the dump grants no rights in the text), Reddit and IMDB (no licence), one-word English
(en1: Stack Exchange comments are 15+ characters, so there are no one-word ones).

Sizes are the internal test halves' (one half: the public set has no dev part). Lines with Korean put in (switching
synthetic) say so in "from" ("changed"), as CC BY-SA asks. Stack Exchange rows carry what its attribution rule
needs: site, question id, author name and user id (data/raw/stackexchange/license.txt).

    python -u experiments/public_sets.py ko     -> data/public/parts/ko.jsonl
    python -u experiments/public_sets.py en     -> data/public/parts/en.jsonl
    python -u experiments/public_sets.py join   -> data/public/nunchi_bench.jsonl (+ typo cells, ids) and manifest.json
row: {"id", "set", "src", "key" (collide), "w": [[keys, labels]], "pts": [idx] | [[idx, category]] (switch, end_extra),
      "text": the text as meant, "from": {...}}
Progress: experiments/logs/public_sets.log"""
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path

import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)

S = Path(__file__).parent
ROOT = S.parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(ROOT))
import end_extra  # noqa: E402
import source_sets  # noqa: E402
import switch_sets  # noqa: E402
from acro_sets import PATTERN as ACRO  # noqa: E402
from amb_en_eval import amb  # noqa: E402
from amb_en_sets import korean_base  # noqa: E402
from amb_ko_sets import EN_ZIPF, SYL  # noqa: E402
from jamo1_sets import JAMO1  # noqa: E402
from num_en_sets import NUM  # noqa: E402
from textcheck import expected  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

import nunchi.sentence_data as sd  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402

PUB = ROOT / "data" / "public"
PARTS = PUB / "parts"
OUT = PUB / "nunchi_bench.jsonl"
LOG = S / "logs" / "public_sets.log"
EVAL = ROOT / "data" / "eval"
SE = ROOT / "data" / "raw" / "stackexchange"
SITES = ("interpersonal", "parenting", "movies", "cooking", "travel", "workplace")

KC_FROM = 12_000_000  # internal sets read KcBERT dump lines 2M..12M
KC_POOL = 400_000  # raw lines the main cell draws from at random, as source_sets does from 2M..2.4M
WORDS = 20_000  # main cell, per source
# points (lines for switch / end_extra) per cell: the internal test halves' kcbert part
WANT_KO = {"switch": 2000, "end_extra": 1250, "amb_en": 2000, "jamo1": 2500, "amb_ko": 2500, "acro": 1000,
           "fused": 1000}
WANT_W1 = {"ko1": 2000, "mix1": 1000}
W1_CAP = 3
SWITCH_EN = 2000  # synthetic lines per Stack Exchange source
NUM_WANT = 1500  # number-word points per source
PER_KEY, MIN_KEY = 60, 15  # key collisions, as collision_sets.py per half
SENT_END = re.compile(r"(?<=[.!?])\s+|\n")
CUT = 20
# the set is shown on a public page: lines holding profanity, a slur, a hate term or an insult are left out of every
# cell, Korean and English, in any form people type them (whole words, jamo shorthand ㅅㅂ, Latin c발, X까, the ㅗ
# gesture), and so are lines with web leftovers (URL parts, article ids). Counts in the log. This makes the set
# cleaner than the internal one: public_eval.py shows by how much per cell.
ABUSE = re.compile("|".join((
    "씨발", "시발(?![점역])", "씨빨", "씨바", "씨팔", "시팔", "쉬발", "슈발", "씹", "십새", "십색", "개새", "개색", "개세끼", "새끼",
    "새기들", "생끼", "병신", "븅신", "빙신", "등신", "찐따", "좆", "좃", "조까", "존만", "존나", "존니", "존내", "졸라",
    "미친", "또라이", "썅", "염병", "지랄", "닥쳐", "꺼져", "애미", "애비", "느금", "니미", "엠창", "호로", "창녀", "걸레",
    "개소리", "개같", "개빡", "개꼰대", "쓰레기", "기레기", "썩을", "빨갱", "틀딱", "한남", "김치녀", "된장녀", "홍어",
    "쪽바리", "쪽발", "왜놈", "짱깨", "짱께", "짱개", "급식충", "맘충", "개돼지", "운지", "일베", "대깨", "문재앙",
    "닭근혜", "쥐박", "좌좀", "좌빨", "수꼴", "토착왜구", "토왜", "매국노", "개독", "개슬람", "똥남아", "흑형", "게이새",
    "충들", "놈", "년아", "년들아", "한녀", "쳐먹", "대갈", "죽여버", "죽어라", "뒤져", "뒈져", "디져")))
ABUSE_FORMS = re.compile(
    r"(?<![A-Za-z])(?:[cCsS]\s?[발빨봘]|[cC]8(?!\d))|[시씨]\s?8|[ㅅㅆ]\s?ㅂ|ㅂ\s?ㅅ|ㅄ|ㅈ\s?ㄹ|ㅈ\s?ㄴ|ㅅ\s?ㄲ|ㅁ\s?ㅊ"
    r"|ㄲ\s?ㅈ|ㄷ\s?ㅊ|ㅈ\s?같|[xX]\s?[까같발]|[gG]\s?랄|(?<![A-Za-z])(?:GR|[cC]\s?pal)(?![A-Za-z])|(?<![가-힣ㄱ-ㅣ])ㅗ+(?![가-힣ㄱ-ㅣ])"
    r"|(?i:\b(?:fuck\w*|f\*+k\w*|shit\w*|bullshit|bitch\w*|cunt\w*|asshole\w*|arsehole\w*|bastard\w*|dick(?:s|head\w*)?"
    r"|cock(?:s|sucker\w*)?|pussy|retard\w*|nigg\w*|fag(?:s|got\w*)?|whore\w*|slut\w*|wtf|stfu|crap\w*|damn\w*|goddamn\w*"
    r"|piss\w*|twat\w*|wank\w*|douche\w*|pricks?|moron\w*|idiot\w*)\b)")
# web leftovers in the cleaned lines: URL parts, query keys before an id, long ids (aid 0004071847 sid1 001, co kr)
JUNK = re.compile(r"(?i)(?<![A-Za-z0-9])(?:www\d*|https?|html?|php|aspx?|com|org|kr|petitions|contid|precseq|newsid|articleid|idxno|nbsp|원본링크)"
                  r"(?![A-Za-z0-9])|(?<![A-Za-z0-9])(?:aid|oid|sid\d?|mid)\s\d|(?<![A-Za-z])co kr(?![A-Za-z])|\d{7,}")


def unfit(text):
    """-> "abuse" or "junk" when a line must stay out of the public set, else None"""
    return "abuse" if ABUSE.search(text) or ABUSE_FORMS.search(text) else "junk" if JUNK.search(text) else None


def unfit_line(ln):
    """unfit on a cleaned line and on the text it types (some KcBERT lines hold decomposed jamo that compose into a
    filtered word)"""
    return unfit(ln) or unfit(text_of(to_sentence(ln)))


def text_of(sent):
    return " ".join(expected(k, y) for k, y in sent)


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def internal_lines(english=False):
    """every line (keys joined by spaces) of the internal eval sets; with english, the en_marked training lines too"""
    used = set()
    for p in EVAL.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if "w" in r:
                used.add(" ".join(k for k, _ in r["w"]))
    if english:
        for p in sd.OUT.glob("en_marked*.txt"):
            used.update(line.rstrip("\n") for line in p.open(encoding="utf-8"))
    return used


def row(set_, src, sent, pts, frm, key=None):
    r = {"set": set_, "src": src}
    if key is not None:
        r["key"] = key
    r.update({"w": [[k, "".join(map(str, y))] for k, y in sent], "pts": pts,
              "text": text_of(sent), "from": frm})
    return r


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    say(f"-> {path}: {len(rows):,} lines")


# ---------------------------------------------------------------- Korean: NSMC test, KcBERT dump lines KC_FROM+

def kc_raw(start, w1):
    """raw KcBERT lines from `start`; each also goes through the one-word cells (w1: their state)"""
    sd.SENT_KCBERT = 10 ** 9
    for i, text in enumerate(itertools.islice(sd.korean_lines("kcbert"), start, None)):
        if i % 1_000_000 == 0:
            say(f"  kcbert line {start + i:,}")
        one_word(text, w1)
        yield text


def one_word(text, w1):
    """word1_sets.py's ko1 / mix1 on one raw line: cleaned with the word minimum 1, at most W1_CAP of a word"""
    if all(w1["got"][c] >= n for c, n in WANT_W1.items()):
        return
    sd.MIN_W = 1
    toks = sd.clean(text, w1["kk"], 3)
    sd.MIN_W = 2
    if not toks:
        return

    def add(src, tok):
        if w1["seen"][tok] >= W1_CAP:
            return
        if unfit_line(tok):
            w1["abuse"] += 1
            return
        sent = to_sentence(tok)
        if len(sent) != 1:
            return
        w1["seen"][tok] += 1
        w1["rows"].append(row("word1", src, sent, [0], {"corpus": "kcbert"}))
        w1["got"][src] += 1

    if w1["got"]["ko1"] < WANT_W1["ko1"] and len(toks) == 1 and sd.HANGUL.match(toks[0]) and len(toks[0]) <= 5:
        add("ko1", toks[0])
    for t in toks:
        if sd.MIXED.match(t) and w1["got"]["mix1"] < WANT_W1["mix1"]:
            add("mix1", t)


def ko_cell(sent, got):
    """-> (set, pts) for the first cell that still wants this line and has a point in it, else None"""
    sw = switch_sets.points(sent)
    if sw and got["switch"] < WANT_KO["switch"]:
        return "switch", sw
    ends = [[i, c] for i, c in sw if c == end_extra.CAT]
    if ends and got["end_extra"] < WANT_KO["end_extra"]:
        return "end_extra", ends
    if not korean_base(sent):
        return None
    cands = {
        "amb_en": lambda: [i for i, (k, y) in enumerate(sent) if set(y) == {0} and amb(k)],
        "jamo1": lambda: [i for i, (k, y) in enumerate(sent) if set(y) == {1} and JAMO1.match(compose(k))],
        "amb_ko": lambda: [i for i, (k, y) in enumerate(sent) if set(y) == {1} and SYL.match(compose(k)) and amb(k)
                           and zipf_frequency(k.lower(), "en") >= EN_ZIPF],
        "acro": lambda: [i for i, (k, y) in enumerate(sent) if set(y) == {0} and ACRO.search(k)],
        "fused": lambda: [i for i, (k, y) in enumerate(sent) if len(set(y)) > 1 and not re.search(r"\d", k)],
    }
    for cell, f in cands.items():
        if got[cell] < WANT_KO[cell]:
            pts = f()
            if pts:
                return cell, pts
    return None


def build_ko():
    used = internal_lines()
    say(f"Korean: {len(used):,} internal eval lines excluded")
    rows, taken = [], set()

    # NSMC test: main only (its ~700 mixed lines are all in the internal switching set)
    lines = [ln for ln in source_sets.held_out_lines("nsmc") if ln not in used]
    n_all = len(lines)
    why = Counter(unfit_line(ln) for ln in lines)
    lines = [ln for ln in lines if not unfit_line(ln)]
    say(f"nsmc: of {n_all:,} lines left out: {why['abuse']:,} abuse, {why['junk']:,} junk")
    random.Random(0).shuffle(lines)
    n = 0
    for ln in lines:
        if n >= WORDS:
            break
        rows.append(row("main", "nsmc", to_sentence(ln), list(range(len(ln.split()))), {"corpus": "nsmc"}))
        taken.add(ln)
        n += len(ln.split())
    say(f"nsmc main: {n:,} words in {len(rows):,} lines (of {len(lines):,} usable)")

    w1 = {"kk": sd.ko_keys(), "seen": Counter(), "got": Counter(), "rows": [], "abuse": 0}
    raw = kc_raw(KC_FROM, w1)
    pool = [ln for ln in source_sets.iter_held_out("kcbert", texts=itertools.islice(raw, KC_POOL)) if ln not in used]
    n_all = len(pool)
    why = Counter(unfit_line(ln) for ln in pool)
    pool = [ln for ln in pool if not unfit_line(ln)]
    say(f"kcbert pool: of {n_all:,} lines left out: {why['abuse']:,} abuse, {why['junk']:,} junk")
    order = pool[:]
    random.Random(0).shuffle(order)
    n = 0
    for ln in order:
        if n >= WORDS:
            break
        rows.append(row("main", "kcbert", to_sentence(ln), list(range(len(ln.split()))), {"corpus": "kcbert"}))
        taken.add(ln)
        n += len(ln.split())
    say(f"kcbert main: {n:,} words from a pool of {len(pool):,} lines")

    got, lines_in, abusive = Counter(), Counter(), Counter()
    rest = (ln for ln in pool if ln not in taken)
    stream = itertools.chain(rest, source_sets.iter_held_out("kcbert", texts=raw))
    for k, ln in enumerate(stream):
        ko_done = all(got[c] >= n for c, n in WANT_KO.items())
        if ko_done and all(w1["got"][c] >= n for c, n in WANT_W1.items()):
            break
        if ko_done or ln in used or ln in taken:
            continue
        sent = to_sentence(ln)
        hit = ko_cell(sent, got)
        if hit is None:
            continue
        if unfit_line(ln):
            abusive[(hit[0], unfit_line(ln))] += 1
            continue
        cell, pts = hit
        taken.add(ln)
        rows.append(row(cell, "kcbert", sent, pts, {"corpus": "kcbert"}))
        got[cell] += 1 if cell in ("switch", "end_extra") else len(pts)
        lines_in[cell] += 1
        if sum(lines_in.values()) % 1000 == 0:
            say("  " + ", ".join(f"{c} {got[c]:,}/{n:,}" for c, n in WANT_KO.items())
                + "; " + ", ".join(f"{c} {w1['got'][c]:,}/{n:,}" for c, n in WANT_W1.items()))
    say("kcbert cells: " + ", ".join(f"{c} {got[c]:,} in {lines_in[c]:,} lines" for c in WANT_KO)
        + "; one word: " + ", ".join(f"{c} {w1['got'][c]:,}" for c in WANT_W1))
    say(f"left out (cell, why): {dict(abusive)}; one-word tokens {w1['abuse']:,}")
    write(PARTS / "ko.jsonl", rows + w1["rows"])


# ---------------------------------------------------------------- English: Stack Exchange, WikiText-103

BLOCK = re.compile(r"<(pre|code|blockquote)\b[^>]*>.*?</\1>", re.S | re.I)
BREAK = re.compile(r"</(p|li|h\d|div|tr)>|<br\s*/?>", re.I)
TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
TICK = re.compile(r"`[^`]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MENTION = re.compile(r"@\w+")


def xml_rows(path):
    for _, el in ET.iterparse(path):
        if el.tag == "row":
            yield el.attrib
            el.clear()


def se_licence(date):
    """Stack Exchange's licence by the date a text was posted (stackoverflow.com/help/licensing)"""
    return "CC BY-SA 2.5" if date < "2011-04-08" else "CC BY-SA 3.0" if date < "2018-05-02" else "CC BY-SA 4.0"


_SITE = {}


def site_maps(site):
    """-> (user id -> display name, post id -> question id)"""
    if site not in _SITE:
        users = {r["Id"]: r.get("DisplayName", "") for r in xml_rows(SE / site / "Users.xml")}
        q_of = {}
        for r in xml_rows(SE / site / "Posts.xml"):
            if r.get("PostTypeId") == "1":
                q_of[r["Id"]] = r["Id"]
            elif r.get("PostTypeId") == "2":
                q_of[r["Id"]] = r.get("ParentId")
        _SITE[site] = users, q_of
    return _SITE[site]


def with_links(m):
    """Stack Exchange's attribution rule (data/raw/stackexchange/license.txt): a direct link to the question, and
    each author's name linked to their profile (deleted users have a name and no profile)"""
    m["question_url"] = f"https://{m['site']}/questions/{m['question']}" if m.get("question") else None
    m["user_url"] = f"https://{m['site']}/users/{m['user']}" if m.get("user") else None
    return m


def se_docs(kind):
    """-> (plain text, attribution) for every post (kind "post": questions and answers) or comment of SITES"""
    for site in SITES:
        users, q_of = site_maps(site)
        if kind == "post":
            for r in xml_rows(SE / site / "Posts.xml"):
                if r.get("PostTypeId") not in ("1", "2"):
                    continue
                t = BREAK.sub("\n", BLOCK.sub(" ", r.get("Body", "")))
                t = URL.sub(" ", unescape(TAG.sub(" ", t)))
                uid = r.get("OwnerUserId")
                yield t, with_links({"site": f"{site}.stackexchange.com", "post": r["Id"], "question": q_of.get(r["Id"]),
                                     "author": users.get(uid) if uid else r.get("OwnerDisplayName", ""), "user": uid,
                                     "license": r.get("ContentLicense") or se_licence(r.get("CreationDate", ""))})
        else:
            for r in xml_rows(SE / site / "Comments.xml"):
                t = URL.sub(" ", MENTION.sub(" ", LINK.sub(r"\1", TICK.sub(" ", unescape(r.get("Text", ""))))))
                uid = r.get("UserId")
                yield t, with_links({"site": f"{site}.stackexchange.com", "comment": r["Id"],
                                     "question": q_of.get(r.get("PostId")),
                                     "author": users.get(uid) if uid else r.get("UserDisplayName", ""), "user": uid,
                                     "license": r.get("ContentLicense") or se_licence(r.get("CreationDate", ""))})


def keep_doc(meta, share):
    h = int(hashlib.md5(f"{meta['site']}/{meta.get('post') or meta.get('comment')}".encode()).hexdigest(), 16)
    return h % 1000 < share * 1000


def held_out_se(kind, src, used, taken, want_lines, share):
    """-> [(line, attribution)]: cleaned, cut lines (source_sets.iter_held_out) from a sample of documents, in
    random order, up to want_lines"""
    docs = [(t, m) for t, m in se_docs(kind) if keep_doc(m, share)]
    rnd = random.Random(0)
    rnd.shuffle(docs)
    say(f"  {src}: {len(docs):,} documents sampled")
    out = []
    for t, m in docs:
        for ln in source_sets.iter_held_out(src, texts=SENT_END.split(t)):
            if ln in used or ln in taken or unfit_line(ln):
                continue
            taken.add(ln)
            out.append((ln, m))
        if len(out) >= want_lines:
            break
    rnd.shuffle(out)
    return out


def cut_lines(texts, hit_of):
    """num_en_sets / collision_sets cutting: the 20 words around the first hit of a sentence -> (line, hit word)"""
    for sent in texts:
        toks = [t for t in filter(None, sd.SPLIT.split(re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", sent)))
                if sd.TOKEN_V2.match(t)]
        if len(toks) < 3:
            continue
        hit = next((i for i, t in enumerate(toks) if hit_of(t)), None)
        if hit is None:
            continue
        lo = max(0, min(hit - 10, len(toks) - CUT))
        yield " ".join(toks[lo:lo + CUT]), toks[hit]


def build_en():
    import pyarrow.parquet as pq  # after torch
    from nunchi.dubeolsik import to_keys
    from nunchi.en_marked import korean_edges, markers

    used = internal_lines(english=True)
    say(f"English: {len(used):,} internal eval and en_marked lines excluded")
    rows, taken = [], set()
    make = switch_sets.synthesizer(random.Random(0))
    for kind, src, share in (("post", "se_post", 0.05), ("comment", "se_comment", 0.1)):
        lines = held_out_se(kind, src, used, taken, 12_000, share)
        n, i = 0, 0
        while n < WORDS and i < len(lines):
            ln, m = lines[i]
            i += 1
            rows.append(row("main", src, to_sentence(ln), list(range(len(ln.split()))), m))
            n += len(ln.split())
        say(f"{src} main: {n:,} words in {i:,} lines")
        made, kinds = 0, Counter()
        for ln, m in lines[i:]:
            if made >= SWITCH_EN:
                break
            toks = ln.split()
            if len(toks) < 3:
                continue
            new, place = make(toks, made)
            if new is None:
                continue
            s = to_sentence(" ".join(new))
            pts = switch_sets.points(s)
            if pts and not unfit(text_of(s)):
                rows.append(row("switch", src, s, pts, dict(m, changed="a Korean word put in")))
                made += 1
                kinds[place] += 1
        say(f"{src} switching (synthetic): {made:,} lines {dict(kinds)}")

    # number words: WikiText-103 as num_en_sets reads it (the internal lines are skipped, so this continues where it
    # stopped), then Stack Exchange posts and comments in random order
    files = sorted((ROOT / "data" / "raw" / "wikitext103").glob("*.parquet"))
    random.Random(1).shuffle(files)
    got = 0
    for f in files:
        for text in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            for ln, _ in cut_lines(SENT_END.split(text or ""), NUM.match):
                if ln in used or ln in taken or unfit_line(ln):
                    continue
                taken.add(ln)
                s = to_sentence(ln)
                pts = [i for i, (k, _) in enumerate(s) if NUM.match(k)]
                if pts:
                    rows.append(row("num_en", "wikitext", s, pts, {"corpus": "wikitext103"}))
                    got += len(pts)
                if got >= NUM_WANT:
                    break
            if got >= NUM_WANT:
                break
        if got >= NUM_WANT:
            break
    say(f"wikitext number words: {got:,}")

    se_all = [(t, m) for kind in ("post", "comment") for t, m in se_docs(kind)]
    random.Random(2).shuffle(se_all)
    say(f"Stack Exchange documents: {len(se_all):,}")
    got = 0
    for t, m in se_all:
        for ln, _ in cut_lines(SENT_END.split(t), NUM.match):
            if ln in used or ln in taken or unfit_line(ln):
                continue
            taken.add(ln)
            s = to_sentence(ln)
            pts = [i for i, (k, _) in enumerate(s) if NUM.match(k)]
            if pts:
                rows.append(row("num_en", "se", s, pts, m))
                got += len(pts)
        if got >= NUM_WANT:
            break
    say(f"se number words: {got:,}")

    keys = {to_keys(w) for w, *_ in markers(*korean_edges())}
    base = lambda t: t if t in keys else re.sub(r"\d+$", "", t) if re.search(r"\d$", t) else None  # noqa: E731
    quick = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, keys), key=len, reverse=True))
                       + r")\d*(?![A-Za-z0-9])")
    per_key, coll = Counter(), []
    for t, m in se_all:
        if not quick.search(t):
            continue
        for ln, w in cut_lines(SENT_END.split(t), lambda tok: base(tok) in keys):
            k = base(w)
            if ln in used or ln in taken or per_key[k] >= PER_KEY or unfit_line(ln):
                continue
            taken.add(ln)
            s = to_sentence(ln)
            pts = [i for i, (tok, y) in enumerate(s) if base(tok) == k and set(y) == {0}]
            if pts:
                coll.append(row("collide", "se", s, pts, m, key=k))
                per_key[k] += len(pts)
    keep = {k for k, c in per_key.items() if c >= MIN_KEY}
    rows += [r for r in coll if r["key"] in keep]
    say(f"key collisions kept ({len(keep)}): " + ", ".join(f"{k} {per_key[k]}" for k in sorted(keep, key=lambda k: -per_key[k]))
        + "; too few: " + ", ".join(f"{k} {c}" for k, c in per_key.most_common() if k not in keep)
        + "; none: " + " ".join(sorted(keys - set(per_key))))
    write(PARTS / "en.jsonl", rows)


# ---------------------------------------------------------------- join: typo cells, ids, manifest

FROM_KEYS = ("corpus", "license", "site", "question", "question_url", "post", "comment", "author", "user", "user_url",
             "changed")
LICENCE = {"nsmc": "CC0-1.0", "kcbert": "CC BY-SA 4.0", "wikitext103": "CC BY-SA"}


def published(r, i):
    """one schema for every row, so Arrow (datasets, the Hub viewer) reads the file: plain indices in pts with the
    switching categories beside them in cats, key / cats null where they do not apply, the same fields in from"""
    pairs = r["set"] in ("switch", "end_extra")
    frm = dict(r["from"])
    frm.setdefault("corpus", "stackexchange" if "site" in frm else None)
    frm.setdefault("license", LICENCE.get(frm["corpus"]))
    return {"id": f"nb{i:06d}", "set": r["set"], "src": r["src"], "key": r.get("key"), "w": r["w"],
            "pts": [p[0] for p in r["pts"]] if pairs else r["pts"], "cats": [p[1] for p in r["pts"]] if pairs else None,
            "text": r["text"], "from": {k: frm.get(k) for k in FROM_KEYS}}


def join():
    from typo_sets import P_POS, SHIFT
    from nunchi.tagger import typo

    rows = [json.loads(line) for name in ("ko", "en") for line in (PARTS / f"{name}.jsonl").open(encoding="utf-8")]
    rnd = random.Random(7)
    extra = []
    for r in rows:
        if r["set"] != "main":
            continue
        s = [(k, [int(c) for c in y]) for k, y in r["w"]]
        w, pts = [], []
        for i, (k, y) in enumerate(s):
            if len(k) >= 2 and rnd.random() < P_POS:
                k2, y2 = typo(k, y, rnd)
                if k2 != k:
                    w.append((k2, y2))
                    pts.append(i)
                    continue
            w.append((k, y))
        if pts and not unfit(text_of(w)):
            extra.append(row("typo_pos", r["src"], w, pts, dict(r["from"], changed="keyboard typos put in")))
        if r["src"] in ("nsmc", "kcbert"):
            w, pts = [], []
            for i, (k, y) in enumerate(s):
                k2 = SHIFT.sub(lambda m: m.group(0).upper(), k) if set(y) == {1} else k
                if k2 != k:
                    pts.append(i)
                w.append((k2, y))
            if pts and not unfit(text_of(w)):
                extra.append(row("typo_shift", r["src"], w, pts, dict(r["from"], changed="Shift held into a vowel")))
    rows += extra
    rows = [published(r, i) for i, r in enumerate(rows)]
    write(OUT, rows)
    cells = Counter()
    for r in rows:
        cells[f"{r['set']} | {r['src']}" + (f" | {r['key']}" if r["key"] else "")] += len(r["pts"])
    manifest = {"file": OUT.name, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "lines": len(rows),
                "points": dict(sorted(cells.items())), "built": time.strftime("%Y-%m-%d")}
    (PUB / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"manifest: {len(rows):,} lines, {sum(cells.values()):,} points, sha256 {manifest['sha256'][:16]}")


if __name__ == "__main__":
    # set iteration order follows Python's per-run string hashing (the translation dictionary the synthetic lines draw
    # from, for one): fixed, so the same build gives the same file
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        sys.exit(subprocess.call([sys.executable, *sys.argv]))
    LOG.parent.mkdir(exist_ok=True)
    what = sys.argv[1]
    if what in ("ko", "all"):
        build_ko()
    if what in ("en", "all"):
        build_en()
    if what in ("join", "all"):
        join()
