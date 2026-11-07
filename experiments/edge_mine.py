"""edge cases the eval sets may miss (log 34): held-out Korean web text of every kind (mC4 ko validation + train
shards 50-59; training reads shards 0-49) run through a model, every word scored, the errors grouped by word shape
and by site, each shape's share of web text set against its share of the dev eval sets.
Lines are cleaned as the eval sets are (source_sets.iter_held_out: Hangul / Latin / digit tokens, long tokens and
Korean typed in English mode dropped), not with the training pool's filters (English-like words only, at most
half Latin), so shapes those filters keep out of training (letter + digit codes) are in. Only lines holding a
Latin or digit token are taken; at most PER_SITE lines per site and PER_TOKEN lines per such token, so one page or
one word cannot fill the set.
    python -u experiments/edge_mine.py build           -> data/eval/edge_web.jsonl  {"site", "w": [[keys, labels]]}
    python -u experiments/edge_mine.py score MODEL     -> experiments/logs/edge_MODEL.jsonl (every word)
    python -u experiments/edge_mine.py report MODEL    -> experiments/logs/edge_report_MODEL.txt"""
import gzip
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))

OUT = S.parent / "data" / "eval" / "edge_web.jsonl"
LOGS = S / "logs"
SHARDS = list(range(50, 60))
LINES = 30_000
PER_SITE = 40
PER_TOKEN = 5
CHUNK = 20_000  # words per scoring step (progress)
HANG = re.compile(r"[가-힣]")
JAMO = re.compile(r"[ㄱ-ㅣ]")
LAT = re.compile(r"[A-Za-z]")
DIG = re.compile(r"\d")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def interesting(tok):
    return bool(LAT.search(tok) or DIG.search(tok))


def _scan(path):
    import source_sets
    import nunchi.sentence_data as sd
    from nunchi.web_mixed import SENT_END, SPAM
    source_sets._KK, source_sets._TRAIN = sd.ko_keys(), set()
    out, per_site = [], Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            site = urlparse(doc.get("url", "")).netloc
            if per_site[site] >= PER_SITE:
                continue
            texts = [t for t in SENT_END.split(doc["text"]) if not SPAM.search(t)]
            for ln in source_sets.iter_held_out("web", texts=texts):
                if per_site[site] < PER_SITE and any(interesting(t) for t in ln.split()):
                    out.append((site, ln))
                    per_site[site] += 1
    return path.name, out


def build():
    from nunchi.sentence_tagger import to_sentence
    from nunchi.web_mixed import RAW as MC4_VALID
    files = [MC4_VALID] + [MC4_VALID.parent / f"c4-ko-train-{i:05d}.json.gz" for i in SHARDS]
    say(f"scanning {len(files)} files in parallel")
    t0, cands = time.time(), []
    with Pool(min(6, len(files))) as pool:
        for n, (name, part) in enumerate(pool.imap_unordered(_scan, files), 1):
            cands += part
            say(f"  {name}: {len(part):,} lines ({n}/{len(files)}, {n / len(files):.0%}; ETA "
                f"{(time.time() - t0) / n * (len(files) - n) / 60:.1f} min)")
    random.Random(0).shuffle(cands)
    seen, per_site, per_tok, rows = set(), Counter(), Counter(), []
    for site, ln in cands:
        toks = ln.split()
        odd = [t for t in toks if interesting(t)]
        if ln in seen or per_site[site] >= PER_SITE or all(per_tok[t] >= PER_TOKEN for t in odd):
            continue
        seen.add(ln)
        per_site[site] += 1
        per_tok.update(odd)
        rows.append({"site": site, "w": [[k, "".join(map(str, y))] for k, y in to_sentence(ln)]})
        if len(rows) >= LINES:
            break
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    say(f"{OUT}: {len(rows):,} lines from {len(per_site):,} sites ({len(cands):,} candidates), "
        f"{sum(len(r['w']) for r in rows):,} words")


def load_rows():
    rows = []
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        rows.append((r["site"], [(k, [int(c) for c in y]) for k, y in r["w"]]))
    return rows


def score(name):
    """every word as system_eval scores it: decided at Space, revised by the fixed rule (error_dump.py)"""
    import system_eval as se
    from textcheck import expected, text_of
    import nunchi.eval_sentence as es
    import nunchi.stream_tagger as st
    from nunchi.models import load
    rows = load_rows()
    its = [(si, i) for si, (_, s) in enumerate(rows) for i in range(len(s))]
    keys = [[k for k, _ in s] for _, s in rows]
    m = load(name)
    out_path = LOGS / f"edge_{name}.jsonl"
    say(f"{name}: {len(its):,} words in {len(rows):,} lines -> {out_path}")
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for c in range(0, len(its), CHUNK):
            part = its[c:c + CHUNK]
            run = lambda d: st.decide(m, [keys[si][:min(len(keys[si]), i + 1 + d)] for si, i in part],  # noqa: E731
                                      [i for _, i in part])
            r0, rk = run(0), {d: run(d) for d in (2, 3, 5)}
            for j, (si, i) in enumerate(part):
                r, revised = r0[j], False
                l3, vl = rk[3][j], es.verdict(rk[3][j])
                t = se.FIRST_T if i == 0 else se.REV_T
                if vl != es.verdict(r, 0.7) and l3[1] >= t and es.verdict(rk[2][j]) == vl and \
                        es.verdict(rk[5][j]) == vl:
                    r, revised = l3, True
                k, y = rows[si][1][i]
                e, o = expected(k, y), text_of(k, r)
                f.write(json.dumps([rows[si][0], si, i, k, e, o, round(r0[j][2], 3), revised],
                                   ensure_ascii=False) + "\n")
            done = c + len(part)
            say(f"  {done:,}/{len(its):,} words ({done / len(its):.0%}; ETA "
                f"{(time.time() - t0) / done * (len(its) - done) / 60:.1f} min)")
    say("done")


def shape(e):
    """word shape class from the expected text"""
    h, j, a, d = bool(HANG.search(e)), bool(JAMO.search(e)), bool(LAT.search(e)), bool(DIG.search(e))
    if a and (h or j):
        cls = "Latin then Hangul (apk의)" if LAT.match(e) else "Hangul then Latin (채널A)"
        return cls + (" + digit" if d else "")
    if a and d:
        return "letters + digits (gpt4, A100)" if LAT.match(e) else "digits + letters (10k, 5G)"
    if a:
        n = len(e)
        case = "lower" if e.islower() else "UPPER" if e.isupper() else "Title" if e[0].isupper() and e[1:].islower() \
            else "MiXed"
        if n == 1:
            return f"Latin, 1 letter ({'lower' if e.islower() else 'upper'})"
        return f"Latin {case}, {'2-3' if n <= 3 else '4+'} letters"
    if (h or j) and d:
        return "digits + Hangul (10점)" if DIG.match(e) else "Hangul + digits"
    if j and not h:
        return "jamo only (ㅋㅋ)"
    if h and j:
        return "Hangul with bare jamo"
    if h:
        return "Hangul"
    if d:
        return "digits only"
    return "other"


def report(name):
    import system_eval as se
    from textcheck import expected
    words = [json.loads(line) for line in (LOGS / f"edge_{name}.jsonl").open(encoding="utf-8")]
    n, bad, ex = Counter(), Counter(), defaultdict(Counter)
    site_n, site_bad = Counter(), Counter()
    for site, si, i, k, e, o, pko, rev in words:
        c = shape(e)
        n[c] += 1
        site_n[site] += 1
        if o != e:
            bad[c] += 1
            site_bad[site] += 1
            ex[c][f"{e}->{o}"] += 1
    # the same shapes in the dev eval sets, and TR5n-102K's error there when its dump exists
    dev = Counter()
    for s, i, g in se.items("dev"):
        dev[shape(expected(*s[i]))] += 1
    dump = LOGS / f"errors_{name}.jsonl"
    dev_bad = Counter(shape(json.loads(line)["exp"]) for line in dump.open(encoding="utf-8")) if dump.exists() \
        else Counter()
    N, D = sum(n.values()), sum(dev.values())
    lines = [f"{name} on held-out web text: {N:,} words, {sum(bad.values()):,} wrong ({100 * sum(bad.values()) / N:.2f}%); "
             f"dev eval sets {D:,} words", "",
             f"{'shape':<36}{'web words':>10}{'share':>7}{'wrong':>7}{'rate':>7}{'+-2se':>7}"
             f"{'dev share':>10}{'dev rate':>9}{'web/dev':>8}"]
    for c, k in sorted(n.items(), key=lambda kv: -bad[kv[0]]):
        r = bad[c] / k
        se2 = 2 * math.sqrt(max(r, 0.5 / k) * (1 - r) / k)
        ds = dev[c] / D
        dr = f"{100 * dev_bad[c] / dev[c]:.2f}" if dev[c] and dump.exists() else "-"
        ratio = f"{(k / N) / ds:.1f}" if ds else "none"
        lines.append(f"{c:<36}{k:>10,}{100 * k / N:>6.1f}%{bad[c]:>7,}{100 * r:>6.2f}%{100 * se2:>6.2f}"
                     f"{100 * ds:>9.2f}%{dr:>9}{ratio:>8}")
    lines += ["", "most common errors per shape (expected->output: count):"]
    for c, _ in bad.most_common():
        lines.append(f"  {c}: " + "; ".join(f"{x} {v}" for x, v in ex[c].most_common(25)))
    lines += ["", "sites with the highest error rate (200+ words; rate, words, errors):"]
    rated = [(site_bad[s] / site_n[s], s) for s in site_n if site_n[s] >= 200]
    for r, s in sorted(rated, reverse=True)[:25]:
        lines.append(f"  {s:<40}{100 * r:6.2f}%{site_n[s]:>7,}{site_bad[s]:>6}")
    text = "\n".join(lines) + "\n"
    (LOGS / f"edge_report_{name}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "score":
        score(sys.argv[2])
    elif cmd == "report":
        report(sys.argv[2])
