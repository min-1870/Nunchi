"""screen short runs on the dev sets with numbers in place, decided at Space, scored by text:
real mixed sentences (+ entry), ambiguous-key English / Korean truth (v3), digits / Hangul+Latin words.
    python -u short_eval.py <title> NAME:file ...   (TR recipe, digit keys)"""
import hashlib
import json
import sys
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import natural_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.ambiguous_eval import OUT_V3  # noqa: E402
from nunchi.sentence_data import OUT as SENT  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402

LOG = S / "logs" / "short_eval_log.txt"
ROOT = Path(__file__).resolve().parent.parent  # the repository
dev_ = device()


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def dev_rows(path, keyed):
    out = []
    for line in path.open(encoding="utf-8"):
        if keyed:
            r = json.loads(line)
            w, t = [(k, [int(c) for c in y]) for k, y in r["w"]], r["t"]
        else:
            w = [(k, [int(c) for c in y]) for k, y in (x.rsplit(":", 1) for x in line.rstrip("\n").split("\t"))]
            t = [i for i, (k, y) in enumerate(w) if any(c.isdigit() for c in k) and not k.isdigit() or
                 (y[0] == KO and 0 in y)]
        if int(hashlib.md5("\t".join(k for k, _ in w).encode()).hexdigest(), 16) % 2 == 0:
            out.append((w, t))
    return out


real = natural_sets.load("dev")
amb = dev_rows(OUT_V3, True)
code = dev_rows(SENT / "web_code_eval.txt", False)
entry = {(si, b) for si, (s, idx) in enumerate(real) for a, b in zip(idx, idx[1:])
         if s[a][1][-1] == KO and len(set(s[b][1])) == 1 and s[b][1][0] != KO}
# entry split by what follows: insert (Korean again after it: 저희 H 모터스) / switch (English goes on)
nxt = {(si, b): (c if c is not None else None) for si, (s, idx) in enumerate(real)
       for b, c in zip(idx, idx[1:] + [None])}
insert = {e for e in entry if nxt[e] is not None and real[e[0]][0][nxt[e]][1][0] == KO}
switch = {e for e in entry if nxt[e] is not None and real[e[0]][0][nxt[e]][1][0] != KO}


def revised_wrong(m, data):
    """wrong words after revising itself (>= 95% sure 3 words after, same answer with 2 and 5 after)"""
    import nunchi.eval_sentence as es
    where = [(si, i) for si, (_, idx) in enumerate(data) for i in idx]
    keys = [[k for k, _ in s] for s, _ in data]
    pre = lambda d: [keys[si][:min(len(keys[si]), i + 1 + d)] for si, i in where]
    tg = [i for _, i in where]
    r0 = st.decide(m, pre(0), tg)
    rk = {d: st.decide(m, pre(d), tg) for d in (2, 3, 5)}
    bad = []
    for n, (si, i) in enumerate(where):
        a, l3 = r0[n], rk[3][n]
        vl = es.verdict(l3)
        take = vl != es.verdict(a, 0.7) and l3[1] >= 0.95 and es.verdict(rk[2][n]) == vl and es.verdict(rk[5][n]) == vl
        if text_of(data[si][0][i][0], l3 if take else a) != expected(*data[si][0][i]):
            bad.append((si, i))
    return bad


def wrong(m, data):
    where = [(si, i) for si, (_, idx) in enumerate(data) for i in idx]
    keys = [[k for k, _ in s] for s, _ in data]
    res = st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])
    return [(si, i) for (si, i), r in zip(where, res) if text_of(data[si][0][i][0], r) != expected(*data[si][0][i])], where


title, specs = sys.argv[1], sys.argv[2:]
say(f"=== {title} (dev, numbers in place, at Space): wrong/words")
for spec in specs:
    name, f = spec.split(":")
    m = st.StreamTagger(left=None, right=None, digits=True)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    m = m.to(dev_).eval()
    b, w = wrong(m, real)
    br = revised_wrong(m, real)
    cells = [f"real {len(b)}/{len(w)} (entry {sum(x in entry for x in b)}/{len(entry)})",
             f"insert {sum(x in insert for x in b)}->{sum(x in insert for x in br)}/{len(insert)}",
             f"switch {sum(x in switch for x in b)}->{sum(x in switch for x in br)}/{len(switch)}"]
    b, w = wrong(m, amb)
    en = [x for x in w if amb[x[0]][0][x[1]][1][0] != KO]
    b_en = sum(amb[si][0][i][1][0] != KO for si, i in b)
    cells.append(f"amb en {b_en}/{len(en)} ko {len(b) - b_en}/{len(w) - len(en)}")
    b, w = wrong(m, code)
    cells.append(f"number words {len(b)}/{len(w)}")
    say(f"  {name:<14} " + "   ".join(cells))
say("done")
