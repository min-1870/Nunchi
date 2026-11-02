"""TR versions side by side on the dev halves (leak filter incl. web50 and the v2 pools).
  A. decided at Space: real (+ entry ko->en), real+typo, particle, ambiguous en / ko, pure ko / en, and the
     digits / Hangul+Latin set (all words, and the words holding digits or Hangul+Latin)
  B. revising itself with the rule in use (>= 95% sure 3 words after, same answer with 2 and 5 after):
     fixed / broken -> wrong after; TX4-long + RV1-long the same way for reference
  C. hand-made sentences (probe)
    python -u tr_eval.py <title> NAME:file:digits ...
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import evalsets as ev  # noqa: E402
import right_eval as re_  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.ambiguous_eval import OUT_V2  # noqa: E402
from nunchi.context import build  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.sentence_data import OUT as SENT  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402
from nunchi.make_noise import with_typos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repository
ev.LOG = str(S / "logs" / "tr_eval_log.txt")
say = ev.say
dev_ = device()


def load(f, left=None, right=None, digits=False):
    m = st.StreamTagger(left=left, right=right, digits=digits)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    return m.to(dev_).eval()


title, specs = sys.argv[1], sys.argv[2:]
models = {}
for spec in specs:
    name, f, dg = spec.split(":")
    models[name] = load(f, digits=dg == "1")
dev = ev.load_sets("dev", drop_leaked=True)
y_of = lambda lab, k: [KO if lab == "ko" else 0] * len(k)
real_lk = [s for n in ("nsmc", "web") for s in dev[n]]
amb, amb_t = [], []
for line in OUT_V2.open(encoding="utf-8"):
    r = json.loads(line)
    if int(hashlib.md5("\t".join(k for k, _ in r["w"]).encode()).hexdigest(), 16) % 2 == 0:
        amb.append([(k, [int(c) for c in y]) for k, y in r["w"]]); amb_t.append(set(r["t"]))
code = []
for line in (SENT / "web_code_eval.txt").open(encoding="utf-8"):
    row = [(k, [int(c) for c in y]) for k, y in (w.rsplit(":", 1) for w in line.rstrip("\n").split("\t"))]
    if int(hashlib.md5("\t".join(k for k, _ in row).encode()).hexdigest(), 16) % 2 == 0:
        code.append(row)
SETS = {"real": [[(k, y_of(l, k)) for l, k in s] for s in real_lk],
        "real+typo": [[(k, y_of(l, k)) for l, k in s] for s in with_typos(real_lk, random.Random(7))],
        "particle": ev.load_keyed("dev", drop_leaked=True)["particle"], "amb": amb, "code": code}
for kind in ("ko", "en"):
    SETS[f"pure {kind}"] = [[(to_keys(w), y_of(l, to_keys(w))) for l, w in s] for s in build()["test"][kind]]
SCOPE = {k: [set(range(len(s))) for s in v] for k, v in SETS.items()}
SCOPE["amb"] = amb_t
SCOPE["code"] = [{i for i, (k, _) in enumerate(s) if not k.isdigit()} for s in code]
ENTRY = [{i for i in range(1, len(s)) if s[i - 1][1][-1] == KO and s[i][1][0] != KO and len(set(s[i][1])) == 1}
         for s in SETS["real"]]
# code words: keys holding a digit, or Hangul keys then Latin in one word (labels KO then EN)
CODE_W = [{i for i, (k, y) in enumerate(s) if not k.isdigit() and (re.search(r"[0-9]", k) or (y[0] == KO and 0 in y))}
          for s in code]


def text_of(keys, r):
    lang, path = es.verdict(r)
    if lang == "split":
        path = list(path) + [path[-1]] * (len(keys) - len(path))  # words longer than MAX_KEYS: path is cut
        out, start = "", 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or path[i] != path[start]:
                out += compose(keys[start:i]) if path[start] else keys[start:i]
                start = i
        return out
    return compose(keys) if lang == "ko" else keys


def expected(k, y):
    out, start = "", 0
    for i in range(1, len(k) + 1):
        if i == len(k) or y[i] != y[start]:
            out += compose(k[start:i]) if y[start] == KO else k[start:i]
            start = i
    return out


def ok(set_name, si, i, r):
    """Right answer? On the digits set by the text that comes out: digits type the same in either mode, so
    a path that differs only on digits is not wrong."""
    k, y = SETS[set_name][si][i]
    if set_name == "code":
        return text_of(k, r) == expected(k, y)
    return re_.correct(r, y)


def run(m, set_name, d=0):
    sents = SETS[set_name]
    where = [(si, i) for si, s in enumerate(sents) for i in sorted(SCOPE[set_name][si])]
    keys = [[k for k, _ in s] for s in sents]
    return where, st.decide(m, [keys[si][:min(len(keys[si]), i + 1 + d)] for si, i in where], [i for _, i in where])


say(f"=== {title} (dev halves, wrong/words)")
say("--- A. decided at Space")
for name, m in models.items():
    cells = []
    for set_name, sents in SETS.items():
        where, res = run(m, set_name)
        bad = [(si, i) for (si, i), r in zip(where, res) if not ok(set_name, si, i, r)]
        if set_name == "real":
            cells.append(f"real {len(bad)}/{len(where)} (entry {sum(i in ENTRY[si] for si, i in bad)}/{sum(map(len, ENTRY))})")
        elif set_name == "amb":
            en = sum(sents[si][i][1][0] != KO for si, i in where)
            b_en = sum(sents[si][i][1][0] != KO for si, i in bad)
            cells.append(f"amb en {b_en}/{en} ko {len(bad) - b_en}/{len(where) - en}")
        elif set_name == "code":
            cells.append(f"code {len(bad)}/{len(where)} (code words {sum(i in CODE_W[si] for si, i in bad)}/{sum(map(len, CODE_W))})")
        else:
            cells.append(f"{set_name} {len(bad)}/{len(where)}")
    say(f"  {name:<12} " + "   ".join(cells))


def revised(base, rev, set_name):
    sents = SETS[set_name]
    where, r0 = run(base, set_name)
    rk = {d: run(rev, set_name, d)[1] for d in (2, 3, 5)}
    fixed = broken = wrong = 0
    for n, (si, i) in enumerate(where):
        a, l3 = r0[n], rk[3][n]
        va, vl = es.verdict(a, 0.7), es.verdict(l3)
        take = vl != va and l3[1] >= 0.95 and es.verdict(rk[2][n]) == vl and es.verdict(rk[5][n]) == vl
        c0, c1 = ok(set_name, si, i, a), ok(set_name, si, i, l3 if take else a)
        fixed += c1 and not c0
        broken += c0 and not c1
        wrong += not c1
    return f"{set_name} +{fixed}/-{broken} -> {wrong}/{len(where)}"


say("--- B. revising itself (95% sure, stable at 2/3/5 words after)")
pairs = [(n, m, m) for n, m in models.items()]
if (ROOT / "models" / "rv1_long.pt").exists():
    pairs.append(("TX4-long + RV1-long", load("tx4_long.pt", 3, 1), load("rv1_long.pt")))
for name, b, r in pairs:
    say(f"  {name:<20} " + "   ".join(revised(b, r, s) for s in ("real", "particle", "amb", "code")))

say("--- C. hand-made sentences, decided at Space")
CASES = ["tx4 모델 학습 끝났어", "그럼 tr 로 하자", "gpt4 보다 나아", "rtx4090 비싸다", "i7 노트북 샀어",
         "s f 영화 좋아해", "별 5 개 줬어", "2019년 12월 3일", "시즌2 는 별로", "채널A 뉴스 봤어", "ㅅㄱ 했어 오늘",
         "ㅋㅋ 웃기다", "ㅠㅠ 슬퍼", "5km 뛰었어"]


for s in CASES:
    keys = ["".join(to_keys(r) if re.match(r"[가-힣ㄱ-ㅣ]", r) else r for r in re.findall(r"[가-힣ㄱ-ㅣ]+|[A-Za-z0-9]+", w))
            for w in s.split()]
    outs = []
    for name, m in models.items():
        res = st.decide(m, [keys[:i + 1] for i in range(len(keys))], list(range(len(keys))))
        outs.append(f"{name}: " + " ".join(text_of(k, r) for k, r in zip(keys, res)))
    say(f"  {s:<22} | " + "  | ".join(outs))
say(f"=== {title} done")
