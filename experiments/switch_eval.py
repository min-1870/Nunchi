"""score models on the switching set (switch_sets.py, dev half), at each point category: wrong at Space and
wrong in the end (after the revision rule in use: the decision 3 words later replaces the one at Space when
it is at least 95% sure and the same 2 and 5 words later; models that decide only at Space keep it).
A category's rate is the mean of the per-source rates (sources with at least MIN_N points of it), so a source
with more lines does not weigh more; then each source's own rate over all its points.

    python -u experiments/switch_eval.py [--seeds A,B] MODEL ...
  MODEL: NAME:file (a 10k stream model with digit keys, models/<file>), a registry name (TR3-51K), or
         BASE+REVISER (TX4-51K+RV1-51K: the first decides at Space, the second revises)
  --seeds A,B: two runs of one recipe with different seeds; their difference is printed as the seed noise.
         When they are the first two models, the others are also shown as a change vs their mean.
"2se": two standard errors of one model's rate from the point counts (binomial), averaged over the models.
Models without digit keys (TR1-51K, TX4-51K, RV1-51K: trained with numbers removed) see a digit as an unknown
key, so their number rows are not comparable. Counts go to experiments/logs/switch_eval.json."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import switch_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.models import load as load_named  # noqa: E402
from nunchi.tagger import device  # noqa: E402

ROOT = S.parent
LOG, JSON_OUT = S / "logs" / "switch_eval_log.txt", S / "logs" / "switch_eval.json"
MIN_N = 20
LOG.write_text("", encoding="utf-8")


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load(spec):
    if ":" in spec:
        m = st.StreamTagger(left=None, right=None, digits=True)
        m.load_state_dict(torch.load(ROOT / "models" / spec.split(":")[1], map_location="cpu"))
        return m.to(device()).eval()
    return load_named(spec)


args, seeds = sys.argv[1:], None
if args and args[0] == "--seeds":
    seeds, args = args[1].split(","), args[2:]
data = switch_sets.load("dev")
where = [(di, i, cat) for di, (_, _, pts) in enumerate(data) for i, cat in pts]
cells = [(cat, data[di][0]) for di, _, cat in where]
n = defaultdict(int)
for c in cells:
    n[c] += 1
keys = [[k for k, _ in s] for _, s, _ in data]


def run(m, d=0):
    return st.decide(m, [keys[di][:min(len(keys[di]), i + 1 + d)] for di, i, _ in where], [i for _, i, _ in where])


bad = {}  # (name, "space" | "end") -> {cell: wrong}
for spec in args:
    name = spec.split(":")[0]
    base_spec, _, rev_spec = spec.partition("+")
    base = load(base_spec)
    # a model built for a fixed window after the word (TX: right=1) decides at Space only
    rev = load(rev_spec) if rev_spec else None if base.right is not None else base
    r0 = run(base)
    rk = {d: run(rev, d) for d in (2, 3, 5)} if rev else {d: r0 for d in (2, 3, 5)}
    for stage in ("space", "end"):
        bad[name, stage] = defaultdict(int)
    for j, ((di, i, _), c) in enumerate(zip(where, cells)):
        k, y = data[di][1][i]
        a, l3 = r0[j], rk[3][j]
        vl = es.verdict(l3)
        take = rev is not None and vl != es.verdict(a, 0.7) and l3[1] >= 0.95 and \
            es.verdict(rk[2][j]) == vl and es.verdict(rk[5][j]) == vl
        for stage, r in (("space", a), ("end", l3 if take else a)):
            if text_of(k, r) != expected(k, y):
                bad[name, stage][c] += 1
    print(f"  scored {name}", flush=True)
JSON_OUT.write_text(json.dumps({"points": {f"{c} | {s}": v for (c, s), v in n.items()},
                                "wrong": {f"{nm} | {stg}": {f"{c} | {s}": v for (c, s), v in b.items()}
                                          for (nm, stg), b in bad.items()}}, ensure_ascii=False, indent=1),
                    encoding="utf-8")

names = list(dict.fromkeys(nm for nm, _ in bad))
cats = sorted({c for c, _ in n})
srcs = sorted({s for _, s in n})
rows = [(c, [(c, s) for s in srcs if n[c, s] >= MIN_N]) for c in cats]
rows += [(f"source {s}", [(c, s) for c in cats if n[c, s]]) for s in srcs]
vs = seeds is not None and names[:2] == seeds


def rate(nm, stage, members, pooled):
    """-> rate %, its standard error %"""
    b = bad[nm, stage]
    if pooled:
        tot = sum(n[c] for c in members)
        r = sum(b[c] for c in members) / tot
        return 100 * r, 100 * math.sqrt(r * (1 - r) / tot)
    rs = [(b[c] / n[c], n[c]) for c in members]
    r = sum(x for x, _ in rs) / len(rs)
    return 100 * r, 100 * math.sqrt(sum(x * (1 - x) / k for x, k in rs)) / len(rs)


for stage, title in (("space", "decided at Space"), ("end", "in the end (after the revision rule)")):
    say(f"=== switching set, dev, {title}: rate %, mean over sources" +
        (f" (change vs mean of {seeds[0]} and {seeds[1]}, %p)" if vs else ""))
    say(f"  {'':<22}{'points':>7}{'src':>4}" + "".join(f"{nm:>{20 if vs else 13}}" for nm in names)
        + (f"{'seed diff':>11}" if seeds else "") + f"{'2se':>7}")
    for label, members in rows:
        if not members:
            continue
        pooled = label.startswith("source")
        r = {nm: rate(nm, stage, members, pooled) for nm in names}
        base = (r[seeds[0]][0] + r[seeds[1]][0]) / 2 if vs else None
        out = []
        for nm in names:
            x = f"{r[nm][0]:>6.1f}"
            if vs:
                x += f" ({r[nm][0] - base:+5.1f})" if nm not in seeds else " " * 8
            out.append(x)
        se = sum(v[1] for v in r.values()) / len(r)
        say(f"  {label:<22}{sum(n[c] for c in members):>7}{1 if pooled else len(members):>4}"
            + "".join(f"{x:>{20 if vs else 13}}" for x in out)
            + (f"{abs(r[seeds[0]][0] - r[seeds[1]][0]):>11.1f}" if seeds else "") + f"{2 * se:>7.1f}")
