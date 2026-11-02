"""did a variant learn more, or only lean further to English? For each model, move the Korean /
English decision by a bias b (Korean when log P(ko) - log P(en) > b; split words as before) and score the
ambiguous-key targets (English / Korean truth) and the real sentences (+ entry) at each b, dev halves with
numbers in place, decided at Space. A variant that only leans sits on the baseline's curve; a real gain
sits below it (fewer English errors at the same Korean errors).
    python -u bias_curve.py NAME:file ..."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import natural_sets  # noqa: E402
from textcheck import expected  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.ambiguous_eval import OUT_V3  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repository
LOG = S / "logs" / "bias_curve_log.txt"
dev_ = device()
BIASES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def text_b(keys, r, b):
    path, pb, pko, pen = r
    if len(set(path)) > 1 and pb >= es.SPLIT_T:
        path = list(path) + [path[-1]] * (len(keys) - len(path))
        out, start = "", 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or path[i] != path[start]:
                out += compose(keys[start:i]) if path[start] else keys[start:i]
                start = i
        return out
    ko = math.log(max(pko, 1e-12)) - math.log(max(pen, 1e-12)) > b
    return compose(keys) if ko else keys


amb = []
for line in OUT_V3.open(encoding="utf-8"):
    r = json.loads(line)
    w = [(k, [int(c) for c in y]) for k, y in r["w"]]
    if int(hashlib.md5("\t".join(k for k, _ in w).encode()).hexdigest(), 16) % 2 == 0:
        amb.append((w, r["t"]))
real = natural_sets.load("dev")
entry = {(si, b) for si, (s, idx) in enumerate(real) for a, b in zip(idx, idx[1:])
         if s[a][1][-1] == KO and len(set(s[b][1])) == 1 and s[b][1][0] != KO}


def decide(m, data):
    where = [(si, i) for si, (_, idx) in enumerate(data) for i in idx]
    keys = [[k for k, _ in s] for s, _ in data]
    return where, st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])


say("=== bias curves (dev, numbers in place, at Space): per bias b, ambiguous en / ko wrong, real wrong (entry)")
say(f"  ambiguous en words {sum(s[i][1][0] != KO for s, t in amb for i in t)}, ko words "
    f"{sum(s[i][1][0] == KO for s, t in amb for i in t)}; real words {sum(len(i) for _, i in real)}, entry {len(entry)}")
for spec in sys.argv[1:]:
    name, f = spec.split(":")
    m = st.StreamTagger(left=None, right=None, digits=True)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    m = m.to(dev_).eval()
    wa, ra = decide(m, amb)
    wr, rr = decide(m, real)
    cells = []
    for b in BIASES:
        bad = [(si, i) for (si, i), r in zip(wa, ra) if text_b(amb[si][0][i][0], r, b) != expected(*amb[si][0][i])]
        en = sum(amb[si][0][i][1][0] != KO for si, i in bad)
        badr = [(si, i) for (si, i), r in zip(wr, rr) if text_b(real[si][0][i][0], r, b) != expected(*real[si][0][i])]
        cells.append(f"b{b:+.1f}: {en}/{len(bad) - en} r{len(badr)}(e{sum(x in entry for x in badr)})")
    say(f"  {name:<6} " + "  ".join(cells))
say("done")
