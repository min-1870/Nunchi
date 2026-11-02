"""ambiguous words in natural context. The ambiguous-key set caps each key at 150 sentences, so
rare keys weigh as much as 해 / 내 / 이; here sentences are drawn as they come (mC4 train shards 55-59: not
in training, which uses 0-49), so each key appears as often as it does in real text. Targets: Korean words of
at most 2 syllables whose keys also read as English (해 = go), and English words whose keys read as Korean.
Built once into amb_natural.jsonl (dev / test by hash); then each model is scored at several decision biases.
    python -u amb_natural_eval.py NAME:file ..."""
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
from textcheck import expected  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.ambiguous_eval import OUT_V3, ambiguous_keys  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.sentence_data import ko_keys  # noqa: E402
from nunchi.sentence_tagger import to_sentence  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402
from nunchi.web_mixed import RAW as MC4_VALID, SENT_END, train_sentence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repository
OUT = ROOT / "data" / "eval" / "amb_natural.jsonl"
LOG = S / "logs" / "amb_natural_log.txt"
N = 8000
BIASES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build():
    from nunchi.sentence_data import clean

    amb = ambiguous_keys()
    kk = ko_keys()
    seen = {"\t".join(k for k, _ in json.loads(line)["w"]) for line in OUT_V3.open(encoding="utf-8")}
    rows = []
    for i in range(55, 60):
        with gzip.open(MC4_VALID.parent / f"c4-ko-train-{i:05d}.json.gz", "rt", encoding="utf-8") as f:
            for line in f:
                for text in SENT_END.split(json.loads(line)["text"]):
                    toks = clean(text, kk, 3)
                    if not toks or len(toks) < 3:
                        continue
                    t = [j for j, w in enumerate(toks)
                         if (re.fullmatch(r"[가-힣]{1,2}", w) and to_keys(w).lower() in amb) or
                         (re.fullmatch(r"[A-Za-z]+", w) and w.lower() in amb)]
                    if t:
                        sent = to_sentence(" ".join(toks))
                        key = "\t".join(k for k, _ in sent)
                        if key not in seen:
                            seen.add(key)
                            rows.append({"w": [[k, "".join(map(str, y))] for k, y in sent], "t": t})
        say(f"  scanned shard {i}: {len(rows):,} sentences so far")
    random.Random(0).shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:N]) + "\n", encoding="utf-8")
    say(f"natural ambiguous set: {min(N, len(rows)):,} sentences (of {len(rows):,}) -> {OUT}")


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
    return compose(keys) if math.log(max(pko, 1e-12)) - math.log(max(pen, 1e-12)) > b else keys


if not OUT.exists():
    build()
data = []
for line in OUT.open(encoding="utf-8"):
    r = json.loads(line)
    w = [(k, [int(c) for c in y]) for k, y in r["w"]]
    if int(hashlib.md5("\t".join(k for k, _ in w).encode()).hexdigest(), 16) % 2 == 0:
        data.append((w, r["t"]))
where = [(si, i) for si, (_, t) in enumerate(data) for i in t]
n_ko = sum(data[si][0][i][1][0] == KO for si, i in where)
say(f"=== ambiguous words in natural context (dev): {len(where)} words, Korean truth {n_ko}, English {len(where) - n_ko}")
say("    per bias b: wrong Korean (became English) / wrong English (became Korean)")
dev_ = device()
keys = [[k for k, _ in s] for s, _ in data]
for spec in sys.argv[1:]:
    name, f = spec.split(":")
    m = st.StreamTagger(left=None, right=None, digits=True)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    m = m.to(dev_).eval()
    res = st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])
    cells = []
    for b in BIASES:
        bad = [(si, i) for (si, i), r in zip(where, res) if text_b(data[si][0][i][0], r, b) != expected(*data[si][0][i])]
        ko = sum(data[si][0][i][1][0] == KO for si, i in bad)
        cells.append(f"b{b:+.1f}: {ko}/{len(bad) - ko}")
    say(f"  {name:<8} " + "   ".join(cells))
say("done")
