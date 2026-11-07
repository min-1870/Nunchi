"""check set for key collisions (en_marked.py v2): English words whose keys are a Korean marker's (WW = ㅉㅉ,
gg = ㅎㅎ, mm = ㅡㅡ ...) in real English lines held out from training. Wrong at Space, per word.
    python -u experiments/collision_eval.py NAME:file ..."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.en_marked import korean_edges, markers  # noqa: E402
from nunchi.tagger import device  # noqa: E402

ROOT = S.parent
data = [[(k, [int(c) for c in y]) for k, y in json.loads(line)["w"]]
        for line in (ROOT / "data" / "eval" / "collision_eval.jsonl").open(encoding="utf-8")]
keys = {to_keys(w) for w, *_ in markers(*korean_edges())}
where = [(si, i) for si, s in enumerate(data) for i, (k, _) in enumerate(s)
         if k in keys or (re.search(r"\d$", k) and re.sub(r"\d+$", "", k) in keys)]
for spec in sys.argv[1:]:
    name, f = spec.split(":")
    m = st.StreamTagger(left=None, right=None, digits=True)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    m = m.to(device()).eval()
    res = st.decide(m, [[k for k, _ in data[si]][:i + 1] for si, i in where], [i for _, i in where])
    bad = Counter()
    for (si, i), r in zip(where, res):
        k, y = data[si][i]
        if text_of(k, r) != expected(k, y):
            bad[re.sub(r"\d+$", "", k)] += 1
    print(f"  {name:<12} wrong {sum(bad.values())}/{len(where)}  {dict(bad.most_common(8))}", flush=True)
