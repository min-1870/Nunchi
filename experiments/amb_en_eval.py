"""English words that are also common Korean words (so = 내, to = 새, a = ㅁ, go = 해) in real Korean sentences of the
switching set and of amb_en_set.jsonl (dev): wrong at Space per model. A word counts if its Hangul reading appears 200+ times in KcBERT +
NSMC and the English one has wordfreq zipf 3+ (one-letter words included).
    python -u experiments/amb_en_eval.py NAME:file ...   (-> experiments/logs/amb_en_scores.json, merged)"""
import json
import sys
from pathlib import Path

import torch
from wordfreq import zipf_frequency

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import switch_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.en_marked import korean_edges  # noqa: E402
from nunchi.tagger import device  # noqa: E402

OUT = S / "logs" / "amb_en_scores.json"
tot = korean_edges()[0]
amb = lambda k: tot.get(compose(k), 0) >= 200 and zipf_frequency(k.lower(), "en") >= 3.0 and not any(c.isdigit() for c in k)


SET = S.parent / "data" / "eval" / "amb_en_set.jsonl"  # experiments/amb_en_sets.py


def points(split):
    """-> (sentences, [(sentence, word)] of the words that count): the real Korean sentences of the switching
    set's half, then the lines of amb_en_set.jsonl (more held-out text, from 2026-11-03) when it is built"""
    data = [s for src, s, _ in switch_sets.load(split) if src in ("ko_casual", "web")]
    where = [(si, i) for si, s in enumerate(data) for i, (k, y) in enumerate(s) if set(y) == {0} and amb(k)]
    if SET.exists():
        for line in SET.open(encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == split:
                where += [(len(data), i) for i in r["pts"]]
                data.append([(k, [int(c) for c in y]) for k, y in r["w"]])
    return data, where


data, where = points("dev")
keys = [[k for k, _ in s] for s in data]


def main():
    scores = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    scores["_n"] = len(where)
    for spec in sys.argv[1:]:
        name, f = spec.split(":")
        m = st.StreamTagger(left=None, right=None, digits=True)
        m.load_state_dict(torch.load(S.parent / "models" / f, map_location="cpu"))
        m = m.to(device()).eval()
        res = st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])
        scores[name] = sum(text_of(data[si][i][0], r) != expected(*data[si][i]) for (si, i), r in zip(where, res))
        print(f"  {name:<12} {scores[name]}/{len(where)}", flush=True)
    OUT.write_text(json.dumps(scores, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
