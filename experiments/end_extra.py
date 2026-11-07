"""more points for the thinnest real cell of the switching set, KO>EN end (a Korean sentence ending in an English
word): the set had ~240 dev points per source. Lines from the same held-out text, kept only if they hold that
point, so the cell's own rate is not tilted (sampling within a cell); lines already in the switching set are
left out. Up to EXTRA lines per source (ko_casual: KcBERT dump lines 2M..2M+READ, which training never reads;
web: mC4 ko validation).
    python -u experiments/end_extra.py build              -> data/eval/switch_end_extra.jsonl
    python -u experiments/end_extra.py score NAME:file ... -> KO>EN end per model on dev, old points + new"""
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "eval" / "switch_end_extra.jsonl"
EXTRA = 2500
READ = 3_000_000
CAT = "KO>EN end"


def build():
    import source_sets
    import switch_sets
    import nunchi.sentence_data as sd
    from nunchi.sentence_tagger import to_sentence

    used = {" ".join(k for k, _ in json.loads(line)["w"]) for line in switch_sets.OUT.open(encoding="utf-8")}
    source_sets.held_out_lines("nsmc")  # loads the training-line filter once
    rows = []

    def take(src, texts):
        n = 0
        for text in texts:
            if n >= EXTRA:
                break
            toks = sd.clean(source_sets.APOS.sub("", text), source_sets._KK, 3)
            if not toks:
                continue
            for i in range(0, len(toks), source_sets.CHUNK):
                piece = toks[i:i + source_sets.CHUNK]
                line = " ".join(piece)
                if sum(not t.isdigit() for t in piece) < sd.MIN_W or line in source_sets._TRAIN or line in used:
                    continue
                if any(len(t) > source_sets.LONG for t in piece) or any(source_sets.wrong_mode(t) for t in piece):
                    continue
                s = to_sentence(line)
                if any(cat == CAT for _, cat in switch_sets.points(s)):
                    used.add(line)
                    rows.append({"split": source_sets._half(line), "src": src,
                                 "w": [[k, "".join(map(str, y))] for k, y in s], "pts": switch_sets.points(s)})
                    n += 1
        print(f"  {src}: {n} lines", flush=True)

    sd.SENT_KCBERT = 2_000_000 + READ
    sd.MAX_W = 10 ** 9
    take("ko_casual", itertools.islice(sd.korean_lines("kcbert"), 2_000_000, None))
    take("web", source_sets.raw_lines("web"))
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"-> {OUT}", flush=True)


def score(specs):
    import numpy as np
    import torch
    import switch_sets
    from textcheck import expected, text_of
    import nunchi.stream_tagger as st
    from nunchi.tagger import device

    data = [r for r in switch_sets.load("dev")]
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "dev":
            data.append((r["src"], [(k, [int(c) for c in y]) for k, y in r["w"]], [tuple(p) for p in r["pts"]]))
    where = [(di, i) for di, (_, _, p) in enumerate(data) for i, c in p if c == CAT]
    keys = [[k for k, _ in s] for _, s, _ in data]
    n = Counter(data[di][0] for di, _ in where)
    out = {"points": n}
    for spec in specs:
        name, f = spec.split(":")
        m = st.StreamTagger(left=None, right=None, digits=True)
        m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
        m = m.to(device()).eval()
        res = st.decide(m, [keys[di][:i + 1] for di, i in where], [i for _, i in where])
        bad = Counter(data[di][0] for (di, i), r in zip(where, res) if text_of(data[di][1][i][0], r) != expected(*data[di][1][i]))
        out[name] = {"wrong": bad, "rate": 100 * float(np.mean([bad[s] / n[s] for s in n]))}
        print(f"  {name:<14} {out[name]['rate']:.2f}%  {dict(bad)} of {dict(n)}", flush=True)
    (S / "logs" / "end_extra_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else score(sys.argv[2:])
