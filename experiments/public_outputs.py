"""a registry model's predictions on the public set (public_sets.py), as bench/SPEC.md asks of any input method: per
line, the text the field holds when the line is done. Every word is decided with the eval rule (system_eval.decisions:
at Space, revised 3 words later when sure), so the words that score.py checks get exactly the answers
public_eval.py counts.

    python -u experiments/public_outputs.py NAME=MODEL      -> data/public/outputs/NAME.jsonl  {"id", "text"}
    python bench/score.py data/public/nunchi_bench.jsonl data/public/outputs/NAME.jsonl"""
import json
import sys
import time
from pathlib import Path

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
from public_sets import OUT as PUBLIC  # noqa: E402
from system_eval import decisions  # noqa: E402
from textcheck import text_of  # noqa: E402

from nunchi.models import load  # noqa: E402

OUTS = PUBLIC.parent / "outputs"


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    name, model = sys.argv[1].split("=")
    rows = [json.loads(line) for line in PUBLIC.open(encoding="utf-8")]
    its = []
    for r in rows:
        s = [(k, [int(c) for c in y]) for k, y in r["w"]]
        its += [(s, i, r["id"]) for i in range(len(s))]
    keys = [[k for k, _ in s] for s, _, _ in its]
    say(f"{len(rows):,} lines, {len(its):,} words; deciding with {model}")
    base = load(model)
    rev = None if base.right is not None else base
    words = {}
    for (s, i, rid), (_, r) in zip(its, decisions(its, keys, base, rev)):
        words.setdefault(rid, {})[i] = text_of(s[i][0], r)
    OUTS.mkdir(parents=True, exist_ok=True)
    out = OUTS / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            w = words[r["id"]]
            f.write(json.dumps({"id": r["id"], "text": " ".join(w[i] for i in range(len(w)))}, ensure_ascii=False) + "\n")
    say(f"-> {out}")


if __name__ == "__main__":
    main()
