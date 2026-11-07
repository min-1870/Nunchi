"""score an ensemble (log 24): several registry models of one kind, their sureness averaged per word (P of the best key path,
P(ko), P(en)), as system_eval scores one model (same items, same revision rule with the averaged numbers). The
key path is the one of the model most sure of its own path. Writes experiments/logs/systems/NAME.json like system_eval.
    python -u experiments/logs/ensemble_eval.py NAME MODEL MODEL [...] [--test]"""
import json
import sys
from collections import Counter

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
import system_eval as se  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.models import load  # noqa: E402


class Ensemble:
    def __init__(self, models):
        self.models = models


_decide = st.decide


def decide(model, prefixes, targets, batch=256):
    if not isinstance(model, Ensemble):
        return _decide(model, prefixes, targets, batch)
    outs = [_decide(m, prefixes, targets, batch) for m in model.models]
    res = []
    for rs in zip(*outs):
        n = len(rs)
        best = max(rs, key=lambda r: r[1])
        res.append((best[0], sum(r[1] for r in rs) / n, sum(r[2] for r in rs) / n, sum(r[3] for r in rs) / n))
    return res


st.decide = decide


def main():
    args = [a for a in sys.argv[1:] if a != "--test"]
    split = "test" if "--test" in sys.argv[1:] else "dev"
    name, members = args[0], args[1:]
    ens = Ensemble([load(m) for m in members])
    its = se.items(split)
    keys = [[k for k, _ in s] for s, _, _ in its]
    n = Counter(g for *_, g in its)
    bad, bad0, late, broke = se.score(its, keys, ens, ens)
    enc = lambda c: {" | ".join(g): v for g, v in c.items() if v}  # noqa: E731
    se.OUT.mkdir(exist_ok=True)
    (se.OUT / (f"{name}.json" if split == "dev" else f"{name}@{split}.json")).write_text(json.dumps(
        {"system": "&".join(members), "split": split, "first_t": se.FIRST_T, "ensemble": members, "wrong": enc(bad),
         "wrong_space": enc(bad0), "late_fix": enc(late), "broke": enc(broke), "points": enc(n)},
        ensure_ascii=False, indent=0), encoding="utf-8")
    N = sum(n.values())
    print(f"  {name} ({split}): {sum(bad.values())} wrong of {N} in the end, {sum(bad0.values())} at Space; "
          f"late fixes {sum(late.values())}, broke {sum(broke.values())}", flush=True)


if __name__ == "__main__":
    main()
