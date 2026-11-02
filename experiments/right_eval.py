"""Late correction: each word decided at Space (k=0) and again after k more words have been typed
(k = 1, 2, 3, 5; fewer if the sentence ends first). Per k: wrong words, and against the k=0 decision,
fixed (wrong -> right) and broken (right -> wrong). A "safe" revision only replaces the k=0 decision when
the later one is at least T sure; that is what a correction you can trust looks like.
Sets: real dev sentences (NSMC + web) and English+particle real sentences (dev halves, leaks removed)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evalsets as ev  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.tagger import KO  # noqa: E402

KS = [0, 1, 2, 3, 5]
TS = [0.9, 0.99]


def load_sets():
    dev = ev.load_sets("dev", drop_leaked=True)
    y_of = lambda lab, k: [KO if lab == "ko" else 0] * len(k)
    real = [[(k, y_of(l, k)) for l, k in s] for n in ("nsmc", "web") for s in dev[n]]
    return {"real": real, "particle": ev.load_keyed("dev", drop_leaked=True)["particle"]}


def correct(r, y):
    v = es.verdict(r)
    if len(set(y)) > 1:
        return v[0] == "split" and v[1] == y
    return v[0] == ("ko" if y[0] == KO else "en")


def run(model, sents, entry_only=False):
    pre, tgt, where = [], [], []
    for si, s in enumerate(sents):
        keys = [k for k, _ in s]
        for i in range(len(s)):
            for k in KS:
                pre.append(keys[:min(len(s), i + 1 + k)])
                tgt.append(i)
                where.append((si, i, k))
    res = st.decide(model, pre, tgt)
    by = {}
    for (si, i, k), r in zip(where, res):
        by.setdefault((si, i), {})[k] = r
    c = Counter()
    for (si, i), rs in by.items():
        y = sents[si][i][1]
        entry = i > 0 and sents[si][i - 1][1][-1] == KO and y[0] != KO
        ok0 = correct(rs[0], y)
        for k in KS:
            okk = correct(rs[k], y)
            for tag, use in (("all", True), ("entry", entry)):
                if not use:
                    continue
                c[tag, k, "n"] += 1
                c[tag, k, "wrong"] += not okk
                c[tag, k, "fixed"] += okk and not ok0
                c[tag, k, "broken"] += ok0 and not okk
                for t in TS:  # revise only when the later decision is sure enough
                    take = k > 0 and rs[k][1] >= t and es.verdict(rs[k]) != es.verdict(rs[0])
                    ok = okk if take else ok0
                    c[tag, k, t, "wrong"] += not ok
                    c[tag, k, t, "fixed"] += ok and not ok0
                    c[tag, k, t, "broken"] += ok0 and not ok
    return c


def report(say, name, c, set_name):
    for tag in ("all", "entry"):
        n = c[tag, 0, "n"]
        say(f"  {name:<10} {set_name:<9} {tag + (' (ko->en)' if tag == 'entry' else ''):<14} n={n:<6}" +
            "  ".join(f"k={k}: {c[tag, k, 'wrong']:>3} wrong (+{c[tag, k, 'fixed']}/-{c[tag, k, 'broken']})" for k in KS))
        for t in TS:
            say(f"  {'':<10} {'':<9} {'  revise if >= ' + str(t):<20}" +
                "  ".join(f"k={k}: {c[tag, k, t, 'wrong']:>3} (+{c[tag, k, t, 'fixed']}/-{c[tag, k, t, 'broken']})" for k in KS[1:]))


if __name__ == "__main__":
    ev.LOG = sys.argv[1]
    sets = load_sets()
    root = Path(r"C:\Users\Minseok\Documents\GitHub\gksdud") / "models"
    for name, path, right in [(a, b, int(c)) for a, b, c in (x.split(":") for x in sys.argv[2:])]:
        m = st.StreamTagger(left=3, right=right)
        import torch
        m.load_state_dict(torch.load(root / path, map_location="cpu"))
        m = m.to(next(ev.load(root / path, 3).parameters()).device).eval()
        for set_name, sents in sets.items():
            report(ev.say, name, run(m, sents), set_name)
    ev.say("done")
