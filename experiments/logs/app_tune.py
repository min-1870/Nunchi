"""tune the app's thresholds for a model (log 25): score dev the way the app acts (app/core/src/decide.rs, typing.rs
revise), for a grid of act / revise / split settings, per cell. The model runs once (at Space, and one word later)
and its outputs are cached; the grid is scored on the CPU.
App rule, per word:
  typed mode  the IME stays in the language the app last left it in: the previous word's language (taken as
              right); a sentence's first word is scored once typed in each mode (half weight each)
  at Space    a path that changes language inside the word, at least SPLIT sure -> split; else the more likely
              language, if not the typed one and at least ACT sure (one-letter words: max(ACT, 0.999), the app's
              FIRST_KEY) -> converted; else left as typed
  revise      once the next word is typed: a word not converted at Space, judged again with the next word; the
              other language, one language in the path, at least REVISE sure -> converted
Wrong = what the app leaves on screen differs from what was meant. Reported: wrong, and of those 'broke' (the
typed mode was right and the app changed it) and 'missed' (typed mode wrong, not converted or converted wrong).
    python -u experiments/logs/app_tune.py NAME [--test | --mine] [--pick=ACT,REVISE,SPLIT]   -> experiments/logs/app_tune_NAME[@test|@mine].txt
--mine scores Min's own typed messages instead (experiments/logs/my_typing.py), per word kind: first word of a
line, switch (the word starts in the other language than the word before ended), mixed (both in one word), or
same language as the word before."""
import itertools
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "experiments"); sys.path.insert(0, "experiments/logs"); sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected  # noqa: E402

from nunchi.tagger import EN, KO  # noqa: E402

FIRST_KEY = 0.999
ACTS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995]
REVS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 2.0]  # 2.0: no revision
SPLITS = [0.5, 0.6, 0.7, 0.8, 0.9]


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell(g):
    if g[0] == "switch":
        return g[1]
    if g[0] == "main":
        return "main KO" if g[1] in ("nsmc", "kcbert", "web") else "main EN"
    if g[0] == "mine":
        return f"mine {g[1]}"
    if g[0] == "word1":
        return f"word1 {g[1]}"
    return {"end_extra": "KO>EN end", "num_en_set": "number EN", "collide": "collisions"}.get(g[0], g[0])


def show(keys, path):
    path = list(path) + [path[-1]] * (len(keys) - len(path))
    return expected(keys, path)


def mine_items():
    import my_typing
    out = []
    for s in my_typing.load():
        for i, (k, y) in enumerate(s):
            kind = "first" if i == 0 else "mixed" if len(set(y)) > 1 else                 "switch" if y[0] != s[i - 1][1][-1] else "same KO" if y[0] == KO else "same EN"
            out.append((s, i, ("mine", kind)))
    return out


def main():
    name = sys.argv[1]
    split = "test" if "--test" in sys.argv else "mine" if "--mine" in sys.argv else "dev"
    cache = Path(f"experiments/logs/app_tune_{name}@{split}.pkl")
    its = mine_items() if split == "mine" else se.items(split)
    keys = [[k for k, _ in s] for s, _, _ in its]
    if cache.exists():
        r0, r1 = pickle.loads(cache.read_bytes())
        say(f"cached outputs: {cache}")
    else:
        import nunchi.stream_tagger as st
        from nunchi.models import load
        m = load(name)
        say(f"{name}: {len(its):,} {split} words, judging at Space")
        r0 = st.decide(m, [k[:i + 1] for k, (_, i, _) in zip(keys, its)], [i for _, i, _ in its])
        say("judging one word later")
        r1 = st.decide(m, [k[:min(len(k), i + 2)] for k, (_, i, _) in zip(keys, its)], [i for _, i, _ in its])
        cache.write_bytes(pickle.dumps((r0, r1)))
    # one row per (word, typed mode): every outcome the thresholds can pick between, computed once
    import numpy as np
    names, cols = {}, {k: [] for k in ("cell", "w", "one", "mixed0", "pb0", "conv0", "p0", "ok_split", "ok_conv",
                                       "has1", "rev1", "p1", "ok_rev", "ok_typed")}
    for j, (s, i, g) in enumerate(its):
        k, y = s[i]
        exp = expected(k, y)
        c = names.setdefault(cell(g), len(names))
        nlet = sum(ch.isascii() and ch.isalpha() for ch in k)
        path0, pb0, pko0, pen0 = r0[j]
        t0, p0 = (KO, pko0) if pko0 > pen0 else (EN, pen0)
        b = r1[j] if i + 1 < len(s) else None
        for typed, w in ([(KO, 0.5), (EN, 0.5)] if i == 0 else [(s[i - 1][1][-1], 1.0)]):
            row = dict(cell=c, w=w, one=nlet <= 1, mixed0=len(set(path0)) > 1, pb0=pb0, conv0=t0 != typed, p0=p0,
                       ok_split=show(k, path0) == exp, ok_conv=show(k, [t0] * len(k)) == exp,
                       ok_typed=show(k, [typed] * len(k)) == exp, has1=b is not None, rev1=False, p1=0.0, ok_rev=False)
            if b is not None:
                path1, _, pko1, pen1 = b
                t1, p1 = (KO, pko1) if pko1 > pen1 else (EN, pen1)
                row.update(rev1=t1 != typed and len(set(path1)) == 1, p1=p1, ok_rev=show(k, [t1] * len(k)) == exp)
            for key, v in row.items():
                cols[key].append(v)
    A = {k: np.array(v) for k, v in cols.items()}
    C = len(names)
    say(f"{len(A['w']):,} (word, mode) rows; scoring {len(ACTS) * len(REVS) * len(SPLITS)} settings")
    n_c = np.bincount(np.array([names[cell(g)] for _, _, g in its]), minlength=C).astype(float)

    def score(act, rev, spl):
        sp = A["mixed0"] & (A["pb0"] >= spl)
        th = np.where(A["one"], max(act, FIRST_KEY), act)
        cv = ~sp & A["conv0"] & (A["p0"] >= th)
        rv = ~sp & ~cv & A["has1"] & A["rev1"] & (A["p1"] >= rev)
        ok = np.where(sp, A["ok_split"], np.where(cv, A["ok_conv"], np.where(rv, A["ok_rev"], A["ok_typed"])))
        bad = ~ok
        wrong = np.bincount(A["cell"], A["w"] * bad, minlength=C)
        broke = np.bincount(A["cell"], A["w"] * (bad & A["ok_typed"]), minlength=C)
        lab = {v: k for k, v in names.items()}
        f = lambda a: Counter({lab[i]: a[i] for i in range(C)})  # noqa: E731
        return f(wrong), f(broke), f(wrong - broke), Counter({lab[i]: n_c[i] for i in range(C)})

    rows = []
    for act, rev, spl in itertools.product(ACTS, REVS, SPLITS):
        rows.append(((act, rev, spl), *score(act, rev, spl)))
    out = Path(f"experiments/logs/app_tune_{name}" + ("" if split == "dev" else f"@{split}") + ".txt")
    n = rows[0][4]
    cells = sorted(n, key=lambda c: (c.startswith("main"), c))
    cur = next(r for r in rows if r[0] == (0.99, 0.99, 0.7))
    with out.open("w", encoding="utf-8") as f:
        def p(line=""):
            print(line, flush=True)
            f.write(line + "\n")
        p(f"{name} ({split}): app rule, {sum(n.values()):,} words; current setting act 0.99 revise 0.99 split 0.7: "
          f"wrong {sum(cur[1].values()):.0f} (broke {sum(cur[2].values()):.0f}, missed {sum(cur[3].values()):.0f})")
        p("best 25 settings by total wrong (revise 2.0 = no revision):")
        p(f"{'act':>6}{'revise':>8}{'split':>7}{'wrong':>8}{'broke':>7}{'missed':>8}")
        for (a, r, s_), w, b, m_, _ in sorted(rows, key=lambda r: sum(r[1].values()))[:25]:
            p(f"{a:>6}{r:>8}{s_:>7}{sum(w.values()):>8.0f}{sum(b.values()):>7.0f}{sum(m_.values()):>8.0f}")
        p("\nact alone (revise 0.99, split 0.7):")
        for a in ACTS:
            r = next(x for x in rows if x[0] == (a, 0.99, 0.7))
            p(f"  act {a:<6} wrong {sum(r[1].values()):>6.0f}  broke {sum(r[2].values()):>5.0f}  missed {sum(r[3].values()):>5.0f}")
        p("\nrevise alone (act 0.99, split 0.7):")
        for rv in REVS:
            r = next(x for x in rows if x[0] == (0.99, rv, 0.7))
            p(f"  revise {rv:<6} wrong {sum(r[1].values()):>6.0f}  broke {sum(r[2].values()):>5.0f}  missed {sum(r[3].values()):>5.0f}")
        # a broken word (right as typed, changed by the app) may cost the user more than a missed one (as without
        # the app): best setting when one broken word counts as W missed ones
        p("\nif one broken word costs W missed ones (best setting for each W):")
        p(f"{'W':>4}{'act':>7}{'revise':>8}{'split':>7}{'wrong':>8}{'broke':>7}{'missed':>8}")
        for W in (1, 2, 3, 5, 10):
            a = min(rows, key=lambda r: W * sum(r[2].values()) + sum(r[3].values()))
            p(f"{W:>4}{a[0][0]:>7}{a[0][1]:>8}{a[0][2]:>7}{sum(a[1].values()):>8.0f}{sum(a[2].values()):>7.0f}"
              f"{sum(a[3].values()):>8.0f}")
        best = min(rows, key=lambda r: sum(r[1].values()))
        pick = next((tuple(map(float, a.split("=", 1)[1].split(","))) for a in sys.argv if a.startswith("--pick=")), None)
        if pick:  # the chosen setting (e.g. on test, once): its row, and its cells instead of the best's
            best = next(r for r in rows if r[0] == pick)
            p(f"\npicked {pick}: wrong {sum(best[1].values()):.0f}  broke {sum(best[2].values()):.0f}  "
              f"missed {sum(best[3].values()):.0f}")
        p(f"\nper cell, % wrong: current (0.99, 0.99, 0.7) vs {'picked' if pick else 'best'} {best[0]}")
        p(f"{'cell':<26}{'n':>7}{'current':>9}{'best':>8}")
        for c in cells:
            p(f"{c:<26}{n[c]:>7}{100 * cur[1][c] / n[c]:>8.2f}%{100 * best[1][c] / n[c]:>7.2f}%")
    say(f"-> {out}")


if __name__ == "__main__":
    main()
