"""the app's switch in the middle of a word (Early, app/core/src/typing.rs key()): after every letter the word so far
is judged with the words before it, and the app switches the word to the other language at once when the model is
sure enough (act_for: max(act, 0.999) for one letter, else act), the path in one language, once per word. With
act 0.85 (log 33) it fires on prefixes that read as a common word: wha = 좀 (P(ko) 0.87), so what shows 좀ㅅ until
Space puts it back. app_tune.py scores the boundary only; this scores the middle, for an Early threshold E of its
own (log 35).
Per scored dev word and typed mode (app_tune's rule: the previous word's language; a sentence's first word once in
each mode, half weight): the first prefix (1 .. n-1 keys, with a letter) where the switch fires, then the app's
boundary rule at Space (split 0.9, act 0.85) from the mode the word is in by then. Reported per E:
  early right  the switch shows the word's own language early
  early wrong  the switch shows the wrong language in the middle (좀ㅅ): a flicker, or worse if Space keeps it
  wrong / broke / missed at the end, as app_tune (revision one word later and the habit prior left out)
The model runs once on every prefix; outputs are cached.
    python -u experiments/logs/early_tune.py NAME [--test]   -> experiments/logs/early_tune_NAME[@test].txt"""
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected  # noqa: E402

from nunchi.tagger import EN, KO  # noqa: E402

FIRST_KEY, ACT, SPLIT = 0.999, 0.85, 0.9
ES = [0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, None]  # None: no switch in the middle
CHUNK = 50_000


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def letters(k):
    return sum(c.isascii() and c.isalpha() for c in k)


def show(keys, path):
    path = list(path) + [path[-1]] * (len(keys) - len(path))
    return expected(keys, path)


def cell(g):
    if g[0] == "switch":
        return g[1]
    if g[0] == "main":
        return "main KO" if g[1] in ("nsmc", "kcbert", "web") else "main EN"
    if g[0] == "word1":
        return f"word1 {g[1]}"
    return {"end_extra": "KO>EN end", "num_en_set": "number EN", "collide": "collisions"}.get(g[0], g[0])


def main():
    name = sys.argv[1]
    split = "test" if "--test" in sys.argv else "dev"
    its = se.items(split)
    keys = [[k for k, _ in s] for s, _, _ in its]
    r0, _ = pickle.loads(Path(f"experiments/logs/app_tune_{name}@{split}.pkl").read_bytes())
    assert len(r0) == len(its), "app_tune's cache is for other items"
    cache = Path(f"experiments/logs/early_tune_{name}@{split}.pkl")
    pre = [(j, L) for j, (s, i, _) in enumerate(its) for L in range(1, min(len(s[i][0]), 24))
           if letters(s[i][0][:L])]
    if cache.exists():
        out = pickle.loads(cache.read_bytes())
        say(f"cached prefix outputs: {cache}")
    else:
        import nunchi.stream_tagger as st
        from nunchi.models import load
        m = load(name)
        say(f"{name}: {len(its):,} {split} words, {len(pre):,} prefixes")
        out, t0 = [], time.time()
        for c in range(0, len(pre), CHUNK):
            part = pre[c:c + CHUNK]
            out += st.decide(m, [keys[j][:its[j][1]] + [its[j][0][its[j][1]][0][:L]] for j, L in part],
                             [its[j][1] for j, _ in part])
            done = c + len(part)
            say(f"  {done:,}/{len(pre):,} prefixes ({done / len(pre):.0%}; ETA "
                f"{(time.time() - t0) / done * (len(pre) - done) / 60:.1f} min)")
        cache.write_bytes(pickle.dumps(out))
    by = {}
    for (j, L), r in zip(pre, out):
        by.setdefault(j, []).append((L, r))
    rows = []  # (cell, weight, typed-right, [(L, target, p, one-letter)], final-word outputs, keys, expected)
    for j, (s, i, g) in enumerate(its):
        k, y = s[i]
        exp = expected(k, y)
        for typed, w in ([(KO, 0.5), (EN, 0.5)] if i == 0 else [(s[i - 1][1][-1], 1.0)]):
            cands = []
            for L, (path, pb, pko, pen) in by.get(j, []):
                t, p = (KO, pko) if pko > pen else (EN, pen)
                if t != typed and len(set(path)) == 1:
                    cands.append((L, t, p, letters(k[:L]) == 1))
            rows.append((cell(g), w, typed, cands, r0[j], k, exp))
    lines = [f"{name} ({split}): the app's switch in the middle of a word, threshold E (one letter: max(E, 0.999));"
             f" boundary act {ACT}, split {SPLIT}; {len(its):,} words",
             f"{'E':>6}{'switches':>10}{'right':>8}{'wrong':>8}{'end wrong':>11}{'broke':>8}{'missed':>8}"]
    per_cell = {}
    for E in ES:
        sw = right = bad_early = wrong = broke = 0.0
        cw = Counter()
        for c, w, typed, cands, (path0, pb0, pko0, pen0), k, exp in rows:
            mode, fired = typed, None
            if E is not None:
                for L, t, p, one in cands:
                    if p >= (max(E, FIRST_KEY) if one else E):
                        fired = t
                        break
            n = len(k)
            if fired is not None:
                mode = fired
                sw += w
                if show(k, [fired] * n) == exp:
                    right += w
                else:
                    bad_early += w
                    cw[c] += w
            t0, p0 = (KO, pko0) if pko0 > pen0 else (EN, pen0)
            if len(set(path0)) > 1 and pb0 >= SPLIT:
                ok = show(k, path0) == exp
            elif t0 != mode and p0 >= (max(ACT, FIRST_KEY) if letters(k) <= 1 else ACT):
                ok = show(k, [t0] * n) == exp
            else:
                ok = show(k, [mode] * n) == exp
            if not ok:
                wrong += w
                broke += w * (show(k, [typed] * n) == exp)
        lines.append(f"{'none' if E is None else E:>6}{sw:>10.0f}{right:>8.0f}{bad_early:>8.0f}{wrong:>11.0f}"
                     f"{broke:>8.0f}{wrong - broke:>8.0f}")
        per_cell[E] = cw
    lines += ["", "wrong switches in the middle, by cell (E = 0.85 / 0.95 / 0.99):"]
    for c, v in per_cell[0.85].most_common(15):
        lines.append(f"  {c:<32}{v:>7.0f}{per_cell[0.95][c]:>7.0f}{per_cell[0.99][c]:>7.0f}")
    lines += ["", "examples of wrong switches at E = 0.85 (prefix -> shown, the word, P):"]
    shown = 0
    for c, w, typed, cands, _, k, exp in rows:
        for L, t, p, one in cands:
            if p >= (max(0.85, FIRST_KEY) if one else 0.85):
                if show(k, [t] * len(k)) != exp and shown < 40:
                    lines.append(f"  [{c}] {k[:L]} -> {show(k[:L], [t] * L)} (word {exp}, P {p:.3f})")
                    shown += 1
                break
    text = "\n".join(lines) + "\n"
    Path(f"experiments/logs/early_tune_{name}" + ("@test" if split == "test" else "") + ".txt").write_text(
        text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
