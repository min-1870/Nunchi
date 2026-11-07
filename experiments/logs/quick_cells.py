"""a quick look at a few cells only (after the revision rule, as system_eval): amb_en, jamo1, amb_ko by default.
    python -u experiments/logs/quick_cells.py NAME ... """
import math, os, sys
from collections import Counter
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import system_eval as se
from textcheck import expected, text_of
import nunchi.stream_tagger as st
import nunchi.eval_sentence as es
from nunchi.models import load

WANT = tuple(os.environ.get("CELLS", "amb_en,jamo1,amb_ko").split(","))
its = [x for x in se.items("dev") if x[2][0] in WANT]
keys = [[k for k, _ in s] for s, _, _ in its]
for name in sys.argv[1:]:
    m = load(name)
    run = lambda d: st.decide(m, [k[:min(len(k), i + 1 + d)] for k, (_, i, _) in zip(keys, its)], [i for _, i, _ in its])
    r0, rk = run(0), {d: run(d) for d in (2, 3, 5)}
    n, bad = Counter(), Counter()
    for j, (s, i, g) in enumerate(its):
        r = r0[j]
        l3, vl = rk[3][j], es.verdict(rk[3][j])
        if vl != es.verdict(r, 0.7) and l3[1] >= 0.95 and es.verdict(rk[2][j]) == vl and es.verdict(rk[5][j]) == vl:
            r = l3
        w = text_of(s[i][0], r) != expected(*s[i])
        n[g[0]] += 1; bad[g[0]] += w
        part = f"{g[0]} {'KO' if g[-1] in ('nsmc', 'kcbert', 'web') else 'EN'}"
        n[part] += 1; bad[part] += w
    print(f"{name:<14} " + "  ".join(f"{c} {100 * bad[c] / n[c]:5.2f}% ({bad[c]}/{n[c]})" for c in sorted(n)), flush=True)
