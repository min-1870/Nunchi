"""what a model still gets wrong on the main set, by kind (2026-11-04)
    python -u experiments/logs/main_errors.py NAME"""
import random, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import system_eval as se
from textcheck import expected, text_of
import nunchi.stream_tagger as st, nunchi.eval_sentence as es
from nunchi.models import load
from nunchi.demo import compose
its = [x for x in se.items("dev") if x[2][0] == "main"]
keys = [[k for k, _ in s] for s, _, _ in its]
m = load(sys.argv[1])
run = lambda d: st.decide(m, [k[:min(len(k), i + 1 + d)] for k, (_, i, _) in zip(keys, its)], [i for _, i, _ in its])
r0, rk = run(0), {d: run(d) for d in (2, 3, 5)}
kinds, ex = Counter(), defaultdict(list)
for j, (s, i, g) in enumerate(its):
    r = r0[j]; l3, vl = rk[3][j], es.verdict(rk[3][j])
    if vl != es.verdict(r, 0.7) and l3[1] >= 0.95 and es.verdict(rk[2][j]) == vl and es.verdict(rk[5][j]) == vl:
        r = l3
    k, y = s[i]; out = text_of(k, r); e = expected(k, y)
    if out == e:
        continue
    src = "KO text" if g[1] in ("nsmc", "kcbert", "web") else "EN text"
    if len(set(y)) > 1:
        kind = "fused, digits" if re.search(r"\d", k) else "fused (both languages in one word)"
    elif set(y) == {1}:
        kind = "Korean, jamo only" if re.match(r"^[ㄱ-ㅣ]+$", compose(k)) else "Korean, 1 syllable" if len(compose(k)) == 1 else "Korean, 2+ syllables"
    else:
        kind = "English, upper case" if k.isupper() else "English, 1-2 letters" if len(k) <= 2 else "English, 3+ letters"
    kinds[(src, kind)] += 1
    ctx = " ".join(compose(kk) if set(yy) == {1} else kk for kk, yy in s[max(0, i - 3):i])
    ex[(src, kind)].append(f"{ctx} [[{e} -> {out}]]")
tot = sum(kinds.values())
print(f"{sys.argv[1]}: {tot} wrong of {len(its):,} main-set words")
random.seed(4)
for (src, kind), c in kinds.most_common():
    print(f"  {c:>4}  {100 * c / tot:4.0f}%  {src:<8} {kind}")
    for e in random.sample(ex[(src, kind)], min(3, len(ex[(src, kind)]))):
        print(f"            {e}")
