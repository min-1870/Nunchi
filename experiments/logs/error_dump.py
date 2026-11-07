"""every word a system gets wrong in the end on dev (the same rule as system_eval.score, first-word rule included),
with its sentence, for reading the errors by hand (log 24: how much of what is left could context fix?).
One JSON line per error: group, sentence (expected text of every word), index, typed keys, expected, output,
the verdict at Space (P(ko), P(en), split sureness) and whether the revision rule changed it.
    python -u experiments/logs/error_dump.py NAME OUT.jsonl"""
import json
import sys

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
import system_eval as se  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.models import load  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
its = se.items("dev")
keys = [[k for k, _ in s] for s, _, _ in its]
m = load(name)
run = lambda d: st.decide(m, [k[:min(len(k), i + 1 + d)] for k, (_, i, _) in zip(keys, its)],  # noqa: E731
                          [i for _, i, _ in its])
r0 = run(0)
print("at Space done", flush=True)
rk = {d: run(d) for d in (2, 3, 5)}
n = 0
with open(out, "w", encoding="utf-8") as f:
    for j, (s, i, g) in enumerate(its):
        r, revised = r0[j], False
        l3, vl = rk[3][j], es.verdict(rk[3][j])
        t = se.FIRST_T if i == 0 else se.REV_T
        if vl != es.verdict(r, 0.7) and l3[1] >= t and es.verdict(rk[2][j]) == vl and es.verdict(rk[5][j]) == vl:
            r, revised = l3, True
        k, y = s[i]
        o, e = text_of(k, r), expected(k, y)
        if o == e:
            continue
        n += 1
        p = r0[j]
        f.write(json.dumps({"g": " | ".join(g), "sent": [expected(*w) for w in s], "i": i, "keys": k, "exp": e,
                            "out": o, "space": [round(p[2], 3), round(p[3], 3), round(p[1], 3)],
                            "later3": [round(l3[2], 3), round(l3[3], 3)], "revised": revised},
                           ensure_ascii=False) + "\n")
print(f"{name}: {n} errors -> {out}", flush=True)
