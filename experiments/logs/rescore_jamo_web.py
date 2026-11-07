"""bring score files to the reviewed single-jamo web set (2026-11-05): their "jamo1 | web" group was scored on all 461 dev
web items; system_eval.items now keeps only the ones a person would type (data/eval/jamo1_review.jsonl). Only that
group is scored again; every other group is left as it is. Files without the group are left alone.
    python -u experiments/logs/rescore_jamo_web.py [FILE.json ...]   (default: every file in experiments/logs/systems/)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
import system_eval as se  # noqa: E402

from nunchi.models import REGISTRY, load  # noqa: E402

OLD = {o.strip(): m["name"] for m in json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
       for o in (m.get("old_name") or "").split(",") if o.strip()}
now = lambda n: OLD.get(n.strip(), n.strip())  # noqa: E731
G = ("jamo1", "web")
K = " | ".join(G)
files = [Path(a) for a in sys.argv[1:]] or sorted(se.OUT.glob("*.json"))
cache = {}
for n, f in enumerate(files, 1):
    s = json.loads(f.read_text(encoding="utf-8"))
    split = s.get("split", "dev")
    if split not in cache:
        its = [x for x in se.items(split) if x[2] == G]
        cache[split] = (its, [[k for k, _ in x[0]] for x in its])
    its, keys = cache[split]
    if K not in s["points"] or s["points"][K] == len(its):
        print(f"[{n}/{len(files)}] {f.name}: nothing to do", flush=True)
        continue
    base_name, _, rev_name = s["system"].partition("+")
    base = load(now(base_name))
    rev = load(now(rev_name)) if rev_name else None if base.right is not None else base
    bad, bad0, late, broke = se.score(its, keys, base, rev)
    old = s["wrong"].get(K, 0), s["points"][K]
    for field, c in (("wrong", bad), ("wrong_space", bad0), ("late_fix", late), ("broke", broke)):
        d = s.setdefault(field, {})
        d.pop(K, None)
        if c.get(G):
            d[K] = c[G]
    s["points"][K] = len(its)
    s["jamo1_reviewed"] = True
    f.write_text(json.dumps(s, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] [{n}/{len(files)}] {f.name}: web jamo {old[0]}/{old[1]} -> "
          f"{s['wrong'].get(K, 0)}/{len(its)}", flush=True)
