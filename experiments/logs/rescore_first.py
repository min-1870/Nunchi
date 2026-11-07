"""bring the score files of experiments/logs/systems/ to the first-word rule (system_eval.FIRST_T, 2026-11-04) without
scoring every word again: the rule changes only the first word of each sentence, so those words are scored under the
old rule (REV_T) and the new one, and the difference is added to the file's counts. Files that already carry
first_t are left alone; TX systems (no revision) only get the mark.
    python -u experiments/logs/rescore_first.py [FILE.json ...]   (default: every file in experiments/logs/systems/)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
import system_eval as se  # noqa: E402

from nunchi.models import REGISTRY, load  # noqa: E402

# score files written before the renaming name their models by the old names
OLD = {o.strip(): m["name"] for m in json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
       for o in (m.get("old_name") or "").split(",") if o.strip()}
now = lambda n: OLD.get(n.strip(), n.strip())  # noqa: E731

files = [Path(a) for a in sys.argv[1:]] or sorted(se.OUT.glob("*.json"))
cache = {}
for n, f in enumerate(files, 1):
    s = json.loads(f.read_text(encoding="utf-8"))
    if "first_t" in s:
        print(f"[{n}/{len(files)}] {f.name}: already on the first-word rule", flush=True)
        continue
    split = s.get("split", "dev")
    if split not in cache:
        every = se.items(split)
        size = {}
        for *_, g in every:
            size[" | ".join(g)] = size.get(" | ".join(g), 0) + 1
        its = [x for x in every if x[1] == 0]
        cache[split] = (its, [[k for k, _ in x[0]] for x in its], size)
    its, keys, size = cache[split]
    # only the groups this file was scored on with the same words (older files hold fewer or smaller sets: the
    # 606-word ambiguous English, the pooled key collisions); their first words keep the old rule
    keep = [j for j, (_, _, g) in enumerate(its) if s["points"].get(" | ".join(g)) == size[" | ".join(g)]]
    sub, subk = [its[j] for j in keep], [keys[j] for j in keep]
    base_name, _, rev_name = s["system"].partition("+")
    base = load(now(base_name))
    rev = load(now(rev_name)) if rev_name else None if base.right is not None else base
    if rev is not None:
        old = se.score(sub, subk, base, rev, first_t=se.REV_T)
        new = se.score(sub, subk, base, rev)
        moved = 0
        for field, o, nw in (("wrong", old[0], new[0]), ("late_fix", old[2], new[2]), ("broke", old[3], new[3])):
            d = s.setdefault(field, {})
            for g in set(o) | set(nw):
                k = " | ".join(g)
                d[k] = d.get(k, 0) + nw[g] - o[g]
                if field == "wrong":
                    moved += nw[g] - o[g]
                if not d[k]:
                    del d[k]
        msg = f"{len(sub):,} first words, wrong {moved:+d}"
    else:
        msg = "no revision (TX): unchanged"
    s["first_t"] = se.FIRST_T
    f.write_text(json.dumps(s, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] [{n}/{len(files)}] {f.name}: {msg}", flush=True)
