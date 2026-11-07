import json, sys, numpy as np
from pathlib import Path
D=Path("experiments/logs/knobs"); pts=json.loads((D/"_points.json").read_text(encoding="utf-8"))
cells=sorted(c for c,v in pts["switch_cells"].items() if v>=20); cats=sorted({c.split(" | ")[0] for c in cells})
for nm in sys.argv[1:]:
    r=json.loads((D/f"{nm}.json").read_text(encoding="utf-8"))
    rate={c:r["switch_cells"].get(c,0)/pts["switch_cells"][c] for c in cells}
    o=100*np.mean([np.mean([rate[c] for c in cells if c.startswith(k+" | ")]) for k in cats])
    print(f"{nm:<8} objective {o:.2f}  switch errors {sum(r['kinds'].values())}  main {dict(sorted(r['main'].items()))}")
