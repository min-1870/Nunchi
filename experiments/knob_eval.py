"""score 10k runs for the knob search: per model, wrong at Space on the switching set (per point category and
source, and per error kind) and on the main one-language set (per source). One JSON per model in
experiments/logs/knobs/; a model already there is skipped.
    python -u experiments/knob_eval.py NAME:file ...
Error kinds (kind_of) were named after reading TR3-51K's errors on the switching set."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import source_sets  # noqa: E402
import switch_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.tagger import device  # noqa: E402

OUT = S / "logs" / "knobs"
SYL = re.compile(r"^[가-힣]+$")


def kind_of(k, y, exp, cat):
    """the error kind of a wrong word at a switching point"""
    d = cat.split()[0].rstrip(",")
    if any(c.isdigit() for c in k):
        return "digits + letters"
    if re.fullmatch(r"[ㄱ-ㅣ]+", exp):
        return f"{d}: jamo only (ㅋㅋ ㅠㅠ) read as English"
    if y[0] == 0 and len(k) <= 2:
        return f"{d}: 1-2 letter English read as Korean" + (" (right after a switch)" if cat.endswith("after") else "")
    if y[0] == 0 and k.islower() and SYL.match(compose(k)):
        return f"{d}: English whose keys make Hangul syllables"
    return f"{d}: other " + ("English read as Korean" if y[0] == 0 else "Korean read as English")


def main():
    OUT.mkdir(exist_ok=True)
    sw = switch_sets.load("dev")
    sw_where = [(di, i, cat) for di, (_, _, pts) in enumerate(sw) for i, cat in pts]
    sw_keys = [[k for k, _ in s] for _, s, _ in sw]
    main_sets = source_sets.load("dev")
    mn = [(src, s) for src in sorted(main_sets) for s in main_sets[src]]
    mn_where = [(di, i) for di, (_, s) in enumerate(mn) for i in range(len(s))]
    mn_keys = [[k for k, _ in s] for _, s in mn]
    dev_ = device()
    for spec in sys.argv[1:]:
        name, f = spec.split(":")
        out = OUT / f"{name}.json"
        if out.exists():
            print(f"  {name}: done before", flush=True)
            continue
        m = st.StreamTagger(left=None, right=None, digits=True)
        m.load_state_dict(torch.load(S.parent / "models" / f, map_location="cpu"))
        m = m.to(dev_).eval()
        cell, kind = Counter(), Counter()
        res = st.decide(m, [sw_keys[di][:i + 1] for di, i, _ in sw_where], [i for _, i, _ in sw_where])
        for (di, i, cat), r in zip(sw_where, res):
            k, y = sw[di][1][i]
            exp = expected(k, y)
            if text_of(k, r) != exp:
                cell[f"{cat} | {sw[di][0]}"] += 1
                kind[kind_of(k, y, exp, cat)] += 1
        main_bad = Counter()
        res = st.decide(m, [mn_keys[di][:i + 1] for di, i in mn_where], [i for _, i in mn_where])
        for (di, i), r in zip(mn_where, res):
            k, y = mn[di][1][i]
            if text_of(k, r) != expected(k, y):
                main_bad[mn[di][0]] += 1
        out.write_text(json.dumps({"model": f, "switch_cells": cell, "kinds": kind, "main": main_bad},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name}: switch {sum(cell.values())} wrong, main {sum(main_bad.values())} wrong", flush=True)
    # denominators, once
    n_out = OUT / "_points.json"
    if not n_out.exists():
        n_cell = Counter(f"{cat} | {sw[di][0]}" for di, _, cat in sw_where)
        n_main = Counter(mn[di][0] for di, _ in mn_where)
        n_out.write_text(json.dumps({"switch_cells": n_cell, "main": n_main}, ensure_ascii=False, indent=1),
                         encoding="utf-8")


if __name__ == "__main__":
    main()
