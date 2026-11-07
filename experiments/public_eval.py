"""score a registry model on the public set (public_sets.py) and on the internal test halves, cell by cell, with the
same rule system_eval.py uses (decide at Space, revise 3 words later), and say for each cell whether the public set
stands in for the internal one: the two error rates differ by less than 2 SE of the difference (two independent
binomial rates: the sentences differ, so no pairing).

Cells pair a public source with the internal one it replaces (public_sets.py): Stack Exchange posts / comments for
IMDB / Reddit, KcBERT for ko_casual (the switching set's Korean casual lines are NSMC + KcBERT). The internal
amb_en cell here is amb_en_set.jsonl only (system_eval also adds the switching set's ambiguous words).

    python -u experiments/public_eval.py NAME=MODEL   (e.g. TR5n-102K=TR5n-102K)
      -> experiments/logs/public/NAME.json  {"public": {group: [wrong, n]}, "internal": {...}, "cells": [...]}
    The internal scores in an existing NAME.json for the same model are reused; only the public set is scored again."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import end_extra  # noqa: E402
import source_sets  # noqa: E402
import switch_sets  # noqa: E402
from public_sets import OUT as PUBLIC  # noqa: E402
from system_eval import score  # noqa: E402

from nunchi.models import load  # noqa: E402

EVAL = S.parent / "data" / "eval"
LOGS = S / "logs" / "public"


def sent_of(r):
    return [(k, [int(c) for c in y]) for k, y in r["w"]]


def public_items():
    out = []
    for line in PUBLIC.open(encoding="utf-8"):
        r = json.loads(line)
        s = sent_of(r)
        if r["set"] == "switch":
            out += [(s, i, ("switch", cat, r["src"])) for i, cat in zip(r["pts"], r["cats"])]
        elif r["set"] == "end_extra":
            out += [(s, i, ("end_extra", r["src"])) for i in r["pts"]]
        elif r["set"] == "collide":
            out += [(s, i, ("collide", r["key"])) for i in r["pts"]]
        else:
            out += [(s, i, (r["set"], r["src"])) for i in r["pts"]]
    return out


def internal_items(split="test"):
    out = []
    for src, s, pts in switch_sets.load(split):
        out += [(s, i, ("switch", cat, src)) for i, cat in pts]
    for line in end_extra.OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out += [(sent_of(r), i, ("end_extra", r["src"])) for i, cat in r["pts"] if cat == end_extra.CAT]
    for src, sents in source_sets.load(split).items():
        out += [(s, i, ("main", src)) for s in sents for i in range(len(s))]
    for line in (EVAL / "collision_set.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out += [(sent_of(r), i, ("collide", r["key"])) for i in r["pts"]]
    review = EVAL / "jamo1_review.jsonl"
    drop = {r["id"] for r in map(json.loads, review.open(encoding="utf-8")) if r["label"] in ("marker", "meta")}
    for fname, group in (("num_en_set.jsonl", "num_en"), ("amb_en_set.jsonl", "amb_en"), ("jamo1_set.jsonl", "jamo1"),
                         ("amb_ko_set.jsonl", "amb_ko"), ("acro_set.jsonl", "acro"), ("fused_set.jsonl", "fused"),
                         ("word1_set.jsonl", "word1")):
        for li, line in enumerate((EVAL / fname).open(encoding="utf-8")):
            r = json.loads(line)
            if r["split"] == split:
                out += [(sent_of(r), i, (group, r["src"])) for i in r["pts"]
                        if not (group == "jamo1" and f"w{li}_{i}" in drop)]
    for line in (EVAL / "typo_set.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out += [(sent_of(r), i, (f"typo_{r['kind']}", r["src"])) for i in r["pts"]]
    return out


def run(model, its):
    base = load(model)
    rev = None if base.right is not None else base
    keys = [[k for k, _ in s] for s, _, _ in its]
    bad = score(its, keys, base, rev)[0]
    n = Counter(g for *_, g in its)
    return {" | ".join(g): [bad[g], n[g]] for g in n}


SE = ("se_post", "se_comment")
IN_EN = ("imdb", "reddit")


def cells(pub, inn):
    """-> [(cell, public (wrong, n), internal (wrong, n))]"""
    def pool(d, match):
        w = n = 0
        for g, (a, b) in d.items():
            if match(g.split(" | ")):
                w, n = w + a, n + b
        return w, n

    out = []
    for src in ("nsmc", "kcbert"):
        out.append((f"main {src}", pool(pub, lambda g, s=src: g == ["main", s]), pool(inn, lambda g, s=src: g == ["main", s])))
    for ps, ins in (("se_post", "imdb"), ("se_comment", "reddit")):
        out.append((f"main English ({ps} / {ins})", pool(pub, lambda g, s=ps: g == ["main", s]),
                    pool(inn, lambda g, s=ins: g == ["main", s])))
    cats = sorted({g.split(" | ")[1] for g in pub if g.startswith("switch")} | {g.split(" | ")[1] for g in inn if g.startswith("switch")})
    for cat in cats:
        if cat.startswith("KO") or cat == "number, KO base":
            out.append((f"switch {cat}", pool(pub, lambda g, c=cat: g[:2] == ["switch", c] and g[2] == "kcbert"),
                        pool(inn, lambda g, c=cat: g[:2] == ["switch", c] and g[2] == "ko_casual")))
        else:
            out.append((f"switch {cat} (synthetic)", pool(pub, lambda g, c=cat: g[:2] == ["switch", c] and g[2] in SE),
                        pool(inn, lambda g, c=cat: g[:2] == ["switch", c] and g[2] in IN_EN)))
    out.append(("KO>EN end, more points", pool(pub, lambda g: g == ["end_extra", "kcbert"]),
                pool(inn, lambda g: g == ["end_extra", "ko_casual"])))
    for s in ("amb_en", "jamo1", "amb_ko", "acro", "fused"):
        out.append((s, pool(pub, lambda g, s=s: g == [s, "kcbert"]), pool(inn, lambda g, s=s: g == [s, "kcbert"])))
    for s in ("ko1", "mix1"):
        out.append((f"one word {s}", pool(pub, lambda g, s=s: g == ["word1", s]), pool(inn, lambda g, s=s: g == ["word1", s])))
    out.append(("number words (wikitext)", pool(pub, lambda g: g == ["num_en", "wikitext"]),
                pool(inn, lambda g: g == ["num_en", "wikitext"])))
    out.append(("number words (se / reddit)", pool(pub, lambda g: g == ["num_en", "se"]),
                pool(inn, lambda g: g == ["num_en", "reddit"])))
    common = {g.split(" | ")[1] for g in pub if g.startswith("collide")} & \
             {g.split(" | ")[1] for g in inn if g.startswith("collide")}
    out.append((f"key collisions, the {len(common)} keys in both",
                pool(pub, lambda g: g[0] == "collide" and g[1] in common),
                pool(inn, lambda g: g[0] == "collide" and g[1] in common)))
    out.append(("key collisions, all keys", pool(pub, lambda g: g[0] == "collide"), pool(inn, lambda g: g[0] == "collide")))
    for src in ("nsmc", "kcbert"):
        for kind in ("pos", "shift"):
            out.append((f"typo {kind} {src}", pool(pub, lambda g, k=kind, s=src: g == [f"typo_{k}", s]),
                        pool(inn, lambda g, k=kind, s=src: g == [f"typo_{k}", s])))
    out.append(("typo pos English", pool(pub, lambda g: g[0] == "typo_pos" and g[1] in SE),
                pool(inn, lambda g: g[0] == "typo_pos" and g[1] in IN_EN)))
    return out


def judge(p, q):
    (a, n), (b, m) = p, q
    if not n or not m:
        return None, None, "no points"
    r1, r2 = a / n, b / m
    se = math.sqrt(r1 * (1 - r1) / n + r2 * (1 - r2) / m) or 1 / max(n, m)
    return r1 - r2, 2 * se, "same" if abs(r1 - r2) <= 2 * se else "public easier" if r1 < r2 else "public harder"


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    name, model = sys.argv[1].split("=")
    pub = run(model, public_items())
    saved = LOGS / f"{name}.json"  # the internal test halves do not change: scored once, reused
    old = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
    inn = old["internal"] if old.get("model") == model and "internal" in old else run(model, internal_items())
    rows = []
    print(f"{'cell':<44}{'public':>18}{'internal test':>18}{'diff pp':>10}{'2 SE':>8}  verdict")
    for cell, p, q in cells(pub, inn):
        d, t, v = judge(p, q)
        rows.append({"cell": cell, "public": p, "internal": q, "diff": d, "two_se": t, "verdict": v})
        fmt = lambda x: f"{x[0]}/{x[1]} {100 * x[0] / x[1]:.2f}%" if x[1] else "-"  # noqa: E731
        print(f"{cell:<44}{fmt(p):>18}{fmt(q):>18}" + (f"{100 * d:>10.2f}{100 * t:>8.2f}" if d is not None else " " * 18)
              + f"  {v}", flush=True)
    (LOGS / f"{name}.json").write_text(json.dumps({"model": model, "public": pub, "internal": inn, "cells": rows},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
