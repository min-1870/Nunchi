"""score a registry model, or a deciding model with a reviser, on everything the results table shows, in the end
(after the revision rule: the reviser's answer 3 words later replaces the one at Space when it is REV_T sure and
the same 2 and 5 words later; for the first word of a sentence, which has no words before it, FIRST_T is enough;
a model built for a fixed window after the word, TX, decides at Space only):
switching set cells (dev, per source), the extra KO>EN end points, main set sources, the key-collision check (per
key from collision_set.jsonl when it is built: groups "collide | KEY"; before, one pool "collide"),
the ambiguous English words in real Korean sentences and number words in real English lines (num_en_set, per
source). One JSON per system in experiments/logs/systems/ ("wrong": in
the end, "wrong_space": at Space, "late_fix": wrong at Space and right in the end, "broke": right at Space and
wrong in the end; per group, groups with 0 left out). Dev halves; --test scores the test halves instead (NAME@test.json; the
key-collision test half is data/eval/collision_test.jsonl, experiments/collision_test.py).
    python -u experiments/system_eval.py [--test] NAME=BASE[+REVISER] ...   (registry names: RV0-17K, TX4-17K+RV1-17K, TX4-51K+RV1-51K)"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
import amb_en_eval  # noqa: E402
import end_extra  # noqa: E402
import source_sets  # noqa: E402
import switch_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.eval_sentence as es  # noqa: E402
import nunchi.stream_tagger as st  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402
from nunchi.en_marked import korean_edges, markers  # noqa: E402
from nunchi.models import load  # noqa: E402

OUT = S / "logs" / "systems"
REV_T = 0.95
FIRST_T = 0.6  # from 2026-11-04: first words were 5x as often wrong as the rest (experiments/logs/policy_cells.py)


def items(split="dev"):
    """-> [(sentence, word index, group)] for every scored word of every part, on that half"""
    out = []
    for src, s, pts in switch_sets.load(split):
        out += [(s, i, ("switch", cat, src)) for i, cat in pts]
    for line in end_extra.OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            s = [(k, [int(c) for c in y]) for k, y in r["w"]]
            out += [(s, i, ("end_extra", r["src"])) for i, cat in r["pts"] if cat == end_extra.CAT]
    for src, sents in source_sets.load(split).items():
        out += [(s, i, ("main", src)) for s in sents for i in range(len(s))]
    per_key = S.parent / "data" / "eval" / "collision_set.jsonl"  # experiments/collision_sets.py, from 2026-11-03
    if per_key.exists():
        for line in per_key.open(encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == split:
                s = [(k, [int(c) for c in y]) for k, y in r["w"]]
                out += [(s, i, ("collide", r["key"])) for i in r["pts"]]
    else:
        keys = {to_keys(w) for w, *_ in markers(*korean_edges())}
        coll = "collision_eval.jsonl" if split == "dev" else "collision_test.jsonl"
        for line in (S.parent / "data" / "eval" / coll).open(encoding="utf-8"):
            s = [(k, [int(c) for c in y]) for k, y in json.loads(line)["w"]]
            out += [(s, i, ("collide",)) for i, (k, _) in enumerate(s)
                    if k in keys or (re.search(r"\d$", k) and re.sub(r"\d+$", "", k) in keys)]
    data, where = amb_en_eval.points(split)
    out += [(data[si], i, ("amb_en",)) for si, i in where]
    # number words in real English lines (experiments/num_en_sets.py), single jamo in Korean (jamo1_sets.py),
    # Korean words whose keys are a common English word (amb_ko_sets.py: 해 = go, 내 = so), upper-case English with
    # the Shift-held shape (acro_sets.py: EU, RM, TM), words of both languages with no space (fused_sets.py: QR코드),
    # one-word input with nothing around it (word1_sets.py, from 2026-11-05: ko1 대박, en1 thanks, mix1 apk의)
    # single jamo from web pages: only the ones a person would type (data/eval/jamo1_review.jsonl, reviewed by hand
    # blind to model output on 2026-11-04: list markers, separators and text about Hangul left out)
    review = S.parent / "data" / "eval" / "jamo1_review.jsonl"
    drop = {r["id"] for r in map(json.loads, review.open(encoding="utf-8")) if r["label"] in ("marker", "meta")}         if review.exists() else set()
    for fname, group in (("num_en_set.jsonl", "num_en_set"), ("jamo1_set.jsonl", "jamo1"),
                         ("amb_ko_set.jsonl", "amb_ko"), ("acro_set.jsonl", "acro"),
                         ("fused_set.jsonl", "fused"), ("word1_set.jsonl", "word1")):
        path = S.parent / "data" / "eval" / fname
        if path.exists():
            for li, line in enumerate(path.open(encoding="utf-8")):
                r = json.loads(line)
                if r["split"] == split:
                    s = [(k, [int(c) for c in y]) for k, y in r["w"]]
                    out += [(s, i, (group, r["src"])) for i in r["pts"]
                            if not (group == "jamo1" and f"w{li}_{i}" in drop)]
    # typos at Space (experiments/typo_sets.py): groups typo_pos | SRC and typo_shift | SRC
    typo = S.parent / "data" / "eval" / "typo_set.jsonl"
    if typo.exists():
        for line in typo.open(encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == split:
                s = [(k, [int(c) for c in y]) for k, y in r["w"]]
                out += [(s, i, (f"typo_{r['kind']}", r["src"])) for i in r["pts"]]
    return out


def score(its, keys, base, rev, first_t=None):
    """-> Counters per group: wrong in the end, wrong at Space, late fixes (wrong at Space, right in the end: the word
    changes on screen), broke (right at Space, wrong in the end)"""
    first_t = FIRST_T if first_t is None else first_t
    run = lambda m, d: st.decide(m, [k[:min(len(k), i + 1 + d)] for k, (_, i, _) in zip(keys, its)],  # noqa: E731
                                 [i for _, i, _ in its])
    r0 = run(base, 0)
    rk = {d: run(rev, d) for d in (2, 3, 5)} if rev else None
    bad, bad0, late, broke = Counter(), Counter(), Counter(), Counter()
    for j, (s, i, g) in enumerate(its):
        r = r0[j]
        w0 = text_of(s[i][0], r) != expected(*s[i])
        bad0[g] += w0
        if rk:
            l3, vl = rk[3][j], es.verdict(rk[3][j])
            t = first_t if i == 0 else REV_T
            if vl != es.verdict(r, 0.7) and l3[1] >= t and es.verdict(rk[2][j]) == vl and es.verdict(rk[5][j]) == vl:
                r = l3
        w = text_of(s[i][0], r) != expected(*s[i])
        bad[g] += w
        late[g] += w0 and not w
        broke[g] += w and not w0
    return bad, bad0, late, broke


def main():
    OUT.mkdir(exist_ok=True)
    args = [a for a in sys.argv[1:] if a != "--test"]
    split = "test" if "--test" in sys.argv[1:] else "dev"
    its = items(split)
    keys = [[k for k, _ in s] for s, _, _ in its]
    n = Counter(g for *_, g in its)
    for spec in args:
        name, sys_ = spec.split("=")
        base_name, _, rev_name = sys_.partition("+")
        base = load(base_name)
        rev = load(rev_name) if rev_name else None if base.right is not None else base
        bad, bad0, late, broke = score(its, keys, base, rev)
        enc = lambda c: {" | ".join(g): v for g, v in c.items() if v}
        (OUT / (f"{name}.json" if split == "dev" else f"{name}@{split}.json")).write_text(json.dumps(
            {"system": sys_, "split": split, "first_t": FIRST_T, "wrong": enc(bad), "wrong_space": enc(bad0),
             "late_fix": enc(late), "broke": enc(broke), "points": enc(n)}, ensure_ascii=False, indent=0), encoding="utf-8")
        N = sum(n.values())
        print(f"  {name} ({split}): {sum(bad.values())} wrong of {N} in the end, {sum(bad0.values())} at Space; "
              f"late fixes {sum(late.values())} ({1000 * sum(late.values()) / N:.2f} per 1000), "
              f"broke {sum(broke.values())} ({1000 * sum(broke.values()) / N:.2f} per 1000)", flush=True)


if __name__ == "__main__":
    main()
