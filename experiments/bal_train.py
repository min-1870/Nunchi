"""balanced short-word lines (log 24): the amb x4 lines (experiments/amb_en_train.py: Korean-base lines holding a
lower-case English word whose keys also read as Korean, go = 해) plus as many Korean-base lines where the same keys
are the Korean word (해, 새, ㄴ), key by key (a line is taken only while every key in it is short). The x4 lines alone taught "short keys in Korean text are English"
(amb_ko 3 -> 14%); with both readings equally common for every key, the word itself says nothing and only the words
around it can decide. Korean-side lines come from KcBERT lines no pool or eval set reads (from line 5M, as
amb_en_train), cleaned as the v3 pools are, with no ambiguous English word in them, minus eval-set and x4 lines.

    python -u experiments/bal_train.py   -> data/sent/kmix_bal.txt (x4 lines, then the Korean-side lines)"""
import itertools
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
from amb_en_eval import amb  # noqa: E402
from amb_en_train import FROM, TO, target  # noqa: E402

import nunchi.sentence_data as sd  # noqa: E402
from nunchi.dubeolsik import to_keys  # noqa: E402

X4 = sd.OUT / "kmix_amb_x4.txt"
OUT = sd.OUT / "kmix_bal.txt"
HANG = re.compile(r"^[가-힣ㄱ-ㅣ]+$")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def en_keys(toks):
    return {t for t in toks if t.isascii() and t.isalpha() and t.islower() and amb(t)}


def main():
    x4 = [line.rstrip("\n") for line in X4.open(encoding="utf-8") if line.strip()]
    need = Counter()
    for line in x4:
        need.update(en_keys(line.split()))
    want = sum(need.values())
    say(f"{len(x4):,} x4 lines, {want:,} English points over {len(need)} keys: "
        + ", ".join(f"{k} {n}" for k, n in need.most_common(15)))
    used = set(x4)
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "amb_en_set.jsonl", "num_en_set.jsonl", "jamo1_set.jsonl",
              "amb_ko_set.jsonl", "collision_set.jsonl", "fused_set.jsonl", "typo_set.jsonl", "word1_set.jsonl"):
        p = S.parent / "data" / "eval" / f
        if p.exists():
            used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in p.open(encoding="utf-8"))
    for name in ("nsmc_v3", "kcbert_v3"):
        used.update(line.rstrip("\n") for line in (sd.OUT / f"{name}.txt").open(encoding="utf-8"))
    kk = sd.ko_keys()
    sd.SENT_KCBERT = TO
    out, got, n = [], Counter(), 0
    for n, text in enumerate(itertools.islice(sd.korean_lines("kcbert"), FROM, None), 1):
        if n % 1_000_000 == 0:
            say(f"  line {FROM + n:,}: {sum(got.values()):,}/{want:,} ({100 * sum(got.values()) / want:.0f}%)")
        toks = sd.clean(text, kk, 3)
        if not toks or en_keys(toks):
            continue
        ko = sum(bool(HANG.match(t)) for t in toks)
        en = sum(t.isascii() and t.isalpha() for t in toks)
        if ko <= en:
            continue
        # every matched key of the line must still be short of its count: lines holding several (ㅅ ㅐ ㄲ ㅣ) would
        # otherwise push the common jamo far past their English side
        hit = {to_keys(t) for t in toks if HANG.match(t)} & set(need)
        line = " ".join(toks)
        if not hit or any(got[k] >= need[k] for k in hit) or line in used:
            continue
        used.add(line)
        out.append(line)
        got.update(hit)
        if all(got[k] >= need[k] for k in need):
            break
    OUT.write_text("".join(f"{line}\n" for line in x4 + out), encoding="utf-8")
    short = {k: need[k] - got[k] for k in need if got[k] < need[k]}
    say(f"{OUT}: {len(x4):,} English-side + {len(out):,} Korean-side lines (KcBERT {FROM:,}-{FROM + n:,}); "
        f"Korean points {sum(got.values()):,} of {want:,}" + (f"; short: {short}" if short else ""))


if __name__ == "__main__":
    assert target  # the English side is amb_en_train's
    main()
