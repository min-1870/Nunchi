"""new training lines for ambiguous English: Korean-base lines holding a lower-case English word that also reads as
Korean (amb_en_eval.amb: b = ㅠ, s = ㄴ, go = 해). In the training text they are rare (0.09% of KcBERT lines, ~40%
kept by sentence_data.clean, as other English lines), so upweighting the ones there only repeats a few hundred
lines; these are new ones from KcBERT lines no pool or eval set reads (from line FROM: training reads the first 2M,
the eval sets 2M-4.4M), cleaned as the v3 pools are, minus eval-set lines.

Counts the same lines in the kmix pool (NSMC + KcBERT v3 lines with Latin) and takes (TIMES - 1) x that many,
so NUNCHI_KMIX_EXTRA=kmix_amb_x4.txt makes them TIMES as common in kmix with every line new.

    python -u experiments/amb_en_train.py [TIMES]   -> data/sent/kmix_amb_x{TIMES}.txt"""
import itertools
import json
import re
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (before pyarrow-using modules: the ROCm DLLs)

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))
from amb_en_eval import amb  # noqa: E402

import nunchi.sentence_data as sd  # noqa: E402

FROM, TO = 5_000_000, 40_000_000
HANG = re.compile(r"^[가-힣ㄱ-ㅣ]+$")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def target(toks):
    """a Korean-base line (more Hangul words than Latin) with a lower-case ambiguous English word"""
    ko = sum(bool(HANG.match(t)) for t in toks)
    en = sum(t.isascii() and t.isalpha() for t in toks)
    return ko > en and any(t.isascii() and t.isalpha() and t.islower() and amb(t) for t in toks)


def main():
    times = float(sys.argv[1]) if len(sys.argv) > 1 else 4
    have = 0
    for name in ("nsmc_v3", "kcbert_v3"):
        for line in (sd.OUT / f"{name}.txt").open(encoding="utf-8"):
            toks = line.split()
            if any(c.isascii() and c.isalpha() for c in line) and target(toks):
                have += 1
    want = int(round((times - 1) * have))
    say(f"kmix pool: {have:,} target lines; taking {want:,} new ones (x{times:g})")
    used = set()
    for f in ("source_sets.jsonl", "switch_sets.jsonl", "amb_en_set.jsonl", "num_en_set.jsonl"):
        p = S.parent / "data" / "eval" / f
        if p.exists():
            used.update(" ".join(k for k, _ in json.loads(line)["w"]) for line in p.open(encoding="utf-8"))
    for name in ("nsmc_v3", "kcbert_v3"):
        used.update(line.rstrip("\n") for line in (sd.OUT / f"{name}.txt").open(encoding="utf-8"))
    kk = sd.ko_keys()
    sd.SENT_KCBERT = TO
    out, n = [], 0
    for n, text in enumerate(itertools.islice(sd.korean_lines("kcbert"), FROM, None), 1):
        if n % 1_000_000 == 0:
            say(f"  line {FROM + n:,}: {len(out):,}/{want:,} ({100 * len(out) / max(want, 1):.0f}%)")
        toks = sd.clean(text, kk, 3)
        if not toks:
            continue
        line = " ".join(toks)
        if line in used or not target(toks):
            continue
        used.add(line)
        out.append(line)
        if len(out) >= want:
            break
    path = sd.OUT / f"kmix_amb_x{times:g}.txt"
    path.write_text("".join(f"{line}\n" for line in out), encoding="utf-8")
    say(f"{path}: {len(out):,} lines from KcBERT lines {FROM:,}-{FROM + n:,}")


if __name__ == "__main__":
    main()
