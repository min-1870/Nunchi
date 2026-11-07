"""does the training cleaner drop Korean lines that hold short English that reads as Korean? (2026-11-03 check)"""
import itertools, re, sys, time
from collections import Counter
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
import torch  # noqa: F401
import nunchi.sentence_data as sd
from amb_en_eval import amb
kk = sd.ko_keys()
HANG = re.compile(r"[가-힣]")
N = 600_000
kept, seen, why = Counter(), Counter(), Counter()
t0 = time.time()
for n, text in enumerate(itertools.islice(sd.korean_lines("kcbert"), N), 1):
    toks = [t for t in filter(None, sd.SPLIT.split(text))]
    if sum(bool(HANG.search(t)) for t in toks) < 2:
        continue
    lat = [t for t in toks if sd.LATIN.match(t)]
    if not lat:
        continue
    a_low = [t for t in lat if amb(t) and t.islower()]
    a_up = [t for t in lat if amb(t) and not t.islower()]
    kind = "ambiguous, lower-case" if a_low else "ambiguous, upper-case" if a_up else "other English"
    seen[kind] += 1
    if sd.clean(text, kk, 3) is not None:
        kept[kind] += 1
    elif a_low:
        for t in a_low:
            if sd.wrong_mode(t, kk):
                why[t] += 1
    if n % 100_000 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {n:,}/{N:,} lines ({100*n/N:.0f}%)", flush=True)
print("Korean lines with Latin words in the first", f"{N:,}", "KcBERT lines (training text):")
for k in seen:
    print(f"  {k:<24} {seen[k]:>7,} lines, kept by clean {kept[k]:>7,} ({100*kept[k]/seen[k]:.0f}%)")
print("  ambiguous lower-case words that the wrong-mode rule drops a line for:",
      ", ".join(f"{w} {c}" for w, c in why.most_common(20)))
