"""error rate per category, the same axes on both sides where they make sense:
  Korean words : form (syllables / jamo only: ㅋㅋ ㅠㅠ), length (1 / 2 / 3+ syllables or jamo), keys read in
                 English (not a word / a rare word / a common word)
  English words: case (lower / Capitalised / ALL CAPS), length (1-2 / 3+ letters), keys read in Korean
                 (not a word / a rare word / a common word)
  number words (3개, gpt4, 5km) and Korean+English words (채널A), from the digits / Hangul+Latin set
Words: dev halves, numbers in place: real mixed sentences + ambiguous words in natural context (+ the digits
set for the last two rows). Decided at Space, default bias. Cells: error rate % (change vs the mean of S0 and
S0-s1 in %p).
    python -u cat_table.py NAME:file ..."""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from wordfreq import zipf_frequency

S = Path(__file__).parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
import natural_sets  # noqa: E402
from textcheck import expected, text_of  # noqa: E402

import nunchi.stream_tagger as st  # noqa: E402
from nunchi.demo import compose  # noqa: E402
from nunchi.sentence_data import OUT as SENT, ko_zipf  # noqa: E402
from nunchi.tagger import KO, device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the repository
LOG = S / "logs" / "cat_table_log.txt"
dev_ = device()
JAMO = set("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡㅃㅉㄸㄲㅆㅒㅖ")


def say(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def amb_level(z):
    return "not a word" if z < 3 else "a rare word" if z < 4.5 else "a common word"


def cats(k, y):
    """-> the categories a word falls in (one per axis)"""
    if any(c.isdigit() for c in k):
        return ["number word"]
    if len(set(y)) > 1:
        return ["Korean+English word"]
    if y[0] == KO:
        text = compose(k)
        jamo = all(c in JAMO for c in text)
        n = len(text)
        return [f"KO form: {'jamo only' if jamo else 'syllables'}", f"KO length: {n if n < 3 else '3+'}",
                f"KO keys in English: {amb_level(zipf_frequency(k.lower(), 'en'))}"]
    case = "ALL CAPS" if k.isupper() and len(k) > 1 else "Capitalised" if k[:1].isupper() else "lower"
    return [f"EN case: {case}", f"EN length: {'1-2' if len(k) <= 2 else '3+'}",
            f"EN keys in Korean: {amb_level(ko_zipf(compose(k)))}"]


def dev_keyed(path, json_rows):
    out = []
    for line in path.open(encoding="utf-8"):
        if json_rows:
            r = json.loads(line)
            w, t = [(k, [int(c) for c in y]) for k, y in r["w"]], r["t"]
        else:
            w = [(k, [int(c) for c in y]) for k, y in (x.rsplit(":", 1) for x in line.rstrip("\n").split("\t"))]
            t = [i for i, (k, y) in enumerate(w) if (any(c.isdigit() for c in k) and not k.isdigit()) or len(set(y)) > 1]
        if int(hashlib.md5("\t".join(k for k, _ in w).encode()).hexdigest(), 16) % 2 == 0:
            out.append((w, t))
    return out


data = list(natural_sets.load("dev")) + dev_keyed(ROOT / "data" / "eval" / "amb_natural.jsonl", True) + \
    dev_keyed(SENT / "web_code_eval.txt", False)
where = [(si, i) for si, (_, idx) in enumerate(data) for i in idx]
wcats = [cats(*data[si][0][i]) for si, i in where]
n = defaultdict(int)
for cs in wcats:
    for c in cs:
        n[c] += 1
ORDER = ["KO form: syllables", "KO form: jamo only", "KO length: 1", "KO length: 2", "KO length: 3+",
         "KO keys in English: not a word", "KO keys in English: a rare word", "KO keys in English: a common word",
         "EN case: lower", "EN case: Capitalised", "EN case: ALL CAPS", "EN length: 1-2", "EN length: 3+",
         "EN keys in Korean: not a word", "EN keys in Korean: a rare word", "EN keys in Korean: a common word",
         "number word", "Korean+English word"]
keys = [[k for k, _ in s] for s, _ in data]
rates = {}
for spec in sys.argv[1:]:
    name, f = spec.split(":")
    m = st.StreamTagger(left=None, right=None, digits=True)
    m.load_state_dict(torch.load(ROOT / "models" / f, map_location="cpu"))
    m = m.to(dev_).eval()
    res = st.decide(m, [keys[si][:i + 1] for si, i in where], [i for _, i in where])
    bad = defaultdict(int)
    for (si, i), r, cs in zip(where, res, wcats):
        if text_of(data[si][0][i][0], r) != expected(*data[si][0][i]):
            for c in cs:
                bad[c] += 1
    rates[name] = {c: 100 * bad[c] / n[c] for c in ORDER if n[c]}
    print(f"  scored {name}", flush=True)
base = {c: (rates["S0"][c] + rates["S0-s1"][c]) / 2 for c in rates["S0"]}
LOG.write_text("", encoding="utf-8")
names = list(rates)
say("=== error rate % per category (change vs mean of S0 and S0-s1, %p); dev, numbers in place, at Space")
say(f"  {'category':<36}{'words':>7}" + "".join(f"{nm:>16}" for nm in names) + f"{'seed diff':>11}")
for c in ORDER:
    if not n[c]:
        continue
    cells = []
    for nm in names:
        v = rates[nm][c]
        cells.append(f"{v:>7.2f}" + (f" ({v - base[c]:+5.2f})" if nm not in ("S0", "S0-s1") else " " * 8))
    say(f"  {c:<36}{n[c]:>7}" + "".join(f"{x:>16}" for x in cells) + f"{abs(rates['S0'][c] - rates['S0-s1'][c]):>11.2f}")
