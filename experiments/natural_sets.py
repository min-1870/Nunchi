"""the real mixed eval sentences (NSMC test + mC4 validation web set) with their numbers in place.
The sets in use were built with every token that is not plain Hangul or Latin removed (10점, 2019, iPhone 15,
3D), the way the v1 training pools were; TR2/TR3 train on text with numbers, so they are scored on what is
really typed. Same sentences, same dev/test halves and leak filter (decided on the original sentence); the
words scored are the original sentence's words (numbers and the other kept tokens are context, not scored).

    build()  -> writes real_natural.jsonl here: {"split", "src", "w": [[keys, labels]], "score": [idx]}
    load(split) -> [(sentence [(keys, labels)], scored idx)]
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parent  # the repository
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parent))  # nunchi
OUT = ROOT / "data" / "eval" / "real_natural.jsonl"


def _half(words):
    return "dev" if int(hashlib.md5(" ".join(words).encode()).hexdigest(), 16) % 2 == 0 else "test"


def build():
    import evalsets as ev
    from nunchi.context import HANGUL_TOKEN, LATIN, MASK, RAW, SPLIT
    from nunchi.sentence_data import TOKEN_V2, clean, ko_keys
    from nunchi.sentence_tagger import to_sentence
    from nunchi.web_mixed import RAW as MC4_VALID, SENT_END, sentence, web_mixed

    rows = []
    # NSMC test: as context.real_mixed(), plus tokens with digits kept as context
    with (RAW / "nsmc_test.txt").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            orig, nat, score = [], [], []
            for t in filter(None, SPLIT.split(parts[1])):
                if HANGUL_TOKEN.match(t):
                    orig.append(("ko", t)); score.append(len(nat)); nat.append(t)
                elif LATIN.match(t) and not MASK.match(t):
                    orig.append(("en", t)); score.append(len(nat)); nat.append(t)
                elif TOKEN_V2.match(t) and re.search(r"[0-9]", t):
                    nat.append(t)
            if len(orig) >= 3 and {lab for lab, _ in orig} == {"ko", "en"} and not ev.in_training(orig):
                rows.append({"split": _half([w for _, w in orig]), "src": "nsmc",
                             "w": [[k, "".join(map(str, y))] for k, y in to_sentence(" ".join(nat))], "score": score})
    # web: the eval sentences; those that had numbers removed are rebuilt from mC4 validation with them
    kk = ko_keys()
    evalset = {" ".join(w for _, w in s): s for s in web_mixed()}
    nat_of = {}
    with gzip.open(MC4_VALID, "rt", encoding="utf-8") as f:
        for line in f:
            for text in SENT_END.split(json.loads(line)["text"]):
                s = sentence(text, kk)
                if not s:
                    continue
                key = " ".join(w for _, w in s)
                if key in evalset and key not in nat_of:
                    nat = clean(text, kk, 3)
                    if nat and [t for t in nat if not re.search(r"[0-9]", t)] == [w for _, w in s]:
                        nat_of[key] = nat
    for key, s in evalset.items():
        if ev.in_training(s):
            continue
        nat = nat_of.get(key, [w for _, w in s])
        score = [j for j, t in enumerate(nat) if not re.search(r"[0-9]", t)]
        rows.append({"split": _half([w for _, w in s]), "src": "web",
                     "w": [[k, "".join(map(str, y))] for k, y in to_sentence(" ".join(nat))], "score": score})
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"real natural: {len(rows)} sentences, {sum(len(r['w']) > len(r['score']) for r in rows)} with numbers in "
          f"place -> {OUT}", flush=True)


def load(split):
    out = []
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            out.append(([(k, [int(c) for c in y]) for k, y in r["w"]], r["score"]))
    return out


if __name__ == "__main__":
    build()
