"""Does previous-word context help? Same fastText model, plus an HMM over word languages.

Sentences (word-level labels):
  ko    : NSMC ratings_test reviews (sentences not used to build the train word list)
  en    : Tatoeba English sentences, original case (Shift changes the keys)
  mixed : ko sentences with 1-2 spans of English words inserted (오늘 deploy 했는데)
  en_ko : en sentences with 1-2 spans of Korean words inserted, drawn by NSMC frequency
          (I had 삼겹살 last night)
  real  : NSMC test reviews that already mix English (SF는 재밌다 -> tokens split on spaces/punct).
          Tokens with English glued to Hangul (SF는) and masked words (OOO) are dropped.

Emission per word is fastText's likelihood ratio ko:en, softened by temperature T.
Context models (transition between consecutive words):
  2-state   : P(same language as previous word) = s
  4-state   : (base language, current language); detours away from the base tend to return
(A recent-N-words ratio prior was also tried; it lost to both on every set and was removed.)
T and the model params are tuned on valid by log loss. s = 0.5 means no context, so
"no context, tuned T" is a fair calibration-only baseline.
"""
import bz2
import math
import random
import re
import sys
from collections import Counter

import fasttext

from nunchi.dubeolsik import shape, to_keys
from nunchi.make_data import english_words, korean_words
from nunchi.train_fasttext import ACT_T, MODELS, ROOT

RAW = ROOT / "data" / "raw"
N_SENT = 2000  # per kind per split
MIN_W, MAX_W = 3, 15
SEED = 2
HANGUL_TOKEN = re.compile(r"^[가-힣ㄱ-ㅣ]+$")
PUNCT = re.compile(r"[.,!?~…\"'()]+")
EN_TOKEN = re.compile(r"[A-Za-z]+")
TEMPS = [1, 2, 3, 5, 8, 12, 20, 30]
SAMES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


def ko_sentences():
    out = []
    with (RAW / "nsmc_test.txt").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            toks = [PUNCT.sub("", t) for t in parts[1].split()]
            toks = [t for t in toks if t]
            if MIN_W <= len(toks) <= MAX_W and all(HANGUL_TOKEN.match(t) for t in toks):
                out.append([("ko", t) for t in toks])
    return out


def en_sentences():
    out = []
    with bz2.open(RAW / "eng_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").split("\t")[-1].replace("'", "")
            toks = EN_TOKEN.findall(text)
            if MIN_W <= len(toks) <= MAX_W:
                out.append([("en", t) for t in toks])
    return out


SPLIT = re.compile(r"[\s.,!?~…\"'()\[\]/:;*^+=-]+")
LATIN = re.compile(r"^[A-Za-z]+$")
MASK = re.compile(r"^O+$")


def real_mixed():
    """Returns (sentences, stats). Only test reviews, so no overlap with the train word list."""
    out, dropped, kept = [], 0, 0
    with (RAW / "nsmc_test.txt").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sent = []
            for t in filter(None, SPLIT.split(parts[1])):
                if HANGUL_TOKEN.match(t):
                    sent.append(("ko", t))
                elif LATIN.match(t) and not MASK.match(t):
                    sent.append(("en", t))
                else:
                    dropped += 1
                    continue
                kept += 1
            labs = {lab for lab, _ in sent}
            if len(sent) >= 3 and labs == {"ko", "en"}:
                out.append(sent)
    return out, dropped / (dropped + kept)


def mixed(base_sents, pick, lab, rnd):
    """Insert 1-2 spans of 1-2 words of the other language; pick(rnd) draws one word."""
    out = []
    for sent in base_sents:
        sent = list(sent)
        for _ in range(rnd.choice([1, 1, 2])):
            span = [(lab, pick(rnd)) for _ in range(rnd.choice([1, 1, 1, 2]))]
            i = rnd.randrange(len(sent) + 1)
            sent[i:i] = span
        out.append(sent)
    return out


def build():
    rnd = random.Random(SEED)
    ko, en = ko_sentences(), en_sentences()
    rnd.shuffle(ko)
    rnd.shuffle(en)
    en_vocab = sorted({w for s in en[: 20 * N_SENT] for _, w in s if len(w) >= 3})
    ko_counts = Counter(w for s in ko for _, w in s)
    ko_words, ko_weights = zip(*ko_counts.items())
    pick_en = lambda r: r.choice(en_vocab)
    pick_ko = lambda r: r.choices(ko_words, ko_weights)[0]
    sets = {}
    for i, split in enumerate(("valid", "test")):
        sets[split] = {
            "ko": ko[i * N_SENT:(i + 1) * N_SENT],
            "en": en[i * N_SENT:(i + 1) * N_SENT],
            "mixed": mixed(ko[(2 + i) * N_SENT:(3 + i) * N_SENT], pick_en, "en", rnd),
            "en_ko": mixed(en[(2 + i) * N_SENT:(3 + i) * N_SENT], pick_ko, "ko", rnd),
        }
    return sets


def ambiguous_keys():
    ko = {to_keys(w) for w, _ in korean_words()}
    return ko & {w for w, _ in english_words()}


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


class Scorer:
    def __init__(self, model):
        self.model, self.cache = model, {}

    def z(self, keys):
        """logit of P(ko) from fastText alone."""
        if keys not in self.cache:
            labels, probs = self.model.predict(f"{keys} _S_{shape(keys)}", k=2)
            p_ko = dict(zip(labels, probs)).get("__label__ko", 0.0)
            self.cache[keys] = logit(float(p_ko))
        return self.cache[keys]


KO, EN = 0, 1


def two_state(s):
    """Current HMM: P(same language as previous word) = s."""
    A = [[s, 1 - s], [1 - s, s]]
    return [0.5, 0.5], [True, False], lambda hist: A


def four_state(q, a, r):
    """States (base, current). The base language stays with q; at the base, the current word
    leaves it with 1 - a; away from the base, it comes back with r (오늘 fork 했는데: fork is a detour)."""
    states = [(B, C) for B in (KO, EN) for C in (KO, EN)]

    def p(i, j):
        (B, C), (B2, C2) = states[i], states[j]
        pb = q if B2 == B else 1 - q
        if C == B2:
            pc = a if C2 == B2 else 1 - a
        else:
            pc = r if C2 == B2 else 1 - r
        return pb * pc
    A = [[p(i, j) for j in range(4)] for i in range(4)]
    init = [0.5 * (a if C == B else 1 - a) for B, C in states]
    return init, [C == KO for _, C in states], lambda hist: A


def decode(zs, model, T, mode="filter"):
    """P(ko) per word from fastText logits zs.

    filter : past words only (decide right at the word boundary)
    lag    : also the next word (decide one word late, i.e. retroactive correction)
    hybrid : act now if the filtered decision is confident, otherwise wait one word
    Returns (probs, retro) where retro marks words decided one word late.
    """
    init, is_ko, A_fn = model
    n = len(init)
    alphas, emits, trans, filt = [], [], [], []
    for z in zs:
        lr = math.exp(max(min(z / T, 50), -50))  # likelihood ratio ko:en
        e = [lr if k else 1.0 for k in is_ko]
        if alphas:
            A = A_fn(filt)
            trans.append(A)
            prior = [sum(alphas[-1][i] * A[i][j] for i in range(n)) for j in range(n)]
        else:
            prior = init
        un = [prior[j] * e[j] for j in range(n)]
        tot = sum(un)
        alphas.append([u / tot for u in un])
        emits.append(e)
        filt.append(sum(a for a, k in zip(alphas[-1], is_ko) if k))
    if mode == "filter":
        return filt, [False] * len(zs)

    lag = []
    for t, alpha in enumerate(alphas):
        if t + 1 == len(zs):  # last word: nothing to wait for
            lag.append(filt[t])
            continue
        A, e = trans[t], emits[t + 1]
        w = [alpha[i] * sum(A[i][j] * e[j] for j in range(n)) for i in range(n)]
        lag.append(sum(x for x, k in zip(w, is_ko) if k) / sum(w))
    if mode == "lag":
        return lag, [t + 1 < len(zs) for t in range(len(zs))]

    out, retro = [], []
    for f, l in zip(filt, lag):
        now = max(f, 1 - f) >= ACT_T
        out.append(f if now else l)
        retro.append(not now)
    return out, retro


def prepare(sents, scorer, amb):
    """Per sentence: fastText logits plus per-word info, computed once."""
    out = []
    if hasattr(scorer, "prefill"):  # batch the tagger instead of one word at a time
        scorer.prefill([to_keys(w) for sent in sents for _, w in sent])
    for sent in sents:
        keys = [to_keys(w) for _, w in sent]
        info = [{"lab": lab, "keys": k, "word": w, "amb": k in amb,
                 "switch": i > 0 and sent[i - 1][0] != lab}
                for i, ((lab, w), k) in enumerate(zip(sent, keys))]
        out.append(([scorer.z(k) for k in keys], info))
    return out


def tokens(prepped, model, T, mode="filter"):
    """Flatten to per-word records."""
    recs = []
    for zs, info in prepped:
        probs, retro = decode(zs, model, T, mode)
        recs += [{**d, "p": p, "retro": r} for d, p, r in zip(info, probs, retro)]
    return recs


def log_loss(recs):
    return -sum(math.log(max(r["p"] if r["lab"] == "ko" else 1 - r["p"], 1e-9)) for r in recs) / len(recs)


def stats(rs):
    """acc, coverage (share decided confidently), error among the confident ones."""
    if not rs:
        return "-"
    ok = sum((r["p"] >= 0.5) == (r["lab"] == "ko") for r in rs)
    acted = [r for r in rs if max(r["p"], 1 - r["p"]) >= ACT_T]
    wrong = sum((r["p"] >= 0.5) != (r["lab"] == "ko") for r in acted)
    return f"acc={ok / len(rs):.4f} cov={len(acted) / len(rs):.3f} err={wrong / max(len(acted), 1):.4f} (n={len(rs)})"


def wrong(r):
    return (r["p"] >= 0.5) != (r["lab"] == "ko")


def main():
    model_file = sys.argv[1] if len(sys.argv) > 1 else "ft_shape_aug.ftz"
    print(f"model: {model_file}")
    sets = build()
    amb = ambiguous_keys()
    if model_file == "tagger":
        from nunchi.tagger import TaggerScorer
        scorer = TaggerScorer()
    else:
        scorer = Scorer(fasttext.load_model(str(MODELS / model_file)))
    valid = prepare([s for kind in sets["valid"].values() for s in kind], scorer, amb)
    test = {kind: prepare(sents, scorer, amb) for kind, sents in sets["test"].items()}
    real, drop_rate = real_mixed()
    print(f"real: {len(real)} sentences, dropped token rate {drop_rate:.3f} (in all test reviews)")
    test["real"] = prepare(real, scorer, amb)

    def tune(make, grid, temps=TEMPS):
        """Pick T and model params by log loss on valid, one-word lookahead (what hybrid uses)."""
        best = min((log_loss(tokens(valid, make(*g), T, "lag")), T, g) for T in temps for g in grid)
        print(f"  tuned {make.__name__}: T={best[1]} params={best[2]} loss={best[0]:.4f}", flush=True)
        return best[1], make(*best[2])

    T0 = min((log_loss(tokens(valid, two_state(0.5), T)), T) for T in TEMPS)[1]
    configs = {
        "no context, tuned T": (T0, two_state(0.5), "filter"),
        "2-state HMM": (*tune(two_state, [(s,) for s in SAMES]), "hybrid"),
        "4-state (base lang)": (*tune(four_state, [(q, a, r) for q in (0.9, 0.97, 0.99) for a in (0.8, 0.9, 0.95)
                                                   for r in (0.3, 0.5, 0.7, 0.9)], [1, 2, 3, 5]), "hybrid"),
    }

    results = {}
    for name, (T, model, mode) in configs.items():
        print(f"\n=== {name} ===")
        for kind, prepped in test.items():
            recs = tokens(prepped, model, T, mode)
            results[name, kind] = recs
            retro = sum(r["retro"] for r in recs) / len(recs)
            print(f"  {kind:<6} all       {stats(recs)}" + (f"  retro={retro:.3f}" if mode == "hybrid" else ""))
            print(f"  {kind:<6} ambig     {stats([r for r in recs if r['amb']])}")
            if kind not in ("ko", "en"):
                print(f"  {kind:<6} sw->en    {stats([r for r in recs if r['switch'] and r['lab'] == 'en'])}")
                print(f"  {kind:<6} sw->ko    {stats([r for r in recs if r['switch'] and r['lab'] == 'ko'])}")

    for name in ("2-state HMM", "4-state (base lang)"):
        for kind in ("real", "en_ko"):
            new, old = results[name, kind], results["no context, tuned T", kind]
            fixed = [(r["keys"], r["word"]) for r, o in zip(new, old) if not wrong(r) and wrong(o)]
            broke = [(r["keys"], r["word"]) for r, o in zip(new, old) if wrong(r) and not wrong(o)]
            print(f"\n{name} vs no context on {kind}: fixed {len(fixed)}, broke {len(broke)}")
            print("  fixed:", fixed[:15])
            print("  broke:", broke[:15])
    for kind in ("real", "en_ko"):
        left = [(r["keys"], r["word"], r["lab"]) for r in results["no context, tuned T", kind]
                if wrong(r) and max(r["p"], 1 - r["p"]) >= ACT_T]
        print(f"\nno context, {kind}, confident mistakes ({len(left)}):", left[:40])


if __name__ == "__main__":
    main()
