"""Does previous-word context help? Same fastText model, plus a 2-state HMM forward filter.

Sentences (word-level labels):
  ko    : NSMC ratings_test reviews (sentences not used to build the train word list)
  en    : Tatoeba English sentences
  mixed : ko sentences with 1-2 spans of English words inserted (오늘 deploy 했는데)

Per word, online (past words only, like the app at a word boundary):
  prior_t   = s * post_{t-1} + (1 - s) * (1 - post_{t-1})      (s = P(same language as previous word))
  logit_t   = logit(p_model) / T + logit(prior_t)
T (temperature, softens fastText's overconfidence) and s are tuned on valid by log loss.
s = 0.5 means no context, so "no context, tuned T" is a fair calibration-only baseline.
"""
import bz2
import math
import random
import re

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
EN_TOKEN = re.compile(r"[a-z]+")
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
            text = line.rstrip("\n").split("\t")[-1].lower().replace("'", "")
            toks = EN_TOKEN.findall(text)
            if MIN_W <= len(toks) <= MAX_W:
                out.append([("en", t) for t in toks])
    return out


def mixed(ko_sents, en_vocab, rnd):
    out = []
    for sent in ko_sents:
        sent = list(sent)
        for _ in range(rnd.choice([1, 1, 2])):
            span = [("en", rnd.choice(en_vocab)) for _ in range(rnd.choice([1, 1, 1, 2]))]
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
    sets = {}
    for i, split in enumerate(("valid", "test")):
        k = ko[i * N_SENT:(i + 1) * N_SENT]
        sets[split] = {
            "ko": k,
            "en": en[i * N_SENT:(i + 1) * N_SENT],
            "mixed": mixed(ko[(2 + i) * N_SENT:(3 + i) * N_SENT], en_vocab, rnd),
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


def decode(sent, scorer, T, s, mode="filter"):
    """P(ko) per word.

    filter : past words only (decide right at the word boundary)
    lag    : also the next word (decide one word late, i.e. retroactive correction)
    hybrid : act now if the filtered decision is confident, otherwise wait one word
    Returns (probs, retro) where retro marks words decided one word late.
    """
    filt, lrs = [], []
    post = 0.5
    for _, w in sent:
        z = scorer.z(to_keys(w)) / T
        prior = s * post + (1 - s) * (1 - post)
        post = sigmoid(z + logit(prior))
        filt.append(post)
        lrs.append(math.exp(max(min(z, 50), -50)))  # likelihood ratio ko:en
    if mode == "filter":
        return filt, [False] * len(sent)

    lag = []
    for t, p in enumerate(filt):
        if t + 1 == len(sent):  # last word: nothing to wait for
            lag.append(p)
            continue
        lr = lrs[t + 1]
        ko = p * (s * lr + (1 - s))
        en = (1 - p) * ((1 - s) * lr + s)
        lag.append(ko / (ko + en))
    if mode == "lag":
        return lag, [t + 1 < len(sent) for t in range(len(sent))]

    out, retro = [], []
    for f, l in zip(filt, lag):
        now = max(f, 1 - f) >= ACT_T
        out.append(f if now else l)
        retro.append(not now)
    return out, retro


def tokens(sents, scorer, T, s, amb, mode="filter"):
    """Flatten to per-word records."""
    recs = []
    for sent in sents:
        probs, retro = decode(sent, scorer, T, s, mode)
        for i, ((lab, w), p, r) in enumerate(zip(sent, probs, retro)):
            keys = to_keys(w)
            recs.append({"lab": lab, "p": p, "keys": keys, "word": w, "amb": keys in amb,
                         "switch": i > 0 and sent[i - 1][0] != lab, "retro": r})
    return recs


def log_loss(recs):
    return -sum(math.log(max(r["p"] if r["lab"] == "ko" else 1 - r["p"], 1e-9)) for r in recs) / len(recs)


def summary(recs):
    def stats(rs):
        if not rs:
            return "-"
        ok = sum((r["p"] >= 0.5) == (r["lab"] == "ko") for r in rs)
        acted = [r for r in rs if max(r["p"], 1 - r["p"]) >= ACT_T]
        wrong = sum((r["p"] >= 0.5) != (r["lab"] == "ko") for r in acted)
        return f"{ok / len(rs):.4f} cov={len(acted) / len(rs):.3f} err={wrong / max(len(acted), 1):.4f} (n={len(rs)})"
    return {"all": stats(recs),
            "ambiguous": stats([r for r in recs if r["amb"]]),
            "switch": stats([r for r in recs if r["switch"]])}


def main():
    sets = build()
    amb = ambiguous_keys()
    scorer = Scorer(fasttext.load_model(str(MODELS / "ft_shape_aug.ftz")))
    valid_all = [s for kind in sets["valid"].values() for s in kind]

    def tune(sames, mode):
        best = min(((log_loss(tokens(valid_all, scorer, T, s, amb, mode)), T, s) for T in TEMPS for s in sames))
        return best[1], best[2], mode

    lag_params = tune(SAMES, "lag")
    configs = {"no context, tuned T": tune([0.5], "filter"),
               "context: past only": tune(SAMES, "filter"),
               "context: +1 word lookahead": lag_params,
               "context: act now or wait 1": (lag_params[0], lag_params[1], "hybrid")}
    for name, (T, s, mode) in configs.items():
        print(f"\n=== {name}: T={T} s={s} ===")
        for kind, sents in sets["test"].items():
            recs = tokens(sents, scorer, T, s, amb, mode)
            m = summary(recs)
            retro = sum(r["retro"] for r in recs) / len(recs) if mode == "hybrid" else 0
            print(f"  {kind:<6} all      {m['all']}" + (f"  retro={retro:.3f}" if retro else ""))
            print(f"  {kind:<6} ambig    {m['ambiguous']}")
            if kind == "mixed":
                print(f"  {kind:<6} switch   {m['switch']}")

    T, s, mode = configs["context: act now or wait 1"]
    recs = tokens(sets["test"]["mixed"], scorer, T, s, amb, mode)
    base = tokens(sets["test"]["mixed"], scorer, *configs["no context, tuned T"][:2], amb)
    fixed = [(r["keys"], r["word"]) for r, b in zip(recs, base)
             if (r["p"] >= 0.5) == (r["lab"] == "ko") and (b["p"] >= 0.5) != (b["lab"] == "ko")]
    broke = [(r["keys"], r["word"]) for r, b in zip(recs, base)
             if (r["p"] >= 0.5) != (r["lab"] == "ko") and (b["p"] >= 0.5) == (b["lab"] == "ko")]
    print(f"\nmixed: context fixed {len(fixed)}, broke {len(broke)}")
    print("  fixed:", fixed[:15])
    print("  broke:", broke[:15])


if __name__ == "__main__":
    main()
