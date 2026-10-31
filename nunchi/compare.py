"""Head-to-head: tuned rules vs fastText models on clean / typo / code test sets.

tuned rule (built from train data + wordfreq only, never from test):
  1. key (ㅋㅋㅋ-style repeats collapsed) is a known Korean train word -> ko
  2. in English dictionary (wordfreq top 30k, minus keys of Korean train words) -> en
  3. decodes to complete syllables only -> ko, else en
"""
import re

import fasttext
import wordfreq

from nunchi.train_fasttext import (FT, MODELS, THREADS, BUCKET, ACT_T, add_weights, load,
                                   metrics, norm_repeats, predict)

EN_WORD = re.compile(r"^[a-z]+$")
EN_DICT_SIZE = 30_000
SETS = ["test", "test_typo", "test_code"]


class Rule:
    def __init__(self, train_rows, use_ko_table=True, en_dict_size=EN_DICT_SIZE):
        self.ko = {norm_repeats(r["keys"]) for r in train_rows if r["lab"] == "ko"} if use_ko_table else set()
        words = [w for w in wordfreq.top_n_list("en", en_dict_size * 2) if EN_WORD.match(w)][:en_dict_size]
        self.en = set(words) - self.ko

    def lookup(self, keys):
        """Steps 1-2 only: the table of known words. None when the word is not in it."""
        if norm_repeats(keys) in self.ko:
            return "ko"
        # exact case: Shift is meaningful in Hangul mode (Tl = 씨, not "tl")
        if keys in self.en:
            return "en"
        return None

    def predict(self, rows):
        return [((self.lookup(r["keys"]) or ("ko" if r["shape"] == "full" else "en")), 1.0) for r in rows]

    def hybrid(self, rows, model_preds):
        """Table hit wins, otherwise the model decides."""
        return [(lab, 1.0) if (lab := self.lookup(r["keys"])) else p for r, p in zip(rows, model_preds)]


def aug_model():
    path = MODELS / "ft_shape_aug.ftz"
    if path.exists():
        return fasttext.load_model(str(path))
    train = str(FT / "train_aug_shape.txt")
    m = fasttext.train_supervised(train, minn=2, maxn=5, dim=32, epoch=25, lr=0.5,
                                  bucket=BUCKET, thread=THREADS, verbose=0)
    m.quantize(input=train, qnorm=True, retrain=True, cutoff=50_000, thread=THREADS, verbose=0)
    m.save_model(str(path))
    return m


def main():
    train = load("train")
    data = {s: load(s) for s in SETS}
    for rows in data.values():
        add_weights(rows)

    rule = Rule(train)
    contenders = {
        "rule: shape only": lambda rows: Rule(train, use_ko_table=False, en_dict_size=0).predict(rows),
        "rule: tuned": rule.predict,
    }
    for name, file, use_shape in (("ft plain .ftz", "ft_plain.ftz", False), ("ft shape .ftz", "ft_shape.ftz", True)):
        m = fasttext.load_model(str(MODELS / file))
        contenders[name] = lambda rows, m=m, s=use_shape: predict(m, rows, s)
    m_aug = aug_model()
    contenders["ft shape+typo-aug .ftz"] = lambda rows: predict(m_aug, rows, True)
    m_plain = fasttext.load_model(str(MODELS / "ft_plain.ftz"))
    contenders["hybrid: table + plain"] = lambda rows: rule.hybrid(rows, predict(m_plain, rows, False))
    contenders["hybrid: table + aug"] = lambda rows: rule.hybrid(rows, predict(m_aug, rows, True))

    for s in SETS:
        rows = data[s]
        weighted = s != "test_code"
        print(f"\n=== {s} (n={len(rows)}) ===")
        print(f"{'':<24}{'acc':>8}{'w_acc':>8}{'cov@' + str(ACT_T):>10}{'err':>8}{'w_err':>8}")
        for name, fn in contenders.items():
            m = metrics(rows, fn(rows))
            w_acc = f"{m['w_acc']:.4f}" if weighted else "-"
            w_err = f"{m['w_err']:.4f}" if weighted else "-"
            print(f"{name:<24}{m['acc']:>8.4f}{w_acc:>8}{m['cov']:>10.4f}{m['err']:>8.4f}{w_err:>8}")

    for s in ("test", "test_typo"):
        rows = data[s]
        hits = [rule.lookup(r["keys"]) for r in rows]
        print(f"\n{s}: table hit rate={sum(h is not None for h in hits) / len(rows):.3f} "
              f"(ko {sum(h == 'ko' for h in hits)}, en {sum(h == 'en' for h in hits)}), "
              f"table wrong={sum(h is not None and h != r['lab'] for h, r in zip(hits, rows))}")
        preds = contenders["hybrid: table + aug"](rows)
        bad = sorted(((r, p) for r, p in zip(rows, preds) if p[0] != r["lab"] and p[1] >= ACT_T),
                     key=lambda x: -x[0]["w"])
        print(f"  hybrid+aug confident mistakes ({len(bad)}):",
              [(r["keys"], r["word"], "table" if rule.lookup(r["keys"]) else "model") for r, _ in bad[:12]])


if __name__ == "__main__":
    main()
