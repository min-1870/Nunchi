"""Rule-only baselines on the same test set, for comparison with the fastText table.

shape rule      : decodes to complete syllables only -> ko, else en
shape + en dict : in English dictionary -> en, else shape rule
                  (dictionary = wordfreq top 200k, which includes the test en words,
                   so this is generous to the rule)
"""
import re

import wordfreq

from nunchi.train_fasttext import add_weights, load, metrics, row

EN_WORD = re.compile(r"^[a-z]+$")


def main():
    test = load("test")
    add_weights(test)
    en_dict = {w for w in wordfreq.top_n_list("en", 200_000) if EN_WORD.match(w)}

    shape_rule = [("ko" if r["shape"] == "full" else "en", 1.0) for r in test]
    dict_rule = [("en", 1.0) if r["keys"] in en_dict else p for r, p in zip(test, shape_rule)]

    print(f"{'model':<26}{'size MB':>8}  {'acc':<6}  {'w_acc':<6}  {'len<=4':<6}  "
          f"{'jamoK':>5}  {'cov@0.99':<6}  {'err':<6}  {'w_err':<6}")
    row("rule: shape", 0, metrics(test, shape_rule))
    row("rule: shape + en dict", 0, metrics(test, dict_rule))


if __name__ == "__main__":
    main()
