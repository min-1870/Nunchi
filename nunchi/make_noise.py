"""Harder test sets built from the (unseen) test words, plus a typo-augmented train set.

test_typo : every test word with one keyboard typo (adjacent-key sub, drop, insert, swap)
test_code : code-ish English tokens (getUser, user_id, kubectl-style concat, user2)
train_aug : train + one typo variant for 30% of train words (for the augmented model)
"""
import random

from nunchi.dubeolsik import shape
from nunchi.make_data import OUT
from nunchi.train_fasttext import load

SEED = 1
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {}
for r, row in enumerate(ROWS):
    for c, ch in enumerate(row):
        near = [row[c + d] for d in (-1, 1) if 0 <= c + d < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS):
                near += [ROWS[rr][cc] for cc in (c - 1, c, c + 1) if 0 <= cc < len(ROWS[rr])]
        NEIGHBORS[ch] = near


def typo(keys: str, rnd: random.Random) -> str:
    for _ in range(10):
        i = rnd.randrange(len(keys))
        ch = keys[i].lower()
        op = rnd.choice(["sub", "drop", "insert", "swap"])
        if op == "sub" and ch in NEIGHBORS:
            out = keys[:i] + rnd.choice(NEIGHBORS[ch]) + keys[i + 1:]
        elif op == "drop" and len(keys) >= 3:
            out = keys[:i] + keys[i + 1:]
        elif op == "insert" and ch in NEIGHBORS:
            out = keys[:i] + rnd.choice(NEIGHBORS[ch]) + keys[i:]
        elif op == "swap" and i + 1 < len(keys):
            out = keys[:i] + keys[i + 1] + keys[i] + keys[i + 2:]
        else:
            continue
        if out != keys and len(out) >= 2:
            return out
    return keys


def code_token(words: list[str], rnd: random.Random) -> str:
    a, b = rnd.sample(words, 2)
    style = rnd.choice(["camel", "snake", "concat", "digit", "pascal"])
    if style == "camel":
        return a + b.capitalize()
    if style == "pascal":
        return a.capitalize() + b.capitalize()
    if style == "snake":
        return f"{a}_{b}"
    if style == "concat":
        return a[: rnd.randint(2, 4)] + b[: rnd.randint(2, 4)]
    return a + str(rnd.randint(1, 9))


def write(name, rows):
    with (OUT / f"{name}.tsv").open("w", encoding="utf-8") as f:
        f.writelines(f"{r['lab']}\t{r['keys']}\t{r['word']}\t{r['freq']}\t{shape(r['keys'])}\n" for r in rows)
    print(name, len(rows))


def main():
    rnd = random.Random(SEED)
    test = load("test")

    write("test_typo", [{**r, "keys": typo(r["keys"], rnd)} for r in test])

    en_words = [r["word"] for r in test if r["lab"] == "en" and 3 <= len(r["word"]) <= 8]
    codes = {code_token(en_words, rnd) for _ in range(3000)}
    write("test_code", [{"lab": "en", "keys": c, "word": c, "freq": 1.0} for c in sorted(codes)])

    train = load("train")
    aug = train + [{**r, "keys": typo(r["keys"], rnd)} for r in train if rnd.random() < 0.3]
    rnd.shuffle(aug)
    with (OUT / "train_aug_shape.txt").open("w", encoding="utf-8") as f:
        f.writelines(f"__label__{r['lab']} {r['keys']} _S_{shape(r['keys'])}\n" for r in aug)
    print("train_aug", len(aug))


if __name__ == "__main__":
    main()
