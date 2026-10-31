"""fastText baseline: compare variants on the test set.

Variants: plain keys, keys + decoded-shape token, each optionally quantized,
plus a size-capped autotune. Each is reported with and without the jamo-slang
whitelist rule (ㅋㅋㅋ, ㅠㅠ ... learned from train, forced to ko).

Weighted metrics weight each test word by its corpus frequency (normalised per
language, ko and en counted 50/50), which is closer to what people actually type.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
FT = ROOT / "data" / "ft"
MODELS = ROOT / "models"
THREADS = 6  # leave half the CPU for the desktop
BUCKET = 200_000
ACT_T = 0.99  # "act only if p >= this"
WHITELIST_MIN_COUNT = 5
REPEAT = re.compile(r"(.)\1{2,}")


def load(split):
    rows = []
    with (FT / f"{split}.tsv").open(encoding="utf-8") as f:
        for line in f:
            lab, keys, word, freq, shp = line.rstrip("\n").split("\t")
            rows.append({"lab": lab, "keys": keys, "word": word, "freq": float(freq), "shape": shp})
    return rows


def norm_repeats(keys):
    return REPEAT.sub(r"\1\1\1", keys)  # zzzzzz -> zzz


def build_whitelist(train_rows):
    counts = Counter()
    for r in train_rows:
        if r["lab"] == "ko" and r["shape"] == "jamo":
            counts[norm_repeats(r["keys"])] += r["freq"]
    return {k for k, c in counts.items() if c >= WHITELIST_MIN_COUNT}


def add_weights(rows):
    totals = Counter()
    for r in rows:
        totals[r["lab"]] += r["freq"]
    for r in rows:
        r["w"] = 0.5 * r["freq"] / totals[r["lab"]]


def predict(model, rows, use_shape, whitelist=None):
    inputs = [f"{r['keys']} _S_{r['shape']}" if use_shape else r["keys"] for r in rows]
    labels, probs = model.predict(inputs, k=1)
    preds = [(l[0].removeprefix("__label__"), float(p[0])) for l, p in zip(labels, probs)]
    if whitelist:
        preds = [("ko", 1.0) if norm_repeats(r["keys"]) in whitelist else p for r, p in zip(rows, preds)]
    return preds


def metrics(rows, preds):
    ok = [p[0] == r["lab"] for r, p in zip(rows, preds)]
    acted = [p[1] >= ACT_T for p in preds]
    short = [i for i, r in enumerate(rows) if len(r["keys"]) <= 4]
    jamo_ko = [i for i, r in enumerate(rows) if r["lab"] == "ko" and r["shape"] == "jamo"]
    w_acted = sum(r["w"] for r, a in zip(rows, acted) if a)
    return {
        "acc": sum(ok) / len(rows),
        "w_acc": sum(r["w"] for r, o in zip(rows, ok) if o),
        "short": sum(ok[i] for i in short) / max(len(short), 1),
        "jamo_ko": f"{sum(ok[i] for i in jamo_ko)}/{len(jamo_ko)}",
        "cov": sum(acted) / len(rows),
        "err": sum(1 for o, a in zip(ok, acted) if a and not o) / max(sum(acted), 1),
        "w_err": sum(r["w"] for r, o, a in zip(rows, ok, acted) if a and not o) / max(w_acted, 1e-12),
    }


def row(name, size_mb, m):
    print(f"{name:<26}{size_mb:>8.1f}  {m['acc']:.4f}  {m['w_acc']:.4f}  {m['short']:.4f}  "
          f"{m['jamo_ko']:>5}  {m['cov']:.4f}  {m['err']:.4f}  {m['w_err']:.4f}")


def main():
    autotune_secs = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    MODELS.mkdir(exist_ok=True)
    train_rows, test = load("train"), load("test")
    add_weights(test)
    wl = build_whitelist(train_rows)
    print(f"whitelist ({len(wl)}): {sorted(wl)}")
    print(f"\n{'model':<26}{'size MB':>8}  {'acc':<6}  {'w_acc':<6}  {'len<=4':<6}  "
          f"{'jamoK':>5}  {'cov@' + str(ACT_T):<6}  {'err':<6}  {'w_err':<6}")

    results = {}

    def evaluate(name, model, path, use_shape):
        size = path.stat().st_size / 1e6
        for suffix, w in (("", None), (" +rule", wl)):
            preds = predict(model, test, use_shape, w)
            results[name + suffix] = (model, use_shape, w, preds)
            row(name + suffix, size, metrics(test, preds))

    for name, use_shape in (("plain", False), ("shape", True)):
        train = str(FT / ("train_shape.txt" if use_shape else "train.txt"))
        model = fasttext.train_supervised(train, minn=2, maxn=5, dim=32, epoch=25, lr=0.5,
                                          bucket=BUCKET, thread=THREADS, verbose=0)
        path = MODELS / f"ft_{name}.bin"
        model.save_model(str(path))
        evaluate(name, model, path, use_shape)

        model.quantize(input=train, qnorm=True, retrain=True, cutoff=50_000, thread=THREADS, verbose=0)
        path = MODELS / f"ft_{name}.ftz"
        model.save_model(str(path))
        evaluate(f"{name} quantized", model, path, use_shape)

    tuned = fasttext.train_supervised(str(FT / "train.txt"), autotuneValidationFile=str(FT / "valid.txt"),
                                      autotuneDuration=autotune_secs, autotuneModelSize="2M",
                                      thread=THREADS, verbose=0)
    path = MODELS / "ft_tuned.ftz"
    tuned.save_model(str(path))
    a = tuned.f.getArgs()
    print(f"(autotune picked minn={a.minn} maxn={a.maxn} dim={a.dim} epoch={a.epoch} "
          f"lr={a.lr:.3f} bucket={a.bucket} wordNgrams={a.wordNgrams})")
    evaluate("autotune 2MB", tuned, path, False)

    _, _, _, preds = results["plain quantized +rule"]
    errors = sorted(((p[1], r, p[0]) for r, p in zip(test, preds) if p[0] != r["lab"]),
                    key=lambda e: (-e[0], -e[1]["freq"]))
    print("\nplain quantized +rule: most confident mistakes")
    for conf, r, pred in errors[:20]:
        print(f"  {r['keys']:<14} true={r['lab']}({r['word']}, freq={r['freq']:.3g}, {r['shape']})  "
              f"pred={pred} p={conf:.3f}")


if __name__ == "__main__":
    main()
