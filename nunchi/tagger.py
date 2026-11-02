"""Key-level language tagger: one label (en / ko) per key of a word, so English + Korean in one word
(githubdp -> github에, PDFvkdlf -> PDF파일) is just a path whose label changes once.

Model: key embedding -> 2-layer BiLSTM -> linear-chain CRF over {en, ko}. A word's confidence is the
probability of its best path; whole-word P(ko) is the probability of the all-ko path.

Data (labels come from the script the text was written in, no annotation):
  pure   : the make_data word rows (same train/valid/test split as fastText)
  mixed  : real Latin+Hangul tokens counted by count_corpus (SF는, TV에서), split by the English part
  synth  : train English word + a Korean ending drawn from the endings seen after English in real text
Typos (make_noise-style: neighbour sub / drop / insert / swap) are added with labels kept aligned.
Training samples words by sqrt(frequency), so common words weigh more than their single row.

    .venv-rocm/Scripts/python -m nunchi.tagger train
    .venv-rocm/Scripts/python -m nunchi.tagger eval
"""
import hashlib
import math
import random
import re
import sys
import time
from collections import Counter

import torch
from torch import nn

from nunchi.dubeolsik import to_keys
from nunchi.make_noise import NEIGHBORS
from nunchi.train_fasttext import ACT_T, MODELS, ROOT, load

KO_SOURCES = ("nsmc", "kcbert", "tatoeba")
MIXED = re.compile(r"^([A-Za-z]+)([가-힣]+)$")
MASK = re.compile(r"^O+$")
MIN_MIXED = 3
EN, KO = 0, 1
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOCAB = {c: i + 2 for i, c in enumerate(CHARS)}  # 0 pad, 1 other
MAX_LEN = 32
TYPO_P = 0.3
SYNTH_PER_EPOCH = 60_000
MIXED_SHARE = 0.1
PATH = MODELS / "tagger.pt"


# ---------------------------------------------------------------- data

def split_of(text):
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "valid" if h == 1 else "train"


def pure_rows(split):
    """(keys, labels, weight) from the make_data split."""
    rows = load(split)
    totals = Counter()
    for r in rows:
        totals[r["lab"]] += math.sqrt(r["freq"])
    # half the mass uniform over word types (rare words still get learned), half by sqrt(frequency)
    n = Counter(r["lab"] for r in rows)
    return [(r["keys"], [KO if r["lab"] == "ko" else EN] * len(r["keys"]),
             0.5 / n[r["lab"]] + 0.5 * math.sqrt(r["freq"]) / totals[r["lab"]]) for r in rows]


def mixed_rows():
    """Real English+Korean tokens -> {split: [(keys, labels, count)]}, and train endings with counts."""
    from nunchi.count_corpus import load as load_counts
    from nunchi.sentence_data import wrong_mode

    counts = Counter()
    for s in KO_SOURCES:
        counts.update(load_counts(s, "mixed"))
    # and the mC4 pool, if built: web text has English + nouns too (K댄스, e북), not only particles
    web = ROOT / "data" / "sent" / "web.txt"
    if web.exists():
        with web.open(encoding="utf-8") as f:
            counts.update(t for line in f for t in line.split() if MIXED.match(t))
    # "English" that was most likely Korean typed in the wrong mode (tk는 = 사는); B급, SK브로드밴드 stay
    ko_keys = {r["keys"].lower() for split in ("train", "valid", "test") for r in load(split) if r["lab"] == "ko"}
    out = {"train": [], "valid": [], "test": []}
    endings = Counter()
    for tok, c in counts.items():
        m = MIXED.match(tok)
        if c < MIN_MIXED or not m or MASK.match(m.group(1)) or wrong_mode(m.group(1), ko_keys):
            continue
        en, ko = m.group(1), to_keys(m.group(2))
        split = split_of(en.lower())
        out[split].append((en + ko, [EN] * len(en) + [KO] * len(ko), c))
        if split == "train":
            endings[m.group(2)] += c
    return out, endings


def synth(en_rows, endings, n, rnd):
    words, wweights = zip(*[(k, w) for k, _, w in en_rows])
    ends, eweights = zip(*endings.items())
    out = []
    for w, e in zip(rnd.choices(words, wweights, k=n), rnd.choices(ends, eweights, k=n)):
        ek = to_keys(e)
        out.append((w + ek, [EN] * len(w) + [KO] * len(ek)))
    return out


def typo(keys, labels, rnd):
    """One keyboard typo, labels kept aligned (an inserted key takes its neighbour's label)."""
    keys, labels = list(keys), list(labels)
    for _ in range(10):
        i = rnd.randrange(len(keys))
        ch = keys[i].lower()
        op = rnd.choice(["sub", "drop", "insert", "swap"])
        if op == "sub" and ch in NEIGHBORS:
            keys[i] = rnd.choice(NEIGHBORS[ch])
        elif op == "drop" and len(keys) >= 3:
            del keys[i], labels[i]
        elif op == "insert" and ch in NEIGHBORS:
            keys.insert(i, rnd.choice(NEIGHBORS[ch]))
            labels.insert(i, labels[i])
        elif op == "swap" and i + 1 < len(keys) and labels[i] == labels[i + 1]:
            keys[i], keys[i + 1] = keys[i + 1], keys[i]
        else:
            continue
        break
    return "".join(keys), labels


def encode(batch):
    """[(keys, labels)] -> ids (B, L), labels (B, L), mask (B, L)."""
    L = min(max(len(k) for k, _ in batch), MAX_LEN)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    labs = torch.zeros(len(batch), L, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.bool)
    for b, (k, y) in enumerate(batch):
        k, y = k[:L], y[:L]
        ids[b, :len(k)] = torch.tensor([VOCAB.get(c, 1) for c in k])
        labs[b, :len(y)] = torch.tensor(y)
        mask[b, :len(k)] = True
    return ids, labs, mask


# ---------------------------------------------------------------- model

class Tagger(nn.Module):
    def __init__(self, emb=64, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 2, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, 2)
        self.trans = nn.Parameter(torch.zeros(2, 2))  # trans[i, j]: label i -> label j
        self.start = nn.Parameter(torch.zeros(2))
        self.end = nn.Parameter(torch.zeros(2))

    def emissions(self, ids):
        h, _ = self.lstm(self.emb(ids))
        return self.out(h)

    def log_z(self, em, mask):
        alpha = self.start + em[:, 0]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t:t + 1], nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def score(self, em, labs, mask):
        s = self.start[labs[:, 0]] + em[:, 0].gather(1, labs[:, :1]).squeeze(1)
        for t in range(1, em.size(1)):
            step = self.trans[labs[:, t - 1], labs[:, t]] + em[:, t].gather(1, labs[:, t:t + 1]).squeeze(1)
            s = s + step * mask[:, t]
        last = labs.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def nll(self, ids, labs, mask):
        em = self.emissions(ids)
        return (self.log_z(em, mask) - self.score(em, labs, mask)).mean()

    @torch.no_grad()
    def decode(self, ids, mask):
        """Best path per word, its probability, and P(all ko)."""
        return self.decode_em(self.emissions(ids), mask)

    @torch.no_grad()
    def decode_em(self, em, mask):
        B, L, _ = em.shape
        score = self.start + em[:, 0]
        back = []
        for t in range(1, L):
            cand = score.unsqueeze(2) + self.trans
            best, idx = cand.max(1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t:t + 1], nxt, score)
            back.append(torch.where(mask[:, t:t + 1], idx, torch.arange(2, device=em.device).expand(B, 2)))
        score = score + self.end
        best, last = score.max(1)
        paths = [last]
        for idx in reversed(back):
            last = idx.gather(1, last.unsqueeze(1)).squeeze(1)
            paths.append(last)
        paths = torch.stack(paths[::-1], 1)
        logz = self.log_z(em, mask)
        all_ko = self.score(em, torch.ones_like(paths), mask)
        all_en = self.score(em, torch.zeros_like(paths), mask)
        return paths, (best - logz).exp(), (all_ko - logz).exp(), (all_en - logz).exp()


# ---------------------------------------------------------------- train / eval

def device():
    if torch.cuda.is_available():
        # ROCm 7.2.1 on Windows: MIOpen's LSTM backward fails to compile its kernels (missing
        # <type_traits>); PyTorch's own RNN kernels work and are ~9x faster than the CPU here.
        torch.backends.cudnn.enabled = False
        return torch.device("cuda")
    return torch.device("cpu")


def train(epochs=30, batch=512, lr=2e-3, seed=0):
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    dev = device()
    pure = pure_rows("train")
    mixed, endings = mixed_rows()
    en_rows = [r for r in pure if r[1][0] == EN]
    # real mixed tokens get MIXED_SHARE of the sampling mass (synthetic ones come on top, SYNTH_PER_EPOCH)
    scale = MIXED_SHARE * sum(w for _, _, w in pure) / max(sum(math.sqrt(c) for _, _, c in mixed["train"]), 1)
    real = [(k, y, math.sqrt(c) * scale) for k, y, c in mixed["train"]]
    pool = pure + real
    weights = [w for _, _, w in pool]
    print(f"device {dev}; pure {len(pure):,}, real mixed {len(real):,} (valid {len(mixed['valid'])}, test {len(mixed['test'])}),"
          f" endings {len(endings):,}", flush=True)
    model = Tagger().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = (len(pool) + SYNTH_PER_EPOCH) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch)
    valid = [(k, y) for k, y, _ in pure_rows("valid")] + [(k, y) for k, y, _ in mixed["valid"]]
    best = 1e9
    for ep in range(epochs):
        t0 = time.time()
        data = [(k, y) for k, y, _ in rnd.choices(pool, weights, k=len(pool))]
        data += synth(en_rows, endings, SYNTH_PER_EPOCH, rnd)
        data = [typo(k, y, rnd) if rnd.random() < TYPO_P and len(k) >= 2 else (k, y) for k, y in data]
        rnd.shuffle(data)
        model.train()
        total = 0.0
        for i in range(steps_per_epoch):
            ids, labs, mask = (x.to(dev) for x in encode(data[i * batch:(i + 1) * batch]))
            loss = model.nll(ids, labs, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        v = evaluate_loss(model, valid, dev)
        print(f"epoch {ep + 1}: train {total / steps_per_epoch:.4f} valid {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best = v
            torch.save(model.state_dict(), PATH)


@torch.no_grad()
def evaluate_loss(model, rows, dev, batch=2048):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(rows), batch):
        ids, labs, mask = (x.to(dev) for x in encode(rows[i:i + batch]))
        total += model.nll(ids, labs, mask).item() * ids.size(0)
        n += ids.size(0)
    return total / n


def load_model():
    model = Tagger()
    model.load_state_dict(torch.load(PATH, map_location="cpu"))
    return model.to(device()).eval()


@torch.no_grad()
def predict(model, keys_list, batch=4096):
    """-> [(labels, p_best, p_all_ko, p_all_en)]"""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(keys_list), batch):
        chunk = keys_list[i:i + batch]
        ids, _, mask = (x.to(dev) for x in encode([(k, [0] * len(k)) for k in chunk]))
        paths, pb, pko, pen = model.decode(ids, mask)
        for b, k in enumerate(chunk):
            out.append((paths[b, :len(k)].tolist(), pb[b].item(), pko[b].item(), pen[b].item()))
    return out


class TaggerScorer:
    """Drop-in for context.Scorer: z(keys) = log P(all ko) / P(all en), plus the best path for
    words that switch language inside (githubdp)."""

    def __init__(self):
        self.model, self.cache = load_model(), {}

    def prefill(self, keys):
        todo = sorted({k for k in keys if k not in self.cache})
        for k, (path, pb, pko, pen) in zip(todo, predict(self.model, todo)):
            self.cache[k] = (math.log(max(pko, 1e-12) / max(pen, 1e-12)), path, pb)

    def z(self, keys):
        if keys not in self.cache:
            self.prefill([keys])
        return self.cache[keys][0]

    def split(self, keys):
        """[(lang, keys part)] when the best path changes language inside the word, else None."""
        self.z(keys)
        path = self.cache[keys][1]
        if len(set(path)) < 2:
            return None
        parts, start = [], 0
        for i in range(1, len(path) + 1):
            if i == len(path) or path[i] != path[start]:
                parts.append(("ko" if path[start] == KO else "en", keys[start:i]))
                start = i
        return parts


def evaluate():
    from nunchi.make_noise import typo as word_typo

    model = load_model()
    rnd = random.Random(1)
    mixed, _ = mixed_rows()
    print(f"{'set':<12}{'n':>7}  {'exact':>7}  {'err@0.99':>9}  {'cov':>6}")
    sets = {"test": load("test"), "test_typo": load("test_typo")}
    for name, rows in sets.items():
        for lab in ("ko", "en"):
            rs = [r for r in rows if r["lab"] == lab]
            want = KO if lab == "ko" else EN
            res = predict(model, [r["keys"] for r in rs])
            exact = sum(all(x == want for x in p) for p, *_ in res)
            acted = [(p, pb) for p, pb, *_ in res if pb >= ACT_T]
            wrong = sum(not all(x == want for x in p) for p, _ in acted)
            print(f"{name + ' ' + lab:<12}{len(rs):>7}  {exact / len(rs):>7.4f}  {wrong / max(len(acted), 1):>9.4f}  {len(acted) / len(rs):>6.3f}")
    for name, rows in (("mixed test", mixed["test"]),
                       ("mixed+typo", [typo(k, y, rnd) + (c,) for k, y, c in mixed["test"]])):
        res = predict(model, [k for k, _, _ in rows])
        exact = sum(p == y for (p, *_), (_, y, _) in zip(res, rows))
        acted = [(p, y) for (p, pb, *_), (_, y, _) in zip(res, rows) if pb >= ACT_T]
        wrong = sum(p != y for p, y in acted)
        print(f"{name:<12}{len(rows):>7}  {exact / len(rows):>7.4f}  {wrong / max(len(acted), 1):>9.4f}  {len(acted) / len(rows):>6.3f}")
        bad = [(k, "".join("k" if x else "e" for x in p)) for (p, *_), (k, y, _) in zip(res, rows) if p != y]
        print("   wrong e.g.", bad[:10])


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
