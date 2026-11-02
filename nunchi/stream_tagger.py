"""Stream tagger: one Transformer over the whole key stream of a sentence, spaces included, instead of
a word layer plus a sentence layer. Every key attends to the keys of the other words directly, so a
word is judged with its neighbours' actual keys (so good / 내가), not a summary of them.

  input      : one token per key (52 letters, other, space), + position in the stream + position in the word
  encoder    : Transformer, 4 layers, d=256, 4 heads; trained from scratch (no LSTM, no MIOpen needed)
  mask       : a key sees every key up to the end of the next word (at most one word of lookahead)
  output     : per key en/ko, CRF inside each word (same CRF as the word tagger)

Data is the sentence tagger's Sampler (real sentences, code-switched ones, single words, English +
particle), cut at a random word so the last word is seen with no lookahead.

    .venv-rocm/Scripts/python -m nunchi.stream_tagger train <log> [steps] [eval_every] [patience] [batch] [workers]
"""
import os
import random
import sys
import time

# memory-efficient attention is off by default on RDNA4 (still "experimental" in ROCm); without it the
# masked attention falls back to the plain math path. Read when attention first runs, so set it here.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn

from nunchi.sentence_tagger import MAX_KEYS, Sampler, load_sentences, to_sentence
from nunchi.tagger import Tagger, VOCAB, device, mixed_rows
from nunchi.train_fasttext import MODELS

PATH = MODELS / "stream_tagger.pt"
BF16 = True  # matmul-heavy, so bf16 autocast pays off on RDNA GPUs; the CRF stays fp32
SPACE = len(VOCAB) + 2
POS = 192  # size of the stream position table (fixed: saved models have it)
MAX_T = 64  # keys kept per stream, latest whole words; 64 lost nothing against 192 and is ~4x faster
# digit keys: older models read every digit as "other" (id 1); a model with digits=True has its own id per
# digit after SPACE (TR2 on: the training text keeps numbers, 3개 / gpt4 / 2019, instead of dropping them)
DIGIT0 = SPACE + 1
PROFILE = os.environ.get("NUNCHI_PROFILE") == "1"  # log where a training step's time goes
prof = [0.0, 0.0, 0.0]


def cut(sent, t_end, left):
    """Input for judging the word t_end words from the end (0: the last word, decided now; 1: the one
    before, decided with the last word as lookahead): with `left`, only that many words before it."""
    if left is None:
        return sent
    return sent[max(0, len(sent) - 1 - t_end - left):]


def encode(batch, targets=None, skip_last=False, digits=False):
    """[[(keys, labels)]] -> stream tensors + per-word index for the CRF.
    ids, pos, wpos, widx (B, T); kidx, labs, cmask (B, W, L); wmask (B, W)
    targets: per row, the word (counted from the end) that gets a label; None = every word.

    Only the last two words of a stream can be labelled honestly: attention layers stack, so with
    4 layers a word reaches 4 words ahead through its neighbours whatever the mask says, while at
    typing time nothing exists beyond the word after it."""
    rows, targets = [], None if targets is None else list(targets)
    for b, sent in enumerate(batch):
        sent = [(k[:MAX_KEYS], y[:MAX_KEYS]) for k, y in sent]
        while sum(len(k) + 1 for k, _ in sent) > MAX_T and len(sent) > 1:
            if targets is not None and targets[b] >= len(sent) - 1:
                sent = sent[:-1]  # the labelled word is first: give up right context, never the word
                targets[b] -= 1
            else:
                sent = sent[1:]  # keep the latest words
        rows.append(sent)
    T = max(sum(len(k) + 1 for k, _ in s) for s in rows)
    W = max(len(s) for s in rows)
    L = max(len(k) for s in rows for k, _ in s)
    B = len(rows)
    # filled in numpy and converted once: per-slice torch writes were a quarter of eval time
    ids = np.zeros((B, T), np.int64)
    wpos = np.zeros((B, T), np.int64)
    widx = np.full((B, T), 10_000, np.int64)  # padding: never visible
    kidx = np.zeros((B, W, L), np.int64)
    labs = np.zeros((B, W, L), np.int64)
    cmask = np.zeros((B, W, L), bool)
    ar = np.arange(max(T, L) + 1)
    for b, sent in enumerate(rows):
        t = 0
        for w, (k, y) in enumerate(sent):
            n = len(k)
            ids[b, t:t + n] = [DIGIT0 + ord(c) - 48 if digits and "0" <= c <= "9" else VOCAB.get(c, 1) for c in k]
            wpos[b, t:t + n] = ar[1:n + 1]
            widx[b, t:t + n + 1] = w  # the space after a word belongs to it
            kidx[b, w, :n] = ar[t:t + n]
            labs[b, w, :n] = y
            if targets is None:  # every word, or (a reviser) every word that has words after it
                cmask[b, w, :n] = not skip_last or w < len(sent) - 1
            else:
                cmask[b, w, :n] = w == len(sent) - 1 - targets[b]
            t += n
            if w + 1 < len(sent):
                ids[b, t] = SPACE
                t += 1
            else:
                widx[b, t] = 10_000
    pos = np.broadcast_to(ar[:T], (B, T))
    cmask = torch.from_numpy(cmask)
    return (torch.from_numpy(ids), torch.from_numpy(pos.copy()), torch.from_numpy(wpos).clamp(max=MAX_KEYS),
            torch.from_numpy(widx), torch.from_numpy(kidx), torch.from_numpy(labs), cmask, cmask.any(2))


NEG = -1e4  # stands in for -inf in the log-space identity (keeps gradients finite)


def log_z_tree(self, em, mask):
    """Tagger.log_z without the per-key loop: with 2 labels, step t is a 2x2 log-space matrix
    M_t[j, k] = trans[j, k] + em_t[k] (identity where the key is padding), and the forward pass is the
    log-space product of the M_t, taken pairwise in log2(L) rounds instead of L small steps (the loop
    launched hundreds of tiny GPU ops per training step and was half its time). Same value."""
    N, L, _ = em.shape
    alpha = self.start + em[:, 0]
    if L > 1:
        m = self.trans.unsqueeze(0).unsqueeze(0) + em[:, 1:].unsqueeze(2)  # (N, L-1, 2, 2): [j, k]
        eye = torch.full((2, 2), NEG, device=em.device, dtype=em.dtype).fill_diagonal_(0.0)
        m = torch.where(mask[:, 1:].unsqueeze(-1).unsqueeze(-1), m, eye)
        while m.size(1) > 1:
            if m.size(1) % 2:
                m = torch.cat([m, eye.expand(N, 1, 2, 2)], 1)
            a, b = m[:, 0::2], m[:, 1::2]  # (N, n, 2, 2) each; product a @ b in log space
            m = torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)
        alpha = torch.logsumexp(alpha.unsqueeze(2) + m[:, 0], dim=1)
    return torch.logsumexp(alpha + self.end, dim=1)


def score_vec(self, em, labs, mask):
    """Tagger.score without the per-key loop: the path's emissions and transitions summed at once."""
    e = em.gather(2, labs.unsqueeze(2)).squeeze(2)  # (N, L)
    s = self.start[labs[:, 0]] + e[:, 0]
    if labs.size(1) > 1:
        s = s + ((e[:, 1:] + self.trans[labs[:, :-1], labs[:, 1:]]) * mask[:, 1:]).sum(1)
    last = labs.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
    return s + self.end[last]


class StreamTagger(nn.Module):
    decode_em = Tagger.decode_em  # same CRF as the word tagger; log_z / score computed without the key loop
    log_z, score = log_z_tree, score_vec

    def __init__(self, d=256, heads=4, layers=4, ff=1024, dropout=0.1, left=None, right=1, digits=False):
        super().__init__()
        self.heads = heads
        self.left = left  # words before the judged one it is given (None: all); the input is cut, see cut()
        # words after its own a key may attend to (1 up to TX4; R5: 5); None: the whole input, for a model
        # that labels every word typed so far at once (the last word = decided now, earlier ones = revised)
        self.right = right
        self.digits = digits
        self.tok = nn.Embedding(DIGIT0 + (10 if digits else 0), d, padding_idx=0)
        self.pos = nn.Embedding(POS, d)
        self.wpos = nn.Embedding(MAX_KEYS + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2)
        self.trans = nn.Parameter(torch.zeros(2, 2))
        self.start = nn.Parameter(torch.zeros(2))
        self.end = nn.Parameter(torch.zeros(2))

    def stream_emissions(self, ids, pos, wpos, widx):
        """(B, T) keys -> (B, T, 2) emissions: the encoder part, fixed shapes (train() may compile it)."""
        B, T = ids.shape
        x = self.tok(ids) + self.pos(pos) + self.wpos(wpos)
        # key i sees key j when j's word is at most one word after i's word (and j is not padding)
        if self.right is None:
            # every key sees every other one but padding: a (B, T) padding mask gives the same outputs at the
            # real keys as the full (B*heads, T, T) mask (padded rows differ, and nothing reads them), faster
            h = self.norm(self.enc(x, src_key_padding_mask=widx >= 10_000))
        else:
            see = widx.unsqueeze(1) < 10_000
            if self.right is not None:
                see = see & (widx.unsqueeze(1) <= widx.unsqueeze(2) + self.right)
            see = see.expand(B, T, T) | torch.eye(T, dtype=torch.bool, device=ids.device)
            mask = (~see).repeat_interleave(self.heads, 0)
            h = self.norm(self.enc(x, mask=mask))
        return self.out(h).float()  # the CRF runs in full precision even under bf16 autocast

    def forward_words(self, ids, pos, wpos, widx, kidx, cmask, wmask):
        B = ids.size(0)
        em_stream = self.stream_emissions(ids, pos, wpos, widx)
        idx = wmask.view(-1).nonzero().squeeze(1)
        Wn, L = kidx.size(1), kidx.size(2)
        b_of = torch.arange(B, device=ids.device).repeat_interleave(Wn)[idx]
        em = em_stream[b_of.unsqueeze(1), kidx.view(B * Wn, L)[idx]]
        return em, cmask.view(B * Wn, L)[idx], idx

    entry_w = 1.0  # train(): NUNCHI_ENTRY_W; a word entering English after a Korean word weighs this much

    def loss(self, ids, pos, wpos, widx, kidx, labs, cmask, wmask):
        em, cm, idx = self.forward_words(ids, pos, wpos, widx, kidx, cmask, wmask)
        y = labs.view(-1, labs.size(2))[idx]
        per_word = self.log_z(em, cm) - self.score(em, y, cm)
        if self.entry_w == 1.0:
            return per_word.mean()
        # entry: all keys English (labels 0), and the word before (labelled too) ends in Korean (1)
        lens = cmask.sum(2)
        last = labs.gather(2, (lens - 1).clamp(min=0).unsqueeze(2)).squeeze(2)
        prev_ko = torch.zeros_like(wmask)
        prev_ko[:, 1:] = (last[:, :-1] == 1) & wmask[:, :-1]
        entry = prev_ko & ((labs == 0) | ~cmask).all(2) & wmask
        w = 1 + (self.entry_w - 1) * entry.view(-1)[idx].float()
        return (per_word * w).sum() / w.sum()


def to_dev(batch, dev):
    return [x.to(dev) for x in batch]


RIGHT_NOW_P = 0.4  # right context up to R > 1: the word is decided at once this often, else 1..R words later


def labelled(samples, rnd, left, one_word, right=1):
    """-> (inputs, targets). one_word: one labelled word per stream, the last (decided now) or one with
    words after it, exactly as when typing: right=1, the one before half the time (one word of
    lookahead); right=R, 1..R words before the end 60% of the time (as many as the sentence has).
    Otherwise every word is labelled (the old way: words further back see further ahead than typing
    allows)."""
    if not one_word:
        return samples, None
    out, tg = [], []
    for s in samples:
        if right == 1:
            t_end = 1 if len(s) >= 2 and rnd.random() < 0.5 else 0
        else:
            t_end = 0 if rnd.random() < RIGHT_NOW_P else min(rnd.randint(1, right), len(s) - 1)
        out.append(cut(s, t_end, left))
        tg.append(t_end)
    return out, tg


_SAMPLER = None  # built once in the parent on Linux; forked workers share its memory


def producer(seed, queue, batch, bucket=16, left=None, one_word=False, right=1, skip_last=False, digits=False):
    """Builds `bucket` batches at a time from samples sorted by length, so little of a batch is padding.
    skip_last: a reviser (RV): label every word but the last one, which the decision at Space covers."""
    rnd = random.Random(seed)
    sampler = _SAMPLER or Sampler()
    while True:
        samples = sampler.batch(batch * bucket, rnd)
        if skip_last:  # a stream of one word has nothing to revise
            samples = [s for s in samples if len(s) > 1]
        samples, tg = labelled(samples, rnd, left, one_word, right)
        order = sorted(range(len(samples)), key=lambda i: sum(len(k) + 1 for k, _ in samples[i]))
        chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
        rnd.shuffle(chunks)
        for c in chunks:
            queue.put(encode([samples[i] for i in c], None if tg is None else [tg[i] for i in c], skip_last, digits))


@torch.no_grad()
def valid_loss(model, rows, dev, batch=256):
    """Loss on the last word of each row (decided now), with the model's own left context."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(rows), batch):
        chunk = [cut(s, 0, model.left) for s in rows[i:i + batch]]
        b = to_dev(encode(chunk, [0] * len(chunk), digits=model.digits), dev)
        total += model.loss(*b).item() * b[0].size(0)
        n += b[0].size(0)
    return total / n


@torch.no_grad()
def path_acc(model, rows, dev, batch=256):
    model.eval()
    ok = n = 0
    for i in range(0, len(rows), batch):
        ids, pos, wpos, widx, kidx, labs, cm, wm = to_dev(encode(rows[i:i + batch], digits=model.digits), dev)
        em, m, idx = model.forward_words(ids, pos, wpos, widx, kidx, cm, wm)
        paths = model.decode_em(em, m)[0]
        y = labs.view(-1, labs.size(2))[idx]
        ok += ((paths == y) | ~m).all(1).sum().item()
        n += m.size(0)
    return ok / n


def train(log_path, steps=300, eval_every=150, patience=5, batch=256, workers=5, lr=1e-3, seed=0):
    import multiprocessing as mp

    log = open(log_path, "a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if sys.platform == "win32":  # keep Windows from idle-sleeping while this runs (app-level)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    seed = int(os.environ.get("NUNCHI_SEED", seed))  # same recipe, another seed: how much results move by chance
    torch.manual_seed(seed)
    left = int(os.environ["NUNCHI_LEFT"]) if os.environ.get("NUNCHI_LEFT") else None
    one_word = os.environ.get("NUNCHI_ONE_WORD") == "1"
    # words after the judged one it learns to use; "all": every word typed so far sees every other one
    right = os.environ.get("NUNCHI_RIGHT", "1")
    right = None if right == "all" else int(right)
    # Linux: build the sampler once and fork the workers, which then share its data; before any GPU use, a
    # forked GPU context is not safe. Windows cannot fork: each spawned worker builds its own sampler.
    global _SAMPLER
    if sys.platform != "win32":
        _SAMPLER = Sampler()
    ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
    if sys.platform != "win32":  # batches through the queue: the default (file descriptors) cost WSL 40-77 ms
        torch.multiprocessing.set_sharing_strategy(os.environ.get("NUNCHI_SHARING", "file_system"))  # a step, this 3
    queue = ctx.Queue(maxsize=32)
    skip_last = os.environ.get("NUNCHI_SKIP_LAST") == "1"  # a reviser: no label on the word just typed
    digits = os.environ.get("NUNCHI_DIGITS") == "1"
    procs = [ctx.Process(target=producer, args=(seed + 1 + i, queue, batch, 16, left, one_word, right, skip_last, digits),
                         daemon=True) for i in range(workers)]
    for p in procs:
        p.start()
    dev = device()
    if dev.type == "cuda":  # leave the rest of the GPU to the desktop (a full card once froze the PC)
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("NUNCHI_VRAM", "0.7")))
    data = _SAMPLER.data if _SAMPLER else load_sentences()
    vr = random.Random(1)
    valid = []
    for name in ("nsmc", "kcbert", "tatoeba", "tatoeba_en"):
        for line in data[name][1][:1500]:
            s = to_sentence(line)[:20]
            valid.append(s[:vr.randint(1, len(s))])
    particles = [[(k, y)] for k, y, _ in mixed_rows()[0]["valid"]]
    out_path = os.environ.get("NUNCHI_OUT", str(PATH))
    # NUNCHI_SIZE="d,heads,layers,ff" for a bigger or smaller model (default 256,4,4,1024)
    size = dict(zip(("d", "heads", "layers", "ff"), map(int, os.environ["NUNCHI_SIZE"].split(",")))) \
        if os.environ.get("NUNCHI_SIZE") else {}
    model = StreamTagger(left=left, right=right, digits=digits, **size)
    model.entry_w = float(os.environ.get("NUNCHI_ENTRY_W", 1))
    if os.environ.get("NUNCHI_INIT"):  # continue from a saved checkpoint
        model.load_state_dict(torch.load(os.environ["NUNCHI_INIT"], map_location="cpu"))
    model.to(dev)
    say(f"stream transformer: {sum(p.numel() for p in model.parameters()):,} parameters; device {dev}; "
        f"batch {batch}; {steps} steps; workers {workers}; left context {left or 'all'}; key budget {MAX_T}; "
        f"labels {'one word per stream' if one_word else 'every word'}; right context up to {right}; "
        f"digit keys {'own ids' if digits else 'as other'} -> {out_path}")
    lr = float(os.environ.get("NUNCHI_LR", lr))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    best, bad, total, n_seen = 1e9, 0, 0.0, 0
    # <out>.ckpt holds everything needed to go on after a stop: weights, optimizer, LR schedule, step.
    # NUNCHI_RESUME=1 continues from it, so a stopped run ends as if it had run through (bar data order)
    ckpt_path = out_path + ".ckpt"
    first = 1
    if os.environ.get("NUNCHI_RESUME") and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, bad, n_seen, first = ck["best"], ck["bad"], ck["n_seen"], ck["step"] + 1
        say(f"  resumed from {ckpt_path} at step {ck['step']}")
    t_start = t_last = time.time()
    model.train()
    deadline = float(os.environ.get("NUNCHI_DEADLINE", "inf"))  # unix time: stop cleanly by then
    for step in range(first, steps + 1):
        if time.time() > deadline:
            say("  deadline reached, stopping")
            break
        t0 = time.perf_counter()
        raw = queue.get()
        t1 = time.perf_counter()
        b = to_dev(raw, dev)
        t2 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16 and dev.type == "cuda"):
            loss = model.loss(*b)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()
        if PROFILE:  # NUNCHI_PROFILE=1: where a step's time goes (queue wait, copy to GPU, compute)
            prof[0] += t1 - t0; prof[1] += t2 - t1; prof[2] += time.perf_counter() - t2
        n_seen += b[0].size(0)
        if step % 50 == 0:
            now = time.time()
            say(f"step {step}/{steps} loss {total / 50:.4f}  {50 * batch / (now - t_last):,.0f} prefixes/s  "
                f"queue {queue.qsize()}  ETA {(now - t_start) / (step - first + 1) * (steps - step) / 60:.1f} min")
            if PROFILE:
                say(f"  per step: queue wait {prof[0] * 20:.0f} ms, copy {prof[1] * 20:.0f} ms, compute {prof[2] * 20:.0f} ms")
                prof[:] = [0.0, 0.0, 0.0]
            total, t_last = 0.0, now
        if step % eval_every == 0 or step == steps:
            v, vp, pa = valid_loss(model, valid, dev), valid_loss(model, particles, dev), path_acc(model, particles, dev)
            model.train()
            say(f"  valid {v:.4f}  particles {vp:.4f}  particle acc {pa:.4f}  (seen {n_seen:,} prefixes)")
            if os.environ.get("NUNCHI_SAVE_LAST"):  # valid set barely switches language: judge by the real eval
                torch.save(model.state_dict(), out_path)
                say("  saved (latest)")
            if v < best:
                best, bad = v, 0
                if not os.environ.get("NUNCHI_SAVE_LAST"):
                    torch.save(model.state_dict(), out_path)
                    say("  saved")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "bad": bad, "n_seen": n_seen}, ckpt_path)
            if bad >= patience:
                say("  no improvement, stopping early")
                break
    for p in procs:
        p.terminate()
    say(f"finished in {(time.time() - t_start) / 60:.1f} min")


def load_model():
    model = StreamTagger()
    model.load_state_dict(torch.load(PATH, map_location="cpu"))
    return model.to(device()).eval()


@torch.no_grad()
def decide(model, prefixes, targets, batch=256):
    """Same contract as sentence_tagger.decide: prefixes [[keys]], target word index -> (path, pb, pko, pen)."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prefixes), batch):
        chunk = prefixes[i:i + batch]
        t_end = [len(s) - 1 - t for s, t in zip(chunk, targets[i:i + batch])]  # counted from the end
        chunk = [cut(s, e, model.left) for s, e in zip(chunk, t_end)]
        # only the judged word is labelled, so the CRF runs on one word per stream, in stream order
        enc = encode([[(k, [0] * len(k)) for k in s] for s in chunk], t_end, digits=getattr(model, "digits", False))
        ids, pos, wpos, widx, kidx, labs, cm, wm = to_dev(enc, dev)
        em, m, idx = model.forward_words(ids, pos, wpos, widx, kidx, cm, wm)
        # one copy back to the CPU: per-sample .item() calls each waited on the GPU (a third of eval time)
        paths, pb, pko, pen = (x.cpu().tolist() for x in model.decode_em(em, m))
        for j, (s, e) in enumerate(zip(chunk, t_end)):
            n = min(len(s[len(s) - 1 - e]), MAX_KEYS)
            out.append((paths[j][:n], pb[j], pko[j], pen[j]))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "train":
        args = [int(a) for a in sys.argv[3:]]
        names = ["steps", "eval_every", "patience", "batch", "workers"]
        train(sys.argv[2], **dict(zip(names, args)))
