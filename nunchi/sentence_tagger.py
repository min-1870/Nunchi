"""Sentence tagger: the key-level word tagger plus a sentence layer, so a word is read together with
the words around it (내가 vs so good), not just their language as the HMM did.

  char layer     : the word tagger's embedding + BiLSTM over each word's keys (initialised from tagger.pt)
  sentence layer : BiLSTM over the words (mean of each word's key states)
  output         : per key, linear over [key state, word's sentence state] -> CRF inside the word
The output layer starts as the word tagger's (sentence half zero), so training starts from it.

Online use: a word is decided from the words typed so far plus at most one word after it. Training
cuts each sentence at a random word, so the model sees every word with 0..n words of right context.

    .venv-rocm/Scripts/python -m nunchi.sentence_tagger train <log file>
"""
import hashlib
from itertools import accumulate
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nunchi.dubeolsik import to_keys
import re

from nunchi.sentence_data import HANGUL, LATIN, MIXED, OUT as SENT
from nunchi.tagger import EN, KO, PATH as WORD_PATH, Tagger, VOCAB, device, pure_rows, typo
from nunchi.train_fasttext import MODELS

PATH = MODELS / "sentence_tagger.pt"
MAX_WORDS, MAX_KEYS = 20, 24
KO_SOURCES = ("nsmc", "kcbert", "tatoeba")
# Sampling mass: Korean and English sentences about even (an earlier 65/25 mix taught the model that
# a short ambiguous word is usually Korean); Korean sources by sqrt(size) so comments do not drown
# reviews; single words keep the word tagger's skills (English + particle) from being forgotten.
SHARE = {"ko": float(os.environ.get("NUNCHI_KO", 0.40)), "en": float(os.environ.get("NUNCHI_EN", 0.30)),
         "synth": float(os.environ.get("NUNCHI_SYNTH", 0.10)), "words": float(os.environ.get("NUNCHI_WORDS", 0.20)),
         # real Korean sentences that already contain English, sampled on their own: the moments the
         # language switches are what the app is for, and they are rare in plain Korean text
         "kmix": float(os.environ.get("NUNCHI_KMIX", 0.0)),
         # the same, from the web (mC4 Korean train split, data/sent/web.txt): every topic, not just reviews
         "web": float(os.environ.get("NUNCHI_WEB", 0.0)),
         # real Korean sentences with their loanwords written in English (data/sent/loan.txt, loanwords.py)
         "loan": float(os.environ.get("NUNCHI_LOAN", 0.0)),
         # a word swapped for its translation (translations.py): English with Korean words (trans_en),
         # Korean with an English noun or adverb (trans_ko); replaces random insertion (synth)
         "trans_en": float(os.environ.get("NUNCHI_TRANS_EN", 0.0)),
         "trans_ko": float(os.environ.get("NUNCHI_TRANS_KO", 0.0))}
# synthetic sentences start from Korean and English sentences ("all"), or English only ("en"): once the
# loanword set gives meaningful English inside Korean, random insertion is only needed the other way
SYNTH_SRC = os.environ.get("NUNCHI_SYNTH_SRC", "all")
# English in a Korean sentence (real or inserted) gets a real Korean ending this often
ENDING_P = float(os.environ.get("NUNCHI_ENDING_P", 0.5))
# English inside Korean text, written in capitals (JSA, APK, BGM), is typed in lower case this often: web text
# capitalises acronyms, people typing often do not (jsa의, apk 파일), and the lower-case form was barely seen
LOWER_P = float(os.environ.get("NUNCHI_LOWER_P", 0.0))
WORDS_MIXED_SHARE = float(os.environ.get("NUNCHI_WORDS_MIXED", 0.3))  # English + particle among single words
# lines of the web / kmix pools holding short English (1-3 letters, or keys that also read as Korean: so, go)
# are drawn this many times as often: short English inside Korean is where the models lean Korean
SHORT_EN_X = float(os.environ.get("NUNCHI_SHORT_EN_X", 1))
# which lines count: both / plain / amb / insert / switch, see Sampler.__init__
SHORT_EN_MODE = os.environ.get("NUNCHI_SHORT_EN_MODE", "both")
# lines holding a Korean word whose keys also read as English (해 = go, 내 = so, 둥 = end) are drawn this many
# times as often, in the Korean sources and the web / kmix pools: the other side of SHORT_EN_X, so the model
# learns to tell the two readings apart instead of leaning one way
AMB_KO_X = float(os.environ.get("NUNCHI_AMB_KO_X", 1))
# how context enters: "start" = only the language a word starts in; "key" = a correction on every key
CTX_MODE = os.environ.get("NUNCHI_CTX", "start")
TYPO_WORD_P = 0.15


# ---------------------------------------------------------------- data

RUN = re.compile(r"[가-힣ㄱ-ㅣ]+|[A-Za-z]+|[0-9]+")


def word(tok, prev=KO):
    """token as written -> (keys, labels). Digits type the same in either mode, so they take a language
    from the letters next to them in the token (3개 -> Korean, gpt4 -> English), else from the word before
    (prev: its last label); a sentence's first number counts as Korean."""
    if HANGUL.match(tok):
        k = to_keys(tok)
        return k, [KO] * len(k)
    if LATIN.match(tok):
        return tok, [EN] * len(tok)
    runs = RUN.findall(tok)
    labs = [KO if HANGUL.match(r) else EN if r.isalpha() else None for r in runs]
    for i in range(len(labs)):
        if labs[i] is None:
            labs[i] = next((x for x in labs[i - 1::-1] if x is not None), None) if i else None
    for i in range(len(labs) - 1, -1, -1):
        if labs[i] is None:
            labs[i] = labs[i + 1] if i + 1 < len(labs) and labs[i + 1] is not None else prev
    keys, y = "", []
    for r, lab in zip(runs, labs):
        k = to_keys(r) if HANGUL.match(r) else r
        keys += k
        y += [lab] * len(k)
    return keys, y


def to_sentence(line):
    out, prev = [], KO
    for t in line.split(" "):
        w = word(t, prev)
        out.append(w)
        prev = w[1][-1]
    return out


def lowered(w):
    """(keys, labels) with the English keys in lower case (JSA의 -> jsa의); Korean keys keep their Shift."""
    k, y = w
    return "".join(c.lower() if lab == EN else c for c, lab in zip(k, y)), y


class LineStore:
    """A read-only list of lines kept as one UTF-8 buffer + offsets. Python str objects took ~3x the
    memory, and every read of one rewrites its reference count, so forked data workers could not share
    them: three workers held 1.8 GB each. Supports len(), [i], iteration (random.choice / choices work)."""

    def __init__(self, lines):
        enc = [line.encode("utf-8") for line in lines]
        self.off = np.zeros(len(enc) + 1, np.int64)
        np.cumsum([len(b) for b in enc], out=self.off[1:])
        self.buf = b"".join(enc)

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return self.buf[self.off[i]:self.off[i + 1]].decode("utf-8")

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_sentences():
    """{source: (train, valid)} of raw lines (converted when sampled); 1% held out for valid by hash."""
    out = {}
    for source in ("nsmc", "kcbert", "tatoeba", "tatoeba_en"):
        tr, va = [], []
        # NUNCHI_POOL_SUFFIX=_v2: the v2 pools (numbers and Hangul+Latin kept), see sentence_data.clean
        fname = source + ("" if source == "tatoeba_en" else os.environ.get("NUNCHI_POOL_SUFFIX", ""))
        with (SENT / f"{fname}.txt").open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                (va if int(hashlib.md5(line.encode()).hexdigest(), 16) % 100 == 0 else tr).append(line)
        out[source] = (LineStore(tr), va)
        print(f"  {source}: train {len(tr):,} valid {len(va):,}", flush=True)
    return out


class Sampler:
    """Draws training prefixes: real sentences, code-switched sentences, and single words."""

    def __init__(self, data=None):
        from nunchi.tagger import mixed_rows

        self.data = data or load_sentences()
        rows = pure_rows("train")
        self.en = [(k, y) for k, y, _ in rows if y[0] == EN]
        self.en_w = [w for _, y, w in rows if y[0] == EN]
        self.ko = [(k, y) for k, y, _ in rows if y[0] == KO]
        self.ko_w = [w for _, y, w in rows if y[0] == KO]
        mixed, endings = mixed_rows()
        self.endings, self.end_w = zip(*[(to_keys(e), c) for e, c in endings.items()])
        self.words = [(k, y) for k, y, _ in rows] + [(k, y) for k, y, _ in mixed["train"]]
        scale = WORDS_MIXED_SHARE * sum(w for _, _, w in rows) / max(sum(math.sqrt(c) for *_, c in mixed["train"]), 1)
        self.words_w = [w for _, _, w in rows] + [math.sqrt(c) * scale for *_, c in mixed["train"]]
        sizes = {s: math.sqrt(len(self.data[s][0])) for s in KO_SOURCES}
        self.kmix = LineStore([l for src in ("nsmc", "kcbert") for l in self.data[src][0]
                               if any(c.isascii() and c.isalpha() for c in l)])
        self.pools = {"kmix": self.kmix}
        for name in ("web", "loan", "trans_en", "trans_ko"):
            self.pools[name] = []
            if SHARE[name] > 0:
                # NUNCHI_WEB_FILE: another web pool (e.g. built from more mC4 shards) in place of web.txt
                fname = os.environ.get("NUNCHI_WEB_FILE", "web.txt") if name == "web" else f"{name}.txt"
                with (SENT / fname).open(encoding="utf-8") as f:
                    self.pools[name] = LineStore([line.rstrip("\n") for line in f if line.strip()])
        self.web, self.loan = self.pools["web"], self.pools["loan"]
        self.pool_cw = {}
        if SHORT_EN_X != 1 or ":" in SHORT_EN_MODE or AMB_KO_X != 1:
            from nunchi.ambiguous_eval import ambiguous_keys

            amb = ambiguous_keys()
            eng = lambda t: t.isascii() and t.isalpha()
            sh = lambda t: eng(t) and (len(t) <= 3 or t.lower() in amb)
            ko = lambda t: t[:1] >= "가" and t[:1] <= "힣"

            def entry(t, after):
                return any(sh(t[i]) and ko(t[i - 1]) and after(t[i + 1]) for i in range(1, len(t) - 1))

            # SHORT_EN_MODE (several joined by +, a line counts if any matches): both (A: any short English),
            # plain (1-3 letters whose keys are not a Korean reading, ERP / jpg), amb (keys that also read as
            # Korean, H / GD / so), insert (short English between Korean words: 저희 H 모터스), switch (short
            # English after Korean, English after it)
            kinds = {"both": lambda t: any(map(sh, t)),
                     "plain": lambda t: any(eng(w) and len(w) <= 3 and w.lower() not in amb for w in t),
                     "amb": lambda t: any(eng(w) and w.lower() in amb for w in t),
                     "insert": lambda t: entry(t, ko), "switch": lambda t: entry(t, eng),
                     # English that is not ambiguous (3+ letters, keys not a Korean reading): pushed out by "amb"
                     "plain3": lambda t: any(eng(w) and len(w) >= 3 and w.lower() not in amb for w in t),
                     # number words (3개, gpt4, 5km, 시즌2) and Hangul+Latin (채널A): upweighting the lines above
                     # pushed these out (R3-2: number words 72-78 -> 92 wrong)
                     "code": lambda t: any(re.search(r"[0-9]", w) and re.search(r"[A-Za-z가-힣]", w) or
                                           re.match(r"^[가-힣]+[A-Za-z]", w) for w in t)}
            # a mode may carry its own multiplier (amb:4+switch:2); without one it takes SHORT_EN_X
            modes = [(kinds[m.split(":")[0]], float(m.split(":")[1]) if ":" in m else SHORT_EN_X)
                     for m in SHORT_EN_MODE.split("+")]

            def short(line):
                """-> the largest multiplier among the modes the line matches (1 if none)"""
                t = line.split(" ")
                return max([x for f, x in modes if f(t)], default=1.0)

            def amb_ko(line):
                return any(ko(w) and len(w) <= 2 and to_keys(w).lower() in amb for w in line.split(" "))

            code_x = max([x for f, x in modes if f is kinds["code"]], default=1.0)

            def weight(line, english):
                # the web / kmix pools take every mode; the Korean sources only the number-word one
                w = short(line) if english else (code_x if code_x != 1 and kinds["code"](line.split(" ")) else 1.0)
                return w * (AMB_KO_X if AMB_KO_X != 1 and amb_ko(line) else 1.0)
            for name in ("web", "kmix"):
                self.pool_cw[name] = list(accumulate(weight(l, True) for l in self.pools[name]))
            if AMB_KO_X != 1 or code_x != 1:
                for src in KO_SOURCES:  # keyed by source: the "ko" kinds draw from self.data[src]
                    self.pool_cw[src] = list(accumulate(weight(l, False) for l in self.data[src][0]))
        extra = ("synth", "words", "kmix", "web", "loan", "trans_en", "trans_ko")
        self.kinds = [("ko", s) for s in KO_SOURCES] + [("en", "tatoeba_en")] + [(k, None) for k in extra]
        self.kind_w = [SHARE["ko"] * sizes[s] / sum(sizes.values()) for s in KO_SOURCES] + \
                      [SHARE["en"]] + [SHARE[k] for k in extra]

        # random.choices(weights=...) re-sums the whole list on every call (O(n) per pick, 60% of data
        # time with 100k+ word lists); cumulative weights once make each pick a binary search
        for name in ("en_w", "ko_w", "words_w", "end_w", "kind_w"):
            setattr(self, name, list(accumulate(getattr(self, name))))

    def with_ending(self, w, rnd):
        """English word + a Korean ending seen after English in real text (MBC + 는)."""
        e = rnd.choices(self.endings, cum_weights=self.end_w)[0]
        return w[0] + e, w[1] + [KO] * len(e)

    def code_switched(self, rnd):
        src = "tatoeba_en" if SYNTH_SRC == "en" else rnd.choice(["nsmc", "kcbert", "tatoeba_en"])
        sent = to_sentence(rnd.choice(self.data[src][0]))
        korean = src != "tatoeba_en"
        other, weights = (self.en, self.en_w) if korean else (self.ko, self.ko_w)
        for _ in range(rnd.choice([1, 1, 2])):
            span = rnd.choices(other, cum_weights=weights, k=rnd.choice([1, 1, 1, 2]))
            if korean and rnd.random() < ENDING_P:
                span[-1] = self.with_ending(span[-1], rnd)
            i = rnd.randrange(len(sent) + 1)
            sent[i:i] = span
        return sent[:MAX_WORDS]

    def one(self, rnd):
        kind, src = rnd.choices(self.kinds, cum_weights=self.kind_w)[0]
        if kind == "synth":
            sent = self.code_switched(rnd)
        elif kind == "words":
            w = rnd.choices(self.words, cum_weights=self.words_w)[0]
            if w[1][0] == EN and w[1][-1] == EN and rnd.random() < 0.2:
                w = self.with_ending(w, rnd)
            sent = [w]
        else:
            pool = self.pools.get(kind) or self.data[src][0]
            key = kind if kind in self.pools else src
            line = rnd.choices(pool, cum_weights=self.pool_cw[key])[0] if key in self.pool_cw else rnd.choice(pool)
            sent = to_sentence(line)[:MAX_WORDS]
            if kind in ("ko", "kmix", "web"):  # English already in Korean text: give it a particle, in its real context
                sent = [self.with_ending(w, rnd) if all(x == EN for x in w[1]) and rnd.random() < ENDING_P else w
                        for w in sent]
        if LOWER_P and kind in ("ko", "kmix", "web", "trans_ko", "words"):
            sent = [lowered(w) if rnd.random() < LOWER_P else w for w in sent]
        sent = [typo(k, y, rnd) if rnd.random() < TYPO_WORD_P and len(k) >= 2 else (k, y) for k, y in sent]
        return sent[:rnd.randint(1, len(sent))]  # online: the words so far, the last one maybe a lookahead

    def batch(self, n, rnd):
        return [self.one(rnd) for _ in range(n)]


def producer(seed, queue, batch):
    """Worker process: builds encoded batches so the GPU never waits on Python."""
    rnd = random.Random(seed)
    sampler = Sampler()
    while True:
        ids, labs, cm, wm = encode(sampler.batch(batch, rnd))
        queue.put((ids.to(torch.uint8), labs.to(torch.uint8), cm, wm))


def encode(batch):
    """[[(keys, labels)]] -> ids, labs, cmask (B, W, L), wmask (B, W)"""
    W = min(max(len(s) for s in batch), MAX_WORDS)
    L = min(max(len(k) for s in batch for k, _ in s), MAX_KEYS)
    ids = torch.zeros(len(batch), W, L, dtype=torch.long)
    labs = torch.zeros(len(batch), W, L, dtype=torch.long)
    cmask = torch.zeros(len(batch), W, L, dtype=torch.bool)
    for b, sent in enumerate(batch):
        for w, (k, y) in enumerate(sent[-W:]):
            k, y = k[:L], y[:L]
            ids[b, w, :len(k)] = torch.tensor([VOCAB.get(c, 1) for c in k])
            labs[b, w, :len(y)] = torch.tensor(y)
            cmask[b, w, :len(k)] = True
    return ids, labs, cmask, cmask.any(2)


# ---------------------------------------------------------------- model

class SentenceTagger(Tagger):
    def __init__(self, ctx=192):
        super().__init__()
        H = self.lstm.hidden_size
        self.sent = nn.LSTM(2 * H, ctx, 1, batch_first=True, bidirectional=True)
        # Context only shifts which language a word *starts* in (a bias on its first key), like a CRF
        # transition across words. Inside the word the frozen word tagger alone decides where the
        # language switches (MBC는 = e e e k k k), so context cannot train that skill away.
        # Starts at zero: with no useful context the model answers exactly like the word tagger.
        self.ctx_start = nn.Linear(2 * H + 2 * ctx, 2)  # mode "key" uses it on every key
        nn.init.zeros_(self.ctx_start.weight)
        nn.init.zeros_(self.ctx_start.bias)
        self.frozen = False

    def init_from_word_tagger(self, path=WORD_PATH, freeze=True):
        """Copy the word tagger in. Frozen (embedding, key BiLSTM, output, CRF), its skills such as
        English + particle cannot be trained away, and no gradient runs through the key layer."""
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        self.frozen = freeze
        for p in (*self.emb.parameters(), *self.lstm.parameters(), *self.out.parameters(),
                  self.trans, self.start, self.end):
            p.requires_grad_(not freeze)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.lstm.eval()  # no dropout inside the frozen key layer: same features as the word tagger
        return self

    def forward_words(self, ids, cmask, wmask):
        """-> emissions (N, L, 2) and key masks (N, L) for the N real words, and their (b, w) index."""
        B, W, L = ids.shape
        idx = wmask.view(-1).nonzero().squeeze(1)
        x = ids.view(B * W, L)[idx]
        cm = cmask.view(B * W, L)[idx]
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            h, _ = self.lstm(self.emb(x))
            word_em = self.out(h)
        pooled = (h * cm.unsqueeze(2)).sum(1) / cm.sum(1, keepdim=True)
        words = torch.zeros(B * W, h.size(2), device=h.device)
        words[idx] = pooled
        lengths = wmask.sum(1).cpu()
        packed = pack_padded_sequence(words.view(B, W, -1), lengths, batch_first=True, enforce_sorted=False)
        c, _ = self.sent(packed)
        c, _ = pad_packed_sequence(c, batch_first=True, total_length=W)
        c = c.reshape(B * W, -1)[idx]
        if CTX_MODE == "key":
            em = word_em + self.ctx_start(torch.cat([h, c.unsqueeze(1).expand(-1, L, -1)], 2))
        else:
            start_bias = self.ctx_start(torch.cat([h[:, 0], c], 1))
            em = torch.cat([word_em[:, :1] + start_bias.unsqueeze(1), word_em[:, 1:]], 1)
        return em, cm, idx

    def loss(self, ids, labs, cmask, wmask):
        em, cm, idx = self.forward_words(ids, cmask, wmask)
        y = labs.view(-1, labs.size(2))[idx]
        return (self.log_z(em, cm) - self.score(em, y, cm)).mean()


# ---------------------------------------------------------------- train

def train(log_path, steps=300, eval_every=150, patience=4, batch=512, lr=2e-3, workers=3, seed=0):
    """Train for `steps` (OneCycle schedule), validate every `eval_every`, stop after `patience`
    validations without improvement. Batches come from `workers` producer processes."""
    import multiprocessing as mp
    from nunchi.tagger import mixed_rows

    log = open(log_path, "a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if sys.platform == "win32":  # keep Windows from idle-sleeping while this process runs (app-level, no setting changed)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(seed)
    dev = device()
    say(f"starting {workers} batch workers (each loads the sentences)")
    queue = mp.get_context("spawn").Queue(maxsize=32)
    procs = [mp.get_context("spawn").Process(target=producer, args=(seed + 1 + i, queue, batch), daemon=True)
             for i in range(workers)]
    for p in procs:
        p.start()

    data = load_sentences()
    valid_rnd = random.Random(1)
    valid = []
    for name in ("nsmc", "kcbert", "tatoeba", "tatoeba_en"):
        for line in data[name][1][:1500]:
            s = to_sentence(line)[:MAX_WORDS]
            valid.append(s[:valid_rnd.randint(1, len(s))])
    particles = [[(k, y)] for k, y, _ in mixed_rows()[0]["valid"]]  # English + particle, alone

    model = SentenceTagger()
    model.init_from_word_tagger()
    model.to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    say(f"trainable parameters {sum(p.numel() for p in trainable):,} of {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    say(f"device {dev}; batch {batch}; {steps} steps; valid {len(valid)} prefixes + {len(particles)} particle words; "
        f"start valid {valid_loss(model, valid, dev):.4f} particles {valid_loss(model, particles, dev):.4f} "
        f"particle acc {path_acc(model, particles, dev):.4f}")
    best, bad, total, n_seen = 1e9, 0, 0.0, 0
    t_start = t_last = time.time()
    model.train()
    for step in range(1, steps + 1):
        ids, labs, cm, wm = queue.get()
        ids, labs, cm, wm = ids.long().to(dev), labs.long().to(dev), cm.to(dev), wm.to(dev)
        loss = model.loss(ids, labs, cm, wm)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()
        n_seen += ids.size(0)
        if step % 50 == 0:
            now = time.time()
            rate = 50 * batch / (now - t_last)
            eta = (now - t_start) / step * (steps - step)
            say(f"step {step}/{steps} loss {total / 50:.4f}  {rate:,.0f} prefixes/s  "
                f"queue {queue.qsize()}  ETA {eta / 60:.1f} min")
            total, t_last = 0.0, now
        if step % eval_every == 0 or step == steps:
            v, vp, pa = valid_loss(model, valid, dev), valid_loss(model, particles, dev), path_acc(model, particles, dev)
            model.train()
            say(f"  valid {v:.4f}  particles {vp:.4f}  particle acc {pa:.4f}  (seen {n_seen:,} prefixes)")
            if v < best:
                best, bad = v, 0
                torch.save(model.state_dict(), PATH)
                say("  saved")
            else:
                bad += 1
                if bad >= patience:
                    say("  no improvement, stopping early")
                    break
    for p in procs:
        p.terminate()
    say(f"finished in {(time.time() - t_start) / 60:.1f} min")


@torch.no_grad()
def path_acc(model, rows, dev, batch=256):
    """Share of words whose best path equals the true labels exactly (e.g. MBC는 -> e e e k k k)."""
    model.eval()
    ok = n = 0
    for i in range(0, len(rows), batch):
        ids, labs, cm, wm = (x.to(dev) for x in encode(rows[i:i + batch]))
        em, m, idx = model.forward_words(ids, cm, wm)
        paths = model.decode_em(em, m)[0]
        y = labs.view(-1, labs.size(2))[idx]
        ok += ((paths == y) | ~m).all(1).sum().item()
        n += m.size(0)
    return ok / n


@torch.no_grad()
def valid_loss(model, rows, dev, batch=256):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(rows), batch):
        ids, labs, cm, wm = (x.to(dev) for x in encode(rows[i:i + batch]))
        total += model.loss(ids, labs, cm, wm).item() * ids.size(0)
        n += ids.size(0)
    return total / n


# ---------------------------------------------------------------- inference

def load_model():
    model = SentenceTagger()
    model.load_state_dict(torch.load(PATH, map_location="cpu"))
    return model.to(device()).eval()


@torch.no_grad()
def decide(model, prefixes, targets, batch=256):
    """prefixes: [[keys]], targets: index of the word to decide in each prefix.
    -> [(path, p_best, p_all_ko, p_all_en)] for that word."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prefixes), batch):
        chunk = prefixes[i:i + batch]
        ids, _, cm, wm = (x.to(dev) for x in encode([[(k, [0] * len(k)) for k in s] for s in chunk]))
        em, m, idx = model.forward_words(ids, cm, wm)
        paths, pb, pko, pen = model.decode_em(em, m)
        W = ids.size(1)
        pos = {v: j for j, v in enumerate(idx.tolist())}
        for b, (s, t) in enumerate(zip(chunk, targets[i:i + batch])):
            offset = len(s) - min(len(s), W)  # encode keeps the last W words
            j = pos[b * W + (t - offset)]
            n = min(len(s[t]), MAX_KEYS)
            out.append((paths[j, :n].tolist(), pb[j].item(), pko[j].item(), pen[j].item()))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "train":
        # train <log> [steps] [eval_every] [patience] [batch] [workers]
        args = [int(a) for a in sys.argv[3:]]
        names = ["steps", "eval_every", "patience", "batch", "workers"]
        train(sys.argv[2], **dict(zip(names, args)))
