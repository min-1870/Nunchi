# Nunchi (gksdud)

**English** | [한국어](README.ko.md)

> Automatic Korean/English input switching for Windows: fixes what you typed in the wrong IME mode, without pressing the 한/영 key.

Type `dkssud` in English mode and it becomes `안녕`. Type `ㅗ디ㅣㅐ` in Korean mode and it becomes `hello`. The IME mode switches along with it.
The repo name `gksdud` is what "한영" looks like when typed in English mode.

> **Status: research.** The classifier and its evaluation work; a key-level tagger handles English with a Korean particle (`MBC는`) too. A sentence-level model is in training. The Windows tray app does not exist yet.

## Principles

- **100% local.** No network access at all. This handles keystrokes, so trust comes first.
- **Do nothing unless confident.** A wrong correction costs more than a missed one.
- **Undo hotkey.** Words you undo are learned as exceptions and never corrected again.
- **Correct within 100ms** of the word boundary.

## How it works

```
capture keys (WH_KEYBOARD_LL)
  → buffer raw keys until a word boundary
  → key tagger: one language label per key (BiLSTM-CRF)
      dkssud   → k k k k k k      → 안녕
      MBCsms   → e e e k k k      → MBC는
  → context: 2-state HMM over words (being replaced by a sentence model)
  → confident now?            correct immediately
    ambiguous?                wait one word, then correct retroactively
    still below 0.99?         do nothing
  → correct: Backspace → switch IME → retype raw keys
```

Everything works on the **raw key sequence** (`dkssud`), not the characters on screen, so input typed in either mode is handled the same way. Labels for training come for free: text written in Hangul is Korean, text written in Latin letters is English.

## Results

All numbers are on held-out data: test words never appear in training (split by word type), and the sentence sets come from text the models never saw.

### Sentences, every word decided

How many words end up in the wrong language when the model must decide every word (as an input method has to). Context: 2-state HMM, act now or wait one word.

| Set | Words | fastText (v3) | **Key tagger** |
|---|---|---|---|
| Korean sentences (NSMC reviews) | 12,819 | 14 | **6** |
| English sentences (Tatoeba) | 14,216 | **0** | 2 |
| Korean + inserted English (synthetic) | 16,154 | 49 | **48** |
| English + inserted Korean (synthetic) | 17,565 | 101 | **57** |
| **Real mixed reviews** (NSMC test) | 8,599 | 88 | **82** |
| Korean sentences + typos | 12,819 | 98 | **36** |
| English sentences + typos | 14,216 | 55 | **28** |
| Real mixed + typos | 8,599 | 149 | **109** |

Typos: each word gets one keyboard typo (neighbour key, dropped, doubled or swapped key) with probability 0.15.

### Sentences, correcting only when confident (≥ 0.99)

"Wrong" is the share of corrections that were mistakes; "corrected" is the share of words the model was confident enough to act on.

| Set | fastText v3: wrong / corrected | **Key tagger: wrong / corrected** |
|---|---|---|
| Korean sentences | 0.05% / 99.0% | **0.00% / 98.0%** |
| English sentences | 0% / 98.7% | 0% / 97.6% |
| Korean + inserted English | 0.08% / 98.7% | **0.01% / 97.4%** |
| English + inserted Korean | 0.05% / 98.0% | **0.00% / 95.3%** |
| **Real mixed reviews** | 0.48% / 97.9% | **0.17% / 95.6%** |

### Single words

| Same test words, every word decided | fastText v3 | **Key tagger** |
|---|---|---|
| Korean words | 99.73% | **99.75%** |
| English words | **99.26%** | 99.09% |
| Korean words + one typo | 97.53% | **98.64%** |
| English words + one typo | **98.23%** | 97.83% |
| English + Korean particle in one word (`MBC는`, `TV에서`), real tokens | not possible | **90.4%** |

The particle test uses real tokens from the corpora whose English part never appears in training.

## What we learned

The order things happened in, and what each step showed.

**1. Single-word scores were far too optimistic.** The first fastText model made 0.02% wrong corrections on single words but 1.03% on real sentences. Short, frequent, ambiguous words (그 `rm`, 내 `so`) are rare in a word list and common in real text. From then on every result was measured on sentences.

**2. Calibration and one-word lookahead made context usable.** fastText says p = 1.000 even when wrong; softening its scores (temperature) cut wrong corrections from 1.03% to 0.39%. A 2-state HMM over words (P(same language as the previous word)) brought Korean sentences to 0.05%, but it pulled the first English word of a Korean sentence toward Korean. Waiting one word before deciding ambiguous words fixed most of that.

**3. The synthetic mixed set hid the real problem.** It inserted random dictionary words (`disgruntled`, `hierarchical`): long, rare, easy. In real reviews the English is short and uppercase (`TV`, `OST`, `CG`), which the first model had never seen in training (wordfreq is all lowercase). A 4-state HMM that remembers the sentence's base language looked best on synthetic data and was worst on real data: it pulled those short English words toward Korean (50% of its confident corrections there were wrong).

**4. Fixing the data beat every context trick.** Rebuilding the training words from real text cut wrong corrections on real mixed reviews from 0.68% to 0.26% without any context:
- English as people type it: cased forms from real sentences and English found inside Korean text.
- A bug: common Korean words (이 `dl`, 수 `tn`, 그 `rm`, ㅋㅋ `zz`) were deleted from training because a noisy English word list contained the same keys. Collisions now go to the far more common side, and close calls (`so` / 내) stay in both languages instead of being dropped.
- Korean typed in the wrong mode inside reviews (`gks` = 한) had been learned as English.
- Korean from news comments (KcBERT) and Tatoeba besides movie reviews, so everyday words (키, 메뉴) are known.

**5. Context has a hard limit, and we measured it.**
- An HMM can move a word's odds by at most s/(1−s), however many words it looks at. A word the model is 99% sure of can never be overturned by later words, so "correct, then undo after the next word" never triggers (0 cases on every set).
- With an oracle that knows the true language of the neighbouring words, context removes most misses in single-language sentences but none at switch points.
- So the HMM mostly raises how many words get corrected; it barely lowers wrong corrections. What is left are words that are real words in both languages (`so` / 내, `go` / 해, `do` / 양). Telling those apart needs the neighbouring *words*, not just their language. That is what the sentence model is for.

**6. A key-level tagger removes the one-language-per-word assumption.** It labels every key, so `githubdp` → `github` + `에` is just a path that switches once. It matches fastText on whole words, is more robust to Korean typos, and handles English + particle (90.4% on real held-out tokens), which a word classifier cannot do at all.

**7. Wrong corrections cost more than they look.** After a correction the app switches the IME mode, so a wrong correction also makes the following words come out in the wrong mode. Missed corrections only matter where the user is actually in the wrong mode: at the start of typing, after switching windows, and where they forget to toggle when changing language. Those are also where context helps least, which is why the word-level model matters most.

## Reproduce

Tested with Python 3.12 on Windows. `PYTHONUTF8=1` is needed so the console can print Korean.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.count_corpus     # word / Latin / Latin+Hangul counts per source
.venv/Scripts/python -m nunchi.make_data        # word train/valid/test
.venv/Scripts/python -m nunchi.make_noise       # typo/code test sets, typo-augmented train set
.venv/Scripts/python -m nunchi.compare          # fastText models (trains ft_shape_aug.ftz) vs rules
.venv/Scripts/python -m nunchi.context          # context on sentences (fastText)
```

The key tagger trains with PyTorch. On an AMD GPU under Windows, AMD's ROCm wheels go in a separate venv (see [AMD's guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)); on CPU, `pip install torch` is enough.

```bash
.venv-rocm/Scripts/python -m nunchi.tagger train
.venv-rocm/Scripts/python -m nunchi.tagger eval
.venv-rocm/Scripts/python -m nunchi.context tagger
.venv/Scripts/python -m nunchi.demo             # try it: http://localhost:8765
```

## Layout

| File | Purpose |
|---|---|
| `nunchi/dubeolsik.py` | Hangul → 두벌식 keys, syllable shape when decoded in Korean mode |
| `nunchi/count_corpus.py` | Token counts per source: Korean words, English in Korean text, English + particle |
| `nunchi/make_data.py` | Word dataset: Korean from the corpora, English from wordfreq + real cased text |
| `nunchi/make_noise.py` | Typo and code-token test sets, typo-augmented training data |
| `nunchi/train_fasttext.py` | fastText training, quantization, frequency-weighted evaluation |
| `nunchi/eval_rules.py`, `compare.py` | Rule baselines; rules vs model vs hybrid on single words |
| `nunchi/tagger.py` | Key-level BiLSTM-CRF tagger (English + particle in one word) |
| `nunchi/context.py` | Sentence sets (incl. real mixed reviews), calibration, HMM context, lookahead |
| `nunchi/demo.py` | Local test page |
| `nunchi/typebench.py` | Real-keystroke benchmark in Notepad |

## Roadmap

- [x] Hangul → 두벌식 key conversion
- [x] fastText baseline + evaluation report
- [x] Context: calibration, HMM over previous words, one-word lookahead
- [x] Real mixed-language test set (NSMC reviews that already mix English)
- [x] Training data from real text (cased English, wider Korean, collision handling)
- [x] Per-key tagging model (switches inside a word: `MBC는`, `github에`)
- [ ] Sentence-level context model (neighbouring words, not just their language) — in training
- [ ] Simulation of real use: wrong-mode starts, forgotten switches, chains after a wrong correction
- [ ] ONNX export
- [ ] Korean-mode output length counter for Backspace (final-consonant carry-over, compound finals) + round-trip tests
- [ ] C# tray app: keyboard hook, IME control, exclusions (password fields, games, elevated windows), undo, retroactive correction
- [ ] Mixed-language test set from developer chat / code comments

## Data sources

- [NSMC](https://github.com/e9t/nsmc) (Naver movie reviews): Korean text, CC0
- [KcBERT pre-training corpus](https://github.com/Beomi/KcBERT) (news comments): Korean text, CC BY-SA 4.0. Used for training only; no data derived from it is in this repository
- [Tatoeba](https://tatoeba.org): English and Korean sentences, CC BY 2.0 FR
- [wordfreq](https://github.com/rspeer/wordfreq): English words and frequencies, data under CC BY-SA 4.0
