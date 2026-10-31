# Nunchi (gksdud)

**English** | [한국어](README.ko.md)

> Automatic Korean/English input switching for Windows: fixes what you typed in the wrong IME mode, without pressing the 한/영 key.

Type `dkssud` in English mode and it becomes `안녕`. Type `ㅗ디ㅣㅐ` in Korean mode and it becomes `hello`. The IME mode switches along with it.
The repo name `gksdud` is what "한영" looks like when typed in English mode.

> **Status: early research.** The classifier (fastText + context) and its evaluation are done. The Windows tray app does not exist yet.

## Principles

- **100% local.** No network access at all. This handles keystrokes, so trust comes first.
- **Do nothing unless confident.** A wrong correction costs more than a missed one.
- **Undo hotkey.** Words you undo are learned as exceptions and never corrected again.
- **Correct within 100ms** of the word boundary.

## How it works (planned)

```
capture keys (WH_KEYBOARD_LL)
  → buffer raw keys until a word boundary
  → classify:
      1. hard rules (password fields, games, undone words)
      2. word table (case-sensitive: Shift matters in Korean mode)
      3. fastText score, calibrated (temperature)
      4. context: 2-state HMM over previous words
  → confident now?            correct immediately
    ambiguous?                wait one word, then correct retroactively
    still below 0.99?         do nothing
  → correct: Backspace → switch IME → retype raw keys
```

Classification looks at the **raw key sequence** (`dkssud`), not the characters on screen, so input typed in either mode is handled the same way.

## Results

### On real sentences

Korean movie reviews (NSMC), English sentences (Tatoeba), and Korean sentences with English words inserted ("mixed").
These follow natural word frequencies, so short common words like 그 (`rm`), 내 (`so`) and 지 (`wl`) show up as often as they do in real typing.

"Wrong" is the share of corrections that were mistakes, correcting only at confidence ≥ 0.99.

| | No context (raw) | No context, calibrated | **Context: act now or wait one word** |
|---|---|---|---|
| Korean sentences | 1.03% wrong | 0.39% wrong | **0.05% wrong** |
| English sentences | 0% wrong | 0% wrong | 0% wrong |
| Mixed sentences | 0.76% wrong | 0.32% wrong | **0.23% wrong** |
| At language switches: corrected / wrong | 99% / 0.65% | 95% / 0.45% | 63% / 0.95% |

- **Context cuts wrong corrections on Korean text by about 8x.** Short words that are ambiguous on their own (그, 내, ㅜㅜ) get resolved by the words around them.
- **Looking one word ahead is what makes context usable.** With past words only, the first English word inside a Korean sentence gets pulled toward Korean: just 27% were corrected, and 2.7% of those were wrong. Waiting for the next word ("오늘 fork 했는데") brings that to 63% / 0.95%. Only 4% of words in Korean text need the wait.
- **Calibration matters.** fastText says p = 1.000 even when wrong. Softening its scores (temperature T=2) alone cuts wrong corrections from 1.03% to 0.39%.
- **Switch points are still the weak spot.** Half the remaining context mistakes there are `zz` (ㅋㅋ) right after an English word. A table rule for ㅋㅋ-style tokens should cover those.

### Single words: rules vs model

Each test word is scored on its own, and the test words never appear in training (split by word type). Words are weighted by how often people type them.

| Method | Size | Clean words | Words with a typo | Code tokens |
|---|---|---|---|---|
| Rule: complete syllables → Korean | 0 | 3.66% wrong | 26.8% wrong | 99.0% |
| Rule: tuned (dictionary + table) | 0 | 0.09% wrong | 25.4% wrong | 99.0% |
| fastText (typo-augmented) | 1.5MB | 0.02% wrong | 0.57% wrong | 99.1% |
| **Hybrid (table + fastText)** | 1.5MB | **0.02% wrong** | **0.52% wrong** | 99.1% |

The code-token column is plain accuracy (`getUser`, `user_id`).

- **Rules are enough for clean, common words.** The tuned rule reaches 99.91% frequency-weighted accuracy, the same as the model.
- **The value of ML is typo robustness.** One typo breaks the syllable structure and the rule calls Korean "English" (`wpoalTdjdy` = 재밌어요 with one typo). On typo'd input the model makes about 1/50 as many wrong corrections.
- **Shift carries information.** A case-insensitive table lookup turns `Tl` (씨) into the English `tl`.
- **Quantization shrinks the model from 37MB to 1.4MB** with no accuracy loss.
- **These single-word numbers are optimistic.** The 0.02% here becomes 0.39% on real sentences (calibrated, no context). The difference comes from short, frequent, ambiguous words that this word-level test under-represents.

## Reproduce

Tested with Python 3.12 on Windows. `PYTHONUTF8=1` is needed so the console can print Korean.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.make_data        # build train/valid/test
.venv/Scripts/python -m nunchi.train_fasttext   # train models + report (6 threads)
.venv/Scripts/python -m nunchi.make_noise       # typo/code test sets, typo-augmented train set
.venv/Scripts/python -m nunchi.compare          # rules vs models vs hybrid (single words)
.venv/Scripts/python -m nunchi.context          # context on real sentences
```

## Layout

| File | Purpose |
|---|---|
| `nunchi/dubeolsik.py` | Hangul → 두벌식 keys, syllable shape when decoded in Korean mode |
| `nunchi/make_data.py` | Synthetic dataset (NSMC eojeol + wordfreq English words) |
| `nunchi/make_noise.py` | Typo and code-token test sets, typo-augmented training data |
| `nunchi/train_fasttext.py` | fastText training, quantization, frequency-weighted evaluation |
| `nunchi/eval_rules.py` | Rule-only baselines |
| `nunchi/compare.py` | Rules vs model vs hybrid on single words |
| `nunchi/context.py` | Calibration + HMM context + one-word lookahead on sentences |

## Roadmap

- [x] Hangul → 두벌식 key conversion
- [x] Synthetic data generator
- [x] fastText baseline + evaluation report
- [x] Context: calibration, HMM over previous words, one-word lookahead
- [ ] PyTorch reimplementation → ONNX
- [ ] Korean-mode output length counter for Backspace (final-consonant carry-over, compound finals) + round-trip tests
- [ ] C# tray app: keyboard hook, IME control, exclusions (password fields, games, elevated windows), undo, retroactive correction
- [ ] Per-key tagging model (switches inside a word: `API를`, `React에서`)
- [ ] Realistic mixed-language test set (developer chat, code comments)

## Data sources

- [NSMC](https://github.com/e9t/nsmc) (Naver movie reviews): Korean text, CC0
- [wordfreq](https://github.com/rspeer/wordfreq): English words and frequencies, data under CC BY-SA 4.0
- [Tatoeba](https://tatoeba.org): English sentences, CC BY 2.0 FR
