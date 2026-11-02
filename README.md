# Nunchi (gksdud)

**English** | [한국어](README.ko.md)

> Automatic Korean/English input switching for Windows: fixes what you typed in the wrong IME mode, without pressing the 한/영 key.

Type `dkssud` in English mode and it becomes `안녕`. Type `ㅗ디ㅣㅐ` in Korean mode and it becomes `hello`. The IME mode switches along with it.
The repo name `gksdud` is what "한영" looks like when typed in English mode.

> **Status: research.** The reference model (TR3-long) is one stream Transformer that decides each word at Space and revises it as more words come; it reads numbers (`3개`, `gpt4`) and English with a Korean particle inside one word (`MBC는`). The earlier pair TX4 (decide) + RV1 (revise) is described below. A Windows app is in progress in `app/`. Current experiments, what each showed and what comes next: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); eval sets and how differences are judged: [docs/EVALUATION.md](docs/EVALUATION.md).

## Principles

- **100% local.** No network access at all. This handles keystrokes, so trust comes first.
- **Do nothing unless confident.** A wrong correction costs more than a missed one.
- **Undo hotkey.** Words you undo are learned as exceptions and never corrected again.
- **Correct within 100ms** of the word boundary.

## How it works

```
capture keys (WH_KEYBOARD_LL)
  → key stream of the last words, spaces included          ... rmfjgek dlqkt rkqtl  MBCsms
  → TX4: one Transformer over the stream, a label per key   ... k k k k k k k  e e e k k k
      sees the 3 words before the word and, at most, the one after it
  → at Space: decide the word (wait one more word if unsure)
  → a word whose best path switches inside it is split:     MBC + 는
  → reviser (RV): once more words are there, may change an earlier word,
      only if it is sure, gives the same answer as words keep coming, and TX4 was unsure
  → correct: Backspace → switch IME → retype raw keys
```

Everything works on the **raw key sequence** (`dkssud`), not the characters on screen, so input typed in either mode is handled the same way. Labels for training come for free: text written in Hangul is Korean, text written in Latin letters is English.

## Results

Dev halves of held-out sets: sentences no model was trained on, with any sentence that also appears in the training text removed. The test halves are kept untouched for a final check. Every word is decided (as an input method has to): at Space, or one word later when the model is below 99% sure; a word is split when its best path is at least 70% sure. (Tuned for TX4 afterwards: waiting from below 70% instead of 99% gives 50 instead of 52 on the first row, with fewer words rewritten.)

**Models:** HMM = key-level word tagger + 2-state HMM over words (the previous approach). TX3, TX4 = the stream Transformer (see [models/registry.json](models/registry.json) for every model and its settings).

### Wrong words

| Set | Words | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| Real mixed sentences (NSMC reviews + web) | 15,032 | 105 | 63 | **52** |
| Real mixed sentences + typos | 15,032 | 183 | **90** | 91 |
| Web sentences with English + particle words | 16,797 | 171 | 97 | **87** |
| English sentences with a word swapped for Korean (generated) | 7,369 | 79 | 31 | **27** |
| Korean sentences with a word swapped for English (generated) | 9,600 | 41 | 22 | **9** |
| Korean only | 12,819 | | **1** | 3 |
| English only | 14,216 | | 4 | 4 |

Typos: each word gets one keyboard typo (neighbour key, dropped, doubled or swapped key) with probability 0.15. The generated sets are built like part of the training data, so they flatter the models trained on it; the real sets are the ones to trust.

### Where the language switches (real mixed sentences)

| | Count | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| Entering English inside Korean | 1,141 | 42 | 37 | **30** |
| Back to Korean after English | 1,266 | 18 | 2 | **1** |
| English + Korean word split exactly (`MBC는`, web set) | 1,694 | 114 | 83 | **76** |

Most of TX4's remaining entry errors are 1-3 letter tokens (`T`, `b`, `gg`): 27 of 30.

### Revising a word later

TX4 decides at Space; a reviser may change a word after 2-5 more words, only when it is sure enough, gives the same answer at 2, 3 and 5 words, and TX4 itself was below 99%. RV0 learns to decide and to revise; RV1 learns only to revise (no label on the word just typed). Fixed / broken against TX4 deciding at Space:

| Set | TX4 wrong | RV0, sure ≥ 90% | **RV1, sure ≥ 90%** | RV0, sure ≥ 99% | **RV1, sure ≥ 99%** |
|---|---|---|---|---|---|
| Words whose keys are common in both languages (`so` / 내, `go` / 해), English truth | 112 of 1,090 | +17 / −1 | +26 / −3 | +4 / −0 | **+9 / −0** |
| Same, Korean truth | 70 of 1,017 | +13 / −0 | +16 / −3 | +9 / −0 | +8 / −0 |
| Real mixed sentences | 56 of 15,032 | +9 / −1 | +12 / −4 | +2 / −0 | **+5 / −0** |
| Web sentences with English + particle words | 97 of 16,797 | +10 / −1 | +16 / −0 | +3 / −0 | +3 / −0 |

Training only to revise makes the reviser more useful: at 99% it fixes 25 words with none broken (RV0: 18), at 90% it fixes more but breaks more. The gains are still small next to the remaining errors.

### Speed

About 17 ms per word on one CPU thread (3.2M parameters).

## What we learned

The order things happened in, and what each step showed.

**1. Single-word scores were far too optimistic.** The first fastText model made 0.02% wrong corrections on single words but 1.03% on real sentences. Short, frequent, ambiguous words (그 `rm`, 내 `so`) are rare in a word list and common in real text. From then on every result was measured on sentences.

**2. Calibration and one-word lookahead made context usable.** Softening fastText's overconfident scores cut wrong corrections from 1.03% to 0.39%; a 2-state HMM over words brought Korean sentences to 0.05% but pulled English inside Korean toward Korean. Waiting one word on ambiguous words fixed most of that.

**3. The synthetic mixed set hid the real problem.** Random dictionary words inserted into sentences are long, rare and easy. Real English inside Korean is short and often uppercase (`TV`, `OST`, `CG`). A 4-state HMM looked best on synthetic data and worst on real data.

**4. Fixing the data beat every context trick.** Rebuilding the training words from real text (cased English, English found in Korean text, collisions sent to the far more common side, Korean typed in the wrong mode removed) cut wrong corrections on real mixed reviews from 0.68% to 0.26% without any context.

**5. An HMM has a hard ceiling.** It can move a word's odds by at most s/(1−s), so a word the model is 99% sure of can never be overturned. What is left are words that are real in both languages (`so` / 내, `go` / 해), which need the neighbouring words themselves, not just their language.

**6. A key-level tagger removes the one-language-per-word assumption.** Labelling every key makes `githubdp` → `github` + `에` just a path that switches once.

**7. Wrong corrections cost more than they look.** The app switches the IME mode after a correction, so one wrong correction makes the following words come out wrong too.

**8. A sentence model over the key stream beats a model over words.** A BiLSTM over word summaries forgot what the word tagger knew (particles). One Transformer over the whole key stream, spaces included, learned both, and became the main model.

**9. Stacked attention leaks the future.** With a mask that lets a word see one word ahead, four layers let it reach four words ahead through its neighbours, while typing never shows more than one. Training now labels only the word being decided (the last one, or the one before it with the last as lookahead), so training sees exactly what typing does.

**10. Three words of left context are enough.** A model trained on the three previous words matched one trained on the whole sentence. Waiting for the next word changes little: a few words either way.

**11. A data filter was deleting exactly the hard cases.** Dropping every sentence with Latin whose keys spell a Korean word (to remove Korean typed in the wrong mode) also removed `so`, `go`, `do`, `work`, `B급` and almost every single letter: in the kept text, short English inside Korean never appeared, so the model turned it into Korean. Deciding by frequency instead (English zipf vs the Korean word's zipf; capitals Shift cannot type in Korean mean English) brought these back. Real mixed-sentence errors went from 76 to 62, and from 63 to 52 with the next step.

**12. Meaningful synthetic data beats random insertion.** Swapping a word for its translation (`the weather` → `the 날씨`, `날씨가` → `weather가`; loanwords like `미팅` / `meeting` are just one more translation) replaced inserting random words, which only helped on tests built the same way. Lower-casing 30% of English inside Korean text taught that people type acronyms in lower case (`jsa`, `apk`).

**13. Single letters are a choice, not an error to fix.** `T`, `b`, `s` inside Korean text are usually Korean jamo in real text (ㅆ, ㅠ, ㄴ) but sometimes English. A rule "never convert one letter" fixed 15 such words on real sentences but broke as many Korean comments with spaced jamo (`ㅅ ㅂ`, `ㅁ ㅊ`), so it is left to the model.

**14. Revising later works only as a separate, careful step.** Training one model to decide now and revise later made its decision at Space worse. A separate reviser with strict limits fixes a few words and almost never breaks one.

**15. Things that did not help.** A higher learning rate with a shorter schedule (clearly worse); code-level speedups (the encoder is 95% of the time); `torch.compile` (3%); WSL instead of native Windows (11% faster, not worth the switch yet).

**How far to trust small differences:** the dev sets hold a few dozen errors per model and every model is trained once. Differences under about ten words are within noise.

## Reproduce

Tested with Python 3.12 on Windows. `PYTHONUTF8=1` is needed so the console can print Korean.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert data/raw/mc4 data/raw/wiktionary
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
curl -L -o data/raw/mc4/c4-ko-validation.json.gz https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ko-validation.tfrecord-00000-of-00001.json.gz
curl -L -o data/raw/wiktionary/kaikki-korean.jsonl https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.count_corpus     # word / Latin / Latin+Hangul counts per source
.venv/Scripts/python -m nunchi.make_data        # word train/valid/test
.venv/Scripts/python -m nunchi.make_noise       # typo/code test sets, typo-augmented train set
.venv/Scripts/python -m nunchi.sentence_data    # sentences per source (data/sent)
.venv/Scripts/python -m nunchi.web_mixed train  # mC4 train shards 0-9 (downloaded) -> mixed sentences
.venv/Scripts/python -m nunchi.web_mixed        # mC4 validation -> web eval set
.venv/Scripts/python -m nunchi.web_mixed particle
.venv/Scripts/python -m nunchi.translations     # translation-swapped sentences, both ways
.venv/Scripts/python -m nunchi.ambiguous_eval   # eval set of words whose keys are common in both languages
```

The models train with PyTorch. On an AMD GPU under Windows, AMD's ROCm wheels go in a separate venv (see [AMD's guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)); on CPU, `pip install torch` is enough. TX4 (about 25 minutes on a Radeon RX 9060 XT):

```bash
NUNCHI_KO=0.25 NUNCHI_EN=0.20 NUNCHI_WORDS=0.15 NUNCHI_KMIX=0.113 NUNCHI_WEB=0.187 \
NUNCHI_TRANS_KO=0.05 NUNCHI_TRANS_EN=0.05 NUNCHI_SYNTH=0 NUNCHI_LOWER_P=0.3 \
NUNCHI_ONE_WORD=1 NUNCHI_LEFT=3 NUNCHI_SAVE_LAST=1 NUNCHI_OUT=models/tx4.pt \
.venv-rocm/Scripts/python -m nunchi.stream_tagger train tx4_log.txt 17000 2125 1000 1024 3
```

A stopped run continues from its checkpoint with `NUNCHI_RESUME=1`. The benchmark scripts behind the tables are not in the repository yet.

## Layout

| File | Purpose |
|---|---|
| `nunchi/dubeolsik.py` | Hangul → 두벌식 keys, syllable shape when decoded in Korean mode |
| `nunchi/count_corpus.py` | Token counts per source: Korean words, English in Korean text, English + particle |
| `nunchi/make_data.py`, `make_noise.py` | Word dataset and typo sets |
| `nunchi/sentence_data.py` | Sentences per source; the frequency-based filter for Korean typed in the wrong mode |
| `nunchi/web_mixed.py` | mC4 Korean: training pool from the train split, eval sets from the validation split |
| `nunchi/translations.py` | Translation-swapped sentences from Wiktionary glosses (loanwords included) |
| `nunchi/loanwords.py` | Loanword ↔ English pairs from Wiktionary etymologies, homographs removed |
| `nunchi/ambiguous_eval.py` | Eval set of words whose keys are common in both languages |
| `nunchi/tagger.py` | Key-level BiLSTM-CRF word tagger (HMM baseline), CRF, English + particle words |
| `nunchi/sentence_tagger.py` | Training data sampler (shares by role), LSTM sentence model |
| `nunchi/stream_tagger.py` | Stream Transformer (TX, RV), training with resume, `decide()` |
| `nunchi/models.py` | Load a model by name with its context and thresholds from `models/registry.json` |
| `nunchi/eval_sentence.py` | Decide every word of a sentence (now / one word later), split words |
| `nunchi/export_onnx.py` | ONNX export for the app |
| `nunchi/context.py` | HMM context experiments, sentence sets incl. real mixed reviews |
| `nunchi/demo.py`, `typebench.py` | Local test page, real-keystroke benchmark |
| `experiments/` | Eval sets with numbers in place, per-category tables, bias curves, `run.py` for training runs from a config |
| `docs/` | Experiment log and evaluation guide |
| `app/` | Windows app (in progress) |

## Roadmap

- [x] Hangul → 두벌식 keys, fastText baseline, HMM context
- [x] Real mixed-language test sets (NSMC reviews, mC4 web text), held-out and de-duplicated
- [x] Per-key tagging model (switches inside a word: `MBC는`, `github에`)
- [x] Sentence model over the key stream (TX4), training that matches typing
- [x] Data: frequency-based wrong-mode filter, mC4, translation swaps, lower-case augmentation
- [x] ONNX export
- [x] Reviser: dedicated model (RV1)
- [ ] One model that decides and revises as well as TX4 + RV1
- [ ] Eval set from real typing logs (the sets above are reviews and web text)
- [ ] Seed-to-seed variance, so small differences can be judged
- [ ] Longer training (loss was still falling at 17,000 steps)
- [ ] Benchmark scripts in the repository
- [ ] Windows app: keyboard hook, IME control, exclusions, undo, revision

## Data sources

- [NSMC](https://github.com/e9t/nsmc) (Naver movie reviews): Korean text, CC0
- [KcBERT pre-training corpus](https://github.com/Beomi/KcBERT) (news comments): Korean text, CC BY-SA 4.0
- [mC4](https://huggingface.co/datasets/allenai/c4) Korean (web text from Common Crawl): ODC-BY, and the Common Crawl terms of use
- [Tatoeba](https://tatoeba.org): English and Korean sentences, CC BY 2.0 FR
- [Wiktionary](https://en.wiktionary.org) Korean entries via [kaikki.org](https://kaikki.org) (Tatu Ylonen, "Wiktextract: Wiktionary as Machine-Readable Structured Data", LREC 2022): CC BY-SA 4.0 / GFDL
- [wordfreq](https://github.com/rspeer/wordfreq): English words and frequencies, data under CC BY-SA 4.0

No data derived from these sources is in this repository; the scripts above rebuild it. Trained weights are not in the repository either; because they are trained on CC BY-SA data, published weights would carry CC BY-SA 4.0.
