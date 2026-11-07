# Nunchi

**English** | [한국어](README.ko.md)

> Automatic Korean/English input switching for Windows: fixes what you typed in the wrong IME mode, without pressing the 한/영 key.

Type `dkssud` in English mode and it becomes `안녕`. Type `ㅗ디ㅣㅐ` in Korean mode and it becomes `hello`. The IME mode switches along with it.

> **Download:** [Windows app](https://github.com/min-1870/Nunchi/releases/latest) · [Model on Hugging Face](https://huggingface.co/min-1870/nunchi-tr5n-102k)
>
> **Status: early release (0.1.0).** The app is unsigned, so Windows may warn you when you run it.
>
> - **Model:** TR5n-102K, one Transformer over the key stream (3.2M parameters, 3.8 ms a word on one CPU thread). It also handles numbers (`3개`, `gpt4`) and English with a Korean particle in one word (`MBC는`).
> - **Docs:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (current state, open problems, every experiment) · [docs/EVALUATION.md](docs/EVALUATION.md) (eval sets, terms, how differences are judged) · [models/registry.json](models/registry.json) (every model and its settings).

## Principles

- **100% local.** No network access at all. This handles keystrokes, so trust comes first.
- **Do nothing unless confident.** A wrong correction costs more than a missed one.
- **Undo hotkey.** Words you undo are learned as exceptions and never corrected again.
- **Correct within 100ms** of the word boundary.

## How it works

```
capture keys (WH_KEYBOARD_LL)
  → key stream of the line so far, spaces included          ... rmfjgek dlqkt rkqtl  MBCsms
  → TR5n: one Transformer over the stream, a label per key   ... k k k k k k k  e e e k k k
  → in the middle of a word: switch the IME early only when very sure (>= 0.99)
  → at Space: convert the word when sure (>= 0.85); split it where its best path
      switches inside (>= 0.9):                              MBC + 는
  → one word later: judge the previous word again (>= 0.8); a word the app changed may go back once
  → per user: short ambiguous words (so / 내) lean toward what this user keeps
  → correct: Backspace → switch IME → retype raw keys
```

Everything works on the **raw key sequence** (`dkssud`), not the characters on screen, so input typed in either mode is handled the same way. Labels for training come for free: text written in Hangul is Korean, text written in Latin letters is English. The thresholds are the app's, tuned once for the shipped model on dev with the app's own rule and checked on test once.

## Results

Dev halves of held-out sets: sentences no model was trained on; the test halves are kept for a final check. Every word is decided, as an input method has to: at Space, then revised when the model is at least 95% sure three words later and gives the same answer two and five words later. Numbers are % of words wrong in the end, per cell (how the sets are built: [docs/EVALUATION.md](docs/EVALUATION.md)). TR5n-102K is trained twice with different random seeds; both are shown.

| Cell | Dev words | TR3-51K | TR4-51K | **TR5n-102K** seed 0 | **TR5n-102K** seed 1 |
|---|---|---|---|---|---|
| Korean → English, start / middle / end of the English part | 1,011 / 4,368 / 2,993 | 3.04 / 2.27 / 4.04 | 2.15 / 1.18 / 2.35 | 2.30 / 1.31 / 2.24 | 2.23 / 1.21 / 2.11 |
| English → Korean, start / middle / end of the Korean part | 1,595 / 2,090 / 1,465 | 1.26 / 5.42 / 12.16 | 0.22 / 1.78 / 4.64 | 0.29 / 2.45 / 5.20 | 0.29 / 1.77 / 4.54 |
| Korean only / English only (main sets) | 60,267 / 39,011 | 0.12 / 0.04 | 0.11 / 0.04 | 0.10 / 0.03 | 0.09 / 0.05 |
| English words that are Korean words too (`so` = 내, `go` = 해) | 3,087 | 14.16 | 7.22 | 6.32 | 6.48 |
| Korean words that are English words too (해 = `go`) | 2,998 | | 1.95 | 2.06 | 1.99 |
| Single jamo in Korean text (ㅋ, ㅠ) | 2,739 | | 7.73 | 7.87 | 7.33 |
| English words typed with a Korean marker's keys (`gee` = ㅎㄷㄷ) | 547 | 19.41 | 11.16 | 12.09 | 12.09 |
| Words with both languages and no space (`QR코드`, `MBC는`) | 1,513 | | | 6.79 | 7.24 |
| Number words in English text (`gpt4`, `5km`) | 2,993 | 3.38 | 0.73 | 0.33 | 0.40 |
| A typo in the word / Shift still held (`RK` for 까) | 14,437 / 3,842 | | | 1.08 / 0.22 | 1.05 / 0.22 |
| Upper-case acronyms (`EU`, `WHO`) | 1,504 | | | 0.20 | 0.30 |
| One word alone: Korean / English / English + particle (`apk의`) | 2,000 / 1,993 / 965 | | | | 0.50 / 1.91 / 7.46 |

Empty cells: the set did not exist when that model was scored. On the 133,050 words every compared model was scored on: TR4-51K 879 wrong, TR5n-102K 840 (seed 0) and 806 (seed 1). Two seeds of one recipe differ by about 35-45 words there, so smaller differences are not results.

**In real use:** on 17,876 words the author typed into Claude Code over 32 days, with the app's thresholds, 0.79% of words come out wrong (about 1 in 126; 1.16% with the earlier thresholds). Most of them are the first word of a line (`so` → 내) and lower-case English with a particle (`wsl로`).

**Speed:** 3.8 ms a word on one thread of a Ryzen 5 5600 (ONNX Runtime, 3.2M parameters).

## What we learned

The six lessons that shaped the model. All of them, in order, with the numbers: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#what-each-step-showed).

- **Score sentences, not words.** The first model made 0.02% wrong corrections on single words and 1.03% on real sentences: short, ambiguous words (그 `rm`, 내 `so`) are rare in word lists and common in text.
- **Fixing the data beat every context trick.** A filter meant to drop Korean typed in the wrong mode was deleting exactly the hard cases (`so`, `go`, `B급`, single letters). Deciding by word frequency instead brought them back; real mixed-sentence errors went from 76 to 52 with the next data step.
- **Label every key, over the whole key stream.** A key-level tagger handles words that switch language inside (`github` + `에`); one Transformer over the key stream, spaces included, beat a model over word summaries.
- **Make training see what typing sees.** Stacked attention lets a word reach several words ahead through its neighbours, while typing shows at most one; training now labels only the word being decided.
- **Seeds move results more than most settings.** The same recipe with another seed differs by 44 words at 51k steps, so every change is trained twice and judged per cell.
- **What is left fails in both directions.** The same strings fail both ways in different sets (ㅎㅎ read as `gg`, `gee` read as ㅎㄷㄷ; 해 read as `go`, `go` read as 해), and every data change aimed at them only moved errors from one side to the other.

## Reproduce

Tested with Python 3.12 on Windows; the commands are for Git Bash. `PYTHONUTF8=1` is needed so the console can print Korean.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert data/raw/mc4 data/raw/wiktionary data/raw/imdb data/raw/goemotions data/raw/wikitext103 data/raw/reddit_big
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
curl -L -o data/raw/mc4/c4-ko-validation.json.gz https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ko-validation.tfrecord-00000-of-00001.json.gz
curl -L -o data/raw/wiktionary/kaikki-korean.jsonl https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl
for s in train test; do curl -L -o data/raw/imdb/$s.parquet https://huggingface.co/api/datasets/stanfordnlp/imdb/parquet/plain_text/$s/0.parquet; done
for s in train validation test; do curl -L -o data/raw/goemotions/$s.parquet https://huggingface.co/api/datasets/google-research-datasets/go_emotions/parquet/simplified/$s/0.parquet; done
for i in 0 1; do curl -L -o data/raw/wikitext103/train_$i.parquet https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-103-raw-v1/train/$i.parquet; done
for s in buildapc gaming relationship_advice todayilearned; do curl -L -o data/raw/reddit_big/${s}_0.parquet https://huggingface.co/api/datasets/HuggingFaceGECLM/REDDIT_comments/parquet/default/$s/0.parquet; done
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

The v3 training pools and the eval sets are built with the commands in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) ("How to run"). The models train with PyTorch. On CPU, `pip install torch` into the same venv is enough; on an AMD GPU under Windows, install AMD's ROCm wheels instead (see [AMD's guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)). TR5n-102K (about 2.5 hours on a Radeon RX 9060 XT), then its scores:

```bash
.venv/Scripts/python -u experiments/run.py experiments/configs/tr5n_102k.json
.venv/Scripts/python -u experiments/system_eval.py TR5n-102K=TR5n-102K
```

A stopped run continues from its checkpoint with `NUNCHI_RESUME=1`.

**Versions used:** [requirements-train.txt](requirements-train.txt) (training, ROCm), [requirements-export.txt](requirements-export.txt) (ONNX export), and [data_versions.sha256](data_versions.sha256) for every raw file (downloaded 2026-10-31 to 2026-11-03). Tatoeba and kaikki.org change over time, so a later download can differ from these; a new training run also differs a little from any other, as data workers and GPU kernels are not deterministic.

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
| `nunchi/stream_tagger.py` | Stream Transformer (TX, RV, TR), training with resume, `decide()` |
| `nunchi/models.py` | Load a model by name with its context and thresholds from `models/registry.json` |
| `nunchi/eval_sentence.py` | Decide every word of a sentence (now / one word later), split words |
| `nunchi/export_onnx.py` | ONNX export for the app |
| `nunchi/context.py` | HMM context experiments, sentence sets incl. real mixed reviews |
| `nunchi/demo.py`, `typebench.py` | Local test page, real-keystroke benchmark |
| `experiments/` | Eval sets with numbers in place, per-category tables, bias curves, `run.py` for training runs from a config |
| `experiments/system_eval.py` | Scores a model on every dev / test set with the revision rule |
| `experiments/logs/app_tune.py`, `early_tune.py` | The app's thresholds, scored with the app's own rule |
| `models/registry.json` | Every model: parent, settings, status |
| `docs/` | Experiment log and evaluation guide |
| `app/` | Windows app |

## Roadmap

- [x] Hangul → 두벌식 keys, fastText baseline, HMM context
- [x] Real mixed-language test sets (NSMC reviews, mC4 web text), held-out and de-duplicated
- [x] Per-key tagging model (switches inside a word: `MBC는`, `github에`)
- [x] Sentence model over the key stream, training that matches typing
- [x] Data: frequency-based wrong-mode filter, mC4, translation swaps, lower-case augmentation, Shift-held typos
- [x] ONNX export
- [x] One model that decides and revises (TR)
- [x] Seed-to-seed variance, so small differences can be judged
- [x] Longer training (102k steps; 204k no better)
- [x] Eval sets per switching point, typos, ambiguous words both ways, one-word input
- [x] Real typing, measured (one person's Claude Code text)
- [ ] Switching in the middle of a word without flashes (`wha` → 좀)
- [ ] Typing from more people, to weigh errors by real use
- [x] Windows app: keyboard hook, IME control, exceptions, undo, revision, habits (0.1.0)

## Data sources

- [NSMC](https://github.com/e9t/nsmc) (Naver movie reviews): Korean text, CC0
- [KcBERT pre-training corpus](https://github.com/Beomi/KcBERT) (news comments): Korean text, CC BY-SA 4.0
- [mC4](https://huggingface.co/datasets/allenai/c4) Korean (web text from Common Crawl): ODC-BY, and the Common Crawl terms of use
- [Tatoeba](https://tatoeba.org): English and Korean sentences, CC BY 2.0 FR
- [Wiktionary](https://en.wiktionary.org) Korean entries via [kaikki.org](https://kaikki.org) (Tatu Ylonen, "Wiktextract: Wiktionary as Machine-Readable Structured Data", LREC 2022): CC BY-SA 4.0 / GFDL
- [wordfreq](https://github.com/rspeer/wordfreq): English words and frequencies, data under CC BY-SA 4.0
- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (Wikipedia articles): English text, CC BY-SA. Used in training.
- [REDDIT_comments](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) (four subreddits): English text, no license stated. About 0.7% of training samples, and some eval sets.
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Reddit comments): Apache-2.0 on its card. Eval sets and a few training lines.
- [IMDB reviews](https://huggingface.co/datasets/stanfordnlp/imdb): license "other", no terms on its card. Eval sets and a few training lines.

No data derived from these sources is in this repository. Trained weights are not in the repository either.

## License

The code is under the [MIT License](LICENSE).

The trained weights are released separately with the app, as is, for trying the app and for research. They were trained on the sources listed above, and those sources' terms apply. Because part of that data is CC BY-SA, the weights are offered under CC BY-SA 4.0 as far as the author is able to do so. No rights are granted in third-party text.

Not every source states a license. If you hold rights in any of this data and object to its use, please open an issue: the model will be retrained without it or taken down.
