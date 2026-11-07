# Evaluation: sets and how results are judged

All sets are split into **dev** and **test** halves by a hash of the sentence; choices (thresholds, data
settings) are made on dev, final comparisons on test. Sentences that also occur in any training pool
(v1, web50, v2, v3) are dropped (`experiments/evalsets.py`, `in_training`).

## Terms

Words used in this file, EXPERIMENTS.md and the READMEs.

| term | meaning |
|---|---|
| dev / test half | every set is split by a hash of the sentence; choices are made on dev, test is used once for a chosen model |
| cell | one row of a results table: one set, or one slice of it (KO>EN end), scored on its own |
| KO>EN, EN>KO | Korean text switching to English, and the other way. **start / middle / end**: where the other-language part sits in the sentence; **after**: the first word back in the base language |
| main sets | whole sentences from one source each, every word scored: mostly one language, so they measure "don't switch wrongly" |
| guard set | a cell that a change aimed elsewhere could break (single jamo guards ambiguous English); watched so a gain on one side is not paid for on the other |
| key-identical pair | two words with the same keys, one in each language (go / 해, gg / ㅎㅎ). **Key collision**: a real English word whose keys are a Korean marker (mm = ㅡㅡ) |
| single jamo | a lone consonant or vowel used as a word in Korean text (ㅋ, ㅠ) |
| 초성체 | Korean written as initial consonants only (ㅅㄱ for 수고); often typed in upper or lower case Latin keys |
| fused word | both languages in one word with no space (QR코드); **English + particle** is the common kind (MBC는) |
| final error | wrong after the revision rule: what stays on screen. **late fixes**: wrong at Space, right after revising. **broke**: right at Space, wrong after revising |
| eval rule / app rule | the fixed rule models are compared with (decide at Space, revise at 95% sure three words later) and the app's own rule (switch mid-word, decide at Space, revise one word later); see below |
| shared cells | the 133,050 dev words every compared model was scored on (Shift-held typos left out); used for totals |
| seed difference | how far two runs of one recipe with different random seeds land apart; a change smaller than it is not a result |
| lean | a change that moves errors from one side of the Korean / English trade-off to the other, along the baseline's bias curve, without learning anything |
| 10k screen | short training runs (10k steps, two seeds each) used to pick changes before a long run |
| knobs | data and training settings a run changes, one letter each (E, L, S, A, N, K, J, X, H, T, W, F); defined in `_scheme` of `models/registry.json` |
| pools v1 / v2 / v3 | versions of the training text: v1 dropped numbers, v2 and v3 keep them (EXPERIMENTS.md logs 4-5) |
| kmix | the training pool of Korean-source lines (NSMC, KcBERT) that hold Latin letters |
| en_marked | English lines with a Korean word or marker put in (`nunchi/en_marked.py`, EXPERIMENTS.md log 10) |
| Zipf | word frequency on the [wordfreq](https://github.com/rspeer/wordfreq) scale: log10 of uses per billion words (3 = once per million) |
| pp | percentage points: 2.0% → 1.5% is −0.5 pp |
| 2 SE | twice the standard error of a difference; about the 95% line for noise |

## Sets (from 2026-11-03)

Built the usual way for a mixed-domain system: a main set with the same amount from each source (macro
average over sources, as Paloma does per domain[^paloma]), plus a set scored only at the points of interest
(words where the language changes, as PIER does for code-switched speech[^pier]). Dev halves choose, test halves report.

| set | file / builder | what it measures | caveats |
|---|---|---|---|
| **main** | `data/eval/source_sets.jsonl`, `experiments/source_sets.py` | 40k words from each source (NSMC, KcBERT, mC4 web, IMDB, Reddit), held out from training, sentences as they come; every word scored | mostly one language per sentence: measures "don't switch wrongly" |
| **switching** | `data/eval/switch_sets.jsonl`, `experiments/switch_sets.py` | only the words where the language changes: KO>EN / EN>KO × start / middle / end, the word after, number words in Korean / English sentences; 4,000 lines per source | real for Korean-base text (NSMC + KcBERT, web); English-base text with Korean is synthetic (IMDB / Reddit with a word translated or an interjection put in) |
| KO>EN end, more points | `data/eval/switch_end_extra.jsonl`, `experiments/end_extra.py` | the thinnest real cell (a Korean sentence ending in an English word): 480 → 2,993 dev points | joined with the switching set's own points for that cell |
| key collisions | from 2026-11-03 `data/eval/collision_set.jsonl`, `experiments/collision_sets.py` (before: `collision_eval.jsonl`, 186 words) | real English lines (Reddit) with words whose keys equal a Korean marker (mm = ㅡㅡ, z = ㅋ, h = ㅗ, gee = 허): up to 60 words per key and half, 12 keys; the cell is the mean over keys | WW, gg, bb, zz: no unused lines left; the seed difference is large (one key can swing 20 of 60) |
| number words, English | `data/eval/num_en_set.jsonl`, `experiments/num_en_sets.py` | words with digits and letters in real WikiText / Reddit lines: 2,993 dev / 3,008 test points; replaces the switching set's 241 synthetic points in that cell for runs scored from 2026-11-03 | mostly ordinals and decades (7th, 1990s) |
| ambiguous English | `experiments/amb_en_eval.py`; more lines: `data/eval/amb_en_set.jsonl`, `experiments/amb_en_sets.py` | English words that are also common Korean words (so = 내, to = 새, a = ㅁ, go = 해) in real Korean-base sentences: the switching set's (606 dev words) and, from 2026-11-03, held-out KcBERT comments (after line 2.4M) and mC4 validation lines: 3,087 dev / 3,087 test words | mostly single letters as Korean comments use them (a, o, c, f, s, k: 'C발', 'F-35', 'KT SK'); to / so / go are a few hundred. Runs scored before 2026-11-03 used the 606 words only |
| single jamo | `data/eval/jamo1_set.jsonl`, `experiments/jamo1_sets.py`; review `data/eval/jamo1_review.jsonl` | one-jamo Hangul words in Korean text (ㅋ, ㅠ, ㄴ: keys z, b, s), the other side of ambiguous English: held-out KcBERT (after line 2.4M) and mC4 validation, 2,739 dev words | the web part was read in full on 2026-11-04, blind to model output: list markers, separators and text about Hangul (342 + 121 of 1,000) are left out on both halves (EXPERIMENTS.md log 24) |
| Korean words that read as English | `data/eval/amb_ko_set.jsonl`, `experiments/amb_ko_sets.py` | Korean words whose keys are a common English word (해 = go, 내 = so, 새 = to; English Zipf 4.5+), the guard for pushing short English toward English: 2,998 dev words | rarer English readings (눈 = sns, 렌 = fps) are not in it |
| upper-case acronyms | `data/eval/acro_set.jsonl`, `experiments/acro_sets.py` | English in Korean text with the Shift-held shape (EU, RM, TM, WHO): the guard for the Shift-held augmentation, 1,504 dev words | a few errors per model: separates nothing smaller than ~1 per 1,000 |
| fused words | `data/eval/fused_set.jsonl`, `experiments/fused_sets.py` | words holding both languages with no space (절충trade, km리뷰, 채널A, MBC는): the typed text must split at the right key, 1,513 dev words | spread over many words: few repeat |
| typos | `data/eval/typo_set.jsonl`, `experiments/typo_sets.py` | the main set's sentences with typos put in, scored on the typo'd words: one keyboard typo (`typo_pos`, 14,437 dev words) or Shift still held after a double consonant (`typo_shift`, RK = 까, 3,842) | synthetic typos; Shift-held is solved since the augmentation (77% → 0.2%) |
| one word alone | `data/eval/word1_set.jsonl`, `experiments/word1_sets.py`, from 2026-11-05 | whole lines of one word: Korean (대박, ≤ 5 syllables, 2,000 dev), English (thanks, FYI, 1,993), English + particle typed alone (apk의, LG가, 965); the guard for single-word samples | runs scored before 2026-11-05 have no such cells |

Per-run scores: `experiments/score_one.sh NAME:file` (all of the above) → `experiments/logs/knobs/`,
`collision_all.json`, `amb_en_scores.json`, `end_extra_scores_all.json` (at Space only); with the revision
rule, the way runs are scored from 2026-11-03: `experiments/system_eval.py [--test]` → `experiments/logs/systems/`.
Not in git.

The key-collision test half is `data/eval/collision_test.jsonl` (`experiments/collision_test.py`): 352 words
from WikiText-103 / Reddit lines not used anywhere else, three times the dev words per key where there were
enough. WW, gg, bb and a few rarer keys had no lines left, so the test half has no WW (68 of the 186 dev words)
and is not comparable with dev. Its lines are also easier: most come from WikiText, where h / mm / z
follow a number as units or model names (65 km h, 300 mm, z68), while the dev lines are casual Reddit text
("ha h ha h", "k r z tt"); TR4-51K gets 12% of test wrong and 48% of the dev words other than WW.

Sets before 2026-11-03 (still used by `cat_table.py`):

| set | file / builder | what it measures | caveats |
|---|---|---|---|
| real mixed, numbers in place | `data/eval/real_natural.jsonl`, `experiments/natural_sets.py` | NSMC test reviews + mC4 validation web sentences with Korean and English, numbers kept as context; scored on the original words | **main set**. Mostly Korean (87% of words) |
| ambiguous words, natural context | `data/eval/amb_natural.jsonl`, `experiments/amb_natural.py` | words whose keys read as a word in both languages (해 = go, so = 내), sentences drawn as they come from mC4 train shards 55-59 | **main set for ambiguous words**; key frequencies as in real text |
| number words | `data/sent/web_code_eval.txt`, `python -m nunchi.web_mixed code` | words with digits (3개, gpt4, 5km) and Korean+English words (채널A), mC4 validation | scored by output text: digits type the same in either mode |
| English + particle | `data/sent/web_particle_eval.txt` | English word + Korean particle in one word (SF는) must split at the right key | numbers removed (v1 build) |
| ambiguous-key set | `data/sent/ambiguous_eval_v3.jsonl`, `python -m nunchi.ambiguous_eval v3` | ambiguous words, each key capped at 150 sentences | **hard contexts**: rates are inflated (Korean 16% here vs 0.5% natural). Reference only |
| old real / ambiguous sets | `web_mixed_v2.txt`, NSMC via `context.real_mixed`, `ambiguous_eval_v2.jsonl` | the sets used before 2026-11-02 | numbers removed: favours models trained on v1 pools |

The gaps these left (jamo-only words, English-based sentences with Korean, too few entry cases) are what the
switching set and the extra end points fill.

## Sets for looking, not for choosing

| set | file / builder | what it is for |
|---|---|---|
| held-out web text | `data/eval/edge_web.jsonl`, `experiments/edge_mine.py` | 30,000 lines with a Latin or digit token from mC4 ko validation + train shards 50-59 (at most 40 a site, 5 a token), cleaned as the eval sets, not as training: which word shapes fail that the sets miss (log 34) |
| the author's typing | `data/eval/my_typing*.jsonl`, `experiments/logs/my_typing.py` (personal data: neither is published) | 17,876 words typed into Claude Code (log 33): the error rate in real use and the share of each word kind. Measured once per shipped setting; never tuned on |

## The app's rule

The app acts differently from the eval rule (`app/core/src/typing.rs`): it switches a word in the middle once the
word so far is sure enough, decides at Space (act, split), judges a word once more one word later (revise; a word
it changed itself may go back once), and keeps a per-user habit prior. Its thresholds are tuned on dev with that
rule, once per shipped model, and checked on test once: `experiments/logs/app_tune.py` (boundary and one word
later: wrong / broke / missed, best setting for each cost of a broken word) and `experiments/logs/early_tune.py`
(the switch in the middle). Model comparisons keep the eval rule.

## Scoring

- A word is right when the text it types equals the text written (`experiments/textcheck.py`): a split word
  (SF는) must split at the right key; digits never count against a model.
- Decided at Space (no word after it) unless a table says "revising": then the rule in use revises a word when
  the model is >= 95% sure with 3 words after it and gives the same answer with 2 and 5 after; from 2026-11-04 the
  first word of a sentence (no words before it) needs only 60% (`system_eval.FIRST_T`, EXPERIMENTS.md log 21).
  Score files carry `first_t` when they are on this rule.

### What is reported (from 2026-11-03)

As streaming speech recognition reports final WER beside how often partial output changes: one number for
what stays on screen, and two for how it got there. All come from `experiments/system_eval.py`, per word.

| measure | definition | where |
|---|---|---|
| **final error** | wrong after the revision rule (what stays on screen) | per cell; the cell average and every adopt / drop decision use it |
| **late fixes** | wrong at Space, right after the revision: the word changes on screen | per 1,000 words, all scored words |
| **broke** | right at Space, wrong after the revision (already counted in final error) | per 1,000 words, all scored words |

Errors at Space = final error - broke + late fixes, so they are not shown on their own. A change is adopted
when final error improves beyond noise and late fixes do not rise beyond noise; two models on the same words
are compared paired (words only one of them gets wrong: a difference larger than 2 sqrt(their sum) is real).

Older runs were scored at Space only (`knob_eval.py`, `score_one.sh`: every 10k screening run up to
tr4-02-10K, and TR3-10K, the 10k baseline); they are not rescored. The results table shows their Space numbers
marked as such and no late fixes / broke; they are compared only with each other.

## Categories (`experiments/cat_table.py`)

The same axes on both sides where they make sense; one word falls in one category per axis.

| axis | Korean words | English words |
|---|---|---|
| form | syllables / jamo only (ㅋㅋ) | — |
| case | — | lower / Capitalised / ALL CAPS |
| length | 1 / 2 / 3+ syllables | 1-2 / 3+ letters |
| keys read in the other language | not a word / a rare word / a common word (Zipf < 3 / < 4.5 / >= 4.5) | same |
| other | number words; Korean+English words | |

## Judging a difference

- **Seed noise**: the same setting with another seed (TR3-10K vs TR3-10K-s1). A change smaller than that is not a result.
  Seed differences on the shared cells (133,050 words): 19 wrong at 10k (TR4-10K pair), 44 at 51k (TR5-51K),
  34 at 102k (TR5n-102K). From 2026-11-04 screens run two seeds and compare seed means
  (`experiments/logs/scr_table.py`): a cell moves when the change is larger than max(2 × the larger seed
  difference, 2 × the binomial se of a difference of two-seed means).
- **Count noise**: with N errors, about sqrt(N) of them can move by chance. Report counts with denominators.
- **Cells** (switching 10, main 5, key collisions 1): a 10k run's change against the mean of TR3-10K and
  TR3-10K-s1 counts only if it is larger than max(2 sqrt(1.5) x the binomial standard error at the baseline
  rate, 2 × the seed difference); smaller changes are shown as "·". The **cell average** is the mean over the
  16 cells of the change in units of that cell's threshold, so every cell weighs the same; it ranks runs,
  the cells decide. Ambiguous English is shown beside them, not in the average.
- **Bias curves** (`experiments/bias_curve.py`, `experiments/amb_natural.py`): move the Korean / English
  decision bias and plot errors on one side against the other. A change that only moves along the baseline's
  curve leans; it is a gain only if it sits below the curve.
- **Weighting by use**: per-category error rates times how often each category occurs in the target user's
  typing (`experiments/error_tree.py` counts raw NSMC / KcBERT text) give expected errors per 1,000 words. The
  ranking flips with the user's English share (~20% is the crossover with current models), so it is applied
  last, not built into the sets.

[^paloma]: I. Magnusson et al., "Paloma: A Benchmark for Evaluating Language Model Fit", NeurIPS 2024 Datasets and Benchmarks. [arXiv:2312.10523](https://arxiv.org/abs/2312.10523)
[^pier]: E. Y. Ugan, N.-Q. Pham, L. Bärmann, A. Waibel, "PIER: A Novel Metric for Evaluating What Matters in Code-Switching", 2025. [arXiv:2501.09512](https://arxiv.org/abs/2501.09512)
