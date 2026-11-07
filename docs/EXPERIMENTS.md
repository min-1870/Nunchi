# Experiments log

What each version tried, what it showed, and why the next step was chosen. Newest state first, then the
log in order. Numbers are wrong words on the **dev** halves unless marked test; how the sets are built and
how differences are judged is in [EVALUATION.md](EVALUATION.md).

## Where things stand (2026-11-05)

Names: models `STRUCTURE[version][-variant]-STEPSK[-sSEED]` (TR4-51K = TR4 at 51k steps; TR5n = the TR5 recipe on
the TR4 size); screening runs `THEME-NN-STEPSK` by the log they belong to (dir = log 7, mark = log 10, reg = log 11,
tr4 = log 12, scr = log 20, siz = log 22, wid = log 23, wrd = log 27, bal = log 28, lwr = log 32). Parents, knobs,
status and earlier names are in `models/registry.json` (log 13; `_scheme` there explains the fields). Results are
final errors after the revision rule, with late fixes and broke words beside them (EVALUATION.md, "What is
reported"); "shared cells" = the 133,050 dev words every compared model was scored on, Shift-held typos left out.

- **App model: TR5n-102K** (`models/tr5n_102k.pt`, log 25): the TR4 size (d 256 × 4 layers, 3.2M parameters,
  3.8 ms on one thread) and recipe + Shift-held typos 8% (H) + single-word samples 15 → 5% (words), 102k steps.
  Shared cells: 840 / 806 wrong (two seeds) against TR4-51K's 879; Shift-held typos 77% → 0.2%. App thresholds
  tuned for it with the app's rule (log 33): act 0.85, revise 0.8, split 0.9 (were 0.99 / 0.99 / 0.7); 0.99 for a
  switch in the middle of a word (log 35); a word the app changed may go back once; a per-user habit prior. On
  17,876 words the author typed: 0.79% wrong (log 33).
- **Scale is spent**: wider pays nothing at 51k (log 25), 204k steps no better than 102k (log 30), two seeds
  averaged no better than the better seed (log 29).
- **Data ratios are spent on the known weak cells**: every change aimed at them moved errors from one side of a
  key-identical pair to the other (amb-01, wid-02, bal-01, lwr-01). Their errors sit on a few strings that fail in
  both directions (log 31): ㅋㅋ / zz, ㅎㅎ / gg, ㄷㄷ / ee, go / 해, to / 새, b / ㅠ.
- **Paused runs**, checkpoints kept: TR3-102K (38k), reg-02-51K (31k), dir-03-51K (8k).
- **Remaining errors** (hand review of TR5n-102K, log 26): ambiguous 46%, context would settle it 30%, needs a
  split inside the word 17%, set problems 8%.

### Open problems
1. Key-identical pairs (log 31): where the line between the two readings belongs depends on how often real users
   mean each one, which the sets fix by construction. A deployment question, not a data one.
2. Codes inside Korean text (TR5n): an upper-case start that reads as 초성체 (TR = ㅆㄲ, tr = ㅅㄱ) is read as
   Korean in Korean context, and the two TR5n-102K seeds disagree on TR5n (P(en) 0.26 vs 0.72). A pair case
   (1.); letter + digit codes in general are 0.5% wrong in web text (log 34).
3. Fused words and lower-case English + particle (k키트, kt면, gp내려가서): 7-8% wrong, spread over many words;
   lower-case English that reads as whole Korean syllables fails 14-19% (log 31). In the author's typing, English + particle
   (wsl로, ml에서) is 34% wrong and the first word of a line 2.5% (log 33).
4. Switching in the middle of a word: the model judges a prefix as a finished word (wha = 좀 at 0.999), so no
   threshold stops it (log 35).

### Next steps (proposed, not decided)
1. Switching in the middle of a word (open problem 4): measure a minimum number of letters on the cached prefix
   outputs; training on prefixes if that costs too many right switches.
2. Tune the app thresholds with dev cells weighted by the word kinds of real typing (log 33), checked on test once.
3. None on the model from the eval sets or web text (logs 30-32, 34). A new kind of typed text (chat, messenger)
   would be the next place to look for gaps; it needs a source first.

## What each step showed

The whole project in order, one lesson per step. The steps up to the reviser came before this log was kept;
the later ones point to their log entries.

**Single-word scores were far too optimistic.** The first fastText model made 0.02% wrong corrections on single
words but 1.03% on real sentences. Short, frequent, ambiguous words (그 `rm`, 내 `so`) are rare in a word list and
common in real text. From then on every result was measured on sentences.

**Calibration and one-word lookahead made context usable.** Softening fastText's overconfident scores cut wrong
corrections from 1.03% to 0.39%; a 2-state HMM over words brought Korean sentences to 0.05% but pulled English
inside Korean toward Korean. Waiting one word on ambiguous words fixed most of that.

**The synthetic mixed set hid the real problem.** Random dictionary words inserted into sentences are long, rare
and easy. Real English inside Korean is short and often upper case (`TV`, `OST`, `CG`). A 4-state HMM looked best
on synthetic data and worst on real data.

**Fixing the data beat every context trick.** Rebuilding the training words from real text (cased English,
English found in Korean text, collisions sent to the far more common side, Korean typed in the wrong mode removed)
cut wrong corrections on real mixed reviews from 0.68% to 0.26% without any context.

**An HMM has a hard ceiling.** It can move a word's odds by at most s/(1−s), so a word the model is 99% sure of
can never be overturned. What is left are words that are real in both languages (`so` / 내, `go` / 해), which need
the neighbouring words themselves, not just their language.

**A key-level tagger removes the one-language-per-word assumption.** Labelling every key makes `githubdp` →
`github` + `에` just a path that switches once.

**Wrong corrections cost more than they look.** The app switches the IME mode after a correction, so one wrong
correction makes the following words come out wrong too.

**A sentence model over the key stream beats a model over words.** A BiLSTM over word summaries forgot what the
word tagger knew (particles). One Transformer over the whole key stream, spaces included, learned both, and became
the main model.

**Stacked attention leaks the future.** With a mask that lets a word see one word ahead, four layers let it reach
four words ahead through its neighbours, while typing never shows more than one. Training now labels only the word
being decided (the last one, or the one before it with the last as lookahead), so training sees exactly what
typing does.

**Three words of left context are enough.** A model trained on the three previous words matched one trained on
the whole sentence. Waiting for the next word changes little: a few words either way.

**A data filter was deleting exactly the hard cases.** Dropping every sentence with Latin whose keys spell a Korean
word (to remove Korean typed in the wrong mode) also removed `so`, `go`, `do`, `work`, `B급` and almost every
single letter: in the kept text, short English inside Korean never appeared, so the model turned it into Korean.
Deciding by frequency instead (English Zipf vs the Korean word's Zipf; capitals Shift cannot type in Korean mean
English) brought these back. Real mixed-sentence errors went from 76 to 62, and from 63 to 52 with the next step.

**Meaningful synthetic data beats random insertion.** Swapping a word for its translation (`the weather` →
`the 날씨`, `날씨가` → `weather가`; loanwords like `미팅` / `meeting` are just one more translation) replaced
inserting random words, which only helped on tests built the same way. Lower-casing 30% of English inside Korean
text taught that people type acronyms in lower case (`jsa`, `apk`).

**Single letters are a choice, not an error to fix.** `T`, `b`, `s` inside Korean text are usually Korean jamo in
real text (ㅆ, ㅠ, ㄴ) but sometimes English. A rule "never convert one letter" fixed 15 such words on real
sentences but broke as many Korean comments with spaced jamo (`ㅅ ㅂ`, `ㅁ ㅊ`), so it is left to the model.

**Revising later works only as a separate, careful step.** Training one model to decide now and revise later made
its decision at Space worse. A separate reviser with strict limits fixes a few words and almost never breaks one.
(At 51k steps one model does both well: log 3.)

**Things that did not help.** A higher learning rate with a shorter schedule (clearly worse); code-level speedups
(the encoder is 95% of the time); `torch.compile` (3%); WSL instead of native Windows (11% faster at the time; equal
after the fixes under Speed).

**Seeds move results more than most settings.** The same recipe with another random seed differs by 44 words at
51k steps (log 25). Every change is now trained twice and judged on the pair, per cell.

**Longer training paid once, size did not.** 51k → 102k steps: −6%, both seeds below all four 51k runs. 204k
steps: no better than 102k. A wider model (7.2M parameters) learned faster but ended equal at 51k, for twice the
latency (logs 25, 30).

**What is left fails in both directions.** Errors read by hand: 46% are words where both readings fit, 30% need
the words around them, 17% a split inside the word. Counted by key, the same strings fail both ways in different
sets (ㅎㅎ read as `gg`, `gee` read as ㅎㄷㄷ; 해 read as `go`, `go` read as 해). Every data change aimed at them
moved errors from one side to the other (logs 26, 31).

**Guard sets catch what totals hide.** Dropping the single-word training samples left the total unchanged but made
English + particle typed alone 1.6× worse; only the one-word set showed it (log 27).

**Space is where the information arrives.** A word's language is guessed wrong 4.4% of the time after three keys
and 0.09% at Space. A model trained on whole words is overconfident on a prefix (`wha` reads as 좀 at 99.9%), so
switching in the middle of a word needs more than a threshold (log 35).

**Real typing is not the eval sets.** The author's own typing has far more same-language words and far fewer hard
cases than the dev sets; its errors sit on the first word of a line and on developer terms with a particle. It is
used to measure, never to tune (log 33).

### Results before the current sets: TX4 + reviser

Sets before 2026-11-03 (EVALUATION.md). Dev halves of held-out sets: sentences no model was trained on, with any
sentence that also appears in the training text removed. Every word is decided: at Space, or one word later when
the model is below 99% sure; a word is split when its best path is at least 70% sure. (Tuned for TX4 afterwards:
waiting from below 70% instead of 99% gives 50 instead of 52 on the first row, with fewer words rewritten.)
HMM = key-level word tagger + 2-state HMM over words; TX3, TX4 = the stream Transformer.

Wrong words:

| Set | Words | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| Real mixed sentences (NSMC reviews + web) | 15,032 | 105 | 63 | **52** |
| Real mixed sentences + typos | 15,032 | 183 | **90** | 91 |
| Web sentences with English + particle words | 16,797 | 171 | 97 | **87** |
| English sentences with a word swapped for Korean (generated) | 7,369 | 79 | 31 | **27** |
| Korean sentences with a word swapped for English (generated) | 9,600 | 41 | 22 | **9** |
| Korean only | 12,819 | | **1** | 3 |
| English only | 14,216 | | 4 | 4 |

Typos: each word gets one keyboard typo (neighbour key, dropped, doubled or swapped key) with probability 0.15. The
generated sets are built like part of the training data, so they flatter the models trained on it; the real sets
are the ones to trust.

Where the language switches (real mixed sentences):

| | Count | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| Entering English inside Korean | 1,141 | 42 | 37 | **30** |
| Back to Korean after English | 1,266 | 18 | 2 | **1** |
| English + Korean word split exactly (`MBC는`, web set) | 1,694 | 114 | 83 | **76** |

Most of TX4's remaining entry errors are 1-3 letter tokens (`T`, `b`, `gg`): 27 of 30.

Revising a word later: TX4 decides at Space; a reviser may change a word after 2-5 more words, only when it is sure
enough, gives the same answer at 2, 3 and 5 words, and TX4 itself was below 99%. RV0 learns to decide and to
revise; RV1 learns only to revise (no label on the word just typed). Fixed / broken against TX4 deciding at Space:

| Set | TX4 wrong | RV0, sure ≥ 90% | **RV1, sure ≥ 90%** | RV0, sure ≥ 99% | **RV1, sure ≥ 99%** |
|---|---|---|---|---|---|
| Words whose keys are common in both languages (`so` / 내, `go` / 해), English truth | 112 of 1,090 | +17 / −1 | +26 / −3 | +4 / −0 | **+9 / −0** |
| Same, Korean truth | 70 of 1,017 | +13 / −0 | +16 / −3 | +9 / −0 | +8 / −0 |
| Real mixed sentences | 56 of 15,032 | +9 / −1 | +12 / −4 | +2 / −0 | **+5 / −0** |
| Web sentences with English + particle words | 97 of 16,797 | +10 / −1 | +16 / −0 | +3 / −0 | +3 / −0 |

Training only to revise makes the reviser more useful: at 99% it fixes 25 words with none broken (RV0: 18); at 90%
it fixes more but breaks more. Speed then: about 17 ms a word on one CPU thread (3.2M parameters).

## How to run

```
python -u experiments/run.py experiments/configs/knobs_9.json         # 10k runs (configs name their runs)
python -u experiments/system_eval.py TR4-51K=TR4-51K TR4-10K=TR4-10K   # the results table: final error, late fixes, broke
python -u experiments/system_eval.py --test TR4-51K=TR4-51K           # test halves: once, for a chosen model
python -u experiments/policy_sweep.py cache TR4-51K; ... sweep TR4-51K # decision policy (log 15)
bash experiments/score_one.sh TR4-10K:tr4_10k.pt                  # older: at Space only, as the 10k runs before 2026-11-03
python -u experiments/knob_table.py                                    # changes vs TR3-10K at Space, marked beyond noise
python -u experiments/mix_search.py fit                                # regression over the knobs (log 11)
```
Data pools: `python -m nunchi.sentence_data v3`, `python -m nunchi.web_mixed train 50 web50_v3.txt v3`,
`python -m nunchi.en_marked v3` (log 10), `python -m nunchi.en_casual` (IMDB / Reddit text for the eval sets).
Eval sets: `experiments/source_sets.py`, `experiments/switch_sets.py`, `experiments/end_extra.py build` (built on
first use into `data/eval/`); the older ones in EVALUATION.md.

## Log

### 1. Longer training of TX4 (17k → 51k) — TX4-51K, TX4-51K-s1
- **Question**: was 17k steps the ceiling? 5× more web data at 17k (TX4-web50) had moved nothing.
- **Result**: real 56 → 48 / 50 (two seeds), particle 97 → 72 / 77, typo 102 → 82 / 92, ambiguous English
  493 → 469 / 461. Ambiguous Korean moved 80 between the two seeds alone.
- **Decision**: steps matter; keep longer runs. The ambiguous-Korean count is too noisy to judge on one seed.

### 2. Reviser at 51k — RV1-51K; the revision rule
- **Result**: without a gate RV1-51K is net positive everywhere (real +23/-15 vs RV1-17K +16/-17). Swept rules on
  dev: revise when >= 95% sure 3 words after and the same with 2 and 5 after (`conf+stable`, T=0.95).
- **Upper bound for 2+ words of lookahead** (no model): only 0.15% of real words have an ambiguous word right
  after an ambiguous word. One word of lookahead covers almost everything.

### 3. One model for both jobs — TR1-51K
- **Question**: RV0-17K (the same recipe at 17k) decided at Space worse than TX4-17K (64 vs 56). Lack of steps, or a
  real conflict between deciding and revising?
- **Result** (51k): decides at Space better than TX4-51K (real 38 vs 46, ambiguous 879 vs 976).
- **Decision**: steps were the cause. TR (one model, full window) replaces TX + RV.

### 4. The training filter dropped numbers — v2 pools, digit keys — TR2-51K
- **Audit** (`nunchi/sentence_data.clean`): every token that was not plain Hangul / Latin / Latin+Hangul was
  silently removed: number+Hangul (10점, 2019년) in 10-19% of lines, letters+digits (gpt4, 5km), Hangul+Latin
  (채널A). Digits were not in the key vocabulary.
- **Change**: v2 pools keep them (`clean(v2=...)`), digits label as their neighbours, own key ids for 0-9.
- **Result**: number words 375 → 41 wrong. But real 38 → 60, short English pushed to Korean.
- **Cause found**: per-word caps in the web pool counted letters inside codes (the s of s10 used up the cap of s)
  and code-only lines (12% of the pool) held no English word → fewer short English words seen.

### 5. v3 pools — TR3-51K
- **Change**: caps count real English words only; code-only web lines kept 30%.
- **Result**: number words keep the gain (37/2497). Real still 53 vs TR1-51K 38 on the old sets.

### 6. The eval sets were biased — sets with numbers in place
- The old eval sets were built the v1 way (numbers removed, neighbours glued: "별 5 개" → "별 개"), which suits
  v1-trained models. 39% of the web eval sentences had numbers removed. Rebuilt with numbers in place
  (`real_natural.jsonl`): TR3-51K 49 vs TR1-51K 40 on real; ambiguous English 402 vs 398 (old sets: 459 vs 366).
- The ambiguous-key set picks hard contexts (each key capped at 150 sentences), which inflated the Korean
  error rate (16% there vs 0.5% in natural context, `amb_natural.jsonl`).
- **Decision**: judge on the number-kept sets; the ambiguous set only as a hard-case reference.

### 7. Short screening runs (10k steps) to pick a data direction
Baseline TR3-10K = TR3 data and recipe at 10k; TR3-10K-s1 = seed 1. Runs in this table are named without
their steps (all 10k): dir-01 = dir-01-10K, and so on (earlier names in the registry's `old_name`). Changes
are judged against both, with bias curves (error trade-off between the two languages as the decision bias
moves) to separate "learned more" from "leans further one way".

| run | change | result | decision |
|---|---|---|---|
| dir-01 | short-English lines ×2 | small, clean gain on ambiguous English | split into parts |
| dir-02 | translation swaps 5 → 15% | none / worse | dropped |
| dir-03 | entry words ×3 in the loss | real better, ambiguous Korean worse (leans, not learns, on that set) | dropped for now |
| dir-04 | 1-3 letter English with no Korean reading ×2 | nothing (4.6 pp more data, no effect) | dropped |
| dir-05 | ambiguous-key English ×2 | the whole gain of dir-01 from 1.1 pp more data | kept |
| dir-06 | Korean-English-Korean lines ×2 | worse | dropped |
| dir-07 | Korean-English-English (switch) lines ×2 | gain | kept |
| dir-08 | ambiguous-key English ×4 | larger gain (-87 vs -35 at matched Korean errors) | kept |
| dir-09 | dir-05 + dir-07, both ×2 | gains roughly add up | kept |
| dir-10 | ambiguous-key English ×4 + switch ×2 | big English gain, Korean 1-syllable worse | Korean side needed |
| **dir-11** | dir-10 + ambiguous Korean lines ×4 | **English gain kept, Korean better too** | **best so far** |
| dir-12 | ambiguous Korean ×4 only | Korean better, English worse | — |
| dir-13 | English-based share 25 → 45% | worse on Korean-based text; English-based not measurable | dropped |
| dir-14 | dir-11 + number-word lines ×2 | numbers back (92 → 70), real 57 → 68 | open: push-out |

Per-category table of all of them: `experiments/cat_table.py` (see EVALUATION.md for the categories).

### 8. Which errors matter: frequency in real text
Each category's frequency per 1,000 words of raw user text (NSMC reviews, KcBERT comments) times its error
rate gives expected errors per 1,000 typed words. In that text English is ~2% of words, so Korean-side errors
dominate; for a user who types both languages about equally, English-side errors dominate (crossover near 20%
English). The priority depends on the target user's mix, so model changes are judged per category and the
mix is applied afterwards.

### 9. Eval sets built per source and per switching point (2026-11-03)
- **Why**: the old sets were mostly Korean (87% of words) and held too few switching points to see changes
  (entry errors 10-30); a total over them hid which side moved.
- **Built** (EVALUATION.md): a main set with 40k words from each of NSMC, KcBERT, web, IMDB, Reddit, and a
  switching set scored only where the language changes (direction × place, the word after, number words),
  real for Korean-base text and synthetic for English-base text; then 2,993 more KO>EN end points, a
  key-collision check (186 words) and ambiguous English in real Korean sentences (606 words).
- **Found while building**: `sentence_data.clean`'s wrong-mode filter dropped real English lines holding a
  word whose keys are a Korean word or marker (WW, mm, gg), from training and from the eval text alike. Most
  "rare English" errors in Korean text were Naver user IDs, whose labels are right.
- **Baseline TR3-10K** (mean of two seeds): main set 0.07-0.24% per source; switching cells 0.2% (EN>KO start)
  to 6.7% (KO>EN end); key collisions 30%; ambiguous English 18%.

### 10. English lines with Korean in them — en_marked (mark-01 to mark-07)
English lines with a Korean interjection, jamo word or conjugated word put in (`nunchi/en_marked.py`), as a
share of the training lines (E).

| run | change | result vs TR3-10K (pp, beyond noise only) | decision |
|---|---|---|---|
| mark-02-10K | E v1 at 10% | EN>KO middle -3.0, end -3.6; number words in English +5.9, key collisions +57 | collisions: the marked lines taught WW = ㅉㅉ |
| mark-05-10K | E v2: + 198 real lines holding colliding words | EN>KO middle -2.7, end -2.9; collisions +22 | too few real lines |
| mark-06-10K | E v3: real English lines per colliding key, 1:1 with the marked lines (WikiText-103, Reddit) | EN>KO middle -1.5, nothing worse | kept |
| mark-07-10K | E v3 at 1:2 | EN>KO middle -2.2, collisions +20 | dropped |

### 11. Knob search by regression (reg-00 to reg-22)
- 20 random knob settings (reg-00 to reg-19, Latin hypercube over E, L, A, S, K) at 10k, then a fit of each
  cell's rate on the knobs and the setting with the lowest predicted cell average under a main-set guard
  (RegMix-style, `experiments/mix_search.py`; reg-20), then two more rounds with the new runs added (reg-21, reg-22).
- **Result**: the picks were no better than the best random run (reg-20 -0.23 / -0.06, reg-21 -0.12 / -0.06,
  reg-22 -0.31 / -0.07 cell average; reg-11 -0.37 / -0.22), and every setting with en_marked v1 paid +30 to
  +60 pp on key collisions. A pick's two seeds differ as much as picks differ from each other.
- **Decision**: stopped; the knobs were chosen from error analysis instead (log 12). Entry loss (from dir-03) and
  switch lines (high in every regression pick) went into tr4-01 / tr4-02.

### 12. From error analysis to TR4 (tr4-01, tr4-02, TR4-10K)
| run | change | result vs TR3-10K (pp, two seeds) | cell average |
|---|---|---|---|
| tr4-01-10K | mark-06 + entry loss 2.25 (from dir-03) | KO>EN middle -1.6 / -1.8, end -2.0 / -2.1, EN>KO after -2.2; KO>EN after +0.3 / +0.5 | -0.32 / -0.31 |
| tr4-02-10K | tr4-01 + switch lines ×2.5 | KO>EN middle -1.9 / -2.2, end -2.5 / -2.6; KO>EN after back within noise; key collisions +9 in one seed | -0.49 / -0.36 |
| **TR4-10K** | tr4-02 + English+particle augmentation 0.5 → 0.25 | KO>EN middle -2.7 / -2.5, end -3.3 / -3.4, EN>KO middle -1.5 / -1.4, after -2.2 / -2.3, ambiguous English -9.7 / -10.0 | **-0.57 / -0.52** |

No main-set source and no key-collision cell worse beyond noise in TR4-10K. Adopted as the TR4 recipe.
- **What is left** (TR4's errors): mostly keys that read the same in both languages (b = ㅠ, gg = ㅎㅎ, WW = ㅉㅉ),
  where only context decides and each side's gain is the other's loss. English + particle split at the
  wrong key (Skylark → Skyla가, rk = 가) is rare (20 of 41k words): left as it is.
- **Decision**: further ratio tuning moves about as much as the seed difference; next is TR4 at 51k, then
  the decision policy.

### 13. Model names (2026-11-03)
- Models say structure, version, steps and seed (`TR3-51K`, `TR4-10K-s1`). A recipe gets the next version
  when it is adopted (cell average better than the 10k baseline beyond noise, no cell worse beyond noise, two
  seeds agree): tr4-02-10K with J 0.25 became TR4-10K.
- Screening runs are named by the log they belong to, `THEME-NN-STEPSK`: `dir` (log 7), `mark` (log 10), `reg`
  (log 11), `tr4` (log 12). The first short ids (A, R3-2, K8, M05, R2) mixed letters across logs (R meant both
  "round 3" and "regression"), and one-letter run ids clashed with the knob letters (E, L, S, J).
- What a run changes is in `knobs` and where it comes from in `parent` (a list when it joins runs, e.g.
  tr4-01 = mark-06 + dir-03; reg-20 is fitted on reg-00 to reg-19), not in the name: lineage is a graph, not a
  tree, so a path in the name would not fit.
- status: production (the model to ship, one), candidate, experiment, archived (why at the end of `what`).
- Files keep their training-time names (`k8_10k.pt` is tr4-02-10K; `tx4_long.pt` / `rv1_long.pt`, which the
  app's ONNX files are named after); score files under `experiments/logs/knobs/` and the run configs use the
  first `old_name`.

### 14. TR4 at 51k (2026-11-03)
`experiments/configs/tr4_51k.json`. Final error (after the revision rule), dev, TR3-51K → TR4-51K, with
2 × the binomial standard error of a difference:

| cell | TR3-51K | TR4-51K | change (2 SE) |
|---|---|---|---|
| KO>EN middle / end | 2.27 / 4.04% | 1.18 / 2.35% | -1.10 (0.64) / -1.69 (1.02) |
| EN>KO middle / end | 5.42 / 12.16% | 1.78 / 4.64% | -3.65 (1.40) / -7.51 (2.41) |
| ambiguous English | 10.56% | 5.61% | -4.95 (3.53) |
| key collisions | 25.3% | 32.3% | +7.0 (9.0): single letters z, h, mm (WW 3 of 68) |
| other switching cells, main sets | | | within noise |
| all scored words | 934 | 501 of 121,443 | |

- The EN>KO regression TR3 showed between 10k and 51k (EN>KO middle / end) does not happen: en_marked gives
  the English-base Korean that TR3's data lacked.
- Late fixes 0.55, broke 0.12 per 1,000 words (67 and 14 of 121,443).
- **Decision**: TR4-51K replaces TR3-51K as the production model.

### 15. Decision policy for TR4-51K (2026-11-03)
`experiments/policy_sweep.py`: the model's outputs at 0-5 words after each word cached once, then 3,905
settings scored on the CPU (Korean / English bias, split threshold, revision lookahead d, sureness T, which
other lookaheads must agree). The current rule (bias 0, split 0.7, d 3, T 0.95, agree at 2 and 5) has cell
average -0.96; the best setting with no cell worse beyond noise (split 0.9, d 5, T 0.8) -1.15. That 0.19 is
below the noise of a cell average (~0.25), and it pays in EN>KO middle (-1.47 → -0.81 pp). **Kept the
current rule.**

### 16. Test halves, once (2026-11-03)
First use of the test halves, for TR4-51K and TR3-51K (`system_eval.py --test`); key collisions on the new
test half without WW (EVALUATION.md). TR4-51K: 522 wrong of 122,728 (TR3-51K 973); late fixes 0.40, broke
0.04 per 1,000 words (49 and 5). Against TR3-51K the dev changes hold (KO>EN middle -1.06, end -2.19; EN>KO middle -3.70, end -8.89 pp;
ambiguous English 11.1 → 6.7%; collisions without WW 17.3 → 11.9%). TR4-51K was a little worse on test than
on dev where the recipe was chosen: KO>EN start 2.08 → 2.94%, EN>KO end 4.64 → 5.75%, EN>KO after
0.84 → 1.59% (TR3-51K moved less there), the expected cost of choosing on dev. Choices stay on dev.

### 17. A larger ambiguous English set (2026-11-03)
606 dev words could not show a change under ~2.6 pp. `experiments/amb_en_sets.py` adds Korean-base lines from
held-out KcBERT comments (after line 2.4M) and mC4 validation, with the same word rule: 3,087 dev / 3,087 test
words (`data/eval/amb_en_set.jsonl`; the first build also took English-base lines and web markup with a stray
N, dropped). Final error on dev, 2 SE about 1 pp:

| model | ambiguous English |
|---|---|
| TR3-51K | 13.7% |
| TR4-10K / -s1 | 10.1% / 8.6% |
| TR4-51K | 7.1% |

Since the scored words now include these, totals changed: TR4-51K 685 wrong of 123,924, late fixes 0.82, broke
0.15 per 1,000 (the new words hold many late fixes). Runs scored earlier keep the 606-word numbers; the results
table marks them and does not compare across the two sets.

### 18. Key collisions per key, number words in real English (2026-11-03)
- **Key collisions** were 186 dev words, 68 of them WW (World War: "WW2", "WW II"), which TR4-51K gets right
  (3 wrong); the keys it got wrong had 13-21 words each, and the test half came from WikiText units (65 km h,
  300 mm), far easier than the casual dev lines. `experiments/collision_sets.py`: both halves from the Reddit
  dumps, up to 60 words per key and half, the cell = mean over keys. 12 keys have 15+ words in both halves (h,
  z, mm, gee, sp, ac, gb, ee, dd, di, dm, eee); WW, gg, bb, zz have none left.
- **Number words in English** were 241 synthetic points with one or two errors. `experiments/num_en_sets.py`:
  words with digits and letters in WikiText / Reddit lines (en_marked and eval lines excluded), 2,993 dev /
  3,008 test points; mostly ordinals (7th) and decades (1990s), then km2, f16, 4x4, mp3.
- Final error on dev:

| model | key collisions (mean over keys) | number words, English |
|---|---|---|
| TR3-51K | 19.4% (gb 60 / 60) | 3.48% |
| TR4-10K / -s1 | 10.8% / 17.9% | 1.76% / 1.59% |
| TR4-51K | 14.1% | 0.73% |

- The collision cell's seed difference (7 pp, mostly gee 1 vs 21 of 60) is larger than any model difference
  in it; number words in English now separate models (TR3-51K → TR4-51K: 105 → 22 wrong of 2,993).

### 19. Ambiguous English: more data, and its guards (2026-11-04)
- **Guards** for pushing short English toward English: `experiments/jamo1_sets.py` (single jamo in Korean text,
  ㅋ / ㅠ / ㄴ: 2,962 dev words) and `experiments/amb_ko_sets.py` (Korean words whose keys are a common English
  word, 내 = so, 해 = go, 걍 = rid, 새 = to: 2,998 dev words; English Zipf 4.5+, since Zipf 3 let in 다 = ek,
  이 = dl). The old 142 single jamo of the main set put TR4-51K at 7%; on 2,962 it is 9.2%.
- **Cause checked first**: the cleaner does not drop Korean lines with lower-case ambiguous English (40% kept,
  other English 43%); such lines are rare (0.09% of KcBERT lines) (`experiments/logs/check_amb_filter.py`).
- **amb-01-10K** (`experiments/amb_en_train.py`, `NUNCHI_KMIX_EXTRA`): TR4-10K + 1,869 new KcBERT lines (after
  line 5M) with such words, ×4 as common in kmix, every line new. Dev, final error, mean of two seeds (TR4-10K
  seeds as the noise):

| cell | TR4-10K | amb-01-10K |
|---|---|---|
| ambiguous English | 9.33% | 5.56% |
| single jamo | 10.82% | 13.7% |
| Korean words that read as English | 3.57% | 11.5% |

- **Bias curve** (`experiments/logs/bias_curve_amb.py`): moving TR4-10K's Korean / English bias trades the
  same way; at the guard error amb-01 has, TR4-10K's curve gives about the same ambiguous English (within
  0.3 pp near the current point, 1.7 pp better only far toward Korean). The data mostly moved along the curve.
- TR4-51K sits well below TR4-10K's curve (7.1% ambiguous English at 5.6% guard error, where the 10k curve
  gives ~12.6%): longer training learned the context that the data ratio did not.
- **Decision**: not adopted; the data ratio is not the lever for this cell. The current bias stays (Korean
  readings are far more common in Korean text). Longer training is noted for later.
- Found by hand: a Korean word typed with Shift still held into the vowel (까 as RK) stays English; no set
  or augmentation holds such typos.

### 20. Overnight 10k screen (2026-11-04)
New cells first: **typos** (`experiments/typo_sets.py`: one keyboard typo in 15% of the main set's words, 14,437
dev; **Shift held** after a double consonant, 까 typed RK, 3,842 dev), **upper-case acronyms** with that shape
(`experiments/acro_sets.py`: JTBC, TK, YTN, WHO, EU; 1,504 dev) and **fused words** with no space between the
languages (`experiments/fused_sets.py`: QR코드, CCTV도, n번방; 1,513 dev; ~12.8% wrong for TR4-10K). TR4-10K
gets 77% of the Shift-held words wrong (TR4-51K 83%) and 1.7% of the position typos.

Eight ideas plus three, two seeds each at 10k against TR4-10K's two seeds (configs `experiments/configs/scr_*.json`,
run in a queue as `experiments/logs/run_queue.sh` does; tables `experiments/logs/scr_table.py`; a change counts beyond
max(2 × the larger seed difference, 2 binomial se)):

| run | change | result | decision |
|---|---|---|---|
| scr-01 | Shift-held typos 15% (`NUNCHI_SHIFT_P`) | Shift-held 77.7 → 0.08%; acronyms 0.15 → 0.40% (within noise); nothing else moved | candidate |
| scr-02 | Shift-held typos 30% | Shift-held 0.03%; acronyms 0.50%, beyond noise | 15% is enough |
| scr-03 | keyboard typos 15 → 25% (`NUNCHI_TYPO_P`) | position typos 1.75 → 1.33%; KO>EN start / middle / after +0.6 / +0.25 / +0.13 (within noise) | on hold |
| scr-04 | keyboard typos 15 → 5% | position typos 1.75 → 3.18%; clean switching cells a little better (KO>EN end -0.58, EN>KO middle -0.59, number EN -0.53) | the typo rate is a trade-off; 15% kept |
| scr-05 | Korean lines ending in English ×3 (mode `end`) | KO>EN end -0.33 (noise 0.66) | dropped |
| scr-06 | entry loss 2.25 → 3.0 | English side better, Korean guards worse, all within noise: leans | dropped |
| scr-07 | en_marked 10 → 15% | EN>KO end -0.71 (noise 2.6) | dropped |
| scr-08 | Shift 15% + keyboard 25% | both typo cells better; acronyms 0.51% (beyond noise), KO>EN start +0.70 | no |
| scr-09 | join neighbouring words of the two languages, 5% (`NUNCHI_JOIN_P`) | fused words -0.42 (noise 1.9) | dropped |
| scr-10 | join, 15% | fused words -1.33 (noise 4.3); single jamo and EN>KO end better, ambiguous English worse: leans Korean | dropped |
| scr-11 | Shift-held typos 8% | Shift-held 0.27%; acronyms 0.28% (within noise) | **recommended** |

- **Shift-held dose** (acronyms, 1,504 dev words; scr-01 on four seeds): 0% 0.15% (Shift-held 77.7%), 8% 0.28%
  (0.27%), 15% 0.40% ± 0.06 (0.08%), 30% 0.50% (0.03%). The acronym cost is real and grows with the rate, a few
  words either way; 8% keeps 99.7% of the Shift-held gain at the smallest cost. No other cell moved at any rate.
- **Keyboard typo rate** trades typo words against clean switching words (5% / 15% / 25%: typos 3.18 / 1.75 /
  1.33%, KO>EN end 2.78 / 3.36 / 3.61%); which side matters more depends on how often a typo is still there at
  Space, so 15% stays until that is known.
- **Decision (proposed, not decided)**: TR4 + Shift-held typos 8% at 51k as the next candidate.

### 21. Policy on the new cells; the first word of a sentence (2026-11-04)
- **Global policy** (`experiments/logs/policy_cells.py`, TR4-51K, every cell): no free setting. A lower split
  threshold (0.5) helps fused words (-1.1 pp) but costs position typos, KO>EN start / end and Shift-held words;
  the bias trades as before. Kept bias 0, split 0.7.
- **Input normalisation** for Shift held (an upper-case vowel after a double consonant made lower case before the
  model: RK → Rk): Shift-held words 83 → 0%, but upper-case acronyms 0.07 → 18% (TK, WHO, JTBC) and KO>EN cells
  +0.4-0.6 pp. Not used; the augmentation (log 20) is the fix.
- **First words** (`experiments/logs/main_errors.py`): on the main set the first word of a sentence is wrong 0.32%
  of the time against 0.06% in the middle (30 of TR4-51K's 95 main-set errors); the same in Korean words that read
  as English (5.2 vs 1.5%) and typos (3.4 vs 1.1%). With no words before it, the Space decision has no context;
  the revision rule rarely takes the later answer at 95%.
- **Rule**: the first word needs 60% sureness to be revised (`system_eval.FIRST_T`), every other word 95%. TR4-51K,
  every cell, first words only (fixed / broke, net, 2 sqrt(sum)):

| half | first words | 0.7 | 0.6 |
|---|---|---|---|
| dev | 15,713 | 83 / 31, +52 (21) | 107 / 40, +67 (24) |
| test | 15,664 | 85 / 35, +50 (22) | 104 / 47, +57 (25) |

- Every score file was brought to the rule (`experiments/logs/rescore_first.py`, first words only; checked
  against a full rescore of TR4-51K: every group equal; files before it in `experiments/logs/systems_before_first/`).
  TR4 models and the 10k screen gain 34-99 words each; the older models lose (RV0 +20, TR1 +18, TR3-51K +30,
  TX4-17K + RV1-17K +17): their later answers are less reliable, so the rule is for TR4 and later.
- The eval sentences stand alone; in the app a first word may have the text before it as context, which shrinks
  the gain there. The rule belongs in the app's revision step.

### 22. Model size (2026-11-04)
- **Latency** (`experiments/logs/latency_sizes.py`: ONNX Runtime on CPU, the app's graph, a 64-key stream, untrained
  weights): TR4 (d 256, 4 layers, 3.2M parameters) takes 3.8 ms on one thread of a Ryzen 5 5600, about 1.1 ms per
  million parameters. By PassMark single-thread ratios a 2018 office laptop (i5-8250U) is ~1.7× slower and a cheap
  one (Celeron N4020) ~2.9×. Published typing-latency data put noticeable delay at 50-100 ms, as low as 15 ms for
  the most sensitive; with 15 ms for the model on the i5-8250U the ceiling is ~7.7M parameters.
- Two sizes at that cost, TR4 recipe, 10k, two seeds, against TR4-10K (final error; total over 150,727 words, the
  Shift-held cell left out):

| run | size | parameters | 1-thread ms | total wrong | cells beyond noise |
|---|---|---|---|---|---|
| TR4-10K | d 256 × 4 layers | 3.2M | 3.8 | 1,676 / 1,706 | |
| siz-01 | d 384 × 4 layers (wider) | 7.2M | 7.8 | 1,579 / 1,619 (-92) | number words in English better |
| siz-02 | d 320 × 6 layers (deeper) | 7.5M | 8.2 | 1,655 / 1,645 (-41) | single jamo worse (+1.9 pp) |

- siz-01 moves nearly every cell the same way (KO>EN middle / end, EN>KO middle, single jamo, fused words, all
  within noise one by one), guards included: more capacity, not a lean. siz-02 helps fused words most but leans
  English on single jamo and Korean words that read as English.
- **Decision (proposed)**: the next long run is siz-01's size (d 384, 6 heads, 4 layers) with the TR4 recipe,
  Shift-held typos 8% and the first-word rule.

### 23. Two screens on the wide size (2026-11-04)
TR4 recipe on siz-01's size (d 384 × 4), 10k, two seeds, against siz-01 (shared cells of log 22, 150k words):
- **wid-01, single-word samples 15 → 5%** (`NUNCHI_WORDS`): 1,554 / 1,555 wrong against 1,579 / 1,619, both
  seeds lower; no cell beyond noise (fused words -1.38, noise 1.68). Counted in 40,000 samples: single-word samples
  held almost no single jamo (15), so they were not what anchors that cell; at 5% single jamo inside sentences
  rose 716 → 848 and single Latin letters 3,397 → 3,789, both sides alike. Taken into the next recipe.
- **wid-02, ambiguous English lines ×4** (amb-01's data, log 19): ambiguous English 8.45 / 9.10 → 5.38 / 5.47%,
  Korean words that read as English 3.50 / 2.75 → 14.26 / 10.28%, single jamo 6.37 / 7.19 → 10.05 / 11.82%.
  The same lean as on the TR4 size, larger. Dropped.

### 24. Single jamo from web pages, reviewed by hand (2026-11-04)
- The web part of the single-jamo cell (461 dev, 539 test) was read in full, blind to model output
  (`data/eval/jamo1_review.jsonl`): typed by a person 525, list markers and separators 342 (ㅡ ㅡ ㅡ, ㅣ 건강 ㅣ,
  ㄴ as a reply arrow), text about Hangul 121, unsure 12. One page often gave 5-30 items of one kind.
- Markers and text about Hangul are left out of the cell on both halves (`system_eval.items`); every dev score file
  was brought to it (`experiments/logs/rescore_jamo_web.py`, the files before it in `systems_before_jamo_review/`). The web part's error fell from about 33% to 7-11% (TR4-10K 34.9 → 7.6%, TR4-51K
  32.8 → 11.3%): most of it had been non-typed jamo.

### 25. TR5: Shift-held typos 8%, single-word samples 5%; width and length (2026-11-04)
Recipe TR5 = TR4 + Shift-held typos 8% (scr-11) + single-word samples 5% (wid-01). Overnight queue
(`experiments/logs/run_queue.sh` with `tr5_queue.txt`), each run scored as soon as it finished; per-cell tables against TR4-51K:
`experiments/logs/tr5_table.py`. Shared cells, 133,050 words:

| run | size | steps | train time | wrong (seeds) |
|---|---|---|---|---|
| TR4-51K | 256 × 4 | 51k | | 879 |
| TR5-51K | 384 × 4 (wide) | 51k | 2 h 04 | 894 / 850 |
| TR5n-51K | 256 × 4 | 51k | 76 min | 893 |
| TR5n-102K | 256 × 4 | 102k | 2 h 35 | 840 / 806 |

- **Width**: nothing at 51k (wide mean 872 against 893 and 879, inside the 44-word seed difference), for twice the
  training time and latency. siz-01's 10k gain was faster learning, not a higher ceiling. The size stays 256 × 4.
- **Recipe**: Shift-held typos 77% → 0.2% (8-10 of 3,842); acronyms 1-4 of 1,504 (TR4-10K 1-2, within noise); no
  other cell beyond noise against TR4-51K.
- **Length**: both 102k seeds sit below all four 51k runs (mean 823 against 879, -6%; seed differences 44 at 51k,
  34 at 102k). One cell at a time it is within noise; ambiguous English and Korean, single jamo, number words in
  English, position typos and the main sets all moved the same way. Late fixes / broke 0.53 / 0.18 per 1,000
  words, as TR4-51K.
- **Decision**: TR5n-102K is the app model (seed 0, `tr5n_102k.pt`).

### 26. What TR5n-102K still gets wrong, read by hand (2026-11-04)
`experiments/logs/error_dump.py` wrote all 1,122 dev errors with their sentences (`errors_tr5n102k.jsonl`); 245 were drawn per set and read
(`experiments/logs/error_sample_view.txt`, labels in `error_review_labels.py`). Shares weighted by set size,
about ±6 pp each:

| class | share | where |
|---|---|---|
| A. ambiguous: both readings fit | 46% | ㅋ / ㅎㅎ in English text, a lone jamo at a Korean line end, letters typed to dodge filters (ㅆl발, 7%) |
| C. the words around it settle it | 30% | key collisions, Korean words that read as English, single jamo, position typos |
| D. needs a split inside the word | 17% | fused words, words glued to numbers, jamo clumps from typos |
| B. set problem | 8% | 한/영 slips left in real text labelled English (dkss, cnti), URL pieces |

- Nothing is taken out of the sets: the B items were found among one model's errors only, so dropping them would
  favour that model, and they are a floor every model shares. Filter-dodging spellings are correct labels (A).

### 27. One-word input: a guard set, and single-word samples 0 vs 5% (2026-11-05)
- **Set** (`experiments/word1_sets.py` → `data/eval/word1_set.jsonl`, in `system_eval` from now on): whole lines
  of one word, dev ko1 2,000 (KcBERT one-word comments, <= 5 syllables: 대박, 아이고야), en1 1,993 (Reddit: thanks,
  FYI), mix1 965 (English + particle cut from held-out KcBERT text and typed alone: apk의, LG가); at most 3 copies
  of a word. Sentence data has 2+ words, so single-word samples are the only place a model sees a word alone.
- **wrd-00 / wrd-01**, TR5n recipe at 10k, two seeds, single-word samples 5% / 0%: English + particle alone
  7.15 / 8.81 → 12.33 / 13.37% (+4.87, noise 3.32); nothing else beyond noise (shared cells 1,651 / 1,674 against
  1,635 / 1,695). **5% stays.** Without the set the totals would have said "remove them".

### 28. Short keys both ways in Korean text — bal-01 (2026-11-05)
- **Data** (`experiments/bal_train.py` → `data/sent/kmix_bal.txt`, `NUNCHI_KMIX_EXTRA`): amb-01's 1,869 lines with
  lower-case ambiguous English, plus 1,865 held-out KcBERT lines holding the same keys as Korean words, matched key
  by key over 26 keys (c / ㅊ 374, o 247, go / 해 164, ...).
- **Result** (TR5n recipe, 10k, two seeds, against wrd-00): ambiguous English 9.17 / 9.65 → 7.32 / 7.94%, Korean
  words that read as English 2.96 / 2.76 → 9.14 / 6.44%; shared cells 1,651 / 1,674 → 1,827 / 1,782. The same
  lean as amb-01. Dropped.
- **Why**: balanced inside the added lines is not balanced in training. Korean lines with 해 / 새 / ㅊ number in the
  millions already, so 1,865 more change nothing; Korean lines with go / to / c as English are rare, so 1,869
  change their share a lot.

### 29. Two seeds averaged — ens-102K (2026-11-05)
`experiments/logs/ensemble_eval.py`: P(path), P(ko), P(en) averaged over TR5n-102K's two seeds, the key path from
the surer model. Shared cells 806, the same as the better seed (mean of the two 823); no cell beyond noise. The two
seeds mostly fail on the same words. Not worth twice the latency.

### 30. Longer again — TR5n-204K (2026-11-05)
- **Prediction** from 10k / 51k / 102k (1,075 / 879 / 823, E = E_inf + B s^-a): about 780 at 204k (E_inf ~610,
  a ~0.34). Stop rule: the 204k seed mean must beat 823 by more than the seed difference.
- **Seed 0** (5 h 08): shared cells **830**, inside the 102k pair (806 / 840). Single jamo 9.31% against 7.87 / 7.33
  (+1.7, about the noise); fused words 6.00 against 6.79 / 7.24; the rest flat.
- Training loss went below the 102k runs' (0.016-0.018 against 0.020) and the validation loss to 0.0024 (102k:
  0.0035 / 0.0030), while the eval did not move. The validation lines are clean one-language text: the model fits
  the training distribution better, and the errors left are of the kind that does not follow it.
- **Seed 1 stopped** at 18:53, 21 minutes in: it would have needed about 750 to pass. Three fitted points, each
  ±20-40, do not support extrapolating to twice the last one.
- **Decision**: training length stops at 102k.

### 31. Inside the weak cells: concentrated or spread? (2026-11-05)
`experiments/logs/slice_errors.py` (output `experiments/logs/slices_tr5n102k.txt`): both TR5n-102K seeds' dev errors
(fresh dumps, `errors_TR5n-102K*.jsonl`) against every scored word of each cell, by key and by word shape. The two
seeds agree almost key for key.

| cell | top 20 keys: errors / words | the keys (errors per seed / words) |
|---|---|---|
| ambiguous English | 96% / 40% | b → ㅠ 35 / 56, go → 해 26 / 63, s, l, m, to, so |
| Korean words that read as English | 100% / 43% (7 keys) | 해 → go 18 / 574, 새 → to 15 / 272 |
| key collisions | 100% / 75% | gee → ㅎㄷㄷ 22 / 60, ee → ㄷㄷ, ac, di, dd → ㅇㅇ |
| EN>KO end | 91% / 39% | ㅎㅎ → gg 19 / 38, ㅉㅉ → WW 14 / 44, ㅋ / ㅋㅋ / ㅋㅋㅋ, ㅠㅠ → bb |
| single jamo | all 30 jamo | upper case (ㅆ ㄲ ㅉ alone) 7.6% against 3.9% |
| fused words, English + particle | 23% / 2%, 43% / 3% | spread; lower case 18.5 / 14.2% against 3.5 / 1.8% upper case |

- The concentrated cells fail on the same strings in opposite directions: gg / ee / dd are read as Korean in one
  set and ㅎㅎ / ㄷㄷ / ㅇㅇ as English in another, go as 해 and 해 as go, with the same kind of words around them.
  Moving the line on such a string trades one cell for the other, as every data change on them did.
- What is left that is not a pair: lower-case English that reads as whole Korean syllables, inside fused words and
  English + particle (mostly read as all Korean), spread over many words.

### 32. English inside Korean typed in lower case more often — lwr-01 (2026-11-05)
`NUNCHI_LOWER_P` 0.3 → 0.6 (TR5n recipe, 10k, two seeds, against wrd-00): the fused and English + particle
errors are mostly lower-case acronyms glued to Korean (k키트, kt면, gp내려가서, s전자).

| cell | wrd-00 | lwr-01 | change | noise |
|---|---|---|---|---|
| fused words | 13.33 / 13.04 | 11.35 / 11.80 | -1.61 | 1.74 |
| English + particle, alone | 7.15 / 8.81 | 7.15 / 7.56 | -0.62 | 3.32 |
| ambiguous English | 9.17 / 9.65 | 8.16 / 7.71 | -1.47 | 1.05, better |
| single jamo | 8.54 / 7.17 | 11.01 / 10.70 | +3.00 | 2.74, worse |
| EN>KO after | 0.09 / 0.07 | 0.88 / 0.86 | +0.79 | 0.11, worse |

Shared cells 1,651 / 1,674 → 1,669 / 1,731. The targets moved within noise and the Korean guards got worse: a lean
toward English again. Dropped.

### 33. App thresholds for TR5n-102K (2026-11-05)
Tuned once, for the app model only, with the app's own rule (`experiments/logs/app_tune.py`; results in
`app_tune_TR5n-102K.txt` and `@test`), choosing the setting where one broken word costs about 2-3 missed ones:
act 0.85, revise 0.8, split 0.9 (were 0.99 / 0.99 / 0.7), in `TUNED["tr5n_102k"]` of `nunchi/export_onnx.py`.
Wrong words under the app rule (missed conversions counted): dev 2,782 → 1,550, test 2,814 → 1,587, broken words
on test 86 → 228. Every switching cell, ambiguous English and acronyms better on test; fused words
(7.2 → 9.9%), English + particle alone (6.3 → 9.0%), key collisions (2.5 → 6.2%) and single jamo
(1.3 → 2.1%) worse, the price of acting earlier. These numbers count differently from `system_eval` and do not
compare with the rest of the log; model comparisons keep the fixed rule (REV_T 0.95, FIRST_T 0.6).

- **Revert once** (built by the app session, measured with the app rule on app_tune's cached outputs): a word the
  app itself changed (at Space or in the middle) is judged once more one word later and may go back to how it was
  typed, at the revise threshold (0.8). Dev wrong 1,550 → 1,538 (broke 236 → 220; 20 reverts, 16 fixed, 4 broke),
  test 1,587 → 1,574. Small: one word later the model rarely turns around (the 3-word figure of the eval rule
  does not apply to the app).
- **Habit prior** v1 (app side, not scored: needs per-user data): per key of up to 3 letters, the language the word
  finally stayed in (undo and retyping count as the user's choice); before the thresholds,
  logit += ln((n_ko + 2) / (n_en + 2)), capped at ±2, from 3 uses on; kept as hashes, `NUNCHI_HABITS=0` turns it off.
- **The author's own typing, measured once** (`experiments/logs/my_typing.py`, `app_tune.py --mine`; data
  `data/eval/my_typing*.jsonl`): 17,876 words typed into Claude Code over 32 days (pastes and system text out; 82
  words left in the wrong mode relabelled as meant, ㅆㄲ5ㅜ → TR5n). Installed setting 142 wrong (0.79% ± 0.07;
  broke 44, missed 98), the old 0.99 / 0.99 / 0.7 207. First word of a line 2.5% wrong (so → 내, rm → 그 at
  P 1.0), English + particle in one word 34% (wsl로, ml에서, kmix는), switching words 5.9%, the rest 0.2%. On this
  text 0.95 / 0.8 / 0.7 does better (124): the dev sets hold far more hard cells than this typing (86% of it is
  same-language words). It is not tuned on: a measuring set, used once; the principled use is its share of word
  kinds as cell weights for tuning on dev. Claude Code text only, no chat.

### 34. Edge cases the sets may miss: held-out web text (2026-11-05)
`experiments/edge_mine.py`: 30,000 lines holding a Latin or digit token from held-out Korean web text (mC4 ko
validation + train shards 50-59; at most 40 lines a site, 5 lines a token; 14,043 sites: news, shopping, travel,
dictionaries, Q&A, code docs), cleaned as the eval sets are, not with the training filters. TR5n-102K, fixed rule,
every word (`edge_report_TR5n-102K.txt`): 849 wrong of 356,809 (0.24%).

| shape | web words | wrong | dev share → web share | dev rate |
|---|---|---|---|---|
| Latin then Hangul (apk의) | 4,600 | 3.3% | 1.56 → 1.3% | 4.9% |
| Hangul then Latin (채널A) | 973 | 13.7% | 0.19 → 0.3% | 21.4% |
| the same with a digit | 574 + 311 | 8.7%, 1.9% | 0.07 → 0.2% | 12.3%, 3.9% |
| Hangul + digits (훈7등, 전2) | 2,651 | 1.4% | 0.24 → 0.7% | 1.3% |
| Hangul | 226,403 | 0.06% | 45.9 → 63.5% | 0.19% |
| Latin lower, 2-3 letters | 14,682 | 0.7% | 11.4 → 4.1% | 1.2% |
| letters + digits (gpt4, A100) | 2,240 | 0.5% | 0.97 → 0.6% | 0.5% |
| every other shape | | <= 0.8% | | |

- No shape is common on the web, clearly worse than on the dev sets and missing from them; no site stands out (the
  worst with 200+ words has 5 errors). Read in context, most errors are page text, not typing: tokens glued by
  removed layout (Videobam로iPhone, 기자l입력2017), table cells (행1, 요15), code (snsTitle, removeEventListener),
  ㅣ as a separator. The rest are the known kinds: fused words and English + particle (kg로, wifi만, d에), and
  key-identical pairs, here with Korean words that type as tech abbreviations (눈 = sns, 렌 = fps, 행 = god,
  새재 → towo).
- Letter + digit codes are no gap in web text (0.5%); TR5n (open problem 2) fails through the 초성체 pair
  TR = ㅆㄲ, not through its digits.
- Not covered: chat and messenger typing (only KcBERT comments stand for it) and English-base text beyond Reddit /
  IMDB. Embedding-based slice discovery is not needed for web text; it would only pay with typed text of a new kind.

### 35. Switching in the middle of a word; when to decide (2026-11-05)
- **Seen**: typing "what" in English mode showed 좀ㅅ until Space. The app switches in the middle of a word after
  any key when the word so far is sure enough (`typing.rs` key(), threshold act, one letter max(act, 0.999)); with
  act 0.85 (log 33) wha = 좀 fires.
- **Measured** (`experiments/logs/early_tune.py`, `early_tune_TR5n-102K*.txt`): the model on all 790,423 prefixes of
  the dev words with their left context, the app's rule at every key, then the boundary rule (act 0.85, split 0.9)
  from the mode the word is in by then (revision and habit prior left out):

| early threshold | switches | right | wrong in the middle | wrong at the end | broke |
|---|---|---|---|---|---|
| 0.85 (act) | 26,848 | 20,860 | 5,988 | 2,502 | 325 |
| 0.95 | 24,244 | 20,328 | 3,916 | 2,494 | 274 |
| 0.99 | 21,924 | 19,336 | 2,589 | 2,466 | 216 |
| 0.999 | 18,834 | 16,868 | 1,965 | 2,460 | 180 |
| no switch in the middle | 0 | 0 | 0 | 2,510 | 152 |

  Test, once: 6,066 / 2,552 at 0.85, 2,644 / 2,522 at 0.99. 0.99 (`early_act`, built by the app session) has 57%
  fewer wrong switches, fewer end errors and broken words; 0.99 → 0.999 would lose 4 right switches per wrong one
  removed. 1,583 of the 2,589 left are one-letter switches (the 0.999 floor, 6,676 right).
- **Not enough**: in sentences the app sees wha at P(ko) 0.9986-0.9997 ("apple so what", "안녕 what"), above any
  threshold. The model learned to judge whole words at Space; given a prefix it judges it as a finished word, and 좀
  is a common one. Options: no switch before N letters (measurable on the cached outputs), a list of prefixes that
  are common words in the other language, or training on prefixes (the label of the word they become). Not decided.
- **When to decide** (TR5n-102K, best guess, dev): one-language words of 4+ keys are wrong 36.7% after 1 key, 7.3%
  after 2, 4.4% after 3, 0.09% at Space, 0.08% one word later. All words: 0.92% at Space, 0.72% one word later
  (-21%); the eval rule's 3 words gain no more (-18%; log 2: one word of lookahead holds almost all of it). Space is
  where the information arrives; the open choice is how freely to change a word one word later.

## Speed (2026-11-02)

- The GPU runs at ~8% of its FP16 peak on this model: time goes to many small operations, not matrix
  multiplies (11 ms of a ~85 ms step). Bigger batches did not raise throughput (2048: +3%, 4096: slower),
  FP16 vs BF16 made no difference.
- **CRF without the per-key loop** (`log_z_tree`, `score_vec` in `nunchi/stream_tagger.py`): 86 → 76 ms a step,
  same loss and gradients. Kept.
- **Pools as one byte buffer** (`LineStore`), and on Linux one sampler shared by forked workers: data workers
  1.8 GB → far less each; WSL training 8.2 → 4.7 GB (WSL gets half of the 16 GB RAM).
- **WSL**: batches through the worker queue took 40-77 ms a step with torch's default file-descriptor sharing;
  `file_system` sharing (now the Linux default) brought it to 3 ms. After that WSL = Windows (GPU compute
  5-10% faster in WSL, not enough to move).
- **Two runs on the GPU at once** (2026-11-03, Windows, 500 steps of the TR4 recipe, batch 1024): alone 11,030
  prefixes/s (compute 80 ms a step); together 6,020 + 6,082 = 12,100 (compute 135-140 ms each), 1.10×, with the
  data queue full in both. Not worth it: the GPU, not the data, is the limit. With NUNCHI_VRAM 0.35 or 0.42 each,
  both runs failed on HIP out-of-memory at ~3 GB allocated (one hung in hipBLASLt), so the cap does not split the
  card between processes on Windows ROCm; 0.7 each ran.
- Tried, no gain: a (B, T) padding mask instead of the full mask (same speed; kept, same output), batches
  padded to few shapes, torch.compile (default mode did not finish compiling in 25 min; with graph capture no
  faster), ROCm TunableOp (did not finish tuning in 15 min).

## Lessons

- Before calling a design a failure, check the training budget (RV0-17K vs TR1-51K).
- Check how an eval set was built before trusting a comparison across data versions (numbers removed; hard
  contexts picked).
- Report counts with denominators, per category, with the noise (seed difference, and about sqrt(N) for N
  errors). A total hides which side moved.
- With a fixed step budget, drawing some lines more often pushes others out; look at every category, not only
  the targeted one.
- Shifting the decision bias is free; a change only counts if it beats the baseline at the same error level on
  the other side (bias curves).
- A falling validation loss on clean text says little about the hard cells: TR5n-204K's went 0.0030 → 0.0024
  with no gain on the eval (log 30).
- Balanced inside the added lines is not balanced in training: what a model learns from extra lines depends on
  how common each reading already is (log 28).
- Before adding data for a weak cell, count its errors by key: if the same strings fail in opposite cells, data
  only moves the line (log 31).
- Score a change with the rule that will use it: the eval rule's late fixes (3 words later) promised far more than
  the app's one-word-later revert gave (log 33).
- A model trained to judge words at Space is overconfident on prefixes: a threshold cannot gate a decision the
  model was never trained to make (log 35).
