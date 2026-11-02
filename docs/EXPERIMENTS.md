# Experiments log

What each version tried, what it showed, and why the next step was chosen. Newest state first, then the
log in order. Numbers are wrong words on the **dev** halves unless marked test; how the sets are built and
how differences are judged is in [EVALUATION.md](EVALUATION.md).

## Where things stand (2026-11-02)

- **Reference model: TR3-long** (`models/tr3_long.pt`, registry name `TR3-long`): one stream Transformer that
  decides each word at Space and revises it as more words come (full window, every word labelled), 51k steps,
  v3 pools (numbers kept), digit keys. 3.2M parameters.
- **Best direction found: R3-2** (10k screening run): lines holding ambiguous-key words drawn more often on
  **both** sides, English (`NUNCHI_SHORT_EN_MODE=amb:4+switch:2`) and Korean (`NUNCHI_AMB_KO_X=4`). Against
  the 10k baseline it cuts English 1-2 letter errors 323 -> 212 and ambiguous Korean 19 -> 8, total 464 -> 367
  (noise about +-21). Costs: number words +15, plain English (3+ letters) +5 to +9 (borderline noise).
- **Weakest spot: short lowercase English inside Korean** (1-2 letters, keys that also read as Korean:
  so, go, s, gg). 11-12% of such words wrong in the baseline; ALL CAPS English is ~0% wrong.
- **Scale (xlong, big) is postponed** on purpose: data, loss and decision policy first. TR3-xlong is paused
  at 38k with its checkpoint kept.

### Open problems
1. Fixing one category by drawing its lines more often pushes others out (fixed step budget): R3-2c brought
   number words back (92 -> 70) but real sentences went 57 -> 68. Hand-tuning multipliers one at a time is slow.
2. Eval gaps: Korean jamo-only words (ㅋㅋ ㅠㅠ ㅜㅜㅜ: 28 in the dev sets), English-based sentences with Korean in
   them (none), entry (ko -> en) errors too few (10-30) to see changes.
3. Real English-based mixed text (English sentences with Korean words) is missing from training: only
   translation swaps (`trans_en`).

### Next steps (proposed)
1. **Automated mixture search**: one objective (expected errors per 1,000 words for the target user's
   language mix, no category worse than the baseline by more than its noise), ~20 random multiplier settings
   as 10k runs overnight, then a regression / Bayesian optimisation to pick the setting (`experiments/run.py`
   with a generated config).
2. **Hard-example weighting in the loss** instead of hand multipliers (weight words the model gets wrong or is
   unsure about), one 10k run against S0 / R3-2.
3. Grow the thin eval sets (jamo, entry, English-based) before judging changes there.
4. Then a long (51k) run of the chosen setting against TR3-long, category by category.

## How to run

```
python -u experiments/run.py experiments/configs/r3_2_branches.json   # 10k runs + evaluations
python -u experiments/cat_table.py S0:short_s0.pt S0-s1:short_s0_s1.pt R3-2:short_r3_2.pt   # per-category table
```
Data pools: `python -m nunchi.sentence_data v3`, `python -m nunchi.web_mixed train 50 web50_v3.txt v3`.
Eval sets with numbers in place: `experiments/natural_sets.py`, `experiments/amb_natural.py` (build on first
use into `data/eval/`).

## Log

### 1. Longer training of TX4 (17k -> 51k) — TX4-long, TX4-long-s1
- **Question**: was 17k steps the ceiling? 5x more web data at 17k (TX4-web50) had moved nothing.
- **Result**: real 56 -> 48 / 50 (two seeds), particle 97 -> 72 / 77, typo 102 -> 82 / 92, ambiguous English
  493 -> 469 / 461. Ambiguous Korean moved 80 between the two seeds alone.
- **Decision**: steps matter; keep longer runs. The ambiguous-Korean count is too noisy to judge on one seed.

### 2. Reviser at 51k — RV1-long; the revision rule
- **Result**: without a gate RV1-long is net positive everywhere (real +23/-15 vs RV1 +16/-17). Swept rules on
  dev: revise when >= 95% sure 3 words after and the same with 2 and 5 after (`conf+stable`, T=0.95).
- **Upper bound for 2+ words of lookahead** (no model): only 0.15% of real words have an ambiguous word right
  after an ambiguous word. One word of lookahead covers almost everything.

### 3. One model for both jobs — TR-long
- **Question**: RV0 (the same recipe at 17k) decided at Space worse than TX4 (64 vs 56). Lack of steps, or a
  real conflict between deciding and revising?
- **Result** (51k): decides at Space better than TX4-long (real 38 vs 46, ambiguous 879 vs 976).
- **Decision**: steps were the cause. TR (one model, full window) replaces TX + RV.

### 4. The training filter dropped numbers — v2 pools, digit keys — TR2-long
- **Audit** (`nunchi/sentence_data.clean`): every token that was not plain Hangul / Latin / Latin+Hangul was
  silently removed: number+Hangul (10점, 2019년) in 10-19% of lines, letters+digits (gpt4, 5km), Hangul+Latin
  (채널A). Digits were not in the key vocabulary.
- **Change**: v2 pools keep them (`clean(v2=...)`), digits label as their neighbours, own key ids for 0-9.
- **Result**: number words 375 -> 41 wrong. But real 38 -> 60, short English pushed to Korean.
- **Cause found**: per-word caps in the web pool counted letters inside codes (the s of s10 used up the cap of s)
  and code-only lines (12% of the pool) held no English word -> fewer short English words seen.

### 5. v3 pools — TR3-long
- **Change**: caps count real English words only; code-only web lines kept 30%.
- **Result**: number words keep the gain (37/2497). Real still 53 vs TR-long 38 on the old sets.

### 6. The eval sets were biased — sets with numbers in place
- The old eval sets were built the v1 way (numbers removed, neighbours glued: "별 5 개" -> "별 개"), which suits
  v1-trained models. 39% of the web eval sentences had numbers removed. Rebuilt with numbers in place
  (`real_natural.jsonl`): TR3-long 49 vs TR-long 40 on real; ambiguous English 402 vs 398 (old sets: 459 vs 366).
- The ambiguous-key set picks hard contexts (each key capped at 150 sentences), which inflated the Korean
  error rate (16% there vs 0.5% in natural context, `amb_natural.jsonl`).
- **Decision**: judge on the number-kept sets; the ambiguous set only as a hard-case reference.

### 7. Short screening runs (10k steps) to pick a data direction
Baseline S0 = TR3 data and recipe at 10k; S0-s1 = seed 1. Changes are judged against both, with bias curves
(error trade-off between the two languages as the decision bias moves) to separate "learned more" from
"leans further one way".

| run | change | result | decision |
|---|---|---|---|
| A | short-English lines x2 | small, clean gain on ambiguous English | split into parts |
| B | translation swaps 5 -> 15% | none / worse | dropped |
| C | entry words x3 in the loss | real better, ambiguous Korean worse (leans, not learns, on that set) | dropped for now |
| A-가 (plain) | 1-3 letter English with no Korean reading x2 | nothing (4.6%p more data, no effect) | dropped |
| A-나 (amb) | ambiguous-key English x2 | the whole gain of A from 1.1%p more data | kept |
| A-3a (insert) | Korean-English-Korean lines x2 | worse | dropped |
| A-3b (switch) | Korean-English-English lines x2 | gain | kept |
| A-나 x4 | stronger | larger gain (-87 vs -35 at matched Korean errors) | kept |
| A-나 + switch | both x2 | gains roughly add up | kept |
| R3-1 | A-나 x4 + switch x2 | big English gain, Korean 1-syllable worse | Korean side needed |
| **R3-2** | R3-1 + ambiguous Korean lines x4 | **English gain kept, Korean better too** | **best so far** |
| R3-3 | ambiguous Korean x4 only | Korean better, English worse | — |
| BAL | English-based share 25 -> 45% | worse on Korean-based text; English-based not measurable | dropped |
| R3-2c | R3-2 + number-word lines x2 | numbers back (92 -> 70), real 57 -> 68 | open: push-out |

Per-category table of all of them: `experiments/cat_table.py` (see EVALUATION.md for the categories).

### 8. Which errors matter: frequency in real text
Each category's frequency per 1,000 words of raw user text (NSMC reviews, KcBERT comments) times its error
rate gives expected errors per 1,000 typed words. In that text English is ~2% of words, so Korean-side errors
dominate; for a user who types both languages about equally, English-side errors dominate (crossover near 20%
English). The priority depends on the target user's mix, so model changes are judged per category and the
mix is applied afterwards.

## Lessons

- Before calling a design a failure, check the training budget (RV0 at 17k vs TR-long at 51k).
- Check how an eval set was built before trusting a comparison across data versions (numbers removed; hard
  contexts picked).
- Report counts with denominators, per category, with the noise (seed difference, and about sqrt(N) for N
  errors). A total hides which side moved.
- With a fixed step budget, drawing some lines more often pushes others out; look at every category, not only
  the targeted one.
- Shifting the decision bias is free; a change only counts if it beats the baseline at the same error level on
  the other side (bias curves).
