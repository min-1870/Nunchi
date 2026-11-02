# Evaluation: sets and how results are judged

All sets are split into **dev** and **test** halves by a hash of the sentence; choices (thresholds, data
settings) are made on dev, final comparisons on test. Sentences that also occur in any training pool
(v1, web50, v2, v3) are dropped (`experiments/evalsets.py`, `in_training`).

## Sets

| set | file / builder | what it measures | caveats |
|---|---|---|---|
| real mixed, numbers in place | `data/eval/real_natural.jsonl`, `experiments/natural_sets.py` | NSMC test reviews + mC4 validation web sentences with Korean and English, numbers kept as context; scored on the original words | **main set**. Mostly Korean (87% of words) |
| ambiguous words, natural context | `data/eval/amb_natural.jsonl`, `experiments/amb_natural.py` | words whose keys read as a word in both languages (해 = go, so = 내), sentences drawn as they come from mC4 train shards 55-59 | **main set for ambiguous words**; key frequencies as in real text |
| number words | `data/sent/web_code_eval.txt`, `python -m nunchi.web_mixed code` | words with digits (3개, gpt4, 5km) and Korean+English words (채널A), mC4 validation | scored by output text: digits type the same in either mode |
| English + particle | `data/sent/web_particle_eval.txt` | English word + Korean particle in one word (SF는) must split at the right key | numbers removed (v1 build) |
| ambiguous-key set | `data/sent/ambiguous_eval_v3.jsonl`, `python -m nunchi.ambiguous_eval v3` | ambiguous words, each key capped at 150 sentences | **hard contexts**: rates are inflated (Korean 16% here vs 0.5% natural). Reference only |
| old real / ambiguous sets | `web_mixed_v2.txt`, NSMC via `context.real_mixed`, `ambiguous_eval_v2.jsonl` | the sets used before 2026-11-02 | numbers removed: favours models trained on v1 pools |

Missing: Korean jamo-only words (ㅋㅋ ㅠㅠ: 28 in dev), English-based sentences with Korean words, enough entry
(ko -> en) cases.

## Scoring

- A word is right when the text it types equals the text written (`experiments/textcheck.py`): a split word
  (SF는) must split at the right key; digits never count against a model.
- Decided at Space (no word after it) unless a table says "revising": then the rule in use revises a word when
  the model is >= 95% sure with 3 words after it and gives the same answer with 2 and 5 after.

## Categories (`experiments/cat_table.py`)

The same axes on both sides where they make sense; one word falls in one category per axis.

| axis | Korean words | English words |
|---|---|---|
| form | syllables / jamo only (ㅋㅋ) | — |
| case | — | lower / Capitalised / ALL CAPS |
| length | 1 / 2 / 3+ syllables | 1-2 / 3+ letters |
| keys read in the other language | not a word / a rare word / a common word (zipf < 3 / < 4.5 / >= 4.5) | same |
| other | number words; Korean+English words | |

## Judging a difference

- **Seed noise**: the same setting with another seed (S0 vs S0-s1). A change smaller than that is not a result.
- **Count noise**: with N errors, about sqrt(N) of them can move by chance. Report counts with denominators.
- **Bias curves** (`experiments/bias_curve.py`, `experiments/amb_natural.py`): move the Korean / English
  decision bias and plot errors on one side against the other. A change that only moves along the baseline's
  curve leans; it is a gain only if it sits below the curve.
- **Weighting by use**: per-category error rates times how often each category occurs in the target user's
  typing (`experiments/error_tree.py` counts raw NSMC / KcBERT text) give expected errors per 1,000 words. The
  ranking flips with the user's English share (~20% is the crossover with current models), so it is applied
  last, not built into the sets.
