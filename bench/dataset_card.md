---
license: cc-by-sa-4.0
language:
- ko
- en
pretty_name: Nunchi benchmark (Korean / English input language)
size_categories:
- 10K<n<100K
tags:
- input-method
- ime
- code-switching
- korean
- keyboard
configs:
- config_name: default
  data_files:
  - split: test
    path: nunchi_bench.jsonl
---

# Nunchi benchmark

Korean is typed on the same keys as English (두벌식 layout), and people switch between the two with the 한/영 key.
Input methods that pick the language themselves read the keys and decide, word by word, whether they were meant
as Korean or English. This benchmark asks how often they get it right: given the keys of a line typed without
pressing 한/영, does the field end up holding the line as it was meant?

- **36,166 lines, 123,069 scored words**, in cells (kinds of word) reported apart: words where the language changes,
  English words whose keys are also a Korean word (so = 내), single jamo (ㅋ), words of both languages with no space
  (MBC는), typos, one-word input, and whole lines that should not switch at all. See [SPEC.md](SPEC.md).
- Real text: Korean movie reviews and news comments, English Stack Exchange posts and comments, Wikipedia.
- One file, one scorer with no dependencies (`score.py`).

Version 1, built 2026-11-08. sha256 of `nunchi_bench.jsonl`: see `manifest.json`. Results on another version are
not comparable.

## Use

```
python score.py nunchi_bench.jsonl predictions.jsonl
```

`predictions.jsonl` holds, per line id, the text the field holds after typing the line's keys as [SPEC.md](SPEC.md)
describes (fresh state, a Space after every word, no 한/영 presses). Report every cell with its count, not one
total.

## Fields

| field | meaning |
|---|---|
| `id` | line id |
| `set`, `src`, `key` | the cell |
| `w` | the words as typed: keys of each word, one label per key (`1` Korean, `0` English) |
| `pts` | the scored words (indices into `w`) |
| `cats` | for `switch` and `end_extra`, each scored word's category; null elsewhere |
| `text` | the line as meant |
| `from` | source and licence of the text; for Stack Exchange, site, question and author links |

## Sources and licences

The set is released under **CC BY-SA 4.0**. Its text comes from:

| source | licence | used for |
|---|---|---|
| [NSMC](https://github.com/e9t/nsmc) (Naver movie reviews, Lucy Park), test file | CC0 1.0 (as stated by the dataset) | `main` |
| [KcBERT pre-training corpus](https://github.com/Beomi/KcBERT) (news comments, Junbum Lee), lines from 12M on | CC BY-SA 4.0 (as stated on the dataset card) | most Korean cells |
| [Stack Exchange data dump](https://archive.org/details/stackexchange) of 2024-03-31: interpersonal, parenting, movies, cooking, travel, workplace | CC BY-SA 2.5 / 3.0 / 4.0 by date | English cells |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (Merity et al., 2016) | CC BY-SA | `num_en` |

**Stack Exchange attribution.** Every Stack Exchange line names its site, links to its question (`question_url`)
and names its author with a link to their profile (`author`, `user_url`; deleted users have a name only), as
[Stack Exchange's licensing terms](https://stackoverflow.com/help/licensing) ask.

**Changes made to the text.** Lines were cut into pieces of at most 20 words, reduced to Hangul, Latin and digit
tokens (no punctuation), and apostrophes inside words removed (don't → dont). Lines holding profanity, slurs,
insults or web leftovers (URL parts, ids) were left out. In the synthetic `switch` lines one English word was
replaced by its Korean translation or a Korean interjection was put in; in the `typo` cells keyboard typos were
put in. Lines that were changed say so in `from.changed`.

No rights are granted in the third-party text beyond its licences. If you hold rights in any of it and object to
its use here, open a discussion on this dataset's page and the lines will be removed in the next version.

## How the set was built

Each cell follows the rules of the set its author uses to evaluate models, and draws from text that set and the
training data do not hold. The builder is
[`experiments/public_sets.py`](https://github.com/min-1870/Nunchi/blob/main/experiments/public_sets.py).

Limitations:
- The Korean text is mostly political news comments (KcBERT) and movie reviews (NSMC).
- English-base lines with Korean in them are synthetic: real English text with one Korean word put in.
- Filtering out abusive lines makes the text cleaner, and easier, than real comments: cases like `C발` or `ㅅㅂ`
  are not in it.
- Key collisions cover the 13 marker keys with at least 15 points in the Stack Exchange text; others are absent.

## Results

Nunchi, model [TR5n-102K](https://huggingface.co/min-1870/nunchi-tr5n-102k) (a script drives the model: each word
decided at Space and revised up to three words later when the model is sure). Error rate per cell, wrong / points:

| cell | source | wrong / points | rate (± 2 SE) |
|---|---|---|---|
| main | NSMC | 12 / 20,001 | 0.06% (± 0.03) |
| main | KcBERT | 5 / 20,005 | 0.02% (± 0.02) |
| main | Stack Exchange posts | 1 / 20,004 | 0.00% (± 0.01) |
| main | Stack Exchange comments | 3 / 20,018 | 0.01% (± 0.02) |
| switch, Korean base, KO>EN start | KcBERT | 17 / 651 | 2.61% (± 1.25) |
| switch, Korean base, KO>EN middle | KcBERT | 10 / 1,527 | 0.65% (± 0.41) |
| switch, Korean base, KO>EN end | KcBERT | 5 / 186 | 2.69% (± 2.37) |
| switch, Korean base, word after | KcBERT | 7 / 1,749 | 0.40% (± 0.30) |
| switch, English base, EN>KO start | Stack Exchange | 4 / 1,333 | 0.30% (± 0.30) |
| switch, English base, EN>KO middle | Stack Exchange | 51 / 1,333 | 3.83% (± 1.05) |
| switch, English base, EN>KO end | Stack Exchange | 121 / 1,331 | 9.09% (± 1.58) |
| KO>EN end, more points | KcBERT | 37 / 1,250 | 2.96% (± 0.96) |
| amb_en (so = 내) | KcBERT | 112 / 2,001 | 5.60% (± 1.03) |
| amb_ko (해 = go) | KcBERT | 22 / 2,500 | 0.88% (± 0.37) |
| jamo1 (ㅋ) | KcBERT | 97 / 2,502 | 3.88% (± 0.77) |
| acro (EU, WHO) | KcBERT | 1 / 1,000 | 0.10% (± 0.20) |
| fused (MBC는) | KcBERT | 103 / 1,000 | 10.30% (± 1.92) |
| word1, Korean | KcBERT | 9 / 2,000 | 0.45% (± 0.30) |
| word1, English + particle | KcBERT | 43 / 1,000 | 4.30% (± 1.28) |
| num_en | WikiText-103 | 1 / 1,500 | 0.07% (± 0.13) |
| num_en | Stack Exchange | 4 / 1,501 | 0.27% (± 0.27) |
| collide, mean over 13 keys | Stack Exchange | 103 / 620 | 19.85% (mean of key rates) |
| typo_pos | all four | 83 / 11,679 | 0.71% (± 0.16) |
| typo_shift | NSMC, KcBERT | 3 / 3,105 | 0.10% (± 0.11) |

Per-cell counts for every category and key: `python score.py` on `outputs/nunchi-TR5n-102K.jsonl`.

## Citation

```
@misc{nunchi_bench_2026,
  title  = {Nunchi benchmark: Korean / English input language from 두벌식 keys},
  author = {Kim, Minseok},
  year   = {2026},
  note   = {Version 1, https://huggingface.co/datasets/min-1870/nunchi-bench}
}
```
