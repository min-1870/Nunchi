# Additions to the model card (huggingface.co/min-1870/nunchi-tr5n-102k)

## 1. Metadata: add under the front matter

```yaml
datasets:
- min-1870/nunchi-bench
```

## 2. Section: put after "Results"

```markdown
## Public benchmark

[Nunchi benchmark](https://huggingface.co/datasets/min-1870/nunchi-bench) v1: 36,166 lines from text that may be
published (NSMC, KcBERT, Stack Exchange, WikiText-103), held out from training, built by the same rules as the dev
sets above, without profanity. % of words wrong in the end, same rule as above. Anyone can score another input
method on it with its `score.py`.

| Cell | Words | Wrong |
|---|---|---|
| Whole lines: Korean (NSMC / KcBERT) / English (Stack Exchange) | 40,006 / 40,022 | 0.04 / 0.01 |
| Korean → English: start / middle / end of the English part | 651 / 1,527 / 1,436 | 2.61 / 0.65 / 2.92 |
| English → Korean: start / middle / end of the Korean part | 1,333 / 1,333 / 1,331 | 0.30 / 3.83 / 9.09 |
| English words that are Korean words too (`so` = 내, `go` = 해) | 2,001 | 5.60 |
| Korean words that are English words too (해 = `go`) | 2,500 | 0.88 |
| Single jamo in Korean text (ㅋ, ㅠ) | 2,502 | 3.88 |
| Both languages with no space (`QR코드`, `MBC는`) | 1,000 | 10.30 |
| A typo in the word / Shift still held | 11,679 / 3,105 | 0.71 / 0.10 |
| English words whose keys are a Korean marker (`mm` = ㅡㅡ, `z` = ㅋ), mean of 13 keys | 620 | 19.85 |

The table above averages each cell over its sources, Korean web text (mC4) among them; this one pools its own
sources, which hold no web text (single jamo: KcBERT 3.56% on dev against 3.88% here, but web 12.18% pulls the
average above to 7.87). Compare models within one table, not across them.
```

Numbers: `python score.py nunchi_bench.jsonl outputs/nunchi-TR5n-102K.jsonl` (2026-11-08). Korean → English end
pools the switching set's 186 points and the 1,250 extra end points (42 wrong of 1,436); whole-line Korean pools
NSMC 12 / 20,001 and KcBERT 5 / 20,005; English 1 / 20,004 and 3 / 20,018.
