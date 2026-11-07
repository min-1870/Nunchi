# Nunchi benchmark: how to run and score an input method

The benchmark asks one thing of a Korean / English input method that picks the language itself: given the keys
someone typed on a 두벌식 keyboard, without pressing 한/영, does the text that ends up in the field read as they meant
it? It is scored word by word, on the words where that is hard, in cells (kinds of word) that are reported apart.

Files: `nunchi_bench.jsonl` (the lines), `score.py` (the scorer, no dependencies), this file.

## A line

```json
{"id": "nb001234", "set": "switch", "src": "kcbert", "key": null,
 "w": [["MOU", "000"], ["rPdirtjeh", "111111111"], ["dksTjTrnaks", "11111111111"]],
 "pts": [0, 1], "cats": ["KO>EN start", "KO>EN after"],
 "text": "MOU 계약서도 안썼구만", "from": {"corpus": "kcbert", "license": "CC BY-SA 4.0", "...": null}}
```

| field | meaning |
|---|---|
| `w` | the words as typed: the keys of each word, and one label per key (`1` Korean, `0` English) |
| `pts` | the words that are scored: indices into `w` |
| `cats` | for `switch` / `end_extra` lines, each scored word's category (beside `pts`); null elsewhere |
| `text` | the line as meant: what the field should hold, words separated by one space |
| `set`, `src`, `key` | the cell the points belong to (below); `key` only for `collide` |
| `from` | where the text comes from and its licence; Stack Exchange lines carry the question and author links |

Keys are QWERTY key names: a lower-case letter is the key, an upper-case letter is the key with Shift (`R` + `k` is
까 in 두벌식), digits are digits. Lines hold no punctuation.

## Typing a line

1. Start from an empty field and a fresh state of the input method (no memory of earlier lines). If it keeps a
   Korean / English state, start in Korean (한) mode, and say so in the report.
2. Type the keys of each word, a Space between words, and one Space after the last word, so that every word is
   followed by a Space. Do not press 한/영, Enter, Backspace or any shortcut.
3. Record the text the field holds when the line is done, as the prediction for the line's `id`.

User learning (personal dictionaries, undo history) is switched off or reset, so every line is typed by a new user.
Speed does not matter: wait for the input method to settle if it works asynchronously.

## Scoring

```
python score.py nunchi_bench.jsonl predictions.jsonl --json scores.json
```

- The prediction and `text` are split on whitespace; a scored word is right when it equals the expected word
  exactly (Unicode NFC). A word holding both languages (`MBC는`) must come out split at the right key.
- A line whose word count differs from `text`, or a line with no prediction, counts all its scored words as wrong;
  the scorer prints how many there were.
- What counts is the final text: a word fixed later in the line (when the next words make the language clear) is
  right.

## Cells

| cell | what is scored | source |
|---|---|---|
| `main` | every word of whole lines, mostly one language: "don't switch wrongly" | NSMC reviews, KcBERT comments, Stack Exchange posts / comments |
| `switch` | words where the language changes (KO>EN / EN>KO; start, middle, end of the line; the word after; number words) | KcBERT (real Korean-base lines); Stack Exchange lines with one Korean word put in (synthetic English-base lines) |
| `end_extra` | more KO>EN end points: a Korean line ending in an English word | KcBERT |
| `amb_en` | English words whose keys are also a common Korean word (so = 내, go = 해) in Korean lines | KcBERT |
| `amb_ko` | Korean words whose keys are a common English word (해 = go) | KcBERT |
| `jamo1` | single jamo used as a word (ㅋ, ㅠ: keys z, b) | KcBERT |
| `acro` | upper-case English with the shape Shift leaves on Korean (EU, WHO) | KcBERT |
| `fused` | words of both languages with no space (QR코드, MBC는) | KcBERT |
| `word1` | a line of one word: Korean (`ko1`), English + particle (`mix1`) | KcBERT |
| `num_en` | words with digits and letters in English lines (5km, 3rd) | WikiText-103, Stack Exchange |
| `collide` | English words whose keys are a Korean marker (mm = ㅡㅡ, z = ㅋ, h = ㅗ), one cell per key | Stack Exchange |
| `typo_pos` | words with one keyboard typo; right = the typo'd keys in the word's language | `main` lines with typos put in |
| `typo_shift` | Korean words with Shift still held after a double consonant (RK = 까) | `main` lines (NSMC, KcBERT) |

Report every cell with its count of points, not one total: the cells measure different things and their sizes are
not their weight in real typing. Rates come with ± 2 SE (about the 95% line); when two methods are run on the same
lines, compare them paired: words only one of them gets wrong, a difference larger than 2 sqrt(their sum) is real.

Lines with profanity, slurs or insults, and lines with web leftovers (URL parts, ids) are not in the set, so the
text is cleaner than real comments are.

## Reporting a result

Name the input method, its version, platform, settings (and the start mode), the date, how the keys were sent
(a script driving the engine, injected key events, a person), and the file's sha256 from `manifest.json`.
Results on another version of the file are not comparable.
