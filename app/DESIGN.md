# Nunchi app: design

Decisions for the Windows app. The model is a given: `models/<name>.onnx` plus `models/<name>.json`
(see `nunchi/export_onnx.py`). Everything model-specific the app needs is read from that JSON.

## Approach

- **Keyboard hook, not an IME.** The app sits next to whatever Korean IME the user already has
  (`WH_KEYBOARD_LL`), and corrects by Backspace → switch IME mode → retype. It runs in its own process,
  so a bug never crashes another app, and installing or quitting it changes nothing else.
- **Stack:** Rust. `windows-rs` for Win32, `ort` (ONNX Runtime) for the model, Tauri + React for UI later.
- **Supported input:** 두벌식 on a US QWERTY layout. Anything else: the app does nothing.

## Layout

```
app/
  core/   pure logic, no Windows: key buffer, 두벌식 composer, encoding, ONNX, Viterbi, decision
  win/    Windows: hook, SendInput, IME mode, logging (UI later)
```

`core` is tested against a golden file written by Python (`export_onnx.py`): same keys in, same path and
probabilities out.

## Words

- A word is a run of letter keys (a–z, A–Z), and digits for a model that reads them (`digit0` in its JSON,
  TR2 on): `3tl` is one word and becomes `3시`, `gpt4` stays. In Korean mode a digit ends the syllable
  before it (`rk1` shows 가1). A word of digits only is not judged (it shows the same in both modes) and never
  revised; the threshold for one key counts letters only. Shift+digit (!@#) is a boundary. For a model
  without digit keys, digits are boundaries as before (`3rhk`: `3` stays, `rhk` is judged).
- Space, Enter, punctuation: boundaries. They type the same in both modes, so they are never deleted or
  retyped.
- Tab is not a boundary but a reset: it often moves focus, and a correction would land in the next field.
- Words longer than `max_keys_per_word` (24) are never corrected (URLs, code).

## When to decide

- **Once, at the boundary key, never again.** No waiting for the next word: the model session measured that
  the next word changes fewer than 30 of 14,267 decisions and fixes as many as it breaks.
- The model gets the current word plus the previous `left_words` (3) words of the buffer.
- **Every letter is judged before it shows** (held about 2 ms). After each letter the word so far is judged:
  - surely the other language (whole word one language, not the start of a word the user undid): the word
    goes out in that language right away, deleting what was shown of it, and the rest of it comes out right
    as it is typed. Sent as keys, not text, so the IME keeps composing (`dks` → 안, then `s` makes 안ㄴ,
    not 아 + ㄴ). Once per word;
  - otherwise: the letter shows as typed.
  The boundary still judges the whole word.
- **Unsure letters are shown, not held.** Holding a letter that leans to the other language until the model
  is sure avoids most deletions (simulated: 82% of wrong-mode words fixed before anything wrong showed,
  first character 17 ms late on average when typing in the right mode), but in use, letters that do not
  show feel like lost keys (ㅜㅜ typed in English mode stayed hidden until Space or a 250 ms pause). So the
  default shows them and replaces a word once sure: in the Notepad test 8 of 9 switches deleted 1–3
  characters first. `NUNCHI_HOLD=1` brings holding back (a Backspace then takes back a held letter without
  reaching the app; held letters go out as typed after 150 ms without a key).
- **Except a word's first letter.** Most of those deletions were one character: the first key shown, the
  second key settling it. So a first letter leaning to the other language waits for the second key (or
  150 ms without one, then it shows as typed); from the second key on nothing is held. Notepad test: 8 of 9
  switches with nothing deleted, against 1 of 9 before. Typed in the right mode, about 14% of words lean
  the other way at their first key (simulation) and show that key one keystroke later.
  `NUNCHI_HOLD_FIRST=0` turns it off.
- A single key always needs 0.999, whatever the main threshold (a 1-key word start, or a 1-key word at its
  boundary): at 0.99, 1.7% of sure first-key judgements in a sentence's first word were wrong (Notepad: `so`
  became 내 from its `s`); none at 0.999.
- **The previous word, once more (reviser).** After a Space, the same model judging the word before the last (TR5n-102K, `models/tr5n_102k.onnx`, which decides and revises; before it a separate RV1), the whole
  window, judging the word before the last) may change the previous word, now that the word after it is
  typed. Strict, always on: only that one word, only if the app has not changed it already, only when both
  it and the last word ended with a Space (no Enter: in a chat app it has sent the message; no
  punctuation), and only at the reviser's threshold, 0.99 (on the model session's dev RV1 then fixed 25 of
  TX4's errors and broke none; at 0.9, 70 fixed and 10 broken). The cursor walks left over the last word
  and the Spaces with the arrow keys, the word is replaced as text, and the cursor walks back, so nothing
  after it is retyped (`ㅑ think` → `i think`: TX4 was 0.96 sure of `i` alone, RV1 0.992 with `think`
  after it). Notepad test: the two revision cases fixed, no revision in the other 13.
- **Going back on its own change, once.** A word the app changed whole into the other language (at the
  boundary or in the middle) is judged once more after the next word, under the same limits, and may only go
  back to how it was typed, at the revise threshold (TR5n-102K: 0.8). `so` alone at the start of a field is
  0.92 Korean and becomes 내; with `what` after it the model is 0.95 sure of so, and it goes back. On the
  model session's dev (app rule, act 0.85 / revise 0.8 / split 0.9) going back at 0.8 fixed 16 broken words
  and broke 4 right changes (wrong 1550 -> 1538); at 0.95, 8 and 2. Checked once on test at 0.8: 14 fixed, 2
  broken (1587 -> 1574). Small: one word later the model rarely turns around.

## Rule

At the boundary, with the IME mode the user typed the word in:

| Condition | Action |
|---|---|
| English mode and P(all ko) ≥ threshold | retype as Korean |
| Korean mode and P(all en) ≥ threshold | retype as English |
| Best path switches inside the word and P(path) ≥ split threshold | retype split (`MBCsms` → MBC는) |
| Otherwise | leave it as typed |

Below the threshold the app trusts the user: most typing is already in the right mode.

**The threshold is chosen by wrong corrections per 1,000 words, not by its probability value** (the model's
probabilities are about 2x overconfident). On the model session's dev set (15,634 words):

| Threshold | Wrong corrections / 1,000 words | Words not acted on |
|---|---|---|
| 0.9 | 2.4 | 0.9% |
| **0.99 (default)** | **1.0** | 3.1% |
| 0.999 | 0.3 | 12.8% |

A wrong correction breaks a word the user typed correctly, which is the common case; a word not acted on
only matters when the user was in the wrong mode. 0.999 was the first default; in use it left words like
`hello` typed in Korean mode (0.998) unfixed, so the default is 0.99 with 0.999 kept for single keys (see
When to decide). `targets` in the JSON would map a per-1,000-words target to a threshold for that model;
there is no setting for it (the app is meant to need no interaction).

## Correction

1. **Hold the boundary key** in the hook. Letting it through first would not work for Enter: in a chat app it
   sends the message before the word is fixed.
2. Backspace until the word is gone. The count follows the Korean IME: the syllable still being composed
   loses one key's jamo per Backspace (녕 → 녀 → ㄴ), committed syllables go whole (`dkssud` in Korean mode:
   안 + 3 keys of 녕 = 4). `core::typing` tracks this, including the user's own Backspaces.
3. Switch mode with an injected 한/영 key (VK_HANGUL), type the finished text as Unicode in one go (안녕,
   MBC는: no composing on screen, fewer events than retyping keys), then send the held boundary key. Everything goes through `SendInput`, so the app receives it in
   order. (Setting the mode by message, `WM_IME_CONTROL`, would bypass the input queue and could switch
   before the Backspaces arrive.) Events go one at a time, 4 ms apart: Notepad dropped up to two thirds of
   the Backspaces sent in one batch; with the current sending 2 ms lost or doubled keys and 3 ms passed
   (3 of 3 runs, both modes), so 4 keeps a margin. A 6-key word fixed at its boundary is about 19 events
   (~75 ms); a mid-word switch after 2 keys about 10 (~40 ms).
   Held letters that go out as typed (most keys) skip the pauses before and after: a key shows about 6 ms
   after it is typed (2 ms judgement, 4 ms between its down and up; 4 more per Shift edge).
   The mode is read with `WM_IME_CONTROL` at the start of every word (it can change without a key, e.g. from
   the taskbar), except the word right after a correction, which uses the mode the correction left. Korean
   means the IME is open *and* in native conversion: in the Claude app (Chromium) 한/영 only opens and
   closes the IME while the conversion mode stays native, so reading the conversion mode alone took English
   for Korean and nothing was corrected there.
4. The IME is left in the language of the word's last part.

**Select instead of Backspace.** Steps 2–3 select the word with Shift+Left (one per character on screen;
the composing syllable is committed by the first) and type over it, at the boundary and mid-word, so the
word changes in one step instead of emptying first (it also feels faster). Undo still uses Backspace.
Where Shift+Left selects nothing the text would land after the word, and trying it is not harmless (in cmd
it starts the console's own selection and the next key is lost: `안녕하세요 d`). So it is decided before
any key is sent, from what the field tells UI Automation, with no list of apps:
- the focused element is an edit field or a document (UIA control type Edit or Document; a terminal's
  screen is static Text, as in Windows Terminal's TermControl), and
- it has a caret (one empty selection) with exactly the word as shown right before it.
Then Shift+Left selects exactly the word: select and type over it. No text, no caret, or not editable →
Backspace, for this field. The word not matching → Backspace this time, and check again at the next
correction (the app may not have shown our last key yet). The check is made at the first correction in a
field (2–3 ms in Notepad, ~22 ms the first time UIA is used) and kept until the focus may have moved:
another window, a mouse click, a Ctrl/Alt/Win combination, Tab or the other reset keys.
`NUNCHI_SELECT=0` goes back to Backspace everywhere. (Replacing one character at a time from the right was tried and
dropped: at 4 ms a step it reads as flicker, not as a smooth change.)

**Keys typed during a correction are held** in the hook and replayed after it, in order, key-ups included, so
they never interleave with the correction and land in the new mode. If the worker does not finish within
500 ms, the hook stops holding keys.

Keys the app sends are tagged (`dwExtraInfo`) and ignored by its own hook.

**Undo** is a hotkey (Pause by default): right after a correction, it clears the corrected word and the
boundary, retypes the keys as typed in the original mode, and adds them to the exceptions.

## Exceptions (learned, no settings)

Names, IDs, the repository's own name (`gksdud` is 한영 typed in English) look like the other language to
the model. The app learns them from what the user does anyway:
- **Learn:** the word right after a correction is the same keys typed in the mode the app corrected them
  away from (the user deleted 한영, pressed 한/영 and typed `gksdud` again). It is left as typed, and the keys
  become an exception: never switched in the middle (while its start is typed) or at the boundary, never
  revised. While that word is being typed back, it is not switched midway either.
- **Unlearn:** the word right after an exception was left alone is the same keys typed in the other mode
  (the user wanted 한영 this time). The exception goes; the word left alone before it stays as it is.
- Only the very next word counts; a click or another window drops the watch (Ctrl+Backspace and 한/영 do
  not: that is how a word is typed back).
- Kept in `%LOCALAPPDATA%\Nunchi\exceptions.json`, as HMAC-SHA256 of each prefix only, never the keys: an
  exception can be a piece of a password typed into a field that does not say it is one (a terminal's
  password prompt). The HMAC key is random and stored DPAPI-encrypted for this Windows user. With each
  word, how sure the model was when it was undone, for a softer rule later (follow the list unless the
  model is now much surer); not used yet: turn it on if unlearning shows up often in logs.
- Test runs (with arguments) keep exceptions in memory unless `NUNCHI_EXCEPTIONS` names a file. 제거
  deletes the folder.
- Tested: learn, unlearn, kept across a restart (and corrected again without the file); the file holds no
  key text.

## Habits (short words)

Short words that read fine both ways (so/내, gg/ㅎㅎ, dd/ㅇㅇ, zz/ㅋㅋ) are about half of the model's errors,
and they go wrong in both directions, so no threshold fixes them; each user, though, keeps meaning one of them.
- Counted: for words of up to 3 letters (letters only), the language the word finally ended up in, once it
  can no longer change (two words on, or when the history is dropped: a click, Enter into a new field). A
  change the user undid or typed back counts as what they kept.
- Used: with n uses as Korean and m as English (at least 3 in all), the model's odds between the two
  one-language readings move by ln((n + 2) / (m + 2)), at most 2 either way (about 7x), before the rule. So
  after 6 uses of so in English, 0.92 Korean becomes about 0.82 and stays so at act 0.85. In the middle of
  a word the habit of the keys so far may only hold back a switch (so before som), never start one.
- Counts halve once they pass 40, so a habit can change.
- Kept with the exceptions, as keyed hashes (`habits` in exceptions.json), written at most every 30 s.
  `NUNCHI_HABITS=0` turns it off. Cannot be scored on the eval sets (no per-user data); tested in the core
  tests and in Notepad (six `so what`, then `so` on a new line stays so).

## Pacing

The app and its IME drop or misplace keys sent too close together, so every event we send waits 4 ms after
the last key event that reached the app, ours or the user's (`keys::wait_gap`); after a quiet moment that is
no wait at all. The exception is Unicode text, which no IME composes: no pause. After a correction (a batch of
events) the next event keeps 4 ms more, recorded instead of slept (`settle`), and the hook holds a key of
the user's that comes within the pause, so the worker never sleeps after a batch. Measured on Notepad
(mid-word correction, Backspace right after it, undo, digits; 72 cases a run): Shift+Left at 0, 1, 2 ms lost
or misplaced keys (4, 3, 1 of 72), none at 4; without the extra pause after a batch, 2 of 216; with it, 0 of
216. Per key in the profile (NUNCHI_PROFILE=1): Space 14 -> 5 ms, a held first letter shown with the second
22 -> 18 ms (12 of it the three pauses between the four events), `10개` corrected mid-word 97 -> 49 ms.

A word of 3 or more characters is selected with Ctrl+Shift+Left (6 events, ~26 ms with the check) instead of
a Shift+Left per character (2 per character + 2: 56 ms for 6). What got selected is read back with UI
Automation and fixed up to the word exactly (Shift+Left for a word the app split, Shift+Right for digits left
before it); anything else: Right puts the caret back, the field falls back to Shift+Left per character.
Notepad: 31 of 31 selected the word exactly; Edge: the slowest Space with a correction 118 -> 55 ms.
`NUNCHI_WORD_SELECT=0` turns it off.

When a click ends or the focus moves to another element (WinEvent focus hook, compared with UI Automation so a
popup does not count), the worker waits 30 ms for things to settle and then looks at the field before any key:
password or not, and the line around the cursor (40-80 ms in Notepad and Edge, none of it on the first key
now). UI Automation is started when the worker starts (its first use costs ~25 ms).

## Buffer resets

Clear the buffer (nothing before it will be touched again) on:

- mouse click (a low-level mouse hook), arrows, Home/End, PageUp/Down, focus change
- Ctrl / Alt / Win combinations, Shift + arrows
- the user pressing 한/영 themselves (the following words keep the mode they chose)

Backspace by the user removes what the IME removes (see Correction); past the start of the word it reopens the
previous word, which is judged again at its next boundary.

## Context from the screen

After a reset the model would judge the next word with nothing before it, the weakest case. So when a word
starts with nothing known (a new field, a click, an arrow, 한/영), the line the cursor is on is read with UI
Automation (200 characters each way, cut at line breaks; edit fields and documents only, never a password
field) and turned back into keys (`compose::words_of`: Hangul to 두벌식 keys; spaces, punctuation, emoji and
other scripts end words).

- Words before the cursor go into the history as context: never revised, and Backspace into one forgets the
  line. The cursor right after a word (no space): that word is dropped, the typed keys join it.
- Words after the cursor (up to 5) are read only for a model that sees the whole window (TR). They are given
  at the word's boundary only: while typing, the word is incomplete, and the model was never trained on an
  incomplete word with words after it. At the boundary the typed word is whole and separate from them.
- Cost: one read per reset, about 2.5 ms in Notepad (5.4 ms the most of 28). What is read is never logged.
  `NUNCHI_CONTEXT=0` turns it off.

## Never act

- Elevated windows: UIPI drops keys sent to them, so a held boundary key would be lost. The hook never holds
  keys there (a window whose process token cannot be read counts as elevated).
- Windows where the IME mode cannot be read.
- CapsLock in Korean mode (the IME types English then). In English mode with CapsLock, keys are kept as
  Shift typed them (their 두벌식 meaning) and only judged at the boundary: a switch in the middle of a word
  would retype keys in Korean mode with CapsLock on. Text put back as English gets CapsLock's case.
  Not tested automatically: CapsLock sent by a program does not toggle on this PC.
- Password fields (UI Automation IsPassword, asked at the first word in a field, and again after Enter or
  Tab: a login form moves on to its password field): keys go through as typed, nothing is judged or logged.
  Tested with a Windows password box next to a plain one. Not covered: a password typed where nothing says
  so, such as a terminal's `sudo`/`ssh` prompt (terminals stay corrected, by choice); what is learned from
  it is stored hashed (see Exceptions).
- The window's input language is not Korean (an English-only keyboard, a Japanese or Chinese IME): the mode
  counts as unknown and keys pass through untouched. Tested by switching Notepad to en-AU and back.

## Running

- In the tray, no window. Right-click: 시작 / 정지 / Windows 시작 시 실행 (check mark) / 제거 / 종료.
  Stopped, the hook lets every key through; started again, what it knew about the screen is dropped.
- Installs itself: run without arguments from anywhere but `%LOCALAPPDATA%\Programs\Nunchi`, it copies the
  exe, `DirectML.dll`, `models/tr5n_102k.onnx` (+ `.json`; a separate reviser only for a model that does not revise itself) there, starts that copy and exits, so the download
  can be deleted. A copy already running there is told to quit first (the tray's quit command), so running
  a newer download is the update. Runs with arguments (tests, another model) stay where they are and do not
  touch autostart. No admin rights needed.
- One copy at a time (a named mutex): a second hook would fix every word twice.
- Starts with Windows: `Nunchi` under `HKCU\...\CurrentVersion\Run`, set on the installed copy's first run
  (marked by `HKCU\Software\Nunchi\AutostartSetUp`), then only the tray's check mark changes it.
- 제거 asks first, then removes the autostart entry, `HKCU\Software\Nunchi` and the install folder (a hidden
  shell deletes it a moment after the process exits).
- Tested: install from a folder that is then deleted, update over a running copy, uninstall; the Notepad
  test (with arguments) installs nothing.
- Icon: a text cursor with a dot (noticing while you type). The exe's icon is a dark tile with a coral dot;
  the tray icon has the cursor in white or dark to match the taskbar (re-read on a theme change) and the coral dot; when
  stopped the dot goes and the cursor greys. Drawn by `win/icons/make_icons.py` (strokes snapped to whole
  pixels at tray sizes), compiled in through `win/nunchi.rc`.
- The model is `models/tr5n_102k.onnx` next to the exe or in a folder above it, or the path given as the first
  argument; an optional second argument is the threshold.

## Model contract (`models/<name>.json`)

| Field | Meaning |
|---|---|
| `vocab`, `max_keys_per_word`, `max_stream_keys`, `left_words` | input encoding |
| `crf` | `trans`, `start`, `end` for Viterbi |
| `lookahead` | words after the judged word the model expects (0) |
| `temperature` | divide emissions by this before Viterbi (1.0 = none) |
| `act_threshold`, `split_threshold` | default thresholds |
| `targets` | wrong corrections per 1,000 words → threshold |

ONNX Runtime runs with one intra-op thread (four threads were 30x slower in the model session's test).

## Logging

Off unless `NUNCHI_DEBUG=1`: the log holds what was typed. Then one line per decided word (keys, mode,
P(all ko), P(all en), best path and P, action, reason) goes to stdout and
`%LOCALAPPDATA%\Nunchi\decisions.jsonl`.
