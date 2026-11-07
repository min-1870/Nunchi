//! What the user has typed, as the screen shows it, and the corrections that follow.
//!
//! The hook holds every letter until the app has decided on it (about 2 ms). After each letter (`key`) the
//! word so far is judged: surely the other language, the word is shown in that language right away (with
//! Backspaces for what was shown already); otherwise the letter is shown as typed, except a word's first
//! letter when it leans to the other language (`hold_first`): it waits for the second key, which usually
//! settles it, so the usual one wrong character never shows. With `hold_unsure`, every letter leaning to the
//! other language but not sure stays held (no deletion later, but keys that do not show feel lost). At a
//! boundary key (also held) the whole word is judged once more. Backspace counts follow the Korean IME:
//! inside the syllable being composed one Backspace removes one key's jamo (녕 -> 녀), a committed syllable
//! goes whole.
//!
//! The user going against us is learned from: when the word right after a correction is the same keys
//! typed in the mode the app corrected them away from (the user pressed 한/영 and typed it again), those
//! keys become an exception and are left as typed; when the word right after an exception was left alone
//! is the same keys typed in the other mode, the exception goes.
//!
//! Short words that read fine both ways (so/내, gg/ㅎㅎ, dd/ㅇㅇ) are where the model cannot know: each user
//! keeps meaning one of them. So the language each short word finally ended up in is counted (once the word
//! can no longer change), and those counts shift the model's odds for the same keys (`prior`).

use crate::compose::{compose, compose_str};
use crate::crf::{Judgement, EN, KO};
use crate::decide::{decide, Action, Mode, Reason, Thresholds};
use crate::exceptions::{Exceptions, Memory};

const HISTORY: usize = 8;
/// Words after the cursor given to the model at a boundary: as many as the model session's revision rule
/// waits for at most.
const AFTER: usize = 5;

/// The habit prior: words of up to HABIT_LETTERS letters, once used HABIT_MIN times, get
/// logit(p_ko) += HABIT_LAMBDA * ln((n_ko + HABIT_A) / (n_en + HABIT_A)), at most HABIT_CAP either way.
const HABIT_LETTERS: usize = 3;
const HABIT_MIN: f64 = 3.0;
const HABIT_A: f64 = 2.0;
const HABIT_LAMBDA: f64 = 1.0;
const HABIT_CAP: f64 = 2.0;

/// Digits at the start of `keys`: they show the same in both modes, so a correction leaves them on screen
/// (3시 typed in English mode: only `tl` is replaced).
fn lead_digits(keys: &str) -> usize {
    keys.chars().take_while(|c| c.is_ascii_digit()).count()
}

/// `parts` without their first `n` keys.
fn drop_front(parts: &[(Mode, String)], mut n: usize) -> Vec<(Mode, String)> {
    let mut out = Vec::new();
    for (m, k) in parts {
        if n >= k.len() {
            n -= k.len();
        } else {
            out.push((*m, k[n..].to_string()));
            n = 0;
        }
    }
    out
}

/// Letter keys in `keys` (digits are keys too, but say nothing about the language alone).
/// A short word of letters only: where the user's habit counts.
fn short(keys: &str) -> bool {
    keys.chars().all(|c| c.is_ascii_alphabetic()) && (1..=HABIT_LETTERS).contains(&keys.len())
}

fn letters(keys: &str) -> usize {
    keys.chars().filter(|c| c.is_ascii_alphabetic()).count()
}

#[derive(Debug, Clone)]
struct Word {
    keys: String,
    /// None: corrected into parts of both languages
    mode: Option<Mode>,
    /// keys[..frozen] are committed characters, each removed whole by one Backspace
    frozen: usize,
    /// switched to the other mode in the middle of the word: the mode it was started in
    early_from: Option<Mode>,
    /// the last `held` keys are held back, not on screen yet
    held: usize,
    /// once committed (in the history): characters it shows, whether the app changed it, and whether a
    /// Space followed it (only then is it a candidate for revision)
    on_screen: usize,
    fixed: bool,
    space_after: bool,
    /// the app changed the whole word to the other language: the mode it was typed in, to go back to once if
    /// the word after it shows the change was wrong
    revert_to: Option<Mode>,
    /// counted into the habits already
    counted: bool,
}

impl Word {
    fn new(mode: Mode) -> Self {
        Word {
            keys: String::new(),
            mode: Some(mode),
            frozen: 0,
            early_from: None,
            held: 0,
            on_screen: 0,
            fixed: false,
            space_after: false,
            revert_to: None,
            counted: false,
        }
    }

    /// Keys on screen (letters are ASCII, so bytes are keys).
    fn shown(&self) -> &str {
        &self.keys[..self.keys.len() - self.held]
    }

    fn take_held(&mut self) -> String {
        let s = self.keys[self.keys.len() - self.held..].to_string();
        self.held = 0;
        s
    }
}

/// What to do with the held letters after a key.
#[derive(Debug, Clone)]
pub enum KeyAction {
    /// keep holding them
    Hold,
    /// show these keys as typed
    Release(String),
    /// show the word in the other language now
    Switch(Outcome),
}

/// Backspaces that clear `keys` typed in `mode`, with keys[..frozen] committed and the rest still composing.
fn clear_count(keys: &str, mode: Mode, frozen: usize) -> usize {
    match mode {
        Mode::En => keys.chars().count(),
        Mode::Ko => {
            let done = compose(&keys[..frozen]).len();
            let tail = compose(&keys[frozen..]);
            match tail.last() {
                None => done,
                Some(last) => done + tail.len() - 1 + (last.end - last.start),
            }
        }
    }
}

/// What `keys` typed in `mode` show, with keys[..frozen] committed and the rest still composing (CapsLock
/// aside).
fn on_screen(keys: &str, mode: Mode, frozen: usize) -> String {
    match mode {
        Mode::En => keys.to_string(),
        Mode::Ko => compose_str(&keys[..frozen]) + &compose_str(&keys[frozen..]),
    }
}

/// Characters `parts` show once committed.
fn shown(parts: &[(Mode, String)]) -> usize {
    parts.iter().map(|(m, k)| clear_count(k, *m, k.len())).sum()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Backspaces to send first
    pub backspaces: usize,
    /// or: the word as it shows, to select (the composing syllable is one character)
    pub shown: String,
    /// then these keys, each part in its mode
    pub parts: Vec<(Mode, String)>,
    /// then the boundary key: the held one for a correction, retyped for an undo
    pub retype_boundary: bool,
    /// the IME mode to leave
    pub mode: Mode,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub keys: String,
    pub typed: Mode,
    pub judgement: Option<Judgement>,
    pub reason: Reason,
    pub plan: Option<Plan>,
    /// with no plan: held keys to show as typed before the boundary
    pub release: String,
}

/// Changing a word before the last one: Left `left` times, Backspace `backspaces` times, type `parts`, Right
/// `right` times.
#[derive(Debug, Clone, PartialEq)]
pub struct Revision {
    pub left: usize,
    pub backspaces: usize,
    pub parts: Vec<(Mode, String)>,
    pub right: usize,
    pub keys: String,
    pub from: Mode,
    /// going back to how the app's own change was typed
    pub revert: bool,
}

/// After a correction (`learn`) or an exception left alone: the next word, if it is `keys` typed in `mode`,
/// is the user typing it the other way.
#[derive(Debug, Clone)]
struct Watch {
    keys: String,
    mode: Mode,
    learn: bool,
    /// how sure the model was of the correction
    p: f64,
}

fn other(m: Mode) -> Mode {
    match m {
        Mode::En => Mode::Ko,
        Mode::Ko => Mode::En,
    }
}

#[derive(Debug, Clone)]
struct Fix {
    keys: String,
    typed: Mode,
    /// what replaced keys[keep..] (the digits before it were left on screen)
    parts: Vec<(Mode, String)>,
    keep: usize,
}

pub struct Typing {
    history: Vec<Word>,
    cur: Option<Word>,
    /// the last correction, undoable until anything else is typed
    last_fix: Option<Fix>,
    /// keys the user undid a correction of: never corrected again
    pub exceptions: Box<dyn Exceptions>,
    /// the word just corrected or left as an exception, to see whether the user types it the other way
    watch: Option<Watch>,
    /// hold letters that lean to the other language until sure, instead of showing them as typed
    pub hold_unsure: bool,
    /// hold a word's first letter while it leans to the other language (the app shows it after a short
    /// pause if no second key comes)
    pub hold_first: bool,
    /// words on screen after the cursor (read when the cursor came), given to the model at a boundary
    after: Vec<String>,
    /// shift the model's odds for short words by what the user ended up with before
    pub habits: bool,
}

impl Default for Typing {
    fn default() -> Self {
        Self::new()
    }
}

impl Typing {
    pub fn new() -> Self {
        Typing {
            history: Vec::new(),
            cur: None,
            last_fix: None,
            exceptions: Box::new(Memory::default()),
            watch: None,
            hold_unsure: false,
            hold_first: false,
            after: Vec::new(),
            habits: true,
        }
    }

    /// Show every held key as typed (before a reset, or when the user pauses): returns them.
    pub fn take_held(&mut self) -> String {
        self.cur.as_mut().map(Word::take_held).unwrap_or_default()
    }

    pub fn has_held(&self) -> bool {
        self.cur.as_ref().is_some_and(|w| w.held > 0)
    }

    /// Forget the words on screen: nothing before the cursor is touched again. Held keys are dropped: take
    /// them first. The last correction is still watched (the user may press 한/영, or delete it with
    /// Ctrl+Backspace, and type it again).
    pub fn reset(&mut self) {
        for i in 0..self.history.len() {
            self.count(i);
        }
        self.history.clear();
        self.cur = None;
        self.last_fix = None;
        self.after.clear();
    }

    /// Nothing known about the text around the cursor: a word is about to start with no words before it.
    pub fn wants_context(&self) -> bool {
        self.cur.is_none() && self.history.is_empty()
    }

    /// Words on screen around the cursor (as keys, see `compose::words_of`), read before a word starts:
    /// the model reads them as the words before and after the one typed. They are never changed (no mode:
    /// Backspace into one forgets the line, as for a word split across modes).
    pub fn set_context(&mut self, before: Vec<String>, after: Vec<String>) {
        let skip = before.len().saturating_sub(HISTORY);
        self.history = before
            .into_iter()
            .skip(skip)
            .map(|keys| Word { frozen: keys.len(), fixed: true, mode: None, keys, ..Word::new(Mode::En) })
            .collect();
        self.after = after.into_iter().take(AFTER).collect();
    }

    /// Words after the judged one in what `boundary` gives the model.
    pub fn after_len(&self) -> usize {
        self.after.len()
    }

    /// The cursor went somewhere else (a click, another window): forget everything.
    pub fn moved(&mut self) {
        self.reset();
        self.watch = None;
    }

    /// The current word is the start of the watched one, typed back the way the user had it.
    fn retyping_back(&self, keys: &str, mode: Mode) -> bool {
        self.watch.as_ref().is_some_and(|w| w.learn && w.mode == mode && w.keys.starts_with(keys))
    }

    /// A word is being typed (the next letter does not start one).
    pub fn in_word(&self) -> bool {
        self.cur.is_some()
    }

    /// A letter key typed in `mode`, held until `key` says otherwise. Returns keys held from before, to show
    /// as typed, if the mode changed inside a word without us seeing the toggle (the word starts over).
    pub fn letter(&mut self, c: char, mode: Mode) -> String {
        self.last_fix = None;
        let mut released = String::new();
        if let Some(w) = &self.cur {
            if w.mode != Some(mode) && !w.keys.is_empty() {
                released = self.take_held();
                self.reset();
            }
        }
        let w = self.cur.get_or_insert_with(|| Word::new(mode));
        w.mode = Some(mode);
        w.keys.push(c);
        w.held += 1;
        released
    }

    /// The user pressed Backspace. False: it removed a held key, so the app must not get it.
    pub fn backspace(&mut self) -> bool {
        self.last_fix = None;
        if let Some(w) = &mut self.cur {
            if w.held > 0 {
                w.keys.pop();
                w.held -= 1;
                if w.keys.is_empty() {
                    self.cur = None;
                }
                return false;
            }
        }
        let Some(w) = &mut self.cur else {
            // the boundary goes: the previous word is being typed again
            match self.history.pop() {
                Some(prev) if prev.mode.is_some() => self.cur = Some(prev),
                Some(_) => self.reset(), // a word split across both modes: not tracked
                None => {}
            }
            return true;
        };
        match w.mode {
            Some(Mode::En) | None => {
                w.keys.pop();
            }
            Some(Mode::Ko) => {
                if w.keys.len() > w.frozen {
                    // only the last syllable is still composing; the ones before it are committed, and
                    // stay separate even where greedy composition would merge them again (가ㄱ is not 각)
                    let tail = compose(&w.keys[w.frozen..]);
                    w.frozen += tail.last().unwrap().start;
                    w.keys.pop();
                } else if let Some(last) = compose(&w.keys[..w.frozen]).last() {
                    w.keys.truncate(last.start);
                    w.frozen = last.start;
                }
            }
        }
        if w.keys.is_empty() {
            self.cur = None;
        }
        true
    }

    /// The word history[i] can no longer change: count the language it ended up in, once (short words of
    /// letters only, in one language).
    fn count(&mut self, i: usize) {
        let w = &mut self.history[i];
        let Some(mode) = w.mode else { return };
        if w.counted || !self.habits || !short(&w.keys) {
            return;
        }
        w.counted = true;
        let keys = w.keys.clone();
        self.exceptions.count(&keys, mode);
    }

    /// The model's judgement of `keys`, shifted by the user's habit for them.
    fn prior(&self, keys: &str, mut j: Judgement) -> Judgement {
        if !self.habits || !short(keys) {
            return j;
        }
        let (ko, en) = self.exceptions.habit(keys);
        if ko + en < HABIT_MIN || j.p_ko <= 0.0 || j.p_en <= 0.0 {
            return j;
        }
        let shift = (HABIT_LAMBDA * ((ko + HABIT_A) / (en + HABIT_A)).ln()).clamp(-HABIT_CAP, HABIT_CAP);
        // between the two one-language readings only: what the paths that change language get stays
        let sum = j.p_ko + j.p_en;
        let r = (j.p_ko / j.p_en).ln() + shift;
        j.p_ko = sum / (1.0 + (-r).exp());
        j.p_en = sum - j.p_ko;
        if j.path.iter().all(|&l| l == j.path[0]) {
            let (label, p) = if j.p_ko > j.p_en { (KO, j.p_ko) } else { (EN, j.p_en) };
            j.path.iter_mut().for_each(|l| *l = label);
            j.p_best = p;
        }
        j
    }

    /// Words for the model: the history and the current word, last one judged.
    fn context(&self) -> Vec<&str> {
        self.history.iter().map(|w| w.keys.as_str()).chain(self.cur.iter().map(|w| w.keys.as_str())).collect()
    }

    /// After a letter: judge the word so far and decide on the held letters. Switching happens once per
    /// word and only for a word wholly in one language (MBC는 waits for its boundary); the plan retypes the
    /// keys as keys in the new mode, so the IME keeps composing and the next keys join in.
    pub fn key<E>(
        &mut self,
        t: &Thresholds,
        judge: impl FnOnce(&[&str]) -> Result<Judgement, E>,
    ) -> Result<KeyAction, E> {
        let Some(w) = &mut self.cur else { return Ok(KeyAction::Hold) };
        if w.held == 0 {
            return Ok(KeyAction::Release(String::new()));
        }
        if w.early_from.is_some() || w.keys.chars().count() > t.max_keys || letters(&w.keys) == 0 {
            return Ok(KeyAction::Release(w.take_held())); // already switched, or too long to judge
        }
        let keys = w.keys.clone();
        let typed = w.mode.expect("current word has a mode");
        if self.exceptions.starts(&keys) || self.retyping_back(&keys, typed) {
            return Ok(KeyAction::Release(self.take_held())); // it may become a word the user undid
        }
        let j = judge(&self.context())?;
        let (target, p) = if j.p_ko > j.p_en { (Mode::Ko, j.p_ko) } else { (Mode::En, j.p_en) };
        // the keys so far may be a short word the user keeps the other way (so, before som): the habit may
        // hold back a switch in the middle (the boundary decides then), never start one
        let h = self.prior(&keys, j.clone());
        let p = if (h.p_ko > h.p_en) == (target == Mode::Ko) { p.min(h.p_ko.max(h.p_en)) } else { 0.0 };
        let hold = self.hold_unsure || self.hold_first && keys.chars().count() == 1;
        let w = self.cur.as_mut().unwrap();
        if target == typed {
            return Ok(KeyAction::Release(w.take_held()));
        }
        if j.path.iter().any(|&l| l != j.path[0]) || p < t.early_for(letters(&keys)) {
            return Ok(if hold { KeyAction::Hold } else { KeyAction::Release(w.take_held()) });
        }
        let k = lead_digits(&keys).min(w.shown().len());
        let plan = Plan {
            backspaces: clear_count(&w.shown()[k..], typed, w.frozen.saturating_sub(k)),
            shown: on_screen(&w.shown()[k..], typed, w.frozen.saturating_sub(k)),
            parts: vec![(target, keys[k..].to_string())],
            retype_boundary: false,
            mode: target,
        };
        *w = Word { keys: keys.clone(), mode: Some(target), early_from: Some(typed), ..Word::new(target) };
        Ok(KeyAction::Switch(Outcome {
            keys,
            typed,
            judgement: Some(j),
            reason: Reason::Early,
            plan: Some(plan),
            release: String::new(),
        }))
    }

    /// A boundary key (Space, Enter, punctuation; `space`: it was Space): judge the current word. `judge` runs
    /// the model.
    pub fn boundary<E>(
        &mut self,
        t: &Thresholds,
        space: bool,
        judge: impl FnOnce(&[&str]) -> Result<Judgement, E>,
    ) -> Result<Option<Outcome>, E> {
        self.last_fix = None;
        let Some(mut w) = self.cur.take() else { return Ok(None) };
        let typed = w.mode.expect("current word has a mode");
        if let Some(watch) = self.watch.take() {
            if watch.keys == w.keys && watch.mode == typed && w.early_from.is_none() {
                if watch.learn {
                    self.exceptions.add(&w.keys, watch.p);
                } else {
                    self.exceptions.remove(&w.keys);
                }
            }
        }
        let out = if letters(&w.keys) == 0 {
            // digits only: they show the same in both modes
            let release = w.take_held();
            Outcome { keys: w.keys.clone(), typed, judgement: None, reason: Reason::SameAsTyped, plan: None, release }
        } else if self.exceptions.contains(&w.keys) {
            let release = w.take_held();
            self.watch = Some(Watch { keys: w.keys.clone(), mode: other(typed), learn: false, p: 0.0 });
            Outcome { keys: w.keys.clone(), typed, judgement: None, reason: Reason::Exception, plan: None, release }
        } else {
            self.cur = Some(w.clone());
            let mut words = self.context();
            words.extend(self.after.iter().map(String::as_str));
            let j = judge(&words);
            self.cur = None;
            let j = self.prior(&w.keys, j?);
            let (action, reason) = decide(&w.keys, typed, &j, t);
            let plan = match action {
                Action::Keep => {
                    if let Some(from) = w.early_from {
                        // switched in the middle of the word: that is the correction to undo
                        self.last_fix = Some(Fix { keys: w.keys.clone(), typed: from, parts: vec![(typed, w.keys.clone())], keep: 0 });
                    }
                    None
                }
                Action::Retype(parts) => {
                    let k = lead_digits(&w.keys).min(w.shown().len());
                    let new = drop_front(&parts, k);
                    let p = Plan {
                        backspaces: clear_count(&w.shown()[k..], typed, w.frozen.saturating_sub(k)),
                        shown: on_screen(&w.shown()[k..], typed, w.frozen.saturating_sub(k)),
                        mode: parts.last().unwrap().0,
                        parts: new.clone(),
                        retype_boundary: true,
                    };
                    w.mode = if parts.len() == 1 { Some(parts[0].0) } else { None };
                    self.last_fix =
                        Some(Fix { keys: w.keys.clone(), typed: w.early_from.unwrap_or(typed), parts: new, keep: k });
                    Some(p)
                }
            };
            // with a plan the held keys are part of the new text; without, they go out as typed
            let release = if plan.is_some() { String::new() } else { w.take_held() };
            w.held = 0;
            if plan.is_some() || w.early_from.is_some() {
                let mode = w.early_from.unwrap_or(typed);
                self.watch = Some(Watch { keys: w.keys.clone(), mode, learn: true, p: j.p_ko.max(j.p_en) });
            }
            Outcome { keys: w.keys.clone(), typed, judgement: Some(j), reason, plan, release }
        };
        w.frozen = w.keys.len(); // the boundary commits the word
        // settled: changed by the app, or left alone as an exception (not revised later either)
        w.fixed = out.plan.is_some() || w.early_from.is_some() || out.reason == Reason::Exception || out.judgement.is_none();
        // changed whole into the other language (at the boundary or in the middle): may go back once
        let from = w.early_from.unwrap_or(typed);
        w.revert_to = (out.judgement.is_some() && w.mode == Some(other(from))).then_some(from);
        w.on_screen = match &out.plan {
            Some(p) => lead_digits(&w.keys) + shown(&p.parts), // the digits left on screen, then the new text
            None => shown(&[(w.mode.expect("kept words have a mode"), w.keys.clone())]),
        };
        w.space_after = space;
        // the word before the last is past revising: what it shows now is what the user kept
        if self.history.len() >= 2 {
            self.count(self.history.len() - 2);
        }
        self.history.push(w);
        if self.history.len() > HISTORY {
            self.history.remove(0);
        }
        Ok(Some(out))
    }

    /// Right after a boundary: let a reviser (`judge` runs it on every word in the history; it judges the word
    /// before the last) change the previous word, now that the word after it is typed. Strict: only that one
    /// word, only if the app has not changed it already, only between Spaces (the cursor can then walk back
    /// over plain characters), and only when sure. The plan moves the cursor left past the last word and the
    /// Spaces, replaces the word, and moves back.
    pub fn revise<E>(
        &mut self,
        t: &Thresholds,
        judge: impl FnOnce(&[&str]) -> Result<Judgement, E>,
    ) -> Result<Option<(Revision, Judgement)>, E> {
        let n = self.history.len();
        if self.cur.is_some() || n < 2 {
            return Ok(None);
        }
        let (prev, last) = (&self.history[n - 2], &self.history[n - 1]);
        let Some(typed) = prev.mode else { return Ok(None) };
        // a word the app changed is looked at once more, and may only go back to how it was typed
        let back = prev.revert_to.filter(|&m| m == other(typed));
        if prev.fixed && back.is_none()
            || !prev.space_after
            || !last.space_after
            || prev.keys.chars().count() > t.max_keys
        {
            return Ok(None);
        }
        if self.exceptions.contains(&prev.keys) {
            return Ok(None);
        }
        self.history[n - 2].revert_to = None;
        let (prev, last) = (&self.history[n - 2], &self.history[n - 1]);
        let words: Vec<&str> = self.history.iter().map(|w| w.keys.as_str()).collect();
        let keys = prev.keys.clone();
        let j = self.prior(&keys, judge(&words)?);
        let (target, p) = if j.p_ko > j.p_en { (Mode::Ko, j.p_ko) } else { (Mode::En, j.p_en) };
        if target == typed || p < t.act || j.path.iter().any(|&l| l != j.path[0]) {
            return Ok(None);
        }
        let past = last.on_screen + 2; // the Space after the last word, the last word, the Space before it
        let r = Revision {
            left: past,
            backspaces: prev.on_screen,
            parts: vec![(target, prev.keys.clone())],
            right: past,
            keys: prev.keys.clone(),
            from: typed,
            revert: prev.fixed,
        };
        let prev = &mut self.history[n - 2];
        prev.mode = Some(target);
        prev.fixed = true;
        prev.on_screen = shown(&r.parts);
        self.last_fix = None;
        Ok(Some((r, j)))
    }

    /// Undo the last correction if nothing was typed since: back to the keys as typed, never correct them
    /// again. The plan clears the corrected word and the boundary after it.
    pub fn undo(&mut self) -> Option<Plan> {
        let fix = self.last_fix.take()?;
        if let Some(w) = self.history.last_mut() {
            w.mode = Some(fix.typed);
        }
        self.exceptions.add(&fix.keys, 1.0);
        self.watch = None;
        Some(Plan {
            backspaces: shown(&fix.parts) + 1,
            shown: String::new(), // undo always uses Backspace
            parts: vec![(fix.typed, fix.keys[fix.keep..].to_string())],
            retype_boundary: true,
            mode: fix.typed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crf::{EN, KO};

    const T: Thresholds = Thresholds { act: 0.999, early: 0.999, first_key: 0.999, split: 0.9, max_keys: 24 };

    fn j(path: &[u8], p_best: f64, p_ko: f64, p_en: f64) -> Judgement {
        Judgement { path: path.to_vec(), p_best, p_ko, p_en }
    }

    /// Type keys that go on screen as typed (each released right away).
    fn typ(t: &mut Typing, s: &str, mode: Mode) {
        for c in s.chars() {
            t.letter(c, mode);
            t.take_held();
        }
    }

    #[test]
    fn shows_unsure_keys_then_replaces_them() {
        let mut t = Typing::new();
        let key = |t: &mut Typing, c: char, jj: Judgement| {
            t.letter(c, Mode::En);
            t.key(&T, |_| Ok::<_, ()>(jj)).unwrap()
        };
        assert!(matches!(key(&mut t, 'd', j(&[KO], 0.9, 0.9, 0.1)), KeyAction::Release(s) if s == "d"));
        assert!(matches!(key(&mut t, 'k', j(&[KO; 2], 0.99, 0.99, 0.01)), KeyAction::Release(s) if s == "k"));
        let KeyAction::Switch(o) = key(&mut t, 's', j(&[KO; 3], 0.9999, 0.9999, 0.0)) else { panic!() };
        let p = o.plan.unwrap();
        assert_eq!((p.backspaces, p.parts), (2, vec![(Mode::Ko, "dks".to_string())])); // dk was on screen
    }

    #[test]
    fn holds_only_the_first_key() {
        let mut t = Typing::new();
        t.hold_first = true;
        let key = |t: &mut Typing, c: char, jj: Judgement| {
            t.letter(c, Mode::En);
            t.key(&T, |_| Ok::<_, ()>(jj)).unwrap()
        };
        // first key leaning Korean: held; second key sure: switched with nothing to delete
        assert!(matches!(key(&mut t, 'd', j(&[KO], 0.9, 0.9, 0.1)), KeyAction::Hold));
        let KeyAction::Switch(o) = key(&mut t, 'k', j(&[KO; 2], 0.9999, 0.9999, 0.0)) else { panic!() };
        assert_eq!(o.plan.unwrap().backspaces, 0);
        // second key still unsure: both go out as typed, nothing held from the second key on
        let mut t = Typing::new();
        t.hold_first = true;
        assert!(matches!(key(&mut t, 's', j(&[KO], 0.9, 0.9, 0.1)), KeyAction::Hold));
        assert!(matches!(key(&mut t, 'o', j(&[KO; 2], 0.93, 0.93, 0.07)), KeyAction::Release(s) if s == "so"));
        // a first key leaning to the typed mode is not held
        let mut t = Typing::new();
        t.hold_first = true;
        assert!(matches!(key(&mut t, 'h', j(&[EN], 0.8, 0.2, 0.8)), KeyAction::Release(s) if s == "h"));
    }

    #[test]
    fn holds_while_leaning_other_then_switches_before_showing() {
        let mut t = Typing::new();
        t.hold_unsure = true;
        let key = |t: &mut Typing, c: char, mode: Mode, jj: Judgement| {
            t.letter(c, mode);
            t.key(&T, |_| Ok::<_, ()>(jj)).unwrap()
        };
        // leaning Korean but not sure: held
        assert!(matches!(key(&mut t, 'd', Mode::En, j(&[KO], 0.9, 0.9, 0.1)), KeyAction::Hold));
        assert!(matches!(key(&mut t, 'k', Mode::En, j(&[KO; 2], 0.99, 0.99, 0.01)), KeyAction::Hold));
        // sure: switch; nothing was shown, so no Backspace
        let KeyAction::Switch(o) = key(&mut t, 's', Mode::En, j(&[KO; 3], 0.9999, 0.9999, 0.0)) else { panic!() };
        let p = o.plan.unwrap();
        assert_eq!((p.backspaces, p.parts), (0, vec![(Mode::Ko, "dks".to_string())]));
        // after the switch, keys go out as typed
        assert!(matches!(key(&mut t, 's', Mode::Ko, j(&[KO; 4], 0.5, 0.5, 0.5)), KeyAction::Release(s) if s == "s"));
        typ(&mut t, "ud", Mode::Ko);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 6], 0.9999, 0.9999, 0.0))).unwrap().unwrap();
        assert_eq!((o.reason, o.plan.is_none()), (Reason::SameAsTyped, true));
        let u = t.undo().unwrap();
        assert_eq!((u.backspaces, u.parts), (3, vec![(Mode::En, "dkssud".to_string())])); // 안녕 + boundary
    }

    #[test]
    fn releases_when_leaning_to_the_typed_mode() {
        let mut t = Typing::new();
        t.letter('h', Mode::En);
        t.letter('e', Mode::En);
        let a = t.key(&T, |_| Ok::<_, ()>(j(&[EN; 2], 0.8, 0.2, 0.8))).unwrap();
        assert!(matches!(a, KeyAction::Release(s) if s == "he"));
        // a held key is taken back by Backspace without reaching the app
        t.letter('k', Mode::En);
        assert!(!t.backspace());
        assert!(t.backspace()); // 'e' is on screen
    }

    #[test]
    fn boundary_shows_held_keys_or_replaces_them() {
        let mut t = Typing::new();
        typ(&mut t, "s", Mode::En);
        t.letter('o', Mode::En); // held: leaning Korean, unsure
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.99, 0.99, 0.01))).unwrap().unwrap();
        assert_eq!((o.plan.is_none(), o.release.as_str()), (true, "o"));
        typ(&mut t, "dk", Mode::En);
        t.letter('s', Mode::En);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 3], 0.9999, 0.9999, 0.0))).unwrap().unwrap();
        assert_eq!((o.plan.unwrap().backspaces, o.release.as_str()), (2, "")); // only dk was on screen
    }

    #[test]
    fn clear_counts() {
        assert_eq!(clear_count("dkssud", Mode::En, 0), 6);
        // 안 committed (1) + 녕 composing (3 keys)
        assert_eq!(clear_count("dkssud", Mode::Ko, 0), 4);
        assert_eq!(clear_count("dkssud", Mode::Ko, 6), 2);
        assert_eq!(clear_count("ekfr", Mode::Ko, 0), 4); // 닭 composing
    }

    #[test]
    fn corrects_english_mode_korean() {
        let mut t = Typing::new();
        typ(&mut t, "dkssud", Mode::En);
        let o = t.boundary(&T, true, |w| {
            assert_eq!(w, ["dkssud"]);
            Ok::<_, ()>(j(&[KO; 6], 0.9999, 0.9999, 0.0))
        });
        let p = o.unwrap().unwrap().plan.unwrap();
        assert_eq!(p.backspaces, 6);
        assert_eq!(p.parts, vec![(Mode::Ko, "dkssud".to_string())]);
        assert_eq!(p.mode, Mode::Ko);
    }

    #[test]
    fn korean_mode_english_counts_composition() {
        let mut t = Typing::new();
        typ(&mut t, "hello", Mode::Ko); // ㅗ디ㅣㅐ: ㅗ, 디 committed, ㅣ, ㅐ composing (1 key)
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 5], 0.9999, 0.0, 0.9999))).unwrap().unwrap();
        assert_eq!(o.plan.unwrap().backspaces, 4);
    }

    #[test]
    fn keeps_below_threshold_and_same_mode() {
        let mut t = Typing::new();
        typ(&mut t, "so", Mode::En);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.99, 0.99, 0.01))).unwrap().unwrap();
        assert_eq!((o.reason, o.plan.is_none()), (Reason::BelowThreshold, true));
        typ(&mut t, "rkwk", Mode::Ko);
        let o = t.boundary(&T, true, |w| {
            assert_eq!(w, ["so", "rkwk"]);
            Ok::<_, ()>(j(&[KO; 4], 1.0, 1.0, 0.0))
        });
        assert_eq!(o.unwrap().unwrap().reason, Reason::SameAsTyped);
    }

    #[test]
    fn revises_the_previous_word() {
        let keep_ko = |_: &[&str]| Ok::<_, ()>(j(&[KO], 0.96, 0.04, 0.96));
        // ㅑ think: `i` typed in Korean mode and left (unsure), `think` then in English
        let mut t = Typing::new();
        typ(&mut t, "i", Mode::Ko);
        t.boundary(&T, true, keep_ko).unwrap();
        typ(&mut t, "think", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 5], 1.0, 0.0, 1.0))).unwrap();
        let (r, _) = t
            .revise(&T, |w| {
                assert_eq!(w, ["i", "think"]);
                Ok::<_, ()>(j(&[EN], 0.9992, 0.0008, 0.9992))
            })
            .unwrap()
            .unwrap();
        // past ` think ` (7), delete ㅑ (1), type i, back 7
        assert_eq!((r.left, r.backspaces, r.parts.clone(), r.right), (7, 1, vec![(Mode::En, "i".into())], 7));
        // done once: the word is now fixed
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("fixed") }).unwrap().is_none());

        // not after Enter or punctuation, and not a word the app already changed
        let mut t = Typing::new();
        typ(&mut t, "i", Mode::Ko);
        t.boundary(&T, false, keep_ko).unwrap();
        typ(&mut t, "think", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 5], 1.0, 0.0, 1.0))).unwrap();
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("not between spaces") }).unwrap().is_none());
        let mut t = Typing::new();
        typ(&mut t, "dkssud", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 6], 1.0, 1.0, 0.0))).unwrap(); // corrected to 안녕
        typ(&mut t, "gkk", Mode::Ko);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 3], 1.0, 1.0, 0.0))).unwrap();
        // a word the app changed is looked at once more: still Korean, so it stays
        assert!(t.revise(&T, |_| Ok::<_, ()>(j(&[KO; 6], 1.0, 1.0, 0.0))).unwrap().is_none());
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("looked at once") }).unwrap().is_none());

        // below the threshold: left alone
        let mut t = Typing::new();
        typ(&mut t, "so", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.93, 0.93, 0.07))).unwrap();
        typ(&mut t, "rkwk", Mode::Ko);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 4], 1.0, 1.0, 0.0))).unwrap();
        assert!(t.revise(&T, |_| Ok::<_, ()>(j(&[KO; 2], 0.98, 0.98, 0.02))).unwrap().is_none());
    }

    #[test]
    fn reverts_a_change_once() {
        // `so` typed in English mode, changed to 내 at Space; with `what` after it the model is sure of so
        let mut t = Typing::new();
        typ(&mut t, "so", Mode::En);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.9995, 0.9995, 0.0005))).unwrap().unwrap();
        assert!(o.plan.is_some());
        typ(&mut t, "dhkt", Mode::Ko);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 4], 1.0, 0.0, 1.0))).unwrap();
        let (r, _) = t
            .revise(&T, |w| {
                assert_eq!(w, ["so", "dhkt"]);
                Ok::<_, ()>(j(&[EN; 2], 0.9995, 0.0005, 0.9995))
            })
            .unwrap()
            .unwrap();
        // delete 내 (one character), type so in English
        assert_eq!((r.backspaces, r.parts.clone(), r.from), (1, vec![(Mode::En, "so".into())], Mode::Ko));
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("settled") }).unwrap().is_none());

        // changed in the middle of the word (Early): also goes back once
        let mut t = Typing::new();
        t.letter('s', Mode::En);
        t.letter('o', Mode::En);
        let a = t.key(&T, |_| Ok::<_, ()>(j(&[KO; 2], 0.9995, 0.9995, 0.0005))).unwrap();
        assert!(matches!(a, KeyAction::Switch(_)));
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.9995, 0.9995, 0.0005))).unwrap();
        typ(&mut t, "dhkt", Mode::Ko);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 4], 1.0, 0.0, 1.0))).unwrap();
        let (r, _) = t.revise(&T, |_| Ok::<_, ()>(j(&[EN; 2], 0.9995, 0.0005, 0.9995))).unwrap().unwrap();
        assert_eq!(r.parts, vec![(Mode::En, "so".into())]);

        // the user undid the change: nothing to go back to
        let mut t = Typing::new();
        typ(&mut t, "so", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.9995, 0.9995, 0.0005))).unwrap();
        t.undo().unwrap();
        typ(&mut t, "what", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN; 4], 1.0, 0.0, 1.0))).unwrap();
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("undone") }).unwrap().is_none());
    }

    #[test]
    fn habits_shift_short_words() {
        let t85 = Thresholds { act: 0.85, early: 0.85, ..T };
        let lean_ko = |_: &[&str]| Ok::<_, ()>(j(&[KO; 2], 0.92, 0.92, 0.08));
        let mut t = Typing::new();
        // `so` kept in English three times (counted once it can no longer change: two words on, or a reset)
        for _ in 0..3 {
            typ(&mut t, "so", Mode::En);
            t.boundary(&t85, true, |_| Ok::<_, ()>(j(&[EN; 2], 0.9, 0.1, 0.9))).unwrap();
            t.reset();
        }
        assert_eq!(t.exceptions.habit("so"), (0.0, 3.0));
        // now 0.92 Korean becomes about 0.82: below 0.85, left as typed
        typ(&mut t, "so", Mode::En);
        let o = t.boundary(&t85, true, lean_ko).unwrap().unwrap();
        assert_eq!(o.reason, Reason::BelowThreshold);
        let p = o.judgement.unwrap().p_ko;
        assert!((0.80..0.85).contains(&p), "{p}");
        // and it holds back a switch in the middle of the word
        t.reset();
        t.letter('s', Mode::En);
        t.letter('o', Mode::En);
        assert!(matches!(t.key(&t85, lean_ko).unwrap(), KeyAction::Release(_)));
        // a word without a habit, or a long one, is as the model says
        t.reset();
        typ(&mut t, "gg", Mode::En);
        assert!(t.boundary(&t85, true, lean_ko).unwrap().unwrap().plan.is_some());
        assert_eq!(t.exceptions.habit("gg"), (0.0, 0.0)); // not yet counted: it may still change
        t.reset();
        assert_eq!(t.exceptions.habit("gg"), (1.0, 0.0));
        typ(&mut t, "dkssud", Mode::En);
        t.boundary(&t85, true, |_| Ok::<_, ()>(j(&[KO; 6], 1.0, 1.0, 0.0))).unwrap();
        t.reset();
        assert_eq!(t.exceptions.habit("dkssud"), (0.0, 0.0));
        // off: nothing counted or shifted
        let mut t = Typing::new();
        t.habits = false;
        typ(&mut t, "so", Mode::En);
        t.boundary(&t85, true, lean_ko).unwrap();
        t.reset();
        assert_eq!(t.exceptions.habit("so"), (0.0, 0.0));
    }

    #[test]
    fn splits() {
        let mut t = Typing::new();
        typ(&mut t, "MBCsms", Mode::Ko);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[EN, EN, EN, KO, KO, KO], 0.95, 0.0, 0.0))).unwrap().unwrap();
        let p = o.plan.unwrap();
        assert_eq!(p.parts, vec![(Mode::En, "MBC".into()), (Mode::Ko, "sms".into())]);
        assert_eq!(p.mode, Mode::Ko);
    }

    #[test]
    fn undo_and_exception() {
        let mut t = Typing::new();
        typ(&mut t, "so", Mode::En);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 2], 0.9999, 0.9999, 0.0))).unwrap();
        let p = t.undo().unwrap();
        assert_eq!(p.backspaces, 2); // 내 + the boundary
        assert_eq!(p.parts, vec![(Mode::En, "so".to_string())]);
        assert!(t.undo().is_none());
        typ(&mut t, "so", Mode::En);
        let o = t.boundary(&T, true, |_| -> Result<Judgement, ()> { panic!("not judged") }).unwrap().unwrap();
        assert_eq!(o.reason, Reason::Exception);
    }

    #[test]
    fn learns_a_correction_typed_back() {
        let ko = |_: &[&str]| Ok::<_, ()>(j(&[KO; 6], 0.9999, 0.9999, 0.0));
        let mut t = Typing::new();
        typ(&mut t, "gksdud", Mode::En);
        assert!(t.boundary(&T, true, ko).unwrap().unwrap().plan.is_some()); // 한영
        // the user deletes it (Ctrl+Backspace: a reset), presses 한/영 and types it again in English
        t.reset();
        for c in "gksdud".chars() {
            t.letter(c, Mode::En);
            let a = t.key(&T, |_| -> Result<Judgement, ()> { panic!("not switched while typed back") });
            assert!(matches!(a.unwrap(), KeyAction::Release(_)));
        }
        let o = t.boundary(&T, true, |_| -> Result<Judgement, ()> { panic!("an exception now") }).unwrap().unwrap();
        assert_eq!((o.reason, o.plan.is_none()), (Reason::Exception, true));
        assert!(t.exceptions.contains("gksdud"));

        // typed the other way right after it was left alone: no longer an exception
        typ(&mut t, "gksdud", Mode::Ko);
        let o = t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 6], 0.9999, 0.9999, 0.0))).unwrap().unwrap();
        assert_eq!(o.reason, Reason::SameAsTyped);
        assert!(!t.exceptions.contains("gksdud"));
        // the one left alone before it stays as it is
        assert!(t.revise(&T, |_| -> Result<Judgement, ()> { panic!("settled") }).unwrap().is_none());
    }

    #[test]
    fn does_not_learn_from_other_words() {
        let ko = |_: &[&str]| Ok::<_, ()>(j(&[KO; 6], 0.9999, 0.9999, 0.0));
        // the same keys typed again in the corrected mode: just the same word again
        let mut t = Typing::new();
        typ(&mut t, "gksdud", Mode::En);
        t.boundary(&T, true, ko).unwrap();
        typ(&mut t, "gksdud", Mode::Ko);
        t.boundary(&T, true, ko).unwrap();
        assert!(!t.exceptions.contains("gksdud"));
        // another word first: the watch is gone
        let mut t = Typing::new();
        typ(&mut t, "gksdud", Mode::En);
        t.boundary(&T, true, ko).unwrap();
        typ(&mut t, "rkwk", Mode::Ko);
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 4], 1.0, 1.0, 0.0))).unwrap();
        typ(&mut t, "gksdud", Mode::En);
        assert!(t.boundary(&T, true, ko).unwrap().unwrap().plan.is_some());
        // a click elsewhere: gone too
        let mut t = Typing::new();
        typ(&mut t, "gksdud", Mode::En);
        t.boundary(&T, true, ko).unwrap();
        t.moved();
        typ(&mut t, "gksdud", Mode::En);
        assert!(t.boundary(&T, true, ko).unwrap().unwrap().plan.is_some());
    }

    #[test]
    fn backspace_follows_the_ime() {
        let mut t = Typing::new();
        typ(&mut t, "dkssud", Mode::Ko); // 안녕, 녕 composing
        t.backspace(); // 안녀
        t.backspace(); // 안ㄴ
        t.backspace(); // 안
        assert_eq!(t.cur.as_ref().unwrap().keys, "dks");
        t.backspace(); // 안 was committed: gone whole
        assert!(t.cur.is_none());

        typ(&mut t, "rkr", Mode::Ko); // 각
        t.boundary(&T, true, |_| Ok::<_, ()>(j(&[KO; 3], 1.0, 1.0, 0.0))).unwrap();
        t.backspace(); // the boundary: 각 is the current word again, committed
        t.backspace(); // 각 goes whole
        assert!(t.cur.is_none());
    }

    #[test]
    fn typing_after_deleting_a_committed_syllable() {
        let mut t = Typing::new();
        typ(&mut t, "rkrk", Mode::Ko); // 가가
        t.backspace(); // 가ㄱ
        t.backspace(); // 가
        t.backspace(); // gone
        typ(&mut t, "rk", Mode::Ko);
        assert_eq!(t.cur.as_ref().unwrap().keys, "rk");
        let mut t = Typing::new();
        typ(&mut t, "rkrk", Mode::Ko);
        t.backspace();
        t.backspace(); // 가 committed (frozen), nothing composing
        typ(&mut t, "k", Mode::Ko); // a new ㅏ, not 가 + ㅏ
        let w = t.cur.as_ref().unwrap();
        assert_eq!((w.keys.as_str(), w.frozen), ("rkk", 2));
        assert_eq!(clear_count(&w.keys, Mode::Ko, w.frozen), 2);
    }

    #[test]
    fn digits_are_part_of_the_word() {
        // 3시 typed in English mode: one word, corrected whole
        let mut t = Typing::new();
        t.letter('3', Mode::En);
        assert!(matches!(t.key(&T, |_| -> Result<Judgement, ()> { panic!("digits alone") }).unwrap(), KeyAction::Release(s) if s == "3"));
        typ(&mut t, "tl", Mode::En);
        let o = t.boundary(&T, true, |w| {
            assert_eq!(w, ["3tl"]);
            Ok::<_, ()>(j(&[KO; 3], 0.9999, 0.9999, 0.0))
        });
        let p = o.unwrap().unwrap().plan.unwrap();
        // the 3 stays on screen: only tl is replaced
        assert_eq!((p.backspaces, p.shown.as_str(), p.parts), (2, "tl", vec![(Mode::Ko, "tl".to_string())]));
        assert_eq!(t.history[0].on_screen, 2); // 3시
        let u = t.undo().unwrap();
        assert_eq!((u.backspaces, u.parts), (2, vec![(Mode::En, "tl".to_string())])); // 시 + the boundary
        // in Korean mode the digit commits the syllable before it: 가1ㅅ clears with 3 Backspaces
        assert_eq!(clear_count("rk1t", Mode::Ko, 0), 3);
        assert_eq!(on_screen("rk1t", Mode::Ko, 0), "가1ㅅ");
        // digits only: kept, not judged, never revised
        let mut t = Typing::new();
        typ(&mut t, "2019", Mode::Ko);
        let o = t.boundary(&T, true, |_| -> Result<Judgement, ()> { panic!("digits alone") }).unwrap().unwrap();
        assert!(o.plan.is_none() && o.judgement.is_none());
        assert!(t.history[0].fixed);
    }

    #[test]
    fn screen_words_are_context() {
        let mut t = Typing::new();
        assert!(t.wants_context());
        t.set_context(vec!["dkssud".into()], vec!["think".into(), "so".into()]);
        assert!(!t.wants_context());
        typ(&mut t, "i", Mode::Ko);
        // while typing: the words before only (the model never saw a word being typed with words after it)
        t.key(&T, |w| {
            assert_eq!(w, ["dkssud", "i"]);
            Ok::<_, ()>(j(&[EN], 0.5, 0.5, 0.5))
        })
        .unwrap();
        assert_eq!(t.after_len(), 2);
        // at the boundary: the words after it too
        t.boundary(&T, true, |w| {
            assert_eq!(w, ["dkssud", "i", "think", "so"]);
            Ok::<_, ()>(j(&[EN], 0.9999, 0.0, 0.9999))
        })
        .unwrap();
        // the screen word before is never revised, and Backspace into it forgets the line
        let ko = |_: &[&str]| Ok::<_, ()>(j(&[KO; 6], 1.0, 1.0, 0.0));
        assert!(t.revise(&T, ko).unwrap().is_none());
        t.backspace(); // the Space: i is the current word again
        t.backspace(); // i
        assert!(!t.wants_context());
        t.backspace(); // the Space before i: into the screen word
        assert!(t.wants_context() && t.after_len() == 0);
    }
}
