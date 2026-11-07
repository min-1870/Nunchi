//! Words the user does not want corrected, learned from what they do: a correction they type back over is
//! added, and an exception they type the other way is removed again (see `Typing`).
//!
//! The app keeps them on disk; what it stores is its business (the Windows app stores keyed hashes, never
//! the keys: an exception can be a piece of a password typed into a plain field). `Memory` keeps them as
//! they are, for tests and runs that save nothing.
//!
//! The same store keeps the user's habits for short words: how often keys ended up in each language (see
//! `Typing`'s habit prior).

use crate::decide::Mode;
use std::collections::HashMap;

pub trait Exceptions: Send {
    /// `keys` is an exception
    fn contains(&self, keys: &str) -> bool;
    /// `keys` is the start of an exception (or one): not to be switched in the middle of the word
    fn starts(&self, keys: &str) -> bool;
    /// add `keys`; `p`: how sure the model was when the user undid it (kept for a softer rule later)
    fn add(&mut self, keys: &str, p: f64);
    fn remove(&mut self, keys: &str);
    /// how often `keys` ended up as Korean and as English
    fn habit(&self, _keys: &str) -> (f64, f64) {
        (0.0, 0.0)
    }
    /// `keys` ended up in `mode`
    fn count(&mut self, _keys: &str, _mode: Mode) {}
}

/// Counts stop growing past this many uses: older uses are halved away, so a habit can change.
pub const HABIT_MAX: f64 = 40.0;

/// Add one use of `mode` to (ko, en), halving both once they pass HABIT_MAX.
pub fn bump(c: &mut (f64, f64), mode: Mode) {
    match mode {
        Mode::Ko => c.0 += 1.0,
        Mode::En => c.1 += 1.0,
    }
    if c.0 + c.1 > HABIT_MAX {
        c.0 /= 2.0;
        c.1 /= 2.0;
    }
}

#[derive(Default)]
pub struct Memory {
    words: HashMap<String, f64>,
    habits: HashMap<String, (f64, f64)>,
}

impl Exceptions for Memory {
    fn contains(&self, keys: &str) -> bool {
        self.words.contains_key(keys)
    }

    fn starts(&self, keys: &str) -> bool {
        self.words.keys().any(|w| w.starts_with(keys))
    }

    fn add(&mut self, keys: &str, p: f64) {
        self.words.insert(keys.to_string(), p);
    }

    fn remove(&mut self, keys: &str) {
        self.words.remove(keys);
    }

    fn habit(&self, keys: &str) -> (f64, f64) {
        self.habits.get(keys).copied().unwrap_or_default()
    }

    fn count(&mut self, keys: &str, mode: Mode) {
        bump(self.habits.entry(keys.to_string()).or_default(), mode);
    }
}
