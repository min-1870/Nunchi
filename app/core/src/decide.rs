//! Judgement of a word + the IME mode it was typed in -> keep it or retype it.

use crate::crf::{Judgement, KO};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Mode {
    En,
    Ko,
}

impl Mode {
    pub fn of(label: u8) -> Mode {
        if label == KO {
            Mode::Ko
        } else {
            Mode::En
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Reason {
    /// the other language, at or above the threshold
    Confident,
    /// language changes inside the word, at or above the split threshold
    Split,
    /// the other language, already sure from the first keys of the word
    Early,
    /// the language it was typed in
    SameAsTyped,
    BelowThreshold,
    TooLong,
    /// the user undid a correction of these keys before
    Exception,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Keep,
    /// retype the word as these parts, each in its mode
    Retype(Vec<(Mode, String)>),
}

pub struct Thresholds {
    pub act: f64,
    /// switching in the middle of a word: the keys so far are a prefix, not yet the word (wha reads as 좀)
    pub early: f64,
    /// for a single key (a 1-key word start, or a 1-key word): one key says little, and at 0.99 1.7% of
    /// sure first-key judgements in a sentence's first word were wrong (none at 0.999)
    pub first_key: f64,
    pub split: f64,
    pub max_keys: usize,
}

impl Thresholds {
    /// Threshold for a word (or word start) of `n` keys.
    /// Threshold for switching in the middle of a word after `n` keys.
    pub fn early_for(&self, n: usize) -> f64 {
        if n <= 1 {
            self.first_key.max(self.early)
        } else {
            self.early
        }
    }

    pub fn act_for(&self, n: usize) -> f64 {
        if n <= 1 {
            self.first_key.max(self.act)
        } else {
            self.act
        }
    }
}

pub fn decide(keys: &str, typed: Mode, j: &Judgement, t: &Thresholds) -> (Action, Reason) {
    if keys.chars().count() > t.max_keys {
        return (Action::Keep, Reason::TooLong);
    }
    if j.path.iter().any(|&l| l != j.path[0]) && j.p_best >= t.split {
        let mut parts: Vec<(Mode, String)> = Vec::new();
        for (c, &l) in keys.chars().zip(&j.path) {
            match parts.last_mut() {
                Some((m, s)) if *m == Mode::of(l) => s.push(c),
                _ => parts.push((Mode::of(l), c.to_string())),
            }
        }
        return (Action::Retype(parts), Reason::Split);
    }
    let (target, p) = if j.p_ko > j.p_en { (Mode::Ko, j.p_ko) } else { (Mode::En, j.p_en) };
    if target == typed {
        (Action::Keep, Reason::SameAsTyped)
    } else if p >= t.act_for(keys.chars().filter(|c| c.is_ascii_alphabetic()).count()) {
        (Action::Retype(vec![(target, keys.to_string())]), Reason::Confident)
    } else {
        (Action::Keep, Reason::BelowThreshold)
    }
}
