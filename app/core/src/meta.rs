//! `models/<name>.json`: everything about the model that lives outside the ONNX graph.

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
pub struct Vocab {
    pub pad: i64,
    pub other: i64,
    pub chars: String,
    pub first_char_id: i64,
    pub space: i64,
    /// models that read digits: the id of '0' (the others follow); None: digits are "other"
    #[serde(default)]
    pub digit0: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Crf {
    pub trans: [[f64; 2]; 2],
    pub start: [f64; 2],
    pub end: [f64; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub model: String,
    pub vocab: Vocab,
    pub max_keys_per_word: usize,
    pub max_stream_keys: usize,
    pub padding_widx: i64,
    /// words before the judged one the model is given; None: all that fit
    pub left_words: Option<usize>,
    /// 0: the model judges the last word given (a decider); 1: the word before it (a reviser)
    #[serde(default)]
    pub judged_from_end: usize,
    /// one model that also revises (TR): the same graph judging the word before the last, at
    /// `revise_act_threshold`
    #[serde(default)]
    pub revises: bool,
    #[serde(default)]
    pub revise_act_threshold: Option<f64>,
    #[serde(default)]
    pub lookahead: usize,
    #[serde(default = "one")]
    pub temperature: f64,
    pub act_threshold: f64,
    /// switching in the middle of a word (a prefix, not yet the word); absent: act_threshold
    #[serde(default)]
    pub early_act_threshold: Option<f64>,
    #[serde(default = "split_default")]
    pub split_threshold: f64,
    /// wrong corrections per 1,000 words -> threshold
    #[serde(default)]
    pub targets: BTreeMap<String, f64>,
    pub crf: Crf,
}

fn one() -> f64 {
    1.0
}

fn split_default() -> f64 {
    0.9
}

impl Meta {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    pub fn parse(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn key_id(&self, c: char) -> i64 {
        match (self.vocab.chars.find(c), self.vocab.digit0, c.to_digit(10)) {
            (Some(i), _, _) => self.vocab.first_char_id + i as i64,
            (None, Some(d0), Some(d)) => d0 + d as i64,
            _ => self.vocab.other,
        }
    }
}
