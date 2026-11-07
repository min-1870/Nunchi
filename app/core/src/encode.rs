//! Words typed so far -> the model's input stream (`stream_tagger.encode` for one stream, after `cut`).

use crate::meta::Meta;

#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    pub ids: Vec<i64>,
    pub pos: Vec<i64>,
    pub wpos: Vec<i64>,
    pub widx: Vec<i64>,
    /// where the judged (last) word's keys are in the stream
    pub judged: std::ops::Range<usize>,
}

/// `words`: everything typed so far; the judged one is `meta.judged_from_end` from the end. Each word is
/// letter keys only.
pub fn encode(words: &[&str], meta: &Meta) -> Stream {
    let back = meta.judged_from_end;
    assert!(words.len() > back);
    let first = match meta.left_words {
        Some(left) => words.len().saturating_sub(1 + back + left),
        None => 0,
    };
    let mut sent: Vec<Vec<char>> =
        words[first..].iter().map(|w| w.chars().take(meta.max_keys_per_word).collect()).collect();
    // keep the latest whole words that fit
    while sent.iter().map(|w| w.len() + 1).sum::<usize>() > meta.max_stream_keys && sent.len() > 1 + back {
        sent.remove(0);
    }
    let judged = sent.len() - 1 - back;
    let t: usize = sent.iter().map(|w| w.len() + 1).sum();
    let mut s = Stream {
        ids: vec![meta.vocab.pad; t],
        pos: (0..t as i64).collect(),
        wpos: vec![0; t],
        widx: vec![meta.padding_widx; t],
        judged: 0..0,
    };
    let mut at = 0;
    for (w, word) in sent.iter().enumerate() {
        for (j, &c) in word.iter().enumerate() {
            s.ids[at + j] = meta.key_id(c);
            s.wpos[at + j] = (j as i64 + 1).min(meta.max_keys_per_word as i64);
        }
        let n = word.len();
        for x in &mut s.widx[at..at + n] {
            *x = w as i64;
        }
        if w + 1 < sent.len() {
            s.widx[at + n] = w as i64; // the space after a word belongs to it
            s.ids[at + n] = meta.vocab.space;
        }
        if w == judged {
            s.judged = at..at + n;
        }
        at += n + 1;
    }
    s
}
