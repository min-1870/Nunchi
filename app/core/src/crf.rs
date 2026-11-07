//! Viterbi and path probabilities over one word's emissions (`Tagger.decode_em`). Labels: 0 en, 1 ko.

use crate::meta::Crf;

pub const EN: u8 = 0;
pub const KO: u8 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    /// best label per key
    pub path: Vec<u8>,
    /// P(best path), P(every key ko), P(every key en)
    pub p_best: f64,
    pub p_ko: f64,
    pub p_en: f64,
}

fn lse2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn path_score(em: &[[f64; 2]], crf: &Crf, path: &[u8]) -> f64 {
    let mut s = crf.start[path[0] as usize] + em[0][path[0] as usize];
    for t in 1..em.len() {
        s += crf.trans[path[t - 1] as usize][path[t] as usize] + em[t][path[t] as usize];
    }
    s + crf.end[path[em.len() - 1] as usize]
}

pub fn decode(em: &[[f64; 2]], crf: &Crf) -> Judgement {
    assert!(!em.is_empty());
    let mut score = [crf.start[0] + em[0][0], crf.start[1] + em[0][1]];
    let mut alpha = score;
    let mut back: Vec<[u8; 2]> = Vec::with_capacity(em.len());
    for e in &em[1..] {
        let mut next = [0.0; 2];
        let mut b = [0u8; 2];
        let mut a = [0.0; 2];
        for j in 0..2 {
            let c0 = score[0] + crf.trans[0][j];
            let c1 = score[1] + crf.trans[1][j];
            // ties go to label 0, as torch's max does
            (next[j], b[j]) = if c1 > c0 { (c1, 1) } else { (c0, 0) };
            next[j] += e[j];
            a[j] = lse2(alpha[0] + crf.trans[0][j], alpha[1] + crf.trans[1][j]) + e[j];
        }
        score = next;
        alpha = a;
        back.push(b);
    }
    let fin = [score[0] + crf.end[0], score[1] + crf.end[1]];
    let mut last = if fin[1] > fin[0] { KO } else { EN };
    let best = fin[last as usize];
    let mut path = vec![last];
    for b in back.iter().rev() {
        last = b[last as usize];
        path.push(last);
    }
    path.reverse();
    let logz = lse2(alpha[0] + crf.end[0], alpha[1] + crf.end[1]);
    let n = em.len();
    Judgement {
        p_best: (best - logz).exp(),
        p_ko: (path_score(em, crf, &vec![KO; n]) - logz).exp(),
        p_en: (path_score(em, crf, &vec![EN; n]) - logz).exp(),
        path,
    }
}
