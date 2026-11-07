//! 두벌식 composition of raw keys, the way the Korean IME shows them.
//!
//! Greedy, left to right (port of `nunchi/demo.py` `compose`), with the key span of every character so the
//! app knows how many Backspaces clear a word.

const CHO: &str = "rRseEfaqQtTdwWczxvg";
const JUNG: [&str; 21] = [
    "k", "o", "i", "O", "j", "p", "u", "P", "h", "hk", "ho", "hl", "y", "n", "nj", "np", "nl", "b", "m",
    "ml", "l",
];
const JONG: [&str; 28] = [
    "", "r", "R", "rt", "s", "sw", "sg", "e", "f", "fr", "fa", "fq", "ft", "fx", "fv", "fg", "a", "q",
    "qt", "t", "T", "d", "w", "c", "z", "x", "v", "g",
];
/// Compatibility jamo ㄱ..ㅣ and their keys, for jamo that stand alone (ㅋㅋ).
const COMPAT: &str = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
const COMPAT_CONS: [&str; 30] = [
    "r", "R", "rt", "s", "sw", "sg", "e", "E", "f", "fr", "fa", "fq", "ft", "fx", "fv", "fg", "a", "q",
    "Q", "qt", "t", "T", "d", "w", "W", "c", "z", "x", "v", "g",
];

/// One character on screen and the keys `start..end` that made it.
#[derive(Debug, Clone, PartialEq)]
pub struct Syl {
    pub ch: char,
    pub start: usize,
    pub end: usize,
}

/// Shift only matters for Q W E R T O P in Korean mode; Shift+S is still ㄴ.
pub fn normalize(keys: &str) -> Vec<char> {
    keys.chars().map(|c| if "QWERTOP".contains(c) { c } else { c.to_ascii_lowercase() }).collect()
}

fn pos<const N: usize>(table: &[&str; N], s: &str) -> Option<usize> {
    table.iter().position(|x| *x == s)
}

fn lone(keys: &str) -> char {
    let i = pos(&COMPAT_CONS, keys)
        .or_else(|| pos(&JUNG, keys).map(|j| COMPAT_CONS.len() + j))
        .expect("jamo keys");
    COMPAT.chars().nth(i).unwrap()
}

/// Characters a key sequence shows in Korean mode. Keys are letters, or digits, which show as they are and
/// end the syllable before them.
pub fn compose(keys: &str) -> Vec<Syl> {
    let k = normalize(keys);
    let mut out = Vec::new();
    // the syllable being built: (keys, start index) for each part
    let (mut cho, mut jung, mut jong) = (String::new(), String::new(), String::new());
    let mut start = 0;

    let flush = |out: &mut Vec<Syl>, cho: &mut String, jung: &mut String, jong: &mut String, start: usize, end: usize| {
        if !cho.is_empty() && !jung.is_empty() {
            let c = CHO.find(cho.as_str()).unwrap();
            let v = pos(&JUNG, jung).unwrap();
            let f = pos(&JONG, jong).unwrap();
            let code = 0xAC00 + ((c * 21 + v) * 28 + f) as u32;
            out.push(Syl { ch: char::from_u32(code).unwrap(), start, end });
        } else {
            // a lone consonant and/or a lone vowel: each is its own character
            let mut s = start;
            for part in [&*cho, &*jung] {
                if !part.is_empty() {
                    out.push(Syl { ch: lone(part), start: s, end: s + part.len() });
                    s += part.len();
                }
            }
        }
        cho.clear();
        jung.clear();
        jong.clear();
    };

    for (i, &c) in k.iter().enumerate() {
        if !c.is_ascii_alphabetic() {
            flush(&mut out, &mut cho, &mut jung, &mut jong, start, i);
            out.push(Syl { ch: c, start: i, end: i + 1 });
            start = i + 1;
            continue;
        }
        let cs = c.to_string();
        let is_cons = CHO.contains(c);
        if is_cons {
            if !cho.is_empty() && !jung.is_empty() && jong.is_empty() && pos(&JONG, &cs).is_some() {
                jong = cs;
            } else if !cho.is_empty() && !jung.is_empty() && !jong.is_empty() && pos(&JONG, &(jong.clone() + &cs)).is_some() {
                jong.push(c);
            } else {
                flush(&mut out, &mut cho, &mut jung, &mut jong, start, i);
                start = i;
                cho = cs;
            }
        } else {
            // a vowel key
            if !cho.is_empty() && !jung.is_empty() && !jong.is_empty() {
                // the last final consonant moves to the next syllable: 각 + ㅏ -> 가가
                let mv = jong.pop().unwrap();
                flush(&mut out, &mut cho, &mut jung, &mut jong, start, i - 1);
                start = i - 1;
                cho = mv.to_string();
                jung = cs;
            } else if !jung.is_empty() && pos(&JUNG, &(jung.clone() + &cs)).is_some() {
                jung.push(c);
            } else if !cho.is_empty() && jung.is_empty() {
                jung = cs;
            } else {
                flush(&mut out, &mut cho, &mut jung, &mut jong, start, i);
                start = i;
                jung = cs;
            }
        }
    }
    flush(&mut out, &mut cho, &mut jung, &mut jong, start, k.len());
    out
}

pub fn compose_str(keys: &str) -> String {
    compose(keys).iter().map(|s| s.ch).collect()
}

/// The keys that type `c` in its own mode: a Hangul syllable or jamo as 두벌식 keys, an ASCII letter (or a
/// digit, with `digits`) as itself. None: not a key the model reads (it ends a word).
fn keys_of_char(c: char, digits: bool) -> Option<String> {
    match c {
        'a'..='z' | 'A'..='Z' => Some(c.to_string()),
        '0'..='9' if digits => Some(c.to_string()),
        '\u{AC00}'..='\u{D7A3}' => {
            let i = c as usize - 0xAC00;
            let (cho, jung, jong) = (i / (21 * 28), i / 28 % 21, i % 28);
            Some(format!("{}{}{}", CHO.chars().nth(cho).unwrap(), JUNG[jung], JONG[jong]))
        }
        _ => {
            let i = COMPAT.chars().position(|x| x == c)?;
            Some(if i < COMPAT_CONS.len() { COMPAT_CONS[i] } else { JUNG[i - COMPAT_CONS.len()] }.to_string())
        }
    }
}

/// Text on screen as the words the model reads, each as the keys that type it: words end at anything that
/// is not a key (spaces, punctuation, emoji, other scripts), as boundary keys end them while typing.
pub fn words_of(text: &str, digits: bool) -> Vec<String> {
    let mut words = vec![String::new()];
    for c in text.chars() {
        match keys_of_char(c, digits) {
            Some(k) => words.last_mut().unwrap().push_str(&k),
            None if words.last().unwrap().is_empty() => {}
            None => words.push(String::new()),
        }
    }
    words.retain(|w| !w.is_empty());
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words() {
        for (keys, want) in [
            ("dkssud", "안녕"),
            ("dkssudgktpdy", "안녕하세요"),
            ("rkqt", "값"),
            ("dz", "ㅇㅋ"),
            ("rhoscksgdk", "괜찮아"),
            ("dlfrdjTek", "읽었다"),
            ("Rkcl", "까치"),
            ("zzzz", "ㅋㅋㅋㅋ"),
            ("hk", "ㅘ"),
            ("rkr", "각"),
            ("rkrk", "가가"),
            ("Dkssud", "안녕"),
        ] {
            assert_eq!(compose_str(keys), want, "{keys}");
        }
    }

    #[test]
    fn digits_end_the_syllable() {
        assert_eq!(compose_str("3tl"), "3시");
        assert_eq!(compose_str("rk1k"), "가1ㅏ"); // the vowel after a digit does not take 가's ㄱ
        assert_eq!(compose_str("dkf2"), "알2");
        let s = compose("rkr1");
        assert_eq!((s[1].ch, s[1].start, s[1].end), ('1', 3, 4));
    }

    #[test]
    fn screen_text_back_to_keys() {
        for w in ["dkssudgktpdy", "rhoscksgdk", "dlfrdjTek", "Rkcl", "zzzz", "rkqt", "ekfr"] {
            assert_eq!(words_of(&compose_str(w), false), vec![w.to_string()], "{w}");
        }
        assert_eq!(words_of("안녕, I think 3시에! MBC는 좋아👍ok", true), ["dkssud", "I", "think", "3tldp", "MBCsms", "whgdk", "ok"]);
        assert_eq!(words_of("3시에", false), ["tldp"]);
        assert!(words_of("  ", true).is_empty());
    }

    #[test]
    fn spans() {
        let s = compose("dkssud");
        assert_eq!((s[0].start, s[0].end, s[1].start, s[1].end), (0, 3, 3, 6));
        let s = compose("rkrk"); // the ㄱ moves: 가(rk) 가(rk)
        assert_eq!((s[0].end, s[1].start), (2, 2));
    }
}
