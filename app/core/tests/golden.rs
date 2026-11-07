//! Same streams and decisions as Python (`nunchi.export_onnx` writes models/<name>_golden.json).

use nunchi_core::encode::encode;
use nunchi_core::model::Model;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    words: Vec<String>,
    /// the judged word from the end (a model that decides and revises has cases for both)
    back: Option<usize>,
    ids: Vec<i64>,
    pos: Vec<i64>,
    wpos: Vec<i64>,
    widx: Vec<i64>,
    path: Vec<u8>,
    p_best: f64,
    p_ko: f64,
    p_en: f64,
}

fn models() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

#[test]
fn golden() {
    // the decider and the reviser the app ships; NUNCHI_MODEL picks one
    let names = match std::env::var("NUNCHI_MODEL") {
        Ok(n) => vec![n],
        Err(_) => vec!["tr5n_102k".to_string()],
    };
    for name in names {
        golden_for(&name);
    }
}

fn golden_for(name: &str) {
    let cases: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(models().join(format!("{name}_golden.json"))).unwrap())
            .unwrap();
    let mut model = Model::load(&models().join(format!("{name}.onnx"))).unwrap();
    let default_back = model.meta.judged_from_end;
    let mut worst: f64 = 0.0;
    for c in &cases {
        model.meta.judged_from_end = c.back.unwrap_or(default_back);
        let words: Vec<&str> = c.words.iter().map(|s| s.as_str()).collect();
        let s = encode(&words, &model.meta);
        assert_eq!((&s.ids, &s.pos, &s.wpos, &s.widx), (&c.ids, &c.pos, &c.wpos, &c.widx), "{:?}", c.words);
        let j = model.judge(&words).unwrap();
        assert_eq!(j.path, c.path, "{:?}", c.words);
        for (a, b) in [(j.p_best, c.p_best), (j.p_ko, c.p_ko), (j.p_en, c.p_en)] {
            worst = worst.max((a - b).abs());
        }
    }
    println!("{name}: {} cases, max prob diff {worst:.2e}", cases.len());
    assert!(worst < 1e-4, "max prob diff {worst}");
}

#[test]
fn compose_matches_python() {
    let pairs: Vec<(String, String)> =
        serde_json::from_str(&std::fs::read_to_string(models().join("compose_golden.json")).unwrap()).unwrap();
    let bad: Vec<_> = pairs.iter().filter(|(k, want)| &nunchi_core::compose::compose_str(k) != want).collect();
    println!("{} words, {} differ", pairs.len(), bad.len());
    assert!(bad.is_empty(), "{:?}", &bad[..bad.len().min(10)]);
}
