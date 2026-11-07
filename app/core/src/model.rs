//! The ONNX graph plus Viterbi: words typed so far -> judgement of one of them.

use crate::crf::{self, Judgement};
use crate::encode::{encode, Stream};
use crate::meta::Meta;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::OnceLock;

/// ONNX Runtime allows one environment per process; every model shares it.
static ENV: OnceLock<Environment> = OnceLock::new();

pub struct Model {
    pub meta: Meta,
    session: Session,
}

type Error = Box<dyn std::error::Error>;

impl Model {
    /// `onnx`: models/<name>.onnx; the JSON next to it is read too.
    pub fn load(onnx: &Path) -> Result<Self, Error> {
        Self::from_memory(&std::fs::read(onnx)?, Meta::load(&onnx.with_extension("json"))?)
    }

    /// The graph's bytes and its JSON, already read (a model built into the exe).
    pub fn from_memory(onnx: &[u8], meta: Meta) -> Result<Self, Error> {
        // one thread: more were 30x slower for this small model on Windows
        if ENV.get().is_none() {
            let _ = ENV.set(ort::init().with_name("nunchi").build()?);
        }
        let env = ENV.get().expect("environment");
        let session = Session::builder(env)?.with_intra_threads(1)?.commit_from_memory(onnx)?;
        Ok(Self { meta, session })
    }

    /// The same model made a reviser (see `load_reviser`).
    pub fn reviser_from_memory(onnx: &[u8], meta: Meta) -> Result<Self, Error> {
        Ok(Self::make_reviser(Self::from_memory(onnx, meta)?))
    }

    fn make_reviser(mut m: Self) -> Self {
        m.meta.judged_from_end = 1;
        m.meta.act_threshold = m.meta.revise_act_threshold.unwrap_or(m.meta.act_threshold);
        m
    }

    /// A model that also revises (`meta.revises`), loaded a second time to judge the word before the last:
    /// the worker then has a decider and a reviser as with two models.
    pub fn load_reviser(onnx: &Path) -> Result<Self, Error> {
        Ok(Self::make_reviser(Self::load(onnx)?))
    }

    /// Judge the word `back` from the end of `words` (the words after it are context the model reads too).
    pub fn judge_at(&mut self, words: &[&str], back: usize) -> Result<Judgement, Error> {
        let keep = std::mem::replace(&mut self.meta.judged_from_end, back);
        let j = self.judge(words);
        self.meta.judged_from_end = keep;
        j
    }

    /// Per-key en/ko scores for a stream.
    pub fn emissions(&mut self, s: &Stream) -> Result<Vec<[f64; 2]>, Error> {
        let t = s.ids.len();
        let input = |v: &Vec<i64>| Tensor::from_array(([1usize, t], v.clone()));
        let out = self.session.run(ort::inputs![
            "ids" => input(&s.ids)?,
            "pos" => input(&s.pos)?,
            "wpos" => input(&s.wpos)?,
            "widx" => input(&s.widx)?,
        ])?;
        let (_, data) = out["emissions"].try_extract_tensor::<f32>()?;
        Ok(data.chunks(2).map(|c| [c[0] as f64, c[1] as f64]).collect())
    }

    /// Judge one of `words`: the last for a decider, the one before it for a reviser (`judged_from_end`).
    /// Letter keys only; at most max_keys_per_word of a word are used.
    pub fn judge(&mut self, words: &[&str]) -> Result<Judgement, Error> {
        let s = encode(words, &self.meta);
        let em = self.emissions(&s)?;
        let temp = self.meta.temperature;
        let word: Vec<[f64; 2]> = em[s.judged.clone()].iter().map(|e| [e[0] / temp, e[1] / temp]).collect();
        Ok(crf::decode(&word, &self.meta.crf))
    }
}
