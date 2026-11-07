//! Nunchi for Windows: fixes words typed in the wrong IME mode. Runs in the tray, no window; right-click
//! the icon to stop or start it.
//!
//!     nunchi.exe [models/name.onnx] [threshold]
//!
//! Without arguments it installs itself into %LOCALAPPDATA%\Programs\Nunchi and runs from there (see
//! install.rs). The model (models/tr5n_102k.onnx and its JSON) is built into the exe: the exe is the whole
//! app. A model given by path (tests) is read from there instead. Only one copy runs
//! at a time. With NUNCHI_DEBUG=1 every decision is logged to stdout and
//! %LOCALAPPDATA%\Nunchi\decisions.jsonl; otherwise nothing is written (the log holds what was typed).
//! Pause undoes the last correction.

#![windows_subsystem = "windows"]

mod autostart;
mod hook;
mod ime;
mod install;
mod store;
mod keys;
mod tray;
mod uia;
mod worker;

use nunchi_core::exceptions::{Exceptions, Memory};
use nunchi_core::meta::Meta;
use nunchi_core::model::Model;
use std::path::PathBuf;
use windows::core::w;
use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};
use windows::Win32::System::Threading::CreateMutexW;

/// The model built into the exe (the repository's models/, which must hold it at build time).
static MODEL_ONNX: &[u8] = include_bytes!("../../../models/tr5n_102k.onnx");
static MODEL_JSON: &str = include_str!("../../../models/tr5n_102k.json");

fn built_in_meta() -> Meta {
    Meta::parse(MODEL_JSON).expect("built-in model JSON")
}

fn main() {
    if !install::ensure() {
        return; // the installed copy took over
    }
    // one copy only: a second hook would fix every word twice
    let _guard = unsafe { CreateMutexW(None, true, w!("Local\\NunchiSingleInstance")) };
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        return;
    }

    let args: Vec<String> = std::env::args().collect();
    let onnx = args.get(1).map(PathBuf::from);
    let model = match &onnx {
        Some(p) => Model::load(p).unwrap_or_else(|e| panic!("loading {}: {e}", p.display())),
        None => Model::from_memory(MODEL_ONNX, built_in_meta()).expect("built-in model"),
    };
    let act = args.get(2).map(|a| a.parse().expect("threshold")).unwrap_or(model.meta.act_threshold);
    // a model that reads digits keeps them inside words (3시, gpt4); for older ones a digit ends the word
    keys::DIGITS.store(model.meta.vocab.digit0.is_some(), std::sync::atomic::Ordering::Relaxed);

    let log = std::env::var("NUNCHI_DEBUG").is_ok_and(|v| v == "1").then(|| {
        let dir = install::data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("decisions.jsonl");
        eprintln!("model {} (threshold {act}), log {}", model.meta.model, path.display());
        std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap()
    });

    // the model revises too (TR), or a reviser sits next to it (models/rv1_long.onnx); else nothing is revised
    let mut reviser = match &onnx {
        None => model.meta.revises.then(|| Model::reviser_from_memory(MODEL_ONNX, built_in_meta()).ok()).flatten(),
        Some(p) if model.meta.revises => Model::load_reviser(p).ok(),
        Some(p) => {
            let rv = p.with_file_name(install::REVISER_FILE);
            rv.exists().then(|| Model::load(&rv).ok()).flatten()
        }
    };

    // warm up: the first run allocates
    let mut model = model;
    let _ = model.judge(&["dkssud"]);
    if let Some(rv) = &mut reviser {
        let _ = rv.judge(&["dkssud", "rkwk"]);
    }

    // the models live on the worker thread; the hook and the tray need this thread's message loop
    // the installed copy keeps what it learns; a test run keeps it in memory unless given a file
    let exceptions: Box<dyn Exceptions> = match std::env::var_os("NUNCHI_EXCEPTIONS") {
        Some(p) => Box::new(store::Store::open(p.into()).expect("exceptions store")),
        None if install::installed() => match store::Store::open(install::data_dir().join("exceptions.json")) {
            Some(s) => Box::new(s),
            None => Box::new(Memory::default()),
        },
        None => Box::new(Memory::default()),
    };
    let w = worker::Worker::new(model, act, reviser, exceptions, log);
    std::thread::spawn(move || w.run());
    if install::installed() {
        autostart::init(); // never for a test run from the build folder
    }
    tray::add().expect("tray icon");
    hook::run().expect("keyboard hook");
}
