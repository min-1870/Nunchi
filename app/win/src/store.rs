//! Exceptions on disk, in %LOCALAPPDATA%\Nunchi\exceptions.json, as keyed hashes only: an exception can be a
//! piece of a password typed into a plain field (a terminal's password prompt), so the keys are never
//! written. Each word is stored as HMAC-SHA256 of every prefix (the last one the whole word: a word being
//! typed that starts an exception is not switched midway). The HMAC key is random, made on first use and
//! kept in the same file encrypted with DPAPI for this Windows user: the file alone, copied elsewhere,
//! cannot be checked against guesses.
//!
//! The same file keeps the habit counts of short words (how often each ended up Korean and English), under
//! the same keyed hashes. They change with almost every short word, so they are written at most every
//! SAVE_EVERY (and with any exception change); a few uses can be lost when the app is closed.

use nunchi_core::decide::Mode;
use nunchi_core::exceptions::{bump, Exceptions};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const SAVE_EVERY: Duration = Duration::from_secs(30);
use windows::Win32::Foundation::{HLOCAL, LocalFree};
use windows::Win32::Security::Cryptography::{
    BCryptGenRandom, BCryptHash, CryptProtectData, CryptUnprotectData, BCRYPT_HMAC_SHA256_ALG_HANDLE,
    BCRYPT_USE_SYSTEM_PREFERRED_RNG, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
};

pub struct Store {
    path: PathBuf,
    key: Vec<u8>,
    /// whole-word hash -> (prefix hashes, how sure the model was of the correction undone)
    words: HashMap<String, (Vec<String>, f64)>,
    /// prefix hash -> words starting with it
    prefixes: HashMap<String, usize>,
    /// word hash -> (uses as Korean, as English)
    habits: HashMap<String, (f64, f64)>,
    saved: Instant,
}

impl Store {
    /// The store at `path`, or a new one if there is none (or it cannot be read: start over).
    pub fn open(path: PathBuf) -> Option<Self> {
        let saved: Option<Value> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let key = saved
            .as_ref()
            .and_then(|v| v["key"].as_str())
            .and_then(unhex)
            .and_then(|b| dpapi(&b, false));
        let (key, fresh) = match key {
            Some(k) => (k, false),
            None => (random_key()?, true),
        };
        let mut s =
            Store { path, key, words: HashMap::new(), prefixes: HashMap::new(), habits: HashMap::new(), saved: Instant::now() };
        if !fresh {
            for w in saved.as_ref().and_then(|v| v["words"].as_array()).into_iter().flatten() {
                let ps: Vec<String> =
                    w["prefixes"].as_array().into_iter().flatten().filter_map(|p| p.as_str()).map(String::from).collect();
                if let Some(whole) = ps.last().cloned() {
                    s.insert(whole, ps, w["p"].as_f64().unwrap_or(1.0));
                }
            }
            for (h, c) in saved.as_ref().and_then(|v| v["habits"].as_object()).into_iter().flatten() {
                if let (Some(ko), Some(en)) = (c[0].as_f64(), c[1].as_f64()) {
                    s.habits.insert(h.clone(), (ko, en));
                }
            }
        }
        Some(s)
    }

    fn hash(&self, keys: &str) -> String {
        let mut out = [0u8; 32];
        unsafe {
            let _ = BCryptHash(BCRYPT_HMAC_SHA256_ALG_HANDLE, Some(&self.key), keys.as_bytes(), &mut out);
        }
        out.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn insert(&mut self, whole: String, ps: Vec<String>, p: f64) {
        if self.words.contains_key(&whole) {
            return;
        }
        for h in &ps {
            *self.prefixes.entry(h.clone()).or_default() += 1;
        }
        self.words.insert(whole, (ps, p));
    }

    fn save(&mut self) {
        self.saved = Instant::now();
        let Some(key) = dpapi(&self.key, true) else { return };
        let words: Vec<Value> = self.words.values().map(|(ps, p)| json!({"prefixes": ps, "p": p})).collect();
        let habits: serde_json::Map<String, Value> =
            self.habits.iter().map(|(h, (ko, en))| (h.clone(), json!([ko, en]))).collect();
        let v = json!({"key": key.iter().map(|b| format!("{b:02x}")).collect::<String>(), "words": words,
                       "habits": habits});
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&self.path, serde_json::to_vec(&v).unwrap());
    }
}

impl Exceptions for Store {
    fn contains(&self, keys: &str) -> bool {
        self.words.contains_key(&self.hash(keys))
    }

    fn starts(&self, keys: &str) -> bool {
        self.prefixes.contains_key(&self.hash(keys))
    }

    fn add(&mut self, keys: &str, p: f64) {
        let ps: Vec<String> = (1..=keys.len()).map(|n| self.hash(&keys[..n])).collect();
        self.insert(ps.last().unwrap().clone(), ps, p);
        self.save();
    }

    fn remove(&mut self, keys: &str) {
        let Some((ps, _)) = self.words.remove(&self.hash(keys)) else { return };
        for h in ps {
            if let Some(n) = self.prefixes.get_mut(&h) {
                *n -= 1;
                if *n == 0 {
                    self.prefixes.remove(&h);
                }
            }
        }
        self.save();
    }

    fn habit(&self, keys: &str) -> (f64, f64) {
        self.habits.get(&self.hash(keys)).copied().unwrap_or_default()
    }

    fn count(&mut self, keys: &str, mode: Mode) {
        let h = self.hash(keys);
        bump(self.habits.entry(h).or_default(), mode);
        if self.saved.elapsed() >= SAVE_EVERY {
            self.save();
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.save();
    }
}

fn random_key() -> Option<Vec<u8>> {
    let mut k = vec![0u8; 32];
    unsafe { BCryptGenRandom(None, &mut k, BCRYPT_USE_SYSTEM_PREFERRED_RNG).ok().ok()? };
    Some(k)
}

/// Encrypt (`protect`) or decrypt `data` for this Windows user.
fn dpapi(data: &[u8], protect: bool) -> Option<Vec<u8>> {
    let input = CRYPT_INTEGER_BLOB { cbData: data.len() as u32, pbData: data.as_ptr() as *mut u8 };
    let mut out = CRYPT_INTEGER_BLOB::default();
    unsafe {
        if protect {
            CryptProtectData(&input, None, None, None, None, CRYPTPROTECT_UI_FORBIDDEN, &mut out).ok()?;
        } else {
            CryptUnprotectData(&input, None, None, None, None, CRYPTPROTECT_UI_FORBIDDEN, &mut out).ok()?;
        }
        let v = std::slice::from_raw_parts(out.pbData, out.cbData as usize).to_vec();
        let _ = LocalFree(Some(HLOCAL(out.pbData as _)));
        Some(v)
    }
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| s.get(i..i + 2).and_then(|b| u8::from_str_radix(b, 16).ok())).collect()
}
