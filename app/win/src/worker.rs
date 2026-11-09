//! Takes keys from the hook in order, keeps `Typing` in step with the screen, and carries out corrections.

use crate::hook::{self, Ev, SHARED, WAKE};
use crate::ime;
use crate::uia::{BeforeCaret, Uia};
use crate::keys::{classify, is_shift, strays, Batch, Class, Shifts};
use nunchi_core::compose::{compose_str, words_of};
use nunchi_core::decide::{Mode, Thresholds};
use nunchi_core::exceptions::Exceptions;
use nunchi_core::model::Model;
use nunchi_core::typing::{KeyAction, Outcome, Plan, Typing};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::Input::KeyboardAndMouse::*;

/// Held keys are shown as typed after this long without a key (a held first key waits at most this).
const IDLE: Duration = Duration::from_millis(150);

/// After a click or a focus change, how long things settle (the app moves its caret) before the field is
/// looked at.
const SETTLE: Duration = Duration::from_millis(30);

/// What woke the worker.
enum Wake {
    Key(Ev),
    /// keys held and the user paused
    Idle,
    /// a click or focus change settled: look at the field now
    Moved,
    /// no key for a moment and none held: check Shift against Windows
    Quiet,
}

/// Threshold for a single key, whatever the main one is.
const FIRST_KEY: f64 = 0.999;

/// Words this long or longer (characters on screen) are selected with Ctrl+Shift+Left in one step (6 events)
/// instead of a Shift+Left per character (2 per character + 2).
const WORD_SELECT_MIN: usize = 3;

/// Characters read around the caret for context (cut to the line it is on).
const AROUND: usize = 200;

pub struct Worker {
    model: Model,
    thresholds: Thresholds,
    /// the model that may change the previous word once the next one is typed, with its thresholds
    reviser: Option<(Model, Thresholds)>,
    typing: Typing,
    /// the IME mode of the current window as the target app will see it next; None: read it again
    mode: Option<Mode>,
    /// `mode` was set by our own correction: use it for the next word instead of reading it
    trust_mode: bool,
    last_hwnd: isize,
    /// Shift+Left selects in the focused field: None until the first correction there checks it
    selects: Option<bool>,
    /// Ctrl+Shift+Left selects the word to replace in this field: None until tried; false once it selected
    /// something that could not be fixed up
    word_select: Option<bool>,
    /// the focused field is a password field: None until its next word checks it
    password: Option<bool>,
    /// UI Automation, made on first use on this thread; None inside: not available
    uia: Option<Option<Uia>>,
    /// the Shift keys the user is holding, as far as the keys that came through say
    shift: Shifts,
    ctrl: bool,
    alt: bool,
    win: bool,
    /// the last boundary key and the Shift state it was typed with, for undo
    last_boundary: Option<(Ev, bool)>,
    /// debug log; None: nothing is written anywhere
    log: Option<File>,
    start: Instant,
}

impl Worker {
    pub fn new(
        model: Model,
        act: f64,
        reviser: Option<Model>,
        exceptions: Box<dyn Exceptions>,
        log: Option<File>,
    ) -> Self {
        let thresholds = Thresholds {
            act,
            // the model's own mid-word threshold, else the act given
            early: model.meta.early_act_threshold.unwrap_or(act),
            first_key: FIRST_KEY,
            split: model.meta.split_threshold,
            max_keys: model.meta.max_keys_per_word,
        };
        // the reviser's own threshold, from its JSON (0.99: no TX4 word broken on the model session's dev)
        let reviser = reviser.map(|rv| {
            let t = Thresholds {
                act: rv.meta.act_threshold,
                early: rv.meta.act_threshold,
                first_key: rv.meta.act_threshold,
                split: rv.meta.split_threshold,
                max_keys: rv.meta.max_keys_per_word,
            };
            (rv, t)
        });
        Worker {
            model,
            thresholds,
            reviser,
            typing: {
                // NUNCHI_HOLD=1: hold unsure letters instead of showing them (the earlier behaviour)
                let mut t = Typing::new();
                t.exceptions = exceptions;
                t.hold_unsure = std::env::var("NUNCHI_HOLD").is_ok_and(|v| v == "1");
                // NUNCHI_HOLD_FIRST=0: show a first key leaning to the other language right away
                t.hold_first = !std::env::var("NUNCHI_HOLD_FIRST").is_ok_and(|v| v == "0");
                // NUNCHI_HABITS=0: no habit prior for short words (nothing counted either)
                t.habits = !std::env::var("NUNCHI_HABITS").is_ok_and(|v| v == "0");
                t
            },
            mode: None,
            trust_mode: false,
            last_hwnd: 0,
            selects: None,
            word_select: None,
            password: None,
            uia: None,
            shift: Shifts::default(),
            ctrl: false,
            alt: false,
            win: false,
            last_boundary: None,
            log,
            start: Instant::now(),
        }
    }

    fn write(&mut self, mut v: serde_json::Value) {
        let Some(log) = &mut self.log else { return };
        v["t"] = json!(self.start.elapsed().as_millis() as u64);
        let line = v.to_string();
        println!("{line}");
        let _ = writeln!(log, "{line}");
        let _ = log.flush();
    }

    pub fn run(mut self) {
        // UI Automation's first use costs ~25 ms: not on the user's first key
        let uia = self.uia.get_or_insert_with(Uia::new);
        let _ = uia.as_ref().map(|u| u.is_password());
        let mut failsafes = 0;
        loop {
            let ev = {
                let mut s = SHARED.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if s.failsafes != failsafes {
                        failsafes = s.failsafes;
                        drop(s);
                        self.write(json!({"event": "failsafe", "count": failsafes}));
                        s = SHARED.lock().unwrap_or_else(|e| e.into_inner());
                    }
                    if let Some(ev) = s.queue.pop_front() {
                        break Wake::Key(ev);
                    }
                    s.busy = false; // everything held has been sent
                    // a click or focus change: wake once it has settled
                    let moved = *hook::MOVED_AT.lock().unwrap_or_else(|e| e.into_inner());
                    let settle = moved.map(|t| SETTLE.saturating_sub(t.elapsed()));
                    if settle == Some(Duration::ZERO) {
                        *hook::MOVED_AT.lock().unwrap_or_else(|e| e.into_inner()) = None;
                        break Wake::Moved;
                    }
                    let (g, wait) = WAKE
                        .wait_timeout(s, settle.map_or(IDLE, |d| d.min(IDLE)))
                        .unwrap_or_else(|e| e.into_inner());
                    s = g;
                    if wait.timed_out() && s.queue.is_empty() && settle.is_none() {
                        break if self.typing.has_held() { Wake::Idle } else { Wake::Quiet };
                    }
                }
            };
            match ev {
                Wake::Moved => self.look_at_field(),
                Wake::Key(ev) if profile() && self.log.is_some() && !ev.up => {
                    // NUNCHI_PROFILE=1 (with NUNCHI_DEBUG=1): how long each key press waited in the queue
                    // and how long handling it took (sending included)
                    let wait = ev.at.elapsed();
                    let t0 = Instant::now();
                    let class = format!("{:?}", classify(ev.vk, self.shift.any()));
                    let kind = class.split('(').next().unwrap_or("").to_string();
                    self.handle(ev);
                    let w = (wait.as_secs_f64() * 100000.0).round() / 100.0;
                    self.write(json!({"event": "key", "class": kind, "held": ev.held, "wait": w, "work": ms(t0)}));
                }
                Wake::Key(ev) => self.handle(ev),
                // the user paused with keys held (leaning to the other language, not sure): show them as
                // typed rather than leave them hidden; the boundary still judges the word
                Wake::Idle if hook::try_hold() => self.flush(),
                Wake::Idle => {}
                Wake::Quiet => self.check_shift(),
            }
        }
    }

    /// Nothing is on its way, ours or the user's, so Windows can be believed about Shift. A release the hook
    /// never saw (the lock screen, an admin window, stopped from the tray) would otherwise read every later
    /// key as shifted. A Shift we pressed back as the user let go of it (the hook stops holding keys when the
    /// worker stalls, so the release can overtake us) has nobody left to release it: we do.
    fn check_shift(&mut self) {
        let os = Shifts::now();
        self.shift.left &= os.left;
        self.shift.right &= os.right;
        let stray = strays(os, self.shift);
        if stray.any() && hook::try_hold() {
            let mut b = Batch::new(stray);
            b.set_shift(false);
            self.send(b);
            self.write(json!({"event": "stray_shift", "left": stray.left, "right": stray.right}));
        }
    }

    /// Show the held keys as typed.
    fn flush(&mut self) {
        let keys = self.typing.take_held();
        self.emit(&keys);
    }

    /// Type keys as they are, in the current mode: one press at a time, each event the usual pause after the
    /// one before (see `send_press`).
    fn emit(&mut self, keys: &str) {
        for c in keys.chars() {
            let mut b = Batch::new(self.shift);
            b.letter(c);
            b.restore_shift();
            let n = b.inputs.len();
            if b.send_press() as usize != n {
                self.write(json!({"event": "error", "error": "SendInput dropped a key"}));
            }
        }
    }

    fn pass(&self, ev: &Ev) {
        if ev.held {
            let mut b = Batch::new(self.shift);
            b.raw(ev.vk, ev.scan, ev.up, ev.extended);
            b.send();
        }
    }

    fn handle(&mut self, ev: Ev) {
        if hook::STALE.swap(false, std::sync::atomic::Ordering::Relaxed) {
            // switched off and on from the tray: keys went by unseen
            self.typing.moved();
            self.mode = None;
            self.new_field();
        }
        if hook::CLICKED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            // the cursor may be anywhere now, maybe in another field of the same window
            self.flush();
            self.typing.moved();
            self.new_field();
        }
        let class = classify(ev.vk, self.shift.any());
        if class == Class::Modifier {
            let v = VIRTUAL_KEY(ev.vk);
            if is_shift(ev.vk) {
                self.shift.set(ev.vk, !ev.up);
            } else if matches!(v, VK_CONTROL | VK_LCONTROL | VK_RCONTROL) {
                self.ctrl = !ev.up;
            } else if matches!(v, VK_MENU | VK_LMENU | VK_RMENU) {
                self.alt = !ev.up;
                self.mode = None; // right Alt is 한/영 on some keyboards
            } else if matches!(v, VK_LWIN | VK_RWIN) {
                self.win = !ev.up;
            }
            self.pass(&ev);
            return;
        }
        if ev.up {
            // a held letter was (or will be) typed by us as a whole key press: drop its release
            if !(ev.held && matches!(class, Class::Letter(_) | Class::Digit(_))) {
                self.pass(&ev);
            }
            return;
        }
        if !matches!(class, Class::Letter(_) | Class::Digit(_)) {
            self.trust_mode = false; // only the word right after a correction
        }
        if ev.hwnd != self.last_hwnd {
            self.flush();
            self.typing.moved();
            self.mode = None;
            self.new_field();
            self.last_hwnd = ev.hwnd;
        }
        if self.ctrl || self.alt || self.win {
            // may move the focus inside the window (to a terminal, the address bar)
            self.flush();
            self.typing.reset();
            self.new_field();
            self.pass(&ev);
            return;
        }
        match class {
            Class::Letter(c) | Class::Digit(c) if !self.typing.in_word() && self.in_password() => {
                // never looked at, never logged
                self.flush();
                self.typing.moved();
                if ev.held {
                    self.emit(&c.to_string());
                }
            }
            Class::Letter(c) | Class::Digit(c) => {
                // read the mode at the start of every word (it can change without a key: the taskbar
                // indicator, another program), except right after a correction: the app may not have
                // processed our 한/영 key yet, and we know what it will be
                if !self.typing.in_word() && !std::mem::take(&mut self.trust_mode)
                    || self.mode.is_none()
                {
                    self.mode = ime::mode(HWND(ev.hwnd as _));
                }
                // keys are kept as Shift typed them (the 두벌식 meaning); CapsLock only flips what
                // English shows. In Korean mode CapsLock makes the IME type English, so it is not
                // tracked; in English mode the word is judged at its boundary only, since a switch in
                // the middle would retype keys in Korean mode with CapsLock on.
                match self.mode {
                    Some(Mode::En) if ev.held && caps_on() => {
                        let earlier = self.typing.letter(c, Mode::En);
                        self.emit(&earlier);
                        let held = self.typing.take_held();
                        self.emit(&held);
                    }
                    Some(m) if ev.held && !caps_on() => {
                        if self.typing.wants_context() {
                            self.read_context();
                        }
                        let earlier = self.typing.letter(c, m);
                        self.emit(&earlier);
                        self.decide_key();
                    }
                    _ => {
                        // not tracked: show what is held, then this key as it is
                        self.flush();
                        self.typing.reset();
                        if ev.held {
                            self.emit(&c.to_string());
                        }
                    }
                }
            }
            Class::Backspace => {
                // a Backspace that takes back a held key never reaches the app
                if self.typing.backspace() {
                    self.pass(&ev);
                }
            }
            Class::Boundary => self.boundary(ev),
            Class::Undo => {
                self.flush();
                self.pass(&ev);
                if let (Some(plan), Some((b, shift))) = (self.typing.undo(), self.last_boundary) {
                    self.write(json!({"event": "undo", "plan": plan_json(&plan)}));
                    self.execute(&plan, b, shift, true);
                }
            }
            Class::ImeKey | Class::Reset | Class::Modifier => {
                self.flush();
                self.typing.reset();
                if class == Class::Reset {
                    self.new_field(); // Tab and the like may move to another field
                }
                if class == Class::ImeKey {
                    self.mode = None;
                }
                self.pass(&ev);
            }
        }
    }

    /// After a held letter: judge the word so far and show, keep holding, or switch.
    fn decide_key(&mut self) {
        let t0 = Instant::now();
        let model = &mut self.model;
        let out = self.typing.key(&self.thresholds, |w| model.judge(w));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match out {
            Ok(KeyAction::Hold) => {}
            Ok(KeyAction::Release(keys)) => self.emit(&keys),
            Ok(KeyAction::Switch(o)) => {
                let plan = o.plan.clone().unwrap();
                self.log_word(&o, ms);
                self.retype_keys(&plan);
            }
            Err(e) => {
                self.write(json!({"event": "error", "error": e.to_string()}));
                self.flush();
                self.typing.reset();
            }
        }
    }

    fn log_word(&mut self, o: &Outcome, ms: f64) {
        let j = o.judgement.as_ref();
        self.write(json!({
            "event": "word",
            "keys": o.keys,
            "typed": o.typed,
            "p_ko": j.map(|j| j.p_ko),
            "p_en": j.map(|j| j.p_en),
            "p_best": j.map(|j| j.p_best),
            "path": j.map(|j| j.path.iter().map(|&l| if l == 1 { 'k' } else { 'e' }).collect::<String>()),
            "reason": o.reason,
            "fix": o.plan.as_ref().map(plan_json),
            "ms": (ms * 100.0).round() / 100.0,
        }));
    }

    /// Mid-word: Backspaces, then the keys so far as keys in the new mode, so the IME keeps composing and the
    /// user's next keys join the same syllable.
    fn retype_keys(&mut self, plan: &Plan) {
        let Some(mode) = self.mode else { return };
        self.clear_word(plan); // selected: the first key retyped replaces it
        let mut b = Batch::new(self.shift);
        if plan.mode != mode {
            b.tap(VK_HANGUL.0);
        }
        for (_, keys) in &plan.parts {
            for c in keys.chars() {
                b.letter(c);
            }
        }
        b.restore_shift();
        self.send(b);
        self.mode = Some(plan.mode);
    }

    fn send(&mut self, b: Batch) {
        let n = b.inputs.len();
        let t0 = Instant::now();
        let sent = b.send();
        if profile() {
            self.write(json!({"event": "send", "inputs": n, "ms": ms(t0)}));
        }
        if sent as usize != n {
            self.write(json!({"event": "error", "error": format!("SendInput sent {sent} of {n}")}));
        }
    }

    fn boundary(&mut self, ev: Ev) {
        if !ev.held {
            // the hook let it through (no word, CapsLock, a window we may not send to): nothing to fix
            self.flush();
            self.typing.reset();
            return;
        }
        if ev.vk != VK_SPACE.0 {
            self.password = None; // Enter or Tab may have moved on to a password field
        }
        let shift = self.shift.any();
        self.last_boundary = Some((ev, shift));
        let t0 = Instant::now();
        let model = &mut self.model;
        let space = ev.vk == VK_SPACE.0;
        let after = self.typing.after_len(); // the words after the cursor follow the judged one
        let out = self.typing.boundary(&self.thresholds, space, |w| model.judge_at(w, after));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match out {
            Ok(Some(o)) => {
                self.log_word(&o, ms);
                match o.plan {
                    Some(plan) => self.execute(&plan, ev, shift, false),
                    None => {
                        self.emit(&o.release);
                        self.pass(&ev);
                    }
                }
                if space {
                    self.revise();
                }
            }
            Ok(None) => self.pass(&ev),
            Err(e) => {
                self.write(json!({"event": "error", "error": e.to_string()}));
                self.flush();
                self.typing.reset();
                self.pass(&ev);
            }
        }
    }

    /// After a Space has gone out: the reviser may change the word before the last one. The cursor walks
    /// left over the last word, the word is replaced as text (whatever the IME mode, which stays as it is),
    /// and the cursor walks back. Keys typed meanwhile are held, as for any correction.
    fn revise(&mut self) {
        let Some((rv, t)) = &mut self.reviser else { return };
        let t0 = Instant::now();
        let out = self.typing.revise(t, |w| rv.judge(w));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (r, j) = match out {
            Ok(Some(x)) => x,
            Ok(None) => return,
            Err(e) => return self.write(json!({"event": "error", "error": e.to_string()})),
        };
        self.write(json!({
            "event": "revise",
            "keys": r.keys,
            "typed": r.from,
            "to": r.parts[0].0,
            "revert": r.revert,
            "p_ko": j.p_ko,
            "p_en": j.p_en,
            "left": r.left,
            "bs": r.backspaces,
            "ms": (ms * 100.0).round() / 100.0,
        }));
        let mut b = Batch::new(self.shift);
        for _ in 0..r.left {
            b.tap_extended(VK_LEFT.0);
        }
        for _ in 0..r.backspaces {
            b.tap(VK_BACK.0);
        }
        b.text(&text_of(&r.parts));
        for _ in 0..r.right {
            b.tap_extended(VK_RIGHT.0);
        }
        b.restore_shift();
        self.send(b);
    }

    /// Backspaces, the mode the word ends in, the finished text in one go (Unicode, no composing on screen),
    /// then the boundary key (the held one, or both edges for undo).
    fn execute(&mut self, plan: &Plan, boundary: Ev, boundary_shift: bool, undo: bool) {
        let Some(mode) = self.mode else { return };
        if undo {
            self.send_backspaces(plan.backspaces);
        } else {
            self.clear_word(plan);
        }
        let mut b = Batch::new(self.shift);
        if plan.mode != mode {
            b.tap(VK_HANGUL.0);
        }
        b.text(&text_of(&plan.parts));
        b.set_shift(boundary_shift);
        b.raw(boundary.vk, boundary.scan, false, boundary.extended);
        if undo {
            b.raw(boundary.vk, boundary.scan, true, boundary.extended);
        }
        b.restore_shift();
        self.send(b);
        self.mode = Some(plan.mode);
        self.trust_mode = true;
    }

    /// Make way for the corrected word: select it with Shift+Left so it changes in one step when typed
    /// over, or Backspace it. Selecting only where the field's text, read with UI Automation, shows the word
    /// right before the caret; checked at the first correction in a field and kept until the focus may have
    /// moved (a mismatch is checked again next time: the app may not have shown our last key yet).
    fn clear_word(&mut self, plan: &Plan) {
        let n = plan.shown.chars().count();
        if n == 0 || !select() || self.selects == Some(false) {
            return self.send_backspaces(plan.backspaces);
        }
        if self.selects.is_none() {
            let t0 = Instant::now();
            let uia = self.uia.get_or_insert_with(Uia::new);
            let seen = uia.as_ref().map_or(BeforeCaret::NoCaret, |u| u.before_caret(n));
            let ok = seen == BeforeCaret::Text(plan.shown.clone());
            if ok {
                self.selects = Some(true);
            } else if matches!(seen, BeforeCaret::NoCaret | BeforeCaret::NotEditable) {
                self.selects = Some(false);
            }
            self.write(json!({"event": "select", "ok": ok, "want": plan.shown, "seen": format!("{seen:?}"), "ms": ms(t0)}));
            if !ok {
                return self.send_backspaces(plan.backspaces);
            }
        }
        if n >= WORD_SELECT_MIN && self.word_select != Some(false) && word_select_on() && self.select_word(&plan.shown) {
            return;
        }
        let mut b = Batch::new(self.shift);
        select_left(&mut b, n);
        b.restore_shift();
        self.send(b);
    }

    /// Select the word before the caret with Ctrl+Shift+Left, then read what got selected (UI Automation) and
    /// fix it up to exactly `shown`: fewer characters are added with Shift+Left (a word the app splits, as at
    /// an apostrophe), more are let go with Shift+Right (digits left on screen before the word). Anything
    /// else: the selection is dropped (Right puts the caret back where it was) and false.
    fn select_word(&mut self, shown: &str) -> bool {
        let t0 = Instant::now();
        let n = shown.chars().count();
        let mut b = Batch::new(self.shift);
        b.ctrl(true);
        b.set_shift(true);
        b.tap_extended(VK_LEFT.0);
        b.set_shift(false);
        b.ctrl(false);
        b.restore_shift();
        self.send(b);
        // the app takes the keys in a moment after they are sent
        let uia = self.uia.get_or_insert_with(Uia::new);
        let mut seen = None;
        for _ in 0..10 {
            seen = uia.as_ref().and_then(|u| u.selection()).filter(|s| !s.is_empty());
            if seen.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let got = seen.as_deref().map_or(0, |s| s.chars().count());
        let fix = match &seen {
            Some(s) if s == shown => Some((0, VK_LEFT)),
            Some(s) if got > n && s.ends_with(shown) => Some((got - n, VK_RIGHT)),
            Some(s) if got < n && shown.ends_with(s.as_str()) => Some((n - got, VK_LEFT)),
            _ => None,
        };
        let mut b = Batch::new(self.shift);
        match fix {
            Some((k, arrow)) => {
                if k > 0 {
                    b.set_shift(true);
                    for _ in 0..k {
                        b.tap_extended(arrow.0);
                    }
                    b.set_shift(false);
                }
            }
            None if got > 0 => b.tap_extended(VK_RIGHT.0), // back to where the caret was
            None => {}
        }
        b.restore_shift();
        self.send(b);
        if fix.is_none() {
            self.word_select = Some(false);
        } else {
            self.word_select = Some(true);
        }
        self.write(json!({"event": "select_word", "ok": fix.is_some(), "want": n, "got": got, "fix": fix.map(|f| f.0), "ms": ms(t0)}));
        fix.is_some()
    }

    /// A click or a focus change has settled, no key waiting: start over in whatever field has the focus now,
    /// and look at it before the user types there (password or not, the line around the cursor), which the
    /// first key would otherwise wait for. Keys held mid-word are shown first.
    fn look_at_field(&mut self) {
        if !hook::try_hold() {
            return; // a key came meanwhile: it goes first, and does this itself
        }
        let t0 = Instant::now();
        let clicked = hook::CLICKED.swap(false, std::sync::atomic::Ordering::Relaxed);
        // a focus event without a new field (a popup, a menu): the word being typed goes on
        let changed = self.uia.get_or_insert_with(Uia::new).as_mut().is_none_or(|u| u.focus_changed());
        if !clicked && !changed {
            return;
        }
        self.flush();
        self.typing.moved();
        self.mode = None;
        self.trust_mode = false;
        self.new_field();
        self.last_hwnd = ime::focus().0 as isize;
        let password = self.in_password();
        if !password {
            self.read_context();
        }
        self.write(json!({"event": "field", "password": password, "ms": ms(t0)}));
    }

    /// A word starts with nothing known around the cursor (a new field, a click, after 한/영 or Tab): read
    /// the line the cursor is on, so the model judges the word with the words before it, and at its boundary
    /// with the words after it too (only a model that reads the whole window, TR, was trained with words
    /// after the judged one). What is read is never logged. NUNCHI_CONTEXT=0: not read.
    fn read_context(&mut self) {
        if !context_on() {
            return;
        }
        let t0 = Instant::now();
        let uia = self.uia.get_or_insert_with(Uia::new);
        let Some((before, after)) = uia.as_ref().and_then(|u| u.around_caret(AROUND, AROUND)) else {
            return self.write(json!({"event": "context", "read": false, "ms": ms(t0)}));
        };
        let digits = crate::keys::DIGITS.load(std::sync::atomic::Ordering::Relaxed);
        let before = before.rsplit(['\n', '\r']).next().unwrap_or("");
        let after = after.split(['\n', '\r']).next().unwrap_or("");
        let mut b = words_of(before, digits);
        // the cursor right after a word: the keys typed now join it, a word the model cannot see whole. A
        // word right after the cursor is kept: the boundary that ends the typed word comes between them, and
        // only the boundary reads the words after.
        if before.chars().last().is_some_and(|c| !words_of(&c.to_string(), digits).is_empty()) {
            b.pop();
        }
        let a = if self.model.meta.revises { words_of(after, digits) } else { Vec::new() };
        self.write(json!({"event": "context", "read": true, "before": b.len(), "after": a.len(), "ms": ms(t0)}));
        self.typing.set_context(b, a);
    }

    /// The focus may be in another field now: check it again before relying on it.
    fn new_field(&mut self) {
        self.selects = None;
        self.word_select = None;
        self.password = None;
    }

    /// The focused field is a password field (asked once per field; unknown counts as not).
    fn in_password(&mut self) -> bool {
        if self.password.is_none() {
            let uia = self.uia.get_or_insert_with(Uia::new);
            self.password = Some(uia.as_ref().and_then(|u| u.is_password()).unwrap_or(false));
        }
        self.password == Some(true)
    }

    fn send_backspaces(&mut self, n: usize) {
        let mut b = Batch::new(self.shift);
        for _ in 0..n {
            b.tap(VK_BACK.0);
        }
        self.send(b);
    }
}

fn ms(t0: Instant) -> f64 {
    (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

/// Replace a word by selecting it with Shift+Left and typing over it (NUNCHI_SELECT=0: Backspace it first).
fn select_left(b: &mut Batch, chars: usize) {
    b.set_shift(true);
    for _ in 0..chars {
        b.tap_extended(VK_LEFT.0);
    }
    b.set_shift(false);
}

fn profile() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NUNCHI_PROFILE").is_ok_and(|v| v == "1"))
}

/// NUNCHI_WORD_SELECT=0: always a Shift+Left per character.
fn word_select_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NUNCHI_WORD_SELECT").is_ok_and(|v| v == "0"))
}

fn context_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NUNCHI_CONTEXT").is_ok_and(|v| v == "0"))
}

fn select() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NUNCHI_SELECT").is_ok_and(|v| v == "0"))
}

/// What `parts` show once typed as text.
fn text_of(parts: &[(Mode, String)]) -> String {
    parts
        .iter()
        .map(|(m, keys)| match m {
            // typed as text, CapsLock does not apply: show the case English mode would
            Mode::En if caps_on() => keys.chars().map(swap_case).collect::<String>(),
            Mode::En => keys.clone(),
            Mode::Ko => compose_str(keys),
        })
        .collect()
}

/// CapsLock as the user's keyboard has it.
fn caps_on() -> bool {
    unsafe { GetKeyState(VK_CAPITAL.0 as i32) & 1 != 0 }
}

fn swap_case(c: char) -> char {
    if c.is_ascii_uppercase() {
        c.to_ascii_lowercase()
    } else {
        c.to_ascii_uppercase()
    }
}

fn plan_json(p: &Plan) -> serde_json::Value {
    json!({
        "bs": p.backspaces,
        "parts": p.parts.iter().map(|(m, k)| json!([m, k])).collect::<Vec<_>>(),
        "mode": p.mode,
    })
}
