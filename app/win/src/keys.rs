//! Key classes and building `SendInput` batches.

use windows::Win32::UI::Input::KeyboardAndMouse::*;

/// dwExtraInfo on every key we send, so our own hook lets it through untouched.
pub const TAG: usize = 0x4E55_4E43; // "NUNC"

/// Between two key events we send; NUNCHI_GAP_MS overrides it for testing. Notepad test with the current
/// sending (a pause before the first key too, letters judged before they show): 2 ms lost or doubled keys
/// (2 of 2 runs), 3 ms passed 3 of 3 in both modes; 4 leaves a margin for a busy PC. (Before those changes
/// 3 ms lost a Backspace in Korean mode in 3 of 3 runs.)
pub static EVENT_GAP: std::sync::LazyLock<std::time::Duration> = std::sync::LazyLock::new(|| {
    let ms = std::env::var("NUNCHI_GAP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    std::time::Duration::from_millis(ms)
});

static START: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(std::time::Instant::now);
/// When the last key event reached the app (ours, or the user's let through), in µs since START.
static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn now_us() -> u64 {
    START.elapsed().as_micros() as u64
}

/// A key event reached the app just now.
pub fn mark_event() {
    LAST.store(now_us(), std::sync::atomic::Ordering::Relaxed);
}

/// The app is busy with what we just sent for `extra` more: the next event waits that long on top of its
/// pause (counted as if the last event were that much later).
fn settle(extra: std::time::Duration) {
    LAST.store(now_us() + extra.as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Less than `g` since the last key event reached the app.
pub fn within(g: std::time::Duration) -> bool {
    now_us() < LAST.load(std::sync::atomic::Ordering::Relaxed) + g.as_micros() as u64
}

/// Wait until `g` has passed since the last key event: the pause the app needs between events is between
/// any two, so after a quiet moment (the user's next key comes tens of ms later) there is nothing to wait for.
fn wait_gap(g: std::time::Duration) {
    let until = LAST.load(std::sync::atomic::Ordering::Relaxed) + g.as_micros() as u64;
    let now = now_us();
    if now < until {
        std::thread::sleep(std::time::Duration::from_micros(until - now));
    }
}

/// What a sent event is, for the pause before it: the IME needs time after some events and not others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// a letter key the IME composes
    Key,
    /// 한/영
    Toggle,
    Back,
    /// arrows (selecting or moving)
    Arrow,
    Shift,
    /// finished characters as Unicode
    Text,
    /// the user's own key, passed on
    Raw,
}

/// The pause before an event of each kind: EVENT_GAP, except Unicode text, which no IME composes. Notepad,
/// 72 cases of the sensitive kind (a correction mid-word, Backspace after it, undo, digits): text without a
/// pause lost nothing, as with it; Shift+Left selecting lost or misplaced keys at 0, 1 and 2 ms (4, 3, 1 of 72)
/// and none at 4; 0-2 ms on every kind lost keys after 한/영, Backspaces and retyped letters. NUNCHI_GAPS
/// overrides, e.g. `text=4`.
fn gap(kind: Kind) -> std::time::Duration {
    static GAPS: std::sync::LazyLock<Vec<(String, u64)>> = std::sync::LazyLock::new(|| {
        std::env::var("NUNCHI_GAPS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|kv| {
                let (k, v) = kv.split_once('=')?;
                Some((k.trim().to_lowercase(), v.trim().parse().ok()?))
            })
            .collect()
    });
    let name = format!("{kind:?}").to_lowercase();
    match GAPS.iter().find(|(k, _)| *k == name) {
        Some((_, ms)) => std::time::Duration::from_millis(*ms),
        None if kind == Kind::Text => std::time::Duration::ZERO,
        None => *EVENT_GAP,
    }
}

/// The model reads digit keys (its JSON has `digit0`): digits are part of a word (3시, gpt4), not its end.
pub static DIGITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Class {
    Letter(char),
    /// a digit inside a word (with `DIGITS`): typed and judged like a letter, the same in both modes
    Digit(char),
    Boundary,
    Backspace,
    Modifier,
    /// 한/영, 한자: the user chose a mode
    ImeKey,
    Undo,
    /// anything that may move the cursor or focus
    Reset,
}

pub fn classify(vk: u16, shift: bool) -> Class {
    let v = VIRTUAL_KEY(vk);
    match v {
        _ if (0x41..=0x5A).contains(&vk) => {
            let c = vk as u8 as char;
            Class::Letter(if shift { c } else { c.to_ascii_lowercase() })
        }
        VK_SPACE | VK_RETURN => Class::Boundary,
        // Shift+digit is a symbol (!@#)
        _ if (0x30..=0x39).contains(&vk) && digits() && !shift => Class::Digit(vk as u8 as char),
        _ if (VK_NUMPAD0.0..=VK_NUMPAD9.0).contains(&vk) && digits() => {
            Class::Digit((b'0' + (vk - VK_NUMPAD0.0) as u8) as char)
        }
        _ if (0x30..=0x39).contains(&vk) => Class::Boundary,
        _ if (VK_NUMPAD0.0..=VK_DIVIDE.0).contains(&vk) => Class::Boundary,
        VK_OEM_1 | VK_OEM_PLUS | VK_OEM_COMMA | VK_OEM_MINUS | VK_OEM_PERIOD | VK_OEM_2
        | VK_OEM_3 | VK_OEM_4 | VK_OEM_5 | VK_OEM_6 | VK_OEM_7 => Class::Boundary,
        VK_BACK => Class::Backspace,
        VK_SHIFT | VK_LSHIFT | VK_RSHIFT | VK_CONTROL | VK_LCONTROL | VK_RCONTROL | VK_MENU
        | VK_LMENU | VK_RMENU | VK_LWIN | VK_RWIN | VK_CAPITAL => Class::Modifier,
        VK_HANGUL | VK_HANJA => Class::ImeKey,
        VK_PAUSE => Class::Undo,
        _ => Class::Reset,
    }
}

pub fn is_shift(vk: u16) -> bool {
    matches!(VIRTUAL_KEY(vk), VK_SHIFT | VK_LSHIFT | VK_RSHIFT)
}

fn input(vk: u16, scan: u16, up: bool, extended: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if up {
        flags |= KEYEVENTF_KEYUP;
    }
    if extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: TAG,
            },
        },
    }
}

fn scan_of(vk: u16) -> u16 {
    unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) as u16 }
}

/// Which Shift keys are down. Windows keeps the two apart: letting go of one leaves the other pressed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Shifts {
    pub left: bool,
    pub right: bool,
}

impl Shifts {
    pub fn any(self) -> bool {
        self.left || self.right
    }

    /// A Shift key went down or up (VK_SHIFT, as another program may send it, counts as the left one).
    pub fn set(&mut self, vk: u16, down: bool) {
        if VIRTUAL_KEY(vk) == VK_RSHIFT {
            self.right = down;
        } else {
            self.left = down;
        }
    }

    /// What Windows has down right now, ours included.
    pub fn now() -> Self {
        let down = |vk: VIRTUAL_KEY| unsafe { GetAsyncKeyState(vk.0 as i32) < 0 };
        Shifts { left: down(VK_LSHIFT), right: down(VK_RSHIFT) }
    }
}

/// The last Shift edge we sent for each side (left, right) was a press.
static OURS: [std::sync::atomic::AtomicBool; 2] =
    [std::sync::atomic::AtomicBool::new(false), std::sync::atomic::AtomicBool::new(false)];

/// Shift keys that are down only because we pressed them: `os` has them down, the user is not holding them.
/// Only our own presses count, so a Shift the hook never saw (it can be off, or dropped by Windows) is left alone.
pub fn strays(os: Shifts, user: Shifts) -> Shifts {
    let check = |i: usize, os: bool, user: bool| {
        if !os {
            OURS[i].store(false, std::sync::atomic::Ordering::Relaxed);
        }
        os && !user && OURS[i].load(std::sync::atomic::Ordering::Relaxed)
    };
    Shifts { left: check(0, os.left, user.left), right: check(1, os.right, user.right) }
}

/// A batch of keys to send, keeping track of the Shift state it leaves.
pub struct Batch {
    pub inputs: Vec<INPUT>,
    /// the kind of each input, for the pause before it
    kinds: Vec<Kind>,
    /// the kind of what is being added
    kind: Kind,
    /// Shift as the target app sees it now (starts as the user's physical state)
    shift: Shifts,
    user_shift: Shifts,
}

impl Batch {
    pub fn new(user_shift: Shifts) -> Self {
        Batch {
            inputs: Vec::new(),
            kinds: Vec::new(),
            kind: Kind::Key,
            shift: user_shift,
            user_shift,
        }
    }

    fn push(&mut self, i: INPUT, kind: Kind) {
        self.inputs.push(i);
        self.kinds.push(kind);
    }

    pub fn tap(&mut self, vk: u16) {
        let scan = scan_of(vk);
        self.kind = match VIRTUAL_KEY(vk) {
            VK_HANGUL => Kind::Toggle,
            VK_BACK => Kind::Back,
            _ => Kind::Key,
        };
        self.push(input(vk, scan, false, false), self.kind);
        self.push(input(vk, scan, true, false), self.kind);
    }

    /// An extended key (the arrows): without the flag their scan code is read as the numpad's.
    pub fn tap_extended(&mut self, vk: u16) {
        let scan = scan_of(vk);
        self.kind = Kind::Arrow;
        self.push(input(vk, scan, false, true), Kind::Arrow);
        self.push(input(vk, scan, true, true), Kind::Arrow);
    }

    fn shift_edge(&mut self, right: bool, down: bool) {
        let vk = if right { VK_RSHIFT.0 } else { VK_LSHIFT.0 };
        self.push(input(vk, scan_of(vk), !down, false), Kind::Shift);
        self.shift.set(vk, down);
        OURS[right as usize].store(down, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_shift(&mut self, down: bool) {
        if down {
            if !self.shift.any() {
                // the key the user holds, so that their release is the one that lets it go
                self.shift_edge(self.user_shift.right && !self.user_shift.left, true);
            }
        } else {
            if self.shift.left {
                self.shift_edge(false, false);
            }
            if self.shift.right {
                self.shift_edge(true, false);
            }
        }
    }

    /// Ctrl down or up (sent as a modifier, paced like Shift).
    pub fn ctrl(&mut self, down: bool) {
        self.push(input(VK_LCONTROL.0, scan_of(VK_LCONTROL.0), !down, false), Kind::Shift);
    }

    pub fn letter(&mut self, c: char) {
        self.set_shift(c.is_ascii_uppercase());
        self.tap(c.to_ascii_uppercase() as u16);
    }

    /// Finished characters, typed as Unicode: no IME composition on the way.
    pub fn text(&mut self, s: &str) {
        self.kind = Kind::Text;
        for u in s.encode_utf16() {
            for up in [false, true] {
                let flags = if up {
                    KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                } else {
                    KEYEVENTF_UNICODE
                };
                self.push(INPUT {
                    r#type: INPUT_KEYBOARD,
                    Anonymous: INPUT_0 {
                        ki: KEYBDINPUT {
                            wVk: VIRTUAL_KEY(0),
                            wScan: u,
                            dwFlags: flags,
                            time: 0,
                            dwExtraInfo: TAG,
                        },
                    },
                }, Kind::Text);
            }
        }
    }

    /// A key exactly as the hook saw it.
    pub fn raw(&mut self, vk: u16, scan: u16, up: bool, extended: bool) {
        self.push(input(vk, scan, up, extended), Kind::Raw);
    }

    /// Put Shift back the way the user is holding it.
    pub fn restore_shift(&mut self) {
        let user = self.user_shift;
        if self.shift.left != user.left {
            self.shift_edge(false, user.left);
        }
        if self.shift.right != user.right {
            self.shift_edge(true, user.right);
        }
    }

    /// One event at a time, each at least a pause after the event before it, ours or the user's (Notepad
    /// dropped a third of the Backspaces sent in one batch, and some even at 2 ms apart; none at 3 ms). The
    /// pause depends on the kind of event (`gap`). After the last one, the hook holds a key of the user's that
    /// comes too soon (`hook::proc`), instead of this waiting a pause every time.
    pub fn send(self) -> u32 {
        let mut sent = 0;
        for (input, &kind) in self.inputs.iter().zip(&self.kinds) {
            wait_gap(gap(kind));
            sent += unsafe {
                SendInput(
                    std::slice::from_ref(input),
                    std::mem::size_of::<INPUT>() as i32,
                )
            };
            mark_event();
        }
        // a correction: the IME is still taking in the burst; whatever comes next (the user's held keys, a
        // key let through) keeps a pause more after it, as the old wait after every batch did (without it a
        // Backspace right after a mid-word correction went wrong in 2 of 72 tries)
        if self.inputs.len() > 1 {
            if let Some(&k) = self.kinds.last() {
                settle(gap(k));
            }
        }
        sent
    }

    /// For one key press the user just made: EVENT_GAP between its events, but none before or after (it
    /// arrives as fast as the user typed it). Whether down and up can go in one call (2 ms instead of 8) is
    /// untested outside Notepad: English showing up in the Claude app turned out to be a misread IME mode.
    pub fn send_press(self) -> u32 {
        let mut sent = 0;
        for (input, &kind) in self.inputs.iter().zip(&self.kinds) {
            // the first one too: a correction may have just gone out (after a quiet moment this is no wait)
            wait_gap(gap(kind));
            sent += unsafe { SendInput(std::slice::from_ref(input), std::mem::size_of::<INPUT>() as i32) };
            mark_event();
        }
        sent
    }
}


fn digits() -> bool {
    DIGITS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u16 = 0x41;
    const L: u16 = VK_LSHIFT.0;
    const R: u16 = VK_RSHIFT.0;
    const LEFT: Shifts = Shifts { left: true, right: false };
    const RIGHT: Shifts = Shifts { left: false, right: true };
    const BOTH: Shifts = Shifts { left: true, right: true };

    /// (key, up) of each event in the batch.
    fn events(b: &Batch) -> Vec<(u16, bool)> {
        b.inputs
            .iter()
            .map(|i| {
                let ki = unsafe { i.Anonymous.ki };
                (ki.wVk.0, ki.dwFlags.contains(KEYEVENTF_KEYUP))
            })
            .collect()
    }

    fn typed(user: Shifts, keys: &str) -> Vec<(u16, bool)> {
        let mut b = Batch::new(user);
        for c in keys.chars() {
            b.letter(c);
        }
        b.restore_shift();
        events(&b)
    }

    #[test]
    fn right_shift_is_let_go_and_put_back_on_the_right() {
        assert_eq!(typed(RIGHT, "a"), [(R, true), (A, false), (A, true), (R, false)]);
    }

    #[test]
    fn a_shifted_key_while_right_is_held_presses_right_not_left() {
        // the case that left Left Shift pressed: lower-case keys retyped, then the key Shift is held for
        assert_eq!(typed(RIGHT, "aA"), [(R, true), (A, false), (A, true), (R, false), (A, false), (A, true)]);
    }

    #[test]
    fn left_shift_is_let_go_and_put_back_on_the_left() {
        assert_eq!(typed(LEFT, "a"), [(L, true), (A, false), (A, true), (L, false)]);
    }

    #[test]
    fn both_shifts_are_let_go_and_put_back() {
        assert_eq!(
            typed(BOTH, "a"),
            [(L, true), (R, true), (A, false), (A, true), (L, false), (R, false)]
        );
    }

    #[test]
    fn with_no_shift_held_nothing_stays_pressed() {
        assert_eq!(typed(Shifts::default(), "A"), [(L, false), (A, false), (A, true), (L, true)]);
    }

    #[test]
    fn nothing_is_sent_when_shift_is_already_as_wanted() {
        assert_eq!(typed(RIGHT, "A"), [(A, false), (A, true)]);
    }
}
