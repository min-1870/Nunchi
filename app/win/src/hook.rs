//! The low-level keyboard hook. It does as little as possible: decide whether to hold a key back, and queue
//! every key for the worker. Letters are held (the worker decides how they show), and so are Backspace and
//! boundary keys inside a word (a Backspace may take back a held letter; the word may be corrected before
//! its boundary goes out). While the worker is busy, every key is held so nothing overtakes the keys it sends.

use crate::ime;
use crate::keys::{classify, Class, TAG};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::*;
use windows::Win32::UI::WindowsAndMessaging::*;

/// Longest the hook holds keys for the worker before giving up on it.
const FAILSAFE: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Copy)]
pub struct Ev {
    pub vk: u16,
    pub scan: u16,
    pub up: bool,
    pub extended: bool,
    /// held back by the hook: the worker must send it
    pub held: bool,
    pub hwnd: isize,
    /// when the hook got it
    pub at: Instant,
}

pub struct Shared {
    pub queue: VecDeque<Ev>,
    pub busy: bool,
    since: Instant,
    /// letters (or Backspace) since the last boundary, in this window
    has_word: bool,
    last_hwnd: isize,
    allowed: HashMap<isize, bool>,
    pub failsafes: u32,
}

pub static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| {
    Mutex::new(Shared {
        queue: VecDeque::new(),
        busy: false,
        since: Instant::now(),
        has_word: false,
        last_hwnd: 0,
        allowed: HashMap::new(),
        failsafes: 0,
    })
});
pub static WAKE: Condvar = Condvar::new();

/// Start holding keys, unless a key press the worker has not seen yet already reached the app (key
/// releases change nothing on screen). True: holding.
pub fn try_hold() -> bool {
    let mut s = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if s.queue.iter().any(|e| !e.up) {
        return false;
    }
    if !s.busy {
        s.busy = true;
        s.since = Instant::now();
    }
    true
}

/// Off from the tray: the hook lets everything through untouched.
static ENABLED: AtomicBool = AtomicBool::new(true);
/// Set when switched on or off: what the worker knew about the screen is stale.
pub static STALE: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    STALE.store(true, Ordering::Relaxed);
}

fn down(vk: VIRTUAL_KEY) -> bool {
    unsafe { GetAsyncKeyState(vk.0 as i32) < 0 }
}

unsafe extern "system" fn proc(code: i32, w: WPARAM, l: LPARAM) -> LRESULT {
    if code < 0 || !enabled() {
        return CallNextHookEx(None, code, w, l);
    }
    let k = &*(l.0 as *const KBDLLHOOKSTRUCT);
    if k.dwExtraInfo == TAG {
        return CallNextHookEx(None, code, w, l);
    }
    let up = matches!(w.0 as u32, WM_KEYUP | WM_SYSKEYUP);
    let vk = k.vkCode as u16;
    let hwnd = ime::focus();
    let h = hwnd.0 as isize;
    let mods = down(VK_CONTROL) || down(VK_MENU) || down(VK_LWIN) || down(VK_RWIN);
    let class = classify(vk, down(VK_SHIFT));

    let mut s = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if s.busy && s.since.elapsed() > FAILSAFE {
        s.busy = false;
        s.failsafes += 1;
    }
    if h != s.last_hwnd {
        s.has_word = false;
        s.last_hwnd = h;
    }
    let allowed = *s
        .allowed
        .entry(h)
        .or_insert_with(|| ime::may_send(HWND(h as _)));
    let hold_key = !up
        && allowed
        && !mods
        && match class {
            Class::Letter(_) | Class::Digit(_) => true,
            Class::Backspace | Class::Boundary => s.has_word,
            _ => false,
        };
    // a key right after one we sent (within the pause the app needs between events) waits for its turn too
    let soon = allowed && crate::keys::within(*crate::keys::EVENT_GAP);
    let held = s.busy || hold_key || soon;
    if !up {
        match class {
            Class::Letter(_) | Class::Digit(_) | Class::Backspace => s.has_word = !mods,
            Class::Modifier => {}
            _ => s.has_word = false,
        }
    }
    if held && !s.busy {
        s.busy = true;
        s.since = Instant::now();
    }
    s.queue.push_back(Ev {
        vk,
        scan: k.scanCode as u16,
        up,
        extended: k.flags.0 & LLKHF_EXTENDED.0 != 0,
        held,
        hwnd: h,
        at: Instant::now(),
    });
    drop(s);
    WAKE.notify_one();
    if held {
        LRESULT(1)
    } else {
        crate::keys::mark_event();
        CallNextHookEx(None, code, w, l)
    }
}

/// Set on a mouse click: the cursor may have moved, so the worker forgets the word it was in.
pub static CLICKED: AtomicBool = AtomicBool::new(false);

/// A click ended or the focus moved, at this time: once things settle the worker looks at the field before
/// the user types (password or not, the line around the cursor), so the first key does not wait for that.
pub static MOVED_AT: Mutex<Option<Instant>> = Mutex::new(None);

fn moved_now() {
    *MOVED_AT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    WAKE.notify_one();
}

unsafe extern "system" fn mouse(code: i32, w: WPARAM, l: LPARAM) -> LRESULT {
    if code >= 0 {
        match w.0 as u32 {
            WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN => {
                SHARED.lock().unwrap_or_else(|e| e.into_inner()).has_word = false;
                CLICKED.store(true, Ordering::Relaxed);
            }
            WM_LBUTTONUP | WM_RBUTTONUP | WM_MBUTTONUP => moved_now(),
            _ => {}
        }
    }
    CallNextHookEx(None, code, w, l)
}

unsafe extern "system" fn focus_moved(
    _: windows::Win32::UI::Accessibility::HWINEVENTHOOK,
    _: u32,
    _: HWND,
    _: i32,
    _: i32,
    _: u32,
    _: u32,
) {
    moved_now();
}

/// Install the hook and pump messages on this thread until the process ends.
pub fn run() -> windows::core::Result<()> {
    unsafe {
        let module = windows::Win32::System::LibraryLoader::GetModuleHandleW(None)?;
        let _hook = SetWindowsHookExW(WH_KEYBOARD_LL, Some(proc), Some(module.into()), 0)?;
        let _mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse), Some(module.into()), 0)?;
        let _focus = windows::Win32::UI::Accessibility::SetWinEventHook(
            EVENT_OBJECT_FOCUS,
            EVENT_OBJECT_FOCUS,
            None,
            Some(focus_moved),
            0,
            0,
            WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS,
        );
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    Ok(())
}
