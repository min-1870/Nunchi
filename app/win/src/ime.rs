//! The foreground window: its IME mode, and whether we may send keys to it.

use nunchi_core::decide::Mode;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, WPARAM};
use windows::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY};
use windows::Win32::System::Threading::{
    OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Input::Ime::{ImmGetDefaultIMEWnd, IME_CMODE_NATIVE};
use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyboardLayout;
use windows::Win32::UI::WindowsAndMessaging::*;

const WM_IME_CONTROL: u32 = 0x0283;
const IMC_GETCONVERSIONMODE: usize = 0x0001;
const IMC_GETOPENSTATUS: usize = 0x0005;

/// The window whose keys we get: the focused one of the foreground thread, else the foreground window.
pub fn focus() -> HWND {
    unsafe {
        let fg = GetForegroundWindow();
        let tid = GetWindowThreadProcessId(fg, None);
        let mut gui = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        if GetGUIThreadInfo(tid, &mut gui).is_ok() && !gui.hwndFocus.is_invalid() {
            gui.hwndFocus
        } else {
            fg
        }
    }
}

fn ime_control(ime: HWND, what: usize) -> Option<usize> {
    let mut result = 0usize;
    let ok = unsafe {
        SendMessageTimeoutW(ime, WM_IME_CONTROL, WPARAM(what), LPARAM(0), SMTO_ABORTIFHUNG, 100, Some(&mut result))
    };
    (ok.0 != 0).then_some(result)
}

/// Korean or English mode of the IME in `hwnd`; None if it cannot be read. Korean needs the IME open and in
/// native (Hangul) conversion: in the Claude app (Chromium) 한/영 only opens and closes the IME, and the
/// conversion mode stays native in English.
///
/// None too when the window's input language is not Korean (an English-only keyboard, a Japanese or Chinese
/// IME): there 한/영 does nothing and the IME's own mode means something else, so the words are left alone.
pub fn mode(hwnd: HWND) -> Option<Mode> {
    if !korean_layout(hwnd) {
        return None;
    }
    let ime = unsafe { ImmGetDefaultIMEWnd(hwnd) };
    if ime.is_invalid() {
        return None;
    }
    let open = ime_control(ime, IMC_GETOPENSTATUS)? != 0;
    let native = ime_control(ime, IMC_GETCONVERSIONMODE)? as u32 & IME_CMODE_NATIVE.0 != 0;
    Some(if open && native { Mode::Ko } else { Mode::En })
}

/// The input language of the thread that owns `hwnd` is Korean (LANGID 0x0412).
fn korean_layout(hwnd: HWND) -> bool {
    const KOREAN: usize = 0x0412;
    unsafe {
        let tid = GetWindowThreadProcessId(hwnd, None);
        GetKeyboardLayout(tid).0 as usize & 0xFFFF == KOREAN
    }
}

/// False for elevated windows (keys we send there are dropped) and ones whose token we cannot read.
pub fn may_send(hwnd: HWND) -> bool {
    unsafe {
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        let Ok(process) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return false;
        };
        let mut token = HANDLE::default();
        let ok = OpenProcessToken(process, TOKEN_QUERY, &mut token).is_ok();
        let mut elevated = true;
        if ok {
            let mut e = TOKEN_ELEVATION::default();
            let mut len = 0u32;
            if GetTokenInformation(
                token,
                TokenElevation,
                Some(&mut e as *mut _ as *mut _),
                std::mem::size_of::<TOKEN_ELEVATION>() as u32,
                &mut len,
            )
            .is_ok()
            {
                elevated = e.TokenIsElevated != 0;
            }
            let _ = CloseHandle(token);
        }
        let _ = CloseHandle(process);
        !elevated
    }
}

