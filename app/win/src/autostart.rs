//! Starting with Windows: a value under HKCU\...\Run pointing at this exe. On by default: set on the first
//! run (marked under HKCU\Software\Nunchi), and from then on only the tray menu changes it.

use windows::core::{w, PCWSTR};
use windows::Win32::System::Registry::*;

const RUN: PCWSTR = w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run");
const NAME: PCWSTR = w!("Nunchi");
const OURS: PCWSTR = w!("Software\\Nunchi");
const SET_UP: PCWSTR = w!("AutostartSetUp");

fn command() -> String {
    let exe = std::env::current_exe().unwrap_or_default();
    format!("\"{}\"", exe.display())
}

fn read_string(key: PCWSTR, name: PCWSTR) -> Option<String> {
    let mut buf = [0u16; 1024];
    let mut size = std::mem::size_of_val(&buf) as u32;
    let ok = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key,
            name,
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr() as *mut _),
            Some(&mut size),
        )
    };
    ok.is_ok().then(|| String::from_utf16_lossy(&buf[..(size as usize / 2).saturating_sub(1)]))
}

fn write_string(key: PCWSTR, name: PCWSTR, value: &str) {
    let wide: Vec<u16> = value.encode_utf16().chain([0]).collect();
    unsafe {
        let _ = RegSetKeyValueW(
            HKEY_CURRENT_USER,
            key,
            name,
            REG_SZ.0,
            Some(wide.as_ptr() as *const _),
            (wide.len() * 2) as u32,
        );
    }
}

pub fn enabled() -> bool {
    read_string(RUN, NAME).is_some()
}

pub fn set(on: bool) {
    if on {
        write_string(RUN, NAME, &command());
    } else {
        unsafe {
            let _ = RegDeleteKeyValueW(HKEY_CURRENT_USER, RUN, NAME);
        }
    }
}

/// At startup: turn it on the first time; if on, point it at this exe (it may have moved).
pub fn init() {
    if read_string(OURS, SET_UP).is_none() {
        set(true);
        write_string(OURS, SET_UP, "1");
    } else if read_string(RUN, NAME).is_some_and(|c| c != command()) {
        set(true);
    }
}
