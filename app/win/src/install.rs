//! Nunchi lives in %LOCALAPPDATA%\Programs\Nunchi. Run from anywhere else (a download), it copies itself
//! there (the model is inside the exe), stops a copy already running there (so running a new download is the update),
//! starts the installed copy and exits. Runs with arguments (tests, a model given by path) stay where they
//! are. Uninstalling removes the autostart entry, our registry key and the folder.

use crate::{autostart, tray};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use windows::core::w;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::System::Registry::{RegDeleteTreeW, HKEY_CURRENT_USER};
use windows::Win32::UI::WindowsAndMessaging::{FindWindowW, PostMessageW, WM_COMMAND};

/// A separate reviser next to a model given by path, for a decider that does not revise itself (optional).
pub const REVISER_FILE: &str = "rv1_long.onnx";
/// What earlier versions installed next to the exe: the model is in the exe now, and DirectML.dll is not
/// needed (delay-loaded, never used).
const OLD: &[&str] = &["models", "DirectML.dll"];

pub fn dir() -> PathBuf {
    PathBuf::from(std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA")).join("Programs").join("Nunchi")
}

/// What it keeps for this user: learned exceptions, the debug log.
pub fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA")).join("Nunchi")
}

fn exe() -> PathBuf {
    std::env::current_exe().expect("current exe")
}

/// Running from the install folder.
pub fn installed() -> bool {
    exe().parent().is_some_and(|d| same(d, &dir()))
}

fn same(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Stop the copy running from the install folder, if any, and wait for it to go.
fn stop_running() {
    let tray = unsafe { FindWindowW(w!("NunchiTray"), None) };
    let Ok(hwnd) = tray else { return };
    unsafe {
        let _ = PostMessageW(Some(hwnd), WM_COMMAND, WPARAM(tray::ID_QUIT), LPARAM(0));
    }
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5) && unsafe { FindWindowW(w!("NunchiTray"), None) }.is_ok() {
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(Duration::from_millis(300)); // the exe file is released a moment after the window
}

fn copy(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to.parent().unwrap())?;
    // the old exe may still be closing
    for _ in 0..20 {
        if std::fs::copy(from, to).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    std::fs::copy(from, to).map(|_| ())
}

/// Called first thing. True: keep running here. False: the installed copy was started, exit.
pub fn ensure() -> bool {
    if std::env::args().len() > 1 || installed() {
        return true;
    }
    stop_running();
    let target = dir();
    if copy(&exe(), &target.join("nunchi.exe")).is_err() {
        return true; // could not install: better to run from here than not at all
    }
    for name in OLD {
        let p = target.join(name);
        let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
    }
    Command::new(target.join("nunchi.exe")).spawn().is_err()
}

/// Remove everything: autostart, our registry key, what it learned, the install folder (after this process
/// has exited).
pub fn uninstall() {
    autostart::set(false);
    unsafe {
        let _ = RegDeleteTreeW(HKEY_CURRENT_USER, w!("Software\\Nunchi"));
    }
    let _ = std::fs::remove_dir_all(data_dir());
    if installed() {
        // the running exe cannot delete its own folder: a hidden shell does it once we are gone
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = Command::new("cmd")
            .raw_arg(format!("/c ping -n 3 127.0.0.1 >nul & rmdir /s /q \"{}\"", dir().display()))
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
    }
}
