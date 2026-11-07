//! The tray icon. Right-click: 시작 (start) / 정지 (stop) / Windows 시작 시 실행 (start with Windows, a check
//! mark) / 제거 (uninstall, asks first) / 종료 (quit). Nothing else to set.

use crate::{autostart, hook, install};
use std::sync::atomic::{AtomicU32, Ordering};
use windows::core::{w, Result, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_DWORD};
use windows::Win32::UI::Shell::*;
use windows::Win32::UI::WindowsAndMessaging::*;

const WM_TRAY: u32 = WM_APP + 1;
const ID_START: usize = 1;
const ID_STOP: usize = 2;
/// Also sent by a newer copy being installed, to make this one quit.
pub const ID_QUIT: usize = 3;
const ID_AUTOSTART: usize = 4;
const ID_UNINSTALL: usize = 5;
/// Explorer sends this when it restarts: the icon has to be added again.
static TASKBAR_CREATED: AtomicU32 = AtomicU32::new(0);

/// Resource ids in nunchi.rc: white glyph for a dark taskbar, dark glyph for a light one; on, off.
const ICON_DARK_ON: u16 = 101;
const ICON_DARK_OFF: u16 = 102;
const ICON_LIGHT_ON: u16 = 103;
const ICON_LIGHT_OFF: u16 = 104;

/// The taskbar is light (Windows' "default app mode" for system surfaces is light).
fn light_taskbar() -> bool {
    let mut value = 0u32;
    let mut size = std::mem::size_of::<u32>() as u32;
    let ok = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!("Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize"),
            w!("SystemUsesLightTheme"),
            RRF_RT_REG_DWORD,
            None,
            Some(&mut value as *mut u32 as *mut _),
            Some(&mut size),
        )
    };
    ok.is_ok() && value == 1
}

fn load_icon(id: u16) -> HICON {
    unsafe {
        let module = GetModuleHandleW(None).unwrap_or_default();
        let size = GetSystemMetrics(SM_CXSMICON);
        LoadImageW(Some(module.into()), PCWSTR(id as usize as *const u16), IMAGE_ICON, size, size, LR_DEFAULTCOLOR)
            .map(|h| HICON(h.0))
            .unwrap_or_default()
    }
}

fn icon(hwnd: HWND, action: NOTIFY_ICON_MESSAGE) {
    let on = hook::enabled();
    let id = match (light_taskbar(), on) {
        (false, true) => ICON_DARK_ON,
        (false, false) => ICON_DARK_OFF,
        (true, true) => ICON_LIGHT_ON,
        (true, false) => ICON_LIGHT_OFF,
    };
    let mut nid = NOTIFYICONDATAW {
        cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: hwnd,
        uID: 1,
        uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
        uCallbackMessage: WM_TRAY,
        hIcon: load_icon(id),
        ..Default::default()
    };
    let tip: Vec<u16> = (if on { "Nunchi: 켜짐" } else { "Nunchi: 꺼짐" }).encode_utf16().collect();
    nid.szTip[..tip.len()].copy_from_slice(&tip);
    unsafe {
        let _ = Shell_NotifyIconW(action, &nid);
        // the shell keeps its own copy
        let _ = DestroyIcon(nid.hIcon);
    }
}

fn menu(hwnd: HWND) {
    unsafe {
        let Ok(m) = CreatePopupMenu() else { return };
        let on = hook::enabled();
        let flag = |grayed: bool| if grayed { MF_STRING | MF_GRAYED } else { MF_STRING };
        let _ = AppendMenuW(m, flag(on), ID_START, w!("시작"));
        let _ = AppendMenuW(m, flag(!on), ID_STOP, w!("정지"));
        let _ = AppendMenuW(m, MF_SEPARATOR, 0, None);
        let checked = if autostart::enabled() { MF_STRING | MF_CHECKED } else { MF_STRING };
        let _ = AppendMenuW(m, checked, ID_AUTOSTART, w!("Windows 시작 시 실행"));
        let _ = AppendMenuW(m, MF_STRING, ID_UNINSTALL, w!("제거"));
        let _ = AppendMenuW(m, MF_SEPARATOR, 0, None);
        let _ = AppendMenuW(m, MF_STRING, ID_QUIT, w!("종료"));
        let mut p = POINT::default();
        let _ = GetCursorPos(&mut p);
        // without this the menu stays open when the user clicks elsewhere
        let _ = SetForegroundWindow(hwnd);
        let _ = TrackPopupMenu(m, TPM_RIGHTBUTTON, p.x, p.y, None, hwnd, None);
        let _ = PostMessageW(Some(hwnd), WM_NULL, WPARAM(0), LPARAM(0));
        let _ = DestroyMenu(m);
    }
}

fn quit(hwnd: HWND) {
    // the icon goes first, or it lingers until the mouse passes over it; then the message loop in
    // hook::run ends and the process with it (the hook is removed with the process)
    hook::set_enabled(false);
    icon(hwnd, NIM_DELETE);
    unsafe { PostQuitMessage(0) };
}

unsafe extern "system" fn proc(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    match msg {
        WM_TRAY if matches!(l.0 as u32, WM_RBUTTONUP | WM_CONTEXTMENU) => menu(hwnd),
        WM_COMMAND if matches!(w.0 & 0xFFFF, ID_START | ID_STOP) => {
            hook::set_enabled(w.0 & 0xFFFF == ID_START);
            icon(hwnd, NIM_MODIFY);
        }
        WM_COMMAND if w.0 & 0xFFFF == ID_AUTOSTART => autostart::set(!autostart::enabled()),
        WM_COMMAND if w.0 & 0xFFFF == ID_UNINSTALL => {
            let answer = MessageBoxW(
                Some(hwnd),
                w!("Nunchi를 제거할까요?\n자동 실행이 꺼지고 설치 폴더가 지워져요."),
                w!("Nunchi"),
                MB_YESNO | MB_ICONQUESTION | MB_SETFOREGROUND,
            );
            if answer == IDYES {
                install::uninstall();
                quit(hwnd);
            }
        }
        WM_COMMAND if w.0 & 0xFFFF == ID_QUIT => quit(hwnd),
        // light / dark taskbar switched
        WM_SETTINGCHANGE => icon(hwnd, NIM_MODIFY),
        m if m != 0 && m == TASKBAR_CREATED.load(Ordering::Relaxed) => icon(hwnd, NIM_ADD),
        _ => return DefWindowProcW(hwnd, msg, w, l),
    }
    LRESULT(0)
}

/// Create the hidden window that owns the icon, and show the icon. Call on the thread that pumps messages.
pub fn add() -> Result<()> {
    unsafe {
        let module = GetModuleHandleW(None)?;
        let class = w!("NunchiTray");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(proc),
            hInstance: module.into(),
            lpszClassName: class,
            ..Default::default()
        };
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("Nunchi"),
            WS_OVERLAPPED,
            0,
            0,
            0,
            0,
            None,
            None,
            Some(module.into()),
            None,
        )?;
        TASKBAR_CREATED.store(RegisterWindowMessageW(w!("TaskbarCreated")), Ordering::Relaxed);
        icon(hwnd, NIM_ADD);
    }
    Ok(())
}
