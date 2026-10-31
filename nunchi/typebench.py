"""Drive a real Windows text box with real keystrokes, to benchmark IME auto-switch tools.

Opens Notepad, sets the IME mode (Hangul / English), sends virtual-key presses (so the IME
and any keyboard hook see them like a person typing), then reads the text back.
"""
import ctypes
import subprocess
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
imm32 = ctypes.WinDLL("imm32")

WM_IME_CONTROL, IMC_GETCONVERSIONMODE, IMC_SETCONVERSIONMODE = 0x283, 0x001, 0x002
WM_GETTEXT, WM_GETTEXTLENGTH = 0x000D, 0x000E
INPUT_KEYBOARD, KEYEVENTF_KEYUP = 1, 0x0002
VK_SHIFT, VK_SPACE, VK_BACK, VK_RETURN, VK_CONTROL = 0x10, 0x20, 0x08, 0x0D, 0x11
MAPVK_VK_TO_VSC = 0


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]


user32.SendMessageW.restype = ctypes.c_ssize_t
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
imm32.ImmGetDefaultIMEWnd.restype = wintypes.HWND
imm32.ImmGetDefaultIMEWnd.argtypes = [wintypes.HWND]


def _key(vk, up=False):
    scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, scan, KEYEVENTF_KEYUP if up else 0, 0, 0))


def press(vk, shift=False, ctrl=False, delay=0.03):
    seq = []
    if ctrl:
        seq.append(_key(VK_CONTROL))
    if shift:
        seq.append(_key(VK_SHIFT))
    seq += [_key(vk), _key(vk, True)]
    if shift:
        seq.append(_key(VK_SHIFT, True))
    if ctrl:
        seq.append(_key(VK_CONTROL, True))
    arr = (INPUT * len(seq))(*seq)
    user32.SendInput(len(seq), arr, ctypes.sizeof(INPUT))
    time.sleep(delay)


def type_keys(keys, delay=0.03):
    """keys: raw key sequence, letters/space only (uppercase = Shift)."""
    for ch in keys:
        if ch == " ":
            press(VK_SPACE, delay=delay)
        elif ch == "\n":
            press(VK_RETURN, delay=delay)
        elif ch.isalpha():
            press(ord(ch.upper()), shift=ch.isupper(), delay=delay)
        else:
            raise ValueError(f"unsupported key {ch!r}")


def focus_hwnd():
    gti = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    user32.GetGUIThreadInfo(0, ctypes.byref(gti))
    return gti.hwndFocus or user32.GetForegroundWindow()


def ime_mode(hwnd=None):
    ime = imm32.ImmGetDefaultIMEWnd(hwnd or focus_hwnd())
    return user32.SendMessageW(ime, WM_IME_CONTROL, IMC_GETCONVERSIONMODE, 0)


def set_ime(hangul, hwnd=None):
    ime = imm32.ImmGetDefaultIMEWnd(hwnd or focus_hwnd())
    user32.SendMessageW(ime, WM_IME_CONTROL, IMC_SETCONVERSIONMODE, 1 if hangul else 0)
    time.sleep(0.1)


def read_text(hwnd=None):
    hwnd = hwnd or focus_hwnd()
    n = user32.SendMessageW(hwnd, WM_GETTEXTLENGTH, 0, 0)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.SendMessageW(hwnd, WM_GETTEXT, n + 1, ctypes.cast(buf, ctypes.c_void_p).value)
    return buf.value


def clear():
    press(ord("A"), ctrl=True)
    press(VK_BACK)


def open_notepad():
    subprocess.Popen(["notepad.exe"])
    time.sleep(2.5)
    return focus_hwnd()


VK_HANGUL = 0x15


def to_mode(hangul, hwnd):
    if bool(ime_mode(hwnd) & 1) != hangul:
        press(VK_HANGUL)
        time.sleep(0.25)


def double_tap():
    press(VK_HANGUL, delay=0.05)
    press(VK_HANGUL, delay=0.05)
    time.sleep(1.0)


# ---------------------------------------------------------------- benchmark
#
# Each case is a sentence prefix that ends right after the user's mistake:
#   start_ko / start_en : the first 1-2 words of a sentence typed in the wrong mode
#   switch_ko_en        : real NSMC review, Korean typed right, then an English word typed in Hangul mode
#   switch_en_ko        : English sentence, then a Korean word typed in English mode (synthetic insert)
# Words are (word, lab, wrong_mode). The "ideal user" double-taps right after the mistake.

def build_cases(n, seed=0):
    import random

    from nunchi.context import build, real_mixed

    rnd = random.Random(seed)
    sets = build()["test"]
    cases = []

    def add(kind, words):
        if len(words) <= 8 and any(w[2] for w in words):
            cases.append({"kind": kind, "words": words})

    for kind, lab, pool in (("start_ko", "ko", sets["ko"]), ("start_en", "en", sets["en"])):
        for sent in rnd.sample(pool, n):
            k = rnd.choice([1, 1, 2])
            add(kind, [(w, lab, True) for _, w in sent[:k]])
    for kind, pool in (("switch_ko_en", real_mixed()[0]), ("switch_en_ko", sets["en_ko"])):
        picked = 0
        for sent in rnd.sample(pool, len(pool)):
            base = sent[0][0]
            i = next((i for i, (lab, _) in enumerate(sent) if lab != base), None)
            if i is None or i == 0 or i >= 8:
                continue
            add(kind, [(w, lab, lab != base) for lab, w in sent[:i + 1]])
            picked += 1
            if picked == n:
                break
    return cases


def truth(case):
    return " ".join(w for w, _, _ in case["words"])


def type_case(case, hwnd):
    """Type each word in the mode the user would be in: the right one, or the wrong one if marked."""
    from nunchi.dubeolsik import to_keys

    for i, (w, lab, wrong) in enumerate(case["words"]):
        to_mode((lab == "ko") != wrong, hwnd)
        type_keys(("" if i == 0 else " ") + to_keys(w))
    time.sleep(0.3)


def norm(text):
    """Compare English case-insensitively: Shift is lost when English is typed in Hangul mode."""
    return " ".join(text.split()).lower()


def word_diff(out, case):
    """(mistake words still wrong, right words broken) by position."""
    got = norm(out).split(" ")
    want = [norm(w) for w, _, _ in case["words"]]
    if len(got) != len(want):
        return sum(wr for _, _, wr in case["words"]), sum(not wr for _, _, wr in case["words"])
    missed = sum(g != t for g, t, (_, _, wr) in zip(got, want, case["words"]) if wr)
    broke = sum(g != t for g, t, (_, _, wr) in zip(got, want, case["words"]) if not wr)
    return missed, broke


def wait_for_person(hwnd, before, timeout):
    """Beep, then wait until the text changes (the person double-tapped) or time runs out."""
    import winsound

    winsound.Beep(880, 120)
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(0.1)
        if read_text(hwnd) != before:
            time.sleep(0.8)  # let the tool finish rewriting
            return read_text(hwnd)
    return read_text(hwnd)


def run(cases, out_path, ours=True, manual=False, timeout=12):
    """manual: the person presses the trigger (for tools that ignore injected keys)."""
    import json

    hwnd = open_notepad()
    model = None
    if ours:
        from nunchi.demo import Model
        model = Model()
    rows = []
    for i, case in enumerate(cases):
        clear()
        time.sleep(0.2)
        type_case(case, hwnd)
        screen = read_text(hwnd)
        if manual:
            tool_out = wait_for_person(hwnd, screen, timeout)
        else:
            double_tap()
            tool_out = read_text(hwnd)
        row = {**case, "truth": truth(case), "screen": screen, "tool": tool_out,
               "tool_diff": word_diff(tool_out, case)}
        if model:
            row["ours"] = model.run(screen, "2state", 0.99)["output"]
            row["ours_diff"] = word_diff(row["ours"], case)
        rows.append(row)
        print(i, case["kind"], repr(screen), "->", repr(tool_out), "| ours:", repr(row.get("ours")), flush=True)
    clear()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def report(path):
    import json
    from collections import defaultdict

    rows = json.load(open(path, encoding="utf-8"))
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        a = agg[r["kind"]]
        a["n"] += 1
        a["mistake_words"] += sum(wr for _, _, wr in r["words"])
        a["right_words"] += sum(not wr for _, _, wr in r["words"])
        for tool in ("tool", "ours"):
            if tool in r:
                a[tool + "_exact"] += norm(r[tool]) == norm(r["truth"])
                a[tool + "_missed"] += r[tool + "_diff"][0]
                a[tool + "_broke"] += r[tool + "_diff"][1]
    for kind, a in agg.items():
        print(f"{kind:<13} n={a['n']:>3} mistake words={a['mistake_words']:>3} right words={a['right_words']:>3}")
        for tool in ("tool", "ours"):
            if a[tool + "_exact"] or a[tool + "_missed"] or a[tool + "_broke"] or tool == "tool":
                print(f"   {tool:<8} sentence right {a[tool + '_exact']:>3}/{a['n']}  "
                      f"mistakes left {a[tool + '_missed']:>3}/{a['mistake_words']}  "
                      f"right words broken {a[tool + '_broke']:>3}/{a['right_words']}")


if __name__ == "__main__":
    import sys

    if sys.argv[1] in ("run", "manual"):
        n, out = int(sys.argv[2]), sys.argv[3]
        run(build_cases(n), out, manual=sys.argv[1] == "manual")
    else:
        report(sys.argv[2])
