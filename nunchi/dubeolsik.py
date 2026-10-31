"""Hangul -> 두벌식 raw key sequence (reverse direction only, for synthetic data)."""

CHO = "rRseEfaqQtTdwWczxvg"  # ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
JUNG = ["k", "o", "i", "O", "j", "p", "u", "P", "h", "hk", "ho", "hl", "y",
        "n", "nj", "np", "nl", "b", "m", "ml", "l"]  # ㅏ..ㅣ
JONG = ["", "r", "R", "rt", "s", "sw", "sg", "e", "f", "fr", "fa", "fq", "ft",
        "fx", "fv", "fg", "a", "q", "qt", "t", "T", "d", "w", "c", "z", "x", "v", "g"]

# Compatibility jamo (ㄱ U+3131 .. ㅣ U+3163), used for standalone jamo like ㅇㅋ, ㅋㅋ
_COMPAT = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_COMPAT_KEYS = ["r", "R", "rt", "s", "sw", "sg", "e", "E", "f", "fr", "fa", "fq", "ft",
                "fx", "fv", "fg", "a", "q", "Q", "qt", "t", "T", "d", "w", "W", "c",
                "z", "x", "v", "g"] + JUNG
COMPAT = dict(zip(_COMPAT, _COMPAT_KEYS))


def is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3 or ch in COMPAT


def to_keys(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            cho, rest = divmod(code, 21 * 28)
            jung, jong = divmod(rest, 28)
            out.append(CHO[cho] + JUNG[jung] + JONG[jong])
        elif ch in COMPAT:
            out.append(COMPAT[ch])
        else:
            out.append(ch)
    return "".join(out)


_CHO_SET = set(CHO)
_JUNG_SET = set(JUNG)
_JONG_SET = set(JONG[1:])
_JAMO_KEYS = set(_COMPAT_KEYS)


def shape(keys: str) -> str:
    """How the key sequence decodes in Hangul mode.

    full  : only complete syllables (dkssud -> 안녕)
    jamo  : only standalone jamo (zg -> ㅋㅎ, sdk is ㄴ+아 so mixed)
    mixed : both
    other : contains keys that are not Hangul keys (digits, symbols)
    Uses a DP that minimises standalone jamo, not the IME's greedy order.
    """
    # Shift only matters for QWERTOP in Hangul mode; Shift+S is still ㄴ
    keys = "".join(k if k in "QWERTOP" else k.lower() for k in keys)
    if any(k not in _JAMO_KEYS for k in keys):
        return "other"
    n = len(keys)
    inf = (n + 1, 0)
    best = [inf] * (n + 1)  # (lone jamo count, -syllable count) at each prefix
    best[0] = (0, 0)
    for i in range(n):
        if best[i] == inf:
            continue
        lone, syl = best[i]
        # one standalone jamo (1 or 2 keys, e.g. ㅘ = hk)
        for L in (1, 2):
            if keys[i:i + L] in _JAMO_KEYS and i + L <= n:
                best[i + L] = min(best[i + L], (lone + 1, syl))
        # one full syllable: cho + jung(1-2) + optional jong(1-2)
        if keys[i] not in _CHO_SET:
            continue
        for jl in (1, 2):
            j = i + 1 + jl
            if j > n or keys[i + 1:j] not in _JUNG_SET:
                continue
            best[j] = min(best[j], (lone, syl - 1))
            for kl in (1, 2):
                if j + kl <= n and keys[j:j + kl] in _JONG_SET:
                    best[j + kl] = min(best[j + kl], (lone, syl - 1))
    if best[n] == inf:
        return "other"
    lone, syl = best[n]
    if lone == 0:
        return "full"
    return "jamo" if syl == 0 else "mixed"


if __name__ == "__main__":
    for w in ["안녕", "안녕하세요", "값", "ㅇㅋ", "괜찮아", "읽었다", "까치"]:
        print(w, to_keys(w), shape(to_keys(w)))
    for k in ["hello", "sdk", "zg", "bnbn", "mmh", "deploy", "react", "a1"]:
        print(k, shape(k))
