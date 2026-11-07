"""Draw Nunchi's icons (a text cursor with a dot) and write them as .ico files next to this script.

    app.ico          the exe: dark rounded tile, white cursor, coral dot
    tray_<bg>_<on|off>.ico
                     tray, for a dark or light taskbar: cursor in the taskbar's ink, coral dot; off: the dot
                     is gone and the cursor greyed (on and off differ in shape and colour)

Shapes are drawn as distance fields on an 8x grid per pixel and averaged, so every size is drawn, not
scaled. Needs numpy only (PNG and ICO are written by hand).

    .venv/Scripts/python app/win/icons/make_icons.py
"""
import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
SS = 8  # samples per pixel per axis

# the glyph, in a 24 x 24 box: an I-beam (stem at x, from top to bottom, serifs reaching half_serif to each
# side) and a dot
STROKE = 2.2
STEM_X, TOP, BOTTOM, HALF_SERIF = 10.5, 3.5, 20.5, 2.5
DOT = ((18, 7), 2.6)

TILE = "#2C2C2A"
CORAL = "#D85A30"
WHITE = "#FFFFFF"
INK = "#2C2C2A"
GREY = "#888780"


def rgb(hex_):
    return np.array([int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def grid(size):
    """Sample centres in pixel units, shape (size*SS, size*SS)."""
    t = (np.arange(size * SS) + 0.5) / SS
    return np.meshgrid(t, t)


def segment(x, y, a, b, r):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    h = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(x - ax - h * dx, y - ay - h * dy) <= r


def circle(x, y, c, r):
    return np.hypot(x - c[0], y - c[1]) <= r


def rounded_rect(x, y, x0, y0, x1, y1, r):
    qx = np.maximum(np.maximum(x0 + r - x, x - (x1 - r)), 0)
    qy = np.maximum(np.maximum(y0 + r - y, y - (y1 - r)), 0)
    return (np.hypot(qx, qy) <= r) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def render(size, layers):
    """layers: [(mask fn(x, y) in pixel units, colour)], painted in order -> RGBA uint8 (size, size, 4)."""
    x, y = grid(size)
    col = np.zeros(x.shape + (3,))
    alpha = np.zeros(x.shape)
    for fn, colour in layers:
        m = fn(x, y)
        col[m] = rgb(colour)
        alpha[m] = 1
    # average premultiplied colour over each pixel's samples
    def down(a):
        return a.reshape(size, SS, size, SS, *a.shape[2:]).mean(axis=(1, 3))
    a = down(alpha)
    c = down(col * alpha[..., None])
    c = np.where(a[..., None] > 0, c / np.maximum(a[..., None], 1e-9), 0)
    return np.dstack([c, a]).clip(0, 1).__mul__(255).round().astype(np.uint8)


def glyph(scale, ox, oy, cursor, dot, hint=False):
    """Mask functions for the glyph drawn at `scale` pixels per unit, offset (ox, oy). hint: snap the cursor's
    strokes to whole pixels (small sizes; otherwise a 1.5 px stroke smears over two pixels)."""
    width = max(1, round(STROKE * scale)) if hint else STROKE * scale

    def snap(v):
        return round(v - width / 2) + width / 2 if hint else v

    # serifs are placed from the snapped stem, so they stay centred on it
    x = snap(ox + STEM_X * scale)
    top, bottom = snap(oy + TOP * scale), snap(oy + BOTTOM * scale)
    half = max(1, round(HALF_SERIF * scale)) if hint else HALF_SERIF * scale
    strokes = [((x, top), (x, bottom)), ((x - half, top), (x + half, top)), ((x - half, bottom), (x + half, bottom))]
    layers = [(lambda px, py, a=a, b=b: segment(px, py, a, b, width / 2), cursor) for a, b in strokes]
    if dot:
        c = (ox + DOT[0][0] * scale, oy + DOT[0][1] * scale)
        layers.append((lambda px, py: circle(px, py, c, DOT[1] * scale), dot))
    return layers


def app_icon(size):
    # tile 96 with the glyph 56 inside at 20, as in the drafts
    k = size / 96
    tile = (lambda x, y: rounded_rect(x, y, 0, 0, size, size, 22 * k), TILE)
    return render(size, [tile] + glyph(56 / 24 * k, 20 * k, 20 * k, WHITE, CORAL, hint=size <= 48))


def tray_icon(size, ink, on):
    return render(size, glyph(size / 24, 0, 0, ink if on else GREY, CORAL if on else None, hint=True))


def png(img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico(path, images):
    data = [png(i) for i in images]
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for img, d in zip(images, data):
        s = img.shape[0]
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(d), offset)
        offset += len(d)
    path.write_bytes(head + entries + b"".join(data))
    print(f"wrote {path.name} ({', '.join(str(i.shape[0]) for i in images)})")


if __name__ == "__main__":
    ico(OUT / "app.ico", [app_icon(s) for s in (16, 20, 24, 32, 40, 48, 64, 256)])
    tray_sizes = (16, 20, 24, 32, 40)
    for bg, ink in (("dark", WHITE), ("light", INK)):
        for on in (True, False):
            ico(OUT / f"tray_{bg}_{'on' if on else 'off'}.ico", [tray_icon(s, ink, on) for s in tray_sizes])
