"""Draw Nunchi's icons (two lanes swapping places) and write them as .ico files next to this script.

    app.ico          the exe: dark rounded tile, coral lane rising, white lane falling over it with a gap
    tray_<bg>_<on|off>.ico
                     tray, no tile: the falling lane in the taskbar's ink, the coral lane cut where it passes
                     under; off: the coral lane is gone and the other is grey (on and off differ in shape and
                     colour)

The shapes follow branding/nunchi-icon.svg (app) and branding/nunchi-mark*.svg (tray). Each is drawn as a
distance field on an 8x grid per pixel and averaged, so every size is drawn, not scaled. Needs numpy only
(PNG and ICO are written by hand).

    .venv/Scripts/python app/win/icons/make_icons.py
"""
import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
SS = 8  # samples per pixel per axis

TILE = "#1C1C1B"
CORAL = "#D85A30"
WHITE = "#FFFFFF"
INK = "#1C1C1B"
GREY = "#888780"

# the lanes of branding/nunchi-icon.svg and nunchi-mark*.svg (96 x 96 units): stroke width, and width of the gap
# cut around the top lane. The lanes span x 12..84 and y 28..68.
UP = ((18, 62), (42, 62), (54, 34), (78, 34))
DOWN = ((18, 34), (42, 34), (54, 62), (78, 62))
STROKE, CUT = 12, 20
MARK_BOX = (12, 28, 84, 68)


def rgb(hex_):
    return np.array([int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def grid(size):
    """Sample centres in pixel units, shape (size*SS, size*SS)."""
    t = (np.arange(size * SS, dtype=np.float32) + 0.5) / SS
    return np.meshgrid(t, t)


def bezier(p, n=96):
    """Points along a cubic Bezier."""
    t = np.linspace(0, 1, n)[:, None]
    p = np.array(p, dtype=float)
    return ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])


def stroke(x, y, points, r):
    """Inside a round-capped stroke of radius r along the polyline `points` (pixel units)."""
    inside = np.zeros(x.shape, bool)
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        # only the samples near this segment
        x0, x1 = min(ax, bx) - r, max(ax, bx) + r
        y0, y1 = min(ay, by) - r, max(ay, by) + r
        cols = slice(max(0, int(x0 * SS)), int(x1 * SS) + 2)
        rows = slice(max(0, int(y0 * SS)), int(y1 * SS) + 2)
        px, py = x[rows, cols], y[rows, cols]
        dx, dy = bx - ax, by - ay
        h = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0, 1)
        inside[rows, cols] |= np.hypot(px - ax - h * dx, py - ay - h * dy) <= r
    return inside


def rounded_rect(x, y, x0, y0, x1, y1, r):
    qx = np.maximum(np.maximum(x0 + r - x, x - (x1 - r)), 0)
    qy = np.maximum(np.maximum(y0 + r - y, y - (y1 - r)), 0)
    return (np.hypot(qx, qy) <= r) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def render(size, layers):
    """layers: [(mask fn(x, y) in pixel units, colour or None to erase)], painted in order -> RGBA uint8."""
    x, y = grid(size)
    col = np.zeros(x.shape + (3,), np.float32)
    alpha = np.zeros(x.shape, np.float32)
    for fn, colour in layers:
        m = fn(x, y)
        if colour is None:
            alpha[m] = 0
        else:
            col[m] = rgb(colour)
            alpha[m] = 1

    # average premultiplied colour over each pixel's samples
    def down(a):
        return a.reshape(size, SS, size, SS, *a.shape[2:]).mean(axis=(1, 3))
    a = down(alpha)
    c = down(col * alpha[..., None])
    c = np.where(a[..., None] > 0, c / np.maximum(a[..., None], 1e-9), 0)
    return np.dstack([c, a]).clip(0, 1).__mul__(255).round().astype(np.uint8)


def lane(curve, k, ox, oy):
    return [(px * k + ox, py * k + oy) for px, py in bezier(curve)]


def app_icon(size):
    k = size / 96
    up, down = lane(UP, k, 0, 0), lane(DOWN, k, 0, 0)
    return render(size, [
        (lambda x, y: rounded_rect(x, y, 0, 0, size, size, 22 * k), TILE),
        (lambda x, y: stroke(x, y, up, STROKE * k / 2), CORAL),
        (lambda x, y: stroke(x, y, down, CUT * k / 2), TILE),
        (lambda x, y: stroke(x, y, down, STROKE * k / 2), WHITE),
    ])


def tray_icon(size, ink, on):
    # the mark's box fills the icon's width, centred vertically
    x0, y0, x1, y1 = MARK_BOX
    k = size / (x1 - x0)
    ox, oy = -x0 * k, (size - (y1 - y0) * k) / 2 - y0 * k
    up, down = lane(UP, k, ox, oy), lane(DOWN, k, ox, oy)
    layers = []
    if on:
        layers += [(lambda x, y: stroke(x, y, up, STROKE * k / 2), CORAL),
                   (lambda x, y: stroke(x, y, down, CUT * k / 2), None)]
    layers.append((lambda x, y: stroke(x, y, down, STROKE * k / 2), ink if on else GREY))
    return render(size, layers)


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
