"""Readable text for the coached screens: Segoe UI through Pillow (OpenCV's Hershey font is hard
to read at a glance), batched so the frame is converted once per draw."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path("C:/Windows/Fonts")
BG = (28, 28, 30)
PANEL = (40, 40, 44)
WHITE, DIM, GREY = (240, 240, 240), (170, 170, 175), (110, 110, 115)
GREEN, AMBER, RED, CYAN, BLUE = (90, 210, 110), (0, 180, 255), (80, 80, 235), (255, 220, 0), (255, 160, 60)


@lru_cache(maxsize=32)
def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        if (FONTS / name).exists():
            return ImageFont.truetype(str(FONTS / name), size)
    return ImageFont.load_default(size=size)


class Typeset:
    """Collect text, then render it all onto a BGR image in one pass."""

    def __init__(self):
        self.items = []

    def add(self, s, xy, size=18, color=WHITE, bold=False, anchor="la"):
        """anchor: Pillow anchors, e.g. 'la' left-top, 'mm' centre, 'ra' right-top."""
        self.items.append((s, xy, size, color, bold, anchor))

    def width(self, s, size=18, bold=False):
        return font(size, bold).getlength(s)

    def wrap(self, s, max_w, size=18, bold=False):
        lines, cur = [], ""
        for word in s.split():
            nxt = (cur + " " + word).strip()
            if cur and self.width(nxt, size, bold) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = nxt
        return lines + ([cur] if cur else [])

    def render(self, img):
        if not self.items:
            return img
        pil = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        for s, xy, size, (b, g, r), bold, anchor in self.items:
            d.text(xy, s, font=font(size, bold), fill=(r, g, b), anchor=anchor)
        img[:] = np.asarray(pil)[:, :, ::-1]
        self.items = []
        return img


def canvas(w, h):
    return np.full((h, w, 3), BG, np.uint8)


def panel(img, x, y, w, h, color=PANEL, r=10):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def fit(img, frame, x, y, w, h):
    """Paste frame scaled to fit inside the box, centred. Returns (x, y, scale) of the paste."""
    s = min(w / frame.shape[1], h / frame.shape[0])
    fw, fh = int(frame.shape[1] * s), int(frame.shape[0] * s)
    ox, oy = x + (w - fw) // 2, y + (h - fh) // 2
    img[oy:oy + fh, ox:ox + fw] = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    return ox, oy, s


def ring(img, c, r, frac, color, thick=6, back=GREY):
    cv2.circle(img, c, r, back, 2, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(img, c, (r, r), -90, 0, 360 * min(frac, 1.0), color, thick, cv2.LINE_AA)


def check(img, x, y, ok, size=14):
    if ok:
        cv2.line(img, (x, y), (x + size // 2, y + size // 2), GREEN, 3, cv2.LINE_AA)
        cv2.line(img, (x + size // 2, y + size // 2), (x + size + 4, y - size // 2), GREEN, 3, cv2.LINE_AA)
    else:
        cv2.line(img, (x, y - size // 2), (x + size, y + size // 2), AMBER, 3, cv2.LINE_AA)
        cv2.line(img, (x + size, y - size // 2), (x, y + size // 2), AMBER, 3, cv2.LINE_AA)
