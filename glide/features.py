"""Per-frame hand geometry, expressed in units that don't depend on camera height or angle."""
import math
from dataclasses import dataclass

import numpy as np

WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
PINKY_MCP = 17
PALM = [0, 5, 9, 13, 17]
# palm bones stay rigid and roughly parallel to the desk, so they give a stable scale
PALM_SEGMENTS = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (5, 17)]
FINGERS = [(5, 8), (9, 12), (13, 16), (17, 20)]  # (mcp, tip)
GESTURE_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP]
NO_PINCH = 9.0


@dataclass
class Obs:
    present: bool
    img: np.ndarray = None        # (21, 2) image pixels
    palm_px: np.ndarray = None    # palm centre, image pixels
    px_per_m: float = 0.0         # image scale at the hand; rises when the hand lifts toward the camera
    pinch_idx: float = NO_PINCH   # thumb-index tip gap / hand length
    pinch_mid: float = NO_PINCH   # thumb-middle tip gap / hand length
    fist: bool = False
    elev_deg: float = 0.0         # index finger elevation from MediaPipe world depth
    chirality: int = 0            # +1 / -1: which way the palm triangle winds in the image
    score: float = 0.0
    hand_m: float = 0.09          # wrist -> middle knuckle, metres (plausibility check)
    frame_wh: tuple = None        # (width, height) of the source frame
    # thumb-tip gap to the index / middle tip in the image, per hand length. From an overhead
    # camera these dip sharply on every pinch or tap, while the 3-D gaps barely move.
    flex8: float = NO_PINCH
    flex12: float = NO_PINCH
    # sine of the palm-triangle winding: ~0.73 flat palm-down, -> 0 as the hand tilts onto its
    # side, negative for the mirrored (other) hand. Graded, so a tilt is not mistaken for a flip.
    palm_sin: float = 1.0
    yaw: float = None             # heading of wrist -> middle knuckle in the image, radians
    ext: tuple = (2.0, 2.0, 2.0, 2.0)   # image tip-to-wrist / knuckle-to-wrist, index..pinky
    two_finger: bool = False      # index + middle extended, ring + pinky curled (scroll pose)
    palm_size: float = 0.0        # longest wrist-to-knuckle bone in the image, px (lift signal)

    def anchor(self, ids):
        return self.img[ids].mean(axis=0) if self.img is not None else self.palm_px

    def sensor(self, k=1.0, side=0.0):
        """Where a mouse would sit under this hand: k along the wrist -> knuckle-line axis (1 = on the
        knuckle line), side = sideways, as a share of that length (+ toward the index finger).
        Swivelling at the wrist moves it the way it moves a mouse."""
        if self.img is None:
            return self.palm_px
        w, line = self.img[0], self.img[[5, 9, 13, 17]].mean(axis=0)
        ax = line - w
        across = self.img[5] - self.img[17]
        perp = np.array([-ax[1], ax[0]])
        perp *= 1.0 if perp @ across >= 0 else -1.0     # point toward the index side for either hand
        return w + k * ax + side * perp

    def palm_in(self, margin):
        """Palm centre inside the frame minus a margin (fraction of each side)."""
        if self.frame_wh is None:
            return True
        w, h = self.frame_wh
        x, y = self.palm_px
        return margin * w <= x <= (1 - margin) * w and margin * h <= y <= (1 - margin) * h

    def tips_in(self, margin):
        if self.frame_wh is None or self.img is None:
            return True
        w, h = self.frame_wh
        t = self.img[GESTURE_TIPS]
        return bool(np.all((t[:, 0] >= margin * w) & (t[:, 0] <= (1 - margin) * w)
                           & (t[:, 1] >= margin * h) & (t[:, 1] <= (1 - margin) * h)))


def _n(v):
    return float(np.linalg.norm(v))


def measure(img, world, score=1.0, frame_wh=None):
    """img: (21, 2) pixels; world: (21, 3) metres (MediaPipe world landmarks)."""
    img, world = np.asarray(img, dtype=float), np.asarray(world, dtype=float)
    ratios = [_n(img[a] - img[b]) / _n(world[a] - world[b])
              for a, b in PALM_SEGMENTS if _n(world[a] - world[b]) > 0.01]
    s = float(np.median(ratios)) if ratios else 0.0
    hand = max(_n(world[WRIST] - world[MIDDLE_MCP]), 1e-3)

    def pinch(tip):
        gap_world = _n(world[THUMB_TIP] - world[tip])
        gap_img = _n(img[THUMB_TIP] - img[tip]) / s if s else gap_world
        # both views must agree the tips are close: a thumb hidden under the finger
        # looks closed in the image but not in depth
        return max(gap_world, gap_img) / hand

    curls = [_n(world[tip] - world[WRIST]) / max(_n(world[mcp] - world[WRIST]), 1e-3)
             for mcp, tip in FINGERS]
    u = world[INDEX_TIP] - world[INDEX_MCP]
    a, b = img[INDEX_MCP] - img[WRIST], img[PINKY_MCP] - img[WRIST]
    hand_img = max(_n(img[WRIST] - img[MIDDLE_MCP]), 1e-3)
    heading = img[MIDDLE_MCP] - img[WRIST]
    # extension seen from above: ~2.0-2.2 for a straight finger, ~1.1-1.25 curled (recorded)
    ext = tuple(_n(img[t] - img[WRIST]) / max(_n(img[m] - img[WRIST]), 1e-3) for m, t in FINGERS)
    return Obs(
        palm_size=max(_n(img[WRIST] - img[m]) for m in (5, 9, 13, 17)),
        ext=ext,
        two_finger=min(ext[0], ext[1]) > 1.7 and max(ext[2], ext[3]) < 1.5 and
                   min(ext[0], ext[1]) - max(ext[2], ext[3]) > 0.4,
        palm_sin=float((a[0] * b[1] - a[1] * b[0]) / max(_n(a) * _n(b), 1e-6)),
        yaw=math.atan2(heading[1], heading[0]),
        flex8=_n(img[THUMB_TIP] - img[INDEX_TIP]) / hand_img,
        flex12=_n(img[THUMB_TIP] - img[MIDDLE_TIP]) / hand_img,
        present=True,
        img=img,
        palm_px=img[PALM].mean(axis=0),
        px_per_m=s,
        pinch_idx=pinch(INDEX_TIP),
        pinch_mid=pinch(MIDDLE_TIP),
        fist=all(c < 1.0 for c in curls),
        elev_deg=math.degrees(math.atan2(-u[2], math.hypot(u[0], u[1]))),
        chirality=1 if a[0] * b[1] - a[1] * b[0] > 0 else -1,
        score=score,
        hand_m=hand,
        frame_wh=tuple(frame_wh) if frame_wh is not None else None,
    )
