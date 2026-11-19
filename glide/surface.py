"""Free-desk detector: where on the camera's view the hand can actually slide.

A desk camera rarely sees an empty desk. A bag, a keyboard or the desk's own edge (with the user's
lap beyond it) can cover a third of the view, and the hand can't go there, yet the workspace fit and
the edge-glide bands assumed the whole frame. This finds the free desk around the hand and returns
the largest rectangle in it, so the gain fits the room that exists and the glide edges sit where the
hand can reach them.

Method (all at about 160 px across, a few ms):
  1. Learn the desk's look (Lab median and spread) next to the hand, or where the hand last was.
  2. Desk-like = close in colour (chroma) and not far darker or brighter (shadows are allowed).
  3. Keep the connected part around the seed, smooth it, and fill small holes (crumbs, cables).
  4. Pick the rectangle that gives the hand the most room to slide, allowing 4% clutter per line.
"""
import math

import cv2
import numpy as np

SCALE_PX = 160            # long side of the working image


def _lab(frame):
    h, w = frame.shape[:2]
    s = SCALE_PX / max(h, w)
    small = cv2.resize(frame, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32), s


def colour_name(bgr):
    """A plain word for an obstacle's colour, for the message ("something blue")."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    hue, sat, val = hsv[0] * 2, hsv[1] / 255, hsv[2] / 255
    if val < 0.22:
        return "dark"
    if sat < 0.18:
        return "light" if val > 0.75 else "grey"
    for name, top in (("red", 15), ("orange", 40), ("yellow", 70), ("green", 165), ("blue", 260),
                      ("purple", 330), ("red", 361)):
        if hue < top:
            return name
    return "coloured"


def best_rect(mask, hand_wh, seed, tol=0.04):
    """Rectangle (x0, y0, x1, y1), end-exclusive, in mask px, that contains the seed and maximises
    the hand's sliding room (w - hand_w) * (h - hand_h). A line counts as free when at most `tol` of
    it is not desk, so a crumb or a cable doesn't cut the room in half."""
    H, W = mask.shape
    hw, hh = hand_wh
    sx, sy = int(np.clip(seed[0], 0, W - 1)), int(np.clip(seed[1], 0, H - 1))
    col = np.vstack([np.zeros((1, W)), np.cumsum(~mask, axis=0)])         # per-column prefix of non-desk
    y1s = np.arange(sy + 1, H + 1)
    best, rect = -1.0, None
    for y0 in range(0, sy + 1):
        hs = y1s - y0
        notok = (col[y1s] - col[y0]) > tol * hs[:, None]                  # (n, W) column blocked in [y0, y1)
        through = ~notok[:, sx]
        if not through.any():
            continue
        lhs = notok[:, :sx][:, ::-1]
        left = np.where(lhs.any(1), sx - lhs.argmax(1), 0)
        rhs = notok[:, sx + 1:]
        right = np.where(rhs.any(1), sx + 1 + rhs.argmax(1), W)
        score = np.maximum(right - left - hw, 1.0) * np.maximum(hs - hh, 1.0)
        score[~through | (hs < hh * 0.6)] = -1
        i = int(score.argmax())
        if score[i] > best:
            best, rect = float(score[i]), (int(left[i]), y0, int(right[i]), int(y1s[i]))
    return rect


def detect(frame, seed_px, hand_box=None, hand_hull=None):
    """Free desk in a BGR frame. seed_px: a point on the desk (the palm, or where it last was).
    hand_box: the hand's (w, h) in frame px. hand_hull: its landmarks in frame px when it is in view
    (they are masked out, and count as desk: the hand is lying on it).

    Returns {"rect": (x0, y0, x1, y1) frame px, "mask": bool array (working scale), "scale",
             "free": share of the view that is free desk, "blocked": {side: colour word}} or None."""
    lab, s = _lab(frame)
    H, W = lab.shape[:2]
    seed = np.asarray(seed_px, float) * s
    hand = np.zeros((H, W), np.uint8)
    if hand_hull is not None and len(hand_hull) >= 3:
        hull = cv2.convexHull(np.round(np.asarray(hand_hull, float) * s).astype(np.int32))
        cv2.fillConvexPoly(hand, hull, 1)
        hand = cv2.dilate(hand, np.ones((5, 5), np.uint8))
    # desk sample: a ring around the hand if it is there, else a disc at the seed
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - seed[0], yy - seed[1])
    reach = (max(hand_box) * s * 0.9) if hand_box is not None else 0.0
    sample = (dist < reach + 10) & (dist > reach * 0.5) & (hand == 0)
    if sample.sum() < 20:
        return None
    px = lab[sample]
    med = np.median(px, axis=0)
    chroma = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
    c_spread = np.median(np.hypot(px[:, 1] - med[1], px[:, 2] - med[2]))
    dl = lab[..., 0] - med[0]
    # shadows (the hand's own, the room's falloff) darken the desk without changing its colour.
    # OpenCV's L runs 0-255: -70 lets a corner 35% darker through, black keyboards are ~-150
    desk = (chroma < max(7.0, 4.0 * c_spread)) & (dl > -70) & (dl < 50)
    desk |= hand > 0
    m = desk.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl = cv2.connectedComponents(m)
    sx, sy = int(np.clip(seed[0], 0, W - 1)), int(np.clip(seed[1], 0, H - 1))
    k = lbl[sy, sx]
    if k == 0:
        return None
    m = (lbl == k).astype(np.uint8)
    # fill holes smaller than a quarter of a hand: crumbs, cables, a pen
    holes = cv2.bitwise_not(m * 255)
    nh, hl, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8))
    hand_area = (hand_box[0] * hand_box[1] * s * s) if hand_box is not None else 400.0
    for i in range(1, nh):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if not touches and area < hand_area / 4:
            m[hl == i] = 1
    mask = m.astype(bool)
    hw, hh = (np.asarray(hand_box, float) * s) if hand_box is not None else (W * 0.3, H * 0.3)
    r = best_rect(mask, (hw, hh), seed)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    blocked = {}
    gx, gy = 0.06 * W, 0.06 * H              # a sliver at the very edge of the view is not an obstacle
    for side, inside, band in (("left", x0 > gx, (slice(None), slice(max(0, x0 - 6), x0))),
                               ("right", x1 < W - gx, (slice(None), slice(x1, min(W, x1 + 6)))),
                               ("top", y0 > gy, (slice(max(0, y0 - 6), y0), slice(x0, x1))),
                               ("bottom", y1 < H - gy, (slice(y1, min(H, y1 + 6)), slice(x0, x1)))):
        if inside:
            region = lab[band][~mask[band]] if (~mask[band]).any() else lab[band].reshape(-1, 3)
            bgr = cv2.cvtColor(np.uint8([[np.median(region, axis=0)]]), cv2.COLOR_LAB2BGR)[0, 0]
            blocked[side] = colour_name(bgr)
    return {"rect": tuple(round(v / s) for v in (x0, y0, x1, y1)), "mask": mask, "scale": s,
            "free": float(mask.mean()), "blocked": blocked}


def same_room(a, b, tol=0.08):
    """Two detections agree: every side within tol of the frame size."""
    if a is None or b is None:
        return False
    span = max(a[2] - a[0], a[3] - a[1], 1)
    return all(abs(p - q) <= tol * span for p, q in zip(a, b))


def desk_side(R, side):
    """Image side ("top", ...) -> where it is for the user: "far", "near", "left" or "right".
    R rows are the desk's right and forward axes in image px."""
    n = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[side]
    right, fwd = np.asarray(R, float) @ n
    if abs(fwd) >= abs(right):
        return "far" if fwd > 0 else "near"
    return "right" if right > 0 else "left"


def room_report(R, rect, frame_wh, hand_box, mm_per_px, blocked, want_mm=(60.0, 100.0)):
    """What the free desk means for the hand, per desk axis (right, forward), in cm, plus how far
    to move each obstacle (not the near side: that is the desk's edge or the user) for want_mm of
    sliding. move_cm None = even out of view it wouldn't be enough."""
    R = np.abs(np.asarray(R, float))
    ext = (rect[2] - rect[0], rect[3] - rect[1])
    span = (frame_wh[0], frame_wh[1])
    free, slide, axis_of = [], [], []
    for k in range(2):
        i = int(R[k].argmax())                       # image axis this desk axis runs along
        axis_of.append(i)
        free.append(round(ext[i] * mm_per_px / 10, 1))
        slide.append(round(max(ext[i] - hand_box[i], 0) * mm_per_px / 10, 1))
    advice = []
    for side, colour in blocked:
        if side == "near":
            continue
        k = 1 if side == "far" else 0
        i = axis_of[k]
        need_px = hand_box[i] + want_mm[k] / mm_per_px - ext[i]
        if need_px <= 0:
            continue
        room_left = span[i] - ext[i]                 # most the free desk could grow on this axis
        advice.append({"side": side, "colour": colour,
                       "move_cm": round(need_px * mm_per_px / 10) if need_px <= room_left else None})
    return {"free_cm": free, "slide_cm": slide, "advice": advice}


def contains(outer, inner, tol):
    """outer covers inner, give or take tol px per side."""
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol and
            outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)
