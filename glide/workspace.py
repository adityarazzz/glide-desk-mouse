"""How far the hand can actually slide in view, and the gain that makes that cover the screen.

A desk camera sees a limited patch of desk and the hand itself fills much of it. Mouse gain tuned
for precision then runs out of hand travel before the cursor crosses the screen. Two remedies:
fit the transfer function to the measured workspace (here), and glide at the frame edges with
RubberEdge-style rate control (engine), so the edge of the view never ends a movement.
"""
import math

import numpy as np

from .filters import gain_curve


def travel_mm(R, frame_wh, bbox_wh, mm_per_px, edge_px=0.0):
    """Free slide distance along the desk (right, forward) axes before the hand's bounding box
    reaches the edge band, in mm. R rows are the desk axes as unit vectors in image px."""
    W, H = frame_wh
    ex, ey = (2 * edge_px, 2 * edge_px) if np.isscalar(edge_px) else edge_px   # band px per image axis
    ux = max(W - bbox_wh[0] - ex, 1.0)
    uy = max(H - bbox_wh[1] - ey, 1.0)
    # an image axis only limits a desk axis that really runs along it: a camera turned 2.6 degrees
    # must not let a cramped forward room shrink the sideways room too
    lim = lambda a: min(u / abs(c) for u, c in ((ux, a[0]), (uy, a[1])) if abs(c) > 0.2)
    return [float(lim(a) * mm_per_px) for a in R]


def scaled_gain(gain_cfg, s):
    """Raise only the fast end and move the knee up: reach grows, the slow precise end stays.

    Measured on a recorded session at s = 2.74: static jitter 0.60 px (NFR-02 <= 1 px), against
    0.93 px when scaling only g_max and 1.31 px when also scaling g_min by sqrt(s).
    """
    return dict(gain_cfg, g_max=gain_cfg["g_max"] * s, v_mid_mm_s=gain_cfg["v_mid_mm_s"] + 20.0 * (s - 1))


AXIS_CAP = 1.3


def fit_axis_gain(travel, screen_wh, cap=AXIS_CAP):
    """Extra gain (>= 1) for the axis that has too little room for its share of the screen.

    A camera low over the desk can leave far less room across than forward (measured 8 x 21 cm)
    while the screen is wide. The engine applies it only to fast strokes, so slow aiming and the
    overall gain are unchanged. It is capped at 1.3, so a fast 45-degree stroke still lands
    steeper than 37 degrees. Beyond that, raising the camera is the real fix.
    """
    nx = screen_wh[0] / max(travel[0], 10.0)
    ny = screen_wh[1] / max(travel[1], 10.0)
    k = float(np.clip(math.sqrt(max(nx, ny) / min(nx, ny)), 1.0, cap))
    return [k, 1.0] if nx >= ny else [1.0, k]


def fit_gain_scale(gain_cfg, travel, screen_wh, cover=0.8, v_nat=200.0, max_scale=3.0, axis=(1.0, 1.0)):
    """Smallest scale (>= 1) at which a natural-speed stroke over `cover` of the workspace crosses
    the whole screen on both axes."""
    need = max(screen_wh[0] / (cover * max(travel[0], 10.0) * axis[0]),
               screen_wh[1] / (cover * max(travel[1], 10.0) * axis[1]))
    px_scale = screen_wh[0] / gain_cfg["reference_width"]
    g = lambda s: gain_curve(v_nat, scaled_gain(gain_cfg, s)) * px_scale
    if g(1.0) >= need:
        return 1.0
    lo, hi = 1.0, max_scale
    if g(hi) < need:
        return max_scale
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) < need else (lo, mid)
    return float(hi)


def camera_advice(frame_mm, hand_mm, travel, want=(150.0, 100.0), hfov_deg=70.0):
    """Rough 'raise the camera by N cm' hint: the view grows linearly with camera height.
    hand_mm: what the hand (and the edge bands, when travel excludes them) takes of the view."""
    if travel[0] >= want[0] and travel[1] >= want[1]:
        return None
    k = max((want[0] + hand_mm[0]) / frame_mm[0], (want[1] + hand_mm[1]) / frame_mm[1])
    height_mm = max(frame_mm) / (2 * math.tan(math.radians(hfov_deg / 2)))
    return max(1, round((k - 1) * height_mm / 10))


def edge_band_px(edge_cfg, frame_wh, bounds=None):
    """Width of the edge-glide band, image px: a share of the frame, but never more than a quarter
    of the free desk, or a small free area would be all band."""
    if not edge_cfg["enabled"]:
        return 0.0
    z = edge_cfg["zone"] * frame_wh[1]
    if bounds is not None:
        z = min(z, 0.25 * min(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    return z


def fit_workspace(R, frame_wh, box, mm_per_px, gain_cfg, edge_cfg, screen_wh, bounds=None):
    """Everything a profile needs about the room to move: travel, gain, axis shape, camera advice.
    bounds: the free-desk rectangle (surface.detect) when something covers part of the view."""
    full_edge = edge_band_px(edge_cfg, frame_wh)
    full = travel_mm(R, frame_wh, box, mm_per_px, full_edge)
    if bounds is not None:
        room = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        # bands only where the free desk meets the view's own edge: a wall (the bag, the desk's edge)
        # has none, the hand glides only once past it (engine._edge). Counting bands at walls too had
        # the fit see 1.1 cm of forward room where 7.1 cm was free, and max out the vertical gain.
        z = edge_band_px(edge_cfg, frame_wh)
        W, H = frame_wh
        walls = [bounds[0] > 2, bounds[1] > 2, bounds[2] < W - 2, bounds[3] < H - 2]   # left top right bottom
        bands = (z * ((not walls[0]) + (not walls[2])), z * ((not walls[1]) + (not walls[3])))
        travel = travel_mm(R, room, box, mm_per_px, bands)
    else:
        travel = full
    frame_mm = travel_mm(R, frame_wh, (0, 0), mm_per_px)
    taken = [f - t for f, t in zip(frame_mm, full)]       # the hand itself plus the two edge bands
    axis = fit_axis_gain(travel, screen_wh)
    return {"travel_mm": travel, "axis_gain": axis,
            "gain_scale": fit_gain_scale(gain_cfg, travel, screen_wh),
            # raising the camera only helps once the desk is clear, so advise on the clear view
            "raise_cm": camera_advice(frame_mm, taken, full)}


def hand_box(obs_list):
    """Median width/height of the hand's landmark bounding box, image px."""
    spans = np.array([np.ptp(o.img, axis=0) for o in obs_list])
    return [float(v) for v in np.median(spans, axis=0)]
