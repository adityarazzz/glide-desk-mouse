"""Learn from your real mouse: which point on your hand moves the way your mouse does.

In the "Show Glide your mouse" session you use your own mouse under the camera while Glide records
the hand's landmarks and the real cursor (Glide doesn't move the cursor while recording). A mouse is
mostly moved by swivelling the wrist and nudging with the fingers, so the cursor follows a point some
way in front of the wrist. For every candidate point, wrist + k * (knuckle line - wrist) + side * (a
sideways step), its desk velocity is compared with the cursor's in short windows, each window with its
own gain, because Windows' pointer acceleration changes the gain with speed. The point that explains
the cursor best is the user's own "mouse sensor"; the engine then follows that point while moving.
"""
import json
from pathlib import Path

import numpy as np

K_GRID = np.round(np.arange(0.0, 1.61, 0.1), 2)
SIDE_GRID = np.round(np.arange(-0.5, 0.51, 0.1), 2)
APPLY_K = (0.4, 1.4)          # beyond this the point is far from any landmark: noisy
WINDOW_S, MIN_PX_S = 0.6, 150.0


def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "img" in r and "cur" in r and str(r.get("label", "")).startswith("mouse"):
            rows.append(r)
    return rows


def _smooth(a, n=3):
    k = np.ones(n) / n
    return np.column_stack([np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])])


def _points(img, k, side):
    w, line = img[:, 0], img[:, [5, 9, 13, 17]].mean(axis=1)
    ax = line - w
    across = img[:, 5] - img[:, 17]
    perp = np.stack([-ax[:, 1], ax[:, 0]], axis=1)
    perp *= np.where((perp * across).sum(axis=1) >= 0, 1.0, -1.0)[:, None]
    return w + k * ax + side * perp


def fit(rows, R, mm_per_px):
    """{"k", "side", "err", "err_now", "windows", "gain_px_mm", "lag_ms"} or None if too little."""
    if len(rows) < 60:
        return None
    t = np.array([r["t"] for r in rows])
    img = np.array([r["img"] for r in rows], float)[:, :, :2]
    cur = _smooth(np.array([r["cur"] for r in rows], float))
    R = np.asarray(R, float)

    def desk_vel(p):
        mm = _smooth(p) @ R.T * mm_per_px
        v = np.gradient(mm, t, axis=0)
        return np.column_stack([v[:, 0], -v[:, 1]])             # forward on the desk = up on screen

    vc = np.gradient(cur, t, axis=0)
    # the cursor is read when the frame is processed, the hand when it was captured: find the lag
    ref = np.hypot(*desk_vel(_points(img, 1.0, 0.0)).T)
    sc = np.hypot(*vc.T)
    dt = float(np.median(np.diff(t)))
    lags = range(0, max(1, int(0.25 / dt)) + 1)
    lag = max(lags, key=lambda L: np.corrcoef(ref[:len(ref) - L], sc[L:])[0, 1] if L < len(ref) - 5 else -1)
    if lag:
        vc, img, t = vc[lag:], img[:-lag], t[:-lag]

    n = max(4, int(round(WINDOW_S / dt)))
    starts = [s for s in range(0, len(t) - n, n // 2)
              if np.median(np.hypot(*vc[s:s + n].T)) > MIN_PX_S and t[s + n - 1] - t[s] < 2 * WINDOW_S]
    if len(starts) < 8:
        return None

    def score(p):
        vp = desk_vel(p)
        errs, gains = [], []
        for s in starts:
            a, b = vp[s:s + n], vc[s:s + n]
            g = float((a * b).sum() / max((a * a).sum(), 1e-9))
            errs.append(((b - g * a) ** 2).sum() / max((b * b).sum(), 1e-9))
            gains.append(g)
        return float(np.mean(errs)), float(np.median(gains))

    best = None
    for k in K_GRID:
        for side in SIDE_GRID:
            e, g = score(_points(img, k, side))
            if best is None or e < best[0]:
                best = (e, g, float(k), float(side))
    e_now, _ = score(img[:, [0, 1, 17]].mean(axis=1))
    return {"k": best[2], "side": best[3], "err": round(best[0], 3), "err_now": round(e_now, 3),
            "windows": len(starts), "gain_px_mm": round(best[1], 1), "lag_ms": round(lag * dt * 1000)}


def sensor_from(result):
    """The point to use, clamped to where landmarks still pin it down."""
    return [float(np.clip(result["k"], *APPLY_K)), float(np.clip(result["side"], -0.4, 0.4))]
