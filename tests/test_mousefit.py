"""Learning the user's own "mouse sensor" point from a session with their real mouse."""
import math
import unittest

import numpy as np

from glide.mousefit import fit, sensor_from

R = [[0.0, 1.0], [1.0, 0.0]]          # image x = forward, image y = right
MM = 0.615


def hand(theta, shift):
    known = {0: (0, 0), 1: (25, 20), 5: (85, 25), 9: (90, 0), 13: (85, -20), 17: (75, -38)}
    pts = np.array([known.get(i, (110 + i, (i % 4 - 1.5) * 18)) for i in range(21)], float)
    c, s = math.cos(theta), math.sin(theta)
    return pts @ np.array([[c, s], [-s, c]]) + [200, 180] + shift


def session(k_true, side_true, seconds=40, fps=17, lag_frames=1, seed=0):
    """Swivels and slides mixed like real mouse use; the cursor follows the true sensor point with a
    speed-dependent gain (Windows acceleration) and arrives lag_frames late."""
    rng = np.random.default_rng(seed)
    n = seconds * fps
    t = np.arange(n) / fps
    theta = 0.35 * np.sin(2 * np.pi * 0.31 * t) * np.sin(2 * np.pi * 0.07 * t + 1)
    slide = np.column_stack([25 * np.sin(2 * np.pi * 0.13 * t + 0.5), 30 * np.sin(2 * np.pi * 0.19 * t)])
    imgs = [hand(th, sh) + rng.normal(0, 0.4, (21, 2)) for th, sh in zip(theta, slide)]
    clean = [hand(th, sh) for th, sh in zip(theta, slide)]
    line = lambda h: h[[5, 9, 13, 17]].mean(0)
    pts = []
    for h in clean:
        ax = line(h) - h[0]
        perp = np.array([-ax[1], ax[0]])
        perp *= 1 if perp @ (h[5] - h[17]) >= 0 else -1
        pts.append(h[0] + k_true * ax + side_true * perp)
    desk = np.array(pts) @ np.array(R).T * MM
    v = np.gradient(desk, t, axis=0)
    speed = np.hypot(*v.T)
    gain = 4 + 8 * speed / (speed + 80)                          # px per mm, rises with speed
    step = np.column_stack([v[:, 0], -v[:, 1]]) * gain[:, None] / fps
    cur = np.cumsum(step, axis=0) + [960, 540]
    cur = np.vstack([np.repeat(cur[:1], lag_frames, 0), cur[:-lag_frames]]) if lag_frames else cur
    return [{"t": float(a), "label": "mouse_free", "img": im.tolist(), "cur": c.tolist()}
            for a, im, c in zip(t, imgs, cur)]


class MouseFitTests(unittest.TestCase):
    def test_finds_the_sensor_point(self):
        res = fit(session(1.1, 0.1), R, MM)
        self.assertAlmostEqual(res["k"], 1.1, delta=0.2)
        self.assertAlmostEqual(res["side"], 0.1, delta=0.2)
        self.assertLess(res["err"], 0.5 * res["err_now"])       # explains the cursor far better

    def test_a_palm_mover_stays_near_the_palm(self):
        res = fit(session(0.3, 0.0, seed=1), R, MM)
        self.assertLess(res["k"], 0.6)
        self.assertEqual(sensor_from(res)[0], 0.4)              # clamped where landmarks pin it down

    def test_too_little_movement(self):
        self.assertIsNone(fit(session(1.0, 0.0)[:40], R, MM))


if __name__ == "__main__":
    unittest.main()
