import copy
import unittest

import numpy as np

from glide.config import DEFAULTS
from glide.engine import DRAG, Engine
from glide.features import Obs
from glide.filters import gain_curve
from glide.replay import VirtualOut
from glide.workspace import camera_advice, fit_gain_scale, scaled_gain, travel_mm

DT = 1 / 30
FRAME = (640, 360)
PROFILE = {"R": [[0.0, 1.0], [1.0, 0.0]], "px_per_m": 1000.0, "chirality": 1}   # image x = forward
HALF = np.array([100.0, 60.0])     # hand bounding box half-size, px


def hand(fwd, right, flex12=1.2):
    """Synthetic hand: anchor landmarks (0, 1, 17) at the point, the rest spread over the box."""
    c = np.array([fwd, right], float)
    img = np.tile(c, (21, 1))
    img[[4, 8, 12, 16, 20]] = c + HALF * [1, 0]            # fingertips forward
    img[[2, 5]] = c + HALF * [0, -1]
    img[[13, 18]] = c + HALF * [0, 1]
    img[[3]] = c - HALF * [1, 0]
    return Obs(present=True, img=img, palm_px=c, px_per_m=1000.0, flex8=0.9, flex12=flex12, chirality=1,
               score=0.99, hand_m=0.08, frame_wh=FRAME)


class Rig:
    def __init__(self):
        self.out = VirtualOut(1920, 8000)               # tall, so upward glides aren't clamped
        self.eng = Engine(copy.deepcopy(DEFAULTS), PROFILE, self.out)
        self.t = 0.0

    def step(self, o):
        self.t += DT
        self.out.t = self.t
        self.eng.update(o, self.t)


class EdgeGlideTests(unittest.TestCase):
    def slide_to_edge(self, r, **kw):
        for i in range(60):
            r.step(hand(320 + i * 4.0, 180, **kw))         # toward the forward (image +x) edge
        return 320 + 59 * 4.0

    def test_holding_at_edge_keeps_cursor_gliding(self):
        r = Rig()
        for i in range(25):
            r.step(hand(470 + i * 2.5, 180))               # gentle slide into the forward band
        x = 470 + 24 * 2.5
        y_at_edge = r.out.pos[1]
        for _ in range(30):
            r.step(hand(x, 180))                           # hand held still inside the band
        self.assertGreater(r.eng.edge_depth, 0)
        self.assertLess(r.out.pos[1], y_at_edge - 150)     # cursor kept moving up for 1 s

    def test_resting_in_band_without_pushing_does_not_drift(self):
        r = Rig()
        for _ in range(40):
            r.step(hand(560, 180))                         # hand put down already near the edge
        before = r.out.cursor()
        for _ in range(30):
            r.step(hand(560, 180))
        self.assertEqual(r.out.cursor(), before)

    def test_pulling_back_stops_glide(self):
        r = Rig()
        x = self.slide_to_edge(r)
        for i in range(10):
            r.step(hand(x - i * 5.0, 180))                 # pull back inward
        self.assertFalse(r.eng.edge_armed)
        before = r.out.cursor()
        for _ in range(15):
            r.step(hand(x - 45.0, 180))
        # only the prediction lead settles back (a few px); an active glide would move ~300 px
        self.assertLess(before[1] - r.out.pos[1], 25)

    def test_glide_carries_a_drag(self):
        r = Rig()
        for _ in range(30):                                # resting on the desk, settled
            r.step(hand(320, 180))
        r.step(hand(320, 180, flex12=1.0))
        for _ in range(3):
            r.step(hand(320, 180, flex12=0.45))            # pinch
        for i in range(60):
            r.step(hand(320 + i * 4.0, 180, flex12=0.45))
        self.assertEqual(r.eng.state, DRAG)
        y = r.out.pos[1]
        for _ in range(20):
            r.step(hand(556, 180, flex12=0.45))
        self.assertIn("left", r.out.held)
        self.assertLess(r.out.pos[1], y - 80)


class WorkspaceTests(unittest.TestCase):
    def test_travel(self):
        R = [[0.0, 1.0], [1.0, 0.0]]                       # right = image y, forward = image x
        right, fwd = travel_mm(R, (640, 360), (200, 120), 0.7)
        self.assertAlmostEqual(right, (360 - 120) * 0.7, places=3)
        self.assertAlmostEqual(fwd, (640 - 200) * 0.7, places=3)

    def test_small_workspace_gets_more_reach(self):
        g = DEFAULTS["gain"]
        roomy = fit_gain_scale(g, [300, 200], (1920, 1080))
        tight = fit_gain_scale(g, [90, 70], (1920, 1080))
        self.assertEqual(roomy, 1.0)
        self.assertGreater(tight, 1.5)
        s = fit_gain_scale(g, [120, 90], (1920, 1080))
        self.assertGreaterEqual(gain_curve(200, scaled_gain(g, s)) * 0.8 * 120, 1919)

    def test_scaling_keeps_precision_end(self):
        g = DEFAULTS["gain"]
        self.assertEqual(scaled_gain(g, 2.5)["g_min"], g["g_min"])

    def test_camera_advice(self):
        self.assertIsNone(camera_advice([400, 300], [190, 100], [200, 150]))
        self.assertGreaterEqual(camera_advice([300, 220], [190, 100], [80, 90]), 1)



class AxisGainTests(unittest.TestCase):
    def test_narrow_room_on_a_wide_screen_gets_more_gain_across(self):
        from glide.workspace import fit_axis_gain
        fx, fy = fit_axis_gain([76, 213], (1920, 1080))
        self.assertAlmostEqual(fx, 1.3)                 # capped: diagonals stay recognisable
        self.assertEqual(fy, 1.0)                       # the roomy axis is left alone

    def test_room_shaped_like_the_screen_stays_isotropic(self):
        from glide.workspace import fit_axis_gain
        fx, fy = fit_axis_gain([160, 90], (1920, 1080))
        self.assertAlmostEqual(fx, 1.0, places=2)

if __name__ == "__main__":
    unittest.main()
