"""Cursor smoothness: repeated camera pictures dropped, filter cap, resting gain, in-between motion."""
import time
import unittest

import numpy as np

from glide.camera import is_repeat
from glide.config import DEFAULTS
from glide.filters import OneEuro2D, gain_curve


class RepeatFrameTests(unittest.TestCase):
    def test_same_picture_is_a_repeat(self):
        a = np.random.default_rng(0).integers(0, 255, (45, 80, 3)).astype(np.int16)
        self.assertTrue(is_repeat(a, a.copy()))

    def test_sensor_noise_is_a_new_frame(self):
        rng = np.random.default_rng(1)
        a = rng.integers(20, 230, (45, 80, 3)).astype(np.int16)
        b = a + rng.integers(-1, 2, a.shape)            # a still scene: only sensor noise differs
        self.assertFalse(is_repeat(a, b))


class FilterCapTests(unittest.TestCase):
    def test_cutoff_capped_while_moving(self):
        # with fc_max the fast-moving point is smoothed (lags a step); without it, it passes through
        free, capped = OneEuro2D(0.4, 0.16, 1.0), OneEuro2D(0.4, 0.16, 1.0, fc_max=8.0)
        for k in range(30):
            x = (k * 10.0, 0.0)                         # 300 mm/s
            a, b = free(x, k / 30), capped(x, k / 30)
        self.assertLess(b[0], a[0] - 3.0)


class RestingGainTests(unittest.TestCase):
    def test_still_hand_gets_exactly_the_minimum_gain(self):
        g = dict(DEFAULTS["gain"], g_max=42.0)          # 3x workspace gain
        self.assertAlmostEqual(gain_curve(0.0, g), g["g_min"])
        plain = g["g_min"] + (g["g_max"] - g["g_min"]) / (1 + np.exp(-(10.0 - g["v_mid_mm_s"]) / g["width"]))
        self.assertLess(gain_curve(10.0, g) - g["g_min"], 0.6 * (plain - g["g_min"]))   # resting wobble

    def test_monotonic(self):
        g = DEFAULTS["gain"]
        v = [gain_curve(s, g) for s in range(0, 400, 5)]
        self.assertTrue(all(b >= a for a, b in zip(v, v[1:])))


class FakeSmooth:
    """SmoothOutput's motion logic without touching the real cursor."""

    @staticmethod
    def make(interval):
        from glide.output_win import SmoothOutput

        class Out(SmoothOutput):
            def __init__(self):
                self.pos = [0, 0]
                super().__init__(lambda: interval)

            def cursor(self):
                return tuple(self.pos)

            def _nudge(self, dx, dy):
                self.pos[0] += dx
                self.pos[1] += dy

        return Out()


class SmoothOutputTests(unittest.TestCase):
    def setUp(self):
        self.out = FakeSmooth.make(0.06)

    def tearDown(self):
        self.out.stop()

    def test_step_spread_over_the_frame_interval(self):
        self.out.move_by(60, -30)
        time.sleep(0.02)
        part = self.out.pos[0]
        self.assertGreater(part, 0)
        self.assertLess(part, 60)                       # not one jump
        time.sleep(0.12)
        self.assertEqual(self.out.pos, [60, -30])

    def test_flush_lands_motion_in_flight(self):
        self.out.move_by(100, 0)
        self.out.flush()                                # buttons and the wheel do this first
        self.assertEqual(self.out.pos, [100, 0])
        time.sleep(0.03)
        self.assertEqual(self.out.pos, [100, 0])

    def test_no_pixels_lost_or_gained(self):
        for k in range(40):
            self.out.move_by(3 if k % 2 else -1, 1)
            time.sleep(0.005)
        time.sleep(0.15)
        self.assertEqual(self.out.pos, [40, 40])


if __name__ == "__main__":
    unittest.main()
