"""AutoGain: the speed is learned from where clicks land, for a simulated user aiming at targets."""
import copy
import math
import unittest

import numpy as np

from glide.autogain import BINS, AutoGain
from glide.config import DEFAULTS
from glide.engine import Engine
from glide.features import Obs
from glide.replay import VirtualOut

PROFILE = {"R": [[0.0, 1.0], [1.0, 0.0]], "px_per_m": 1000.0, "chirality": 1}
DT = 1 / 30
OPEN8, OPEN12 = 0.9, 1.2


def obs(right, flex12=OPEN12):
    return Obs(present=True, palm_px=np.array([0.0, right]), px_per_m=1000.0, flex8=OPEN8, flex12=flex12,
               fist=False, chirality=1, score=0.99, hand_m=0.08)


class User:
    """Aims a fast stroke of h mm (planned for the right gain), then creeps onto the target and pinches."""

    def __init__(self, mult, learn=True):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["autogain"]["enabled"] = True
        self.out = VirtualOut()
        self.eng = Engine(cfg, PROFILE, self.out)
        self.eng.autogain.mult[:] = mult
        self.learn = learn
        self.t, self.x = 0.0, 0.0
        for _ in range(30):
            self.step()

    def step(self, flex12=OPEN12):
        self.t += DT
        self.out.t = self.t
        self.eng.update(obs(self.x, flex12), self.t)

    def stroke(self, h, frames=14):
        x0 = self.x
        for i in range(1, frames + 1):                   # minimum-jerk profile
            s = i / frames
            self.x = x0 + h * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
            self.step()
        for _ in range(4):
            self.step()

    def trial(self, h, target, px_per_mm):
        before = self.out.pos[0]
        self.stroke(h)
        primary = self.out.pos[0]
        for _ in range(120):                             # corrections onto the target
            err = target - self.out.pos[0]
            if abs(err) <= 4:
                break
            if abs(err) > 60:                            # far off: another quick aimed movement
                self.stroke(0.93 * err / px_per_mm)
            else:                                        # close: creep
                self.x += math.copysign(min(1.2, abs(err) / 6), err)
                self.step()
        for _ in range(3):
            self.step()
        self.step(OPEN12 * 0.85)                         # quick pinch = click
        for _ in range(3):
            self.step(OPEN12 * 0.4)
        for _ in range(8):
            self.step()
        return primary - before


def planned_stroke(distance_px):
    """The stroke that, with the right gain (multiplier 1), covers 93% of the distance."""
    lo, hi = 2.0, 200.0
    for _ in range(16):
        mid = (lo + hi) / 2
        u = User(1.0)
        moved = u.out.pos[0]
        u.stroke(mid)
        if u.out.pos[0] - moved < 0.93 * distance_px:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class AutoGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = planned_stroke(400)

    def learn_from(self, start_mult, trials=30):
        u = User(start_mult)
        for k in range(trials):
            sign = 1 if k % 2 == 0 else -1
            target = u.out.pos[0] + sign * 400
            u.trial(sign * self.h, target, 0.93 * 400 / self.h)
        return u.eng.autogain

    def effective(self, ag):
        """The multiplier where this user's strokes run (travel-weighted, as the learner sees it)."""
        u = User(ag.mult.copy())
        before = u.out.pos[0]
        u.stroke(self.h)
        return (u.out.pos[0] - before) / (0.93 * 400)    # 1.0 = the stroke covers 93% again

    def test_too_fast_learns_to_slow_down(self):
        ag = self.learn_from(1.4)
        self.assertGreater(ag.n, 20)                     # clicks it learned from
        self.assertAlmostEqual(self.effective(ag), 1.0, delta=0.12)   # started at ~1.4

    def test_too_slow_learns_to_speed_up(self):
        ag = self.learn_from(0.7)
        self.assertGreater(ag.n, 20)
        self.assertAlmostEqual(self.effective(ag), 1.0, delta=0.15)   # started at ~0.7

    def test_a_stroke_with_glide_or_lift_teaches_nothing(self):
        ag = AutoGain(DEFAULTS["autogain"])
        ag.frame(0.0, (0, 0), 100.0, clean=True)
        ag.frame(0.1, (200, 0), 5.0, clean=False)        # glided / lifted on the way
        self.assertIsNone(ag.click((400, 0)))
        self.assertEqual(ag.n, 0)

    def test_bins_interpolate_smoothly(self):
        ag = AutoGain(DEFAULTS["autogain"], {"mult": [1.0, 1.2, 1.0, 1.0, 1.0], "n": 3})
        self.assertAlmostEqual(ag.factor(BINS[1]), 1.2)
        self.assertTrue(1.0 < ag.factor((BINS[1] + BINS[2]) / 2) < 1.2)


if __name__ == "__main__":
    unittest.main()
