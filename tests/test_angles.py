import copy
import math
import unittest

import numpy as np

from glide.config import DEFAULTS
from glide.engine import SCROLL, Engine
from glide.features import Obs
from glide.replay import VirtualOut

DT = 1 / 30
# real orientation from this desk: fingers point image +x, the user's right is image +y
PROFILE = {"R": [[0.0, 1.0], [1.0, 0.0]], "px_per_m": 1000.0, "chirality": 1}


def obs(fwd=300.0, right=180.0, yaw_deg=0.0, palm_sin=0.73, fist=False, two=False):
    return Obs(present=True, palm_px=np.array([fwd, right]), px_per_m=1000.0, flex8=0.9, flex12=1.2,
               chirality=1 if palm_sin >= 0 else -1, score=0.99, hand_m=0.08, fist=fist, two_finger=two,
               palm_sin=palm_sin, yaw=math.radians(yaw_deg))


class Rig:
    def __init__(self, **over):
        cfg = copy.deepcopy(DEFAULTS)
        for k, v in over.items():
            cfg[k].update(v)
        self.out = VirtualOut(8000, 8000)
        self.out.screen_width = lambda: 1920
        self.eng = Engine(cfg, PROFILE, self.out)
        self.t = 0.0

    def run(self, n, **kw):
        for _ in range(n):
            self.t += DT
            self.out.t = self.t
            self.eng.update(obs(**kw), self.t)


class MouseFrameTests(unittest.TestCase):
    """A mouse's sensor turns with the hand: a sweep pivoting at the elbow runs the cursor straight."""

    def sweep(self, axes, pivot_mm=300.0, deg=30.0, n=30):
        r = Rig(pointer={"axes": axes})
        r.run(30, fwd=300.0, right=180.0)
        x0, y0 = r.out.cursor()
        ys = []
        for i in range(n + 1):                         # the forearm swings about the elbow
            a = math.radians(-deg / 2 + deg * i / n)
            fwd = 300.0 - pivot_mm + pivot_mm * math.cos(a)
            right = 180.0 + pivot_mm * math.sin(a)
            r.run(1, fwd=fwd, right=right, yaw_deg=math.degrees(a))
            ys.append(r.out.pos[1])
        dx = r.out.pos[0] - x0
        bow = max(ys) - min(ys)                        # how far the path bends up or down
        return dx, bow

    def test_elbow_sweep_runs_straight_like_a_mouse(self):
        dx, bow = self.sweep("mouse")
        self.assertGreater(dx, 200)
        self.assertLess(bow, 0.08 * dx)

    def test_fixed_axes_bend_the_same_sweep(self):
        dx_m, bow_m = self.sweep("mouse")
        dx_f, bow_f = self.sweep("fixed")
        self.assertGreater(bow_f, 2 * bow_m)


class AdaptiveAxesTests(unittest.TestCase):
    def stroke_along_fingers(self, axes):
        r = Rig(pointer={"axes": axes})
        yaw = 40.0                                     # arm comes in 40 deg rotated
        r.run(90, yaw_deg=yaw)                         # rests a moment: axes re-align
        x0, y0 = r.out.cursor()
        u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
        for i in range(30):                            # push along the fingers
            p = np.array([300.0, 180.0]) + u * i * 3.0
            r.run(1, fwd=p[0], right=p[1], yaw_deg=yaw)
        dx, dy = r.out.pos[0] - x0, r.out.pos[1] - y0
        return math.degrees(math.atan2(dx, -dy))       # 0 = straight up on screen

    def test_adaptive_axes_make_finger_direction_up(self):
        self.assertLess(abs(self.stroke_along_fingers("adaptive")), 8)

    def test_fixed_axes_stay_on_the_desk(self):
        self.assertGreater(abs(self.stroke_along_fingers("fixed")), 30)

    def test_mouse_axes_also_make_finger_direction_up(self):
        self.assertLess(abs(self.stroke_along_fingers("mouse")), 8)

    def test_adaptive_axes_do_not_turn_during_a_stroke(self):
        r = Rig(pointer={"axes": "adaptive"})
        r.run(10)
        for i in range(30):                            # fast stroke with the wrist yawing 30 deg
            r.run(1, fwd=300.0, right=180.0 + i * 8.0, yaw_deg=i)
        self.assertLess(abs(r.eng.status()["axes_deg"]), 2)


class TiltTests(unittest.TestCase):
    def test_hand_tilted_on_its_side_is_not_rejected(self):
        r = Rig()
        r.run(10)
        x0 = r.out.pos[0]
        for i in range(30):                            # palm_sin near zero but still positive
            r.run(1, right=180.0 + i * 3.0, palm_sin=0.2)
        self.assertGreater(r.out.pos[0] - x0, 50)

    def test_edge_on_flicker_holds_instead_of_losing(self):
        r = Rig()
        r.run(10)
        r.run(4, palm_sin=-0.1)                        # momentary flip while edge-on
        self.assertNotEqual(r.eng.state, "IDLE")

    def test_mirrored_hand_is_rejected(self):
        r = Rig()
        r.run(10)
        before = r.out.cursor()
        for i in range(15):
            r.run(1, right=180.0 + i * 4.0, palm_sin=-0.7)
        self.assertEqual(r.out.cursor(), before)


class RateScrollTests(unittest.TestCase):
    def test_fist_held_forward_keeps_scrolling(self):
        r = Rig()
        r.run(10)
        r.run(6, two=True)
        for i in range(12):
            r.run(1, fwd=300.0 + i * 4.0, two=True)   # nudge 45 mm forward
        n0 = sum(e[2][0] for e in r.out.events if e[1] == "scroll")
        r.run(30, fwd=344.0, two=True)                # hold still there for 1 s
        n1 = sum(e[2][0] for e in r.out.events if e[1] == "scroll")
        self.assertEqual(r.eng.state, SCROLL)
        self.assertLess(n1 - n0, -600)                 # kept scrolling (natural: content up)

    def test_small_hold_does_not_autoscroll(self):
        r = Rig()
        r.run(10)
        r.run(6, two=True)
        for i in range(5):
            r.run(1, fwd=300.0 + i * 3.0, two=True)   # 12 mm: past the dead zone, inside rate_mm
        n0 = sum(e[2][0] for e in r.out.events if e[1] == "scroll")
        r.run(30, fwd=312.0, two=True)
        n1 = sum(e[2][0] for e in r.out.events if e[1] == "scroll")
        self.assertGreater(n1 - n0, -60)

    def test_fast_flick_scrolls_further_than_slow(self):
        def total(speed):
            r = Rig()
            r.run(10)
            r.run(6, two=True)
            n = int(30 / (speed * DT))
            for i in range(n + 1):
                r.run(1, fwd=300.0 + min(i * speed * DT, 30.0), two=True)
            r.run(1, fwd=330.0, two=False)
            return -sum(e[2][0] for e in r.out.events if e[1] == "scroll")
        self.assertGreater(total(400), 1.5 * total(60))



def edge_obs(right, fwd=300.0, scale=1000.0):
    """A hand with a real landmark box, so the engine can see it entering the edge band.
    Image y is the user's right here; the frame is 640 x 360 and the band starts 43 px in."""
    o = obs(fwd=fwd, right=right)
    o.img = np.array([[fwd + dx, right + dy] for dx in (-40, 0, 40) for dy in (-30, 0, 30)] * 3, float)[:21]
    o.frame_wh, o.px_per_m, o.palm_size = (640, 360), scale, scale * 0.1
    return o


class ExitGlideTests(unittest.TestCase):
    def rig(self):
        r = Rig()
        r.t = 0.0
        for _ in range(25):
            r.t += DT
            r.out.t = r.t
            r.eng.update(edge_obs(180.0), r.t)
        return r

    def feed(self, r, o, n=1):
        for _ in range(n):
            r.t += DT
            r.out.t = r.t
            r.eng.update(o, r.t)

    def glide_out(self, r):
        y = 180.0
        while y < 326:                                  # slide right into the band
            y += 6.0
            self.feed(r, edge_obs(y))
        self.feed(r, edge_obs(326.0), 12)               # held in the band: gliding
        return r.out.pos[0]

    def test_gliding_out_of_view_keeps_going_until_the_hand_returns(self):
        r = self.rig()
        x = self.glide_out(r)
        self.feed(r, Obs(present=False), 15)            # hand pushed out of the view
        x_out = r.out.pos[0]
        self.assertGreater(x_out - x, 150)
        self.feed(r, edge_obs(200.0), 10)               # back in the middle: glide stops
        x_back = r.out.pos[0]
        self.feed(r, edge_obs(200.0), 10)
        self.assertLess(abs(r.out.pos[0] - x_back), 3)

    def test_leaving_without_gliding_leaves_the_cursor_put(self):
        r = self.rig()
        x = r.out.pos[0]
        self.feed(r, Obs(present=False), 20)
        self.assertEqual(r.out.pos[0], x)
        self.assertEqual(r.eng.exits, 1)

    def test_no_glide_after_a_lift(self):
        r = self.rig()
        self.glide_out(r)
        for k in range(1, 7):                           # lifted while at the edge, then gone
            self.feed(r, edge_obs(326.0, scale=1000 * (1 + 0.05 * k)))
        x = r.out.pos[0]
        self.feed(r, Obs(present=False), 15)
        self.assertLess(abs(r.out.pos[0] - x), 3)

    def test_exit_glide_has_a_time_limit(self):
        r = self.rig()
        self.glide_out(r)
        self.feed(r, Obs(present=False), int(1.6 / DT))
        x = r.out.pos[0]
        self.feed(r, Obs(present=False), 20)
        self.assertEqual(r.out.pos[0], x)

if __name__ == "__main__":
    unittest.main()
