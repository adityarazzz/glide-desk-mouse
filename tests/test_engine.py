import copy
import math
import unittest

import numpy as np

from glide.config import DEFAULTS
from glide.engine import CLUTCH, DRAG, HOLD, IDLE, PINCH, SCROLL, TRACK, Engine
from glide.features import Obs
from glide.filters import OneEuro2D
from glide.replay import VirtualOut

DT = 1 / 30
# image x = user's forward, image y = user's right (how the ZEB camera sees the desk); 1 px = 1 mm
PROFILE = {"R": [[0.0, 1.0], [1.0, 0.0]], "px_per_m": 1000.0, "chirality": 1}
OPEN8, OPEN12 = 0.9, 1.2            # open-hand thumb gaps measured on the desk camera


def obs(fwd=0.0, right=0.0, flex8=OPEN8, flex12=OPEN12, scale=1000.0, fist=False, score=0.99, two=False):
    return Obs(present=True, palm_px=np.array([fwd, right]), px_per_m=scale, flex8=flex8, flex12=flex12,
               fist=fist, chirality=1, score=score, hand_m=0.08, two_finger=two)


class Rig:
    def __init__(self, **cfg_over):
        cfg = copy.deepcopy(DEFAULTS)
        for k, v in cfg_over.items():
            cfg[k].update(v)
        self.out = VirtualOut()
        self.eng = Engine(cfg, PROFILE, self.out)
        self.t = 0.0

    def run(self, n, **kw):
        for _ in range(n):
            self.step(**kw)

    def step(self, o=None, typing=False, **kw):
        self.t += DT
        self.out.t = self.t
        self.eng.update(o if o is not None else obs(**kw), self.t, typing)

    def kinds(self):
        return [(e[1], e[2]) for e in self.out.events if e[1] != "scroll"]

    def pinch(self, frames, depth=0.4, **kw):
        """Quick dip of the thumb-middle gap (a deliberate pinch dips within ~100 ms)."""
        self.step(flex12=OPEN12 * 0.85, **kw)
        self.run(frames, flex12=OPEN12 * depth, **kw)


class PointerTests(unittest.TestCase):
    def test_still_noisy_hand_does_not_jitter(self):
        r, rng = Rig(), np.random.default_rng(0)
        r.run(10)
        start = r.out.cursor()
        path = []
        for _ in range(90):
            fwd, right = rng.normal(0, 0.4, 2)          # measured anchor noise is ~0.3-0.4 mm
            r.step(fwd=fwd, right=right)
            path.append(r.out.cursor())
        self.assertLessEqual(max(np.hypot(p[0] - start[0], p[1] - start[1]) for p in path), 2.0)

    def test_axes(self):
        r = Rig()
        r.run(5)
        x0, y0 = r.out.cursor()
        for i in range(30):
            r.step(right=i * 3.0)                       # hand slides to the user's right
        self.assertGreater(r.out.pos[0] - x0, 50)
        self.assertLess(abs(r.out.pos[1] - y0), 5)
        y1 = r.out.pos[1]
        for i in range(30):
            r.step(fwd=i * 3.0, right=87.0)             # hand slides away from the user
        self.assertLess(r.out.pos[1] - y1, -50)         # cursor goes up

    def test_fast_moves_get_more_gain_per_mm(self):
        def px_for(speed):
            r = Rig()
            r.run(20)                                   # hand settled after arriving
            x0, n = r.out.pos[0], int(60 / (speed * DT))
            for i in range(n + 1):
                r.step(right=i * speed * DT)
            r.run(20, right=n * speed * DT)
            return r.out.pos[0] - x0
        self.assertGreater(px_for(400), 2.5 * px_for(30))

    def test_hand_sliding_into_view_does_not_fling_the_cursor(self):
        r = Rig()
        before = r.out.cursor()
        for i in range(6):                              # arrives moving at 150 mm/s, then stops
            r.step(right=i * 5.0)
        r.run(10, right=25.0)
        self.assertLess(abs(r.out.pos[0] - before[0]), 40)

    def test_moves_normally_once_the_hand_has_settled(self):
        r = Rig()
        r.run(20)
        x0 = r.out.pos[0]
        for i in range(6):
            r.step(right=i * 5.0)
        self.assertGreater(r.out.pos[0] - x0, 60)

    def test_reacquire_elsewhere_does_not_jump(self):
        r = Rig()
        r.run(10)
        r.run(3, o=Obs(present=False))
        before = r.out.cursor()
        r.run(10, fwd=120.0, right=-80.0)
        self.assertEqual(r.out.cursor(), before)

    def test_single_frame_glitch_is_ignored(self):
        r = Rig()
        r.run(10)
        before = r.out.cursor()
        r.step(right=150.0)                             # tracker teleports for one frame
        r.run(10)
        self.assertLessEqual(abs(r.out.pos[0] - before[0]), 1)

    def test_lift_clutch(self):
        r = Rig()
        r.run(5)
        before = r.out.cursor()
        r.run(6, scale=1450.0)                          # lifted high toward the camera
        for i in range(20):
            r.step(right=i * 4.0, scale=1450.0)
        self.assertEqual(r.eng.state, CLUTCH)
        self.assertEqual(r.out.cursor(), before)

    def test_hand_size_wobble_is_not_a_lift(self):
        r = Rig()
        r.run(20)
        for i in range(90):                             # +-8 %: perspective and pose noise
            r.step(right=40 * np.sin(i / 9), scale=1000 * (1 + 0.08 * np.sin(i / 4)))
            self.assertNotEqual(r.eng.state, CLUTCH)

    def test_fist_forming_is_not_a_lift(self):
        r = Rig()
        r.run(20)
        r.run(20, scale=1300.0, fist=True)              # curling fingers shift the knuckle points
        self.assertNotEqual(r.eng.state, CLUTCH)

    def test_typing_guard_blocks_motion_and_clicks(self):
        r = Rig()
        r.run(10)
        before = r.out.cursor()
        for i in range(20):
            r.step(right=i * 4.0, typing=True)
        r.pinch(4, right=76.0, typing=True)
        r.run(4, right=76.0, typing=True)
        self.assertEqual(r.out.cursor(), before)
        self.assertEqual(r.out.events, [])

    def test_other_hand_is_ignored(self):
        r = Rig()
        r.run(5)
        o = obs(right=50.0)
        o.chirality, o.palm_sin = -1, -0.7
        before = r.out.cursor()
        r.run(10, o=o)
        self.assertEqual(r.out.cursor(), before)


class ClickTests(unittest.TestCase):
    def test_quick_pinch_clicks_where_it_started(self):
        r = Rig()
        r.run(10)
        for i in range(10):
            r.step(right=i * 5.0)
        r.pinch(4, right=45.0)
        self.assertEqual(r.eng.state, PINCH)
        r.run(4, right=45.0)
        self.assertEqual(r.kinds(), [("left", True), ("left", False)])
        self.assertEqual(r.out.events[0][3], r.out.events[1][3])

    def test_click_shift_does_not_become_a_drag(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(2)
        r.run(3, right=9.0, flex12=OPEN12 * 0.4)        # landmarks shift ~9 mm while pinching
        r.run(4)
        self.assertEqual(r.kinds(), [("left", True), ("left", False)])

    def test_pinch_hold_is_right_click(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(32)                                     # ~1.07 s held still (default hold 0.9 s)
        self.assertEqual(r.eng.state, HOLD)
        r.run(4)
        self.assertEqual(r.kinds(), [("right", True), ("right", False)])

    def test_pinch_and_move_drags(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(2)
        x0 = r.out.pos[0]
        for i in range(30):
            r.step(right=i * 2.0, flex12=OPEN12 * 0.4)
        self.assertEqual(r.eng.state, DRAG)
        self.assertGreater(r.out.pos[0] - x0, 40)
        r.run(4, right=58.0)
        self.assertEqual(r.kinds(), [("left", True), ("left", False)])

    def test_slow_drift_into_pinch_pose_is_not_a_click(self):
        r = Rig()
        r.run(10)
        for i in range(40):                             # gap closes over 1.3 s
            r.step(flex12=OPEN12 * (1 - 0.6 * i / 39))
        r.run(10)
        self.assertEqual(r.out.events, [])

    def test_low_confidence_frames_cannot_click(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(4, score=0.7)
        r.run(4)
        self.assertEqual(r.out.events, [])

    def test_release_works_after_pose_drift(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(3)
        r.run(4, flex12=OPEN12 * 0.75)                  # opens, but to a new, more closed rest pose
        self.assertEqual(r.kinds(), [("left", True), ("left", False)])
        self.assertEqual(r.eng.state, TRACK)

    def test_lost_hand_releases_held_drag(self):
        r = Rig()
        r.run(30)                                       # resting on the desk, settled
        r.pinch(2)
        for i in range(25):
            r.step(right=i * 2.0, flex12=OPEN12 * 0.4)
        r.run(3, o=Obs(present=False))
        self.assertIn("left", r.out.held)               # short dropout keeps the drag
        r.run(10, o=Obs(present=False))
        self.assertNotIn("left", r.out.held)
        self.assertEqual(r.eng.state, IDLE)

    def test_dwell_click(self):
        r = Rig(dwell={"enabled": True})
        r.run(5)
        for i in range(20):
            r.step(right=i * 4.0)
        r.run(30, right=76.0)
        self.assertEqual(r.kinds(), [("left", True), ("left", False)])


class ScrollTests(unittest.TestCase):
    def fist_move(self, r, mm, frames=30, **pose):
        pose = pose or {"two": True}                    # two-finger slide is the scroll gesture
        r.run(4, **pose)
        for i in range(frames):
            r.step(fwd=i * mm / frames, **pose)

    def test_fist_and_move_scrolls_naturally(self):
        r = Rig()
        r.run(10)
        self.fist_move(r, 60.0)
        self.assertEqual(r.eng.state, SCROLL)
        wheel = [e[2][0] for e in r.out.events if e[1] == "scroll"]
        self.assertTrue(wheel and all(v < 0 for v in wheel))   # hand forward = content up
        self.assertEqual(r.kinds(), [])

    def test_fist_dead_zone(self):
        r = Rig()
        r.run(10)
        self.fist_move(r, 6.0)
        self.assertEqual([e for e in r.out.events if e[1] == "scroll"], [])

    def test_scroll_inertia(self):
        r = Rig()
        r.run(10)
        self.fist_move(r, 90.0, frames=15)
        n = len([e for e in r.out.events if e[1] == "scroll"])
        r.run(15, fwd=90.0)                             # fist opens, hand stops
        self.assertGreater(len([e for e in r.out.events if e[1] == "scroll"]), n)

    def test_curled_grip_fist_does_nothing_by_default(self):
        r = Rig()
        r.run(10)
        x0 = r.out.pos[1]
        self.fist_move(r, 60.0, fist=True)
        self.assertEqual(r.eng.state, TRACK)            # moves the cursor like any hand pose
        self.assertLess(r.out.pos[1], x0 - 40)
        self.assertEqual([e for e in r.out.events if e[1] == "scroll"], [])

    def test_fist_scroll_option(self):
        r = Rig(scroll={"gesture": "fist"}, fist={"action": "scroll"})
        r.run(10)
        self.fist_move(r, 60.0, fist=True)
        self.assertEqual(r.eng.state, SCROLL)

    def test_fist_clutch_option(self):
        r = Rig(fist={"action": "clutch"})
        r.run(10)
        before = r.out.cursor()
        self.fist_move(r, 60.0, fist=True)
        self.assertEqual(r.out.cursor(), before)
        self.assertEqual(r.out.events, [])


class FilterTests(unittest.TestCase):
    def test_one_euro_reduces_noise_and_converges(self):
        f, rng = OneEuro2D(0.6, 0.1, 1.0), np.random.default_rng(1)
        raw = rng.normal(0, 0.5, (300, 2))
        out = np.array([f(x, i * DT) for i, x in enumerate(raw)])
        self.assertLess(out[50:].std(), 0.5 * raw.std())
        for i in range(300, 345):
            y = f([100.0, 0.0], i * DT)
        self.assertGreater(y[0], 95.0)



class RepositionTests(unittest.TestCase):
    """Slide, lift, carry the hand back, set it down: the cursor must stay where the lift began."""

    def reposition(self, lift=1.25):
        r = Rig()
        r.run(20)
        for i in range(15):                             # a stroke to the right moves the cursor
            r.step(right=i * 4.0)
        r.run(5, right=56.0)
        held = r.out.cursor()
        x = 56.0
        for i in range(6):                              # lifting while already heading back
            x -= 4.0
            r.step(right=x, scale=1000 * (1 + (lift - 1) * (i + 1) / 6))
        for _ in range(15):                             # carried back in the air
            x -= 4.0
            r.step(right=x, scale=1000 * lift)
        for i in range(6):                              # set down, still drifting a little
            x -= 2.0
            r.step(right=x, scale=1000 * (lift - (lift - 1) * (i + 1) / 6))
        r.run(15, right=x)
        return r, held, x

    def test_carrying_the_hand_back_leaves_the_cursor_put(self):
        r, held, _ = self.reposition()
        self.assertLess(abs(r.out.pos[0] - held[0]), 25)

    def test_a_small_lift_of_a_few_centimetres_works(self):
        r, held, _ = self.reposition(lift=1.18)
        self.assertLess(abs(r.out.pos[0] - held[0]), 25)

    def test_moves_normally_after_setting_down(self):
        r, held, x = self.reposition()
        x0 = r.out.pos[0]
        for i in range(15):
            r.step(right=x + i * 4.0)
        self.assertEqual(r.eng.state, TRACK)
        self.assertGreater(r.out.pos[0] - x0, 100)

class ForwardReachTests(unittest.TestCase):
    """Under the desk camera the palm looks bigger the further forward it is (1.3 per mille per px
    recorded). A long push up the screen must not read as a lift and freeze the cursor."""

    GRAD = 0.0013

    def scale(self, fwd, lift=1.0):
        return 1000 * math.exp(self.GRAD * fwd) * lift

    def push(self, r, start, end, step=6.0):
        f = start
        while (end - f) * np.sign(end - start) > 0:
            f += step * np.sign(end - start)
            r.step(fwd=f, scale=self.scale(f))
        return f

    def test_long_push_forward_keeps_moving_the_cursor_up(self):
        r = Rig()
        r.run(20, scale=self.scale(0))
        y0 = r.out.pos[1]
        self.push(r, 0, 200)                             # 200 mm forward at 180 mm/s
        self.assertFalse(r.eng.lifted)
        self.assertNotIn("lift", r.eng.last_event)
        self.assertLess(r.out.pos[1], y0 - 400)          # went a long way up the screen

    def test_back_and_forth_strokes_never_lift(self):
        r = Rig()
        r.run(20, scale=self.scale(0))
        f, lifts = 0.0, 0
        for end in (180, 20, 200, 0, 160, 40):
            f = self.push(r, f, end, step=5.0)
            r.run(10, fwd=f, scale=self.scale(f))
            lifts += r.eng.lifted
        self.assertEqual(lifts, 0)

    def test_learns_the_desk_gradient(self):
        r = Rig()
        r.run(20, scale=self.scale(0))
        f = 0.0
        for end in (150, 0, 150, 0):
            f = self.push(r, f, end, step=3.0)
            r.run(10, fwd=f, scale=self.scale(f))
        self.assertAlmostEqual(r.eng.lift_grad[0], self.GRAD, delta=0.3 * self.GRAD)

    def test_real_lift_at_the_far_end_still_works(self):
        r = Rig()
        r.run(20, scale=self.scale(0))
        f = 0.0
        for end in (180, 0, 180):                      # it has seen the desk once
            f = self.push(r, f, end)
            r.run(15, fwd=f, scale=self.scale(f))
        for i in range(6):                               # lift 5 cm in place
            r.step(fwd=f, scale=self.scale(f, 1 + 0.25 * (i + 1) / 6))
        self.assertTrue(r.eng.lifted)


class SensorPointTests(unittest.TestCase):
    """Like a mouse: swivelling at the wrist moves the cursor; a pinch forming never jumps it."""

    @staticmethod
    def hand(theta=0.0, shift=0.0, curl=0.0):
        pts = np.zeros((21, 2))
        known = {0: (0, 0), 1: (25, 20), 5: (85 - curl, 25), 9: (90 - curl, 0), 13: (85 - curl, -20), 17: (75 - curl, -38)}
        for i in range(21):
            pts[i] = known.get(i, (110 + i, (i % 4 - 1.5) * 18))
        c, s_ = math.cos(theta), math.sin(theta)
        return pts @ np.array([[c, s_], [-s_, c]]) + [200 + shift, 180]

    def obs_of(self, img, flex12=OPEN12):
        o = Obs(present=True, palm_px=img[[0, 5, 9, 13, 17]].mean(0), px_per_m=1000.0, flex8=OPEN8,
                flex12=flex12, chirality=1, score=0.99, hand_m=0.08)
        o.img, o.frame_wh = img, (640, 360)
        return o

    def swivel(self, sensor):
        r = Rig(pointer={"sensor": sensor})
        r.run(25, o=self.obs_of(self.hand()))
        y0 = r.out.pos[1]
        for k in range(1, 13):
            r.step(o=self.obs_of(self.hand(math.radians(20) * k / 12)))
        r.run(8, o=self.obs_of(self.hand(math.radians(20))))
        return abs(r.out.pos[0] - 960) + abs(r.out.pos[1] - y0)

    def test_wrist_swivel_moves_the_cursor_like_a_mouse(self):
        self.assertGreater(self.swivel([1.0, 0.0]), 2.5 * self.swivel([0.0, 0.0]))

    def test_knuckles_shifting_in_a_pinch_do_not_move_the_cursor(self):
        r = Rig()
        r.run(25, o=self.obs_of(self.hand()))
        held = list(r.out.pos)
        for k in range(1, 7):                            # pinch forms: knuckles pull back 12 px
            r.step(o=self.obs_of(self.hand(curl=2.0 * k), flex12=OPEN12 * (1 - 0.1 * k)))
        r.run(6, o=self.obs_of(self.hand(curl=12.0), flex12=OPEN12 * 0.4))
        self.assertLess(abs(r.out.pos[0] - held[0]) + abs(r.out.pos[1] - held[1]), 6)
        self.assertNotEqual(r.eng.state, DRAG)


class ReturnTests(unittest.TestCase):
    """Taking the hand off the desk and putting it back must not disturb the cursor."""

    BASE = SensorPointTests.hand()

    def obs(self, fwd=0.0, right=0.0, size=1.0, present=True):
        if not present:
            return Obs(present=False)
        c = self.BASE[[0, 5, 9, 13, 17]].mean(0)
        img = (self.BASE - c) * size + c + [fwd, right]
        o = Obs(present=True, palm_px=img[[0, 5, 9, 13, 17]].mean(0), px_per_m=1000.0 * size, flex8=OPEN8,
                flex12=OPEN12, chirality=1, score=0.99, hand_m=0.08, palm_size=90.0 * size)
        o.img, o.frame_wh = img, (640, 360)
        return o

    def settled(self):
        r = Rig()
        r.run(30, o=self.obs())
        return r

    def moved(self, r, start):
        return abs(r.out.pos[0] - start[0]) + abs(r.out.pos[1] - start[1])

    def test_quick_return_still_settles(self):
        r = self.settled()
        r.run(5, o=self.obs(present=False))             # gone for 0.17 s, less than lost_ms
        start = list(r.out.pos)
        for i in range(4):                               # back 40 mm away, still sliding in
            r.step(o=self.obs(fwd=40 + 6 * i))
        self.assertLess(self.moved(r, start), 5)

    def test_hand_put_back_from_above_waits_until_it_is_down(self):
        r = self.settled()
        r.run(20, o=self.obs(present=False))
        start = list(r.out.pos)
        for i in range(8):                               # coming down from 5 cm while drifting sideways
            r.step(o=self.obs(fwd=30 - 3 * i, right=4 * i, size=1.3 - 0.3 * i / 7))
            if i < 6:
                self.assertLess(self.moved(r, start), 5)
        for i in range(25):                              # down: it moves the cursor again
            r.step(o=self.obs(fwd=9, right=28 + 4 * i))
        self.assertGreater(self.moved(r, start), 60)

    def test_sliding_in_from_the_side_moves_nothing_until_in_view(self):
        r = self.settled()
        r.run(20, o=self.obs(present=False))
        start = list(r.out.pos)
        for i in range(6):                               # palm still crossing the bottom of the view
            r.step(o=self.obs(right=200 - 12 * i))
        self.assertLess(self.moved(r, start), 5)

    def test_taking_the_hand_away_leaves_the_cursor_put(self):
        r = self.settled()
        start = list(r.out.pos)
        for i in range(1, 7):                            # pulled out of view fast, toward you
            r.step(o=self.obs(fwd=-30 * i))
        r.run(10, o=self.obs(present=False))
        self.assertLess(self.moved(r, start), 60)        # the flick out barely counts

    def test_the_hand_is_remembered(self):
        r = self.settled()
        base, lever = r.eng.lift_base, r.eng.lever_len
        r.run(20, o=self.obs(present=False))
        self.assertEqual((r.eng.lift_base, r.eng.lever_len), (base, lever))


class PhantomClickTests(unittest.TestCase):
    """Clicks nobody made: a hand landing and relaxing, a lift that starts with a pinch-like closing,
    a long semi-closed hand that drifts (read as pinch-and-hold)."""

    def test_fingers_relaxing_as_the_hand_lands_do_not_click(self):
        r = Rig()
        r.run(30)
        r.run(20, o=Obs(present=False))
        r.run(5)                                         # back on the desk
        for k in range(8):                               # fingers relax: the gap closes by 40%
            r.step(flex12=OPEN12 * (1 - 0.05 * k))
        r.run(10, flex12=OPEN12 * 0.6)
        self.assertEqual(r.kinds(), [])

    def test_a_lift_starting_with_a_pinch_is_not_a_click(self):
        r = Rig()
        r.run(30)
        r.pinch(2)                                       # the hand closes a little...
        for k in range(6):                               # ...and lifts
            r.step(flex12=OPEN12 * 0.4, scale=1000 * (1 + 0.06 * (k + 1)))
        r.run(4, flex12=OPEN12, scale=1360.0)            # opens in the air
        self.assertEqual(r.kinds(), [])

    def test_a_drifting_long_press_is_not_a_right_click(self):
        r = Rig()
        r.run(30)
        r.pinch(2)
        for k in range(32):                              # held 1 s while wandering 12 mm (< drag_mm)
            r.step(flex12=OPEN12 * 0.4, right=12.0 * k / 31)
        self.assertNotIn(("right", True), r.kinds())


if __name__ == "__main__":
    unittest.main()
