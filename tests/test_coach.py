"""The coached recorder, driven by a simulated user built from the demo hand itself."""
import unittest

import numpy as np

from glide.coach import BEAT_S, BEATS, Coach, Step
from glide.demo import DEMOS, View, envelope, landmarks_px
from glide.features import Obs, measure
from glide.record import SCRIPTS

DT = 1 / 30


def hand(pose, jitter=0.0, rng=None, side="right"):
    img, world = landmarks_px(pose, side)
    if jitter:
        img = img + rng.normal(0, jitter, img.shape)
    return measure(img, world, 0.99, (640, 360))


class SimUser:
    """Obeys the coach: rests still while waiting, performs the step's demo on GO."""

    def __init__(self, coach, present=lambda t: True, perform=True):
        self.c, self.present, self.perform = coach, present, perform
        self.rng = np.random.default_rng(0)
        self.t = 0.0
        self.hand = "right"

    def run(self, seconds):
        for _ in range(int(seconds / DT)):
            self.t += DT
            c = self.c
            if c.done:
                return
            s = c.step
            d = DEMOS.get(s.demo or s.label, DEMOS["rest"])
            if not self.present(self.t):
                obs = Obs(present=False)
            elif c.state in ("respond", "result") and s.kind == "cued" and self.perform:
                obs = hand(d.pose(0.0, g=envelope(c.since_go(), *d.env)), 0.3, self.rng, self.hand)
            elif c.state == "run":
                obs = hand(d.pose(self.t), 0.3, self.rng, self.hand)
            else:
                obs = hand(d.pose(0.0, g=0.0) if d.act else d.base, 0.3, self.rng, self.hand)
            c.feed(obs, self.t, {"t": self.t})


CUED = [Step("pinch_mid", "Pinch", "On GO, pinch.", kind="cued", reps=3)]


class CoachTests(unittest.TestCase):
    def test_cued_step_runs_practice_then_trials_all_seen(self):
        c = Coach(CUED)
        SimUser(c).run(60)
        self.assertTrue(c.done)
        self.assertEqual(c.summary[0]["trials"], 3)
        self.assertEqual(c.summary[0]["seen"], 3)
        labels = [x["label"] for x in c.cues]
        self.assertEqual(labels, ["practice", "pinch_mid", "pinch_mid", "pinch_mid"])

    def test_nothing_starts_while_the_user_could_still_be_reading(self):
        c = Coach(CUED)
        SimUser(c).run(60)
        self.assertGreater(c.cues[0]["t"], c.min_intro + BEATS * BEAT_S)

    def test_cues_are_spaced_beyond_the_analyzer_window(self):
        c = Coach(CUED)
        SimUser(c).run(60)
        gaps = np.diff([x["t"] for x in c.cues])
        self.assertGreater(gaps.min(), 1.6 + BEATS * BEAT_S)

    def test_missed_gesture_is_reported_and_practice_repeats_once(self):
        c = Coach(CUED)
        SimUser(c, perform=False).run(60)
        self.assertEqual(c.summary[0]["seen"], 0)
        self.assertEqual(sum(1 for x in c.cues if x["label"] == "practice"), 2)

    def test_no_cue_while_the_hand_is_away(self):
        c = Coach(CUED)
        away = lambda t: not (2.0 < t < 20.0)
        SimUser(c, present=away).run(30)
        self.assertTrue(all(not (2.0 < x["t"] < 20.0) for x in c.cues))

    def test_still_step_clock_stops_while_hand_is_gone(self):
        steps = [Step("rest", "Still", "Hold still.", kind="still", seconds=3)]
        c = Coach(steps)
        SimUser(c, present=lambda t: t < 6.0 or t > 30.0).run(25)
        self.assertFalse(c.done)

    def test_still_step_measures_orientation(self):
        c = Coach(SCRIPTS["basic"][:1])
        SimUser(c).run(15)
        self.assertTrue(c.done)
        right, fwd = c.R
        self.assertGreater(-fwd[1], 0.9)               # demo hand's fingers point image up

    def test_left_hand_orientation_points_right_to_the_users_right(self):
        c = Coach(SCRIPTS["basic"][:1], hand="left")
        u = SimUser(c)
        u.hand = "left"
        u.run(15)
        right, fwd = c.R
        self.assertGreater(-fwd[1], 0.9)
        self.assertGreater(right[0], 0.9)              # image +x is the user's right in the demo view

    def test_scroll_pose_check_passes_for_the_demo_pose(self):
        c = Coach([s for s in SCRIPTS["precision"] if s.label == "two_scroll"])
        SimUser(c).run(40)
        self.assertGreater(c.summary[0]["check_frac"], 0.9)

    def test_redo_drops_the_step_data(self):
        c = Coach(CUED)
        u = SimUser(c)
        u.run(12)
        self.assertTrue(c.cues)
        c.redo()
        self.assertEqual(c.cues, [])
        self.assertEqual(len(c.rows), c.row0)

    def test_every_script_step_has_a_demo(self):
        for name, steps in SCRIPTS.items():
            for s in steps:
                self.assertIn(s.demo or s.label, DEMOS, f"{name}:{s.label}")


class PacedCalibrationTests(unittest.TestCase):
    def run_cal(self, present=lambda t: True):
        from glide.calibrate import Calibrator
        from glide.config import DEFAULTS
        cal, t, rng = Calibrator(DEFAULTS), 0.0, np.random.default_rng(1)
        d = DEMOS["pinch_mid"]
        for _ in range(int(40 / DT)):
            t += DT
            if not present(t):
                obs = Obs(present=False)
            else:
                g = envelope(t - cal.cues[-1][0], *d.env) if cal.cues and cal.phase == "sample" else 0.0
                obs = hand(d.pose(0.0, g=g), 0.2, rng)
            if cal.feed(obs, t):
                return cal, t
        return cal, t

    def test_cues_wait_for_the_hand_and_count_in(self):
        cal, _ = self.run_cal()
        self.assertIn("click", cal.profile)
        gaps = np.diff([c for c, _ in cal.cues])
        self.assertGreaterEqual(gaps.min(), 1.5 + 3 * 0.5)

    def test_no_cue_while_hand_is_away(self):
        cal, _ = self.run_cal(present=lambda t: not (3.0 < t < 15.0))
        self.assertTrue(all(not (3.0 < c < 15.0) for c, _ in cal.cues))


class DemoPoseTests(unittest.TestCase):
    """The demos must teach poses that the real feature extractor accepts."""

    def test_scroll_demo_is_two_finger_pose(self):
        self.assertTrue(hand(DEMOS["two_scroll"].base).two_finger)

    def test_flat_and_grip_are_not_scroll_poses(self):
        for name in ("rest", "grip", "move_slow"):
            self.assertFalse(hand(DEMOS[name].base).two_finger, name)

    def test_fist_demo_is_a_fist(self):
        self.assertTrue(hand(DEMOS["fist"].pose(0, g=1.0)).fist)

    def test_pinch_demo_closes_the_gap_with_a_straight_finger(self):
        o = hand(DEMOS["pinch_mid"].pose(0, g=1.0))
        self.assertLess(o.pinch_mid, 0.15)
        self.assertGreater(o.ext[1], 1.70)              # engine's min_ext: the pinching finger stays straight

    def test_side_demo_reads_as_tilted_not_mirrored(self):
        o = hand(DEMOS["side"].base)
        self.assertTrue(0 < o.palm_sin < 0.45)


class ViewTests(unittest.TestCase):
    def test_turns_any_camera_orientation_fingers_up(self):
        for fwd, right in [((1, 0), (0, 1)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)), ((0, -1), (1, 0)),
                           ((1, 0), (0, -1))]:
            v = View([right, fwd], (640, 360))
            self.assertTrue(np.allclose(v.M @ fwd, (0, -1)), (fwd, right))
            self.assertTrue(np.allclose(v.M @ right, (1, 0)), (fwd, right))
            p = v.pts([(0, 0), (639, 359)])
            self.assertTrue((p >= 0).all() and (p[:, 0] < v.size[0]).all() and (p[:, 1] < v.size[1]).all())


if __name__ == "__main__":
    unittest.main()
