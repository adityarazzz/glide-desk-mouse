"""Golden replay (test plan 6.1): a real cued session must keep its measured quality."""
import unittest
from pathlib import Path

import numpy as np

from glide.analyze import metrics
from glide.config import load_config
from glide.replay import load_session, session_profile
from glide.setup import throughput

GOLDEN = Path(__file__).parent / "data" / "golden_session.jsonl"
GOLDEN_SCROLL = Path(__file__).parent / "data" / "golden_scroll.jsonl"   # first coached recording


@unittest.skipUnless(GOLDEN.exists(), "no golden session")
class GoldenReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(path=Path("__no_user_config__"))
        cls.frames, cls.cues = load_session(GOLDEN)
        cls.m = metrics(cls.frames, cls.cues, cls.cfg, session_profile(cls.frames, cls.cfg))

    def test_no_ghost_clicks(self):
        self.assertEqual(self.m["ghosts"], 0, self.m["ghost_at"])

    def test_click_recall(self):
        self.assertGreaterEqual(self.m["click_recall"], 16 / 18)       # 12/18 before the peak-relative onset

    def test_clicks_do_not_turn_into_drags(self):
        self.assertEqual(self.m["drags_in_click_phases"], 0)
        self.assertEqual(self.m["extra_presses"], 0)

    def test_fists_do_nothing(self):
        # a curled mouse grip reads as a fist, so by default fists must be inert
        self.assertEqual(self.m["presses_in_fist_phase"], 0)

    def test_no_false_reposition(self):
        self.assertEqual(self.m["false_lifts"], 0)

    def test_static_jitter_meets_nfr02(self):
        self.assertLessEqual(self.m["jitter_px"], 1.0)

    def test_recorded_strokes_reach_across_screen(self):
        # with the old fixed gain the same strokes swept only 31% of the screen width
        self.assertGreaterEqual(self.m["reach_screen"][0], 0.5)

    def test_filter_lag(self):
        self.assertLessEqual(abs(self.m["lag_ms"]), 30)

    def test_retimed_to_uniform_clock(self):
        dt = np.diff([t for t, _, _ in self.frames])
        self.assertLess(dt.std(), 1e-3)


@unittest.skipUnless(GOLDEN_SCROLL.exists(), "no coached scroll session")
class GoldenScrollTests(unittest.TestCase):
    """Two-finger scroll vs the poses it must not be confused with (coached session)."""

    @classmethod
    def setUpClass(cls):
        cfg = load_config(path=Path("__no_user_config__"))
        frames, cues = load_session(GOLDEN_SCROLL)
        cls.m = metrics(frames, cues, cfg, session_profile(frames, cfg))

    def test_scroll_pose_scrolls_throughout(self):
        self.assertGreaterEqual(self.m["scroll_frac_by_phase"]["two_scroll"], 0.95)
        self.assertEqual(self.m["presses_in_two_scroll"], 0)

    def test_mouse_grip_and_fist_never_scroll_or_click(self):
        for lab in ("grip_move", "fist", "hover_rest"):
            self.assertEqual(self.m["scroll_frac_by_phase"][lab], 0.0, lab)
        self.assertEqual(self.m["presses_in_fist_phase"], 0)
        self.assertEqual(self.m["ghosts"], 0)
        self.assertEqual(self.m["false_lifts"], 0)      # 2 with the old fixed-threshold lift


class ThroughputTests(unittest.TestCase):
    def test_iso_throughput(self):
        a, b = np.array([0.0, 0.0]), np.array([380.0, 0.0])
        rng = np.random.default_rng(0)
        trials = [(a, b, b + [rng.normal(0, 8), rng.normal(0, 8)], 1.0) for _ in range(9)]
        tp, mt = throughput(trials)
        self.assertAlmostEqual(mt, 1.0)
        self.assertTrue(3.0 < tp < 5.0)                 # log2(380/~30 + 1) ~ 3.8 bits in 1 s


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(GOLDEN.exists() and GOLDEN_SCROLL.exists(), "no golden sessions")
class MouseLikeUseTests(unittest.TestCase):
    """The prop-mouse video (2026-10-28 13:17): on a cramped free desk the cursor slammed to the top
    and bottom (edge bands glided during ordinary moves), and a mouse grip fell back to the palm base,
    which hardly moves when the wrist swivels."""

    @staticmethod
    def replay(path, labels, bounds=None):
        from glide.engine import Engine
        from glide.replay import VirtualOut
        cfg = load_config(path=Path("__no_user_config__"))
        frames, _ = load_session(path)
        out = VirtualOut()
        eng = Engine(cfg, session_profile(frames, cfg), out)
        if bounds:
            eng.set_bounds(bounds, (640, 360))
        rows = []
        for t, lab, o in frames:
            out.t = t
            eng.update(o, t)
            if lab in labels and o.present:
                rows.append((eng.edge_depth > 0, out.pos[1] in (0, out.h - 1), eng.sensor_w))
        return np.array(rows, float)

    def test_cramped_desk_does_not_glide_during_ordinary_moves(self):
        r = self.replay(GOLDEN, ("move_slow", "move_fast"), bounds=(116, 0, 420, 328))
        # what still glides here is the hand really pushing past a wall or reaching the view's edge
        # (this session was recorded on a clear desk, so it crosses where the bag would be)
        self.assertLess(r[:, 0].mean(), 0.25)            # was 0.72-0.84 with bands at the walls
        self.assertLess(r[:, 1].mean(), 0.10)            # cursor pinned top/bottom: was 0.59-0.71

    def test_mouse_grip_moves_like_a_mouse(self):
        r = self.replay(GOLDEN_SCROLL, ("grip_move",))
        self.assertGreater(r[:, 2].mean(), 0.9)          # knuckle lever in use: was 0.19


EXIT_TILT = Path(__file__).parent / "data" / "exit_tilt.jsonl"   # Ctrl+Alt+X capture, 2026-10-28 21:04


@unittest.skipUnless(EXIT_TILT.exists(), "no exit capture")
class DoubtfulFrameTests(unittest.TestCase):
    """The hand tilting off the desk: one frame scored 0.67 and its landmarks flung the cursor 1160 px
    across the screen before the leave rollback brought it back (NFR-07: no action on uncertain state)."""

    @staticmethod
    def biggest_step(gate):
        import copy
        from glide.replay import replay
        cfg = copy.deepcopy(load_config(path=Path("__no_user_config__")))
        cfg["hand"]["min_track_score"] = gate
        frames, _ = load_session(EXIT_TILT)
        _, trace = replay(frames, cfg, session_profile(frames, cfg))
        cur = np.array([c for *_, c in trace], float)
        return float(np.hypot(*np.diff(cur, axis=0).T).max())

    def test_doubtful_leap_does_not_move_the_cursor(self):
        self.assertGreater(self.biggest_step(0.0), 600)    # the recorded flash, without the gate
        self.assertLess(self.biggest_step(0.70), 400)      # ordinary steps in this capture: <= 170 px
