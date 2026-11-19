"""Pointing test (ISO 9241-411 multi-directional tapping): scoring, saving, comparing devices."""
import math
import tempfile
import unittest

import numpy as np

from glide import pointing
from glide.setup import throughput


def ring_trials(amp=400.0, spread=0.0, mt=1.0, n=9, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        a = np.array([0.0, 0.0])
        b = a + amp * np.array([math.cos(ang), math.sin(ang)])
        c = b + rng.normal(0, spread, 2) if spread else b.copy()
        out.append({"a": a.tolist(), "b": b.tolist(), "c": c.tolist(), "t": mt})
    return out


class ScoreTests(unittest.TestCase):
    def test_matches_the_iso_formula(self):
        tr = ring_trials(spread=8.0, mt=0.8)
        s = pointing.score_round(tr, 32)
        dx = []
        for r in tr:
            a, b, c = (np.array(r[k]) for k in "abc")
            u = (b - a) / np.linalg.norm(b - a)
            dx.append((c - a) @ u - np.linalg.norm(b - a))
        de = np.mean([(np.array(r["c"]) - np.array(r["a"])) @ ((np.array(r["b"]) - np.array(r["a"])) / 400.0) for r in tr])
        ide = math.log2(de / (4.133 * np.std(dx)) + 1)
        self.assertAlmostEqual(s["tp"], ide / 0.8, places=1)
        self.assertEqual(s["n"], 9)

    def test_same_as_the_setup_practice_score(self):
        tr = ring_trials(spread=6.0, mt=1.2, seed=3)
        tp, mt = throughput([(np.array(r["a"]), np.array(r["b"]), np.array(r["c"]), r["t"]) for r in tr])
        self.assertAlmostEqual(pointing.score_round(tr, 40)["tp"], tp, places=2)

    def test_misses_are_clicks_outside_the_circle(self):
        tr = ring_trials()
        tr[0]["c"] = [tr[0]["b"][0] + 30, tr[0]["b"][1]]      # 30 px off a 32 px circle
        self.assertEqual(pointing.score_round(tr, 32)["errors"], 1)
        self.assertEqual(pointing.score_round(tr, 64)["errors"], 0)

    def test_test_score_is_the_mean_of_rounds(self):
        rounds = [{"w": 32, "trials": ring_trials(spread=5, mt=1.0)}, {"w": 64, "trials": ring_trials(spread=9, mt=0.5)}]
        s = pointing.score(rounds)
        self.assertAlmostEqual(s["tp"], np.mean([r["tp"] for r in s["rounds"]]), places=2)
        self.assertEqual(s["trials"], 18)

    def test_too_few_clicks_are_not_scored(self):
        self.assertIsNone(pointing.score([{"w": 32, "trials": ring_trials(n=3)}]))


class SaveTests(unittest.TestCase):
    def test_save_history_and_compare(self):
        with tempfile.TemporaryDirectory() as d:
            slow = [{"w": 32, "trials": ring_trials(spread=6, mt=1.6)}]
            fast = [{"w": 32, "trials": ring_trials(spread=6, mt=0.8)}]
            self.assertIsNone(pointing.save("joystick", fast, folder=d))     # unknown device
            m = pointing.save("mouse", fast, [1200, 700], folder=d)
            import os, time
            os.utime(next(iter(__import__("pathlib").Path(d).glob("pointing_mouse_*"))), (time.time() - 5,) * 2)
            g = pointing.save("glide", slow, [1200, 700], folder=d)
            hist = pointing.history(folder=d)
            self.assertEqual([h["device"] for h in hist], ["glide", "mouse"])
            self.assertNotIn("raw", hist[0])                               # the list stays small
            c = pointing.compare(hist)
            self.assertAlmostEqual(c["ratio"], round(g["tp"] / m["tp"], 2))
            self.assertAlmostEqual(c["ratio"], 0.5, places=1)               # same accuracy, twice the time


if __name__ == "__main__":
    unittest.main()
