"""Runs the real MediaPipe model on a still frame from the desk camera, if one is present."""
import os
import unittest
from pathlib import Path

import cv2

from glide.calibrate import Calibrator
from glide.config import DEFAULTS, ROOT
from glide.tracker import HandTracker

FRAME = Path(os.environ.get("GLIDE_TEST_FRAME", ROOT / "tests" / "data" / "desk_hand.jpg"))


@unittest.skipUnless(FRAME.exists(), "no sample frame")
class RealFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = cv2.imread(str(FRAME))
        cls.img = cv2.resize(img, (640, 360))
        tr = HandTracker(ROOT / DEFAULTS["model"], DEFAULTS["hand"])
        cls.obs = tr.detect(cls.img, 1.0)
        tr.close()

    def test_hand_found_with_sane_geometry(self):
        o = self.obs
        self.assertTrue(o.present)
        self.assertGreater(o.px_per_m, 500)
        self.assertGreater(o.pinch_idx, 0.4)            # resting flat hand is not pinching
        self.assertFalse(o.fist)

    def test_rest_calibration_orientation(self):
        cal = Calibrator(DEFAULTS)
        for i in range(45):
            cal.feed(self.obs, i / 30)
        right, fwd = cal.profile["R"]
        # in this frame the fingers point to image +x and the user's right is image +y
        self.assertGreater(fwd[0], 0.8)
        self.assertGreater(right[1], 0.8)


if __name__ == "__main__":
    unittest.main()
