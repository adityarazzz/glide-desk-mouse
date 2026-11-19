"""NFR-06 privacy: camera frames stay in memory, recordings hold landmarks only, nothing goes out."""
import re
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent / "glide"
SOURCES = {p.name: p.read_text(encoding="utf-8") for p in PKG.glob("*.py")}


class NoImageWritesTests(unittest.TestCase):
    def test_no_code_writes_images_or_video(self):
        for name, src in SOURCES.items():
            # (VideoWriter_fourcc only names the camera's pixel format; a writer is VideoWriter(...))
            for bad in (r"cv2\.imwrite", r"VideoWriter\(", r"\.save\([^)]*\.(png|jpe?g|bmp|mp4|avi)"):
                self.assertIsNone(re.search(bad, src), f"{name} writes media ({bad})")

    def test_recorded_rows_hold_landmarks_not_pictures(self):
        from glide.record import frame_row

        class Tracker:
            raw = (np.zeros((21, 2)), np.zeros((21, 3)), 0.99)
        row = frame_row(np.zeros((360, 640, 3), np.uint8), Tracker(), 1.0, cursor=(10, 20))
        self.assertLessEqual(set(row), {"t", "wh", "cur", "img", "world", "score"})
        self.assertEqual(np.asarray(row["img"]).shape, (21, 2))              # 21 points, not pixels

    def test_only_the_local_instance_is_ever_contacted(self):
        # the one outgoing request in the package asks an already running Glide (127.0.0.1) to show itself
        users = [n for n, s in SOURCES.items() if "urlopen(" in s]
        self.assertLessEqual(set(users), {"desktop.py", "report.py"})
        for name in ("requests", "http.client", "smtplib", "ftplib"):
            self.assertFalse(any(re.search(rf"^\s*(import|from) {re.escape(name)}\b", s, re.M) for s in SOURCES.values()), name)

    def test_server_binds_to_loopback_only(self):
        self.assertIn('Server(("127.0.0.1", port)', SOURCES["server.py"])
        self.assertNotIn("0.0.0.0", SOURCES["server.py"])


if __name__ == "__main__":
    unittest.main()
