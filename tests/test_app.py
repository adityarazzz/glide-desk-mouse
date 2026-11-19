"""The desktop app layer: the local server's guards, settings, the setup flow and the demo data."""
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from glide import service as service_mod
from glide.calibrate import rest_profile
from glide.config import DEFAULTS
from glide.demo import DEMOS, export, landmarks_px
from glide.features import measure
from glide.onboard import Learn
from glide.server import serve

DT = 1 / 30


def hand_obs(pose, rng):
    img, world = landmarks_px(pose)
    return measure(img + rng.normal(0, 0.3, img.shape), world, 0.99, (640, 360))


class FakeService:
    def __init__(self):
        self.posted = []
        self.cond = __import__("threading").Condition()
        self.snapshot, self.snap_seq, self.jpeg, self.jpeg_seq, self.viewers = {}, 0, None, 0, 0
        self.camera_thumbs = {}

    def post(self, action, **kw):
        self.posted.append((action, kw))

    def settings(self):
        return {"values": {}}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = FakeService()
        cls.url, cls.httpd = serve(cls.svc)
        cls.base, cls.token = cls.url.split("/?t=")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def status(self, req):
        try:
            return urllib.request.urlopen(req, timeout=3).status
        except urllib.error.HTTPError as e:
            return e.code

    def post(self, body, token=None, host=None):
        h = {"Content-Type": "application/json"}
        if token:
            h["X-Glide-Token"] = token
        if host:
            h["Host"] = host
        return self.status(urllib.request.Request(self.base + "/api/cmd", json.dumps(body).encode(), h, method="POST"))

    def test_page_is_served_without_a_token(self):
        self.assertEqual(self.status(self.base + "/"), 200)
        self.assertEqual(self.status(self.base + "/app.js"), 200)

    def test_api_needs_the_token(self):
        self.assertEqual(self.status(self.base + "/api/settings"), 403)
        self.assertEqual(self.status(self.base + f"/api/settings?t={self.token}"), 200)

    def test_commands_need_the_token_header(self):
        n = len(self.svc.posted)
        self.assertEqual(self.post({"action": "enable"}), 403)
        self.assertEqual(self.post({"action": "enable"}, token="wrong"), 403)
        self.assertEqual(len(self.svc.posted), n)
        self.assertEqual(self.post({"action": "enable", "on": False}, token=self.token), 200)
        self.assertEqual(self.svc.posted[-1], ("enable", {"on": False}))

    def test_foreign_host_header_is_refused(self):
        # a DNS-rebinding page reaches 127.0.0.1 under its own host name
        self.assertEqual(self.post({"action": "enable"}, token=self.token, host="evil.example:47620"), 403)

    def test_only_files_of_the_web_folder_are_served(self):
        for path in ("/..%2Fservice.py", "/../config.py", "/%2E%2E/app.py"):
            self.assertNotEqual(self.status(self.base + path), 200, path)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self.patch = mock.patch.object(service_mod, "CONFIG_PATH", self.tmp)
        self.patch.start()
        self.svc = service_mod.Service.__new__(service_mod.Service)   # no camera, no tracker
        self.svc.cfg = service_mod.load_config(self.tmp)
        self.svc.profile, self.svc.overlay, self.svc.cam, self.svc.toast = None, None, None, None
        self.svc.out = mock.Mock()
        self.svc._sync_overlay = lambda: None

    def tearDown(self):
        self.patch.stop()

    def test_values_are_clamped_and_saved(self):
        self.svc._set("gain.speed", 9)
        self.assertEqual(json.loads(self.tmp.read_text())["gain"]["speed"], 2.0)
        self.assertEqual(self.svc.cfg["gain"]["speed"], 2.0)

    def test_unknown_paths_and_choices_are_ignored(self):
        self.svc._set("hand.glitch_mm", 1)
        self.svc._set("scroll.gesture", "wave")
        self.assertFalse(self.tmp.exists())

    def test_preset_round_trip(self):
        self.svc._set("preset", "accessibility")
        self.assertEqual(self.svc.cfg["preset"], "accessibility")
        self.svc._set("preset", "")
        self.assertNotIn("preset", json.loads(self.tmp.read_text()))

    def test_changing_the_hand_asks_for_recalibration(self):
        self.svc._set("hand.dominant", "left")
        self.assertIn("Recalibrate", self.svc.toast)


class LearnTests(unittest.TestCase):
    def test_move_lesson_passes_with_the_demo_hand(self):
        rng = np.random.default_rng(0)
        rest = [hand_obs(DEMOS["rest"].base, rng) for _ in range(40)]
        profile = rest_profile(rest, DEFAULTS, (1920, 1080))
        learn = Learn(DEFAULTS, profile, (1920, 1080))
        d, t = DEMOS["move_slow"], 0.0
        for _ in range(int(12 / DT)):
            t += DT
            learn.feed(hand_obs(d.pose(t), rng), t)
            if learn.done[0]:
                break
        self.assertTrue(learn.done[0])
        self.assertEqual(learn.state()["lessons"][0]["name"], "Move")


class DemoExportTests(unittest.TestCase):
    def test_every_demo_exports_complete_frames(self):
        e = export()
        for name, d in e.items():
            self.assertEqual(len(d["frames"][0]), 27, name)
            if d["move"]:
                self.assertEqual(len(d["move"]), round(d["loop"] * 30), name)
        self.assertEqual(len(e["pinch_mid"]["frames"]), 21)

    def test_left_hand_is_mirrored(self):
        r, l = export("right")["rest"]["frames"][0], export("left")["rest"]["frames"][0]
        self.assertAlmostEqual(r[4][0], -l[4][0], places=1)


if __name__ == "__main__":
    unittest.main()


class SessionEndTests(unittest.TestCase):
    """With no mouse plugged in, the hand is the only pointer: a finished session must hand it back."""

    def make(self):
        from unittest import mock
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg = S.load_config(path=Path("__none__"))
        svc.out, svc.overlay, svc.engine = mock.Mock(), None, None
        svc.out.cursor.return_value = (500, 400)
        svc.mouse_watch = None
        return svc

    def test_no_mouse_is_noticed(self):
        from glide.features import Obs
        svc = self.make()
        o = Obs(present=True)
        self.assertFalse(svc._no_mouse(o, 0.0))
        self.assertFalse(svc._no_mouse(o, 5.0))
        self.assertTrue(svc._no_mouse(o, 8.5))           # 8 s of hand, pointer never moved

    def test_a_real_mouse_is_never_mistaken(self):
        from glide.features import Obs
        svc = self.make()
        o = Obs(present=True)
        svc._no_mouse(o, 0.0)
        svc.out.cursor.return_value = (600, 400)         # it moved once
        svc._no_mouse(o, 1.0)
        svc.out.cursor.return_value = (600, 400)
        self.assertFalse(svc._no_mouse(o, 30.0))         # then held still: still a mouse


class NeverStuckTests(unittest.TestCase):
    """A mouse-less user must never be stranded: a frozen camera reconnects, recalibration can be
    cancelled (Esc) and gives up by itself when it can't see a hand."""

    def make(self):
        import glide.service as S
        from glide.onboard import Calibrate
        svc = S.Service.__new__(S.Service)
        svc.cfg = S.load_config(path=Path("__none__"))
        svc.out, svc.overlay, svc.toast = mock.Mock(), None, None
        svc.screen_wh = (1920, 1080)
        svc.profile = {"R": [[0.0, 1.0], [1.0, 0.0]], "px_per_m": 1600.0}
        svc.mode, svc.setup_step, svc.setup_only_calibrate = "setup", "calibrate", True
        svc.calib, svc.calib_seen = Calibrate(svc.cfg, svc.screen_wh), None
        svc.placement = svc.learn = None
        return svc

    def test_cancel_keeps_the_old_calibration(self):
        svc = self.make()
        old = svc.profile
        svc._command("setup_cancel", {})
        self.assertEqual(svc.mode, "run")
        self.assertIs(svc.profile, old)

    def test_gives_up_when_no_hand_is_seen(self):
        from glide.features import Obs
        svc = self.make()
        svc._setup(Obs(present=False), None, 0.0, False)
        svc._setup(Obs(present=False), None, 30.0, False)
        self.assertEqual(svc.mode, "setup")
        svc._setup(Obs(present=False), None, 61.0, False)
        self.assertEqual(svc.mode, "run")
        self.assertIn("previous calibration", svc.toast)

    def test_a_frozen_camera_is_let_go_without_waiting(self):
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        hung = mock.Mock()
        hung.stop.side_effect = lambda: __import__("time").sleep(5)    # a read that never returns
        svc.cam = hung
        t0 = __import__("time").perf_counter()
        svc._drop_camera()
        self.assertIsNone(svc.cam)
        self.assertLess(__import__("time").perf_counter() - t0, 0.5)


class CameraFinderTests(unittest.TestCase):
    """Windows renumbers cameras when another one appears or fails: find the one that works."""

    def test_finds_the_camera_with_a_picture(self):
        import glide.camera as C

        def fake_open(i, cfg):
            if i == 1:
                return None                               # gone
            cap = mock.Mock()
            img = np.full((36, 64, 3), 150 if i == 2 else 2, np.uint8)   # 0: black (lid shut)
            cap.read.return_value = (True, img)
            return cap
        with mock.patch.object(C, "open_capture", fake_open):
            self.assertEqual(C.find_working({}, skip=(3,)), 2)

    def test_service_switches_and_remembers(self):
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg = S.load_config(path=Path("__none__"))
        svc.cfg["camera"]["index"] = 1
        svc.cam_search_t, svc.toast = -1e9, None
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "CONFIG_PATH", Path(d) / "config.json"), \
                mock.patch.object(S, "find_working", lambda cfg, skip, dark: 0), \
                mock.patch.object(S, "list_cameras", lambda: []):
            self.assertTrue(svc._switch_camera(skip=(1,)))
            self.assertEqual(json.loads((Path(d) / "config.json").read_text())["camera"]["index"], 0)
            self.assertFalse(svc._switch_camera(skip=(0,)))      # not again right away
        self.assertIn("switched to camera 0", svc.toast)


LAPTOP = {"index": 0, "name": "Integrated Camera", "id": "vid_04f2&pid_b725"}
ZEB = {"index": 1, "name": "ZEB LIVE PRO", "id": "vid_1bcf&pid_2281"}


class CameraIdTests(unittest.TestCase):
    """The desk camera is remembered by its device id, not its number (2026-11-19: the laptop's own
    camera came back as camera 0 and Glide quietly showed it)."""

    def run_with(self, cams, camera_cfg):
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "config.json"
        path.write_text(json.dumps({"camera": camera_cfg}))
        with mock.patch.object(S, "CONFIG_PATH", path), mock.patch.object(S, "list_cameras", lambda: cams):
            svc.cfg = S.load_config(path)
            ok = svc._resolve_camera()
        return svc, ok, json.loads(path.read_text())["camera"]

    def test_follows_the_camera_when_windows_renumbers_it(self):
        svc, ok, saved = self.run_with([LAPTOP, ZEB], {"index": 0, "id": ZEB["id"], "name": ZEB["name"]})
        self.assertTrue(ok)
        self.assertEqual(svc.cfg["camera"]["index"], 1)
        self.assertEqual(saved["index"], 1)

    def test_unplugged_desk_camera_is_reported_not_replaced(self):
        svc, ok, saved = self.run_with([LAPTOP], {"index": 1, "id": ZEB["id"], "name": ZEB["name"]})
        self.assertFalse(ok)
        self.assertEqual(svc.cam_missing, "ZEB LIVE PRO")
        self.assertEqual(saved["id"], ZEB["id"])                  # still the choice for when it's back

    def test_old_setting_on_the_laptop_camera_moves_to_the_desk_camera(self):
        svc, ok, saved = self.run_with([LAPTOP, ZEB], {"index": 0})
        self.assertTrue(ok)
        self.assertEqual((saved["index"], saved["id"]), (1, ZEB["id"]))

    def test_a_deliberate_laptop_camera_choice_is_kept(self):
        svc, ok, saved = self.run_with([LAPTOP, ZEB], {"index": 0, "id": LAPTOP["id"], "name": LAPTOP["name"]})
        self.assertTrue(ok)
        self.assertEqual(saved["index"], 0)

    def test_dark_desk_camera_does_not_fail_over_to_the_laptop_camera(self):
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg = S.load_config(path=Path("__none__"))
        svc.cfg["camera"].update(index=1, id=ZEB["id"])
        svc.cam_search_t, svc.camera_names = -1e9, {0: LAPTOP["name"], 1: ZEB["name"]}
        with mock.patch.object(S, "find_working", lambda cfg, skip, dark: 0):
            self.assertFalse(svc._switch_camera(skip=(1,)))
