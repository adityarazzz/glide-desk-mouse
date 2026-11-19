"""Free-desk detector: a bag, a keyboard or the desk's edge in view shrinks the workspace to the free
desk, and the glide edges follow it."""
import copy
import unittest

import cv2
import numpy as np

from glide.config import DEFAULTS
from glide.engine import Engine
from glide.features import Obs
from glide.replay import VirtualOut
from glide.surface import desk_side, detect, same_room
from glide.workspace import fit_workspace

W, H = 640, 360
BEIGE, BLUE, DARK = (178, 196, 208), (120, 80, 45), (25, 25, 28)       # BGR
HAND = (208, 150)                                                       # recorded hand box, px
R = [[0.0, 1.0], [1.0, 0.0]]          # image x = forward, image y = right (as on the desk camera)


def desk(seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = BEIGE
    # the room's light falls off toward the corners (a dim room under one lamp)
    yy, xx = np.mgrid[0:H, 0:W]
    fall = 1.0 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    img *= fall[..., None]
    img += rng.normal(0, 3, img.shape)
    return img


def done(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def scene():
    """The user's desk: a blue bag over the far third, the lap (blue jeans) past the near edge,
    a keyboard poking in from the left, a crumb in the middle."""
    img = desk()
    img[:, 430:] = BLUE                  # bag on the far side (image x = forward)
    img[:, :90] = (150, 110, 80)         # jeans beyond the desk edge
    img[:40, 180:300] = DARK             # keyboard corner on the left
    cv2.circle(img, (300, 250), 9, (60, 90, 80), -1)                     # a crumb
    return done(img)


class DetectTests(unittest.TestCase):
    def test_free_desk_between_the_bag_and_the_desk_edge(self):
        res = detect(scene(), (260, 200), HAND)
        x0, y0, x1, y1 = res["rect"]
        self.assertLessEqual(x1, 436)                                    # stops at the bag
        self.assertGreaterEqual(x1, 410)
        self.assertGreaterEqual(x0, 86)                                  # and at the desk's edge
        self.assertLessEqual(x0, 110)
        self.assertGreater(y1 - y0, 280)                                 # the crumb didn't cut it
        self.assertEqual(res["blocked"].get("right"), "blue")

    def test_clear_desk_is_the_whole_view(self):
        res = detect(done(desk(1)), (320, 180), HAND)
        x0, y0, x1, y1 = res["rect"]
        self.assertTrue(x0 <= 8 and y0 <= 8 and x1 >= W - 8 and y1 >= H - 8, res["rect"])
        self.assertEqual(res["blocked"], {})

    def test_hand_in_view_counts_as_desk(self):
        img = scene()
        pts = np.array([(180, 140), (330, 120), (370, 200), (330, 260), (200, 250)], float)
        cv2.fillConvexPoly(img, pts.astype(np.int32), (120, 150, 200))   # skin over the desk
        res = detect(img, (260, 190), HAND, hand_hull=pts)
        self.assertGreater(res["rect"][3] - res["rect"][1], 280)

    def test_fast(self):
        import time
        img = scene()
        t = time.perf_counter()
        for _ in range(5):
            detect(img, (260, 200), HAND)
        self.assertLess((time.perf_counter() - t) / 5, 0.05)

    def test_sides_for_the_user(self):
        self.assertEqual(desk_side(R, "right"), "far")                   # image +x is forward
        self.assertEqual(desk_side(R, "left"), "near")
        self.assertEqual(desk_side(R, "bottom"), "right")

    def test_agreement(self):
        self.assertTrue(same_room((100, 0, 430, 360), (104, 3, 426, 360)))
        self.assertFalse(same_room((100, 0, 430, 360), (100, 0, 600, 360)))


class FreeDeskWorkspaceTests(unittest.TestCase):
    def test_gain_fits_the_free_desk(self):
        args = (R, (W, H), HAND, 0.615, DEFAULTS["gain"], DEFAULTS["edge"], (1920, 1080))
        whole = fit_workspace(*args)
        free = fit_workspace(*args, bounds=(96, 0, 430, 360))
        self.assertLess(free["travel_mm"][1], whole["travel_mm"][1] * 0.6)   # forward room
        self.assertGreaterEqual(free["gain_scale"], whole["gain_scale"])
        self.assertEqual(free["raise_cm"], whole["raise_cm"])             # clear the desk first

    def test_glide_starts_at_the_bag_not_the_frame_edge(self):
        cfg = copy.deepcopy(DEFAULTS)
        prof = {"R": R, "px_per_m": 1626.0, "chirality": 1}
        out = VirtualOut()
        eng = Engine(cfg, prof, out)
        eng.set_bounds((96, 0, 430, 360))
        hand = np.array([[-100, -70], [100, -70], [100, 70], [-100, 70]] + [[0, 0]] * 17, float)

        def frame(cx, t):
            o = Obs(present=True, palm_px=np.array([cx, 180.0]), px_per_m=1626.0, flex8=0.9, flex12=1.2,
                    chirality=1, score=0.99, hand_m=0.08)
            o.img = hand + [cx, 180.0]
            o.frame_wh = (W, H)
            eng.update(o, t)

        t, cx = 0.0, 260.0
        for _ in range(30):                                             # settle in the middle
            t += 1 / 17
            frame(cx, t)
        while cx < 320:                                                 # up to the bag: fingertips 1 cm short
            t += 1 / 17
            cx += 8
            frame(cx, t)
        self.assertEqual(eng.edge_depth, 0.0)                           # near it is not a push
        while cx < 360:                                                 # push past its edge
            t += 1 / 17
            cx += 8
            frame(cx, t)
        self.assertGreater(eng.edge_depth, 0.3)                         # gliding up
        self.assertGreater(eng.edge_dir[1], 0.9)


class CrampedDeskTests(unittest.TestCase):
    """A bag leaving barely more than the hand: resting doesn't glide, pushing past the bag glides up."""

    def setUp(self):
        self.eng = Engine(copy.deepcopy(DEFAULTS), {"R": R, "px_per_m": 1626.0, "chirality": 1}, VirtualOut())
        self.eng.set_bounds((148, 24, 388, 360))          # the user's desk: 240 px forward for a 208 px hand
        self.t = 0.0

    def frame(self, cx):
        hand = np.array([[-104, -75], [104, -75], [104, 75], [-104, 75]] + [[0, 0]] * 17, float)
        o = Obs(present=True, palm_px=np.array([cx, 190.0]), px_per_m=1626.0, flex8=0.9, flex12=1.2,
                chirality=1, score=0.99, hand_m=0.08)
        o.img, o.frame_wh = hand + [cx, 190.0], (W, H)
        self.t += 1 / 17
        self.eng.update(o, self.t)

    def test_resting_in_the_middle_does_not_glide(self):
        for _ in range(40):
            self.frame(268.0)
        self.assertEqual(self.eng.edge_depth, 0.0)

    def test_pushing_past_the_bag_glides_up_but_the_desk_edge_never_glides(self):
        for _ in range(30):
            self.frame(268.0)
        for cx in range(270, 304, 3):                      # fingertips pushed past the bag's edge
            self.frame(float(cx))
        self.assertGreater(self.eng.edge_depth, 0.3)
        self.assertGreater(self.eng.edge_dir[1], 0.9)      # forward = up the screen
        for cx in range(300, 226, -4):                     # wrist back over the desk's edge, your side
            self.frame(float(cx))
        self.assertEqual(self.eng.edge_depth, 0.0)         # the forearm rests there: no glide


class AdviceTests(unittest.TestCase):
    def test_says_how_far_to_move_the_bag(self):
        from glide.surface import room_report
        rep = room_report(R, (148, 24, 388, 360), (W, H), HAND, 0.615, [["far", "blue"], ["near", "blue"]])
        self.assertEqual(rep["slide_cm"][1], 2.0)
        self.assertEqual(rep["advice"], [{"side": "far", "colour": "blue", "move_cm": 8}])


if __name__ == "__main__":
    unittest.main()


class DeskWatchTests(unittest.TestCase):
    """While the hand is in view the free desk may only grow (it once shrank to 2.9 x 2.4 cm)."""

    def make(self):
        from unittest import mock
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg = copy.deepcopy(DEFAULTS)
        svc.profile = {"R": R, "px_per_m": 1626.0, "chirality": 1, "frame_wh": [W, H], "hand_box_px": list(HAND)}
        svc.engine = Engine(svc.cfg, svc.profile, VirtualOut())
        svc.desk, svc.screen_wh, svc.toast = S.DeskWatch(), (1920, 1080), None
        svc.desk.palm = np.array([260.0, 180.0])
        svc.desk.applied = (96, 0, 430, 360)
        self.patch = mock.patch.object(S, "save_profile", lambda p: None)
        self.patch.start()
        return svc

    def tearDown(self):
        self.patch.stop()

    def res(self, rect):
        return {"rect": rect, "blocked": {"right": "blue"}, "free": 0.5}

    def test_seen_with_the_hand_it_never_shrinks(self):
        svc = self.make()
        for _ in range(3):
            svc._desk_result(self.res((200, 60, 430, 330)), hand_in_view=True)
        self.assertEqual(svc.desk.applied, (96, 0, 430, 360))

    def test_a_clear_look_can_shrink_it(self):
        svc = self.make()
        for _ in range(2):
            svc._desk_result(self.res((96, 0, 360, 360)), hand_in_view=False)
        self.assertEqual(svc.desk.applied, (96, 0, 360, 360))

    def test_seen_with_the_hand_it_can_grow(self):
        svc = self.make()
        for _ in range(2):
            svc._desk_result(self.res((60, 0, 520, 360)), hand_in_view=True)
        self.assertEqual(svc.desk.applied, (60, 0, 520, 360))


class StartupRefitTests(unittest.TestCase):
    """Starting with a saved free desk refits the room (a shadowed import once crashed this path)."""

    def test_use_profile_with_a_saved_free_desk(self):
        from unittest import mock
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg, svc.out, svc.screen_wh = copy.deepcopy(DEFAULTS), VirtualOut(), (1920, 1080)
        prof = {"R": R, "px_per_m": 1626.0, "chirality": 1, "frame_wh": [W, H], "hand_box_px": list(HAND),
                "travel_mm": [76.0, 213.0],
                "surface": {"rect": [116, 0, 420, 328], "blocked": [["far", "blue"]], "frame_wh": [W, H]}}
        with mock.patch.object(S, "save_profile", lambda p: None):
            svc._use_profile(prof)
        self.assertIsNotNone(svc.engine)
        self.assertLess(prof["travel_mm"][1], 213.0)                # refitted to the free desk
        self.assertEqual(svc.engine.bounds, (116.0, 0.0, 420.0, 328.0))

    def test_no_new_look_before_the_hand_is_seen(self):
        """A camera moved between sessions must not have its new view read as the desk."""
        from unittest import mock
        import glide.service as S
        svc = S.Service.__new__(S.Service)
        svc.cfg, svc.out, svc.screen_wh = copy.deepcopy(DEFAULTS), VirtualOut(), (1920, 1080)
        prof = {"R": R, "px_per_m": 1626.0, "chirality": 1, "frame_wh": [W, H], "hand_box_px": list(HAND),
                "surface": {"rect": [116, 0, 420, 328], "blocked": [["far", "blue"]], "frame_wh": [W, H],
                            "seed": [260.0, 180.0]}}
        with mock.patch.object(S, "save_profile", lambda p: None):
            svc._use_profile(prof)
        svc.profile, svc.cam = prof, None                           # no camera size: counts as calibrated
        with mock.patch.object(S.threading, "Thread") as th:
            svc._watch_desk(Obs(present=False), np.zeros((H, W, 3), np.uint8), 10.0)
        th.assert_not_called()                                      # nothing to look from yet
