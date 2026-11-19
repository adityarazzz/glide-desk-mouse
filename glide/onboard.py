"""First-run flow for the app, as frame-driven state machines (no windows of their own).

camera -> placement -> calibrate -> learn -> practice, one decision per screen. The checks, the
calibrator and the lessons are the measured ones from glide.setup / glide.calibrate; the app
only draws them.
"""
import numpy as np

from .calibrate import Calibrator, orientation
from .engine import Engine
from .replay import VirtualOut
from .setup import LESSONS, FIST_SCROLL, placement_checks

STEPS = ["welcome", "camera", "placement", "calibrate", "learn", "practice", "done"]


class Placement:
    def __init__(self, cfg, screen_wh):
        self.cfg, self.screen_wh = cfg, screen_wh
        self.window, self.checks, self.score = [], [], 0
        self.R = None

    def feed(self, obs, frame, fps):
        self.window = (self.window + [obs])[-45:]
        self.checks = placement_checks(frame, self.window, fps, self.cfg, self.screen_wh)
        self.score = int(100 * sum(bool(ok) for _, ok, _ in self.checks) / len(self.checks))
        seen = [o for o in self.window if o.present]
        if len(seen) >= 20:                          # live orientation for the turned view
            right, fwd = orientation(seen[-20:], self.cfg["hand"]["dominant"])
            self.R = [right.tolist(), fwd.tolist()]

    def state(self):
        return {"score": self.score,
                "checks": [{"name": n, "ok": bool(ok), "tip": "" if ok else tip} for n, ok, tip in self.checks]}


class Calibrate:
    def __init__(self, cfg, screen_wh):
        self.cal = Calibrator(cfg, screen_wh)
        self.profile, self.t, self.beat = None, 0.0, None
        self.events = []

    def feed(self, obs, t):
        self.events, self.t = [], t
        n = len(self.cal.cues)
        done = self.cal.feed(obs, t)
        k = self.cal.countdown(t)
        if k and k != self.beat:
            self.events.append("beat")
        if len(self.cal.cues) != n:
            self.events.append("go")
        self.beat = k
        if done and self.profile is None:
            self.profile = done
            self.events.append("step_done")

    def skip(self):
        self.cal.skip()

    def state(self):
        c = self.cal
        go_age = self.t - c.cues[-1][0] if c.cues and c.step == "clicks" and c.phase == "sample" else None
        return {"step": c.step or "done", "prompt": c.prompt, "progress": min(1.0, c.progress()),
                "countdown": c.countdown(self.t), "go": c.flash(self.t), "waiting": c.waiting(),
                "message": c.message, "cues": len(c.cues), "go_age": go_age, "done": self.profile is not None}


class Learn:
    """The gesture lessons against a virtual screen: nothing is injected yet."""

    def __init__(self, cfg, profile, screen_wh):
        self.out = VirtualOut(*screen_wh)
        self.eng = Engine(cfg, profile, self.out)
        self.lessons = LESSONS[:-1] + [FIST_SCROLL if cfg["scroll"]["gesture"] == "fist" else LESSONS[-1]]
        self.i, self.passed_at, self.seen = 0, None, self.eng.event_seq
        self.done = [False] * len(self.lessons)
        self.last = np.array(self.out.cursor(), float)
        self.events = []
        self._reset()

    def _reset(self):
        self.stats = {"travel": 0.0, "edge": 0, "left click": 0, "right click": 0, "drag": 0, "scroll": 0, "lift": 0}

    @property
    def finished(self):
        return self.i >= len(self.lessons)

    def skip(self):
        if not self.finished:
            self.i, self.passed_at = self.i + 1, None
            self._reset()

    def feed(self, obs, t):
        self.events = []
        self.out.t = t
        self.eng.update(obs, t)
        pos = np.array(self.out.cursor(), float)
        self.stats["travel"] += float(np.hypot(*(pos - self.last)))
        self.stats["edge"] += self.eng.edge_depth > 0
        self.last = pos
        if self.eng.event_seq != self.seen:
            self.seen = self.eng.event_seq
            if self.eng.last_event in self.stats:
                self.stats[self.eng.last_event] += 1
            self.events.append("tick")
        if self.finished:
            return
        if self.passed_at is None and self.lessons[self.i][2](self.stats):
            self.done[self.i], self.passed_at = True, t
            self.events.append("ok")
        if self.passed_at is not None and t - self.passed_at > 0.9:
            self.i, self.passed_at = self.i + 1, None
            self._reset()

    def state(self):
        st = self.eng.status()
        x, y = self.out.cursor()
        return {"i": self.i, "finished": self.finished,
                "lessons": [{"name": n, "how": how, "demo": d, "done": ok}
                            for (n, how, _, d), ok in zip(self.lessons, self.done)],
                "cursor": [x / self.out.w, y / self.out.h], "state": st["state"], "hold": st["hold"],
                "passed": self.passed_at is not None}
