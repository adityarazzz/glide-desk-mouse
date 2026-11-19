"""Self-paced, coached recording sessions: what to do is shown, not raced.

The first recorder ran each phase on a fixed timer and flashed NOW every 1.8 s. People were still
reading when the clock started, couldn't tell which movement was meant, and missed cues. Every
rule below fixes one of those, following practice from gesture-dataset collection and usability:

- show, don't tell: every step has an animated demo hand (glide.demo), viewed like your own hand
- one short instruction per step; the step never starts while you could still be reading it
  (minimum intro time from its word count at ~240 wpm, then you start it: rest your hand still
  for a moment, or press SPACE)
- user control: P pause, R redo the step, S skip it, ESC stop and keep the finished steps
- a predictable rhythm instead of a surprise flash: 3-2-1 countdown beats with sound, then GO.
  A warned stimulus gives faster, steadier reactions than an unwarned one (foreperiod studies)
- each trial waits until your hand is back, open and steady, so trials never pile up
- one practice trial per gesture, not scored, with a hint if it wasn't seen
- feedback after every trial and a live check during movement steps ("scroll pose seen"), so
  the recording also tells us when a pose is impossible for the camera to see
- the clock stops while your hand is out of view

Ground truth for glide.analyze is unchanged: the step label on each frame and the cue times.
Frames spent reading, waiting or practising are labelled "wait" / "practice", which the
analyzer scores neither as clicks nor as quiet time.
"""
import collections
import math
from dataclasses import dataclass

import numpy as np

from .calibrate import orientation

BEAT_S = 0.6           # countdown beat
BEATS = 3
RESPOND_S = 1.9        # recorded after GO; >= the analyzer's 1.6 s cue window
RESULT_S = 0.8
READY_S = 0.4          # open + steady this long before a trial's countdown
READY_TIMEOUT_S = 5.0  # then start anyway: some people never hold fully still
DWELL_S = 1.2          # resting still this long starts a step hands-free
# palm speed in hand lengths / s, from the net movement over SPEED_S. Frame-to-frame differences
# are mostly landmark noise: a resting hand measured 0.23 that way, 0.02 over the window.
SPEED_S = 0.25
STILL = 0.12
STEADY = 0.4
WPM = 240


@dataclass
class Step:
    label: str
    title: str
    how: str
    kind: str = "free"        # still | free | cued
    seconds: float = 6.0      # still / free: seconds of good recording
    reps: int = 5             # cued trials
    check: str = None         # live check for free steps (see CHECKS)
    tip: str = ""             # shown when a trial or the live check fails
    demo: str = None          # glide.demo.DEMOS key, default = label
    practice: bool = True


CHECKS = {   # name: (test, live text when it holds, when it doesn't)
    "still": (lambda c, o: c.speed < STILL, "Nice and still", "Hold still"),
    "moving": (lambda c, o: c.speed > 0.3, "Moving", "Keep sliding"),
    "fast": (lambda c, o: c.speed > 2.0, "Fast", "Faster flicks"),
    "two": (lambda c, o: o.two_finger, "Scroll pose seen", "Scroll pose not seen"),
    "side": (lambda c, o: o.palm_sin < 0.45, "On its side", "Tilt further onto the side"),
    "turned": (lambda c, o: 25 <= abs(c.turn_deg(o)) <= 80, "Turned", "Turn the hand ~45 degrees"),
    "curled": (lambda c, o: max(o.ext) < 1.65 and not o.fist, "Curled grip", "Curl the fingers a bit more"),
}


def gesture_of(label):
    return {"pinch_idx": "pinch8", "tap": "tap", "lift": "lift", "reposition": "lift", "fist": "fist"}.get(label, "pinch12")


class Coach:
    def __init__(self, steps, R=None, hand="right"):
        self.steps, self.hand = steps, hand
        self.i = 0
        self.rows, self.cues, self.summary = [], [], []
        self.R, self.yaw0 = R, None
        self.paused = False
        self.events, self.t, self.hand_px = [], 0.0, 100.0
        self.prev = None                     # t of the previous frame
        self.speed = 0.0
        self.track = collections.deque()     # (t, palm_px) over SPEED_S
        self.recent = collections.deque()    # (t, flex8, flex12) for the open-hand reference
        self.live = collections.deque(maxlen=12)
        self._begin_step(None)

    # ---- public ------------------------------------------------------------------------

    @property
    def step(self):
        return self.steps[self.i] if self.i < len(self.steps) else None

    @property
    def done(self):
        return self.i >= len(self.steps)

    def go(self):
        """SPACE: start the step now / skip the wait for an open hand."""
        if self.state == "intro":
            self.forced = True
        elif self.state == "ready":
            self.ready_forced = True

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused and self.state in ("count", "respond", "result"):
            self._enter("ready", self.t)             # resume with a fresh trial

    def redo(self):
        if not self.done:
            del self.rows[self.row0:]
            del self.cues[self.cue0:]
            self._begin_step(self.t)

    def skip(self):
        if not self.done:
            self.summary.append({"label": self.step.label, "skipped": True})
            del self.rows[self.row0:]
            del self.cues[self.cue0:]
            self._next_step()

    def feed(self, obs, t, row):
        """One camera frame. row: the frame's data; the coach labels and keeps it."""
        self.events = []
        self.t = t
        self._measure(obs, t)
        label = "wait"
        if not self.done and not self.paused:
            label = getattr(self, "_" + self.state)(obs, t)
        row["label"] = label
        self.rows.append(row)
        return label

    # ---- progress the UI reads -----------------------------------------------------------

    def countdown(self):
        """3, 2, 1 during the countdown, else None."""
        if self.state != "count":
            return None
        return BEATS - int((self.t - self.t_state) / BEAT_S)

    def fill(self):
        """0..1 progress of whatever is being waited for (dwell to start, time recorded)."""
        if self.state == "intro":
            return min(1.0, self.dwell / DWELL_S)
        if self.state == "run":
            return min(1.0, self.good_s / self.step.seconds)
        if self.state == "respond":
            return min(1.0, (self.t - self.t_state) / RESPOND_S)
        return 0.0

    def intro_left(self):
        if self.state != "intro":
            return 0.0
        return self.min_intro if self.t_state is None else max(0.0, self.t_state + self.min_intro - self.t)

    def live_ok(self):
        """Live check result over the last few frames, or None when the step has no check."""
        if not self.live:
            return None
        return sum(self.live) > len(self.live) / 2

    def since_go(self):
        return self.t - self.t_go if self.t_go is not None else None

    # ---- states ----------------------------------------------------------------------------

    def _begin_step(self, t):
        self.row0, self.cue0 = len(self.rows), len(self.cues)
        self.trial, self.practice_left, self.seen = 0, 0, 0
        self.good_s, self.checked, self.passed = 0.0, 0, 0
        self.results, self.stills = [], []
        self.live.clear()
        self.t_go = None
        s = self.step
        if s is not None:
            self.practice_left = 2 if (s.kind == "cued" and s.practice) else 0
            words = len((s.title + " " + s.how).split())
            self.min_intro = min(6.0, 1.5 + words * 60 / WPM)
        self._enter("intro", t)

    def _enter(self, state, t):
        self.state, self.t_state = state, t
        self.dwell, self.forced, self.ready_s, self.ready_forced, self.beats = 0.0, False, 0.0, False, 0
        if state == "intro":
            self.t_state = None                     # set on the first frame of the step
        if state == "respond":
            self.t_go = t

    def _intro(self, obs, t):
        if self.t_state is None:
            self.t_state = t
        if obs.present and self.speed < STEADY:
            self.dwell += self.dt
        else:
            self.dwell = 0.0
        if self.forced or (t - self.t_state >= self.min_intro and self.dwell >= DWELL_S):
            self.events.append("start")
            self._enter("ready" if self.step.kind == "cued" else "count", t)
        return "wait"

    def _ready(self, obs, t):
        ok = obs.present and self.speed < STEADY and self._open(obs)
        self.ready_s = self.ready_s + self.dt if ok else 0.0
        waited = t - self.t_state
        if self.ready_forced or self.ready_s >= READY_S or (obs.present and waited > READY_TIMEOUT_S):
            self._enter("count", t)
        return "wait"

    def _count(self, obs, t):
        if not obs.present:                        # hand gone: wait for it again
            self._enter("ready", t)
            return "wait"
        k = int((t - self.t_state) / BEAT_S)
        if k >= BEATS:
            self.events.append("go")
            if self.step.kind == "cued":
                self.base = obs
                self.hit = False
                self.cues.append({"t": t, "label": self._trial_label()})
                self._enter("respond", t)
                return self._trial_label()
            self._enter("run", t)
            return self.step.label
        if self.beats < k + 1:
            self.beats = k + 1
            self.events.append("beat")
        return "wait"

    def _respond(self, obs, t):
        if obs.present and not self.hit:
            self.hit = self._detect(obs)
        if t - self.t_state >= RESPOND_S:
            practice = self.practice_left > 0
            self.results.append((practice, self.hit))
            self.events.append("ok" if self.hit else "miss")
            if practice:
                # one practice trial, or two when the first one wasn't seen
                self.practice_left = 0 if self.hit else self.practice_left - 1
            else:
                self.trial += 1
                self.seen += self.hit
            self._enter("result", t)
        return self._trial_label()

    def _result(self, obs, t):
        if t - self.t_state >= RESULT_S:
            if self.trial >= self.step.reps:
                self._finish_step()
            else:
                self._enter("ready", t)
        return "wait"

    def _run(self, obs, t):
        s = self.step
        if not obs.present:
            return s.label
        test = CHECKS.get(s.check)
        if test and t - self.t_state > 0.8:         # give the hand time to get into the pose
            ok = bool(test[0](self, obs))
            self.live.append(ok)
            self.checked += 1
            self.passed += ok
        if s.kind == "still":
            if self.speed < STILL:
                self.good_s += self.dt
                self.stills.append(obs)
            else:
                return "wait"                       # moved: not part of the still recording
        else:
            self.good_s += self.dt
        if self.good_s >= s.seconds:
            self._finish_step()
        return s.label

    # ---- helpers ---------------------------------------------------------------------------

    def _trial_label(self):
        return "practice" if self.practice_left > 0 else self.step.label

    def _finish_step(self):
        s = self.step
        rec = {"label": s.label}
        if s.kind == "cued":
            rec.update(trials=self.trial, seen=self.seen)
        if s.check:
            rec["check"] = s.check
            rec["check_frac"] = round(self.passed / self.checked, 3) if self.checked else None
        if s.kind == "still" and self.stills:
            right, fwd = orientation(self.stills, self.hand)   # fresh: the camera may have moved
            self.R = [right.tolist(), fwd.tolist()]
            if self.yaw0 is None:
                self.yaw0 = float(np.median([o.yaw for o in self.stills]))
        self.summary.append(rec)
        self.rows.append({"step": rec})
        self.events.append("step_done")
        self._next_step()

    def _next_step(self):
        self.i += 1
        self._begin_step(self.t)

    def _measure(self, obs, t):
        self.dt = 0.0 if self.prev is None else min(0.2, max(0.0, t - self.prev))
        self.prev = t
        if not obs.present:
            self.speed = 0.0
            self.track.clear()
            return
        hand = max(float(np.hypot(*(obs.img[9] - obs.img[0]))), 1.0)
        self.track.append((t, obs.palm_px.copy()))
        while t - self.track[0][0] > SPEED_S:
            self.track.popleft()
        t0, p0 = self.track[0]
        if t - t0 >= 0.1:
            self.speed = float(np.hypot(*(obs.palm_px - p0))) / hand / (t - t0)
        self.hand_px = hand
        self.recent.append((t, obs.flex8, obs.flex12))
        while self.recent and t - self.recent[0][0] > 3.0:
            self.recent.popleft()

    def _open(self, obs):
        if obs.fist:
            return False
        m8 = max(r[1] for r in self.recent)
        m12 = max(r[2] for r in self.recent)
        return obs.flex8 > 0.85 * m8 and obs.flex12 > 0.85 * m12

    def _detect(self, obs):
        g, b = gesture_of(self.step.label), self.base
        if g == "pinch8":
            return obs.flex8 < 0.72 * b.flex8
        if g == "pinch12":
            return obs.flex12 < 0.72 * b.flex12
        if g == "tap":
            d = (obs.img[8] - obs.palm_px) - (b.img[8] - b.palm_px)
            return float(np.hypot(*d)) / self.hand_px > 0.12 or obs.flex8 < 0.8 * b.flex8
        if g == "lift":                              # the same palm-size signal the engine uses
            return obs.palm_size > 1.12 * b.palm_size
        if g == "fist":
            return obs.fist
        return False

    def turn_deg(self, obs):
        if self.yaw0 is None or obs.yaw is None:
            return 0.0
        d = obs.yaw - self.yaw0
        return math.degrees((d + math.pi) % (2 * math.pi) - math.pi)
