"""Guided personalisation (PRD J1): rest -> clicks. Produces profile.json.

rest    hand flat and still: desk orientation, image scale, chirality
clicks  5 cued pinches: how deep this user's pinch dips the thumb-to-finger gap, which sets the
        personal press / release thresholds (the 10-tap personalisation of TRD 4.6, adapted)
"""
import math
import time

import numpy as np

from .features import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST
from .workspace import fit_workspace, hand_box

PROMPTS = {
    "rest": "Rest your hand flat on the desk, fingers pointing away from you. Hold still.",
    "clicks": "On GO: touch THUMB tip to MIDDLE fingertip, then open. SPACE = skip",
}
REST_FRAMES = 40
CLICKS = 5
# self-paced cues (see glide.coach): wait for an open hand, count 3-2-1, GO, sample the pinch
BEAT_S, BEATS, SAMPLE_S = 0.5, 3, 1.5
READY_S, READY_TIMEOUT_S = 0.3, 4.0


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def orientation(obs_list, dominant="right"):
    """Desk axes in image px: (right, forward) unit vectors from resting hand poses."""
    fwd = _unit(np.mean([_unit(o.img[MIDDLE_MCP] - o.img[WRIST]) for o in obs_list], axis=0))
    across = np.mean([_unit(o.img[PINKY_MCP] - o.img[INDEX_MCP]) for o in obs_list], axis=0)
    # palm down, the pinky is on the user's right for a right hand and on the left for a left hand
    if dominant == "left":
        across = -across
    right = np.array([-fwd[1], fwd[0]])
    if right @ across < 0:
        right = -right
    return right, fwd


def rest_profile(obs_list, cfg, screen_wh=(1920, 1080)):
    right, fwd = orientation(obs_list, cfg["hand"]["dominant"])
    s = float(np.median([o.px_per_m for o in obs_list]))
    p = {
        "R": [right.tolist(), fwd.tolist()],
        "px_per_m": s,
        "chirality": int(np.sign(np.sum([o.chirality for o in obs_list])) or 1),
        "rest_flex": [float(np.median([o.flex8 for o in obs_list])),
                      float(np.median([o.flex12 for o in obs_list]))],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    frame_wh = obs_list[0].frame_wh
    if frame_wh is not None:
        box = hand_box(obs_list)
        p.update(frame_wh=list(frame_wh), hand_box_px=box,
                 **fit_workspace(p["R"], frame_wh, box, 1000.0 / s, cfg["gain"], cfg["edge"], screen_wh))
    return p


def pinch_duration(samples):
    """Seconds the gap stayed below halfway between open (1.0) and its deepest point."""
    lo = min(r for _, r in samples)
    below = [t for t, r in samples if r < (1.0 + lo) / 2]
    return (max(below) - min(below)) if len(below) > 1 else 0.1


def thresholds_from_dips(dips, cfg):
    """Press / release ratios placed inside this user's measured dip depth."""
    depth = 1.0 - float(np.median(dips))
    k = cfg["click"]
    return {"enter": float(np.clip(1 - 0.55 * depth, 0.62, k["enter"] + 0.05)),
            "deep": float(np.clip(1 - 0.85 * depth, 0.45, k["deep"])),
            "exit": float(np.clip(1 - 0.30 * depth, 0.80, 0.92))}


class Calibrator:
    def __init__(self, cfg, screen_wh=(1920, 1080)):
        self.cfg = cfg
        self.screen_wh = screen_wh
        self.steps = ["rest", "clicks"]
        self.i = 0
        self.rest = []
        self.prev_palm = None
        self.profile = {}
        self.message = ""
        self.cues = []
        self.phase, self.t_phase, self.ready_s, self.prev_t = "ready", None, 0.0, None

    @property
    def step(self):
        return self.steps[self.i] if self.i < len(self.steps) else None

    @property
    def prompt(self):
        return PROMPTS.get(self.step, "")

    def progress(self):
        return len(self.rest) / REST_FRAMES if self.step == "rest" else len(self.cues) / CLICKS

    def flash(self, t):
        """True right after a cue, for the HUD's GO prompt."""
        return self.step == "clicks" and bool(self.cues) and t - self.cues[-1][0] < 0.5

    def countdown(self, t):
        """3, 2, 1 while counting in the next pinch, else None."""
        if self.step != "clicks" or self.phase != "count":
            return None
        return BEATS - int((t - self.t_phase) / BEAT_S)

    def waiting(self):
        return self.step == "clicks" and self.phase == "ready"

    def skip(self):
        if self.step == "clicks":
            self._finish_clicks(skipped=True)

    def feed(self, obs, t):
        """Returns the finished profile once all steps are done, else None."""
        if self.step == "rest":
            self._feed_rest(obs)
        elif self.step == "clicks":
            self._feed_clicks(obs, t)
        return self.profile if self.step is None else None

    def _feed_rest(self, obs):
        if not obs.present:
            self.prev_palm = None
            return
        if self.prev_palm is not None and np.linalg.norm(obs.palm_px - self.prev_palm) > 3.0:
            self.rest.clear()                       # moved: start the still window again
        self.prev_palm = obs.palm_px
        self.rest.append(obs)
        if len(self.rest) >= REST_FRAMES:
            self.profile.update(rest_profile(self.rest, self.cfg, self.screen_wh))
            tr = self.profile.get("travel_mm")
            self.message = (f"Room to slide: {tr[0] / 10:.0f} x {tr[1] / 10:.0f} cm, cursor reach x"
                            f"{self.profile['gain_scale']:.1f}" if tr else "Orientation locked.")
            self.i += 1

    def _feed_clicks(self, obs, t):
        dt = 0.0 if self.prev_t is None else min(0.2, t - self.prev_t)
        self.prev_t = t
        if self.t_phase is None:
            self.t_phase = t
        b8, b12 = self.profile["rest_flex"]
        ratio = min(obs.flex8 / b8, obs.flex12 / b12) if obs.present else None
        if self.phase == "ready":                   # hand back and open before the next cue
            self.ready_s = self.ready_s + dt if ratio is not None and ratio > 0.88 else 0.0
            if self.ready_s >= READY_S or (ratio is not None and t - self.t_phase > READY_TIMEOUT_S):
                self.phase, self.t_phase = "count", t
        elif self.phase == "count":
            if ratio is None:
                self.phase, self.t_phase, self.ready_s = "ready", t, 0.0
            elif t - self.t_phase >= BEATS * BEAT_S:
                self.cues.append((t, []))
                self.phase, self.t_phase = "sample", t
        else:
            if ratio is not None:
                self.cues[-1][1].append((t, ratio))
            if t - self.t_phase >= SAMPLE_S:
                if len(self.cues) >= CLICKS:
                    self._finish_clicks()
                else:
                    self.phase, self.t_phase, self.ready_s = "ready", t, 0.0

    def _finish_clicks(self, skipped=False):
        dips = [min(r for _, r in v) for _, v in self.cues if v]
        if not skipped and len(dips) >= 3 and np.median(dips) < 0.8:
            self.profile["click"] = thresholds_from_dips(dips, self.cfg)
            # right-click hold: comfortably longer than this user's own natural pinch
            durs = [pinch_duration(v) for _, v in self.cues if v]
            self.profile["click"]["long_ms"] = float(np.clip(2.2 * 1000 * max(durs), 800, 2000))
            self.message = (f"Pinch depth {1 - np.median(dips):.0%}: press below "
                            f"{self.profile['click']['enter']:.2f}, release above {self.profile['click']['exit']:.2f}")
        else:
            self.message = "Using default click thresholds."
        self.i += 1


def describe(profile):
    fwd = profile["R"][1]
    k = profile.get("click", {})
    tr = profile.get("travel_mm")
    room = f", room {tr[0] / 10:.0f}x{tr[1] / 10:.0f} cm, reach x{profile['gain_scale']:.1f}" if tr else ""
    return (f"forward axis {math.degrees(math.atan2(fwd[1], fwd[0])):.0f} deg in image, "
            f"scale {profile['px_per_m']:.0f} px/m, click enter {k.get('enter', 'default')}{room}")
