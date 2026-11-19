"""Deterministic replay of recorded sessions (test plan 6.1 "Replay" level).

Session files (logs/*.jsonl) hold timestamped landmarks only, never images (NFR-06):
  {"t", "label", "wh", "img", "world", "score"}   one per frame ("img" absent = no hand)
  {"cue": {"t", "label"}}                         when the recorder prompted a gesture
  {"meta": {...}} / {"step": {...}}               coached sessions: script and per-step summary
"""
import json
from pathlib import Path

import numpy as np

from .calibrate import rest_profile
from .engine import Engine
from .features import Obs, measure


class VirtualOut:
    """Output adapter that records instead of injecting; same interface as WinOutput."""

    def __init__(self, width=1920, height=1080):
        self.w, self.h = width, height
        self.pos = [width // 2, height // 2]
        self.held = set()
        self.events = []      # (t, kind, detail, pos)
        self.t = 0.0

    def screen_width(self):
        return self.w

    def cursor(self):
        return tuple(self.pos)

    def move_to(self, x, y):
        self.pos = [int(min(max(x, 0), self.w - 1)), int(min(max(y, 0), self.h - 1))]

    def move_by(self, dx, dy):
        self.move_to(self.pos[0] + dx, self.pos[1] + dy)

    def button(self, name, down):
        if down != (name in self.held):
            (self.held.add if down else self.held.discard)(name)
            self.events.append((self.t, name, down, tuple(self.pos)))

    def click(self, name):
        self.button(name, True)
        self.button(name, False)

    def scroll(self, v, h=0):
        self.events.append((self.t, "scroll", (v, h), tuple(self.pos)))

    def release_all(self):
        for n in list(self.held):
            self.button(n, False)


def load_session(path):
    frames, cues = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "cue" in r:
            cues.append(r["cue"])
        elif "t" not in r:                          # session meta / step summaries
            continue
        elif "img" in r:
            frames.append((r["t"], r["label"], measure(r["img"], r["world"], r.get("score", 1.0), r.get("wh"))))
        else:
            frames.append((r["t"], r["label"], Obs(present=False)))
    return retime(frames), cues


def load_meta(path):
    """Coached sessions: {"script", "summary": [per-step results]}, else None."""
    with open(path, encoding="utf-8") as f:
        first = json.loads(f.readline() or "{}")
    return first.get("meta")


def retime(frames):
    """Sessions recorded before driver timestamps were used carry bursty arrival times; every
    frame was processed, so a uniform clock over the same span is the faithful reconstruction."""
    t = np.array([f[0] for f in frames])
    if len(t) < 3 or np.mean(np.diff(t) < 0.008) < 0.2:
        return frames
    u = np.linspace(t[0], t[-1], len(t))
    return [(float(ui), label, obs) for ui, (_, label, obs) in zip(u, frames)]


def session_profile(frames, cfg, labels=("rest", "hover_rest", "move_slow", "rot_left", "side")):
    """Reference pose for replay: the first phase with enough flat (non-fist) frames. Users don't
    always hold the prompted pose (one 'rest' phase was 79% fist), so fall back down the list."""
    for label in labels:
        flat = [o for _, l, o in frames if l == label and o.present and not o.fist]
        if len(flat) >= 40:
            return rest_profile(flat, cfg)
    return rest_profile([o for _, _, o in frames if o.present][:60], cfg)


def replay(frames, cfg, profile, typing_labels=()):
    """Run the engine over a session. Returns (out, trace) with trace = [(t, label, state, cursor)]."""
    out = VirtualOut()
    eng = Engine(cfg, profile, out)
    trace = []
    for t, label, obs in frames:
        out.t = t
        eng.update(obs, t, typing=label in typing_labels)
        trace.append((t, label, eng.state, out.cursor()))
    return out, trace
