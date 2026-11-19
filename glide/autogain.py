"""Learn the user's own pointer speed from where their clicks land (after AutoGain, Lee et al., CHI 2020).

A pointing movement is a fast ballistic part aimed at the target, then corrections (Meyer et al. 1988,
optimized submovement model). Every fast submovement aims at the spot the user finally clicks: if they
keep ending past it, the gain at the speeds they ran at is too high; if they keep stopping well short,
too low. Aiming normally stops a little short (a submovement covers ~93% of the way), so that is the
goal, not 100%.

The gain is a multiplier in hand-speed bins, applied on top of the fitted transfer function and
interpolated in log speed. Each submovement updates the bins in proportion to the cursor travel it made
at their speeds (a stroke also passes slower speeds as it speeds up and slows down). Updates are small,
geometric and clamped, so a few odd clicks can't swing it and the user's own adaptation keeps up.
"""
import math

import numpy as np

BINS = (40.0, 80.0, 140.0, 220.0, 350.0)          # hand speed, mm/s
LOG_BINS = [math.log(b) for b in BINS]


def bin_weights(v):
    x = math.log(max(v, BINS[0]))
    w = np.zeros(len(BINS))
    if x >= LOG_BINS[-1]:
        w[-1] = 1.0
        return w
    i = max(j for j in range(len(BINS)) if LOG_BINS[j] <= x)
    f = (x - LOG_BINS[i]) / (LOG_BINS[i + 1] - LOG_BINS[i])
    w[i], w[i + 1] = 1 - f, f
    return w


class AutoGain:
    def __init__(self, cfg, saved=None):
        self.cfg = cfg
        saved = saved or {}
        self.mult = np.array(saved.get("mult", [1.0] * len(BINS)), float)
        self.n = int(saved.get("n", 0))
        self._last = None                                 # cursor on the previous frame
        self.reset()

    def reset(self):
        self.subs = []                                    # finished submovements: (start, end, bin travel)
        self.cur = None                                   # the one in progress
        self.t_end = None

    def saved(self):
        return {"mult": [round(float(m), 4) for m in self.mult], "n": self.n}

    def factor(self, v):
        """Gain multiplier at hand speed v (mm/s)."""
        if not self.cfg.get("enabled", True):
            return 1.0
        return float(bin_weights(v) @ self.mult)

    def frame(self, t, cursor, speed, clean):
        k = self.cfg
        c = np.asarray(cursor, float)
        prev, self._last = self._last, c
        if not clean or prev is None:
            self.reset()
            return
        if self.cur is None:
            if speed > k["v_on"]:
                self.cur = {"start": prev, "travel": np.zeros(len(BINS)), "peak": speed}
            elif self.t_end is not None and t - self.t_end > k["max_correct_s"]:
                self.reset()                              # wandered off without clicking
            return
        self.cur["travel"] += bin_weights(speed) * float(np.hypot(*(c - prev)))
        self.cur["peak"] = max(self.cur["peak"], speed)
        if speed < max(k["end_share"] * self.cur["peak"], k["v_still"]):
            self.subs.append((self.cur["start"], c, self.cur["travel"]))
            self.cur, self.t_end = None, t

    def click(self, pos):
        """A press started at pos: learn from the submovements that aimed here. Returns the first
        one's 'need' (> 1: it fell short, more gain wanted), or None if nothing was learned."""
        k = self.cfg
        subs, self.subs, self.cur, self.t_end = self.subs, [], None, None
        if not k.get("enabled", True) or not subs:
            return None
        pos = np.asarray(pos, float)
        if float(np.hypot(*(pos - subs[0][0]))) < k["min_move_px"]:
            return None
        first = None
        for start, end, travel in subs:
            d = end - start
            P = float(np.hypot(*d))
            if P < k["min_primary_px"] or travel.sum() <= 0:
                continue
            along = float((pos - start) @ (d / P))
            if along <= 0:
                continue
            need = (along / P) * k["cover"]
            step = float(np.clip(need ** k["rate"], 1 - k["max_step"], 1 + k["max_step"]))
            self.mult = np.clip(self.mult * step ** (travel / travel.sum()), k["min_mult"], k["max_mult"])
            first = need if first is None else first
        if first is not None:
            self.n += 1
        return first
