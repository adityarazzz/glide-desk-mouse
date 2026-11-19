"""Signal filters for the pointer pipeline (TRD 4.7)."""
import math

import numpy as np


def ema_alpha(fc, dt):
    r = 2 * math.pi * fc * dt
    return r / (r + 1)


class OneEuro2D:
    """1-euro filter on a 2-D point; one shared cutoff driven by speed magnitude [R3]."""

    def __init__(self, fc_min=1.2, beta=0.06, d_cutoff=1.0, fc_max=None):
        self.fc_min, self.beta, self.d_cutoff, self.fc_max = fc_min, beta, d_cutoff, fc_max
        self.reset()

    def reset(self):
        self.x = None
        self.dx = np.zeros(2)
        self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros(2), t
            return self.x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += ema_alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        fc = self.fc_min + self.beta * float(np.hypot(*self.dx))
        if self.fc_max:
            fc = min(fc, self.fc_max)
        self.x += ema_alpha(fc, dt) * (x - self.x)
        return self.x.copy()


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(min(z, 50), -50)))


def gain_curve(v, g):
    """Logistic transfer function: precise when slow, fast flicks cross the screen.

    Anchored at zero speed: a plain logistic still leaks 2-3% of g_max into a resting hand, which at
    the 3x workspace gain was a third of the gain while holding still (static jitter 0.65 -> 0.50 px).
    """
    s0 = _sigmoid(-g["v_mid_mm_s"] / g["width"])
    rise = max(_sigmoid((v - g["v_mid_mm_s"]) / g["width"]) - s0, 0.0) / (1 - s0)
    return g["g_min"] + (g["g_max"] - g["g_min"]) * rise
