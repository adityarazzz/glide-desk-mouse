"""Live debug view (FR-09): camera, landmarks, state, signals and latency."""
import collections

import cv2
import numpy as np

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
STATE_COLORS = {"TRACK": (80, 200, 80), "PINCH": (255, 255, 255), "DRAG": (255, 140, 60),
                "HOLD": (0, 176, 255), "SCROLL": (26, 140, 255), "CLUTCH": (170, 170, 170),
                "PAUSED": (60, 60, 220), "IDLE": (120, 120, 120)}
HELP = "q quit  p pause  c calibrate  +/- speed  [ ] rotate"


def text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, label, val, lo, hi, marks=(), color=(255, 255, 255)):
    cv2.rectangle(img, (x, y), (x + w, y + 8), (60, 60, 60), -1)
    f = float(np.clip((val - lo) / (hi - lo), 0, 1))
    cv2.rectangle(img, (x, y), (x + int(f * w), y + 8), color, -1)
    for m in marks:
        mx = x + int(np.clip((m - lo) / (hi - lo), 0, 1) * w)
        cv2.line(img, (mx, y - 2), (mx, y + 10), (0, 255, 255), 1)
    text(img, f"{label} {val:.2f}", (x + w + 6, y + 8), 0.4)


def draw_hand(img, obs, color=(0, 200, 255)):
    if not obs.present:
        return
    pts = obs.img.astype(int)
    for a, b in BONES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_edge_band(img, cfg, active=False):
    """Inner outline of the RubberEdge band: reaching past it with the hand keeps the cursor gliding."""
    if not cfg["edge"]["enabled"]:
        return
    h, w = img.shape[:2]
    z = int(cfg["edge"]["zone"] * h)
    cv2.rectangle(img, (z, z), (w - z, h - z), (0, 200, 255) if active else (110, 110, 110), 1, cv2.LINE_AA)


class Hud:
    def __init__(self):
        self.trace = collections.deque(maxlen=120)

    def draw(self, frame, obs, engine=None, cal=None, fps=0.0, lat_ms=0.0, paused=False, typing=False, t=0.0):
        img = frame.copy()
        h, w = img.shape[:2]
        st = engine.status() if engine else {}
        if engine:
            draw_edge_band(img, engine.cfg, st["edge"] > 0)
        draw_hand(img, obs)
        if obs.present and engine:
            c = obs.anchor(engine.cfg["pointer"]["anchor"]).astype(int)
            cv2.circle(img, tuple(c), 7, STATE_COLORS.get(st.get("state"), (255, 255, 255)), -1)
            r, f = engine.R[0], engine.R[1]       # desk axes: green = forward (screen up), red = right
            cv2.arrowedLine(img, tuple(c), tuple((c + f * 60).astype(int)), (0, 255, 0), 2, tipLength=0.25)
            cv2.arrowedLine(img, tuple(c), tuple((c + r * 60).astype(int)), (0, 0, 255), 2, tipLength=0.25)

        state = "PAUSED" if paused else st.get("state", "CALIBRATING")
        text(img, state, (10, 28), 0.8, STATE_COLORS.get(state, (255, 255, 0)), 2)
        text(img, f"{fps:4.1f} fps  {lat_ms:4.0f} ms capture->out", (w - 270, 22), 0.5)
        if typing:
            text(img, "typing guard", (w - 130, 44), 0.5, (0, 200, 255))
        if fps and fps < 20:
            text(img, "Low fps: add light on the desk", (w - 260, 66), 0.5, (0, 140, 255))

        if engine and obs.present:
            y = h - 88
            bar(img, 10, y, 150, "click dip", st["ratio"], 0.3, 1.1, (st["enter"], st["exit"]), (255, 255, 255))
            bar(img, 10, y + 16, 150, "lift", st["scale"], 0.8, 1.4,
                (engine.cfg["clutch"]["lift_on"],), (200, 200, 200))
            glide = f"   EDGE GLIDE {st['edge']:.0%}" if st["edge"] > 0 else ""
            text(img, f"fist {'YES' if st['fist'] else 'no'}   speed {st['speed']:4.0f} mm/s   {st['event']}{glide}",
                 (10, y + 44), 0.45)
            self.trace.append(st["ratio"])
            self._plot(img, w - 200, h - 88, 190, 60, st["enter"], st["exit"])

        if cal is not None:
            cv2.rectangle(img, (0, 40), (w, 100), (0, 0, 0), -1)
            text(img, f"Calibration {cal.i + 1}/{len(cal.steps)}", (10, 60), 0.55, (0, 255, 255))
            text(img, cal.prompt, (10, 82), 0.45)
            cv2.rectangle(img, (10, 90), (10 + int(min(cal.progress(), 1) * (w - 20)), 96), (0, 255, 255), -1)
            if cal.message:
                text(img, cal.message, (10, 118), 0.45, (0, 255, 0))
            k = cal.countdown(t)
            if k:
                text(img, str(k), (w // 2 - 20, h // 2 + 20), 2.5, (255, 255, 0), 5)
            elif cal.flash(t):
                text(img, "GO", (w // 2 - 50, h // 2 + 20), 2.5, (0, 255, 0), 5)
            elif cal.waiting():
                text(img, "open your hand", (w // 2 - 90, h // 2 + 20), 0.8, (200, 200, 200), 2)
        text(img, HELP, (10, h - 6), 0.38, (200, 200, 200))
        return img

    def _plot(self, img, x, y, w, h, enter, exit_):
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
        lo, hi = 0.3, 1.1
        to_y = lambda v: y + h - int(np.clip((v - lo) / (hi - lo), 0, 1) * h)
        cv2.line(img, (x, to_y(enter)), (x + w, to_y(enter)), (0, 255, 255), 1)
        cv2.line(img, (x, to_y(exit_)), (x + w, to_y(exit_)), (0, 140, 140), 1)
        pts = [(x + int(i * w / self.trace.maxlen), to_y(v)) for i, v in enumerate(self.trace)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, (255, 255, 255), 1, cv2.LINE_AA)
        text(img, "click dip (press below yellow)", (x + 4, y + 12), 0.36)
