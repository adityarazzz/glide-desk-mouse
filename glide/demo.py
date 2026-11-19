"""Animated demo hand: shows each gesture instead of describing it, seen the way you see your own
hand (fingers pointing up the screen), plus the view transform that turns the camera image the
same way.

The hand is a 21-landmark skeleton in desk millimetres (x = your right, y = forward, z = up),
built from a few pose parameters. Because it uses MediaPipe's landmark layout, tests run each
demo pose through glide.features.measure: a demo can never teach a pose the detector rejects.
"""
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

# right hand, palm down, wrist at the origin. mcp id: (knuckle xy, splay deg, phalanx lengths)
FINGERS = {5: ((-20, 90), -6, (40, 24, 20)), 9: ((-3, 94), 0, (44, 28, 22)),
           13: ((13, 89), 6, (41, 26, 21)), 17: ((27, 80), 12, (32, 20, 18))}
THUMB_REST = np.array([(-22, 18, 14), (-38, 40, 14), (-48, 60, 12), (-54, 78, 10)], float)
THUMB_FIST = np.array([(-22, 18, 14), (-32, 38, 20), (-26, 54, 28), (-8, 62, 30)], float)
PALM_OUTLINE = np.array([(-30, 8, 14), (-27, 86, 14), (-3, 102, 14), (16, 97, 14), (34, 84, 14),
                         (31, 8, 14)], float)
KNUCKLE_Z = 15.0
FLEX = (75.0, 100.0, 65.0)          # joint angles at full curl, knuckle -> tip
CAMERA_MM = 450.0                   # perspective of an overhead camera this high


@dataclass(frozen=True)
class Pose:
    curl: tuple = (0.05, 0.05, 0.05, 0.05)   # index..pinky, 0 straight .. 1 tucked under
    thumb: float = 0.0          # 0 resting beside the hand .. 1 folded across (fist)
    pinch: float = 0.0          # thumb tip travelling to the pinch finger's tip
    pinch_to: int = 12          # 8 index, 12 middle
    tap: float = 0.0            # index finger raised off the desk
    lift: float = 0.0           # whole hand raised, mm
    yaw: float = 0.0            # deg, + turns the fingers to your left
    roll: float = 0.0           # deg, + rolls onto the little-finger edge (handshake)
    dx: float = 0.0
    dy: float = 0.0


def _finger(mcp, splay, lengths, curl, raise_deg=0.0):
    d = np.array([math.sin(math.radians(splay)), math.cos(math.radians(splay))])
    p = np.array([mcp[0], mcp[1], KNUCKLE_Z], float)
    phi, pts = -math.radians(raise_deg), [p]
    for length, a in zip(lengths, FLEX):
        phi += math.radians(a * curl)
        p = p + length * np.array([d[0] * math.cos(phi), d[1] * math.cos(phi), -math.sin(phi)])
        pts.append(p)
    return pts


def skeleton(pose, hand="right"):
    """(27, 3) mm: the 21 MediaPipe landmarks, then the palm outline."""
    pts = np.zeros((27, 3))
    pts[0] = (0, 0, 18)
    for (mcp_id, (mcp, splay, lengths)), c in zip(FINGERS.items(), pose.curl):
        if mcp_id == pose.pinch_to - 3:             # the pinching finger bends a little to meet the thumb
            c = min(1.0, c + 0.08 * pose.pinch)
        pts[mcp_id:mcp_id + 4] = _finger(mcp, splay, lengths, c, 28 * pose.tap if mcp_id == 5 else 0.0)
    thumb = THUMB_REST + (THUMB_FIST - THUMB_REST) * pose.thumb
    if pose.pinch > 0:
        tip = pts[pose.pinch_to]
        mid = thumb[1] + 0.55 * (tip - thumb[1]) + (-9, 0, 4)
        target = np.array([thumb[0], thumb[1], mid, tip])
        thumb = thumb + (target - thumb) * pose.pinch
    pts[1:5] = thumb
    pts[21:] = PALM_OUTLINE
    # roll about the forward axis: the thumb side rises
    r = math.radians(pose.roll)
    x, z = pts[:, 0].copy(), pts[:, 2].copy()
    pts[:, 0], pts[:, 2] = x * math.cos(r), z - x * math.sin(r)
    if hand == "left":
        pts[:, 0] *= -1
    y = math.radians(pose.yaw)
    cx, cy = 0.0, 50.0
    x, yy = pts[:, 0] - cx, pts[:, 1] - cy
    pts[:, 0] = cx + x * math.cos(y) - yy * math.sin(y) + pose.dx
    pts[:, 1] = cy + x * math.sin(y) + yy * math.cos(y) + pose.dy
    pts[:, 2] += pose.lift
    return pts


def project(pts, center, scale, z=None):
    """Desk mm -> panel px as seen from the overhead camera (things nearer it look bigger)."""
    z = pts[:, 2] if z is None else z
    f = CAMERA_MM / (CAMERA_MM - np.clip(z, -50, 300))
    return np.stack([center[0] + pts[:, 0] * scale * f, center[1] - (pts[:, 1] - 95) * scale * f], 1)


def landmarks_px(pose, hand="right", center=(320, 180), scale=1.4):
    """What the camera would report for this pose: (21, 2) image px and (21, 3) world metres."""
    pts = skeleton(pose, hand)[:21]
    return project(pts, center, scale), pts / 1000.0


# ---- gestures ---------------------------------------------------------------------------

def envelope(s, rise=0.15, hold=0.25, fall=0.2):
    """Gesture activation s seconds after the cue: 0 -> 1 -> 0."""
    if s < 0:
        return 0.0
    if s < rise:
        return s / rise
    if s < rise + hold:
        return 1.0
    return max(0.0, 1.0 - (s - rise - hold) / fall)


def _smooth(u):
    return u * u * (3 - 2 * u)


def _keys(t, keys):
    """Piecewise smooth interpolation through (time, value) keys, looping over the last time."""
    t = t % keys[-1][0]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * _smooth((t - t0) / max(t1 - t0, 1e-6))
    return keys[-1][1]


REST = Pose()
GRIP = Pose(curl=(0.45, 0.45, 0.5, 0.5), thumb=0.25)
TWO = Pose(curl=(0.03, 0.03, 1.0, 1.0), thumb=0.55)
FIST = dict(curl=(1.0, 1.0, 1.0, 1.0), thumb=1.0)


@dataclass
class Demo:
    base: Pose = REST
    act: dict = None                 # fields blended in by the activation (cued gestures)
    env: tuple = (0.15, 0.25, 0.2)   # rise, hold, fall seconds
    move: object = None              # t -> (dx, dy) mm
    path: object = None              # [(x, y)] mm drawn as the movement guide
    note: object = None              # t -> short caption ("hold", "lift up") or None
    period: float = 2.4              # intro loop: one gesture every period seconds
    loop: float = None               # length of the movement loop, when it differs from period

    def pose(self, t, g=None):
        """t: seconds into the loop. g: activation from a live cue, else the demo's own rhythm."""
        if self.act and g is None:
            g = envelope(t % self.period - 0.6, *self.env)
        p = self.blend(g or 0.0)
        if self.move:
            dx, dy = self.move(t)
            p = replace(p, dx=p.dx + dx, dy=p.dy + dy)
        return p

    def blend(self, g):
        """The base pose with the gesture blended in by activation g, before any movement."""
        p = self.base
        if not self.act:
            return p
        vals = {}
        for k, v in self.act.items():
            b = getattr(p, k)
            vals[k] = tuple(bi + (vi - bi) * g for bi, vi in zip(b, v)) if isinstance(v, tuple) else \
                v if k == "pinch_to" else b + (v - b) * g
        return replace(p, **vals)


def _circle(r, period):
    return lambda t: (r * math.cos(2 * math.pi * t / period), r * math.sin(2 * math.pi * t / period))


def _circle_path(r):
    return [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]


def _flicks(t):
    s = math.tanh(4 * math.sin(2 * math.pi * t / 0.9)) * 45
    return (s, 0.0) if (t % 6) < 3 else (0.0, s)


SCROLL_KEYS = [(0, 0), (0.8, 40), (2.2, 40), (3.0, 0), (3.8, -40), (5.2, -40), (6.0, 0)]
PINCH_MID = dict(pinch=1.0, pinch_to=12)
DEMOS = {
    "rest": Demo(),
    "get_ready": Demo(),
    "hover_rest": Demo(base=GRIP),
    "move_slow": Demo(move=_circle(45, 4.0), path=_circle_path(45), loop=4.0),
    "move_fast": Demo(move=_flicks, path=[(-45, -70), (45, -70), (-45, -70)], loop=6.0),
    "fine": Demo(move=_circle(7, 3.0), path=_circle_path(7), note=lambda t: "tiny", loop=3.0),
    "pinch_idx": Demo(act=dict(pinch=1.0, pinch_to=8)),
    "pinch_mid": Demo(act=PINCH_MID),
    "tap": Demo(act=dict(tap=1.0), env=(0.15, 0.2, 0.15)),
    "lift": Demo(act=dict(lift=70.0), env=(0.3, 0.6, 0.3), note=lambda t: "lift up"),
    "fist": Demo(act=FIST, env=(0.3, 0.6, 0.3)),
    "rot_left": Demo(base=Pose(yaw=45), move=_circle(30, 4.0), path=_circle_path(30), loop=4.0),
    "rot_left_pinch": Demo(base=Pose(yaw=45), act=PINCH_MID),
    "rot_right": Demo(base=Pose(yaw=-45), move=_circle(30, 4.0), path=_circle_path(30), loop=4.0),
    "rot_right_pinch": Demo(base=Pose(yaw=-45), act=PINCH_MID),
    "side": Demo(base=Pose(roll=70), move=lambda t: (35 * math.sin(2 * math.pi * t / 3), 0.0),
                 path=[(-35, -70), (35, -70), (-35, -70)], loop=3.0),
    "side_pinch": Demo(base=Pose(roll=70), act=PINCH_MID),
    "grip": Demo(base=GRIP, move=_circle(35, 4.0), path=_circle_path(35), loop=4.0),
    "grip_move": Demo(base=GRIP, move=_circle(35, 4.0), path=_circle_path(35), loop=4.0),
    "grip_pinch": Demo(base=GRIP, act=PINCH_MID),
    "two_scroll": Demo(base=TWO, move=lambda t: (0.0, 35 * math.sin(2 * math.pi * t / 2.4)),
                       path=[(85, -35), (85, 35), (85, -35)], loop=2.4),
    "scroll_long": Demo(base=TWO, move=lambda t: (0.0, _keys(t, SCROLL_KEYS)), path=[(85, -40), (85, 40), (85, -40)],
                        note=lambda t: "hold" if 0.8 < t % 6 < 2.2 or 3.8 < t % 6 < 5.2 else None, loop=6.0),
    # setup lessons
    "reach": Demo(move=lambda t: (_keys(t, [(0, 0), (1.0, 110), (3.0, 110), (3.8, 0), (4.4, 0)]), 0.0),
                  path=[(0, -70), (110, -70)], note=lambda t: "hold at the edge" if 1.0 < t % 4.4 < 3.0 else None,
                  loop=4.4),
    # lift-to-reposition: slide on the desk, lift a little, carry back, set down, slide again
    "reposition": Demo(act=dict(lift=55.0), env=(0.25, 1.0, 0.25), period=3.2, loop=3.2,
                       move=lambda t: (0.0, _keys(t, [(0, -25), (0.55, 25), (0.9, 25), (1.7, -25), (3.2, -25)])),
                       path=[(85, -25), (85, 25), (85, -25)],
                       note=lambda t: "slide" if t % 3.2 < 0.6 else "lift and carry back" if t % 3.2 < 1.85
                       else "put down" if t % 3.2 < 2.3 else None),
    "hold": Demo(act=PINCH_MID, env=(0.15, 1.4, 0.2), period=3.0, note=lambda t: "hold still"),
    "drag": Demo(act=PINCH_MID, env=(0.15, 1.6, 0.2), period=3.2,
                 move=lambda t: (_keys(t, [(0, 0), (0.9, 0), (2.3, 70), (2.6, 70), (3.2, 0)]), 0.0),
                 path=[(0, -70), (70, -70)], loop=3.2),
}



def export(hand="right", steps=21, fps=30):
    """Every demo as data for the app's canvas renderer: skeletons (27 x 3 mm) sampled over the
    gesture activation, the movement loop sampled in time, and the captions."""
    out = {}
    for name, d in DEMOS.items():
        loop = d.loop or d.period
        gs = [k / (steps - 1) for k in range(steps)] if d.act else [0.0]
        out[name] = {
            "frames": [np.round(skeleton(d.blend(g), hand), 1).tolist() for g in gs],
            "period": d.period, "loop": loop, "env": list(d.env), "act": bool(d.act),
            "ring": (d.act or {}).get("pinch_to", 8 if d.act and "tap" in d.act else None),
            "move": [[round(v, 1) for v in d.move(i / fps)] for i in range(int(round(loop * fps)))]
            if d.move else None,
            "path": [list(p) for p in d.path] if d.path else None,
            "notes": [d.note(i / 10) for i in range(int(round(loop * 10)))] if d.note else None,
        }
    return out


# ---- drawing ----------------------------------------------------------------------------

SKIN, SKIN_DARK, EDGE, SHADOW = (190, 205, 228), (130, 145, 170), (60, 70, 90), (38, 38, 38)
LIMBS = [(0, 1, 17), (1, 2, 16), (2, 3, 15), (3, 4, 14)] + \
        [(m + i, m + i + 1, w) for m, w in ((5, 15), (9, 15), (13, 14), (17, 13)) for i in range(3)]


def _dashed(img, pts, color):
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        if i % 2 == 0:
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, 2, cv2.LINE_AA)


def draw(img, demo, t, g=None, hand="right", center=None, scale=1.5, path_color=(255, 220, 0)):
    """Draws the demo hand at time t into img (in place). g overrides the gesture activation."""
    h, w = img.shape[:2]
    center = center or (w // 2, h // 2)
    pose = demo.pose(t, g)
    pts = skeleton(pose, hand)
    if demo.path:
        q = project(np.array([(x, y + 50, 0) for x, y in demo.path], float), center, scale)
        if len(q) > 5:                              # loops: dashed track
            _dashed(img, q, path_color)
        else:                                       # strokes: arrows
            for a, b in zip(q, q[1:]):
                cv2.arrowedLine(img, tuple(map(int, a)), tuple(map(int, b)), path_color, 2, cv2.LINE_AA,
                                tipLength=0.12)
        if demo.move:                               # where the hand is on its track right now
            dot = project(np.array([(pose.dx - demo.base.dx, pose.dy - demo.base.dy + 50, 0)]), center, scale)[0]
            cv2.circle(img, tuple(map(int, dot)), 6, path_color, -1, cv2.LINE_AA)
    # shadow on the desk: separates from the hand as it lifts
    sh = pts.copy()
    sh[:, 0] += pts[:, 2] * 0.35
    sh[:, 1] -= pts[:, 2] * 0.45
    _hand(img, project(sh, center, scale, z=np.zeros(len(sh))), sh, scale, shadow=True)
    _hand(img, project(pts, center, scale), pts, scale)
    if pose.pinch > 0.9 or pose.tap > 0.9:
        tip = project(pts[[pose.pinch_to if pose.pinch > 0.9 else 8]], center, scale)[0]
        cv2.circle(img, tuple(map(int, tip)), int(14 * scale), (90, 230, 90), 2, cv2.LINE_AA)
    return pose


def _hand(img, q, pts, scale, shadow=False):
    palm = q[21:].astype(np.int32)
    cv2.fillPoly(img, [palm], SHADOW if shadow else SKIN, cv2.LINE_AA)
    if not shadow:
        cv2.polylines(img, [palm], True, EDGE, 2, cv2.LINE_AA)
    # fingers low to high, thumb on top; a segment folding back toward the wrist is tucked
    # under the palm and drawn darker
    order = sorted(LIMBS[4:], key=lambda s: pts[s[1], 2]) + LIMBS[:4]
    for a, b, wid in order:
        tucked = a > 4 and np.hypot(*pts[b, :2]) < np.hypot(*pts[a, :2]) - 2
        col = SHADOW if shadow else SKIN_DARK if tucked else SKIN
        th = max(2, int(wid * scale))
        pa, pb = tuple(map(int, q[a])), tuple(map(int, q[b]))
        if not shadow:
            cv2.line(img, pa, pb, EDGE, th + 3, cv2.LINE_AA)
        cv2.line(img, pa, pb, col, th, cv2.LINE_AA)
    if not shadow:
        for i in (4, 8, 12, 16, 20):
            cv2.circle(img, tuple(map(int, q[i])), max(2, int(3 * scale)), (235, 240, 250), -1, cv2.LINE_AA)


# ---- camera view, turned so your fingers point up -----------------------------------------

class View:
    """A signed 90-degree turn/flip of the camera image that puts your forward up and your right
    on the right, so the live view matches the demo and your own sense of direction."""

    def __init__(self, R=None, size=(640, 360)):
        w, h = size
        self.M, best = np.eye(2), -9.0
        if R is not None:
            right, fwd = np.asarray(R[0], float), np.asarray(R[1], float)
            for swap in (False, True):
                for a in (1, -1):
                    for b in (1, -1):
                        M = np.array([[0, a], [b, 0]] if swap else [[a, 0], [0, b]], float)
                        s = (M @ fwd) @ (0, -1) + (M @ right) @ (1, 0)
                        if s > best:
                            self.M, best = M, s
        swap = self.M[0, 0] == 0
        self.size = (h, w) if swap else (w, h)
        # offsets keep the turned image inside the canvas
        corners = np.array([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], float) @ self.M.T
        self.o = -corners.min(axis=0)
        self.A = np.hstack([self.M, self.o[:, None]])

    def frame(self, img):
        return cv2.warpAffine(img, self.A, self.size, flags=cv2.INTER_NEAREST)

    def pts(self, p):
        return np.asarray(p, float) @ self.M.T + self.o
