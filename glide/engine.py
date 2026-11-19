"""Interaction engine (TRD 4.7-4.8): hand observations in, pointer actions out.

Gesture map, chosen from what an overhead webcam measurably separates (see glide.analyze):
  slide hand               move cursor (relative, speed-dependent gain, short prediction)
  quick pinch / index tap  left click (two quick ones = double click)
  pinch, then move         drag
  pinch, hold still        right click (long press)
  fist + move              scroll with inertia (fist.action = "clutch" makes it a clutch)
  lift hand high / away    clutch

States: IDLE TRACK PINCH DRAG HOLD SCROLL CLUTCH PAUSED
  PINCH = pressed but undecided; nothing is emitted until it becomes a click, drag or right click,
  so a gesture that turns out to be a fist or a glitch is cancelled cleanly.
"""
import collections
import math

import numpy as np

from .autogain import AutoGain
from .filters import OneEuro2D, ema_alpha, gain_curve
from .workspace import edge_band_px, scaled_gain

IDLE, TRACK, PINCH, DRAG, HOLD, SCROLL, CLUTCH, PAUSED = (
    "IDLE", "TRACK", "PINCH", "DRAG", "HOLD", "SCROLL", "CLUTCH", "PAUSED")
FRAME_S = 1 / 30


def rotation(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


class Engine:
    def __init__(self, cfg, profile, out):
        self.cfg, self.out = cfg, out
        f = cfg["filter"]
        self.filt = OneEuro2D(f["fc_min_hz"], f["beta"], f["d_cutoff_hz"], f.get("fc_max_hz"))
        self.state = IDLE
        self.t_last = None
        self.prev_pos = None      # filtered anchor, desk mm
        self.emitted = None       # filtered anchor + prediction lead, desk mm
        self.vel = np.zeros(2)    # mm/s
        self.speed = 0.0
        self.gain_speed = 0.0     # the speed the gain curve sees: rises slower than it falls
        self.acc = np.zeros(2)    # sub-pixel cursor remainder
        self.scroll_acc = np.zeros(2)
        self.momentum = np.zeros(2)
        self.lost_since = None
        self.last_raw, self.glitches = None, 0
        self.t_accepted = None    # last frame that moved the tracker (skipped glitch frames don't count)
        self.history = collections.deque(maxlen=90)   # (t, cursor xy, anchor mm)
        self.base = None          # open-hand flex level
        self.window = collections.deque()
        self.ratio = self.r_min = 1.0
        self.pressing = False
        self.press_cnt = self.release_cnt = 0
        self.t_peak = None
        self.recent = collections.deque()   # (t, ratio) over the onset window
        self.poses = {}
        self.scale_f, self.lifted = 1.0, False
        self.lift_base, self.lift_peak, self.lift_onset, self.lift_t = None, 1.0, 0.0, 0.0
        self.glide_rate, self.glide_since, self.exit_glide = None, None, None
        self.exits = 0            # times the hand left the view without gliding (for the app's tip)
        self.lift_trace = collections.deque()   # (t, ratio) for the onset rollback
        self.lift_ref = None      # image px the size model is centred on
        self.bounds = None        # free-desk rectangle in image px (surface.detect); None = whole view
        self.sensor_w, self.sensor_off = 1.0, np.zeros(2)   # see _point
        self.wall, self.near_side = [False] * 4, 3
        self.released_t = -9.0
        self.lever_len = self.lever_back = self.lever_off = None
        self.reacquired, self.arriving, self.landed_t, self.partial_since = False, False, -9.0, None
        self.arrive_min = None
        self.t_prev = None
        self.steady_mm = self.steady0 = None
        self.settle_until = 0.0
        self.acquired_t = -9.0
        self.onset_t, self.rb_pos, self.anchor_mm = 0.0, None, None
        self.dwell_anchor, self.dwell_t, self.dwell_armed, self.dwell_from = None, 0.0, False, None
        self.last_event = ""
        self.event_seq = 0
        self.scroll_anchor, self.scrolling = None, False
        self.edge_armed, self.edge_depth, self.edge_dir = False, 0.0, np.zeros(2)
        self.mirrored = 0
        self.yaw_off = 0.0        # adaptive axes: hand heading relative to calibration, desk radians
        self.rate_scroll = False
        sw = getattr(out, "screen_width", lambda: cfg["gain"]["reference_width"])()
        self.screen_scale = sw / cfg["gain"]["reference_width"]
        self.set_profile(profile)

    # ---- profile -------------------------------------------------------------------------

    def set_profile(self, p):
        self.profile = p
        self.R = np.array(p["R"], dtype=float) @ rotation(self.cfg["orient_offset_deg"])
        self.s0 = float(p["px_per_m"])
        self.mm_per_px = 1000.0 / self.s0
        self.chirality = int(p.get("chirality", 0))
        self.click = dict(self.cfg["click"], **p.get("click", {}))
        self.gain = scaled_gain(self.cfg["gain"], p.get("gain_scale", 1.0))   # fitted to the workspace
        self.axis_gain = np.array(p.get("axis_gain", [1.0, 1.0]), float)      # room shape vs screen shape
        self.lift_grad = np.array(p.get("lift_grad", [0.0, 0.0]), float)      # learned, see _size_fit
        self.sensor = tuple(p.get("sensor") or self.cfg["pointer"].get("sensor", [0.0, 0.0]))   # see _point
        self.autogain = AutoGain(self.cfg["autogain"], p.get("autogain"))                      # learned speed
        self.lift_prior = self.lift_grad.copy()
        self.size_S, self.size_b = np.zeros((3, 3)), np.zeros(3)
        fwd = self.R[1]
        self.yaw0 = math.atan2(fwd[1], fwd[0])                    # calibrated heading in the image
        self.det = float(np.sign(np.linalg.det(self.R))) or 1.0   # image -> desk may be mirrored

    def set_bounds(self, rect, frame_wh=None):
        """Free-desk rectangle (x0, y0, x1, y1) in image px, or None for the whole view."""
        self.bounds = None if rect is None else tuple(float(v) for v in rect)
        W, H = frame_wh or self.profile.get("frame_wh") or (640, 360)
        if self.bounds is None:
            self.wall = [False] * 4
        else:                                           # sides (left, top, right, bottom) set in from the view
            b = self.bounds
            self.wall = [b[0] > 2, b[1] > 2, b[2] < W - 2, b[3] < H - 2]
        # the user's side: the image side whose outward normal points back toward them
        normals = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.near_side = int(np.argmin([self.R[1] @ n for n in normals]))

    def rotate(self, deg):
        self.cfg["orient_offset_deg"] += deg
        self.set_profile(self.profile)

    def to_mm(self, px):
        """Image px -> (right, forward) desk mm in the user's frame."""
        return (self.R @ np.asarray(px, dtype=float)) * self.mm_per_px

    # ---- main update ---------------------------------------------------------------------

    def update(self, obs, t, typing=False, paused=False):
        # a near-zero interval means a late-delivered frame without a real timestamp
        dt = FRAME_S if self.t_last is None or t - self.t_last < 0.005 else t - self.t_last
        self.t_prev, self.t_last = self.t_last, t
        self._coast(dt)
        if paused:
            self._release()
            self._lose()
            self.momentum[:] = 0
            self.state = PAUSED
            return
        was_lifted = self.lifted                        # _accept resets it when the hand is gone
        mm, lost = self._accept(obs)
        if mm is None:
            if lost and self.lost_since is None:
                self._leave(t, was_lifted)
            if self.exit_glide is not None:
                self._exit_glide(t, dt)
            if lost or self.lost_since is not None:
                self.lost_since = self.lost_since or t
                if t - self.lost_since > self.cfg["hand"]["lost_ms"] / 1000:
                    self._release()
                    self.state = IDLE
            return
        self.lost_since = None
        self.exit_glide = None                          # back in view: the hand takes over again
        if self.state in (IDLE, PAUSED) or self.reacquired:
            # any return counts, also within lost_ms (the state was still TRACK then and a hand
            # put back quickly got no settling at all)
            if self.state in (IDLE, PAUSED):
                self.state = TRACK
            self.acquired_t = t                         # hand just arrived: it is usually still sliding in
            self.arriving, self.reacquired = True, False

        c = self.cfg
        # after a skipped (doubtful or glitch) frame the motion since the last accepted frame spans
        # two frame intervals: measure speed over both, or it reads double and the gain overshoots
        dt_move = dt if self.t_accepted is None else max(t - self.t_accepted, dt)
        self.t_accepted = t
        d, pos = self._track(mm, t, dt_move)
        self._axes(obs, dt)
        glide = self._edge(obs, dt)
        h = c["hand"]
        since = t - self.acquired_t
        if glide is not None:
            glide = None if since < h["entry_edge_ms"] / 1000 else glide * self.screen_scale
        if glide is not None and self.state == TRACK:   # remembered in case the hand leaves the view
            self.glide_rate = glide / max(dt, 1e-3)
            self.glide_since = self.glide_since if self.glide_since is not None else t
        else:
            self.glide_since = None
        # putting the hand down is not a stroke: hold until it is fully in view and down on the desk,
        # then fade the motion in
        partial = self._partial(obs)
        self.partial_since = (self.partial_since or t) if partial else None
        if self.arriving:
            landed = self._landed(obs)
            size = self._norm_size(obs)[0]
            self.arrive_min = size if self.arrive_min is None else min(self.arrive_min, size)
            if (not partial and landed and since >= h["entry_hold_ms"] / 1000) or since > h["arrive_max_ms"] / 1000:
                self.arriving, self.landed_t = False, t
                if self.lift_base is None or not landed:
                    # first visit: it came down to its smallest; stayed "high" that long: its rest now
                    self.lift_base = self.arrive_min if self.lift_base is None else size
                self.arrive_min = None
        entry = 0.0 if self.arriving else             float(np.clip((t - self.landed_t) / max(h["entry_ramp_ms"] / 1000, 1e-3), 0, 1))
        if partial:
            entry = 0.0                                 # part of the palm is out of view: its landmarks are guesses
        fist = self._debounce("fist", obs.fist)
        two = self._debounce("two", obs.two_finger)
        scroll_pose = two if c["scroll"]["gesture"] == "two_finger" else (fist and c["fist"]["action"] == "scroll")
        lift = None if self.arriving else             self._lift(obs, t, dt, busy=obs.fist or obs.two_finger or self.pressing or self.ratio < 0.85)
        if lift == "up" and self.state == PINCH:
            # the hand closes a little as it starts to lift: that pinch was the lift, not a click
            self.pressing, self.press_cnt = False, 0
            self.state = TRACK
        if lift == "up" and self.state in (TRACK, CLUTCH):
            # the lift was recognised a few frames in: undo the motion it already made
            self.out.move_to(*self._cursor_at(self.lift_onset))
            self.momentum[:] = 0
            self._note("lift")
        elif lift == "down":
            self.acquired_t = t                         # setting the hand down: settle like an arrival
            self.landed_t = t + c["hand"]["entry_hold_ms"] / 1000
        # no clicks while a returning hand settles: the open finger gap is relearned from its first
        # frames, and fingers spreading or relaxing as it lands then look like a pinch
        settled = not self.arriving and t - self.landed_t >= c["click"]["settle_ms"] / 1000
        can_press = (self.state == TRACK and not typing and not obs.fist and not obs.two_finger and settled
                     and not self.lifted and self.speed < self.click["max_speed_mm_s"]
                     and obs.palm_in(c["pad"]["margin"]) and obs.tips_in(c["pad"]["tip_margin"]))
        started, released = self._press(obs, dt, t, can_press)
        st = self.state

        if st == PINCH:
            if scroll_pose or fist or obs.fist:        # a fist / scroll pose forming, not a pinch
                self.pressing = False
                if scroll_pose:
                    self._start_scroll(t)
                else:
                    self.state = TRACK
            elif released:
                self.out.click("left")
                self._event("left click", t)
            elif np.hypot(*(self.steady_mm - self.steady0)) > self.click["drag_mm"]:
                # measured on the steady palm point: the knuckle line shifts ~10 mm as a pinch forms
                self.out.button("left", True)
                self.state = DRAG
                self._move(self.steady_mm - self.steady0)   # catch up with the motion so far
                self._note("drag")
            elif t - self.onset_t >= self.click["long_ms"] / 1000:
                if self._held_still(t):
                    self.out.click("right")
                    self.state = HOLD
                    self._note("right click")
                else:
                    # held that long while drifting: a hand curling into a grip or relaxing, which
                    # read as pinch-and-hold (3 of 8 recorded phantom presses became right clicks)
                    self.pressing, self.press_cnt = False, 0
                    self.state = TRACK
        elif st == DRAG:
            if scroll_pose:                            # switching to the scroll pose ends a drag
                self.out.button("left", False)
                self.pressing = False
                self._start_scroll(t)
            elif released:
                self.out.move_to(*self._cursor_at(t - self.click["release_rollback_ms"] / 1000))
                self.out.button("left", False)
                self._event("drag end", t)
            else:
                self._move(d, glide)                   # edge glide carries drags across the screen too
        elif st == HOLD:
            if released:
                self.state = TRACK
                self.settle_until = t + self.click["release_rollback_ms"] / 1000
        elif st == SCROLL:
            if not scroll_pose:
                if self.scrolling and not self.rate_scroll and self.speed > c["scroll"]["inertia_min_mm_s"]:
                    self.momentum = self.vel.copy()
                self.state, self.rate_scroll = TRACK, False
                self.settle_until = t + c["scroll"]["settle_ms"] / 1000
            elif t >= self.settle_until and not typing:
                if self.scroll_anchor is None:
                    self.scroll_anchor = pos
                if self.scrolling:
                    self._scroll(d)
                    self._rate_scroll(pos, dt)
                    if glide is not None:              # fist held at the edge keeps scrolling
                        e = c["edge"]
                        k = -1 if c["scroll"]["natural"] else 1
                        self.scroll_acc += self.edge_dir * e["scroll_units_s"] * self.edge_depth ** e["power"] * dt * k
                        self._flush_scroll()
                elif np.hypot(*(pos - self.scroll_anchor)) > c["scroll"]["start_mm"]:
                    self.scrolling = True              # dead zone: a curled resting hand won't scroll
                    self._note("scroll")
        elif self.lifted or (fist and c["fist"]["action"] == "clutch"):
            self.state = CLUTCH
        elif scroll_pose:
            self._start_scroll(t)
        elif started:
            self.state = PINCH
            self.onset_t = self.t_peak if self.t_peak is not None else t
            self.rb_pos = self._cursor_at(self.onset_t - FRAME_S)
            self.autogain.click(self.rb_pos)            # where the user meant to go: learn from it
            self.anchor_mm = self._mm_at(self.onset_t - FRAME_S, pos)
            self.steady0 = self._steady_at(self.onset_t - FRAME_S)
            self.out.move_to(*self.rb_pos)             # click stabilizer: freeze at the onset point
            self.momentum[:] = 0
        else:
            self.state = TRACK
            if not typing and t >= self.settle_until:
                self._move(d * entry, glide)
                self._dwell(t)
        # learn only from plain pointing: no glide, lift, arrival or other state in the way
        clean = self.state == TRACK and glide is None and entry >= 1.0 and not self.lifted
        if self.state != PINCH:
            self.autogain.frame(t, self.out.cursor(), self.speed, clean)
        self.history.append((t, self.out.cursor(), pos, self.steady_mm))

    # ---- input conditioning --------------------------------------------------------------

    def _accept(self, obs):
        """(anchor desk mm, lost). mm is None for no hand / other hand (lost) or a glitch frame."""
        if not obs.present:
            self._lose()
            return None, True
        if self.chirality:
            # the other (mirrored) hand sits steadily at -0.73; pinching in tilted poses flips the
            # palm triangle briefly to -0.4..-0.5 (recorded), so only a deep, sustained flip counts
            s = obs.palm_sin * self.chirality
            self.mirrored = self.mirrored + 1 if s < -0.5 else 0
            if self.mirrored >= 10:
                self._lose()
                return None, True
            if self.mirrored:                          # deep flip: don't act on it while deciding
                return None, False
        self.steady_mm = self.to_mm(obs.anchor(self.cfg["pointer"]["anchor"]))   # gesture decisions
        mm = self.to_mm(self._point(obs))
        h = self.cfg["hand"]
        step = np.hypot(*(mm - self.last_raw)) if self.last_raw is not None else 0.0
        jump = step > h["glitch_mm"]
        # A frame the tracker doubts that would also leap: a hand tilting off the desk (0.58-0.67)
        # flung the cursor 1160 px in one frame. Doubtful frames that barely move are kept, and
        # pointing only: skipping frames as a pinch forms cost a rotated-hand pinch its click.
        doubtful = (obs.score < h.get("min_track_score", 0.0) and step > h.get("doubt_mm", 0.0)
                    and self.state not in (PINCH, DRAG, HOLD, SCROLL))
        if jump or doubtful or not 0.04 < obs.hand_m < 0.16:
            self.glitches += 1
            if self.glitches <= 3:
                return None, False                     # hold the state, skip this frame
            self._lose()                               # it really moved: re-anchor without a jump
        self.glitches = 0
        self.last_raw = mm
        return mm, False

    def _point(self, obs):
        """The tracked point, image px: the mouse-like sensor point while moving, the steady palm
        anchor while a pinch, scroll or fist is forming (its fingers reshape the knuckle line). The
        blend is continuous: each change of weight is offset so the point never jumps."""
        p = self.cfg["pointer"]
        steady = obs.anchor(p["anchor"])
        k, side = self.sensor
        if (k == 0.0 and side == 0.0) or obs.img is None:
            return steady
        lever = self._lever(obs, steady, k, side)
        # A pinch closing (a quick dip from the last moment's level) with a finger straight enough to
        # click (click.min_ext). Holding a mouse curls the fingers: the gap then wanders by 0.13-0.21
        # with no pinch possible, and a dip test alone kept the steady point 58-95% of the time.
        # The hand reopening after a click is held off for 250 ms.
        peak = max((r for _, r in self.recent), default=self.ratio)
        can_click = max(obs.ext[0], obs.ext[1]) >= self.click["min_ext"]
        recent = self.t_last is not None and self.t_last - self.released_t < 0.25
        busy = (self.pressing or recent or (can_click and peak - self.ratio > 0.10)
                or self.state in (PINCH, DRAG, HOLD, SCROLL) or obs.two_finger)
        dt = FRAME_S if self.t_last is None or self.t_prev is None else max(self.t_last - self.t_prev, 1e-3)
        step = dt / max(p.get("sensor_blend_ms", 120.0) / 1000, 1e-3)
        w = max(self.sensor_w - step, 0.0) if busy else min(self.sensor_w + step, 1.0)
        self.sensor_off -= (w - self.sensor_w) * lever
        self.sensor_w = w
        return steady + w * lever + self.sensor_off

    def _lever(self, obs, steady, k, side):
        """Steady anchor -> sensor point as a rigid lever: the hand's direction (wrist -> knuckle line)
        times a slowly learned length. Swivelling turns it exactly like the knuckle line, but the
        knuckles sliding along the hand as a pinch forms (they shorten it by ~10 mm) don't move it:
        with the raw knuckle point the worst click landed 57-164 px from its target (recorded)."""
        img = obs.img
        a = img[[5, 9, 13, 17]].mean(axis=0) - img[0]
        n = float(np.hypot(*a))
        ids = self.cfg["pointer"].get("lever_dir", [17])
        r = img[ids].mean(axis=0) - img[0]               # the knuckles that move least in any click
        if n < 1e-6 or not r.any():
            return np.zeros(2)
        ang_r = math.atan2(r[1], r[0])
        off = math.remainder(math.atan2(a[1], a[0]) - ang_r, 2 * math.pi)
        if self.lever_off is None:
            self.lever_off = off
        elif not (self.pressing or self.state in (PINCH, DRAG, HOLD, SCROLL)):
            self.lever_off += 0.03 * math.remainder(off - self.lever_off, 2 * math.pi)
        u = np.array([math.cos(ang_r + self.lever_off), math.sin(ang_r + self.lever_off)])
        perp = np.array([-u[1], u[0]])
        perp *= 1.0 if perp @ (img[5] - img[17]) >= 0 else -1.0
        back = float((steady - img[0]) @ u)              # how far along the hand the anchor sits
        if self.lever_len is None:
            self.lever_len, self.lever_back = n, back
        elif not (self.pressing or self.state in (PINCH, DRAG, HOLD, SCROLL)):
            f = 0.03                                     # ~2 s at 17 fps: follows the hand, not a pinch
            self.lever_len += f * (n - self.lever_len)
            self.lever_back += f * (back - self.lever_back)
        L = self.lever_len
        return (k * L - self.lever_back) * u + side * L * perp

    def _track(self, mm, t, dt):
        pos = self.filt(mm, t)
        if self.prev_pos is not None:
            self.vel += ema_alpha(self.cfg["filter"]["speed_cutoff_hz"], dt) * ((pos - self.prev_pos) / dt - self.vel)
        self.speed = float(np.hypot(*self.vel))
        g = self.cfg["gain"]
        hz = g.get("rise_hz", 1e3) if self.speed > self.gain_speed else g.get("fall_hz", 1e3)
        self.gain_speed += ema_alpha(hz, dt) * (self.speed - self.gain_speed)
        self.prev_pos = pos
        p = self.cfg["pointer"]
        lead = np.zeros(2)
        if self.state in (TRACK, DRAG) and p["predict_ms"] > 0:
            ramp = np.clip((self.speed - p["predict_v0"]) / (p["predict_v1"] - p["predict_v0"]), 0, 1)
            lead = self.vel * (p["predict_ms"] / 1000) * ramp
        target = pos + lead
        d = np.zeros(2) if self.emitted is None else target - self.emitted
        self.emitted = target
        return d, pos

    def _press(self, obs, dt, t, can_press):
        """Relative dip of the thumb-to-finger gaps below their recent open level; returns edges.

        The open level is a high percentile of the last base_window_s (frozen while pressed), so it
        follows the hand's pose as it drifts and ignores single-frame spikes. A press releases when
        the gap recovers part-way from that press's own deepest point, so it cannot stick when the
        resting pose changes while pressed.
        """
        k = self.click
        vals = (obs.flex8, obs.flex12)
        reliable = obs.score >= k["min_score"]
        can_press = can_press and reliable
        if not self.pressing and reliable:
            self.window.append((t, vals))
            while self.window and t - self.window[0][0] > k["base_window_s"]:
                self.window.popleft()
        if self.base is None or not self.pressing:
            arr = np.array([v for _, v in self.window])
            self.base = np.percentile(arr, 85, axis=0) if len(arr) >= 5 else np.array(vals)
        ratios = [v / max(b, 1e-3) for v, b in zip(vals, self.base)]
        i = int(np.argmin(ratios))
        self.ratio = float(ratios[i])
        # a curled hand has tiny gaps whose relative dips are noise: every recorded real pinch/tap
        # moved its gap by >= 0.35 hand-lengths, grip-pose ghosts by 0.06-0.28
        can_press = can_press and self.base[i] - vals[i] >= k["min_dip"]
        # in a real pinch the thumb comes to a straight finger (extension 1.74-2.31 in 24/25 recorded
        # presses); a finger curling while the hand slides in a mouse grip shortens (1.33-1.68)
        can_press = can_press and obs.ext[i] >= k["min_ext"]
        if not self.pressing:
            # a deliberate pinch dips fast; a hand slowly settling into a pinch-like pose does not.
            # The dip is measured from the highest ratio in the onset window, which is also where
            # the click is placed (the stabilizer rolls the cursor back to it)
            self.recent.append((t, self.ratio))
            while t - self.recent[0][0] > k["max_onset_ms"] / 1000:
                self.recent.popleft()
            peak = max(r for _, r in self.recent)
            # the dip starts at the last frame still near the peak: the click goes there
            self.t_peak = max(tr for tr, r in self.recent if r >= peak - 0.03)
            if peak - self.ratio < k["onset_drop"]:
                can_press = False
            self.press_cnt = self.press_cnt + 1 if self.ratio < k["enter"] and can_press else 0
            if can_press and (self.ratio < k["deep"] or self.press_cnt >= k["confirm_frames"]):
                self.pressing, self.release_cnt, self.r_min = True, 0, self.ratio
                return True, False
            return False, False
        self.r_min = min(self.r_min, self.ratio)
        up = min(k["exit"], self.r_min + k["recover"] * (1 - self.r_min))
        self.release_cnt = self.release_cnt + 1 if self.ratio > up else 0
        if self.release_cnt >= k["confirm_frames"] or self.ratio > 0.97:
            self.pressing, self.press_cnt = False, 0
            self.released_t = self.t_last
            return False, True
        return False, False

    def _debounce(self, name, raw, frames=2):
        on, cnt = self.poses.get(name, (False, 0))
        cnt = cnt + 1 if raw != on else 0
        if cnt >= frames:
            on, cnt = raw, 0
        self.poses[name] = (on, cnt)
        return on

    # ---- outputs -------------------------------------------------------------------------

    def _axes(self, obs, dt):
        """Which way is 'forward' for the hand's motion.

        "mouse": the axes turn with the hand all the time, like a mouse's sensor, which is fixed to
        the mouse and so to the hand. Sweeping a mouse by pivoting at the elbow or wrist moves it on
        an arc while it turns with the forearm, and the cursor still runs straight across. In fixed
        desk axes the same sweep bent the cursor path by the hand's turn (4-14 deg per stroke, up
        to ~20, recorded). Frozen while a pinch, drag or scroll is on and while the hand is arriving
        or partly out of view, so finger motion and guessed landmarks can't twist it.
        "adaptive": follow the heading slowly and only while nearly still. "fixed": calibration axes.
        """
        p = self.cfg["pointer"]
        if obs.yaw is None:
            return
        if p["axes"] == "mouse":
            if (self.pressing or self.arriving or self._partial(obs)
                    or self.state in (PINCH, DRAG, HOLD, SCROLL) or obs.two_finger):
                return
            target = math.remainder(obs.yaw - self.yaw0, 2 * math.pi) * self.det
            lim = math.radians(p.get("mouse_max_deg", 75.0))
            target = max(-lim, min(lim, target))
            self.yaw_off += ema_alpha(p.get("mouse_hz", 6.0), dt) * math.remainder(target - self.yaw_off, 2 * math.pi)
            return
        if p["axes"] != "adaptive":
            return
        if self.state not in (TRACK, CLUTCH) or self.speed > p["axes_max_speed"]:
            return
        delta = math.remainder(obs.yaw - self.yaw0, 2 * math.pi) * self.det
        err = math.remainder(delta - self.yaw_off, 2 * math.pi)
        if abs(math.degrees(err)) > p["axes_deadband_deg"]:
            self.yaw_off += (1.0 - math.exp(-dt / p["axes_tau_s"])) * err

    def _move(self, d_mm, glide_px=None):
        if self.yaw_off:
            c, s = math.cos(-self.yaw_off), math.sin(-self.yaw_off)
            d_mm = np.array([c * d_mm[0] - s * d_mm[1], s * d_mm[0] + c * d_mm[1]])
        v = self.gain_speed
        g = gain_curve(v, self.gain) * self.cfg["gain"]["speed"] * self.screen_scale * self.autogain.factor(v)
        # room-shape gain only on fast strokes: reach needs it, slow aiming keeps its precision
        ramp = float(np.clip((v - 80.0) / 120.0, 0.0, 1.0))   # full above 200 mm/s
        a = 1.0 + (self.axis_gain - 1.0) * ramp
        self.acc += np.array([d_mm[0] * a[0], -d_mm[1] * a[1]]) * g   # forward on the desk = up on screen
        if glide_px is not None:
            self.acc += np.array([glide_px[0], -glide_px[1]])
        step = np.trunc(self.acc)
        if step.any():
            self.out.move_by(int(step[0]), int(step[1]))
            self.acc -= step

    def _leave(self, t, was_lifted):
        """The hand just left the view. If it left by gliding out through the edge band, the user
        is reaching further than the camera sees, so the glide carries on until the hand is back."""
        e = self.cfg["edge"]
        gliding = (self.glide_since is not None and t - self.glide_since >= e["exit_min_ms"] / 1000
                   and not was_lifted and self.state == TRACK)
        self.exit_glide = (self.glide_rate.copy(), t) if gliding else None
        if not gliding and self.state == TRACK and not was_lifted:
            self.exits += 1
            # taking the hand away is not a stroke: undo its last moments (motion-blurred, half out of
            # view), back to before it started going, like a mouse lifted off the pad
            h = self.cfg["hand"]
            back = (self.t_prev or t) - h["leave_rollback_ms"] / 1000
            if self.partial_since is not None:
                back = min(back, self.partial_since)
            calm = self._calm_before(t - h["leave_stroke_ms"] / 1000)
            if calm is not None:                         # the whole pull-away, from where it began
                back = min(back, calm)
            if self.history and t - back < 0.8:
                self.out.move_to(*self._cursor_at(back))
        self.partial_since = None
        self.glide_since = None

    def _exit_glide(self, t, dt):
        rate, t0 = self.exit_glide
        if t - t0 > self.cfg["edge"]["exit_glide_ms"] / 1000:
            self.exit_glide = None
            return
        self.edge_depth = 1.0                           # the overlay keeps showing the glide arrow
        self._move(np.zeros(2), rate * dt)

    def _lift(self, obs, t, dt, busy):
        """Lift-to-reposition, like lifting a mouse. Returns "up", "down" or None.

        Seen from above, a lifted hand looks bigger: about 1.25x for 5 cm under this camera. The
        size is the longest palm bone in the image. Finger poses don't stretch it the way they do
        the world-scale estimate (fists and pinches moved that by 15-45%). It is corrected for where the
        hand is in the view (_size_fit), then compared with a baseline that slowly follows the resting
        hand. The baseline doesn't learn while a pinch, fist or scroll pose is
        forming, so those can't read as lifts. The hand counts as down again once it has given back
        most of the rise.
        """
        k = self.cfg["clutch"]
        size, dp, raw = self._norm_size(obs)
        if self.lift_base is None:
            self.lift_base = size
            return None
        r = size / self.lift_base
        if not self.lifted and not busy and abs(math.log(r)) < 0.25:
            self._size_fit(dp, math.log(raw), dt)
        self.scale_f = r
        self.lift_trace.append((t, r))
        while t - self.lift_trace[0][0] > k["rollback_ms"] / 1000:
            self.lift_trace.popleft()
        if not self.lifted:
            # during a pinch the size isn't learned from (busy), but a clear lift still counts: the
            # hand closes a little as it starts lifting, and that pinch must not become a click
            pinch_lift = self.state == PINCH and r > k["lift_on"] + k["pinch_lift_extra"]
            if r > k["lift_on"] and (not busy or pinch_lift):
                calm = [tt for tt, rr in self.lift_trace if rr < 1 + (r - 1) * 0.2]
                onset = calm[-1] if calm else self.lift_trace[0][0]
                start = self._mm_at(onset, None)
                if start is not None and self.prev_pos is not None and                         np.hypot(*(self.prev_pos - start)) > k["stroke_mm"]:
                    # grew while sliding far: the desk's perspective, not a lift (real lifts rise
                    # within 70-130 ms over 8-16 mm). This spot's size is the rest size now.
                    self.lift_base = size
                    self.lift_trace.clear()
                    return None
                self.lifted, self.lift_peak, self.lift_t = True, r, t
                self.lift_onset = onset
                return "up"
            if not busy:
                self.lift_base += (1 - math.exp(-dt / k["base_tau_s"])) * (size - self.lift_base)
            return None
        self.lift_peak = max(self.lift_peak, r)
        landed = r < k["lift_off"] or r < 1 + (self.lift_peak - 1) * k["land_keep"]
        if landed or t - self.lift_t > k["max_lift_s"]:     # held up for long: that's the new rest
            self.lifted, self.lift_base = False, size
            self.lift_trace.clear()
            return "down"
        return None

    def _norm_size(self, obs):
        """(palm size as it would look at the reference spot, offset from that spot, raw size)."""
        raw = obs.palm_size or obs.px_per_m
        if self.lift_ref is None:
            self.lift_ref = np.array(obs.palm_px, float)
        dp = np.asarray(obs.palm_px, float) - self.lift_ref
        return raw * math.exp(-float(self.lift_grad @ dp)), dp, raw

    def _landed(self, obs):
        """A returning hand is down once it looks its resting size again (it comes back from above,
        bigger, still drifting sideways). Unknown rest size: assume down."""
        if self.lift_base is None:
            return True
        return self._norm_size(obs)[0] / self.lift_base <= self.cfg["clutch"]["lift_off"]

    def _partial(self, obs):
        """Part of the palm out of the view, where the tracker can only guess the landmarks."""
        if obs.img is None or obs.frame_wh is None:
            return False
        W, H = obs.frame_wh
        p = obs.img[[0, 1, 5, 9, 13, 17]]
        m = 0.01
        return bool((p[:, 0] < m * W).any() or (p[:, 0] > (1 - m) * W).any() or
                    (p[:, 1] < m * H).any() or (p[:, 1] > (1 - m) * H).any())

    def _size_fit(self, dp, log_size, dt):
        """How the resting palm's image size changes across the view: log size = c + grad . dp.

        Under this camera the palm looks 0.9-1.65 per mille bigger per pixel forward (six recorded
        sessions), so across the 21 cm of forward room it grows by 30-60%, far past the lift
        threshold: pushing up the screen read as a lift and froze the cursor. Fitted by least
        squares with a ~60 s memory, ridge-pulled to the saved gradient when the hand hasn't moved
        around enough to tell.
        """
        k = self.cfg["clutch"]
        a = np.array([1.0, dp[0], dp[1]])
        f = math.exp(-dt / k["grad_tau_s"])
        self.size_S = self.size_S * f + np.outer(a, a)
        self.size_b = self.size_b * f + a * log_size
        lam = k["grad_ridge"]
        M = self.size_S + np.diag([1e-6, lam, lam])
        rhs = self.size_b + np.array([0.0, *(lam * self.lift_prior)])
        try:
            g = np.linalg.solve(M, rhs)[1:]
        except np.linalg.LinAlgError:
            return
        self.lift_grad = np.clip(g, -0.004, 0.004)

    def _edge(self, obs, dt):
        """RubberEdge glide in (right, forward) screen px this frame, or None.

        depth 0..1 = how far the hand's bounding box has entered the band along each side of the free
        desk (the whole view unless set_bounds found a bag, a keyboard or the desk's edge in it).
        It arms only when the hand moves outward into the band, so a hand that merely rests near
        an edge never drifts, and disarms when the hand pulls back inward. Past the edge (onto the
        bag, off the desk) the depth stays at 1, so it keeps gliding.
        """
        e = self.cfg["edge"]
        self.edge_depth = 0.0
        if not e["enabled"] or obs.img is None or obs.frame_wh is None:
            self.edge_armed = False
            return None
        W, H = obs.frame_wh
        z = max(edge_band_px(e, (W, H)), 1.0)
        (x0, y0), (x1, y1) = obs.img.min(axis=0), obs.img.max(axis=0)
        bounds = self.bounds if self.bounds is not None else (0, 0, W, H)
        frame_edge = (0, 0, W, H)
        # The view's own edges: a band inside them (the hand runs out of sight there). Free-desk walls
        # (a bag, the desk's edge) only glide once the hand pushes past them, and never on the user's
        # side, where the wrist and forearm rest anyway. As bands, they glided 33-84% of the time
        # during ordinary moves on a cramped desk and pinned the cursor to the top or bottom.
        def depth(gap_frame, gap_wall, side):
            if self.wall[side]:
                return 0.0 if self.near_side == side else float(np.clip(-gap_wall / z, 0.0, 1.0))
            return float(np.clip((z - gap_frame) / z, 0.0, 1.0))
        push = np.array([depth(W - x1, bounds[2] - x1, 2) - depth(x0, x0 - bounds[0], 0),
                         depth(H - y1, bounds[3] - y1, 3) - depth(y0, y0 - bounds[1], 1)])
        depth = min(float(np.hypot(*push)), 1.0)
        if depth == 0.0:
            self.edge_armed = False
            return None
        u = self.R @ push
        u /= max(float(np.hypot(*u)), 1e-6)
        out_speed = float(self.vel @ u)
        if not self.edge_armed and out_speed > e["arm_mm_s"]:
            self.edge_armed = True
        elif self.edge_armed and out_speed < -2 * e["arm_mm_s"]:
            self.edge_armed = False
        if not self.edge_armed:
            return None
        self.edge_depth, self.edge_dir = depth, u
        return u * e["rate_px_s"] * depth ** e["power"] * dt

    def _scroll(self, d_mm):
        s = self.cfg["scroll"]
        k = s["units_per_mm"] * (1 + s["accel"] * min(self.speed / 200.0, 3.0))   # fast flicks go further
        k *= -1 if s["natural"] else 1
        self.scroll_acc += np.array([d_mm[0], d_mm[1]]) * k
        self._flush_scroll()

    def _rate_scroll(self, pos, dt):
        """Joystick zone: fist held beyond rate_mm from where it closed keeps the page moving, so a
        long page never needs a long stroke. Locked to the dominant axis to avoid sideways drift."""
        s = self.cfg["scroll"]
        off = pos - self.scroll_anchor
        if abs(off[1]) > 2 * abs(off[0]):
            off = np.array([0.0, off[1]])
        elif abs(off[0]) > 2 * abs(off[1]):
            off = np.array([off[0], 0.0])
        dist = float(np.hypot(*off))
        self.rate_scroll = dist > s["rate_mm"]
        if not self.rate_scroll:
            return
        depth = min((dist - s["rate_mm"]) / s["rate_span_mm"], 1.0)
        v = off / dist * s["rate_units_s"] * depth ** 1.3 * (-1 if s["natural"] else 1)
        self.scroll_acc += v * dt
        self._flush_scroll()

    def _flush_scroll(self):
        s = self.cfg["scroll"]
        step = np.trunc(self.scroll_acc / s["min_step"]) * s["min_step"]
        if step.any():
            self.out.scroll(int(step[1]), int(step[0]))
            self.scroll_acc -= step

    def _coast(self, dt):
        """Scroll inertia: keep scrolling after the fist opens, decaying exponentially."""
        if not self.momentum.any():
            return
        self._scroll(self.momentum * dt)
        self.momentum *= math.exp(-dt / self.cfg["scroll"]["inertia_tau_s"])
        if np.hypot(*self.momentum) < 8.0:
            self.momentum[:] = 0

    def _start_scroll(self, t):
        self.momentum[:] = 0
        self.scroll_acc[:] = 0
        self.scroll_anchor, self.scrolling = None, False
        self.state = SCROLL
        self.settle_until = t + self.cfg["scroll"]["settle_ms"] / 1000   # forming the pose shifts the anchor

    def _dwell(self, t):
        dw = self.cfg["dwell"]
        if not dw["enabled"]:
            return
        cur = np.array(self.out.cursor())
        if self.dwell_anchor is None or np.hypot(*(cur - self.dwell_anchor)) > dw["radius_px"]:
            self.dwell_anchor, self.dwell_t = cur, t
            if self.dwell_from is None or np.hypot(*(cur - self.dwell_from)) > dw["arm_px"]:
                self.dwell_armed = True
        elif self.dwell_armed and t - self.dwell_t >= dw["ms"] / 1000:
            self.out.click("left")
            self.dwell_armed, self.dwell_from = False, cur
            self._note("dwell click")

    def _event(self, name, t):
        self.state = TRACK
        self._note(name)
        self.settle_until = t + self.click["release_rollback_ms"] / 1000   # release shifts the anchor

    def _note(self, name):
        """Record a user-visible event; the app turns event_seq changes into a feedback tick."""
        self.last_event = name
        self.event_seq += 1

    # ---- helpers -------------------------------------------------------------------------

    def _cursor_at(self, t_target):
        for ts, p, *_ in reversed(self.history):
            if ts <= t_target:
                return p
        return self.history[0][1] if self.history else self.out.cursor()

    def _held_still(self, t):
        """A deliberate press-and-hold: the steady palm point stayed within hold_mm over the hold."""
        pts = [s for ts, _, _, s in self.history if ts >= self.onset_t and s is not None]
        if len(pts) < 2:
            return True
        pts = np.array(pts)
        return float(np.hypot(*(pts - pts[0]).T).max()) <= self.click["hold_mm"]

    def _calm_before(self, t_min, still_mm_s=60.0):
        """Latest history time after t_min at which the hand was moving slowly, or None."""
        h = list(self.history)
        for (t1, _, p1, *_), (t0, _, p0, *_) in zip(reversed(h), list(reversed(h))[1:]):
            if t0 < t_min:
                return None
            if p0 is not None and p1 is not None and np.hypot(*(p1 - p0)) / max(t1 - t0, 1e-3) < still_mm_s:
                return t1
        return None

    def _steady_at(self, t_target):
        for ts, *_, m in reversed(self.history):
            if ts <= t_target:
                return m
        return self.history[0][3] if self.history else self.steady_mm

    def _mm_at(self, t_target, default):
        for ts, _, m, *_ in reversed(self.history):
            if ts <= t_target:
                return m
        return self.history[0][2] if self.history else default

    def _lose(self):
        # the resting size and the lever belong to the hand and the desk, not to this visit: kept, so
        # a returning hand is judged against them (relearning them from its first, worst frames made
        # the cursor creep for ~2 s). A stale rest size is replaced after arrive_max_ms.
        self.lifted = False
        self.sensor_w, self.sensor_off = 1.0, np.zeros(2)
        self.reacquired = True
        self.lift_trace.clear()
        self.edge_armed, self.edge_depth = False, 0.0
        self.filt.reset()
        self.prev_pos = self.emitted = self.last_raw = self.t_accepted = None
        self.vel[:] = 0
        self.speed = self.gain_speed = 0.0
        self.acc[:] = 0
        self.base = None
        self.window.clear()
        self.press_cnt = 0
        self.recent.clear()

    def _release(self):
        self.out.release_all()
        self.pressing = False
        if self.state in (PINCH, DRAG, HOLD, SCROLL):
            self.state = TRACK

    def status(self):
        hold = 0.0
        if self.state == PINCH and self.t_last is not None:
            hold = min(max((self.t_last - self.onset_t) / (self.click["long_ms"] / 1000), 0.0), 1.0)
        return {"state": self.state, "speed": self.speed, "lifted": self.lifted, "scale": self.scale_f,
                "ratio": self.ratio, "enter": self.click["enter"], "exit": self.click["exit"],
                "fist": self.poses.get("fist", (False,))[0], "two": self.poses.get("two", (False,))[0], "event": self.last_event, "hold": hold,
                "edge": self.edge_depth, "edge_dir": (float(self.edge_dir[0]), float(-self.edge_dir[1])),
                "rate_scroll": self.rate_scroll, "axes_deg": math.degrees(self.yaw_off)}
