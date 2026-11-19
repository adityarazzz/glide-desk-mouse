"""First-run setup wizard (PRD J1, doc 10.1): camera -> placement -> calibrate -> learn -> practice.

1 Camera     probe every camera, pick the one with the best frame rate and light
2 Placement  live checklist: light, fps, whole hand in view, room to move, tracking confidence
3 Calibrate  desk orientation + personal pinch depth (glide.calibrate)
4 Learn      each gesture taught live against a virtual screen; nothing is injected yet
5 Practice   ISO 9241-411 multi-directional target task with the real cursor -> throughput
"""
import json
import math
import time

import cv2
import numpy as np

from .calibrate import Calibrator, describe, orientation
from .camera import Camera, open_capture
from . import demo
from .config import CONFIG_PATH, ROOT, _merge, save_profile
from .engine import Engine
from .feedback import play, tick
from .hud import STATE_COLORS, draw_edge_band, draw_hand, text
from .replay import VirtualOut
from .tracker import HandTracker
from .workspace import fit_workspace, hand_box

WIN = "Glide setup"
W, H = 960, 600
GREEN, RED, AMBER, CYAN, GREY = (90, 210, 90), (80, 80, 230), (0, 180, 255), (255, 255, 0), (150, 150, 150)
STEPS = ["Camera", "Placement", "Calibrate", "Learn", "Practice"]


class Abort(Exception):
    pass


def blank():
    return np.full((H, W, 3), 28, np.uint8)


def show(img):
    cv2.imshow(WIN, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        raise Abort
    return k


def header(img, step, sub=""):
    cv2.rectangle(img, (0, 0), (W, 74), (45, 45, 45), -1)
    text(img, f"Glide setup  -  {STEPS[step]}", (20, 34), 0.85, CYAN, 2)
    for i, name in enumerate(STEPS):
        x = W - 420 + i * 84
        col = GREEN if i < step else CYAN if i == step else GREY
        cv2.circle(img, (x, 26), 7, col, -1 if i <= step else 1, cv2.LINE_AA)
        text(img, name, (x - 30, 50), 0.38, col)
    if sub:
        text(img, sub, (20, 64), 0.5)
    text(img, "ESC exits setup", (W - 140, H - 10), 0.4, GREY)


def paste(img, frame, x, y, w):
    h = int(frame.shape[0] * w / frame.shape[1])
    img[y:y + h, x:x + w] = cv2.resize(frame, (w, h))
    return h


def mark(img, ok, x, y):
    if ok is True:
        cv2.line(img, (x, y), (x + 6, y + 7), GREEN, 3, cv2.LINE_AA)
        cv2.line(img, (x + 6, y + 7), (x + 18, y - 8), GREEN, 3, cv2.LINE_AA)
    elif ok is False:
        cv2.line(img, (x, y - 7), (x + 14, y + 7), RED, 3, cv2.LINE_AA)
        cv2.line(img, (x + 14, y - 7), (x, y + 7), RED, 3, cv2.LINE_AA)
    else:
        cv2.circle(img, (x + 7, y), 6, GREY, 1, cv2.LINE_AA)


# ---- step 1: camera ---------------------------------------------------------------------

def probe(cam_cfg, idx):
    cap = open_capture(idx, cam_cfg)
    if cap is None:
        return None
    ok, last = cap.read()
    if not ok:
        cap.release()
        return None
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        ok, f = cap.read()
        if ok:
            n, last = n + 1, f
    cap.release()
    bright = float(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY).mean())
    fps = n / 1.2
    score = 2 * (fps >= 24) + (50 <= bright <= 210) + fps / 30 + bright / 255
    return {"index": idx, "fps": fps, "bright": bright, "score": score, "thumb": last}


MODES = [(1280, 720), (640, 360), (640, 480)]   # ties go to the first: capture high, average down


def coverage(frame, ref, orb, bf):
    """Area of `frame`'s view as a fraction of the reference view, via ORB feature homography."""
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kr, dr = orb.detectAndCompute(g(ref), None)
    k, d = orb.detectAndCompute(g(frame), None)
    if dr is None or d is None:
        return None
    mt = sorted(bf.match(d, dr), key=lambda x: x.distance)[:400]
    if len(mt) < 20:
        return None
    Hm, inl = cv2.findHomography(np.float32([k[x.queryIdx].pt for x in mt]),
                                 np.float32([kr[x.trainIdx].pt for x in mt]), cv2.RANSAC, 4.0)
    if Hm is None or inl.sum() < 15:
        return None
    h, w = frame.shape[:2]
    c = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), Hm).reshape(-1, 2)
    return float(cv2.contourArea(c)) / (ref.shape[0] * ref.shape[1])


def widest_mode(cam_cfg, idx):
    """Many webcams crop the sensor in 4:3 modes (the ZEB shows 76% of the width at 640x480).
    Measure each candidate mode's real view against a 1280x720 reference; keep the widest at 24+ fps."""
    def grab(w, h):
        cap = open_capture(idx, dict(cam_cfg, width=w, height=h))
        if cap is None:
            return None, 0.0
        f = None
        for _ in range(12):
            ok, f = cap.read()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 0.6:
            n += cap.read()[0]
        cap.release()
        return f, n / 0.6
    ref, _ = grab(1280, 720)
    if ref is None:
        return None
    orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best = None
    for w, h in MODES:
        f, fps = grab(w, h)
        if f is None or fps < 24:
            continue
        area = coverage(f, ref, orb, bf)
        if area is not None and (best is None or area > best[0] + 0.02):
            best = (area, (f.shape[1], f.shape[0]))
    return best


def pick_camera(cfg):
    img = blank()
    header(img, 0, "Looking for cameras...")
    show(img)
    found = [c for c in (probe(cfg["camera"], i) for i in range(5)) if c]
    if not found:
        raise RuntimeError("No camera could be opened. Plug in the camera and close apps using it.")
    best = max(found, key=lambda c: c["score"])
    while True:
        img = blank()
        header(img, 0, "Press the camera's number, or ENTER for the recommended one.")
        tw = min(300, (W - 40) // len(found) - 10)
        for j, c in enumerate(found):
            x = 20 + j * (tw + 10)
            h = paste(img, c["thumb"], x, 110, tw)
            col = GREEN if c is best else GREY
            cv2.rectangle(img, (x - 2, 108), (x + tw + 2, 112 + h), col, 2)
            text(img, f"[{c['index']}]{'  recommended' if c is best else ''}", (x, 140 + h), 0.6, col, 2)
            text(img, f"{c['fps']:.0f} fps   light {c['bright']:.0f}/255", (x, 166 + h), 0.5)
            if c["fps"] < 24:
                text(img, "slow - needs more light", (x, 190 + h), 0.45, AMBER)
        text(img, "Use the camera that looks DOWN at your desk, beside the keyboard.", (20, H - 60), 0.55)
        k = show(img)
        chosen = best["index"] if k in (13, 10) else next((c["index"] for c in found if k == ord(str(c["index"]))), None)
        if chosen is not None:
            img = blank()
            header(img, 0, "Measuring the field of view of each capture mode...")
            show(img)
            mode = widest_mode(cfg["camera"], chosen)
            if mode:
                cfg["camera"]["width"], cfg["camera"]["height"] = mode[1]
            return chosen


# ---- step 2: placement ------------------------------------------------------------------

def room(seen, cfg, screen_wh):
    """(travel_mm, gain_scale, raise_cm) for the hand as it rests now."""
    R = orientation(seen, cfg["hand"]["dominant"])
    mm_px = 1000.0 / float(np.median([o.px_per_m for o in seen]))
    w = fit_workspace(R, seen[-1].frame_wh, hand_box(seen), mm_px, cfg["gain"], cfg["edge"], screen_wh)
    return w["travel_mm"], w["gain_scale"], w["raise_cm"]


def placement_checks(frame, window, fps, cfg=None, screen_wh=(1920, 1080)):
    seen = [o for o in window if o.present]
    bright = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    h, w = frame.shape[:2]
    present = len(seen) / max(len(window), 1)
    score = float(np.median([o.score for o in seen])) if seen else 0.0
    inside = bool(seen) and all(np.all((o.img > [0.02 * w, 0.02 * h]) & (o.img < [0.98 * w, 0.98 * h])) for o in seen[-10:])
    span = float(np.median([np.linalg.norm(o.img[0] - o.img[12]) for o in seen])) / h if seen else 0.0
    if seen and cfg is not None:
        tr, scale, raise_cm = room(seen, cfg, screen_wh)
        room_check = (f"Room to slide {tr[0] / 10:.0f} x {tr[1] / 10:.0f} cm", scale < 2.5,
                      f"Raise the camera about {raise_cm} cm for more room" if raise_cm
                      else "Move the hand to the middle of the view")
    else:
        room_check = ("Room to slide", seen and span <= 0.62, "Raise the camera: hand fills the view")
    return [
        ("Light", 60 <= bright <= 215, "Too dark: switch on a desk lamp" if bright < 60 else "Too bright: avoid glare"),
        ("Frame rate 24+ fps", fps >= 24, "Camera slow: usually too little light"),
        ("Hand seen steadily", present >= 0.9, "Rest your hand flat under the camera"),
        ("Tracking confidence", score >= 0.9, "Show the whole hand; avoid hard shadows"),
        ("Whole hand inside view", inside, "Fit wrist and fingertips inside the view"),
        room_check,
        ("Hand big enough", seen and span >= 0.25, "Lower the camera: hand looks small"),
    ]


def placement(cam, tracker, cfg, screen_wh):
    window, seq, all_ok_since, forced = [], 0, None, False
    while True:
        got = cam.get(seq)
        if got is None:
            continue
        seq, frame, t = got
        obs = tracker.detect(frame, t)
        window = (window + [obs])[-45:]
        checks = placement_checks(frame, window, cam.fps, cfg, screen_wh)
        passed = sum(bool(ok) for _, ok, _ in checks)
        score = int(100 * passed / len(checks))
        img = blank()
        header(img, 1, "Rest your hand flat on the desk where you want to use it.")
        view = frame.copy()
        draw_hand(view, obs)
        paste(img, view, 20, 90, 560)
        text(img, f"Placement score {score}", (600, 120), 0.8, GREEN if score >= 85 else AMBER, 2)
        for i, (name, ok, tip) in enumerate(checks):
            y = 160 + i * 52
            mark(img, bool(ok), 600, y)
            text(img, name, (630, y + 6), 0.55, (230, 230, 230))
            if not ok:
                text(img, tip, (630, y + 26), 0.42, AMBER)
        if score == 100:
            all_ok_since = all_ok_since or t
            text(img, "All good - continuing...", (600, H - 40), 0.6, GREEN, 2)
            if t - all_ok_since > 2.0:
                return
        else:
            all_ok_since = None
            text(img, "ENTER to continue anyway" if not forced else "ENTER again to continue anyway",
                 (600, H - 40), 0.55, GREY)
        k = show(img)
        if k in (13, 10):
            if score >= 70 or forced:
                return
            forced = True


# ---- step 3: calibrate ------------------------------------------------------------------

def demo_panel(img, x, y, w, h, name, t, g=None, hand="right"):
    """Animated 'do this' hand beside the camera view (glide.demo)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 44), -1)
    text(img, "DO THIS", (x + 10, y + 20), 0.45, GREY)
    demo.draw(img[y + 26:y + h - 4, x + 4:x + w - 4], demo.DEMOS[name], t, g=g, hand=hand,
              scale=min(w, h) / 330)


def calibrate(cam, tracker, cfg, screen_wh):
    cal, seq, t0, beats = Calibrator(cfg, screen_wh), 0, None, None
    while True:
        got = cam.get(seq)
        if got is None:
            continue
        seq, frame, t = got
        t0 = t if t0 is None else t0
        obs = tracker.detect(frame, t)
        n_cues = len(cal.cues)
        done = cal.feed(obs, t)
        k = cal.countdown(t)
        if k != beats and k:
            play("beat")
        if len(cal.cues) != n_cues:
            play("go")
        beats = k
        img = blank()
        header(img, 2, cal.prompt)
        view = frame.copy()
        draw_hand(view, obs)
        h = paste(img, view, 20, 90, 580)
        cv2.rectangle(img, (20, 575), (20 + int(min(cal.progress(), 1) * 580), 585), CYAN, -1)
        if cal.step == "clicks":
            g = demo.envelope(t - cal.cues[-1][0]) if cal.cues and cal.phase == "sample" else 0.0
            demo_panel(img, 620, 90, 320, h, "pinch_mid", t - t0, g, cfg["hand"]["dominant"])
        else:
            demo_panel(img, 620, 90, 320, h, "rest", t - t0, None, cfg["hand"]["dominant"])
        c = (310, 90 + h // 2)
        if k:
            text(img, str(k), (c[0] - 22, c[1] + 25), 2.4, CYAN, 5)
        elif cal.flash(t):
            text(img, "GO", (c[0] - 55, c[1] + 30), 2.6, GREEN, 5)
        elif cal.waiting():
            text(img, "open your hand", (c[0] - 95, c[1] + 10), 0.8, (230, 230, 230), 2)
        if cal.message:
            text(img, cal.message, (30, 120), 0.55, GREEN)
        if show(img) == ord(" "):
            cal.skip()
        if done:
            return done


# ---- step 4: learn the gestures ---------------------------------------------------------

LESSONS = [   # name, instruction, passed, demo
    ("Move", "Slide your hand around on the desk", lambda s: s["travel"] > 700, "move_slow"),
    ("Reach", "Slide toward the EDGE of the view and hold: the cursor keeps gliding", lambda s: s["edge"] > 20,
     "reach"),
    ("Reposition", "Lift your hand a few centimetres, carry it back, put it down", lambda s: s["lift"] >= 1,
     "reposition"),
    ("Click", "Quick pinch: thumb tip to MIDDLE fingertip, then open. Twice.", lambda s: s["left click"] >= 2,
     "pinch_mid"),
    ("Right click", "Pinch and HOLD STILL until the ring fills up", lambda s: s["right click"] >= 1, "hold"),
    ("Drag", "Pinch, keep holding, and slide your hand", lambda s: s["drag"] >= 1, "drag"),
    ("Scroll", "Index + middle straight, ring + little finger curled: slide forward / back",
     lambda s: s["scroll"] >= 1, "two_scroll"),
]
FIST_SCROLL = ("Scroll", "Make a FIST and move it forward / back (grab the page)", lambda s: s["scroll"] >= 1,
               "fist")


def learn(cam, tracker, cfg, profile):
    out = VirtualOut()
    eng = Engine(cfg, profile, out)
    seq, i, passed_at, seen = 0, 0, None, eng.event_seq
    stats = {"travel": 0.0, "edge": 0, "left click": 0, "right click": 0, "drag": 0, "scroll": 0, "lift": 0}
    lessons = LESSONS[:-1] + [FIST_SCROLL if cfg["scroll"]["gesture"] == "fist" else LESSONS[-1]]
    done = [False] * len(lessons)
    t_lesson = None
    last_pos = np.array(out.cursor(), float)
    while i < len(lessons):
        got = cam.get(seq)
        if got is None:
            continue
        seq, frame, t = got
        obs = tracker.detect(frame, t)
        out.t = t
        eng.update(obs, t)
        pos = np.array(out.cursor(), float)
        stats["travel"] += float(np.hypot(*(pos - last_pos)))
        stats["edge"] += eng.edge_depth > 0
        last_pos = pos
        if eng.event_seq != seen:
            seen = eng.event_seq
            if eng.last_event in stats:
                stats[eng.last_event] += 1
            tick()
        if passed_at is None and lessons[i][2](stats):
            done[i], passed_at = True, t
            tick()
        if passed_at is not None and t - passed_at > 0.8:
            i, passed_at, t_lesson = i + 1, None, None
            stats = {k: 0 for k in stats}
            stats["travel"] = 0.0

        img = blank()
        header(img, 3, (lessons[i][1] if i < len(lessons) else "") + "   (SPACE = skip)")
        view = frame.copy()
        draw_edge_band(view, cfg, eng.edge_depth > 0)
        draw_hand(view, obs)
        h = paste(img, view, 20, 90, 440)
        if i < len(lessons):
            t_lesson = t if t_lesson is None else t_lesson
            demo_panel(img, 20, 100 + h, 440, H - 130 - h, lessons[i][3], t - t_lesson, None,
                       cfg["hand"]["dominant"])
        sx, sy, sw = 490, 90, 450                                     # virtual screen
        sh = int(sw * out.h / out.w)
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (70, 70, 70), -1)
        cx, cy = int(sx + pos[0] * sw / out.w), int(sy + pos[1] * sh / out.h)
        st = eng.status()
        col = STATE_COLORS.get(st["state"], GREY)
        cv2.circle(img, (cx, cy), 10, col, -1 if st["state"] in ("PINCH", "DRAG", "HOLD") else 2, cv2.LINE_AA)
        if st["state"] == "PINCH":
            cv2.ellipse(img, (cx, cy), (16, 16), -90, 0, 360 * st["hold"], AMBER, 3, cv2.LINE_AA)
        text(img, st["state"], (sx + 8, sy + 20), 0.5, col)
        for j, (name, _, _, _) in enumerate(lessons):
            y = sy + sh + 40 + j * 34
            mark(img, True if done[j] else None, sx, y)
            text(img, name, (sx + 30, y + 6), 0.6, CYAN if j == i else (220, 220, 220), 2 if j == i else 1)
        if show(img) == ord(" ") and i < len(lessons):
            i, passed_at, t_lesson = i + 1, None, None


# ---- step 5: practice (ISO 9241-411 multi-directional task) -------------------------------

N_TARGETS, RADIUS, TARGET_W = 9, 190, 56
ORDER = [(k * (N_TARGETS // 2 + 1)) % N_TARGETS for k in range(N_TARGETS + 1)]


def throughput(trials):
    """Effective throughput (bits/s): IDe / MT with We = 4.133 * SD of endpoint error on the axis."""
    if len(trials) < 4:
        return None, None
    dx, de, mt = [], [], []
    for a, b, c, t in trials:
        u = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        proj = float((c - a) @ u)
        dx.append(proj - float(np.linalg.norm(b - a)))
        de.append(proj)
        mt.append(t)
    we = max(4.133 * float(np.std(dx)), 1.0)
    ide = math.log2(np.mean(de) / we + 1)
    return ide / float(np.mean(mt)), float(np.mean(mt))


def practice(cam, tracker, cfg, profile):
    from .feedback import Overlay
    from .output_win import KeyWatcher, WinOutput
    out = WinOutput()
    eng = Engine(cfg, profile, out)
    keys = KeyWatcher(cfg["guard"]["typing_ms"])
    overlay = Overlay(out.cursor) if cfg["feedback"]["overlay"] else None
    if overlay:
        overlay.start()
    centre = np.array([W / 2, H / 2 + 40])
    targets = [centre + RADIUS * np.array([math.sin(2 * math.pi * k / N_TARGETS), -math.cos(2 * math.pi * k / N_TARGETS)])
               for k in range(N_TARGETS)]
    clicks = []
    cv2.setMouseCallback(WIN, lambda ev, x, y, *_: clicks.append((time.perf_counter(), np.array([x, y], float)))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    k_i, trials, errors, prev, seq, seen = 0, [], 0, None, 0, eng.event_seq
    result_until = None
    try:
        while True:
            got = cam.get(seq)
            if got is None:
                continue
            seq, frame, t = got
            obs = tracker.detect(frame, t)
            typing, events = keys.poll()
            if "quit" in events:
                raise Abort
            eng.update(obs, t, typing)
            if overlay:
                st = eng.status()
                overlay.set(st["state"], st["hold"])
            if eng.event_seq != seen:
                seen = eng.event_seq
                if cfg["feedback"]["sound"]:
                    tick()
            while clicks and result_until is None:
                tc, p = clicks.pop(0)
                target = targets[ORDER[k_i]]
                if np.linalg.norm(p - target) > TARGET_W / 2:
                    errors += k_i > 0
                if prev is not None:
                    trials.append((prev[1], target, p, tc - prev[0]))
                prev = (tc, p)
                k_i += 1
                if k_i >= len(ORDER):
                    result_until = time.perf_counter() + 6.0
            img = blank()
            header(img, 4, "Move with your hand and pinch on the highlighted circle.  (SPACE = skip)")
            paste(img, frame, W - 200, H - 170, 180)
            for j, c in enumerate(targets):
                active = result_until is None and j == ORDER[min(k_i, len(ORDER) - 1)]
                cv2.circle(img, tuple(c.astype(int)), TARGET_W // 2, GREEN if active else (80, 80, 80),
                           -1 if active else 2, cv2.LINE_AA)
            if k_i == 0:
                text(img, "Click the green circle to start", (W // 2 - 150, int(centre[1]) + 6), 0.6, CYAN)
            if result_until is not None:
                tp, mt = throughput(trials)
                rate = errors / max(len(trials), 1)
                profile["practice"] = {"throughput_bits_s": tp, "mean_time_s": mt, "error_rate": rate}
                cv2.rectangle(img, (W // 2 - 260, 230), (W // 2 + 260, 380), (40, 40, 40), -1)
                text(img, f"Throughput {tp:.2f} bits/s" if tp else "Done", (W // 2 - 230, 275), 0.9, GREEN, 2)
                text(img, f"{mt:.2f} s per target, {rate:.0%} misses" if mt else "", (W // 2 - 230, 315), 0.6)
                text(img, "(ISO 9241-411; a mouse usually scores about 4-5 bits/s)", (W // 2 - 230, 350), 0.45, GREY)
                if time.perf_counter() > result_until:
                    return profile.get("practice")
            if show(img) == ord(" "):
                return None
    finally:
        out.release_all()
        if overlay:
            overlay.stop()
        cv2.setMouseCallback(WIN, lambda *a: None)


# ---- wizard -----------------------------------------------------------------------------

def save_camera(cam_cfg):
    over = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    _merge(over, {"camera": {k: cam_cfg[k] for k in ("index", "width", "height")}})
    CONFIG_PATH.write_text(json.dumps(over, indent=2), encoding="utf-8")


def run_setup(cfg):
    """Returns the new profile, or None if the user left setup."""
    from .output_win import WinOutput
    screen_wh = WinOutput().screen_size()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cam = tracker = None
    try:
        cfg["camera"]["index"] = pick_camera(cfg)
        save_camera(cfg["camera"])
        cam = Camera(cfg["camera"])
        cam.start()
        tracker = HandTracker(ROOT / cfg["model"], cfg["hand"])
        placement(cam, tracker, cfg, screen_wh)
        profile = calibrate(cam, tracker, cfg, screen_wh)
        save_profile(profile)
        learn(cam, tracker, cfg, profile)
        practice(cam, tracker, cfg, profile)
        save_profile(profile)
        print("Setup complete:", describe(profile))
        return profile
    except Abort:
        return None
    finally:
        if cam:
            cam.stop()
        if tracker:
            tracker.close()
        cv2.destroyWindow(WIN)
