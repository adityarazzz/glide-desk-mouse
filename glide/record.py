"""Coached session recorder: self-paced gesture steps -> logs/session_*.jsonl for glide.analyze.

Each step shows an animated demo next to your own (turned) camera view, waits until you are
ready, counts down 3-2-1 with sound, then tells you whether the gesture was seen. See
glide.coach for the design rules. Landmarks and timestamps only; camera frames are never written
(NFR-06). Nothing is injected.

Keys: SPACE start / continue, P pause, R redo step, S skip step, ESC stop and keep finished steps.
"""
import json
import time

import cv2

from . import demo, ui
from .coach import CHECKS, Coach, Step, gesture_of
from .config import ROOT, load_profile
from .demo import DEMOS, View, envelope
from .feedback import play
from .hud import draw_hand

GET_READY = Step("get_ready", "Get set", "Rest your hand flat on the desk, fingers pointing away from you.",
                 kind="still", seconds=1.5, demo="rest")
RELAX = Step("hover_rest", "Relax", "Rest your hand however feels natural, as if on a mouse.", seconds=5)
PINCH_TIP = "Touch the tips together firmly, then open wide."
BASIC = [
    GET_READY,
    Step("rest", "Stay still", "Keep your hand flat and completely still.", kind="still", seconds=6, check="still"),
    Step("move_slow", "Slow circles", "Slide your hand slowly in circles on the desk.", seconds=7, check="moving"),
    Step("move_fast", "Quick flicks", "Flick your hand quickly left-right, then forward-back.", seconds=5,
         check="fast", tip="Faster: short, snappy flicks."),
    Step("pinch_idx", "Pinch with index", "On GO, touch your thumb tip to your index fingertip, then open.",
         kind="cued", tip=PINCH_TIP),
    Step("pinch_mid", "Pinch with middle", "On GO, touch your thumb tip to your middle fingertip, then open.",
         kind="cued", tip=PINCH_TIP),
    Step("tap", "Finger tap", "On GO, lift your index finger a little and tap the desk.", kind="cued",
         tip="Lift the finger about 2 cm before tapping."),
    Step("lift", "Lift the hand", "On GO, lift your whole hand up high, then put it back down.", kind="cued",
         tip="Lift higher, toward the camera."),
    Step("fist", "Fist", "On GO, close your hand into a fist, then open it flat again.", kind="cued",
         tip="Close all fingers fully."),
    RELAX,
]
# different hand angles and long scrolls: the cases that broke in real use
ANGLES = [
    GET_READY,
    Step("rest", "Stay still", "Keep your hand flat and completely still.", kind="still", seconds=5, check="still"),
    Step("rot_left", "Turned left", "Turn your hand about 45 degrees to the left and slide it around.",
         seconds=7, check="turned"),
    Step("rot_left_pinch", "Turned left: pinch", "Keep it turned. On GO, pinch thumb and middle finger.",
         kind="cued", reps=4, demo="rot_left_pinch", tip=PINCH_TIP),
    Step("rot_right", "Turned right", "Turn your hand about 45 degrees to the right and slide it around.",
         seconds=7, check="turned"),
    Step("rot_right_pinch", "Turned right: pinch", "Keep it turned. On GO, pinch thumb and middle finger.",
         kind="cued", reps=4, tip=PINCH_TIP),
    Step("side", "On its side", "Tilt your hand onto its little-finger edge, like a handshake, and slide.",
         seconds=7, check="side"),
    Step("side_pinch", "On its side: pinch", "Stay on the side. On GO, pinch thumb and middle finger.",
         kind="cued", reps=4, tip=PINCH_TIP),
    Step("grip", "Mouse grip", "Curl your fingers as if holding a mouse, and slide around.", seconds=7,
         check="curled"),
    Step("grip_pinch", "Mouse grip: pinch", "Keep the grip. On GO, pinch thumb and middle finger.",
         kind="cued", reps=4, tip=PINCH_TIP),
    Step("scroll_long", "Long scroll", "Scroll pose: push forward and hold, then pull back and hold.",
         seconds=10, check="two", tip="Index + middle straight, ring + little finger tucked under."),
    RELAX,
]
# fine pointing on the current capture pipeline + the two-finger scroll pose vs a curled grip
PRECISION = [
    GET_READY,
    Step("rest", "Stay still", "Keep your hand completely still.", kind="still", seconds=8, check="still"),
    Step("fine", "Tiny movements", "Make tiny, slow movements, as if aiming at a small button.", seconds=10),
    Step("move_slow", "Slow circles", "Slide your hand slowly in circles on the desk.", seconds=7, check="moving"),
    Step("move_fast", "Quick flicks", "Flick your hand quickly left-right, then forward-back.", seconds=6,
         check="fast"),
    Step("pinch_mid", "Quick pinch", "On GO, a QUICK pinch: thumb tip to middle fingertip, then open.",
         kind="cued", tip=PINCH_TIP),
    Step("two_scroll", "Scroll pose", "Index + middle straight, ring + little finger curled. Slide forward/back.",
         seconds=10, check="two", tip="Tuck the ring and little finger fully under the palm."),
    Step("grip_move", "Mouse grip", "Curl your fingers as if holding a mouse, and slide around.", seconds=8,
         check="curled"),
    RELAX,
]
# short check of the scroll pose against the poses it must not be confused with
SCROLL = [
    GET_READY,
    Step("rest", "Stay still", "Keep your hand flat and still.", kind="still", seconds=3, check="still"),
    PRECISION[6],
    PRECISION[7],
    Step("fist", "Fist", "On GO, close your hand into a fist, then open it flat again.", kind="cued", reps=4),
    RELAX,
]
# lift-to-reposition against the poses it must not be confused with
REPOSITION = [
    GET_READY,
    Step("move_slow", "Slide around", "Slide your hand slowly around on the desk.", seconds=5, check="moving"),
    Step("reposition", "Lift and carry back", "On GO, lift your hand a few centimetres, carry it back toward you, "
         "and put it down.", kind="cued", reps=6, tip="Lift a little higher, like lifting a mouse off the pad."),
    Step("pinch_mid", "Quick pinch", "On GO, a quick pinch: thumb tip to middle fingertip, then open.",
         kind="cued", reps=4, tip=PINCH_TIP),
    Step("fist", "Fist", "On GO, close your hand into a fist, then open it flat again.", kind="cued", reps=3),
    Step("grip_move", "Mouse grip", "Curl your fingers as if holding a mouse, and slide around.", seconds=6,
         check="curled"),
    RELAX,
]
# your real mouse under the camera: Glide records the hand and the real cursor (glide.mousefit)
MOUSE_HOW = "Use your real mouse under the camera"
MOUSE = [
    Step("mouse_hold", "Hold your mouse", "Put your mouse under the camera and hold it the way you always do.",
         kind="still", seconds=2, demo="grip_move", practice=False),
    Step("mouse_free", "Use it as usual", "Move the pointer around the screen the way you normally do: "
         "across, up and down, big moves and small ones.", seconds=20, demo="grip_move", practice=False,
         tip="Keep your hand and the mouse under the camera."),
    Step("mouse_across", "Left and right", "Sweep the pointer left and right across the screen a few times.",
         seconds=8, demo="grip_move", practice=False),
    Step("mouse_updown", "Up and down", "Now sweep it up and down the screen a few times.",
         seconds=8, demo="grip_move", practice=False),
    Step("mouse_small", "Small moves", "Make small, careful moves, as if pointing at single words.",
         seconds=8, demo="grip_move", practice=False),
]
SCRIPTS = {"basic": BASIC, "angles": ANGLES, "precision": PRECISION, "scroll": SCROLL, "reposition": REPOSITION,
           "mouse": MOUSE}
WINDOW = "Glide record"
W, H = 1040, 660
CAM = (24, 104, 600, 420)
DEMO = (648, 104, 368, 420)
KEYS = "SPACE start    P pause    R redo step    S skip step    ESC stop & save"
NOW_WORDS = {"pinch8": "Pinch!", "pinch12": "Pinch!", "tap": "Tap!", "lift": "Lift!", "fist": "Fist!"}


class Screen:
    def __init__(self, hand="right"):
        self.hand = hand
        self.toast, self.toast_t = None, -9.0
        self.step_t0, self.step_i = 0.0, -1

    def draw(self, coach, frame, obs, view, t):
        img, ts = ui.canvas(W, H), ui.Typeset()
        s = coach.step
        if coach.i != self.step_i:
            self.step_i, self.step_t0 = coach.i, t
        n = len(coach.steps)

        # header: where you are
        ts.add(f"Step {min(coach.i + 1, n)} of {n}", (24, 16), 16, ui.DIM)
        ts.add(s.title if s else "All done", (24, 38), 34, ui.WHITE, bold=True)
        for j in range(n):
            c = (W - 24 - (n - j) * 22 + 8, 34)
            if j < coach.i:
                cv2.circle(img, c, 7, ui.GREEN, -1, cv2.LINE_AA)
            elif j == coach.i:
                cv2.circle(img, c, 8, ui.CYAN, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 7, ui.GREY, 2, cv2.LINE_AA)

        # your hand, turned so the fingers point up like in the demo
        ui.panel(img, *CAM)
        v = frame.copy()
        draw_hand(v, obs)
        ox, oy, sc = ui.fit(img, view.frame(v), CAM[0] + 8, CAM[1] + 30, CAM[2] - 16, CAM[3] - 38)
        ts.add("YOUR HAND", (CAM[0] + 14, CAM[1] + 8), 14, ui.DIM, bold=True)
        cx, cy = CAM[0] + CAM[2] // 2, CAM[1] + CAM[3] // 2 + 14

        # the demo: what to do
        ui.panel(img, *DEMO)
        ts.add("DO THIS", (DEMO[0] + 14, DEMO[1] + 8), 14, ui.DIM, bold=True)
        if s is not None:
            d = DEMOS.get(s.demo or s.label, DEMOS["rest"])
            sub = img[DEMO[1] + 30:DEMO[1] + DEMO[3] - 30, DEMO[0] + 4:DEMO[0] + DEMO[2] - 4]
            g = None
            if s.kind == "cued" and coach.state in ("ready", "count", "respond", "result"):
                g = envelope(coach.since_go(), *d.env) if coach.state in ("respond", "result") else 0.0
            demo.draw(sub, d, t - self.step_t0, g=g, hand=self.hand, scale=1.12)
            note = d.note(t - self.step_t0) if d.note else None
            if note:
                ts.add(note, (DEMO[0] + DEMO[2] // 2, DEMO[1] + DEMO[3] - 18), 18, ui.AMBER, True, "mm")

        # instruction + status
        y = CAM[1] + CAM[3] + 18
        if s is not None:
            for k, line in enumerate(ts.wrap(s.how, W - 48, 22)):
                ts.add(line, (24, y + k * 28), 22, ui.WHITE)
        status, col = self._status(coach, obs, ts, img, (cx, cy), gesture_of(s.label) if s else None)
        ts.add(status, (24, H - 70), 22, col, bold=True)
        if s is not None and s.kind == "cued":
            self._trials(img, ts, coach)
        if t - self.toast_t < 1.8 and self.toast:
            ui.panel(img, W - 330, H - 82, 306, 40, (40, 70, 45))
            ts.add(self.toast, (W - 177, H - 62), 17, ui.WHITE, False, "mm")
        if coach.paused:
            cv2.rectangle(img, (CAM[0], cy - 40), (CAM[0] + CAM[2], cy + 40), (20, 20, 20), -1)
            ts.add("Paused  -  press P to continue", (cx, cy), 26, ui.AMBER, True, "mm")
        ts.add(KEYS, (W // 2, H - 18), 14, ui.GREY, False, "mm")
        return ts.render(img)

    def _status(self, coach, obs, ts, img, c, gesture):
        st, s = coach.state, coach.step
        if s is None:
            return "Saving...", ui.GREEN
        if coach.paused:
            return "Paused", ui.AMBER
        if not obs.present and st in ("intro", "ready", "count", "run"):
            msg = "Hand out of view - the clock is paused" if st == "run" else "Put your hand under the camera"
            ts.add("No hand seen", c, 24, ui.AMBER, True, "mm")
            return msg, ui.AMBER
        if st == "intro":
            if coach.intro_left() > 0:
                return "Watch the demo on the right", ui.DIM
            ui.ring(img, c, 46, coach.fill(), ui.CYAN)
            ts.add("hold still", c, 16, ui.WHITE, False, "mm")
            return "Ready? Rest your hand still for a moment to start  (or press SPACE)", ui.CYAN
        if st == "ready":
            return "Open your hand and hold it steady", ui.DIM
        if st == "count":
            k = coach.countdown()
            ui.ring(img, c, 60, 1 - ((coach.t - coach.t_state) % 0.6) / 0.6, ui.CYAN, 8)
            ts.add(str(k), c, 64, ui.CYAN, True, "mm")
            return ("Get ready..." if s.kind == "cued" else "Starting..."), ui.CYAN
        if st == "respond":
            word = NOW_WORDS.get(gesture, "Now!")
            if coach.hit:
                ui.check(img, c[0] - 18, c[1], True, 28)
                return "Seen", ui.GREEN
            if coach.since_go() < 0.6:
                ts.add("GO", c, 80, ui.GREEN, True, "mm")
            return word, ui.GREEN
        if st == "result":
            practice, hit = coach.results[-1]
            ui.check(img, c[0] - 18, c[1], hit, 28)
            pre = "Practice: " if practice else ""
            if hit:
                return pre + "Got it" + (" - now for real" if practice and coach.practice_left == 0 else ""), ui.GREEN
            return pre + "Not seen. " + (s.tip or "Try a bit bigger."), ui.AMBER
        if st == "run":
            frac = coach.fill()
            x0, x1, yb = CAM[0] + 16, CAM[0] + CAM[2] - 16, CAM[1] + CAM[3] - 10
            cv2.line(img, (x0, yb), (x1, yb), ui.GREY, 6, cv2.LINE_AA)
            cv2.line(img, (x0, yb), (int(x0 + (x1 - x0) * frac), yb), ui.CYAN, 6, cv2.LINE_AA)
            left = max(0.0, s.seconds - coach.good_s)
            ok = coach.live_ok()
            if ok is None:
                return f"Keep going - {left:.0f} s", ui.CYAN
            _, yes, no = CHECKS[s.check]
            ui.check(img, CAM[0] + 20, CAM[1] + 56, ok, 18)
            ts.add(yes if ok else no, (CAM[0] + 50, CAM[1] + 44), 20, ui.GREEN if ok else ui.AMBER, True)
            return (f"{yes} - {left:.0f} s" if ok else f"{no}. {s.tip}".strip()), ui.GREEN if ok else ui.AMBER
        return "", ui.DIM

    def _trials(self, img, ts, coach):
        s = coach.step
        x = W - 24 - s.reps * 26
        ts.add("Practice" if coach.practice_left > 0 else "Trials", (x - 12, H - 70), 18, ui.DIM, False, "ra")
        real = [hit for practice, hit in coach.results if not practice]
        for j in range(s.reps):
            c = (x + j * 26 + 9, H - 58)
            if j < len(real):
                cv2.circle(img, c, 9, ui.GREEN if real[j] else ui.AMBER, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 9, ui.CYAN if j == len(real) else ui.GREY, 2, cv2.LINE_AA)


def record(cam, tracker, script="basic", hand="right"):
    steps = SCRIPTS[script]
    profile = load_profile()
    coach = Coach(steps, profile.get("R") if profile else None, hand)
    screen, seq, view_R = Screen(hand), 0, None
    view = View(coach.R, cam.size)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while not coach.done:
        got = cam.get(seq)
        if got is None:
            continue
        seq, frame, t_cap = got
        obs = tracker.detect(frame, t_cap)
        row = frame_row(frame, tracker, t_cap)
        i = coach.i
        coach.feed(obs, t_cap, row)
        if coach.R is not view_R:                   # orientation measured: turn the view to match
            view_R, view = coach.R, View(coach.R, cam.size)
        for e in coach.events:
            if e != "start":
                play(e)
        if coach.i != i and coach.summary:
            rec = coach.summary[-1]
            screen.toast = (f"Done: {steps[i].title}" + (f"  -  {rec['seen']}/{rec['trials']} seen"
                                                         if "seen" in rec else ""))
            screen.toast_t = t_cap
        cv2.imshow(WINDOW, screen.draw(coach, frame, obs, view, t_cap))
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            del coach.rows[coach.row0:]             # keep only finished steps
            del coach.cues[coach.cue0:]
            break
        if k == ord(" "):
            coach.go()
        elif k == ord("p"):
            coach.toggle_pause()
        elif k == ord("r"):
            coach.redo()
        elif k == ord("s"):
            coach.skip()
    cv2.destroyWindow(WINDOW)
    path = save_session(coach, script, hand)
    for r in coach.summary:
        print("  ", r)
    return path


def frame_row(frame, tracker, t, cursor=None):
    """One recorded frame: landmarks (and the real cursor position) only, never the image."""
    row = {"t": t, "wh": [frame.shape[1], frame.shape[0]]}
    if cursor is not None:
        row["cur"] = [int(cursor[0]), int(cursor[1])]
    if tracker.raw is not None:
        img, world, score = tracker.raw
        row.update(img=img.round(2).tolist(), world=world.round(5).tolist(), score=round(float(score), 3))
    return row


def save_session(coach, script, hand):
    """Writes the finished steps to logs/; None when nothing was finished."""
    if not [r for r in coach.summary if not r.get("skipped")]:
        return None
    (ROOT / "logs").mkdir(exist_ok=True)
    path = ROOT / "logs" / time.strftime(f"session_{script}_%Y%m%d_%H%M%S.jsonl")
    meta = {"meta": {"script": script, "recorder": "coach-1", "hand": hand, "summary": coach.summary}}
    lines = [meta] + coach.rows + [{"cue": c} for c in coach.cues]
    path.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    return path
