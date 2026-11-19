"""Glide as a background service: camera -> tracker -> engine on one thread, in one of three modes.

run     the hand drives the real cursor (or is paused)
setup   first-run flow / recalibration (glide.onboard); nothing is injected until practice
record  a coached training session (glide.coach)

The app UI never touches the pipeline directly. It posts commands, which run on the pipeline thread
between frames, and it reads the latest snapshot (JSON) and preview frame (JPEG) the thread publishes.
"""
import collections
import json
import queue
import threading
import time

import cv2
import numpy as np

import re

from .camera import Camera, find_working
from .camlist import cameras as list_cameras
from .camlist import index_of
from . import __version__, pointing
from .coach import CHECKS, Coach, gesture_of
from .config import CONFIG_PATH, ROOT, load_config, load_profile, save_profile
from .demo import View
from .engine import TRACK, Engine
from .features import Obs
from .feedback import OverlayProcess, play
from .onboard import STEPS, Calibrate, Learn, Placement
from .output_win import KeyWatcher, SmoothOutput, WinOutput
from .surface import contains, desk_side, detect, room_report, same_room
from .workspace import edge_band_px, fit_workspace
from .record import SCRIPTS, frame_row, save_session
from .tracker import HandTracker

# settings the app may change: dotted path -> kind (range, bool or choices)
SETTINGS = {
    "gain.speed": (0.5, 2.0), "click.long_ms": (600, 2000), "scroll.units_per_mm": (6.0, 30.0),
    "scroll.natural": bool, "scroll.gesture": ("two_finger", "fist"), "feedback.overlay": bool,
    "feedback.sound": bool, "hand.dominant": ("right", "left"), "edge.enabled": bool,
    "camera.index": int, "preset": ("", "accessibility", "fast"),
}
RECALIBRATE_AFTER = ("hand.dominant", "camera.index")
# a laptop's own camera faces the user, not the desk: without a remembered choice, prefer another
BUILT_IN = re.compile(r"integrated|built-?in|internal|front|facetime|user.?facing|\bir\b", re.I)
SCRIPT_INFO = {   # what each training session is for, shown in the app
    "mouse": ("Show Glide your mouse", "Needs a plugged-in mouse. Use it under the camera: Glide learns "
              "which part of your hand moves like it", 1),
    "reposition": ("Reposition", "Lifting the hand to move it back, against pinches and fists", 2.5),
    "scroll": ("Scroll check", "The scroll pose against a mouse grip and a fist", 1.5),
    "basic": ("Basic gestures", "Still, moving, pinches, taps, lifts and fists", 4),
    "precision": ("Precision", "Tiny movements, quick pinches and the scroll pose", 3),
    "angles": ("Hand angles", "Turned, tilted and curled hands, with pinches", 5),
}


def get_path(cfg, path):
    node = cfg
    for k in path.split("."):
        node = node.get(k) if isinstance(node, dict) else None
    return node


# Ctrl+Alt+X keeps this much. 6 s caught a click in only 1 of 18 reports (2026-10-28): reaching
# for the keys with the other hand takes longer than that
MISFIRE_KEEP_S = 15.0
STALL_REOPEN_S = 3.0      # no picture for this long: reconnect the camera
DARK_LEVEL = 12.0         # mean grey below this: the camera sees nothing (lights off, lens covered)
RECAL_GIVE_UP_S = 60.0    # recalibrating without a hand in view this long: keep the old calibration
DARK_SWITCH_S = 8.0       # only black this long: look for another camera that shows a picture
CAM_SEARCH_EVERY_S = 30.0  # at most this often (each look opens every camera briefly)


class DeskWatch:
    """State of the free-desk check (Service._watch_desk)."""
    EVERY_S = 3.0

    def __init__(self):
        self.palm = self.hull = None              # last clean hand position on the desk, image px
        self.last, self.busy, self.result = -1e9, False, None
        self.cand, self.agree, self.applied = None, 0, None


class Service:
    def __init__(self, paused=False):
        self.cfg = load_config(CONFIG_PATH)
        self.profile = load_profile()
        self.cam = None
        self.out = (SmoothOutput(lambda: self.cam.interval if self.cam else 1 / 30)
                    if self.cfg["pointer"].get("smooth_output", True) else WinOutput())
        self.screen_wh = self.out.screen_size()
        self.keys = KeyWatcher(self.cfg["guard"]["typing_ms"])
        self.tracker = HandTracker(ROOT / self.cfg["model"], self.cfg["hand"])
        self.cam, self.cam_error = None, None
        self.cam_stalled, self.stall_since = False, None
        self.cam_search_t, self.dark_since = -1e9, None
        self.camera_names, self.cam_missing = {}, None   # index -> name; name of an unplugged desk camera
        self.engine = None
        self.desk = DeskWatch()
        self.enabled = not paused
        self.mode, self.setup_step, self.setup_only_calibrate = "run", None, False
        self.placement = self.calib = self.learn = None
        self.coach = self.script = None
        self.record_result = None
        self.commands = queue.Queue()
        self.cond = threading.Condition()
        self.snapshot, self.snap_seq = {}, 0
        self.jpeg, self.jpeg_seq, self.viewers = None, 0, 0
        self.cameras, self.camera_thumbs = None, {}
        self.toast = None
        self.overlay = None
        self.on_quit = None
        self.running = False
        self.seen_event, self.last_hand_t = 0, 0.0
        self.recent_rows = collections.deque()          # landmarks of the last few seconds (misfires)
        self.click_times = collections.deque(maxlen=20)
        self.mm_px = None
        self._use_profile(self.profile)
        if not self.calibrated:
            self.mode, self.setup_step = "setup", "welcome"

    # ---- lifecycle ---------------------------------------------------------------------

    def start(self):
        self.running = True
        self._sync_overlay()
        self.thread = threading.Thread(target=self._loop, name="glide-pipeline", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=3)
        try:
            self._keep_size_map(0.0, force=True)
        except Exception:
            pass
        self.out.release_all()
        if hasattr(self.out, "stop"):
            self.out.stop()
        if self.cam:
            self.cam.stop()
        if self.overlay:
            self.overlay.stop()
        self.tracker.close()

    def post(self, action, **kw):
        """Thread-safe: queue a UI command for the pipeline thread."""
        self.commands.put((action, kw))

    # ---- profile / camera ----------------------------------------------------------------

    @property
    def calibrated(self):
        p = self.profile
        if not p:
            return False
        size = self.cam.size if self.cam else None
        return size is None or tuple(p.get("frame_wh") or ()) == tuple(size)

    def _use_profile(self, p):
        if p and "axis_gain" not in p and p.get("hand_box_px") and p.get("frame_wh"):
            # calibrated before the room-shape fit existed: derive it from what was measured
            p.update(fit_workspace(p["R"], p["frame_wh"], p["hand_box_px"], 1000.0 / p["px_per_m"],
                                   self.cfg["gain"], self.cfg["edge"], self.screen_wh))
            save_profile(p)
        self.profile = p
        self.engine = Engine(self.cfg, p, self.out) if p else None
        self.desk = DeskWatch()
        surf = (p or {}).get("surface")
        if self.engine and surf and tuple(surf.get("frame_wh") or ()) == tuple(p.get("frame_wh") or ()):
            self.engine.set_bounds(surf["rect"] if surf.get("blocked") else None)   # last known; re-checked live
            if surf.get("blocked") and p.get("hand_box_px"):
                # refit on start: the fit may have been made by an older rule (it once saw 1.1 cm of
                # forward room where 7.1 cm was free)
                fit = fit_workspace(p["R"], p["frame_wh"], p["hand_box_px"], 1000.0 / p["px_per_m"],
                                    self.cfg["gain"], self.cfg["edge"], self.screen_wh, bounds=surf["rect"])
                if [round(v, 1) for v in fit["travel_mm"]] != [round(v, 1) for v in p.get("travel_mm", [])]:
                    p.update(fit)
                    save_profile(p)
                    self.engine.set_profile(p)
                    self.engine.set_bounds(surf["rect"])
            self.desk.applied = tuple(surf["rect"])
            # The saved seed is not used to look again: the free desk is re-checked only once the
            # hand has been seen this session. A camera turned away from the desk between sessions
            # (2026-11-19: it looked at the room) was otherwise read as a cramped "desk" and refitted.

    def _resolve_camera(self):
        """Find the desk camera's current number from its remembered device id. Windows renumbers
        cameras whenever one appears or fails (2026-11-19: the laptop's own camera came back as 0
        and Glide quietly showed it instead of the desk camera). False: the desk camera is unplugged."""
        cams = list_cameras()
        self.camera_names = {c["index"]: c["name"] for c in cams}
        self.cam_missing = None
        if not cams:                                      # no Media Foundation list: numbers only
            return True
        c = self.cfg["camera"]
        if c.get("id"):
            i = index_of(c["id"], cams)
            if i is None:
                self.cam_missing = c.get("name") or "Your desk camera"
                return False
            if i != c["index"]:
                self._save_camera(i, cams)
            return True
        # chosen before cameras were remembered by id: keep that choice unless it is the laptop's own
        # camera and another one is plugged in, then remember the camera itself
        pick = c["index"] if c["index"] < len(cams) else 0
        if BUILT_IN.search(cams[pick]["name"]):
            pick = next((x["index"] for x in cams if not BUILT_IN.search(x["name"])), pick)
        self._save_camera(pick, cams)
        return True

    def _save_camera(self, index, cams=None):
        """Remember camera <index> by its device id and name (config.json), so a renumbering finds it."""
        cams = list_cameras() if cams is None else cams
        info = next((x for x in cams if x["index"] == index), None)
        over = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        cam = over.setdefault("camera", {})
        cam["index"] = int(index)
        if info:
            cam["id"], cam["name"] = info["id"], info["name"]
        else:
            cam.pop("id", None)
            cam.pop("name", None)
        CONFIG_PATH.write_text(json.dumps(over, indent=2), encoding="utf-8")
        self.cfg = load_config(CONFIG_PATH)

    def _open_camera(self):
        if self.cam:
            self.cam.stop()
            self.cam = None
        if not self._resolve_camera():
            self.cam_error = (f"{self.cam_missing} isn't connected. Plug it in, or choose another camera "
                              "in Settings.")
            self.seq = 0
            return
        try:
            self.cam = Camera(self.cfg["camera"])
            self.cam.start()
            self.cam_error = None
        except RuntimeError as e:
            self.cam, self.cam_error = None, str(e)
            if self._switch_camera(skip=(self.cfg["camera"]["index"],)):
                return self._open_camera()
        self.seq = 0

    def _switch_camera(self, skip=()):
        """The camera at our number won't open or shows only black: look for the one that works and
        use it. A new number for the same camera needs no recalibration (same picture size)."""
        now = time.perf_counter()
        if now - self.cam_search_t < CAM_SEARCH_EVERY_S:
            return False
        if self.cfg["camera"].get("id") and self.camera_names:
            # the remembered camera is plugged in (else _resolve_camera says so): it's the right one,
            # just dark or slow to start. Switching to whatever else shows a picture would pick the
            # laptop's own camera
            return False
        self.cam_search_t = now
        old = self.cfg["camera"]["index"]
        i = find_working(self.cfg["camera"], skip=skip, dark=DARK_LEVEL)
        if i is None or i == old:
            return False
        self._save_camera(i)
        self.toast = (f"Camera {old} stopped giving a picture, so Glide switched to camera {i}. "
                      "If that's the wrong one, choose it in Settings.")
        return True

    def _drop_camera(self):
        """Let go of a camera that stopped sending pictures without waiting on it (its read can hang),
        so the loop opens it again."""
        old, self.cam = self.cam, None
        if old is not None:
            threading.Thread(target=old.stop, name="glide-camera-drop", daemon=True).start()

    def _probe_cameras(self):
        from .setup import probe
        active = self.cfg["camera"]["index"]
        listed = list_cameras()
        names = {c["index"]: c["name"] for c in listed}
        found = []
        for i in (sorted(names) if names else range(5)):
            name = names.get(i, f"Camera {i}")
            if i == active and self.cam:
                found.append({"index": i, "name": name, "fps": round(self.cam.fps), "active": True})
                continue
            c = probe(self.cfg["camera"], i)
            if c:
                self.camera_thumbs[i] = cv2.imencode(".jpg", cv2.resize(c["thumb"], (320, 180)))[1].tobytes()
                found.append({"index": i, "name": name, "fps": round(c["fps"]), "active": False,
                              "dark": c["bright"] < 60, "built_in": bool(BUILT_IN.search(name))})
        self.cameras = found

    # ---- settings ------------------------------------------------------------------------

    def settings(self):
        return {"values": {p: get_path(self.cfg, p) if p != "preset" else (self.cfg.get("preset") or "")
                           for p in SETTINGS},
                "long_ms": (self.profile or {}).get("click", {}).get("long_ms", self.cfg["click"]["long_ms"]),
                "cameras": self.cameras, "version": __version__,
                "scripts": [{"id": k, "title": t, "what": w, "minutes": m} for k, (t, w, m) in SCRIPT_INFO.items()]}

    def _set(self, path, value):
        kind = SETTINGS.get(path)
        if kind is None:
            return
        if kind is bool:
            value = bool(value)
        elif kind is int:
            value = int(value)
        elif isinstance(kind, tuple) and isinstance(kind[0], (int, float)):
            value = float(np.clip(float(value), *kind))
        elif value not in kind:
            return
        over = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        if path == "preset":
            over.pop("preset", None)
            if value:
                over["preset"] = value
        else:
            node = over
            *parents, leaf = path.split(".")
            for k in parents:
                node = node.setdefault(k, {})
            node[leaf] = value
        CONFIG_PATH.write_text(json.dumps(over, indent=2), encoding="utf-8")
        self.cfg = load_config(CONFIG_PATH)
        if path == "click.long_ms" and self.profile and "click" in self.profile:
            self.profile["click"]["long_ms"] = value        # the personal value would otherwise win
            save_profile(self.profile)
        self.keys = KeyWatcher(self.cfg["guard"]["typing_ms"])
        self.out.release_all()
        self._use_profile(self.profile)
        self._sync_overlay()
        if path == "camera.index":
            self._save_camera(value)                     # remember the camera itself, not its number
            self._open_camera()
        if path in RECALIBRATE_AFTER:
            self.toast = "Recalibrate so Glide learns the new setup"

    def _reset_settings(self):
        keep = {}
        if CONFIG_PATH.exists():
            old = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if "camera" in old:
                keep["camera"] = old["camera"]                 # the camera choice is not a preference
        CONFIG_PATH.write_text(json.dumps(keep, indent=2), encoding="utf-8")
        self.cfg = load_config(CONFIG_PATH)
        self._use_profile(self.profile)
        self._sync_overlay()
        self.toast = "Settings reset"

    def _sync_overlay(self):
        want = self.cfg["feedback"]["overlay"]
        if want and self.overlay is None:
            self.overlay = OverlayProcess()
            self.overlay.start()
        elif not want and self.overlay is not None:
            self.overlay.stop()
            self.overlay = None

    # ---- commands ------------------------------------------------------------------------

    def _command(self, action, kw):
        if action == "enable":
            self.enabled = bool(kw.get("on", not self.enabled))
            if not self.enabled:
                self.out.release_all()
        elif action == "set":
            self._set(kw["path"], kw["value"])
        elif action == "sensor" and self.profile:               # learned from the user's own mouse
            self.profile["sensor"] = [float(v) for v in kw["value"]]
            save_profile(self.profile)
            if self.engine:
                self.engine.set_profile(self.profile)
        elif action == "reset_settings":
            self._reset_settings()
        elif action == "cameras":
            threading.Thread(target=self._probe_cameras, daemon=True).start()
        elif action == "setup":
            self.setup_only_calibrate = False           # a fresh start: full setup or just calibration
            self._enter_setup(kw.get("step", "welcome"), only_calibrate=kw.get("step") == "calibrate")
        elif action == "setup_next":
            self._setup_next()
        elif action == "setup_back":
            i = STEPS.index(self.setup_step)
            if i > 0:
                self._enter_setup(STEPS[i - 1])
        elif action == "setup_skip":
            if self.setup_step == "calibrate" and self.calib:
                self.calib.skip()
            elif self.setup_step == "learn" and self.learn:
                self.learn.skip()
        elif action == "setup_cancel" and self.setup_only_calibrate and self.profile:
            self._cancel_recalibration("Recalibration cancelled. Your previous calibration is kept.")
        elif action == "setup_finish":
            self.mode, self.setup_step, self.setup_only_calibrate = "run", None, False
            self.toast = "Glide is on"
        elif action == "record":
            self._start_record(kw.get("script", "scroll"))
        elif action == "coach" and self.coach is not None and self.record_result is None:
            if kw["key"] in ("go", "toggle_pause", "redo", "skip"):
                getattr(self.coach, kw["key"])()
            elif kw["key"] == "stop":
                del self.coach.rows[self.coach.row0:]
                del self.coach.cues[self.coach.cue0:]
                self._finish_record()
        elif action == "pointing_save":
            # positions and times only; scored again here so the saved numbers come from tested code
            rec = pointing.save(kw.get("device"), kw.get("rounds") or [], kw.get("board"))
            if rec:
                self.toast = f"Pointing test saved: {rec['tp']:.2f} bits/s"
        elif action == "record_close":
            self.coach, self.record_result, self.mode = None, None, "run"
        elif action == "quit" and self.on_quit:
            self.on_quit()

    def _cancel_recalibration(self, why):
        self.mode, self.setup_step, self.setup_only_calibrate, self.calib = "run", None, False, None
        self.toast = why

    def _enter_setup(self, step, only_calibrate=False):
        self.out.release_all()
        self.mode, self.setup_step = "setup", step
        if only_calibrate:
            self.setup_only_calibrate = True
        if step == "camera" and self.cameras is None:
            threading.Thread(target=self._probe_cameras, daemon=True).start()
        if step == "placement":
            self.placement = Placement(self.cfg, self.screen_wh)
        if step == "calibrate":
            self.calib = Calibrate(self.cfg, self.screen_wh)
            self.calib_seen = None
        if step == "learn":
            if not self.profile:
                return self._enter_setup("calibrate")
            self.learn = Learn(self.cfg, self.profile, self.screen_wh)
        if step == "practice":
            self.enabled = True                       # practice uses the real cursor

    def _setup_next(self):
        if self.setup_step == "calibrate" and self.setup_only_calibrate:
            self.setup_only_calibrate = False
            self.mode, self.setup_step = "run", None
            self.toast = "Calibrated"
            return
        i = STEPS.index(self.setup_step)
        if i + 1 < len(STEPS):
            self._enter_setup(STEPS[i + 1])

    def _start_record(self, script):
        if script not in SCRIPTS:
            return
        self.out.release_all()
        self.mode, self.script, self.record_result = "record", script, None
        self.mouse_watch = None
        R = (self.profile or {}).get("R")
        self.coach = Coach(SCRIPTS[script], R, self.cfg["hand"]["dominant"])
        self.step_t0, self.step_i, self.coach_toast = 0.0, -1, None

    def _finish_record(self):
        path = save_session(self.coach, self.script, self.cfg["hand"]["dominant"])
        self.record_result = {"path": str(path) if path else None, "summary": self.coach.summary,
                              "metrics": None, "analyzing": bool(path)}
        if path:
            threading.Thread(target=self._analyze, args=(path,), daemon=True).start()

    def _analyze(self, path):
        from .analyze import metrics
        from .replay import load_session, session_profile
        if self.script == "mouse":
            return self._learn_mouse(path)
        try:
            frames, cues = load_session(path)
            cfg = load_config()
            m = metrics(frames, cues, cfg, session_profile(frames, cfg))
            keep = ("click_hits", "ghosts", "jitter_px", "two_scroll_units", "presses_in_fist_phase")
            self.record_result["metrics"] = {k: (v if not isinstance(v, float) else round(v, 2))
                                             for k, v in m.items() if k in keep and v is not None}
        except Exception as e:                             # analysis is a bonus; never break the app
            self.record_result["metrics"] = {"error": str(e)}
        self.record_result["analyzing"] = False

    def _learn_mouse(self, path):
        from .mousefit import fit, load_rows, sensor_from
        res = None
        try:
            p = self.profile
            res = fit(load_rows(path), p["R"], 1000.0 / p["px_per_m"]) if p else None
        except Exception as e:
            self.record_result["metrics"] = {"error": str(e)}
        if res:
            self.record_result["mouse"] = dict(res, sensor=sensor_from(res))
            self.post("sensor", value=sensor_from(res))   # applied on the pipeline thread
        elif "metrics" not in self.record_result or self.record_result["metrics"] is None:
            self.record_result["mouse"] = {"error": "too little"}
        self.record_result["analyzing"] = False

    # ---- the pipeline --------------------------------------------------------------------

    def _loop(self):
        self._open_camera()
        if self.cam and self.profile and not self.calibrated and self.mode == "run":
            self._enter_setup("calibrate", only_calibrate=True)   # capture mode changed since then
        retry_at = 0.0
        while self.running:
            while not self.commands.empty():
                self._command(*self.commands.get())
            if self.cam is None:
                if time.perf_counter() > retry_at:
                    self._open_camera()
                    retry_at = time.perf_counter() + 2.0
                self._publish(None, None, time.perf_counter(), {})
                time.sleep(0.1)
                continue
            got = self.cam.get(self.seq, timeout=1.0)
            if got is None:                               # camera stalled: fail safe
                self.out.release_all()
                now = time.perf_counter()
                self.stall_since = self.stall_since or now
                # say so (the screen kept showing the last picture before), then reconnect
                self.cam_error, self.cam_stalled = "stalled", True
                if now - self.stall_since > STALL_REOPEN_S:
                    self._drop_camera()
                    self.stall_since, retry_at = None, 0.0
                self._publish(None, None, now, {})
                continue
            if self.cam_stalled:
                self.cam_error, self.cam_stalled = None, False
            self.stall_since = None
            if self.cam.brightness < DARK_LEVEL:
                # only black for a while: maybe our number now belongs to another camera (a laptop's
                # own behind its lid). Look for one that shows a picture; at night none does, and the
                # app says to turn a light on instead
                self.dark_since = self.dark_since or time.perf_counter()
                if time.perf_counter() - self.dark_since > DARK_SWITCH_S:
                    self.dark_since = None
                    if self._switch_camera(skip=(self.cfg["camera"]["index"],)):
                        self._drop_camera()               # the loop opens the one that works
                        continue
            else:
                self.dark_since = None
            self.seq, frame, t = got
            idle = t - self.last_hand_t > self.cfg["idle"]["after_s"] and self.mode == "run"
            if idle and self.seq % self.cfg["idle"]["every_n"]:
                obs = Obs(present=False)
            else:
                obs = self.tracker.detect(frame, t)
            if obs.present:
                self.last_hand_t = t
            typing, hot = self.keys.poll()
            if "quit" in hot and self.on_quit:
                self.on_quit()
            if "toggle" in hot:
                self.enabled = not self.enabled
            if self.mode == "run":
                # with the cursor, so a report also shows a jump, not only a click
                self.recent_rows.append(frame_row(frame, self.tracker, t, cursor=self.out.cursor()))
                while self.recent_rows and t - self.recent_rows[0]["t"] > MISFIRE_KEEP_S:
                    self.recent_rows.popleft()
            if "misfire" in hot:
                self._save_misfire(t)
            extra = getattr(self, "_" + self.mode)(obs, frame, t, typing)
            self._publish(obs, frame, t, extra)

    def _sound(self, name):
        if self.cfg["feedback"]["sound"]:
            play(name)

    def _run(self, obs, frame, t, typing):
        st = {}
        if self.engine is not None:
            self.engine.update(obs, t, typing, paused=not self.enabled)
            st = self.engine.status()
            self._exit_tip(t)
            self._keep_size_map(t)
            self._watch_desk(obs, frame, t)
            if self.engine.event_seq != self.seen_event:
                self.seen_event = self.engine.event_seq
                if self.engine.last_event in ("left click", "right click", "drag", "dwell click"):
                    self.click_times.append((t, self.engine.last_event))
                self._sound("tick")
            st = {"state": "PAUSED" if not self.enabled else st["state"], "hold": st["hold"],
                  "edge": st["edge"], "event": st["event"], "event_seq": self.engine.event_seq,
                  "learned_clicks": self.engine.autogain.n}
        if self.overlay:
            e = self.engine.status() if self.engine else None
            self.overlay.set("PAUSED" if not self.enabled else (e["state"] if e else "IDLE"),
                             e["hold"] if e else 0.0, e["edge_dir"] if e and e["edge"] > 0 else None)
        return {"engine": st}

    def _watch_desk(self, obs, frame, t):
        """Every few seconds, find the free desk (surface.detect) on a worker thread. A new free
        area is used once two checks in a row agree, so a hand or a passing arm never changes it."""
        d, e, p = self.desk, self.engine, self.profile
        if not (e and p and p.get("hand_box_px")) or frame is None or not self.calibrated:
            return
        if obs.present and e.state == TRACK and not e.lifted:
            d.palm, d.hull = np.array(obs.palm_px, float), np.array(obs.img, float)
        elif obs.present:
            return                                        # pinching, scrolling, lifted: not a clean look
        else:
            d.hull = None                                 # hand away: the whole desk is visible
        if d.result is not None:
            (res, seen), d.result = d.result, None
            self._desk_result(res, hand_in_view=seen)
        if d.busy or d.palm is None or t - d.last < DeskWatch.EVERY_S:
            return
        d.last, d.busy = t, True
        args = (frame.copy(), d.palm.copy(), p["hand_box_px"], None if d.hull is None else d.hull.copy())

        def work():
            try:
                d.result = (detect(*args) or False, args[3] is not None)
            except Exception:
                d.result = (False, False)
            d.busy = False
        threading.Thread(target=work, name="glide-desk", daemon=True).start()

    def _desk_result(self, res, hand_in_view=False):
        d, p = self.desk, self.profile
        if not res:
            return
        rect, (hw, hh) = res["rect"], p["hand_box_px"]
        if rect[2] - rect[0] < hw * 1.15 or rect[3] - rect[1] < hh * 1.15:
            return                                        # smaller than the hand: a misread, not a desk
        W, H = p["frame_wh"]
        if hand_in_view and d.applied is not None and not contains(rect, d.applied, 0.04 * W):
            # the hand, its shadow and the arm cover desk: seen with them, the desk may only grow.
            # (Shrinking it mid-use squeezed one session to 2.9 x 2.4 cm.) Shrinks need a clear look.
            return
        d.agree = d.agree + 1 if same_room(rect, d.cand) else 1
        d.cand = rect
        if d.agree < 2 or same_room(rect, d.applied, 0.04):
            return
        first, before = d.applied is None, d.applied
        d.applied = rect
        self._keep_size_map(0.0, force=True)              # set_profile reloads the saved size map
        full = rect[0] <= 0.06 * W and rect[1] <= 0.06 * H and rect[2] >= 0.94 * W and rect[3] >= 0.94 * H
        bounds = None if full else rect
        p.update(fit_workspace(p["R"], p["frame_wh"], p["hand_box_px"], 1000.0 / p["px_per_m"],
                               self.cfg["gain"], self.cfg["edge"], self.screen_wh, bounds=bounds))
        R = np.array(p["R"], float)
        blocked = [[desk_side(R, side), colour] for side, colour in res["blocked"].items()]
        p["surface"] = {"rect": list(bounds) if bounds else [0, 0, W, H], "blocked": blocked,
                        "frame_wh": list(p["frame_wh"]), "seed": [round(float(v), 1) for v in d.palm],
                        **room_report(p["R"], rect, p["frame_wh"], p["hand_box_px"],
                                      1000.0 / p["px_per_m"], blocked)}
        save_profile(p)
        self.engine.set_profile(p)
        self.engine.set_bounds(bounds)
        area = lambda r: max(r[2] - r[0], 1) * max(r[3] - r[1], 1)
        if not first and abs(area(rect) / area(before) - 1) > 0.25:      # say so only when it matters
            cm = p["surface"]["slide_cm"]
            self.toast = (f"Your free desk changed: the hand now has {cm[0]:g} cm to slide left-right "
                          f"and {cm[1]:g} cm forward-back." if blocked else
                          "The desk is clear now: your hand has the whole view again.")

    def _save_misfire(self, t):
        """Ctrl+Alt+X, "that click was a mistake": keep the landmarks (never images, NFR-06) around the
        last click, so phantom clicks can be studied on the user's own hand instead of guessed at."""
        if not self.recent_rows:
            return
        clicks = [(ct, kind) for ct, kind in self.click_times if t - ct < MISFIRE_KEEP_S]
        (ROOT / "logs").mkdir(exist_ok=True)
        path = ROOT / "logs" / time.strftime("misfire_%Y%m%d_%H%M%S.jsonl")
        meta = {"meta": {"script": "misfire", "reported_t": t, "clicks": clicks,
                         "hand": self.cfg["hand"]["dominant"]}}
        rows = [dict(r, label="misfire") for r in self.recent_rows]
        path.write_text("\n".join(json.dumps(r) for r in [meta] + rows), encoding="utf-8")
        self.toast = ("Thanks. Glide kept the moments before that click so it can learn from it."
                      if clicks else "Thanks. Glide sent no click in the last 15 seconds, but it kept them anyway.")
        self._sound("ok")

    def _keep_size_map(self, t, force=False):
        """Save the learned palm-size-across-the-view gradient (engine._size_fit) once a minute, so
        the next start knows the desk's perspective before the hand has moved around."""
        if not (self.engine and self.profile) or (not force and t - getattr(self, "_grad_saved", t - 60) < 60):
            return
        self._grad_saved = t
        g = [round(float(v), 6) for v in self.engine.lift_grad]
        old = self.profile.get("lift_grad", [0.0, 0.0])
        learned = self.engine.autogain.saved()           # the speed learned from the user's clicks
        changed = max(abs(a - b) for a, b in zip(g, old)) > 5e-5 or learned != self.profile.get("autogain")
        if changed:
            self.profile["lift_grad"] = g
            self.profile["autogain"] = learned
            save_profile(self.profile)

    def _exit_tip(self, t):
        """Leaving the view (without gliding) three times in a minute: say once how to reach far."""
        n = self.engine.exits
        if n != getattr(self, "_exits_seen", 0):
            self._exits_seen = n
            self._exit_times = [x for x in getattr(self, "_exit_times", []) if t - x < 60] + [t]
            if len(self._exit_times) >= 3 and not getattr(self, "_exit_tip_shown", False):
                self._exit_tip_shown = True
                self.toast = ("Out of room? Lift your hand a little and carry it back, "
                              "or push into the edge of the view to glide.")

    def _setup(self, obs, frame, t, typing):
        s = {"step": self.setup_step, "only_calibrate": self.setup_only_calibrate}
        if self.overlay:
            self.overlay.set("IDLE")
        if self.setup_step == "placement" and self.placement:
            self.placement.feed(obs, frame, self.cam.fps)
            s["placement"] = self.placement.state()
        elif self.setup_step == "calibrate" and self.calib:
            if obs.present or self.calib_seen is None:
                self.calib_seen = t
            if self.setup_only_calibrate and self.profile and t - self.calib_seen > RECAL_GIVE_UP_S:
                # never strand a user (with no mouse) in a calibration that can't see their hand
                self._cancel_recalibration("Glide couldn't see your hand for a minute, so it kept your "
                                           "previous calibration.")
                return {"setup": s}
            self.calib.feed(obs, t)
            for e in self.calib.events:
                self._sound(e)
            if self.calib.profile is not None and self.profile is not self.calib.profile:
                save_profile(self.calib.profile)
                self._use_profile(self.calib.profile)
            s["calibrate"] = self.calib.state()
        elif self.setup_step == "learn" and self.learn:
            self.learn.feed(obs, t)
            for e in self.learn.events:
                self._sound(e)
            s["learn"] = self.learn.state()
        elif self.setup_step == "practice":
            s.update(self._run(obs, frame, t, typing))
        elif self.setup_step == "camera":
            s["cameras"] = self.cameras
        return {"setup": s}

    def _record(self, obs, frame, t, typing):
        c = self.coach
        if self.overlay:
            self.overlay.set("IDLE")
        if self.record_result is None:
            i = c.i
            c.feed(obs, t, frame_row(frame, self.tracker, t, self.out.cursor()))
            for e in c.events:
                if e != "start":
                    self._sound(e)
            if c.i != i:
                self.step_t0 = t
                if c.summary:
                    rec, title = c.summary[-1], c.steps[i].title
                    self.coach_toast = (f"Done: {title}" + (f" - {rec['seen']}/{rec['trials']} seen"
                                                            if "seen" in rec else ""), t)
            if c.done:
                self._finish_record()
            elif self.script == "mouse" and self._no_mouse(obs, t):
                # the hand moves but the pointer never does: nothing is plugged in to learn from
                self.record_result = {"path": None, "summary": c.summary, "metrics": None, "analyzing": False,
                                      "mouse": {"error": "no_mouse"}}
            return {"record": self._coach_state(t)}
        # finished: the result screen is up, so the hand drives the pointer again (else, with no
        # mouse plugged in, nothing could reach the Done button)
        extra = self._run(obs, frame, t, typing)
        extra["record"] = self._coach_state(t)
        return extra

    def _no_mouse(self, obs, t):
        """Mouse session: 8 s with the hand in view and the pointer still within 20 px."""
        cur = np.array(self.out.cursor(), float)
        if self.mouse_watch == "seen":
            return False
        if self.mouse_watch is None or not obs.present:
            if obs.present and self.mouse_watch is None:
                self.mouse_watch = (t, cur)
            return False
        t0, c0 = self.mouse_watch
        if float(np.hypot(*(cur - c0))) > 20:
            self.mouse_watch = "seen"                     # it moved: a mouse is there
            return False
        return t - t0 > 8.0

    def _coach_state(self, t):
        c, s = self.coach, self.coach.step
        r = {"script": self.script, "title_session": SCRIPT_INFO[self.script][0], "n": len(c.steps),
             "i": c.i, "done": c.done, "paused": c.paused, "result": self.record_result}
        if self.coach_toast and t - self.coach_toast[1] < 2.0:
            r["toast"] = self.coach_toast[0]
        if s is None:
            return r
        check = CHECKS.get(s.check)
        live = c.live_ok() if c.state == "run" else None
        r.update(label=s.label, title=s.title, how=s.how, kind=s.kind, tip=s.tip, demo=s.demo or s.label,
                 state=c.state, countdown=c.countdown(), fill=c.fill(), intro_left=c.intro_left(),
                 reps=s.reps, practice=c.practice_left > 0,
                 trials=[hit for practice, hit in c.results if not practice],
                 last=(list(c.results[-1]) if c.results else None),
                 hit=bool(getattr(c, "hit", False)) if c.state == "respond" else False,
                 go_age=c.since_go() if c.state in ("respond", "result") and s.kind == "cued" else None,
                 cued=s.kind == "cued", gesture=gesture_of(s.label),
                 live=live, live_text=(check[1] if live else check[2]) if check and live is not None else None,
                 left=max(0.0, s.seconds - c.good_s), demo_t=t - self.step_t0)
        return r

    # ---- publishing ------------------------------------------------------------------------

    def _view_R(self):
        if self.mode == "record" and self.coach is not None and self.coach.R is not None:
            return self.coach.R
        if self.mode == "setup":
            if self.calib is not None and self.calib.cal.profile.get("R") is not None:
                return self.calib.cal.profile["R"]
            if self.placement is not None and self.placement.R is not None:
                return self.placement.R
        return (self.profile or {}).get("R")

    def _publish(self, obs, frame, t, extra):
        snap = {"mode": self.mode, "enabled": self.enabled, "calibrated": self.calibrated,
                "has_profile": self.profile is not None, "camera_error": self.cam_error,
                "camera_stalled": self.cam_stalled,
                "camera_index": self.cfg["camera"]["index"], "toast": self.toast,
                "camera_name": self.cfg["camera"].get("name"), "camera_missing": self.cam_missing,
                "hand_side": self.cfg["hand"]["dominant"], "t": round(time.time(), 3)}
        self.toast = None
        if frame is not None:
            view = View(self._view_R(), self.cam.size)
            vw, vh = view.size
            snap["dark"] = self.cam.brightness < DARK_LEVEL
            snap.update(fps=round(self.cam.fps, 1), repeats=round(self.cam.repeats, 2), latency_ms=round((time.perf_counter() - t) * 1000),
                        view=[vw, vh])
            if obs.present:
                self.mm_px = 1000.0 / obs.px_per_m if self.mm_px is None else \
                    self.mm_px + 0.05 * (1000.0 / obs.px_per_m - self.mm_px)
                pts = view.pts(obs.img) / [vw, vh]
                snap["hand"] = {"pts": np.round(pts, 4).tolist(), "score": round(float(obs.score), 2)}
            elif self.profile:
                self.mm_px = self.mm_px or 1000.0 / self.profile["px_per_m"]
            snap["mm_per_px"] = round(self.mm_px, 4) if self.mm_px else None
            e = self.cfg["edge"]
            bounds = self.engine.bounds if self.engine else None
            snap["edge_inset"] = edge_band_px(e, self.cam.size, bounds) / min(vw, vh)
            if bounds is not None:
                norm = lambda c: np.round(np.concatenate([c.min(axis=0) / [vw, vh], c.max(axis=0) / [vw, vh]]), 4).tolist()
                snap["work"] = norm(view.pts([(bounds[0], bounds[1]), (bounds[2], bounds[3])]))
                hb = self.profile.get("hand_box_px") if self.profile else None
                if hb:                                    # the engine's bands (half the slack when cramped)
                    z = edge_band_px(e, self.cam.size, bounds)
                    zx = min(z, max((bounds[2] - bounds[0] - hb[0]) / 2, 6.0))
                    zy = min(z, max((bounds[3] - bounds[1] - hb[1]) / 2, 6.0))
                    snap["glide"] = norm(view.pts([(bounds[0] + zx, bounds[1] + zy), (bounds[2] - zx, bounds[3] - zy)]))
            surf = (self.profile or {}).get("surface")
            snap["desk"] = ({k: surf.get(k) for k in ("blocked", "free_cm", "slide_cm", "advice")}
                            if surf and surf.get("blocked") and bounds is not None else None)
            if self.profile and self.profile.get("travel_mm"):
                snap["room_cm"] = [round(v / 10) for v in self.profile["travel_mm"]]
                snap["raise_cm"] = self.profile.get("raise_cm")
            if self.viewers > 0:
                ok, buf = cv2.imencode(".jpg", view.frame(frame), [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    with self.cond:
                        self.jpeg, self.jpeg_seq = buf.tobytes(), self.jpeg_seq + 1
        snap.update(extra)
        with self.cond:
            self.snapshot, self.snap_seq = snap, self.snap_seq + 1
            self.cond.notify_all()
