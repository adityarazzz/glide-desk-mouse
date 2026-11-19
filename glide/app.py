"""Glide main loop: camera -> landmarks -> features -> engine -> Windows input."""
import argparse
import ctypes
import sys
import time

from . import output_win  # noqa: F401  (sets DPI awareness before any window exists)
from .calibrate import Calibrator, describe
from .config import PRESETS, ROOT, load_config, load_profile, save_profile
from .engine import Engine
from .features import Obs
from .output_win import KeyWatcher, WinOutput

WINDOW = "Glide"


def single_instance():
    """Two copies would fight over the cursor."""
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\GlideDeskMouse")
    return handle if ctypes.windll.kernel32.GetLastError() != 183 else None   # ERROR_ALREADY_EXISTS


def list_cameras():
    import cv2
    import numpy as np
    from .camera import open_capture
    tiles = []
    for i in range(6):
        cap = open_capture(i, {"fourcc": "MJPG", "width": 640, "height": 480, "fps": 30})
        ok, f = cap.read() if cap else (False, None)
        if cap:
            cap.release()
        if ok:
            print(f"camera {i}: {f.shape[1]}x{f.shape[0]}")
            f = cv2.resize(f, (320, 240))
            cv2.putText(f, f"--camera {i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(f)
    if not tiles:
        print("No cameras could be opened (is another app using them?)")
        return
    cv2.imshow("Glide cameras (any key closes)", np.hstack(tiles))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Glide bare-hand desk mouse (P1 prototype)")
    ap.add_argument("--setup", action="store_true", help="run the first-time setup wizard again")
    ap.add_argument("--calibrate", action="store_true", help="redo just the calibration")
    ap.add_argument("--record", nargs="?", const="basic", choices=["basic", "angles", "precision", "scroll", "reposition"],
                    help="record a coached session for glide.analyze (basic | angles | precision | scroll)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="precision (default), accessibility, fast")
    ap.add_argument("--camera", type=int, help="camera index (see --list-cameras)")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--no-preview", action="store_true", help="run without the debug window")
    ap.add_argument("--classic", action="store_true", help="the developer HUD instead of the app window")
    ap.add_argument("--paused", action="store_true", help="open the app with the hand mouse paused")
    args = ap.parse_args(argv)
    if args.list_cameras:
        return list_cameras()
    legacy = (args.classic, args.setup, args.calibrate, args.record, args.preset, args.camera is not None,
              args.no_preview)
    if not any(legacy):
        from .desktop import run
        return run(single_instance, args.paused)
    mutex = single_instance()
    if mutex is None:
        sys.exit("Glide is already running.")

    import cv2
    from .camera import Camera
    from .feedback import Overlay, tick
    from .hud import Hud
    from .tracker import HandTracker

    cfg = load_config(preset=args.preset)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    profile = None if args.calibrate else load_profile()
    if args.setup or (profile is None and not args.calibrate and not args.record):
        from .setup import run_setup
        profile = run_setup(cfg) or load_profile()
        if profile is None:
            sys.exit("Setup was not finished. Run glide.bat again to continue.")
        cfg = load_config(preset=args.preset)              # picks up the chosen camera
    preview = cfg["preview"] and not args.no_preview
    if profile is None and not preview:
        sys.exit("Calibration needs the preview window; run without --no-preview.")

    cam = Camera(cfg["camera"])
    tracker = HandTracker(ROOT / cfg["model"], cfg["hand"])
    if args.record:
        from .record import record
        cam.start()
        try:
            path = record(cam, tracker, args.record, cfg["hand"]["dominant"])
            print(f"Saved {path}\nAnalyze with: .venv\\Scripts\\python -m glide.analyze {path}" if path else "Cancelled.")
        finally:
            cam.stop()
            tracker.close()
        return

    out, keys, hud = WinOutput(), KeyWatcher(cfg["guard"]["typing_ms"]), Hud()
    screen_wh = out.screen_size()
    if profile and tuple(profile.get("frame_wh") or ()) != tuple(cam.size):
        print(f"Capture mode is now {cam.size[0]}x{cam.size[1]}: recalibrating (hand still, then 5 pinches).")
        profile = None
    engine = Engine(cfg, profile, out) if profile else None
    cal = None if profile else Calibrator(cfg, screen_wh)
    overlay = Overlay(out.cursor) if cfg["feedback"]["overlay"] else None
    if overlay:
        overlay.start()
    paused, seen_event, last_hand_t, skipped = False, 0, 0.0, 0
    cam.start()
    print(f"Glide running on camera {cfg['camera']['index']} {cam.size}"
          f"{' preset ' + cfg['preset'] if cfg.get('preset') else ''}. "
          f"Ctrl+Alt+G pause/resume, Ctrl+Alt+Q quit.")
    if profile:
        print("Profile:", describe(profile))

    seq = 0
    try:
        while True:
            got = cam.get(seq, timeout=1.0)
            if got is None:                         # camera stalled: fail safe
                out.release_all()
                continue
            seq, frame, t_cap = got
            # no hand for a while: run the (expensive) palm detector on every n-th frame only
            idle = t_cap - last_hand_t > cfg["idle"]["after_s"]
            if idle and seq % cfg["idle"]["every_n"]:
                obs = Obs(present=False)
                skipped += 1
            else:
                obs = tracker.detect(frame, t_cap)
            if obs.present:
                last_hand_t = t_cap
            typing, events = keys.poll()
            if "quit" in events:
                break
            if "toggle" in events:
                paused = not paused
                print("paused" if paused else "resumed")

            if cal is not None:
                done = cal.feed(obs, t_cap)
                if done:
                    save_profile(done)
                    print("Calibrated:", describe(done))
                    engine = engine or Engine(cfg, done, out)
                    engine.set_profile(done)
                    cal = None
            else:
                engine.update(obs, t_cap, typing, paused)
                st = engine.status()
                if overlay:
                    overlay.set("PAUSED" if paused else st["state"], st["hold"],
                                st["edge_dir"] if st["edge"] > 0 else None)
                if engine.event_seq != seen_event:
                    seen_event = engine.event_seq
                    if cfg["feedback"]["sound"]:
                        tick()
            lat_ms = (time.perf_counter() - t_cap) * 1000

            if preview:
                img = hud.draw(frame, obs, engine if cal is None else None, cal,
                               cam.fps, lat_ms, paused, typing, t_cap)
                cv2.imshow(WINDOW, img)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if k == ord("p"):
                    paused = not paused
                elif k == ord("c"):
                    if engine:
                        engine.update(Obs(present=False), t_cap, paused=True)
                    cal = Calibrator(cfg, screen_wh)
                elif k == ord(" ") and cal:
                    cal.skip()
                elif k in (ord("+"), ord("=")) and engine:
                    cfg["gain"]["speed"] *= 1.15
                elif k == ord("-") and engine:
                    cfg["gain"]["speed"] /= 1.15
                elif k == ord("[") and engine:
                    engine.rotate(-5)
                elif k == ord("]") and engine:
                    engine.rotate(5)
    except KeyboardInterrupt:
        pass
    finally:
        out.release_all()                           # never leave a button held down
        if overlay:
            overlay.stop()
        cam.stop()
        tracker.close()
        cv2.destroyAllWindows()
