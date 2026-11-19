"""Defaults, presets, plus optional overrides from config.json and the calibrated profile.json.

Defaults are tuned on recorded sessions (logs/, `python -m glide.analyze`), not guessed.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
PROFILE_PATH = ROOT / "profile.json"

DEFAULTS = {
    "camera": {
        "index": 1,            # a number Windows may change: the camera is found by "id" when set
        "id": None,            # USB vid/pid of the chosen camera (glide.camlist), remembered on choosing
        "name": None,          # its name, for messages ("ZEB LIVE PRO isn't connected")
        # 16:9 = the full sensor (640x480 crops it to 76% of the width, measured). Captured at
        # 1280x720 and area-averaged to process_width: 3x less frame-to-frame landmark noise than
        # the camera's own 640x360 mode, at the same inference cost (measured)
        "width": 1280,
        "height": 720,
        "process_width": 640,
        "fps": 30,
        "fourcc": "MJPG",
        "backend": "msmf",     # "msmf" (30 fps on the ZEB) or "dshow"
        "exposure": None,      # None = auto; otherwise log2 seconds, e.g. -5 = ~1/32 s
        # in a dim room the ZEB's auto exposure runs ~60 ms and it repeats each picture to fill 30 fps
        # (44% repeats measured, all modes). A repeat reads as "the hand stopped" and the next frame as
        # a double step, so repeats are dropped and the pipeline runs on the real, unique frames
        "drop_repeats": True,
    },
    "model": "models/hand_landmarker.task",
    "hand": {
        "dominant": "right",
        "min_detection_conf": 0.6,
        "min_tracking_conf": 0.5,
        "lost_ms": 300,        # fail-safe: release buttons after this long without a hand
        "glitch_mm": 60.0,     # one-frame jump larger than this is a tracking glitch, not motion
        # a frame the tracker itself doubts doesn't move the cursor (NFR-07: no action on an uncertain
        # state). A hand tilting away scored 0.58-0.67 and flung the cursor 1100-1160 px in one frame;
        # under 1% of tracked frames score below this (p1 = 0.74 over 22 000 recorded frames)
        "min_track_score": 0.70,
        "doubt_mm": 12.0,      # ...when it would also jump this far (about 180 mm/s at 15 fps)
        # a hand arriving in view is still sliding in (130-230 mm/s recorded): hold the cursor,
        # then fade motion in, and keep edge glide off, or putting the hand down flings the cursor
        "entry_hold_ms": 150,
        "entry_ramp_ms": 300,
        "entry_edge_ms": 600,
        "arrive_max_ms": 1000,     # longest hold for a returning hand to come fully into view and down
        "leave_rollback_ms": 150,  # taking the hand away: undo this much of its last motion,
        "leave_stroke_ms": 600,    # or back to where the pull-away began, if that was this recent
    },
    "pointer": {
        # landmarks averaged into the cursor anchor: wrist, thumb base, pinky knuckle moved least
        # while pinching/tapping in recorded sessions (finger knuckles shift 2-3x more)
        "anchor": [0, 1, 17],
        # ...but while just moving, follow where a mouse would sit: on the knuckle line. The palm base
        # is right at the wrist's pivot, so swivelling the wrist (how a mouse is mostly moved) moved it
        # 20-25 mm per radian against 53-70 for the knuckle line (recorded): you had to slide the whole
        # palm. Pinches, scrolls and fists switch back to the steady anchor within sensor_blend_ms,
        # offset so the cursor never jumps. The "Show Glide your mouse" session personalises k/side.
        "sensor": [1.0, 0.0],
        "sensor_blend_ms": 120.0,
        # the lever's direction comes from wrist -> pinky knuckle (plus a learned angle): it moves least
        # in pinches and taps. Clicks then landed as close as with the palm base (p90 8.8 vs 10.5 px)
        # where the knuckle-line direction let the worst land 91 px off (37 recorded clicks)
        "lever_dir": [17],
        "predict_ms": 35.0,    # velocity extrapolation, ramped in with speed (TRD 4.7); tuned
        "predict_v0": 40.0,    # mm/s: no prediction below this (keeps the still cursor still)
        "predict_v1": 200.0,   # mm/s: full prediction above this
        # "mouse": the axes turn with the hand, as a mouse's sensor does (it is fixed to the mouse),
        # so a sweep pivoting at the elbow or wrist still moves the cursor straight; see engine._axes.
        # "adaptive": "forward" follows where the fingers point, re-aligning slowly while the hand
        # is nearly still; never rotates mid-stroke. "fixed": desk axes locked at calibration.
        "axes": "mouse",
        "mouse_hz": 6.0,              # smoothing of the hand's heading (landmark noise ~1 deg)
        "mouse_max_deg": 75.0,        # beyond this the hand is tilted or on its side, not turned
        "axes_tau_s": 1.5,
        "axes_deadband_deg": 4.0,     # gestures wobble the heading +-6 deg; ignore small changes
        "axes_max_speed": 100.0,      # mm/s: only re-align when moving slower than this
        # spread each frame's cursor step over the frame interval (WinOutput thread at ~250 Hz):
        # at 17 fps the cursor otherwise jumps up to 100+ px at a time
        "smooth_output": True,
    },
    # 1-euro filter on the anchor in desk millimetres (beta is unit-dependent, so tuned for mm)
    # tuned on a recorded session: 0.42 px static jitter (NFR-02 <= 1 px) with ~0 ms filter lag
    # retuned on the 720p->640 pipeline: jitter 1.02 -> 0.95 px (precision), 0.60 -> 0.65 (golden),
    # lag <= 5 ms; lower beta was smoother still but made fast flicks lag
    # fc_max: at speed the 1-euro cutoff would open to 30+ Hz and pass the hand's 1.8-2.8 mm frame-to-
    # frame wobble straight into a 40 px/mm gain; capping it at 8 Hz cut cursor shake while moving by
    # a quarter for +5 ms lag (recorded sessions)
    "filter": {"fc_min_hz": 0.4, "beta": 0.16, "d_cutoff_hz": 1.0, "speed_cutoff_hz": 6.0, "fc_max_hz": 8.0},
    # speed-dependent gain in screen px per desk mm, defined for a 1920 px wide screen
    # The gain follows its own speed estimate: it rises at 8 Hz so speed noise can't make the gain
    # flicker mid-stroke, and falls at 16 Hz so it is already low as the hand slows onto a target.
    # With fc_max and the zero-anchored curve (filters.gain_curve), on the recorded sessions: shake
    # while moving -32%, static jitter 0.65 -> 0.50 px, reach 0.59 -> 0.54 of the screen, lag +5 ms
    "gain": {"g_min": 2.0, "g_max": 14.0, "v_mid_mm_s": 135.0, "width": 50.0, "speed": 1.0,
             "reference_width": 1920, "rise_hz": 8.0, "fall_hz": 16.0},
    # learn the user's speed from where their clicks land (glide.autogain, after AutoGain, CHI 2020)
    "autogain": {"enabled": True, "v_on": 40.0, "v_still": 15.0, "end_share": 0.25, "restart_share": 0.6,
                 "max_correct_s": 4.0, "min_primary_px": 40, "min_move_px": 60, "cover": 0.93,
                 "rate": 0.35, "max_step": 0.10, "min_mult": 0.6, "max_mult": 1.6},
    "orient_offset_deg": 0.0,
    # click = a quick dip of the thumb-to-finger image gap relative to its running baseline;
    # works for thumb+middle pinch (strongest), thumb+index pinch and index taps alike
    "click": {
        "enter": 0.70,         # dip ratio that starts a press
        "deep": 0.55,          # dip this deep starts a press without waiting for confirmation
        "exit": 0.86,          # dip ratio that ends it
        "confirm_frames": 2,
        "max_speed_mm_s": 250.0,  # fast motion makes the finger landmarks noisy
        "min_score": 0.90,        # real pinches/taps scored >= 0.98; tracking glitches 0.6-0.87
        "max_onset_ms": 300,      # a press must dip by onset_drop within this long: slow pose drift
        "onset_drop": 0.25,       # isn't a click. Measured from the recent peak, not from a fixed level,
                                  # because the open ratio wanders 0.85-0.95 and a fixed level started
                                  # the onset clock seconds early (11 of 21 misses in recorded sessions)
        "base_window_s": 1.2,     # open level = 85th percentile of this recent window
        "recover": 0.5,           # release once the gap recovers this fraction from its minimum
        "drag_mm": 18.0,       # held displacement from onset that turns a press into a drag (clicks shift 5-10 mm)
        "long_ms": 900,        # held still this long = right click (calibration personalises it:
                               # natural pinches lasted 0.17-0.47 s in one session, 1.1-1.2 s in another)
        "min_ext": 1.70,       # pinching finger must stay straight (curled-finger ghosts <= 1.68)
        "min_dip": 0.25,       # absolute gap change, hand-lengths (curled-grip noise 0.06-0.28; 0.30 lost a real tap)
        "release_rollback_ms": 100,
        "settle_ms": 500,      # after a hand lands: learn its open finger gap before any press counts
        "hold_mm": 8.0,        # a right click needs the hand this still for the whole hold
    },
    # a curled mouse grip reads as a fist half the time (recorded), so by default a fist does nothing;
    # "clutch" or "scroll" (with scroll.gesture = "fist") make it act
    "fist": {"action": "none"},
    # two-finger slide = scroll (PRD 3.5, the trackpad convention): index + middle out, ring + pinky
    # curled. Unlike a fist it cannot be confused with a curled mouse grip.
    "scroll": {"gesture": "two_finger", "settle_ms": 150,
               "units_per_mm": 14.0, "natural": True, "min_step": 20,
               "start_mm": 8.0,      # fist must move this far before scrolling (curled resting hands)
               "accel": 1.5,         # fast flicks scroll up to (1 + 3*accel)x further, like a trackpad
               # joystick zone (hybrid position/rate, as RubberEdge / browser autoscroll): hold the
               # fist this far from where it closed and the page keeps scrolling, faster the further
               "rate_mm": 20.0, "rate_span_mm": 40.0, "rate_units_s": 2400.0,
               "inertia_tau_s": 0.35, "inertia_min_mm_s": 60.0},
    # immediate feedback (Meta / Apple / Ultraleap hand guidelines): ring at the cursor + soft tick
    "feedback": {"overlay": True, "sound": True},
    # lift-to-reposition (engine._lift): palm-bone image size against a slow resting baseline.
    # Recorded: rests and poses stay below 1.15 of the baseline (0 false lifts in 4 sessions);
    # a 5 cm lift under a camera ~25 cm above the hand reads ~1.25
    # The size is first corrected for where the hand is in the view (engine._size_fit): forward the
    # palm looks up to 60% bigger, which read as a lift and froze the cursor going up the screen.
    # stroke_mm: a rise that came with this much sliding is perspective, not a lift (real lifts 8-16 mm)
    "clutch": {"lift_on": 1.15, "lift_off": 1.07, "land_keep": 0.35, "base_tau_s": 2.0,
               "rollback_ms": 450, "max_lift_s": 4.0, "stroke_mm": 40.0, "pinch_lift_extra": 0.05,
               "grad_tau_s": 60.0, "grad_ridge": 1.0e5},
    "pad": {"margin": 0.03, "tip_margin": 0.01},     # FR-07 pad zone, fraction of the frame
    # RubberEdge (Casiez et al., UIST 2007): pushing the hand into the band at the frame edge keeps
    # the cursor gliding at a rate set by how deep it goes, so the view's edge never ends a move
    "edge": {"enabled": True,
             "zone": 0.12,         # band width, fraction of frame height
             "rate_px_s": 1400.0,  # glide speed at full depth (scaled to screen width)
             "power": 1.6,         # rate ~ depth^power: starts from zero, no velocity jump
             "arm_mm_s": 30.0,     # only arms when the hand moves INTO the band (resting near it won't drift)
             "scroll_units_s": 1500.0,
             # gliding out through the band means "further than the camera sees": keep going until
             # the hand is back (or this long), only if the glide had already run this long
             "exit_glide_ms": 1500, "exit_min_ms": 150},
    "dwell": {"enabled": False, "ms": 800, "radius_px": 6, "arm_px": 30},
    "guard": {"typing_ms": 500},
    "idle": {"after_s": 1.0, "every_n": 3},          # run the detector less often with no hand
    "preview": True,
}

PRESETS = {
    "precision": {},
    "accessibility": {   # tremor filter, dwell click, more tolerant timing (FR-10)
        "filter": {"fc_min_hz": 0.3, "beta": 0.03},
        "gain": {"g_min": 1.5, "g_max": 9.0},
        "pointer": {"predict_ms": 0.0},
        "click": {"drag_mm": 18.0, "long_ms": 1000, "confirm_frames": 3},
        "dwell": {"enabled": True},
    },
    "fast": {
        "gain": {"g_min": 2.5, "g_max": 18.0, "v_mid_mm_s": 120.0},
        "pointer": {"predict_ms": 30.0},
    },
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=CONFIG_PATH, preset=None):
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path)
    if path.exists():
        _merge(cfg, json.loads(path.read_text(encoding="utf-8")))
    preset = preset or cfg.get("preset")
    if preset:
        _merge(cfg, copy.deepcopy(PRESETS[preset]))
        cfg["preset"] = preset
    return cfg


def load_profile(path=PROFILE_PATH):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_profile(profile, path=PROFILE_PATH):
    Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")
