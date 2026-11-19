"""Test report: every requirement of the project pack -> test points -> evidence, in one command.

  python -m glide.report                 automated tests + recorded-session replay + saved results
  python -m glide.report --live 20       also measure the running Glide for 20 s (fps, latency,
                                         CPU, memory, network) -- start Glide first

Writes TEST_REPORT.md (for people) and TEST_REPORT.json (for tools) next to this package.

Levels follow the test plan (doc 6.1):
  Unit     property tests on synthetic hands and trajectories (tests/)
  Replay   the full pipeline on recorded, labelled sessions (landmarks only, never images)
  Live     measured on the running app with the real camera
  Manual   needs a person or extra equipment; results are entered in tests/manual_results.json
"""
import argparse
import ctypes
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from . import pointing
from .config import ROOT, load_config

TESTS = ROOT / "tests"
DATA = TESTS / "data"
MANUAL = TESTS / "manual_results.json"
OUT_MD, OUT_JSON = ROOT / "TEST_REPORT.md", ROOT / "TEST_REPORT.json"

# quiet phases of the coached recordings: ordinary use, and deliberately awkward poses
NORMAL_QUIET = ("rest", "move_slow", "move_fast", "hover_rest", "fine", "reposition")
STRESS_QUIET = ("lift", "rot_left", "rot_right", "side", "grip", "grip_move")

# ---- the traceability matrix ---------------------------------------------------------------
# tests: unittest id prefixes (module.Class[.test]); metric: key in the measured values, checked
# by its rule; manual: what a person does. P1 = the webcam prototype (doc 8, phase P1).
POINTS = [
    # functional requirements (doc 3.6)
    dict(id="TP-01", req="FR-01", title="Tracks one dominant hand; the other hand and mirrored poses are ignored",
         level="Unit", tests=["test_engine.PointerTests.test_other_hand_is_ignored", "test_angles.TiltTests",
                              "test_features.RealFrameTests"]),
    dict(id="TP-02", req="FR-01", title="Camera frame rate (unique pictures per second)", level="Live",
         metric="live_fps", rule=(">=", 24), unit="fps",
         note="Spec asks 60 fps on the Tier 1 IR camera. P1 runs on a 30 fps webcam; 24 is the placement "
              "check's floor. A dim room halves it (auto exposure), so the demo needs a desk lamp."),
    dict(id="TP-03", req="FR-02", title="Desk-millimetre scale and axes from a 2 s still-hand calibration",
         level="Unit", tests=["test_features.RealFrameTests.test_rest_calibration_orientation",
                              "test_engine.PointerTests.test_axes", "test_workspace.WorkspaceTests",
                              "test_angles.AdaptiveAxesTests", "test_angles.MouseFrameTests"]),
    dict(id="TP-04", req="FR-02", title="Recalibration takes 30 s or less", level="Manual",
         manual="Settings > Recalibrate. Time from pressing it to 'Calibrated' (2 s still hand + 5 paced pinches)."),
    dict(id="TP-05", req="FR-03", title="Pinch contact detected with hysteresis; drift, slow poses and doubtful frames never press",
         level="Unit", tests=["test_engine.ClickTests", "test_engine.PhantomClickTests"]),
    dict(id="TP-06", req="FR-03", title="Cued clicks detected in the golden recorded session", level="Replay",
         metric="click_recall", rule=(">=", 16 / 18), unit="", fmt="pct",
         note="v1.0 target is 97% with the Tier 1 IR touch sensor; P1 detects a pinch seen from above."),
    dict(id="TP-07", req="FR-04", title="Relative cursor with a speed-dependent gain that learns from your clicks",
         level="Unit", tests=["test_engine.PointerTests.test_fast_moves_get_more_gain_per_mm",
                              "test_smoothness.RestingGainTests", "test_autogain", "test_workspace.AxisGainTests"]),
    dict(id="TP-08", req="FR-04", title="Recorded strokes reach across the screen", level="Replay",
         metric="reach_x", rule=(">=", 0.5), unit="of screen width", fmt="pct"),
    dict(id="TP-09", req="FR-05", title="Every gesture is emitted as a native OS event: click, drag, right click, scroll, clutch",
         level="Unit", tests=["test_engine.ClickTests", "test_engine.ScrollTests", "test_engine.RepositionTests",
                              "test_smoothness.SmoothOutputTests", "test_angles.RateScrollTests"]),
    dict(id="TP-10", req="FR-05", title="Scroll pose scrolls; mouse grip and fist never scroll or click", level="Replay",
         tests=["test_replay.GoldenScrollTests"], metric="scroll_frac", rule=(">=", 0.95), unit="of the scroll step", fmt="pct"),
    dict(id="TP-11", req="FR-06", title="Click stabilizer: the click lands where the pinch began", level="Unit",
         tests=["test_engine.ClickTests.test_quick_pinch_clicks_where_it_started",
                "test_engine.ClickTests.test_click_shift_does_not_become_a_drag",
                "test_engine.SensorPointTests.test_knuckles_shifting_in_a_pinch_do_not_move_the_cursor",
                "test_replay.GoldenReplayTests.test_clicks_do_not_turn_into_drags"]),
    dict(id="TP-12", req="FR-06", title="Click drift in the golden session (median)", level="Replay",
         metric="click_drift_median_px", rule=("<=", 3.0), unit="px"),
    dict(id="TP-13", req="FR-07", title="Intent gate: typing guard, view margins, pause hotkey and tray toggle",
         level="Unit", tests=["test_engine.PointerTests.test_typing_guard_blocks_motion_and_clicks",
                              "test_surface.CrampedDeskTests", "test_workspace.EdgeGlideTests"],
         note="The palm-flat pause gesture is not in P1; Ctrl+Alt+G and the tray pause instead."),
    dict(id="TP-14", req="FR-08", title="Guided onboarding and coached training: self-paced, demonstrated, checked",
         level="Unit", tests=["test_coach", "test_app.LearnTests", "test_app.DemoExportTests"]),
    dict(id="TP-15", req="FR-08", title="First run, unboxing to first click, in 3 minutes or less", level="Manual",
         manual="Settings > Run full setup with a stopwatch: camera, placement, calibration, gestures, practice."),
    dict(id="TP-16", req="FR-09", title="Live view: camera, landmarks, state, fps and latency on Home", level="Manual",
         manual="Home shows the hand skeleton, the state word, Hand/Camera/Room readouts. Visual check in the demo."),
    dict(id="TP-17", req="FR-10", title="Profiles: Steady (tremor filter, dwell click), Fast; settings round-trip",
         level="Unit", tests=["test_app.SettingsTests", "test_engine.ClickTests.test_dwell_click"],
         note="Multi-monitor profiles are not in P1."),
    # non-functional requirements (doc 3.7)
    dict(id="TP-18", req="NFR-01", title="Pipeline latency, capture to cursor command (P50)", level="Live",
         metric="live_latency_p50_ms", rule=("<=", 90), unit="ms",
         note="Tier 0 motion-to-photon target is P50 90 ms. This is the software part; TP-19 measures end to end."),
    dict(id="TP-19", req="NFR-01", title="Motion-to-photon latency with a 240 fps phone camera", level="Manual",
         manual="Film the hand and the screen in slow motion; count frames from the hand starting to move to the "
                "cursor starting to move (10 trials, report P50/P95). 1 frame = 4.2 ms."),
    dict(id="TP-20", req="NFR-01", title="Filter and prediction add no lag over raw tracking", level="Replay",
         metric="lag_ms", rule=("<=", 30), unit="ms", absval=True),
    dict(id="TP-21", req="NFR-02", title="Static cursor jitter with the hand still (1080p)", level="Replay",
         tests=["test_replay.GoldenReplayTests.test_static_jitter_meets_nfr02", "test_engine.PointerTests.test_still_noisy_hand_does_not_jitter"],
         metric="jitter_px", rule=("<=", 1.0), unit="px"),
    dict(id="TP-22", req="NFR-03", title="Robustness: camera renumbered, frozen or dark; desk cluttered; hand leaving fast",
         level="Unit", tests=["test_app.CameraIdTests", "test_app.NeverStuckTests", "test_app.CameraFinderTests",
                              "test_surface", "test_replay.DoubtfulFrameTests", "test_engine.ReturnTests"]),
    dict(id="TP-23", req="NFR-03", title="Lighting and surface matrix (5-2000 lux; wood, white, black, patterned)",
         level="Manual", manual="Repeat the pointing test under a lamp, room light and daylight, on two desk surfaces."),
    dict(id="TP-24", req="NFR-04", title="Fairness across skin tones, hand sizes, accessories", level="Manual",
         manual="Needs a participant pool (doc 6.1 robustness matrix); not possible with one user.",
         note="Planned for the user study."),
    dict(id="TP-25", req="NFR-05", title="Engine compute per frame (filters, gestures, gain)", level="Replay",
         metric="engine_us_p50", rule=("<=", 2000), unit="us"),
    dict(id="TP-26", req="NFR-05", title="CPU use of the running app, share of all cores", level="Live",
         metric="live_cpu_total_pct", rule=("<=", 10), unit="%",
         note="Spec assumes a GPU/NPU delegate; P1 runs MediaPipe on the CPU. Measure with the hand moving in "
              "view: with no hand, Glide checks only every few frames."),
    dict(id="TP-27", req="NFR-05", title="Memory of the running app (working set)", level="Live",
         metric="live_ram_mb", rule=("<=", 300), unit="MB", target="v1.0",
         note="The hand model and MediaPipe take about 110 MB, the .NET runtime hosting the window about 25 MB. "
              "Limiting NumPy's math library to one thread cut private memory by 354 MB."),
    dict(id="TP-28", req="NFR-06", title="Privacy: no image is ever written; the API is local and token-protected",
         level="Unit", tests=["test_privacy", "test_app.ServerTests"]),
    dict(id="TP-29", req="NFR-06", title="No network connection leaves the computer (running app)", level="Live",
         metric="live_remote_connections", rule=("==", 0), unit="connections"),
    dict(id="TP-30", req="NFR-07", title="Safety: kill-switch hotkey, release on tracking loss, no click on an uncertain state",
         level="Unit", tests=["test_engine.ClickTests.test_lost_hand_releases_held_drag",
                              "test_engine.ClickTests.test_low_confidence_frames_cannot_click",
                              "test_engine.PointerTests.test_single_frame_glitch_is_ignored",
                              "test_replay.DoubtfulFrameTests"],
         note="Ctrl+Alt+G (pause) and Ctrl+Alt+Q (quit) are checked live in the demo."),
    dict(id="TP-31", req="NFR-08", title="Accessibility: every control has a label; reachable by keyboard; one-handed",
         level="Unit", tests=["test_ui"]),
    dict(id="TP-32", req="NFR-09", title="Runs without administrator rights", level="Manual",
         manual="Start Glide.vbs from a standard user account.", note="Signed installers are a v1.0 item."),
    # success metrics (doc 3.8) and the P1 exit criterion (doc 8)
    dict(id="TP-33", req="3.8 Trust", title="No ghost press while resting, sliding or hovering (every coached recording)",
         level="Replay", metric="ghosts_normal", rule=("==", 0), unit="presses", show="ghosts_normal_text",
         note="A regression gate on the recordings' ordinary-use phases. Minutes of data cannot show the "
              "per-hour target; TP-38 does."),
    dict(id="TP-34", req="3.8 Trust", title="Ghost presses in awkward poses (tilted, curled, mouse grip, lifting)",
         level="Replay", metric="ghosts_stress", unit="presses", rule=None, show="ghosts_stress_text",
         note="These phases ask for poses that look like a pinch from above. Reported, not gated: telling them "
              "apart needs the touch signal of the Tier 1 IR camera (doc 2.1, P2)."),
    dict(id="TP-35", req="3.8 Typing", title="No click while typing", level="Unit",
         tests=["test_engine.PointerTests.test_typing_guard_blocks_motion_and_clicks"]),
    dict(id="TP-36", req="3.8 Performance", title="Pointing throughput (ISO 9241-411) relative to the same person's mouse",
         level="Manual", metric="pointing_ratio", rule=(">=", 0.80), fmt="pct", unit="of mouse",
         manual="Training > Pointing test, once With Glide and once With mouse or touchpad.",
         note="The PRD target (80%) is after 30 minutes of practice."),
    dict(id="TP-37", req="P1 exit (M1)", title="Browse the web with Glide for 10 minutes", level="Manual",
         manual="Open a browser; search, open links, scroll, switch tabs for 10 minutes using only Glide."),
    dict(id="TP-38", req="3.8 Trust", title="Soak: unintended clicks per hour of real use (target < 0.2)", level="Manual",
         manual="Use Glide for an hour of ordinary work; press Ctrl+Alt+X after every click you did not mean. "
                "Count logs/misfire_*.jsonl files that contain a click (doc 6.1 asks 8 h for v1.0)."),
]


# ---- automated tests -----------------------------------------------------------------------

class _Collect(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.outcome = {}

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcome[test.id()] = "pass"

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcome[test.id()] = "fail"

    def addError(self, test, err):
        super().addError(test, err)
        self.outcome[test.id()] = "error"

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcome[test.id()] = "skip"


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(TESTS), top_level_dir=str(ROOT))
    with open(os.devnull, "w") as sink:
        runner = unittest.TextTestRunner(stream=sink, resultclass=_Collect, verbosity=0)
        t0 = time.perf_counter()
        res = runner.run(suite)
    out = {k.replace("tests.", "", 1): v for k, v in res.outcome.items()}
    return out, time.perf_counter() - t0


def tests_for(prefixes, outcome):
    return {k: v for k, v in outcome.items() if any(k == p or k.startswith(p + ".") for p in prefixes)}


# ---- replay measurements ---------------------------------------------------------------------

def replay_metrics():
    from .analyze import QUIET_LABELS, metrics
    from .engine import Engine
    from .replay import VirtualOut, load_session, session_profile
    cfg = load_config(path=Path("__no_user_config__"))
    m = {}
    frames, cues = load_session(DATA / "golden_session.jsonl")
    prof = session_profile(frames, cfg)
    g = metrics(frames, cues, cfg, prof)
    hits = g["click_hits"]
    m.update(click_recall=g["click_recall"], click_hits=hits, jitter_px=round(g["jitter_px"], 2), lag_ms=g["lag_ms"],
             reach_x=g["reach_screen"][0], ghosts_golden=g["ghosts"],
             click_drift_median_px=g["click_drift_px"][0] if g["click_drift_px"] else None,
             click_drift_max_px=g["click_drift_px"][1] if g["click_drift_px"] else None,
             drags_in_click_phases=g["drags_in_click_phases"], false_lifts=g["false_lifts"])
    # engine compute time per frame
    out, eng = VirtualOut(), Engine(cfg, prof, VirtualOut())
    eng.out = out
    times = []
    for t, _, o in frames:
        out.t = t
        t0 = time.perf_counter()
        eng.update(o, t)
        times.append((time.perf_counter() - t0) * 1e6)
    m["engine_us_p50"], m["engine_us_p95"] = round(float(np.median(times))), round(float(np.percentile(times, 95)))
    sf, sc = load_session(DATA / "golden_scroll.jsonl")
    s = metrics(sf, sc, cfg, session_profile(sf, cfg), with_jitter=False)
    m["scroll_frac"] = s["scroll_frac_by_phase"].get("two_scroll")
    # ghost presses per hour of quiet time, over every coached recording
    files = [DATA / "golden_session.jsonl", DATA / "golden_scroll.jsonl"]
    seen = {p.stat().st_size for p in files}
    for f in sorted(glob.glob(str(ROOT / "logs" / "session_*.jsonl"))):
        if Path(f).stat().st_size not in seen:                 # the golden files are copies of two of these
            files.append(Path(f))
    quiet = {"normal": 0.0, "stress": 0.0}
    ghosts = {"normal": 0, "stress": 0}
    used = []
    for f in files:
        fr, cu = load_session(f)
        if not cu:
            continue
        r = metrics(fr, cu, cfg, session_profile(fr, cfg), with_jitter=False)
        T = np.array([t for t, _, _ in fr])
        L = np.array([lab for _, lab, _ in fr])
        dt = np.diff(T)
        for grp, labs in (("normal", NORMAL_QUIET), ("stress", STRESS_QUIET)):
            quiet[grp] += float(dt[np.isin(L[:-1], labs)].sum())
            ghosts[grp] += sum(1 for _, lab in r["ghost_at"] if lab in labs)
        used.append(f.name)
    for grp in quiet:
        m[f"ghosts_{grp}"] = ghosts[grp]
        m[f"quiet_{grp}_min"] = round(quiet[grp] / 60, 1)
        m[f"ghosts_{grp}_text"] = f"{ghosts[grp]} in {quiet[grp] / 60:.1f} min"
    m["recordings"] = used
    return m


# ---- live measurements -----------------------------------------------------------------------

class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + \
               [(n, ctypes.c_size_t) for n in ("peak_ws", "ws", "qpp", "qp", "qnpp", "qnp", "pagefile", "peak_pagefile")]


def _proc_times(pid):
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)                   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        c, e, kt, ut = _FT(), _FT(), _FT(), _FT()
        k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut))
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        k.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        cpu = ((kt.hi << 32) | kt.lo) + ((ut.hi << 32) | ut.lo)   # 100 ns units
        return cpu / 1e7, pmc.ws / 2 ** 20, pmc.pagefile / 2 ** 20  # pagefile usage = private bytes
    finally:
        k.CloseHandle(h)


def _connections(pid):
    """(listening addresses, remote non-loopback peers) of a process, from netstat."""
    txt = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20).stdout
    listen, remote = [], []
    for line in txt.splitlines():
        p = line.split()
        if len(p) >= 5 and p[0] == "TCP" and p[-1] == str(pid):
            if p[3] == "LISTENING":
                listen.append(p[1])
            elif not re.match(r"(127\.0\.0\.1|\[::1\]|0\.0\.0\.0):", p[2]):
                remote.append(p[2])
    return listen, remote


def live_metrics(seconds):
    inst = ROOT / "models" / ".instance.json"
    if not inst.exists():
        return {"live_error": "Glide is not running"}
    info = json.loads(inst.read_text(encoding="utf-8"))
    port = info["base"].rsplit(":", 1)[1]
    txt = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20).stdout
    pid = next((int(p.split()[-1]) for p in txt.splitlines()
                if f"127.0.0.1:{port}" in p and "LISTENING" in p), None)
    if pid is None:
        return {"live_error": "Glide's local port is not listening"}
    lat, fps, rep, cam, hand = [], [], [], None, []
    before = _proc_times(pid)
    t0 = time.perf_counter()
    with urllib.request.urlopen(info["base"] + "/events?t=" + info["token"], timeout=10) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            s = json.loads(line[6:])
            if s.get("latency_ms") is not None:
                lat.append(s["latency_ms"])
            if s.get("fps"):
                fps.append(s["fps"])
                rep.append(s.get("repeats") or 0)
            cam = s.get("camera_name") or cam
            hand.append(bool(s.get("hand")))
            if time.perf_counter() - t0 > seconds:
                break
    wall = time.perf_counter() - t0
    after = _proc_times(pid)
    listen, remote = _connections(pid)
    m = {"live_seconds": round(wall, 1), "live_camera": cam, "live_hand_share": round(float(np.mean(hand)), 2) if hand else 0}
    if lat:
        m.update(live_latency_p50_ms=round(float(np.median(lat))), live_latency_p95_ms=round(float(np.percentile(lat, 95))))
    if fps:
        m.update(live_fps=round(float(np.median(fps)), 1), live_repeats=round(float(np.median(rep)), 2))
    if before and after:
        core = (after[0] - before[0]) / wall * 100
        m.update(live_cpu_core_pct=round(core, 1), live_cpu_total_pct=round(core / (os.cpu_count() or 1), 1),
                 live_ram_mb=round(after[1]), live_private_mb=round(after[2]))
    m.update(live_listening=listen, live_remote_connections=len(remote))
    return m


# ---- results -----------------------------------------------------------------------------------

def check(rule, v):
    if rule is None or v is None:
        return None
    op, x = rule
    return {">=": v >= x, "<=": v <= x, "==": v == x}[op]


def fmt(v, p):
    if v is None:
        return "not measured"
    if p.get("fmt") == "pct":
        return f"{v * 100:.0f}%"
    if isinstance(v, float):
        v = round(v, 2)
    return f"{v} {p.get('unit', '')}".strip()


def evaluate(outcome, measured, manual):
    rows = []
    for p in POINTS:
        tests = tests_for(p.get("tests", []), outcome)
        bad = [k for k, v in tests.items() if v in ("fail", "error")]
        val = measured.get(p.get("metric")) if p.get("metric") else None
        if val is not None and p.get("absval"):
            val = abs(val)
        ok = check(p.get("rule"), val)
        man = manual.get(p["id"]) or {}
        if p.get("tests") and not tests:
            status = "NO TESTS"                               # a renamed test must not pass silently
        elif ok is False and p.get("target") == "v1.0" and not bad:
            status = "GAP (v1.0)"                             # a v1.0 budget the P1 prototype doesn't meet yet
        elif bad or ok is False or man.get("result") == "fail":
            status = "FAIL"
        elif man.get("result") == "pass" or ok is True or (tests and not p.get("metric")):
            status = "PASS"
        elif val is not None and p.get("rule") is None:
            status = "REPORTED"
        else:
            status = "TO DO"                                  # needs the live app or a person
        crit = ""
        if p.get("rule"):
            op, x = p["rule"]
            crit = f"{op} {fmt(x, p)}"
        rows.append({**{k: p.get(k) for k in ("id", "req", "title", "level", "note", "manual")},
                     "criterion": crit or ("all tests pass" if p.get("tests") else "done"),
                     "measured": (measured.get(p["show"]) if p.get("show") else fmt(val, p)) if p.get("metric") and val is not None
                     else (man.get("value") or (fmt(val, p) if p.get("metric") else "")),
                     "tests": f"{sum(v == 'pass' for v in tests.values())}/{len(tests)}" if tests else "",
                     "failed": bad, "status": status, "manual_note": man.get("note", "")})
    return rows


def machine():
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    return {"os": platform.platform(), "cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version()}


def write_md(rows, outcome, secs, measured, mach):
    n = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "TO DO", "REPORTED", "NO TESTS", "GAP (v1.0)")}
    passed = sum(v == "pass" for v in outcome.values())
    L = [f"# Glide P1 test report",
         "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m glide.report`"
         f"{' --live ' + str(int(measured['live_seconds'])) if measured.get('live_seconds') else ''}. "
         f"Requirement IDs refer to the project pack (`Glide_BareHand_Desk_Mouse_Project_Docs.pdf`).",
         "",
         f"Machine: {mach['cpu']} ({mach['cores']} logical cores), {mach['os']}, Python {mach['python']}."
         + (f" Camera: {measured['live_camera']}." if measured.get("live_camera") else ""),
         "",
         "## Summary",
         "",
         f"- **Automated tests: {passed}/{len(outcome)} pass** ({secs:.0f} s)",
         f"- **Test points: {n['PASS']} pass, {n['FAIL']} fail, {n['GAP (v1.0)']} v1.0 gap, {n['REPORTED']} reported, "
         f"{n['TO DO']} to do** (of {len(rows)}; 'to do' = needs a person or equipment, see the procedures below)",
         "- Status key: PASS meets the criterion; FAIL misses it; GAP (v1.0) misses a budget the project pack sets "
         "for the v1.0 native build, not for this Python prototype; REPORTED is measured without a gate.",
         "",
         "## Traceability matrix",
         "",
         "| Test point | Requirement | What is checked | Level | Criterion | Measured | Tests | Status |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['req']} | {r['title']} | {r['level']} | {r['criterion']} | {r['measured']} | "
                 f"{r['tests']} | **{r['status']}** |")
    notes = [r for r in rows if r.get("note") or r.get("failed") or r.get("manual_note")]
    if notes:
        L += ["", "### Notes", ""]
        for r in notes:
            bits = [r.get("note") or "", r.get("manual_note") or ""]
            if r["failed"]:
                bits.append("Failed: " + ", ".join(r["failed"]))
            L.append(f"- **{r['id']}**: " + " ".join(b for b in bits if b))
    todo = [r for r in rows if r["level"] == "Manual"]
    L += ["", "## Manual test procedures", "",
          "Enter results in `tests/manual_results.json` (`{\"TP-xx\": {\"result\": \"pass\", \"value\": \"...\", "
          "\"note\": \"...\"}}`) and run the report again.", ""]
    for r in todo:
        L.append(f"- **{r['id']} ({r['req']}) {r['title']}.** {r.get('manual') or ''}")
    m = measured
    L += ["", "## Measurements", "",
          "| Quantity | Value |", "|---|---|",
          f"| Cued clicks detected (golden session) | {m.get('click_hits')} |",
          f"| Click drift, median / max | {m.get('click_drift_median_px')} / {m.get('click_drift_max_px')} px |",
          f"| Static jitter (hand still, 1080p) | {m.get('jitter_px')} px |",
          f"| Filter + prediction lag vs raw tracking | {m.get('lag_ms')} ms |",
          f"| Engine time per frame, P50 / P95 | {m.get('engine_us_p50')} / {m.get('engine_us_p95')} us |",
          f"| Ghost presses, ordinary use | {m.get('ghosts_normal')} in {m.get('quiet_normal_min')} min |",
          f"| Ghost presses, awkward poses | {m.get('ghosts_stress')} in {m.get('quiet_stress_min')} min |",
          f"| Recordings replayed | {len(m.get('recordings', []))} |"]
    if m.get("live_seconds"):
        L += [f"| Live: camera rate (unique pictures) | {m.get('live_fps')} fps ({int((m.get('live_repeats') or 0) * 100)}% repeats dropped) |",
              f"| Live: capture to cursor command, P50 / P95 | {m.get('live_latency_p50_ms')} / {m.get('live_latency_p95_ms')} ms |",
              f"| Live: CPU, one core / all cores | {m.get('live_cpu_core_pct')}% / {m.get('live_cpu_total_pct')}% |",
              f"| Live: memory, working set / private | {m.get('live_ram_mb')} / {m.get('live_private_mb')} MB |",
              f"| Live: listening on | {', '.join(m.get('live_listening') or [])} |",
              f"| Live: hand in view during the sample | {int(m.get('live_hand_share', 0) * 100)}% |"]
    elif m.get("live_error"):
        L.append(f"| Live measurements | not taken: {m['live_error']} |")
    if m.get("pointing"):
        for d, t in m["pointing"].items():
            L.append(f"| Pointing test, {d} ({t['when']}) | {t['tp']} bits/s, {t['mt']} s per click, "
                     f"{int(t['error_rate'] * 100)}% missed |")
    L += ["", "## How to reproduce", "",
          "```", ".venv\\Scripts\\python -m unittest discover -s tests -t .   # automated tests",
          ".venv\\Scripts\\python -m glide.report --live 20             # this report (start Glide first)",
          ".venv\\Scripts\\python -m glide.analyze tests\\data\\golden_session.jsonl   # replay metrics in detail", "```", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=float, default=0, help="seconds to measure the running app")
    args = ap.parse_args(argv)
    print("Running the automated tests...")
    outcome, secs = run_tests()
    print(f"  {sum(v == 'pass' for v in outcome.values())}/{len(outcome)} pass")
    print("Replaying the recorded sessions...")
    measured = replay_metrics()
    hist = pointing.history()
    cmp_ = pointing.compare(hist)
    if cmp_:
        measured["pointing_ratio"] = cmp_["ratio"]
    measured["pointing"] = {d: next(h for h in hist if h["device"] == d) for d in pointing.DEVICES
                            if any(h["device"] == d for h in hist)}
    if args.live:
        print(f"Measuring the running app for {args.live:g} s...")
        measured.update(live_metrics(args.live))
    manual = json.loads(MANUAL.read_text(encoding="utf-8")) if MANUAL.exists() else {}
    manual = {k: v for k, v in manual.items() if k.startswith("TP-")}
    rows = evaluate(outcome, measured, manual)
    mach = machine()
    write_md(rows, outcome, secs, measured, mach)
    OUT_JSON.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": mach,
                                    "tests": outcome, "measured": measured, "points": rows}, indent=1, default=str),
                        encoding="utf-8")
    for r in rows:
        print(f"  {r['id']:6s} {r['status']:9s} {r['req']:14s} {r['measured'] or r['tests']}")
    print(f"Wrote {OUT_MD.name} and {OUT_JSON.name}")
    return 0 if not any(r["status"] == "FAIL" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
