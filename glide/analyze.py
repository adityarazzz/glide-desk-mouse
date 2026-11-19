"""Replay recorded sessions and report spec metrics; optionally grid-search the tunables.

  python -m glide.analyze logs/session_*.jsonl            report with current config
  python -m glide.analyze logs/session_*.jsonl --tune     search click + pointer parameters
  python -m glide.analyze ... --tune --apply              write the winners to config.json

Ground truth comes from the recorder's labelled phases and timed cues.
"""
import argparse
import copy
import glob
import itertools
import json

import numpy as np

from .config import CONFIG_PATH, load_config
from .engine import Engine
from .replay import VirtualOut, load_meta, load_session, session_profile

CLICK_LABELS = ("pinch_idx", "pinch_mid", "tap", "rot_left_pinch", "rot_right_pinch", "side_pinch",
                "grip_pinch")
QUIET_LABELS = ("rest", "move_slow", "move_fast", "hover_rest", "lift", "rot_left", "rot_right", "side",
                "grip", "fine", "grip_move", "reposition")
LIFT_LABELS = ("lift", "reposition")
CUE_WINDOW = (-0.25, 1.6)


def run(frames, cfg, profile):
    out = VirtualOut()
    eng = Engine(cfg, profile, out)
    raw, emitted, states, cursor = [], [], [], []
    for t, label, obs in frames:
        out.t = t
        eng.update(obs, t)
        raw.append(eng.last_raw.copy() if eng.last_raw is not None else None)
        emitted.append(eng.emitted.copy() if eng.emitted is not None else None)
        states.append(eng.state)
        cursor.append(out.cursor())
    out.trace = np.array(cursor, float)
    return out, raw, emitted, states


def reach(frames, cfg, profile, labels=("move_slow", "move_fast")):
    """Fraction of the screen (w, h) the recorded strokes sweep: answers 'does my hand's range
    cover the screen'. Unclamped, from the summed cursor deltas of each move phase."""
    out, *_ = run(frames, cfg, profile)
    L = np.array([l for _, l, _ in frames])
    best = np.zeros(2)
    for lab in labels:
        tr = out.trace[L == lab]
        if len(tr) > 10:
            best = np.maximum(best, np.ptp(tr, axis=0) / [out.w, out.h])
    return [round(float(b), 2) for b in best]


def metrics(frames, cues, cfg, profile, with_jitter=True):
    out, raw, emitted, states = run(frames, cfg, profile)
    T = np.array([t for t, _, _ in frames])
    L = np.array([l for _, l, _ in frames])
    label_at = lambda t: L[min(np.searchsorted(T, t), len(L) - 1)]
    presses = [(t, kind) for t, kind, down, _ in out.events if kind in ("left", "right") and down]
    m = {}

    # clicks vs cues
    click_cues = [c["t"] for c in cues if c["label"] in CLICK_LABELS]
    # a hit = the press was recognised near the cue (its outcome may come later: long holds
    # release after the window, or turn into a right click / drag)
    st_all = np.array(states)
    onsets = [T[i] for i in range(1, len(st_all)) if st_all[i] == "PINCH" and st_all[i - 1] != "PINCH"]
    hit = [any(c + CUE_WINDOW[0] <= t <= c + CUE_WINDOW[1] for t in onsets) for c in click_cues]
    m["click_hits"] = f"{sum(hit)}/{len(click_cues)}" if click_cues else "n/a"
    m["click_recall"] = sum(hit) / len(click_cues) if click_cues else None
    # ghosts: any press or scroll during phases where the user did not gesture
    ghosts = [t for t, k in presses if label_at(t) in QUIET_LABELS]
    ghosts += [t for t, kind, _, _ in out.events if kind == "scroll" and label_at(t) in QUIET_LABELS]
    quiet_s = sum(np.diff(T)[np.isin(L[:-1], QUIET_LABELS)])
    m["ghosts"] = len(ghosts)
    m["ghosts_per_min"] = len(ghosts) / max(quiet_s / 60, 1e-9)
    # extra presses inside click phases beyond one per cue, and clicks that became drags
    in_click = [t for t in onsets if label_at(t) in CLICK_LABELS]
    m["extra_presses"] = max(0, len(in_click) - sum(hit)) if click_cues else None
    st = np.array(states)
    m["drags_in_click_phases"] = int(sum(1 for i in range(1, len(st)) if st[i] == "DRAG" and st[i - 1] != "DRAG"
                                         and L[i] in CLICK_LABELS))
    m["clicks_per_phase"] = {lab: sum(1 for t, k in presses if label_at(t) == lab and k == "left")
                             for lab in CLICK_LABELS if lab in L}
    # click precision (FR-06): where the press landed vs where the cursor was 100 ms before the
    # pinch began; the stabilizer should make this ~0 even though pinching shifts the hand 5-10 mm
    drift = []
    # lift-to-reposition: cued lifts recognised, and clutches where nobody lifted
    lifts = [T[i] for i in range(1, len(st_all)) if st_all[i] == "CLUTCH" and st_all[i - 1] != "CLUTCH"]
    lift_cues = [c["t"] for c in cues if c["label"] in LIFT_LABELS]
    if lift_cues:
        m["lift_hits"] = f"{sum(any(c - 0.3 <= t <= c + 1.6 for t in lifts) for c in lift_cues)}/{len(lift_cues)}"
    m["false_lifts"] = len([t for t in lifts if label_at(t) not in LIFT_LABELS + ("wait", "practice")
                            and not any(c - 0.3 <= t <= c + 2.5 for c in lift_cues)])
    on_idx = [i for i in range(1, len(st_all)) if st_all[i] == "PINCH" and st_all[i - 1] != "PINCH"]
    downs = [(t, pos) for t, kind, down, pos in out.events if kind in ("left", "right") and down is True]
    for j, i in enumerate(on_idx):
        # only this press's own button event: a cancelled press (fist, scroll pose) has none
        t_next = T[on_idx[j + 1]] if j + 1 < len(on_idx) else np.inf
        nxt = [pos for t, pos in downs if T[i] <= t < t_next]
        if nxt:
            ref = out.trace[max(i - 3, 0)]
            drift.append(float(np.hypot(*(np.array(nxt[0]) - ref))))
    m["click_drift_px"] = (round(float(np.median(drift)), 1), round(float(np.max(drift)), 1)) if drift else None
    m["tracked_by_phase"] = {lab: round(float(np.mean([s not in ("IDLE",) for s, l in zip(st, L) if l == lab])), 2)
                             for lab in dict.fromkeys(L)}
    m["hits_by_phase"] = {lab: f"{sum(h for h, c in zip(hit, click_cues) if label_at(c) == lab)}/"
                               f"{sum(1 for c in click_cues if label_at(c) == lab)}"
                          for lab in dict.fromkeys(label_at(c) for c in click_cues)}
    m["scroll_frac_by_phase"] = {lab: round(float(np.mean([s == "SCROLL" for s, l in zip(st, L) if l == lab])), 2)
                                 for lab in dict.fromkeys(L) if lab in ("two_scroll", "grip_move", "grip", "fist",
                                                                        "move_slow", "hover_rest")}
    if "two_scroll" in L:
        m["two_scroll_units"] = int(sum(abs(e[2][0]) for e in out.events if e[1] == "scroll"
                                        and label_at(e[0]) == "two_scroll"))
        m["presses_in_two_scroll"] = sum(1 for t, k in presses if label_at(t) == "two_scroll")
    if "scroll_long" in L:
        m["scroll_long_units"] = int(sum(abs(e[2][0]) for e in out.events
                                         if e[1] == "scroll" and label_at(e[0]) == "scroll_long"))
    fist_cues = [c["t"] for c in cues if c["label"] == "fist"]
    if "fist" in L:
        entries = [T[i] for i in range(1, len(st)) if st[i] == "SCROLL" and st[i - 1] != "SCROLL"]
        m["fist_scroll_entries"] = (f"{sum(any(c <= t <= c + 1.6 for t in entries) for c in fist_cues)}/{len(fist_cues)}"
                                    if fist_cues else len([t for t in entries if label_at(t) == 'fist']))
        m["presses_in_fist_phase"] = sum(1 for t, k in presses if label_at(t) == "fist")

    # static jitter (NFR-02): cursor spread in 1-s windows of the rest phase, px
    m["jitter_px"] = jitter(frames, cfg, profile) if with_jitter else None
    m["reach_screen"] = reach(frames, cfg, profile) if with_jitter else None
    m["ghost_at"] = [(round(t - T[0], 1), label_at(t)) for t in ghosts]
    m["lag_ms"] = lag(T, L, raw, emitted)
    return m


def jitter(frames, cfg, profile):
    out = VirtualOut()
    eng = Engine(cfg, profile, out)
    pos, labels, T = [], [], []
    for t, label, obs in frames:
        eng.update(obs, t)
        pos.append(out.cursor())
        labels.append(label)
        T.append(t)
    pos, labels, T = np.array(pos, float), np.array(labels), np.array(T)
    spreads = []
    for lab in ("rest",):
        idx = np.where(labels == lab)[0]
        if len(idx) < 40:
            continue
        idx = idx[15:]                                  # settle after the phase change
        for s in range(0, len(idx) - 30, 15):
            if idx[s + 29] - idx[s] != 29:              # coached sessions label only the still stretches
                continue
            w = pos[idx[s:s + 30]]
            spreads.append(float(np.sqrt(((w - w.mean(axis=0)) ** 2).sum(axis=1).mean())))
    return float(np.median(spreads)) if spreads else None   # median: the user is not perfectly still


def lag(T, L, raw, emitted):
    """Delay of the filtered (+predicted) anchor behind the raw anchor during the move phases, ms."""
    idx = [i for i in range(len(T)) if L[i] in ("move_slow", "move_fast") and raw[i] is not None
           and emitted[i] is not None]
    if len(idx) < 60:
        return None
    t = T[idx]
    R = np.array([raw[i] for i in idx])
    E = np.array([emitted[i] for i in idx])
    best = None
    for tau in np.arange(-0.06, 0.25, 0.005):
        Rs = np.column_stack([np.interp(t - tau, t, R[:, k]) for k in range(2)])
        err = float(np.mean(np.hypot(*(E - Rs).T)))
        if best is None or err < best[1]:
            best = (tau, err)
    return round(best[0] * 1000)


def score(ms):
    """Ghost clicks are the trust killer (PRD 3.8), so they dominate; then misses and drags."""
    s = 0.0
    for m in ms:
        s += 10 * (m["click_recall"] or 0) - 6 * m["ghosts"] - 2 * (m["extra_presses"] or 0) \
             - 3 * m["drags_in_click_phases"] - 2 * m.get("presses_in_fist_phase", 0)
    return s


def tune(sessions, cfg):
    best = None
    grid = itertools.product([0.70, 0.75, 0.80], [0.86, 0.92], [0.55, 0.62],
                             [1, 2], [250.0, 450.0], [10.0, 14.0, 18.0])
    for enter, exit_, deep, confirm, vmax, drag in grid:
        if deep >= enter or exit_ <= enter:
            continue
        c = copy.deepcopy(cfg)
        c["click"].update(enter=enter, exit=exit_, deep=deep, confirm_frames=confirm,
                          max_speed_mm_s=vmax, drag_mm=drag)
        ms = [metrics(f, q, c, p, with_jitter=False) for f, q, p in sessions]
        s = score(ms)
        if best is None or s > best[0]:
            best = (s, c["click"].copy(), ms)
    click = best[1]
    # anchor: pick the landmark set that gives the fewest click-induced drags and least jitter
    best_a = None
    for anchor in ([0, 1, 17], [0, 17], [0, 1, 5, 17], [0, 5, 9, 13, 17]):
        c = copy.deepcopy(cfg)
        c["click"].update(click)
        c["pointer"]["anchor"] = anchor
        ms = [metrics(f, q, c, p, with_jitter=False) for f, q, p in sessions]
        ms = [dict(m, jitter_px=jitter(f, c, p)) for m, (f, q, p) in zip(ms, sessions)]
        s = score(ms) - sum(m["jitter_px"] or 0 for m in ms)
        if best_a is None or s > best_a[0]:
            best_a = (s, anchor)
    # pointer filter: least lag with static jitter at or below 1 px (NFR-02)
    best_f = None
    for fc, beta, pred in itertools.product([0.6, 0.9, 1.2, 1.6, 2.2], [0.02, 0.04, 0.06, 0.1, 0.16],
                                            [0.0, 15.0, 25.0, 35.0]):
        c = copy.deepcopy(cfg)
        c["click"].update(click)
        c["pointer"]["anchor"] = best_a[1]
        c["filter"].update(fc_min_hz=fc, beta=beta)
        c["pointer"]["predict_ms"] = pred
        js = [jitter(f, c, p) for f, q, p in sessions]
        lags = [lag(*_lag_inputs(f, c, p)) for f, q, p in sessions]
        js = [j for j in js if j is not None]
        lags = [x for x in lags if x is not None]
        if not lags:
            continue
        jit, lg = max(js) if js else 0.0, float(np.mean(lags))
        cost = lg + (1000 if jit > 1.0 else 0) + 5 * jit
        if best_f is None or cost < best_f[0]:
            best_f = (cost, fc, beta, pred, jit, lg)
    return click, best_a[1], best_f


def _lag_inputs(frames, cfg, profile):
    _, raw, emitted, _ = run(frames, cfg, profile)
    return (np.array([t for t, _, _ in frames]), np.array([l for _, l, _ in frames]), raw, emitted)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    paths = [p for pat in args.sessions for p in sorted(glob.glob(pat))]
    cfg = load_config()
    sessions = []
    for p in paths:
        frames, cues = load_session(p)
        sessions.append((frames, cues, session_profile(frames, cfg)))
    for p, (f, q, prof) in zip(paths, sessions):
        print(f"\n{p}: {len(f)} frames, {len(q)} cues")
        meta = load_meta(p)
        if meta:                                     # what the coach saw while recording
            for rec in meta["summary"]:
                print("  step", " ".join(f"{k}={v}" for k, v in rec.items()))
        for k, v in metrics(f, q, cfg, prof).items():
            print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 3)}")
    if not args.tune:
        return
    click, anchor, (cost, fc, beta, pred, jit, lg) = tune(sessions, cfg)
    print("\nTuned:")
    print("  click  ", {k: click[k] for k in ("enter", "exit", "deep", "confirm_frames", "max_speed_mm_s", "drag_mm")})
    print("  anchor ", anchor)
    print(f"  filter  fc_min={fc} beta={beta} predict_ms={pred}  ->  jitter {jit:.2f} px, lag {lg:.0f} ms")
    tuned = copy.deepcopy(cfg)
    tuned["click"].update(click)
    tuned["pointer"].update(anchor=anchor, predict_ms=pred)
    tuned["filter"].update(fc_min_hz=fc, beta=beta)
    for p, (f, q, prof) in zip(paths, sessions):
        m = metrics(f, q, tuned, prof)
        print(f"  {p}: hits {m['click_hits']}, ghosts {m['ghosts']}, drags-in-clicks "
              f"{m['drags_in_click_phases']}, jitter {m['jitter_px']}, lag {m['lag_ms']} ms")
    if args.apply:
        over = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        over.setdefault("click", {}).update({k: click[k] for k in ("enter", "exit", "deep", "confirm_frames",
                                                                   "max_speed_mm_s", "drag_mm")})
        over.setdefault("pointer", {}).update(anchor=anchor, predict_ms=pred)
        over.setdefault("filter", {}).update(fc_min_hz=fc, beta=beta)
        CONFIG_PATH.write_text(json.dumps(over, indent=2), encoding="utf-8")
        print(f"\nWrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
