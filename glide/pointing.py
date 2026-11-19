"""Pointing test: ISO 9241-411 multi-directional tapping, scored the standard way.

The app shows nine circles in a ring (four rounds: two distances x two widths). The first click
of a round starts its clock; each later click is one trial from the previous circle (a) to the
highlighted one (b), landing at c after t seconds. Per round (Soukoreff & MacKenzie 2004):

    dx  = endpoint deviation along the a->b axis,   De = mean effective distance
    We  = 4.133 * SD(dx)                            IDe = log2(De / We + 1)
    TP  = IDe / MT  (bits/s)                        MT = mean movement time

The test's throughput is the mean of the rounds' throughputs. Error rate = clicks outside the
circle. Results are saved to logs/pointing_*.json (positions and times only) so the test report
can compare Glide with the same person's mouse (PRD 3.8: target 80%).
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import ROOT

DEVICES = ("glide", "mouse")
MIN_TRIALS = 4


def score_round(trials, width):
    """trials: [{"a": [x, y], "b": [x, y], "c": [x, y], "t": s}]; width: target diameter, px."""
    rows = [(np.asarray(r["a"], float), np.asarray(r["b"], float), np.asarray(r["c"], float), float(r["t"]))
            for r in trials if r.get("t", 0) > 0]
    if len(rows) < MIN_TRIALS:
        return None
    dx, de, mt, miss = [], [], [], 0
    for a, b, c, t in rows:
        L = float(np.linalg.norm(b - a))
        u = (b - a) / max(L, 1e-6)
        proj = float((c - a) @ u)
        dx.append(proj - L)
        de.append(proj)
        mt.append(t)
        miss += float(np.linalg.norm(c - b)) > width / 2
    we = max(4.133 * float(np.std(dx)), 1.0)
    ide = math.log2(max(float(np.mean(de)), 0.0) / we + 1)
    amp = float(np.mean([np.linalg.norm(b - a) for a, b, _, _ in rows]))
    return {"w": float(width), "a": round(amp, 1), "id": round(math.log2(amp / width + 1), 2),
            "we": round(we, 1), "ide": round(ide, 2), "mt": round(float(np.mean(mt)), 3),
            "tp": round(ide / float(np.mean(mt)), 2), "errors": int(miss), "n": len(rows)}


def score(rounds):
    """rounds: [{"w": px, "trials": [...]}] -> per-round scores and the test's summary."""
    per = [s for s in (score_round(r.get("trials") or [], float(r.get("w") or 0) or 1.0) for r in rounds) if s]
    if not per:
        return None
    n = sum(p["n"] for p in per)
    return {"rounds": per, "tp": round(float(np.mean([p["tp"] for p in per])), 2),
            "mt": round(float(np.mean([p["mt"] for p in per])), 3),
            "error_rate": round(sum(p["errors"] for p in per) / n, 3), "trials": n}


def save(device, rounds, board=None, folder=None):
    """Scores and saves one finished test. Returns the saved record, or None if it can't be scored."""
    if device not in DEVICES:
        return None
    res = score(rounds)
    if res is None:
        return None
    rec = {"device": device, "when": time.strftime("%Y-%m-%d %H:%M:%S"), **res,
           "board": [int(v) for v in (board or [])][:2], "raw": rounds}
    folder = Path(folder or ROOT / "logs")
    folder.mkdir(exist_ok=True)
    (folder / time.strftime(f"pointing_{device}_%Y%m%d_%H%M%S.json")).write_text(json.dumps(rec), encoding="utf-8")
    return rec


def history(limit=20, folder=None):
    """Saved tests, newest first, without the raw clicks."""
    out = []
    files = sorted(Path(folder or ROOT / "logs").glob("pointing_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:limit]:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: r.get(k) for k in ("device", "when", "tp", "mt", "error_rate", "trials")})
    return out


def compare(hist):
    """Glide's best recent throughput as a share of the mouse's (latest of each), or None."""
    g = next((h for h in hist if h["device"] == "glide"), None)
    m = next((h for h in hist if h["device"] == "mouse"), None)
    if not (g and m and m.get("tp")):
        return None
    return {"glide": g["tp"], "mouse": m["tp"], "ratio": round(g["tp"] / m["tp"], 2)}
