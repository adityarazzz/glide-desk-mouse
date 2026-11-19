"""Capture thread: always hands the newest frame to the pipeline and drops stale ones (TRD 4.4)."""
import threading
import time

import cv2
import numpy as np


BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}


def open_capture(index, cam):
    # Media Foundation: the ZEB LIVE PRO only offers YUY2 at ~17 fps through DirectShow, 30 fps here
    cap = cv2.VideoCapture(index, BACKENDS[cam.get("backend", "msmf")])
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    cap.set(cv2.CAP_PROP_FPS, cam["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cam.get("exposure") is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, cam["exposure"])
    return cap


def is_repeat(a, b):
    """A repeated picture decodes to the same pixels (mean difference < 0.05 measured on the ZEB);
    two real frames of even a still scene differ by sensor noise (0.4 or more at this size)."""
    return float(np.abs(a - b).mean()) < 0.1


def find_working(cam_cfg, skip=(), n=4, frames=15, dark=12.0):
    """Index of the first camera that opens and shows a real picture (not black), or None.

    Windows numbers only the cameras that work right now, so the desk camera's number changes when
    another one (a laptop's own, behind a closed lid) appears or fails: index 1 became index 0.
    """
    for i in range(n):
        if i in skip:
            continue
        cap = open_capture(i, cam_cfg)
        if cap is None:
            continue
        try:
            got = [f for ok, f in (cap.read() for _ in range(frames)) if ok]
        finally:
            cap.release()
        if got and float(np.mean([f.mean() for f in got[-3:]])) >= dark:
            return i
    return None


class Camera(threading.Thread):
    def __init__(self, cam_cfg):
        super().__init__(daemon=True)
        self.cap = open_capture(cam_cfg["index"], cam_cfg)
        if self.cap is None or not self.cap.read()[0]:
            raise RuntimeError(
                f"Camera {cam_cfg['index']} could not be opened. Close other apps using it "
                f"(Windows Camera, Zoom, Teams...) or pick another with --list-cameras.")
        cw, ch = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pw = cam_cfg.get("process_width") or cw
        # capture high, average down: the camera's own 640x360 mode aliases (2.0 mm frame-to-frame
        # anchor noise, measured) while 1280x720 area-downscaled to 640 gives 0.66 mm, same inference cost
        self.size = (pw, round(ch * pw / cw)) if pw < cw else (cw, ch)
        self.capture_size = (cw, ch)
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.fps, self.interval = 0.0, 1 / 30
        self.drop_repeats = cam_cfg.get("drop_repeats", True)
        self.repeats = 0.0        # share of delivered frames that repeated the previous picture
        self.brightness = 128.0   # mean grey level (0-255): near 0 = lights off or the lens covered
        self.running = True

    def run(self):
        """Media Foundation hands frames over in bursts (two ~1 ms apart, then a gap) while its
        sample timestamps are exactly periodic, so capture time comes from the driver clock,
        mapped onto perf_counter by the smallest arrival-minus-capture offset seen (TRD 4.4)."""
        last_t, offset, thumb = None, None, None
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos > 0:
                off = now - pos
                # creep upward slowly so clock drift between camera and PC can't accumulate
                offset = off if offset is None else min(off, offset + 2e-5)
                t = pos + offset
            else:
                t = now
            if self.size != self.capture_size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            small = cv2.resize(frame, (80, 45), interpolation=cv2.INTER_AREA).astype(np.int16)
            self.brightness += 0.1 * (float(small.mean()) - self.brightness)
            if self.drop_repeats:
                repeat = thumb is not None and is_repeat(small, thumb) and t - last_t < 0.2
                self.repeats += 0.02 * (repeat - self.repeats)
                if repeat:
                    continue                              # same picture again: not a new sample
                thumb = small
            if last_t is not None and t > last_t:
                self.interval += 0.05 * (t - last_t - self.interval)
                self.fps = 1.0 / max(self.interval, 1e-3)
            last_t = t
            with self.cond:
                self.frame, self.t, self.seq = frame, t, self.seq + 1
                self.cond.notify_all()

    def get(self, after_seq, timeout=1.0):
        """Block until a frame newer than after_seq arrives; returns (seq, frame, t) or None."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > after_seq, timeout):
                return None
            return self.seq, self.frame, self.t

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        self.cap.release()
