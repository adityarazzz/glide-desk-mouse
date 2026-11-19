"""MediaPipe HandLandmarker wrapper (TRD 5.1). One hand keeps it in tracking mode, so the
palm detector only runs when the hand is lost."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .features import Obs, measure


class HandTracker:
    def __init__(self, model_path, hand_cfg):
        vision = mp.tasks.vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=hand_cfg["min_detection_conf"],
            min_hand_presence_confidence=hand_cfg["min_tracking_conf"],
            min_tracking_confidence=hand_cfg["min_tracking_conf"],
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.last_ms = -1
        self.raw = None   # last (img, world, score) for the session recorder

    def detect(self, frame_bgr, t):
        ms = max(int(t * 1000), self.last_ms + 1)   # VIDEO mode needs strictly increasing timestamps
        self.last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(image, ms)
        if not res.hand_landmarks:
            self.raw = None
            return Obs(present=False)
        h, w = frame_bgr.shape[:2]
        img = np.array([(p.x * w, p.y * h) for p in res.hand_landmarks[0]])
        world = np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
        score = res.handedness[0][0].score if res.handedness else 1.0
        self.raw = (img, world, score)
        return measure(img, world, score, (w, h))

    def close(self):
        self.lm.close()
