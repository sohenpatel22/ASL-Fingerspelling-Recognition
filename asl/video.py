from __future__ import annotations

from pathlib import Path

import numpy as np

from asl.features import hands_to_row


class NoHandsDetected(ValueError):
    """Raised when a video has no frames, or MediaPipe never finds a hand in it."""


def extract_landmarks(
    video_path: str | Path, min_detection_confidence: float = 0.5
) -> tuple[np.ndarray, float]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    rows: list[np.ndarray] = []
    detected = 0
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                left = right = None
                if result.multi_hand_landmarks:
                    detected += 1
                    for lms, handed in zip(
                        result.multi_hand_landmarks, result.multi_handedness, strict=True
                    ):
                        xy = np.array([[lm.x, lm.y] for lm in lms.landmark], dtype=np.float32)
                        if handed.classification[0].label == "Left":
                            left = xy
                        else:
                            right = xy
                rows.append(hands_to_row(left, right))
    finally:
        cap.release()

    if not rows or detected == 0:
        raise NoHandsDetected("No hands detected - use good lighting and keep hands in frame.")
    return np.vstack(rows), detected / len(rows)
