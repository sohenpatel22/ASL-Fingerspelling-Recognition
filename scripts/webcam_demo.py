from __future__ import annotations

import argparse

import cv2
import mediapipe as mp
import numpy as np

from asl.checkpoint import resolve_checkpoint
from asl.features import hands_to_row
from asl.infer import Predictor

MIN_FRAMES = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="path to weights (default: resolve)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(resolve_checkpoint(args.checkpoint))
    cap = cv2.VideoCapture(args.camera)
    frames: list[np.ndarray] = []
    recording, prediction = False, ""

    with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if recording:
                left = right = None
                for lms, handed in zip(
                    result.multi_hand_landmarks or [], result.multi_handedness or [], strict=False
                ):
                    xy = np.array([[lm.x, lm.y] for lm in lms.landmark], dtype=np.float32)
                    if handed.classification[0].label == "Left":
                        left = xy
                    else:
                        right = xy
                frames.append(hands_to_row(left, right))

            shown = cv2.flip(frame, 1)
            status = f"RECORDING ({len(frames)} frames)" if recording else "SPACE to record"
            cv2.putText(shown, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(shown, f"Pred: {prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("ASL Fingerspelling", shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                if not recording:
                    if len(frames) >= MIN_FRAMES and any(f.any() for f in frames):
                        prediction = predictor.predict_landmarks(np.vstack(frames)).text
                        print(f"Prediction: {prediction}")
                    frames.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
