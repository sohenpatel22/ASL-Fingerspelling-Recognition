"""
Video processing pipeline for ASL Fingerspelling Recognition.

Converts video frames to MediaPipe hand landmarks using the same format
as training data (84-dimensional vectors per frame).

"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import mediapipe as mp

from config import FEATURE_SIZE, MAX_FRAMES, NORMALIZE_LANDMARKS


class MediaPipeExtractor:
    """Extract hand landmarks from video using MediaPipe."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single frame.
        
        Args:
            frame: (H, W, 3) BGR image
        
        Returns:
            landmarks: (84,) vector [left_hand_21*2, right_hand_21*2]
                      or None if hands not detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks is None:
            return None
        
        landmarks_list = []
        handedness_list = []
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmarks_list.append(hand_landmarks)
            handedness_list.append(handedness.classification[0].label)
        
        # Standardize to [left_hand, right_hand]
        left_hand = None
        right_hand = None
        
        for landmarks, handedness in zip(landmarks_list, handedness_list):
            if handedness == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks
        
        # Convert to 84-dim vector
        landmarks_vector = self._landmarks_to_vector(left_hand, right_hand)
        
        return landmarks_vector
    
    def _landmarks_to_vector(
        self,
        left_hand: Optional,
        right_hand: Optional,
    ) -> np.ndarray:
        """Convert hand landmarks to 84-dim vector."""
        vector = np.zeros(FEATURE_SIZE, dtype=np.float32)
        
        # Left hand (indices 0-41): 21 landmarks * 2 coords
        if left_hand is not None:
            for i, landmark in enumerate(left_hand.landmark):
                vector[i*2] = landmark.x
                vector[i*2 + 1] = landmark.y
        
        # Right hand (indices 42-83)
        if right_hand is not None:
            for i, landmark in enumerate(right_hand.landmark):
                vector[42 + i*2] = landmark.x
                vector[42 + i*2 + 1] = landmark.y
        
        return vector
    
    def extract_landmarks_from_video(
        self,
        video_path: Path,
        every_n_frames: int = 1,
    ) -> Tuple[np.ndarray, int]:
        """
        Extract landmarks from video file.
        
        Args:
            video_path: path to video file
            every_n_frames: extract every nth frame (for speed)
        
        Returns:
            landmarks: (T, 84) array
            fps: frames per second of video
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        landmarks_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract every nth frame
            if frame_idx % every_n_frames == 0:
                landmarks = self.extract_landmarks_from_frame(frame)
                
                if landmarks is not None:
                    landmarks_list.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        if not landmarks_list:
            raise ValueError("No hands detected in video")
        
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        
        return landmarks_array, fps
    
    def extract_landmarks_from_webcam(
        self,
        duration_seconds: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract landmarks from webcam stream.
        
        Args:
            duration_seconds: duration to record
            max_frames: maximum frames to extract
        
        Returns:
            landmarks: (T, 84) array
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames_count = int(duration_seconds * fps) if max_frames is None else max_frames
        
        landmarks_list = []
        frame_count = 0
        
        while frame_count < max_frames_count:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            landmarks = self.extract_landmarks_from_frame(frame)
            
            if landmarks is not None:
                landmarks_list.append(landmarks)
            
            # Display frame with hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks (simplified)
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            cv2.imshow("Recording fingerspelling... Press ESC to stop", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not landmarks_list:
            raise ValueError("No hands detected in webcam stream")
        
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        
        return landmarks_array


def preprocess_landmarks(
    landmarks: np.ndarray,
    normalize: bool = NORMALIZE_LANDMARKS,
) -> np.ndarray:
    """
    Preprocess landmark sequence to match training format.
    
    Args:
        landmarks: (T, 84) raw MediaPipe landmarks
        normalize: whether to normalize to [-1, 1] range
    
    Returns:
        preprocessed: (T, 84) padded/truncated and optionally normalized
    """
    # Clip to valid range
    if normalize:
        landmarks = np.clip(landmarks, 0.0, 1.0)
        landmarks = landmarks * 2 - 1  # scale to [-1, 1]
    
    # Pad or truncate
    T = landmarks.shape[0]
    if T < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - T, FEATURE_SIZE), dtype=np.float32)
        landmarks = np.vstack([landmarks, padding])
    elif T > MAX_FRAMES:
        landmarks = landmarks[:MAX_FRAMES]
    
    return landmarks


def video_to_landmarks(
    video_source: Path,
    extract_every_n_frames: int = 1,
) -> Tuple[np.ndarray, str]:
    """
    End-to-end pipeline: video -> MediaPipe -> landmarks -> preprocessed.
    
    Args:
        video_source: path to video file
        extract_every_n_frames: frame sampling rate
    
    Returns:
        landmarks: (MAX_FRAMES, 84) preprocessed array
        video_name: name of video file
    """
    extractor = MediaPipeExtractor()
    
    # Extract from video
    raw_landmarks, fps = extractor.extract_landmarks_from_video(
        video_source,
        every_n_frames=extract_every_n_frames,
    )
    
    # Preprocess
    landmarks = preprocess_landmarks(raw_landmarks)
    
    video_name = Path(video_source).stem
    
    return landmarks, video_name
