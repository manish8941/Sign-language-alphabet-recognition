from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .config import InferenceConfig
from .features import build_feature_vector


@dataclass(slots=True)
class DetectionResult:
    landmarks: np.ndarray | None
    feature_vector: np.ndarray | None
    handedness: str | None


class HandTracker:
    def __init__(
        self,
        config: InferenceConfig | None = None,
        static_image_mode: bool = False,
    ) -> None:
        self.config = config or InferenceConfig()
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return DetectionResult(None, None, None)

        hand_landmarks = results.multi_hand_landmarks[0]
        points = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        feature_vector = build_feature_vector(points)

        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return DetectionResult(points, feature_vector, handedness)

    def annotate(
        self,
        frame_bgr: np.ndarray,
        results: Any,
        label: str | None = None,
        confidence: float | None = None,
    ) -> np.ndarray:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

        if label is not None:
            text = label if confidence is None else f"{label} ({confidence:.2f})"
            cv2.putText(
                frame_bgr,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame_bgr

    def process_and_annotate(self, frame_bgr: np.ndarray) -> tuple[DetectionResult, np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        detection = DetectionResult(None, None, None)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            points = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            feature_vector = build_feature_vector(points)
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            detection = DetectionResult(points, feature_vector, handedness)

        annotated = self.annotate(frame_bgr.copy(), results)
        return detection, annotated
