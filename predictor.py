from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import InferenceConfig, ProjectPaths
from .image_features import crop_from_roi, extract_image_feature


class LiveASLPredictor:
    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        paths = ProjectPaths()
        default_model = paths.artifacts_dir / "asl_classifier.joblib"
        self.model_path = model_path or default_model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train the model first."
            )

        self.config = InferenceConfig()
        if confidence_threshold is not None:
            self.config.confidence_threshold = confidence_threshold

        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            self.model = bundle["model"]
            self.image_size = int(bundle.get("image_size", self.config.image_size))
        else:
            self.model = bundle
            self.image_size = self.config.image_size

        self.prediction_history: deque[str] = deque(maxlen=self.config.smoothing_window)

    def close(self) -> None:
        return None

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, float]:
        roi, (x1, y1, x2, y2) = crop_from_roi(frame, self.config.roi_size)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = "Uncertain"
        confidence = 0.0

        try:
            feature_vector = extract_image_feature(roi, image_size=self.image_size)
            probabilities = self.model.predict_proba([feature_vector])[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            candidate_label = str(self.model.classes_[best_index])

            if confidence >= self.config.confidence_threshold:
                self.prediction_history.append(candidate_label)
                label = Counter(self.prediction_history).most_common(1)[0][0]
        except Exception:
            label = "Prediction error"

        status = f"{label} ({confidence:.2f})" if confidence > 0 else label
        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Keep your hand inside the box. Press Q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, label, confidence


def run_live_prediction(camera_index: int = 0, model_path: Path | None = None) -> None:
    predictor = LiveASLPredictor(model_path=model_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        predictor.close()
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from webcam.")

            frame = cv2.flip(frame, 1)
            annotated, _, _ = predictor.predict_frame(frame)
            cv2.imshow("ASL Alphabet Recognition", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.release()
        predictor.close()
        cv2.destroyAllWindows()
