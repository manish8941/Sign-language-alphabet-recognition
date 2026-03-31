from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize_landmarks(landmarks: Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray(list(landmarks), dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError("Expected 21 hand landmarks with x, y, z coordinates.")

    wrist = points[0]
    centered = points - wrist
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
    if scale <= 1e-6:
        scale = 1.0

    normalized = centered / scale
    return normalized.flatten()


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-6:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def build_feature_vector(landmarks: Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray(list(landmarks), dtype=np.float32)
    normalized = normalize_landmarks(points)

    finger_triplets = (
        (0, 1, 2),
        (1, 2, 4),
        (0, 5, 6),
        (5, 6, 8),
        (0, 9, 10),
        (9, 10, 12),
        (0, 13, 14),
        (13, 14, 16),
        (0, 17, 18),
        (17, 18, 20),
    )
    angles = [
        calculate_angle(points[a], points[b], points[c]) / 180.0
        for a, b, c in finger_triplets
    ]

    fingertip_indices = (4, 8, 12, 16, 20)
    wrist = points[0]
    fingertip_distances = []
    for index in fingertip_indices:
        fingertip_distances.append(float(np.linalg.norm(points[index] - wrist)))
    fingertip_distances = np.asarray(fingertip_distances, dtype=np.float32)
    if np.max(fingertip_distances) > 1e-6:
        fingertip_distances = fingertip_distances / np.max(fingertip_distances)

    return np.concatenate(
        [
            normalized.astype(np.float32),
            np.asarray(angles, dtype=np.float32),
            fingertip_distances.astype(np.float32),
        ]
    )

