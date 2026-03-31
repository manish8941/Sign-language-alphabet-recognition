from __future__ import annotations

import cv2
import numpy as np


def crop_from_roi(image: np.ndarray, roi_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    size = min(roi_size, height, width)
    x1 = max((width - size) // 2, 0)
    y1 = max((height - size) // 2, 0)
    x2 = x1 + size
    y2 = y1 + size
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_image(image: np.ndarray, image_size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)


def extract_image_feature(image: np.ndarray, image_size: int = 64) -> np.ndarray:
    processed = preprocess_image(image, image_size=image_size)
    normalized = processed.astype(np.float32) / 255.0

    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    bins = np.int32(angle / 45.0) % 8
    cell_size = max(image_size // 8, 1)
    hog_features: list[float] = []
    for y in range(0, image_size, cell_size):
        for x in range(0, image_size, cell_size):
            cell_bins = bins[y : y + cell_size, x : x + cell_size]
            cell_mag = magnitude[y : y + cell_size, x : x + cell_size]
            hist = np.bincount(
                cell_bins.ravel(),
                weights=cell_mag.ravel(),
                minlength=8,
            ).astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist /= norm
            hog_features.extend(hist.tolist())

    stats = np.asarray(
        [
            float(normalized.mean()),
            float(normalized.std()),
            float(normalized.min()),
            float(normalized.max()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([normalized.flatten(), np.asarray(hog_features, dtype=np.float32), stats])
