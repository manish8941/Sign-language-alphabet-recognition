from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import InferenceConfig, ProjectPaths, STATIC_ASL_LABELS
from .image_features import crop_from_roi, extract_image_feature


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _draw_demo_pattern(label: str, sample_index: int, image_size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed=(ord(label) * 1000 + sample_index))
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

    base_color = (
        int(40 + (ord(label) * 3) % 180),
        int(50 + (ord(label) * 5) % 160),
        int(60 + (ord(label) * 7) % 140),
    )

    center = (
        int(image_size * 0.5 + rng.integers(-18, 19)),
        int(image_size * 0.5 + rng.integers(-18, 19)),
    )
    radius = int(image_size * 0.26 + rng.integers(-10, 11))
    cv2.circle(canvas, center, radius, base_color, thickness=3)

    line_count = 3 + (ord(label) % 4)
    for index in range(line_count):
        angle = (index * 180.0 / max(line_count, 1)) + (ord(label) % 25) + rng.integers(-8, 9)
        length = int(image_size * (0.2 + 0.05 * index))
        x2 = int(center[0] + length * np.cos(np.radians(angle)))
        y2 = int(center[1] + length * np.sin(np.radians(angle)))
        cv2.line(canvas, center, (x2, y2), base_color, thickness=4)

    rect_w = int(image_size * (0.18 + ((ord(label) + sample_index) % 6) * 0.02))
    rect_h = int(image_size * (0.12 + ((ord(label) + sample_index) % 5) * 0.025))
    rect_x = int(image_size * 0.15 + rng.integers(-10, 11))
    rect_y = int(image_size * 0.68 + rng.integers(-12, 13))
    cv2.rectangle(
        canvas,
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        base_color,
        thickness=3,
    )

    font_scale = 2.5 + float(rng.uniform(-0.2, 0.2))
    thickness = 5
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max((image_size - text_size[0]) // 2 + int(rng.integers(-12, 13)), 10)
    text_y = max((image_size + text_size[1]) // 2 + int(rng.integers(-12, 13)), text_size[1] + 10)
    cv2.putText(
        canvas,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    noise = rng.normal(0, 10, canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def generate_demo_dataset(samples_per_class: int = 20, output_dir: Path | None = None) -> dict[str, object]:
    paths = ProjectPaths()
    paths.ensure()
    target_root = output_dir or paths.raw_data_dir
    target_root.mkdir(parents=True, exist_ok=True)

    created = 0
    for label in STATIC_ASL_LABELS:
        label_dir = target_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples_per_class):
            image = _draw_demo_pattern(label, sample_index)
            output_path = label_dir / f"demo_{label}_{sample_index:03d}.jpg"
            cv2.imwrite(str(output_path), image)
            created += 1

    summary = {
        "type": "demo_dataset",
        "output_dir": str(target_root),
        "samples_per_class": samples_per_class,
        "classes": STATIC_ASL_LABELS,
        "total_images": created,
        "note": "This synthetic dataset is only for pipeline verification. It is not suitable for real ASL recognition.",
    }
    return summary


def collect_samples(
    label: str,
    samples: int,
    camera_index: int = 0,
    countdown: int = 60,
) -> None:
    label = label.upper()
    if label not in STATIC_ASL_LABELS:
        raise ValueError(
            f"Label '{label}' is unsupported. Use one of: {', '.join(STATIC_ASL_LABELS)}"
        )

    paths = ProjectPaths()
    paths.ensure()
    target_dir = paths.raw_data_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)

    config = InferenceConfig()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    frames_until_next_shot = countdown
    collected = 0
    try:
        while collected < samples:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from webcam.")

            frame = cv2.flip(frame, 1)
            roi, (x1, y1, x2, y2) = crop_from_roi(frame, config.roi_size)
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            status_text = f"Label: {label}  Saved: {collected}/{samples}"
            cv2.putText(
                annotated,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frames_until_next_shot -= 1
            cv2.putText(
                annotated,
                f"Keep your hand inside the box. Capturing in: {max(frames_until_next_shot, 0)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if frames_until_next_shot <= 0:
                output_path = target_dir / f"{label}_{collected:04d}.jpg"
                cv2.imwrite(str(output_path), roi)
                collected += 1
                frames_until_next_shot = countdown

            cv2.imshow("ASL Sample Collector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


def _collect_images_in_dir(label_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for image_path in sorted(label_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((label_dir.name, image_path))
    return items


def _iter_image_paths(data_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    direct_label_dirs = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_dir() and path.name.upper() in STATIC_ASL_LABELS
    ]
    if direct_label_dirs:
        for label_dir in direct_label_dirs:
            items.extend(_collect_images_in_dir(label_dir))
        return items

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_dir.name.upper() not in STATIC_ASL_LABELS:
                continue
            items.extend(_collect_images_in_dir(label_dir))
    return items


def build_feature_dataset(data_dir: Path | None = None) -> dict[str, object]:
    paths = ProjectPaths()
    paths.ensure()
    source_dir = data_dir or paths.raw_data_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_dir}")

    image_items = _iter_image_paths(source_dir)
    if not image_items:
        raise RuntimeError(
            f"No images found in {source_dir}. Add folders like data/raw/A, data/raw/B, ..."
        )

    config = InferenceConfig()
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    skipped_labels: list[str] = []

    for label, image_path in image_items:
        normalized_label = label.upper()
        if normalized_label not in STATIC_ASL_LABELS:
            skipped_labels.append(normalized_label)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append(str(image_path))
            continue

        try:
            feature_vector = extract_image_feature(image, image_size=config.image_size)
        except Exception:
            skipped.append(str(image_path))
            continue

        features.append(feature_vector)
        labels.append(normalized_label)

    if not features:
        raise RuntimeError("No valid image features were extracted from the dataset.")

    feature_matrix = np.vstack(features).astype(np.float32)
    label_array = np.asarray(labels)

    output_npz = paths.processed_dir / "asl_image_dataset.npz"
    np.savez_compressed(output_npz, X=feature_matrix, y=label_array)

    summary = {
        "samples": int(len(labels)),
        "classes": sorted(set(labels)),
        "class_distribution": dict(Counter(labels)),
        "skipped_files": skipped,
        "skipped_labels": sorted(set(skipped_labels)),
        "output_file": str(output_npz),
        "feature_count": int(feature_matrix.shape[1]),
        "feature_mode": "image",
    }

    with (paths.processed_dir / "dataset_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def build_dataset_cli() -> None:
    parser = argparse.ArgumentParser(description="Build ASL image-feature dataset from images.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing class folders with images.",
    )
    args = parser.parse_args()
    summary = build_feature_dataset(args.data_dir)
    print(json.dumps(summary, indent=2))


def collect_cli() -> None:
    parser = argparse.ArgumentParser(description="Collect webcam samples for one ASL label.")
    parser.add_argument("--label", required=True, help="ASL letter to collect.")
    parser.add_argument("--samples", type=int, default=200, help="Number of images to save.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--countdown",
        type=int,
        default=15,
        help="Frames to wait between captures.",
    )
    args = parser.parse_args()
    collect_samples(
        label=args.label,
        samples=args.samples,
        camera_index=args.camera_index,
        countdown=args.countdown,
    )


def generate_demo_dataset_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset for pipeline testing.")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = generate_demo_dataset(
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    build_dataset_cli()
