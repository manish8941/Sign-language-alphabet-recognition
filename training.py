from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectPaths, TrainingConfig


def load_processed_dataset(dataset_file: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    paths = ProjectPaths()
    paths.ensure()
    npz_path = dataset_file or (paths.processed_dir / "asl_image_dataset.npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {npz_path}. Run the dataset builder first."
        )
    data = np.load(npz_path, allow_pickle=False)
    return data["X"], data["y"]


def train_model(dataset_file: Path | None = None) -> dict[str, object]:
    paths = ProjectPaths()
    paths.ensure()
    config = TrainingConfig()
    X, y = load_processed_dataset(dataset_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_split=config.min_samples_split,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    labels = sorted(np.unique(y))
    model_bundle = {
        "model": model,
        "feature_mode": "image",
        "image_size": 64,
        "labels": labels,
    }
    joblib.dump(model_bundle, paths.artifacts_dir / "asl_classifier.joblib")
    figure, axis = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=labels,
        xticks_rotation=45,
        ax=axis,
        colorbar=False,
    )
    axis.set_title("ASL Alphabet Recognition Confusion Matrix")
    figure.tight_layout()
    figure.savefig(paths.artifacts_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)

    metrics = {
        "accuracy": accuracy,
        "labels": labels,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "classification_report": report,
        "model_file": str(paths.artifacts_dir / "asl_classifier.joblib"),
        "confusion_matrix_file": str(paths.artifacts_dir / "confusion_matrix.png"),
    }
    with (paths.artifacts_dir / "training_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train ASL alphabet classifier.")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional processed dataset .npz file.",
    )
    args = parser.parse_args()
    metrics = train_model(args.dataset_file)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_cli()
