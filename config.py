from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STATIC_ASL_LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


@dataclass(slots=True)
class ProjectPaths:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.artifacts_dir = self.root_dir / "artifacts"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 350
    max_depth: int | None = 24
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass(slots=True)
class InferenceConfig:
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    max_num_hands: int = 1
    smoothing_window: int = 6
    confidence_threshold: float = 0.55
    roi_size: int = 320
    image_size: int = 64
