from __future__ import annotations

import argparse
from pathlib import Path

from .environment import build_environment_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASL alphabet recognition project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect webcam images for one label.")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--samples", type=int, default=200)
    collect_parser.add_argument("--camera-index", type=int, default=0)
    collect_parser.add_argument("--countdown", type=int, default=15)

    build_parser = subparsers.add_parser(
        "build-dataset", help="Extract image features from image folders."
    )
    build_parser.add_argument("--data-dir", type=Path, default=None)

    demo_parser = subparsers.add_parser(
        "generate-demo-data", help="Generate a synthetic demo dataset for pipeline testing."
    )
    demo_parser.add_argument("--samples-per-class", type=int, default=20)
    demo_parser.add_argument("--output-dir", type=Path, default=None)

    demo_run_parser = subparsers.add_parser(
        "demo-run", help="Generate demo data, build the dataset, and train the model."
    )
    demo_run_parser.add_argument("--samples-per-class", type=int, default=20)
    demo_run_parser.add_argument("--output-dir", type=Path, default=None)

    train_parser = subparsers.add_parser("train", help="Train the classifier.")
    train_parser.add_argument("--dataset-file", type=Path, default=None)

    predict_parser = subparsers.add_parser("predict", help="Run live webcam prediction.")
    predict_parser.add_argument("--camera-index", type=int, default=0)
    predict_parser.add_argument("--model-path", type=Path, default=None)

    subparsers.add_parser("doctor", help="Check Python/dependency readiness.")
    subparsers.add_parser("gui", help="Launch the desktop GUI.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        from .dataset import collect_samples

        collect_samples(
            label=args.label,
            samples=args.samples,
            camera_index=args.camera_index,
            countdown=args.countdown,
        )
    elif args.command == "build-dataset":
        from .dataset import build_feature_dataset

        summary = build_feature_dataset(args.data_dir)
        print(summary)
    elif args.command == "generate-demo-data":
        from .dataset import generate_demo_dataset

        summary = generate_demo_dataset(
            samples_per_class=args.samples_per_class,
            output_dir=args.output_dir,
        )
        print(summary)
    elif args.command == "demo-run":
        from .dataset import build_feature_dataset, generate_demo_dataset
        from .training import train_model
        from .config import ProjectPaths

        paths = ProjectPaths()
        paths.ensure()
        demo_output_dir = args.output_dir or (paths.data_dir / "demo_raw")
        demo_summary = generate_demo_dataset(
            samples_per_class=args.samples_per_class,
            output_dir=demo_output_dir,
        )
        dataset_summary = build_feature_dataset(demo_output_dir)
        metrics = train_model()
        print(
            {
                "demo_dataset": demo_summary,
                "built_dataset": dataset_summary,
                "training": metrics,
            }
        )
    elif args.command == "train":
        from .training import train_model

        metrics = train_model(args.dataset_file)
        print(metrics)
    elif args.command == "predict":
        from .predictor import run_live_prediction

        run_live_prediction(camera_index=args.camera_index, model_path=args.model_path)
    elif args.command == "doctor":
        print(build_environment_report())
    elif args.command == "gui":
        from .gui import launch_gui

        launch_gui()
