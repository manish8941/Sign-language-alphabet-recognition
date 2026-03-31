from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import ProjectPaths, STATIC_ASL_LABELS
from .environment import build_environment_report, inspect_environment


class ASLProjectGUI:
    def __init__(self) -> None:
        self.paths = ProjectPaths()
        self.paths.ensure()
        self.root = tk.Tk()
        self.root.title("ASL Alphabet Recognition")
        self.root.geometry("980x720")
        self.root.minsize(900, 640)

        self.dataset_dir_var = tk.StringVar(value=str(self.paths.raw_data_dir))
        self.label_var = tk.StringVar(value=STATIC_ASL_LABELS[0])
        self.samples_var = tk.StringVar(value="200")
        self.camera_var = tk.StringVar(value="0")
        self.model_var = tk.StringVar(value=str(self.paths.artifacts_dir / "asl_classifier.joblib"))

        self._build_layout()
        self.refresh_environment()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        title = ttk.Label(
            container,
            text="Sign Language Alphabet Recognition",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text="Collect data, train the model, and run live prediction from one desktop app.",
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        control_tab = ttk.Frame(notebook, padding=12)
        status_tab = ttk.Frame(notebook, padding=12)
        guide_tab = ttk.Frame(notebook, padding=12)
        notebook.add(control_tab, text="Controls")
        notebook.add(status_tab, text="Environment")
        notebook.add(guide_tab, text="Guide")

        self._build_controls(control_tab)
        self._build_status(status_tab)
        self._build_guide(guide_tab)

        log_frame = ttk.LabelFrame(container, text="Logs", padding=12)
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.log_text = tk.Text(log_frame, wrap="word", height=14, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

    def _build_controls(self, parent: ttk.Frame) -> None:
        dataset_frame = ttk.LabelFrame(parent, text="Dataset", padding=12)
        dataset_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(dataset_frame, text="Raw dataset folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(dataset_frame, textvariable=self.dataset_dir_var, width=70).grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=(6, 0)
        )
        ttk.Button(dataset_frame, text="Browse", command=self._browse_dataset_dir).grid(
            row=1, column=1, sticky="e", pady=(6, 0)
        )
        ttk.Button(
            dataset_frame,
            text="Build Image Dataset",
            command=lambda: self._run_cli_command(["build-dataset", "--data-dir", self.dataset_dir_var.get()]),
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Button(
            dataset_frame,
            text="Generate Demo Data",
            command=lambda: self._run_cli_command(["generate-demo-data", "--output-dir", self.dataset_dir_var.get()]),
        ).grid(row=2, column=1, sticky="e", pady=(12, 0))
        dataset_frame.columnconfigure(0, weight=1)

        collect_frame = ttk.LabelFrame(parent, text="Sample Collection", padding=12)
        collect_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(collect_frame, text="Label").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            collect_frame,
            textvariable=self.label_var,
            values=STATIC_ASL_LABELS,
            width=8,
            state="readonly",
        ).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(6, 0))
        ttk.Label(collect_frame, text="Samples").grid(row=0, column=1, sticky="w")
        ttk.Entry(collect_frame, textvariable=self.samples_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(0, 12), pady=(6, 0)
        )
        ttk.Label(collect_frame, text="Camera Index").grid(row=0, column=2, sticky="w")
        ttk.Entry(collect_frame, textvariable=self.camera_var, width=12).grid(
            row=1, column=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            collect_frame,
            text="Collect Samples",
            command=self._collect_samples,
        ).grid(row=1, column=3, sticky="w", padx=(16, 0), pady=(6, 0))

        model_frame = ttk.LabelFrame(parent, text="Training and Prediction", padding=12)
        model_frame.pack(fill="x")
        ttk.Label(model_frame, text="Model path").grid(row=0, column=0, sticky="w")
        ttk.Entry(model_frame, textvariable=self.model_var, width=70).grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=(6, 12)
        )
        ttk.Button(model_frame, text="Browse", command=self._browse_model_file).grid(
            row=1, column=1, sticky="e", pady=(6, 12)
        )
        ttk.Button(model_frame, text="Train Model", command=lambda: self._run_cli_command(["train"])).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Button(
            model_frame,
            text="Run Live Prediction",
            command=self._predict_live,
        ).grid(row=2, column=1, sticky="e")
        model_frame.columnconfigure(0, weight=1)

    def _build_status(self, parent: ttk.Frame) -> None:
        button_row = ttk.Frame(parent)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Refresh Check", command=self.refresh_environment).pack(anchor="w")

        self.status_text = tk.Text(parent, wrap="word", font=("Consolas", 10))
        self.status_text.pack(fill="both", expand=True, pady=(12, 0))

    def _build_guide(self, parent: ttk.Frame) -> None:
        guide = """Recommended workflow

1. Use your normal Python environment.
2. Install dependencies from requirements.txt.
3. Either generate demo data or collect samples for each ASL letter you want to support.
4. Build the image-feature dataset.
5. Train the classifier.
6. Run live prediction from the webcam.

Notes

- This project now uses direct image classification instead of MediaPipe in the main flow.
- The demo dataset is only for verifying that the pipeline runs end to end.
- Keep your hand centered inside the green guide box during collection and prediction.
- J and Z are omitted by default because they are dynamic gestures."""
        guide_label = ttk.Label(parent, text=guide, justify="left")
        guide_label.pack(anchor="w")

    def _browse_dataset_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.dataset_dir_var.get() or str(self.paths.root_dir))
        if selected:
            self.dataset_dir_var.set(selected)

    def _browse_model_file(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(self.paths.artifacts_dir),
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if selected:
            self.model_var.set(selected)

    def _collect_samples(self) -> None:
        self._run_cli_command(
            [
                "collect",
                "--label",
                self.label_var.get(),
                "--samples",
                self.samples_var.get(),
                "--camera-index",
                self.camera_var.get(),
            ]
        )

    def _predict_live(self) -> None:
        command = [
            "predict",
            "--camera-index",
            self.camera_var.get(),
        ]
        model_path = self.model_var.get().strip()
        if model_path:
            command.extend(["--model-path", model_path])
        self._run_cli_command(command)

    def refresh_environment(self) -> None:
        report = build_environment_report()
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, report)

    def _run_cli_command(self, args: list[str]) -> None:
        status = inspect_environment()
        if not status.ready_for_full_runtime and any(
            item in args for item in ("collect", "build-dataset", "train", "predict")
        ):
            messagebox.showwarning(
                "Missing Dependencies",
                "The environment checker reports missing core modules. Install the required packages before running the full workflow.",
            )

        command = [sys.executable, str(self.paths.root_dir / "app.py"), *args]
        self._append_log(f"\n$ {' '.join(command)}\n")

        def runner() -> None:
            process = subprocess.run(
                command,
                cwd=self.paths.root_dir,
                capture_output=True,
                text=True,
            )
            output = process.stdout.strip()
            errors = process.stderr.strip()
            if output:
                self._append_log(output + "\n")
            if errors:
                self._append_log(errors + "\n")
            self._append_log(f"[exit code: {process.returncode}]\n")

        threading.Thread(target=runner, daemon=True).start()

    def _append_log(self, text: str) -> None:
        self.log_text.after(0, self._append_log_on_ui, text)

    def _append_log_on_ui(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui() -> None:
    app = ASLProjectGUI()
    app.run()
