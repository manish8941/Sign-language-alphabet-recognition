from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


REQUIRED_MODULES = ("cv2", "numpy", "sklearn", "joblib", "matplotlib")
OPTIONAL_MODULES = ("mediapipe",)


@dataclass(slots=True)
class EnvironmentStatus:
    python_version: str
    available_modules: dict[str, bool]
    optional_modules: dict[str, bool]

    @property
    def missing_modules(self) -> list[str]:
        return [name for name, ok in self.available_modules.items() if not ok]

    @property
    def ready_for_full_runtime(self) -> bool:
        return all(self.available_modules.values())


def inspect_environment() -> EnvironmentStatus:
    available: dict[str, bool] = {}
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            available[name] = True
        except Exception:
            available[name] = False

    optional: dict[str, bool] = {}
    for name in OPTIONAL_MODULES:
        try:
            importlib.import_module(name)
            optional[name] = True
        except Exception:
            optional[name] = False

    version_info = sys.version_info
    return EnvironmentStatus(
        python_version=f"{version_info.major}.{version_info.minor}.{version_info.micro}",
        available_modules=available,
        optional_modules=optional,
    )


def build_environment_report() -> str:
    status = inspect_environment()
    lines = [
        f"Python version: {status.python_version}",
        "Core modules:",
    ]
    for name, ok in status.available_modules.items():
        lines.append(f"  - {name}: {'OK' if ok else 'missing'}")
    lines.append("Optional modules:")
    for name, ok in status.optional_modules.items():
        lines.append(f"  - {name}: {'OK' if ok else 'missing'}")

    if status.ready_for_full_runtime:
        lines.append("Environment status: ready for full project execution.")
    else:
        lines.append("Environment status: not ready for full project execution.")

    return "\n".join(lines)
