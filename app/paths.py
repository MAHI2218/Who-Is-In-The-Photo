"""
Runtime path resolution helpers for both source execution and PyInstaller bundles.
"""

from pathlib import Path
import sys


def get_model_path(model_filename: str) -> str:
    """Resolve an ONNX model path across source and frozen PyInstaller runs."""
    candidates = []

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "models" / model_filename)
        candidates.append(Path(sys.executable).resolve().parent / "models" / model_filename)

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "models" / model_filename)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])
