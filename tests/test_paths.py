"""
Unit tests for runtime model path resolution across source and PyInstaller bundles.
"""

import sys
from pathlib import Path
from app.paths import get_model_path


def test_get_model_path_source():
    path = Path(get_model_path("face_detection_yunet_2023mar.onnx"))
    assert path.name == "face_detection_yunet_2023mar.onnx"
    assert path.parent.name == "models"
    assert path.exists()


def test_get_model_path_frozen_meipass(monkeypatch, tmp_path):
    bundled_models = tmp_path / "models"
    bundled_models.mkdir()
    bundled_model = bundled_models / "face_recognition_sface_2021dec.onnx"
    bundled_model.write_bytes(b"onnx")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    resolved = get_model_path("face_recognition_sface_2021dec.onnx")
    assert Path(resolved) == bundled_model
