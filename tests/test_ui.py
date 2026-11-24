"""
Unit and integration tests for PySide6 MainWindow and UI workflow.
"""

import os
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from app.employee_loader import EmployeeRecord, LoaderReport
from app.face_recognizer import MatchResult
from app.ui_main_window import MainWindow, ScaledImageViewer


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_initialization(qapp):
    window = MainWindow()

    # Verify window title
    assert "Who Is in the Photo?" in window.windowTitle()

    # Verify required buttons exist
    assert window.btn_select_folder is not None
    assert window.btn_upload_group is not None
    assert window.btn_identify is not None
    assert window.btn_save_image is not None

    # Verify Identify and Save buttons are initially disabled
    assert not window.btn_identify.isEnabled()
    assert not window.btn_save_image.isEnabled()

    # Verify labels and image viewer
    assert window.lbl_employee_status is not None
    assert window.lbl_group_status is not None
    assert window.image_viewer is not None
    assert isinstance(window.image_viewer, ScaledImageViewer)

    # Verify default threshold
    assert window.slider_threshold.value() == 36  # ~0.36
    assert window.lbl_threshold_val.text() == "0.36"

    window.close()


def test_ui_complete_workflow_integration(qapp, monkeypatch):
    """
    Tests the complete UI interaction pipeline:
    1. Employee folder loaded -> status updated
    2. Group photo loaded -> display updated
    3. Identify People executed -> results list & stat badges populated
    4. Save button enabled
    """
    window = MainWindow()

    # Avoid blocking modal message boxes in headless testing
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *args, **kwargs: None)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *args, **kwargs: None)

    # 1. Simulate employee database loaded with 2 profiles
    mock_records = [
        EmployeeRecord(
            name="Priya Sharma",
            file_path="dummy/priya.jpg",
            embedding=np.zeros(128, dtype=np.float32),
            bounding_box=(10, 10, 50, 50),
            landmarks=np.zeros((5, 2)),
        ),
        EmployeeRecord(
            name="Rahul Verma",
            file_path="dummy/rahul.jpg",
            embedding=np.zeros(128, dtype=np.float32),
            bounding_box=(10, 10, 50, 50),
            landmarks=np.zeros((5, 2)),
        ),
    ]
    report = LoaderReport(total_images_found=2, valid_records=mock_records)
    window._on_load_finished(report)

    assert len(window.reference_records) == 2
    assert "2 employee profiles indexed" in window.lbl_employee_status.text()
    # Group photo not yet loaded, identify should still be disabled
    assert not window.btn_identify.isEnabled()

    # 2. Simulate group photo upload
    dummy_group = np.zeros((600, 800, 3), dtype=np.uint8)
    window.current_group_bgr = dummy_group
    window.lbl_group_status.setText("✅ group_photo.jpg (800×600)")
    window._update_identify_button_state()

    # Both employee database and group photo are loaded -> identify should be enabled!
    assert window.btn_identify.isEnabled()

    # 3. Simulate recognition completed with 2 identified faces and 1 unknown face
    mock_results = [
        MatchResult(
            face_index=0,
            bounding_box=(50, 50, 80, 80),
            landmarks=np.zeros((5, 2)),
            identity="Priya Sharma",
            similarity_score=0.88,
            is_known=True,
            detection_score=0.99,
        ),
        MatchResult(
            face_index=1,
            bounding_box=(150, 50, 80, 80),
            landmarks=np.zeros((5, 2)),
            identity="Rahul Verma",
            similarity_score=0.79,
            is_known=True,
            detection_score=0.96,
        ),
        MatchResult(
            face_index=2,
            bounding_box=(250, 50, 80, 80),
            landmarks=np.zeros((5, 2)),
            identity="Unknown",
            similarity_score=0.15,
            is_known=False,
            detection_score=0.92,
        ),
    ]

    annotated_dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    window._on_recognition_finished(annotated_dummy, mock_results)

    # Verify stat counters
    assert window.stat_total.val_lbl.text() == "3"
    assert window.stat_known.val_lbl.text() == "2"
    assert window.stat_unknown.val_lbl.text() == "1"

    # Verify save button is enabled
    assert window.btn_save_image.isEnabled()

    # Verify results cards created in results layout (3 cards + 1 stretch)
    assert window.results_layout.count() >= 4

    window.close()


def test_ui_error_handling_empty_database(qapp, monkeypatch):
    """Tests error notification when selected folder has no images."""
    window = MainWindow()

    warn_called = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda parent, title, msg: warn_called.append((title, msg))
    )

    empty_report = LoaderReport(total_images_found=0, valid_records=[])
    window._on_load_finished(empty_report)

    assert len(warn_called) == 1
    assert "No Photos Found" in warn_called[0][0]
    assert not window.btn_identify.isEnabled()

    window.close()
