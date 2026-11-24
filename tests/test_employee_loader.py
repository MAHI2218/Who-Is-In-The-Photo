"""
Unit tests for EmployeeLoader and employee reference ingestion.
"""

import os
import tempfile
import numpy as np
import cv2
import pytest
from app.employee_loader import EmployeeLoader, EmployeeRecord, LoaderReport


def test_extract_name_from_filename():
    assert EmployeeLoader.extract_name_from_filename("Priya Sharma.jpg") == "Priya Sharma"
    assert EmployeeLoader.extract_name_from_filename("John Doe.PNG") == "John Doe"
    assert EmployeeLoader.extract_name_from_filename("Rahul_Verma.jpeg") == "Rahul Verma"
    assert EmployeeLoader.extract_name_from_filename("Alice.jpg") == "Alice"


def test_corrupt_file_handling():
    loader = EmployeeLoader()
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupt_file = os.path.join(tmp_dir, "Invalid Person.jpg")
        with open(corrupt_file, "w") as f:
            f.write("This is not an image file!")

        record, err, face_count = loader.process_single_image(corrupt_file)
        assert record is None
        assert "corrupt" in err.lower() or "not a valid image" in err.lower()
        assert face_count == -1


def test_empty_canvas_no_face_handling():
    loader = EmployeeLoader()
    with tempfile.TemporaryDirectory() as tmp_dir:
        black_canvas_file = os.path.join(tmp_dir, "No Face Person.jpg")
        blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(black_canvas_file, blank_img)

        record, err, face_count = loader.process_single_image(black_canvas_file)
        assert record is None
        assert face_count == 0
        assert "no face" in err.lower()


def test_folder_batch_report_summary():
    loader = EmployeeLoader()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create one blank image (no face)
        cv2.imwrite(os.path.join(tmp_dir, "Blank.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
        # Create one corrupt file
        with open(os.path.join(tmp_dir, "Corrupt.png"), "wb") as f:
            f.write(b"bad bytes")
        # Create a non-image file (should be ignored by extension filter)
        with open(os.path.join(tmp_dir, "notes.txt"), "w") as f:
            f.write("ignore me")

        report = loader.load_from_folder(tmp_dir)
        assert report.total_images_found == 2
        assert len(report.skipped_no_face) == 1
        assert len(report.skipped_invalid_file) == 1
        assert report.valid_count == 0
        assert "Blank.jpg" in report.summary()
