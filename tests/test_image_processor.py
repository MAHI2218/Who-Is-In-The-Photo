"""
Unit and integration tests for GroupImageProcessor pipeline.
"""

import numpy as np
import pytest
from app.employee_loader import EmployeeRecord
from app.face_recognizer import MatchResult
from app.image_processor import GroupImageProcessor, COLOR_KNOWN, COLOR_UNKNOWN


def test_no_faces_detected_handling():
    processor = GroupImageProcessor()
    # Blank canvas with no faces
    blank_img = np.zeros((400, 400, 3), dtype=np.uint8)
    references = []

    annotated, results = processor.process_group_photo(blank_img, references)

    assert annotated is not None
    assert annotated.shape == blank_img.shape
    assert len(results) == 0


def test_corrupt_group_photo_input():
    processor = GroupImageProcessor()
    annotated, results = processor.process_group_photo("non_existent_file.jpg", [])
    assert annotated is None
    assert len(results) == 0


def test_draw_annotations_styling():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    dummy_results = [
        MatchResult(
            face_index=0,
            bounding_box=(50, 50, 100, 100),
            landmarks=np.array([[70, 70], [130, 70], [100, 100], [80, 130], [120, 130]]),
            identity="Priya Sharma",
            similarity_score=0.885,
            is_known=True,
        ),
        MatchResult(
            face_index=1,
            bounding_box=(250, 50, 100, 100),
            landmarks=np.array([[270, 70], [330, 70], [300, 100], [280, 130], [320, 130]]),
            identity="Unknown",
            similarity_score=0.150,
            is_known=False,
        ),
    ]

    annotated = GroupImageProcessor.draw_annotations(img, dummy_results, draw_landmarks=True)

    # Annotated image should be same shape and have modified pixels
    assert annotated.shape == (500, 500, 3)
    assert not np.array_equal(annotated, img)


def test_end_to_end_group_processing_with_mocks(monkeypatch):
    processor = GroupImageProcessor(recognition_threshold=0.40)

    # Create 2 reference employees: "Priya Sharma" and "Rahul Verma"
    priya_emb = np.zeros(128, dtype=np.float32)
    priya_emb[0] = 1.0

    rahul_emb = np.zeros(128, dtype=np.float32)
    rahul_emb[1] = 1.0

    ref_priya = EmployeeRecord(
        name="Priya Sharma",
        file_path="dummy/priya.jpg",
        embedding=priya_emb,
        bounding_box=(10, 10, 50, 50),
        landmarks=np.zeros((5, 2)),
    )
    ref_rahul = EmployeeRecord(
        name="Rahul Verma",
        file_path="dummy/rahul.jpg",
        embedding=rahul_emb,
        bounding_box=(10, 10, 50, 50),
        landmarks=np.zeros((5, 2)),
    )

    references = [ref_priya, ref_rahul]

    # Mock group photo with 3 faces:
    # Face 0 -> Matches Priya
    # Face 1 -> Matches Rahul
    # Face 2 -> Unknown person
    mock_detected_faces = np.array([
        [50, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.98],
        [200, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
        [350, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92],
    ], dtype=np.float32)

    # Mock detector to return 3 faces
    class MockDetector:
        def setInputSize(self, size):
            pass
        def detect(self, img):
            return 3, mock_detected_faces

    processor.detector = MockDetector()

    # Face embeddings corresponding to the 3 faces
    face0_emb = np.copy(priya_emb)
    face1_emb = np.copy(rahul_emb)
    face2_emb = np.zeros(128, dtype=np.float32)
    face2_emb[10] = 1.0  # Unrelated orthogonal vector

    emb_iter = iter([face0_emb, face1_emb, face2_emb])
    monkeypatch.setattr(processor.recognizer, "extract_embedding", lambda img, info: next(emb_iter))

    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    annotated, results = processor.process_group_photo(test_img, references)

    assert annotated is not None
    assert len(results) == 3

    # Verify each face processed separately and correctly matched
    assert results[0].identity == "Priya Sharma"
    assert results[0].is_known is True

    assert results[1].identity == "Rahul Verma"
    assert results[1].is_known is True

    assert results[2].identity == "Unknown"
    assert results[2].is_known is False
