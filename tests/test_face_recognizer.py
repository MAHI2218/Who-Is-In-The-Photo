"""
Unit tests for FaceRecognizer, cosine similarity, thresholding, and duplicate avoidance.
"""

import numpy as np
import pytest
from app.employee_loader import EmployeeRecord
from app.face_recognizer import FaceRecognizer, MatchResult, DEFAULT_SFACE_THRESHOLD


def create_dummy_record(name: str, embedding: np.ndarray) -> EmployeeRecord:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return EmployeeRecord(
        name=name,
        file_path=f"dummy/{name}.jpg",
        embedding=embedding,
        bounding_box=(10, 10, 100, 100),
        landmarks=np.zeros((5, 2)),
    )


def test_cosine_similarity_math():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert pytest.approx(FaceRecognizer.compute_cosine_similarity(v1, v2), 0.0001) == 1.0
    assert pytest.approx(FaceRecognizer.compute_cosine_similarity(v1, v3), 0.0001) == 0.0


def test_threshold_rejection_unknown():
    recognizer = FaceRecognizer(cosine_threshold=0.50)

    # Reference employee: Alice
    ref_emb = np.zeros(128, dtype=np.float32)
    ref_emb[0] = 1.0
    records = [create_dummy_record("Alice", ref_emb)]

    # Query 1: Exactly Alice (sim = 1.0 >= 0.50) -> Should match Alice
    name, score, is_known = recognizer.match_face_to_references(ref_emb, records)
    assert is_known is True
    assert name == "Alice"
    assert pytest.approx(score, 0.001) == 1.0

    # Query 2: Unrelated person (sim = 0.10 < 0.50) -> Should return "Unknown"
    query_unknown = np.zeros(128, dtype=np.float32)
    query_unknown[0] = 0.10
    query_unknown[1] = 0.99
    query_unknown = query_unknown / np.linalg.norm(query_unknown)

    name, score, is_known = recognizer.match_face_to_references(query_unknown, records)
    assert is_known is False
    assert name == "Unknown"
    assert score < 0.50


def test_configurable_threshold():
    recognizer = FaceRecognizer(cosine_threshold=0.60)
    assert recognizer.cosine_threshold == 0.60

    recognizer.set_threshold(0.35)
    assert recognizer.cosine_threshold == 0.35


def test_duplicate_avoidance_in_group_photo():
    """
    If two faces in the group photo both resemble 'Priya',
    the higher-similarity face must be assigned 'Priya', and
    the other face must NOT be assigned 'Priya'.
    """
    recognizer = FaceRecognizer(cosine_threshold=0.30)

    # Reference database with Priya and Bob
    priya_emb = np.zeros(128, dtype=np.float32)
    priya_emb[0] = 1.0
    bob_emb = np.zeros(128, dtype=np.float32)
    bob_emb[1] = 1.0

    references = [
        create_dummy_record("Priya Sharma", priya_emb),
        create_dummy_record("Bob Smith", bob_emb),
    ]

    # Two detected faces in group:
    # Face 0 is a 95% match for Priya
    face0_emb = np.copy(priya_emb)
    face0_emb[2] = 0.1
    face0_emb = face0_emb / np.linalg.norm(face0_emb)

    # Face 1 is also similar to Priya (60% match), but weaker
    face1_emb = np.copy(priya_emb)
    face1_emb[2] = 0.7
    face1_emb = face1_emb / np.linalg.norm(face1_emb)

    # Mock detected_faces array with dummy bounding boxes
    mock_faces = np.array([
        [10, 10, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
        [80, 80, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],
    ], dtype=np.float32)

    # Monkeypatch extract_embedding to return our synthetic embeddings
    embeddings = [face0_emb, face1_emb]
    recognizer.extract_embedding = lambda img, info: embeddings.pop(0)

    results = recognizer.recognize_group_faces(
        group_image=np.zeros((200, 200, 3), dtype=np.uint8),
        detected_faces=mock_faces,
        reference_records=references,
        prevent_duplicates=True,
    )

    assigned_names = [r.identity for r in results]

    # Exactly one face should be "Priya Sharma"
    assert assigned_names.count("Priya Sharma") == 1
    assert results[0].identity == "Priya Sharma"
    # Second face should NOT be Priya Sharma (should be Unknown because Bob is 0% match)
    assert results[1].identity == "Unknown"
