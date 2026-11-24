"""
Face Recognition & Identity Matching Module (SFace).

This module takes detected faces from a group photo, generates 128-dimensional
SFace biometric embeddings, compares them against reference employee profiles using
cosine similarity, and resolves identities.

Features:
- Configurable cosine similarity threshold (default: 0.363 as recommended for SFace).
- Rejection of unknown faces below threshold (marks as 'Unknown').
- Duplicate prevention: Ensures each employee identity is matched at most once
  per group photo (the highest-confidence face receives the match).
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from app.employee_loader import EmployeeRecord


# Recommended cosine similarity threshold for SFace
# SFace cosine similarity ranges from -1.0 to +1.0.
# Benchmark threshold for SFace 1:1 verification is ~0.363.
# Higher threshold = stricter matching (fewer false positives).
# Lower threshold = more relaxed matching (higher recall).
DEFAULT_SFACE_THRESHOLD = 0.363


@dataclass
class MatchResult:
    """Represents recognition result for a single face in a group photo."""
    face_index: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray                   # 5 facial landmarks (5, 2)
    identity: str                           # Employee Name or "Unknown"
    similarity_score: float                 # Cosine similarity (-1.0 to 1.0)
    is_known: bool                          # True if matched to an employee, False if "Unknown"
    detection_score: float = 0.0            # YuNet face detection confidence

    @property
    def display_label(self) -> str:
        """Formatted label for bounding box overlays."""
        if self.is_known:
            return f"{self.identity} ({self.similarity_score * 100:.1f}%)"
        return f"Unknown ({self.similarity_score * 100:.1f}%)"


class FaceRecognizer:
    """
    Offline Face Recognizer using OpenCV's SFace model.
    Computes 128-D embeddings and performs cosine similarity matching.
    """

    def __init__(
        self,
        sface_model_path: Optional[str] = None,
        cosine_threshold: float = DEFAULT_SFACE_THRESHOLD,
    ):
        """
        Initializes SFace recognizer.

        Args:
            sface_model_path: Path to 'face_recognition_sface_2021dec.onnx'.
            cosine_threshold: Minimum cosine similarity to accept a match.
        """
        project_root = Path(__file__).resolve().parent.parent
        self.sface_path = sface_model_path or str(project_root / "models" / "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model file not found at: {self.sface_path}")

        self.cosine_threshold = float(cosine_threshold)

        # Initialize native OpenCV FaceRecognizerSF
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config="",
        )

    def set_threshold(self, threshold: float) -> None:
        """Dynamically adjusts the matching threshold (e.g. from a UI slider)."""
        self.cosine_threshold = float(threshold)

    def extract_embedding(self, image: np.ndarray, face_info: np.ndarray) -> np.ndarray:
        """
        Aligns, crops, and extracts a normalized 128-D embedding for a single face.

        Args:
            image: Full BGR input image.
            face_info: Detection output from YuNet containing bounding box and landmarks.
                       Shape: (14,) -> [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc]

        Returns:
            L2-normalized 128-dimensional embedding vector (shape: (128,)).
        """
        # Geometrically align and crop the face using YuNet 5 facial landmarks
        aligned_face = self.recognizer.alignCrop(image, face_info)

        # Extract 128-dimensional feature embedding
        feature = self.recognizer.feature(aligned_face)

        # Flatten and normalize embedding to unit length (L2 norm = 1.0)
        embedding = feature.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two unit-normalized embeddings.
        Since vectors are L2-normalized: Cosine Similarity = dot_product(a, b).
        Output range is [-1.0, 1.0].
        """
        return float(np.dot(embedding_a, embedding_b))

    def match_face_to_references(
        self,
        query_embedding: np.ndarray,
        reference_records: List[EmployeeRecord],
        excluded_names: Optional[Set[str]] = None,
    ) -> Tuple[str, float, bool]:
        """
        Finds the closest matching employee for a single face embedding.

        Args:
            query_embedding: 128-D embedding of the face to identify.
            reference_records: List of registered EmployeeRecord objects.
            excluded_names: Optional set of names to exclude (e.g. if already assigned).

        Returns:
            (identity_name, best_similarity_score, is_match)
        """
        if not reference_records:
            return "Unknown", 0.0, False

        excluded = excluded_names or set()
        best_name = "Unknown"
        best_score = -1.0

        for record in reference_records:
            if record.name in excluded:
                continue

            similarity = self.compute_cosine_similarity(query_embedding, record.embedding)
            if similarity > best_score:
                best_score = similarity
                best_name = record.name

        # Enforce threshold check:
        # If best similarity is below threshold, reject match and return "Unknown"
        is_known = (best_score >= self.cosine_threshold)
        final_identity = best_name if is_known else "Unknown"

        return final_identity, max(0.0, best_score), is_known

    def recognize_group_faces(
        self,
        group_image: np.ndarray,
        detected_faces: Optional[np.ndarray],
        reference_records: List[EmployeeRecord],
        prevent_duplicates: bool = True,
    ) -> List[MatchResult]:
        """
        Performs end-to-end recognition on all detected faces in a group photo.

        Args:
            group_image: BGR group photograph.
            detected_faces: NumPy array of detected faces from YuNet (N, 15).
            reference_records: List of registered EmployeeRecord profiles.
            prevent_duplicates: If True, ensures no single employee identity is assigned
                                more than once in the same group photograph.

        Returns:
            List of MatchResult objects, one for each detected face.
        """
        if detected_faces is None or len(detected_faces) == 0:
            return []

        num_faces = len(detected_faces)
        num_refs = len(reference_records)

        # 1. Extract embeddings for all detected faces in the group photo
        group_embeddings: List[np.ndarray] = []
        for face_info in detected_faces:
            emb = self.extract_embedding(group_image, face_info)
            group_embeddings.append(emb)

        # If no reference records are loaded, all faces are Unknown
        if num_refs == 0:
            results = []
            for idx, face_info in enumerate(detected_faces):
                box = (int(face_info[0]), int(face_info[1]), int(face_info[2]), int(face_info[3]))
                landmarks = face_info[4:14].reshape((5, 2))
                det_score = float(face_info[14]) if len(face_info) > 14 else 1.0
                results.append(
                    MatchResult(
                        face_index=idx,
                        bounding_box=box,
                        landmarks=landmarks,
                        identity="Unknown",
                        similarity_score=0.0,
                        is_known=False,
                        detection_score=det_score,
                    )
                )
            return results

        # 2. Build pairwise similarity matrix: shape (num_faces, num_refs)
        # We stack reference embeddings into a matrix (num_refs, 128)
        ref_matrix = np.vstack([rec.embedding for rec in reference_records])  # (M, 128)
        query_matrix = np.vstack(group_embeddings)                            # (N, 128)

        # Matrix multiplication computes all pairwise cosine similarities in one vectorized step:
        # sim_matrix[i, j] = cosine similarity between group face i and employee j
        sim_matrix = np.matmul(query_matrix, ref_matrix.T)                    # (N, M)

        # 3. Resolve identities
        assigned_results: Dict[int, Tuple[str, float, bool]] = {}

        if prevent_duplicates:
            # Greedy Global Best-Match Assignment:
            # Collect all candidate (score, face_idx, ref_idx) pairs and sort descending.
            # Greedily accept the highest score, claiming both that face and that employee identity.
            candidate_pairs = []
            for face_idx in range(num_faces):
                for ref_idx in range(num_refs):
                    score = float(sim_matrix[face_idx, ref_idx])
                    candidate_pairs.append((score, face_idx, ref_idx))

            # Sort candidate matches by highest similarity first
            candidate_pairs.sort(key=lambda item: item[0], reverse=True)

            claimed_faces: Set[int] = set()
            claimed_employees: Set[str] = set()

            for score, face_idx, ref_idx in candidate_pairs:
                emp_name = reference_records[ref_idx].name

                if face_idx in claimed_faces:
                    continue
                if emp_name in claimed_employees:
                    continue

                if score >= self.cosine_threshold:
                    assigned_results[face_idx] = (emp_name, score, True)
                    claimed_faces.add(face_idx)
                    claimed_employees.add(emp_name)

            # Any face that didn't get a valid match above threshold is Unknown
            for face_idx in range(num_faces):
                if face_idx not in claimed_faces:
                    best_raw_score = float(np.max(sim_matrix[face_idx]))
                    assigned_results[face_idx] = ("Unknown", max(0.0, best_raw_score), False)

        else:
            # Independent matching per face (duplicates allowed)
            for face_idx in range(num_faces):
                best_ref_idx = int(np.argmax(sim_matrix[face_idx]))
                best_score = float(sim_matrix[face_idx, best_ref_idx])
                if best_score >= self.cosine_threshold:
                    assigned_results[face_idx] = (reference_records[best_ref_idx].name, best_score, True)
                else:
                    assigned_results[face_idx] = ("Unknown", max(0.0, best_score), False)

        # 4. Assemble final MatchResult list
        final_results: List[MatchResult] = []
        for face_idx, face_info in enumerate(detected_faces):
            box = (int(face_info[0]), int(face_info[1]), int(face_info[2]), int(face_info[3]))
            landmarks = face_info[4:14].reshape((5, 2))
            det_score = float(face_info[14]) if len(face_info) > 14 else 1.0

            identity, sim_score, is_known = assigned_results[face_idx]

            final_results.append(
                MatchResult(
                    face_index=face_idx,
                    bounding_box=box,
                    landmarks=landmarks,
                    identity=identity,
                    similarity_score=sim_score,
                    is_known=is_known,
                    detection_score=det_score,
                )
            )

        return final_results
