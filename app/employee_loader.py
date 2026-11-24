"""
Employee Reference Directory Loader & Face Encoder.

This module scans a designated folder of employee reference portraits, extracts
identities from filenames, validates that each portrait contains exactly one face,
and computes SFace 128-dimensional biometric embeddings for offline matching.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class EmployeeRecord:
    """Represents a validated employee reference profile."""
    name: str
    file_path: str
    embedding: np.ndarray  # 128-dim normalized SFace feature vector
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray  # (5, 2) facial key points


@dataclass
class LoaderReport:
    """Summary report detailing loaded employees and skipped/invalid files."""
    total_images_found: int = 0
    valid_records: List[EmployeeRecord] = field(default_factory=list)
    skipped_no_face: List[Tuple[str, str]] = field(default_factory=list)      # (path, reason)
    skipped_multi_face: List[Tuple[str, str, int]] = field(default_factory=list)  # (path, reason, count)
    skipped_invalid_file: List[Tuple[str, str]] = field(default_factory=list) # (path, reason)

    @property
    def valid_count(self) -> int:
        return len(self.valid_records)

    @property
    def has_warnings(self) -> bool:
        return bool(self.skipped_no_face or self.skipped_multi_face or self.skipped_invalid_file)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EMPLOYEE REFERENCE FOLDER LOAD SUMMARY",
            "=" * 60,
            f"Total image files found:       {self.total_images_found}",
            f"Successfully indexed profiles: {self.valid_count}",
            f"Skipped (No face detected):    {len(self.skipped_no_face)}",
            f"Skipped (Multiple faces):      {len(self.skipped_multi_face)}",
            f"Skipped (Corrupt/Unreadable):  {len(self.skipped_invalid_file)}",
            "=" * 60,
        ]
        if self.skipped_no_face:
            lines.append("\n[Warning: No Face Detected]")
            for p, r in self.skipped_no_face:
                lines.append(f"  - {Path(p).name}: {r}")
        if self.skipped_multi_face:
            lines.append("\n[Warning: Multiple Faces Detected (ambiguous identity)]")
            for p, r, count in self.skipped_multi_face:
                lines.append(f"  - {Path(p).name}: found {count} faces")
        if self.skipped_invalid_file:
            lines.append("\n[Warning: Unreadable or Corrupt Images]")
            for p, r in self.skipped_invalid_file:
                lines.append(f"  - {Path(p).name}: {r}")
        return "\n".join(lines)


class EmployeeLoader:
    """
    Scans, detects, validates, and encodes reference portraits from a directory.
    Uses YuNet for face detection & landmarks, and SFace for 128-D embedding extraction.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        yunet_model_path: Optional[str] = None,
        sface_model_path: Optional[str] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        # Resolve default model paths relative to project root if not explicitly provided
        project_root = Path(__file__).resolve().parent.parent
        self.yunet_path = yunet_model_path or str(project_root / "models" / "face_detection_yunet_2023mar.onnx")
        self.sface_path = sface_model_path or str(project_root / "models" / "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found at: {self.yunet_path}")
        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model not found at: {self.sface_path}")

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize detector with default dummy size; dynamic input size set per image
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config="",
        )

    @staticmethod
    def extract_name_from_filename(filename: str) -> str:
        """
        Extracts clean person name from image filename.
        Example: 'Priya Sharma.jpg' -> 'Priya Sharma'
        """
        stem = Path(filename).stem
        # Clean up any duplicate spaces or underscores
        cleaned = stem.replace("_", " ").strip()
        return cleaned

    @staticmethod
    def read_image_unicode(file_path: str) -> Optional[np.ndarray]:
        """
        Robustly reads an image file supporting Unicode and Windows paths.
        Standard cv2.imread fails on Windows with non-ASCII characters;
        np.fromfile + cv2.imdecode works universally.
        """
        try:
            with open(file_path, "rb") as f:
                bytes_array = bytearray(f.read())
            np_arr = np.asarray(bytes_array, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def process_single_image(self, file_path: str) -> Tuple[Optional[EmployeeRecord], Optional[str], Optional[int]]:
        """
        Loads, detects faces, and extracts biometric embedding from a single reference photo.

        Returns:
            (record, error_message, face_count)
            If successful, error_message is None.
        """
        img = self.read_image_unicode(file_path)
        if img is None or img.size == 0:
            return None, "File is corrupted or not a valid image format", -1

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return None, f"Image dimensions too small ({w}x{h})", -1

        # Set dynamic input size for YuNet
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)

        face_count = 0 if faces is None else len(faces)

        if face_count == 0:
            return None, "No face detected in reference photo", 0

        if face_count > 1:
            return None, f"Ambiguous reference: found {face_count} faces (expected exactly 1)", face_count

        # Exactly 1 face found: extract bounding box and 5 facial landmarks
        primary_face = faces[0]
        box = (int(primary_face[0]), int(primary_face[1]), int(primary_face[2]), int(primary_face[3]))
        landmarks = primary_face[4:14].reshape((5, 2))

        # Align and crop face using SFace recognizer
        aligned_face = self.recognizer.alignCrop(img, primary_face)

        # Extract 128-D biometric feature vector
        feature = self.recognizer.feature(aligned_face)

        # L2 Normalize feature embedding vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        person_name = self.extract_name_from_filename(file_path)

        record = EmployeeRecord(
            name=person_name,
            file_path=os.path.abspath(file_path),
            embedding=feature.flatten(),
            bounding_box=box,
            landmarks=landmarks,
        )

        return record, None, 1

    def load_from_folder(
        self,
        folder_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> LoaderReport:
        """
        Scans folder_path, processes all supported images, validates single-face criteria,
        and returns a comprehensive LoaderReport.

        Args:
            folder_path: Directory path containing employee photos.
            progress_callback: Optional callback receiving (current_index, total_count, current_filename).
        """
        folder = Path(folder_path)
        report = LoaderReport()

        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Provided path is not a valid directory: {folder_path}")

        # Collect all image files matching supported extensions
        image_files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        # Sort for deterministic processing order
        image_files.sort(key=lambda p: p.name.lower())
        report.total_images_found = len(image_files)

        for idx, img_path in enumerate(image_files, start=1):
            str_path = str(img_path)
            if progress_callback:
                progress_callback(idx, report.total_images_found, img_path.name)

            record, err, face_count = self.process_single_image(str_path)

            if record is not None:
                report.valid_records.append(record)
            else:
                if face_count == 0:
                    report.skipped_no_face.append((str_path, err or "No face"))
                elif face_count > 1:
                    report.skipped_multi_face.append((str_path, err or "Multiple faces", face_count))
                else:
                    report.skipped_invalid_file.append((str_path, err or "Corrupt file"))

        return report


def select_folder_interactively() -> Optional[str]:
    """Helper to open a native desktop folder selection dialog."""
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication([])
        selected_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Employee Reference Photos Folder",
            os.getcwd(),
        )
        return selected_dir if selected_dir else None
    except ImportError:
        # Fallback to tkinter if PySide6 isn't initialized
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory(title="Select Employee Reference Photos Folder")


if __name__ == "__main__":
    import sys

    print("=== Employee Reference Folder Loader ===")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None

    if not target_dir:
        print("No folder path supplied via argument. Opening folder dialog...")
        target_dir = select_folder_interactively()

    if not target_dir:
        print("Folder selection cancelled.")
        sys.exit(0)

    print(f"Scanning directory: {target_dir}")
    loader = EmployeeLoader()

    def on_progress(current, total, filename):
        print(f"[{current}/{total}] Processing: {filename}")

    report = loader.load_from_folder(target_dir, progress_callback=on_progress)
    print("\n" + report.summary())
