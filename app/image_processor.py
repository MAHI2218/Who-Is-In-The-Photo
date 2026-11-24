"""
Group Photograph Processing & Annotation Pipeline.

This module unifies YuNet face detection, employee reference indexing, and
SFace recognition to process a group photograph end-to-end.

Responsibilities:
1. Detect all faces in a group photograph using YuNet.
2. Extract SFace embeddings and match each face against the reference database.
3. Mark matched individuals with their name and matching percentage.
4. Mark unrecognized individuals as 'Unknown'.
5. Render clean, auto-scaling visual bounding boxes and text headers over the image.
6. Return both the annotated BGR image and a structured list of MatchResult objects.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from app.employee_loader import EmployeeLoader, EmployeeRecord
from app.face_recognizer import FaceRecognizer, MatchResult, DEFAULT_SFACE_THRESHOLD


# Visual Theme Colors (BGR format for OpenCV)
COLOR_KNOWN = (46, 204, 113)    # Emerald Green for identified employees
COLOR_UNKNOWN = (30, 144, 255)  # Amber / Orange for unlisted individuals
COLOR_TEXT = (255, 255, 255)    # White text for high contrast labels


class GroupImageProcessor:
    """
    Combines YuNet face detector and SFace recognizer into a cohesive
    group photo analysis pipeline.
    """

    def __init__(
        self,
        yunet_model_path: Optional[str] = None,
        sface_model_path: Optional[str] = None,
        detection_threshold: float = 0.6,
        recognition_threshold: float = DEFAULT_SFACE_THRESHOLD,
    ):
        """
        Initializes the processor and loads offline models.

        Args:
            yunet_model_path: Path to YuNet ONNX model.
            sface_model_path: Path to SFace ONNX model.
            detection_threshold: Minimum confidence score for YuNet to detect a face.
            recognition_threshold: Minimum cosine similarity for SFace to match a known employee.
        """
        project_root = Path(__file__).resolve().parent.parent
        self.yunet_path = yunet_model_path or str(project_root / "models" / "face_detection_yunet_2023mar.onnx")
        self.sface_path = sface_model_path or str(project_root / "models" / "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found at: {self.yunet_path}")
        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model not found at: {self.sface_path}")

        self.detection_threshold = float(detection_threshold)
        self.recognition_threshold = float(recognition_threshold)

        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.detection_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Initialize SFace recognizer
        self.recognizer = FaceRecognizer(
            sface_model_path=self.sface_path,
            cosine_threshold=self.recognition_threshold,
        )

    def set_recognition_threshold(self, threshold: float) -> None:
        """Dynamically adjusts the face recognition similarity threshold."""
        self.recognition_threshold = float(threshold)
        self.recognizer.set_threshold(self.recognition_threshold)

    def set_detection_threshold(self, threshold: float) -> None:
        """Dynamically adjusts the face detection confidence threshold."""
        self.detection_threshold = float(threshold)
        self.detector.setScoreThreshold(self.detection_threshold)

    @staticmethod
    def _read_image(image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """Loads image safely across operating systems and Unicode paths."""
        if isinstance(image_input, np.ndarray):
            return image_input.copy()

        file_path = str(image_input)
        try:
            with open(file_path, "rb") as f:
                bytes_array = bytearray(f.read())
            np_arr = np.asarray(bytes_array, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    @staticmethod
    def draw_annotations(
        image: np.ndarray,
        results: List[MatchResult],
        draw_landmarks: bool = False,
    ) -> np.ndarray:
        """
        Renders styled bounding boxes and text labels on the image.
        Dynamically adapts line thickness and font scale according to image resolution.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Calculate responsive scale based on image dimensions
        diagonal = np.sqrt(h**2 + w**2)
        base_scale = max(0.45, diagonal / 1800.0)
        thickness = max(1, int(round(diagonal / 700.0)))
        font_scale = max(0.4, base_scale * 0.7)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for res in results:
            x, y, bw, bh = res.bounding_box

            # Clamp coordinates to image boundaries
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            color = COLOR_KNOWN if res.is_known else COLOR_UNKNOWN

            # 1. Draw outer face bounding box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, thickness)

            # 2. Draw facial landmarks (optional)
            if draw_landmarks and res.landmarks is not None:
                for pt in res.landmarks:
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(annotated, (px, py), max(2, thickness), (0, 255, 255), -1)

            # 3. Prepare label text
            label_text = res.display_label
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Determine label placement (prefer above bounding box; flip below if at top edge)
            padding = 4
            header_h = text_h + baseline + (padding * 2)
            header_w = text_w + (padding * 2)

            if y - header_h >= 0:
                header_y1 = y - header_h
                header_y2 = y
                text_baseline_y = y - baseline - padding
            else:
                header_y1 = y
                header_y2 = y + header_h
                text_baseline_y = y + text_h + padding

            header_x1 = x
            header_x2 = min(w, x + header_w)

            # Draw filled background header pill
            cv2.rectangle(
                annotated,
                (header_x1, header_y1),
                (header_x2, header_y2),
                color,
                -1,
            )

            # Draw text inside header
            cv2.putText(
                annotated,
                label_text,
                (header_x1 + padding, text_baseline_y),
                font,
                font_scale,
                COLOR_TEXT,
                max(1, thickness - 1),
                lineType=cv2.LINE_AA,
            )

        return annotated

    def process_group_photo(
        self,
        group_image_input: Union[str, Path, np.ndarray],
        reference_records: List[EmployeeRecord],
        prevent_duplicates: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MatchResult]]:
        """
        Executes complete group photo detection, recognition, and annotation pipeline.

        Args:
            group_image_input: File path or NumPy image array of the group photo.
            reference_records: In-memory list of validated employee profiles from EmployeeLoader.
            prevent_duplicates: Ensures each employee is assigned at most once per photo.

        Returns:
            Tuple of:
            - annotated_image (np.ndarray): Group image with bounding boxes & labels rendered.
            - results (List[MatchResult]): Structured list of detected faces and their identities.
        """
        image = self._read_image(group_image_input)
        if image is None or image.size == 0:
            return None, []

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return image.copy(), []

        # 1. Configure dynamic input size for YuNet detector
        self.detector.setInputSize((w, h))

        # 2. Detect all faces in the group photo
        _, detected_faces = self.detector.detect(image)

        # 3. Handle zero faces detected
        if detected_faces is None or len(detected_faces) == 0:
            # Return unannotated copy of the original image with empty results list
            return image.copy(), []

        # 4. Perform SFace recognition and identity resolution for each face separately
        recognition_results = self.recognizer.recognize_group_faces(
            group_image=image,
            detected_faces=detected_faces,
            reference_records=reference_records,
            prevent_duplicates=prevent_duplicates,
        )

        # 5. Render bounding boxes and identity headers
        annotated_image = self.draw_annotations(image, recognition_results)

        return annotated_image, recognition_results
