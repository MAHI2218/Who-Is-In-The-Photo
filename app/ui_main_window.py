"""
User Interface Module for 'Who Is in the Photo?'
Built with PySide6.

Features:
- Clean, modern desktop layout with dark-slate theme and intuitive workflow.
- Responsive image viewer maintaining aspect ratio on resize.
- Background QThread processing to prevent UI freezing during model inference.
- Results sidebar with summary statistics and per-person identification cards.
- Configurable similarity threshold slider.
- High-resolution annotated image export.
"""

import os
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.employee_loader import EmployeeLoader, EmployeeRecord, LoaderReport
from app.face_recognizer import DEFAULT_SFACE_THRESHOLD, MatchResult
from app.image_processor import GroupImageProcessor


# ----------------------------------------------------------------------
# Background Workers (QThread) to ensure responsive UI
# ----------------------------------------------------------------------

class EmployeeLoadingWorker(QThread):
    """Background worker for scanning and embedding employee reference photos."""
    progress = Signal(int, int, str)
    finished = Signal(object)  # LoaderReport
    error = Signal(str)

    def __init__(self, folder_path: str, loader: EmployeeLoader):
        super().__init__()
        self.folder_path = folder_path
        self.loader = loader

    def run(self):
        try:
            report = self.loader.load_from_folder(
                self.folder_path,
                progress_callback=lambda cur, tot, name: self.progress.emit(cur, tot, name)
            )
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(str(e))


class RecognitionWorker(QThread):
    """Background worker for group face detection and matching."""
    finished = Signal(object, list)  # (annotated_np_array, List[MatchResult])
    error = Signal(str)

    def __init__(
        self,
        processor: GroupImageProcessor,
        group_image: np.ndarray,
        reference_records: List[EmployeeRecord],
    ):
        super().__init__()
        self.processor = processor
        self.group_image = group_image
        self.reference_records = reference_records

    def run(self):
        try:
            annotated_img, results = self.processor.process_group_photo(
                group_image_input=self.group_image,
                reference_records=self.reference_records,
                prevent_duplicates=True,
            )
            self.finished.emit(annotated_img, results)
        except Exception as e:
            self.error.emit(str(e))


# ----------------------------------------------------------------------
# Resizable Image Viewer Widget
# ----------------------------------------------------------------------

class ScaledImageViewer(QLabel):
    """Custom QLabel that scales its displayed pixmap preserving aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 350)
        self._pixmap: Optional[QPixmap] = None

    def set_image_pixmap(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            super().setPixmap(scaled)
        else:
            super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


# ----------------------------------------------------------------------
# Main Application Window
# ----------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main Application Window for 'Who Is in the Photo?'."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Who Is in the Photo? - Offline Face Recognition")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)

        # Core Pipeline Engines
        self.loader = EmployeeLoader()
        self.processor = GroupImageProcessor(recognition_threshold=DEFAULT_SFACE_THRESHOLD)

        # State Variables
        self.reference_records: List[EmployeeRecord] = []
        self.current_group_bgr: Optional[np.ndarray] = None
        self.current_annotated_bgr: Optional[np.ndarray] = None
        self.current_results: List[MatchResult] = []

        # Background Worker handles
        self.load_worker: Optional[EmployeeLoadingWorker] = None
        self.recog_worker: Optional[RecognitionWorker] = None

        # Build UI & Styles
        self._apply_stylesheet()
        self._init_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(12)

        # 1. Header Bar
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        app_title = QLabel("Who Is in the Photo?")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel("Offline Desktop Face Recognition & Attendance Analysis Platform")
        app_subtitle.setObjectName("appSubtitle")

        title_box.addWidget(app_title)
        title_box.addWidget(app_subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        root_layout.addLayout(header_layout)

        # 2. Main Body Splitter (Controls / Photo Display / Results Sidebar)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)

        # Left Control Panel
        control_panel = self._create_control_panel()
        main_splitter.addWidget(control_panel)

        # Center Photo Viewer
        center_viewer_panel = self._create_photo_viewer_panel()
        main_splitter.addWidget(center_viewer_panel)

        # Right Results Panel
        results_panel = self._create_results_panel()
        main_splitter.addWidget(results_panel)

        main_splitter.setSizes([290, 650, 340])
        root_layout.addWidget(main_splitter, 1)

        # 3. Bottom Status and Progress Bar
        status_bar_layout = QHBoxLayout()
        self.status_label = QLabel("Ready. Select employee folder to begin.")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(12)

        status_bar_layout.addWidget(self.status_label, 1)
        status_bar_layout.addWidget(self.progress_bar, 0)
        root_layout.addLayout(status_bar_layout)

    def _create_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        panel_title = QLabel("WORKFLOW CONTROLS")
        panel_title.setObjectName("sectionHeader")
        layout.addWidget(panel_title)

        # Step 1: Employee Database
        step1_box = QFrame()
        step1_box.setObjectName("stepBox")
        s1_layout = QVBoxLayout(step1_box)
        s1_layout.setContentsMargins(10, 10, 10, 10)
        s1_layout.setSpacing(8)

        s1_title = QLabel("1. Employee Database")
        s1_title.setObjectName("stepTitle")
        s1_layout.addWidget(s1_title)

        self.btn_select_folder = QPushButton("📁 Select Employee Folder")
        self.btn_select_folder.setObjectName("btnPrimary")
        self.btn_select_folder.clicked.connect(self._on_select_folder_clicked)
        s1_layout.addWidget(self.btn_select_folder)

        self.lbl_employee_status = QLabel("No employee folder selected.")
        self.lbl_employee_status.setObjectName("lblSubtext")
        self.lbl_employee_status.setWordWrap(True)
        s1_layout.addWidget(self.lbl_employee_status)

        layout.addWidget(step1_box)

        # Step 2: Group Photograph
        step2_box = QFrame()
        step2_box.setObjectName("stepBox")
        s2_layout = QVBoxLayout(step2_box)
        s2_layout.setContentsMargins(10, 10, 10, 10)
        s2_layout.setSpacing(8)

        s2_title = QLabel("2. Target Group Photo")
        s2_title.setObjectName("stepTitle")
        s2_layout.addWidget(s2_title)

        self.btn_upload_group = QPushButton("🖼️ Upload Group Photo")
        self.btn_upload_group.setObjectName("btnSecondary")
        self.btn_upload_group.clicked.connect(self._on_upload_group_clicked)
        s2_layout.addWidget(self.btn_upload_group)

        self.lbl_group_status = QLabel("No group photo uploaded.")
        self.lbl_group_status.setObjectName("lblSubtext")
        self.lbl_group_status.setWordWrap(True)
        s2_layout.addWidget(self.lbl_group_status)

        layout.addWidget(step2_box)

        # Step 3: Match Configuration & Execution
        step3_box = QFrame()
        step3_box.setObjectName("stepBox")
        s3_layout = QVBoxLayout(step3_box)
        s3_layout.setContentsMargins(10, 10, 10, 10)
        s3_layout.setSpacing(8)

        s3_title = QLabel("3. Identification & Tolerance")
        s3_title.setObjectName("stepTitle")
        s3_layout.addWidget(s3_title)

        # Slider for threshold
        slider_label_box = QHBoxLayout()
        slider_title = QLabel("Similarity Threshold:")
        slider_title.setObjectName("lblSubtext")
        self.lbl_threshold_val = QLabel(f"{DEFAULT_SFACE_THRESHOLD:.2f}")
        self.lbl_threshold_val.setObjectName("lblThresholdValue")
        slider_label_box.addWidget(slider_title)
        slider_label_box.addStretch()
        slider_label_box.addWidget(self.lbl_threshold_val)
        s3_layout.addLayout(slider_label_box)

        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(20, 70)  # 0.20 to 0.70
        self.slider_threshold.setValue(int(DEFAULT_SFACE_THRESHOLD * 100))
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        s3_layout.addWidget(self.slider_threshold)

        self.btn_identify = QPushButton("🔍 Identify People")
        self.btn_identify.setObjectName("btnIdentify")
        self.btn_identify.setEnabled(False)
        self.btn_identify.clicked.connect(self._on_identify_clicked)
        s3_layout.addWidget(self.btn_identify)

        layout.addWidget(step3_box)

        # Step 4: Export Result
        step4_box = QFrame()
        step4_box.setObjectName("stepBox")
        s4_layout = QVBoxLayout(step4_box)
        s4_layout.setContentsMargins(10, 10, 10, 10)
        s4_layout.setSpacing(8)

        s4_title = QLabel("4. Save Output")
        s4_title.setObjectName("stepTitle")
        s4_layout.addWidget(s4_title)

        self.btn_save_image = QPushButton("💾 Save Annotated Photo")
        self.btn_save_image.setObjectName("btnSecondary")
        self.btn_save_image.setEnabled(False)
        self.btn_save_image.clicked.connect(self._on_save_image_clicked)
        s4_layout.addWidget(self.btn_save_image)

        layout.addWidget(step4_box)
        layout.addStretch()

        return panel

    def _create_photo_viewer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("viewerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        viewer_header = QLabel("PHOTOGRAPH VIEWER")
        viewer_header.setObjectName("sectionHeader")
        top_row.addWidget(viewer_header)
        top_row.addStretch()

        self.lbl_photo_meta = QLabel("")
        self.lbl_photo_meta.setObjectName("lblSubtext")
        top_row.addWidget(self.lbl_photo_meta)
        layout.addLayout(top_row)

        # Central Image View
        self.image_viewer = ScaledImageViewer()
        self.image_viewer.setText(
            "No Photograph Loaded\n\n"
            "1. Select an employee reference folder\n"
            "2. Upload a group photograph\n"
            "3. Click 'Identify People'"
        )
        self.image_viewer.setStyleSheet("color: #71717a; font-size: 14px; font-weight: 500;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_viewer)
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("photoScrollArea")
        layout.addWidget(scroll_area, 1)

        return panel

    def _create_results_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("resultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        results_header = QLabel("IDENTIFICATION RESULTS")
        results_header.setObjectName("sectionHeader")
        layout.addWidget(results_header)

        # Stats Cards Layout
        stats_box = QFrame()
        stats_box.setObjectName("statsBox")
        sb_layout = QHBoxLayout(stats_box)
        sb_layout.setContentsMargins(6, 8, 6, 8)

        self.stat_total = self._create_stat_badge("Faces", "0", "#60a5fa")
        self.stat_known = self._create_stat_badge("Identified", "0", "#34d399")
        self.stat_unknown = self._create_stat_badge("Unknown", "0", "#fbbf24")

        sb_layout.addWidget(self.stat_total)
        sb_layout.addWidget(self.stat_known)
        sb_layout.addWidget(self.stat_unknown)
        layout.addWidget(stats_box)

        # Scrollable list for individual face result cards
        self.results_scroll = QScrollArea()
        self.results_scroll.setObjectName("resultsScroll")
        self.results_scroll.setWidgetResizable(True)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()

        self.results_scroll.setWidget(self.results_container)
        layout.addWidget(self.results_scroll, 1)

        # Initial Empty State Label
        self.empty_results_label = QLabel("Run 'Identify People' to see detected individual profiles and confidence scores.")
        self.empty_results_label.setObjectName("lblSubtext")
        self.empty_results_label.setWordWrap(True)
        self.results_layout.insertWidget(0, self.empty_results_label)

        return panel

    def _create_stat_badge(self, label: str, value: str, color: str) -> QWidget:
        badge = QFrame()
        badge.setObjectName("statBadge")
        b_layout = QVBoxLayout(badge)
        b_layout.setContentsMargins(4, 4, 4, 4)
        b_layout.setSpacing(2)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("statVal")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")

        name_lbl = QLabel(label)
        name_lbl.setObjectName("lblSubtext")
        name_lbl.setAlignment(Qt.AlignCenter)

        b_layout.addWidget(val_lbl)
        b_layout.addWidget(name_lbl)
        badge.val_lbl = val_lbl  # store reference to update later
        return badge

    # ------------------------------------------------------------------
    # Event Handlers & Business Logic
    # ------------------------------------------------------------------

    @Slot()
    def _on_select_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Employee Reference Photos Folder",
            os.getcwd(),
        )
        if not folder:
            return

        self.btn_select_folder.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Scanning directory: {Path(folder).name}...")

        # Run loading in background QThread
        self.load_worker = EmployeeLoadingWorker(folder, self.loader)
        self.load_worker.progress.connect(self._on_load_progress)
        self.load_worker.finished.connect(self._on_load_finished)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()

    @Slot(int, int, str)
    def _on_load_progress(self, current: int, total: int, filename: str):
        pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"Indexing employee photos: [{current}/{total}] {filename}")

    @Slot(object)
    def _on_load_finished(self, report: LoaderReport):
        self.btn_select_folder.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.reference_records = report.valid_records

        if report.total_images_found == 0:
            self.lbl_employee_status.setText("⚠️ No image files (.jpg, .jpeg, .png) found.")
            self.status_label.setText("Warning: Selected folder contains no supported photos.")
            QMessageBox.warning(
                self,
                "No Photos Found",
                "No supported image files (.jpg, .jpeg, .png) were found in the selected folder.\n\n"
                "Please choose a directory containing employee portraits."
            )
            self._update_identify_button_state()
            return

        if report.valid_count == 0:
            self.lbl_employee_status.setText("⚠️ 0 valid portraits indexed (all images skipped).")
            self.status_label.setText("Warning: No valid employee profiles could be indexed.")
            QMessageBox.warning(
                self,
                "No Valid Profiles",
                f"Found {report.total_images_found} images, but none had exactly 1 clear face.\n\n"
                f"• {len(report.skipped_no_face)} had no face detected\n"
                f"• {len(report.skipped_multi_face)} had multiple faces\n"
                f"• {len(report.skipped_invalid_file)} were corrupt/unreadable"
            )
            self._update_identify_button_state()
            return

        self.lbl_employee_status.setText(
            f"✅ {report.valid_count} employee profiles indexed."
        )

        if report.has_warnings:
            warn_msg = (
                f"Loaded {report.valid_count} valid portraits out of {report.total_images_found} images.\n\n"
                f"• {len(report.skipped_no_face)} skipped (no face detected)\n"
                f"• {len(report.skipped_multi_face)} skipped (multiple faces)\n"
                f"• {len(report.skipped_invalid_file)} skipped (unreadable/corrupt)"
            )
            QMessageBox.information(self, "Employee Loading Summary", warn_msg)

        self.status_label.setText(f"Ready: {report.valid_count} employee reference profiles loaded.")
        self._update_identify_button_state()

    @Slot(str)
    def _on_load_error(self, err_msg: str):
        self.btn_select_folder.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Loading Error", f"Failed to load folder:\n{err_msg}")
        self.status_label.setText("Error loading employee folder.")

    @Slot()
    def _on_upload_group_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Group Photograph",
            os.getcwd(),
            "Image Files (*.jpg *.jpeg *.png *.bmp)",
        )
        if not file_path:
            return

        # Load image via OpenCV
        img = self.loader.read_image_unicode(file_path)
        if img is None:
            QMessageBox.warning(
                self,
                "Invalid Image",
                f"Could not decode the selected image file:\n{file_path}\n\nThe file may be corrupted or in an unsupported format."
            )
            return

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            QMessageBox.warning(
                self,
                "Image Too Small",
                f"The selected image dimensions ({w}×{h} px) are too small to reliably detect faces."
            )
            return

        self.current_group_bgr = img
        self.current_annotated_bgr = None
        self.current_results = []

        self.lbl_group_status.setText(f"✅ {Path(file_path).name} ({w}×{h})")
        self.lbl_photo_meta.setText(f"{w}×{h} px")

        # Display raw group photo
        pixmap = self._cv2_to_qpixmap(img)
        self.image_viewer.set_image_pixmap(pixmap)

        self.btn_save_image.setEnabled(False)
        self._clear_results_panel()
        self.status_label.setText("Group photo loaded. Click 'Identify People' to process.")
        self._update_identify_button_state()

    @Slot()
    def _on_threshold_changed(self, val: int):
        float_val = val / 100.0
        self.lbl_threshold_val.setText(f"{float_val:.2f}")
        self.processor.set_recognition_threshold(float_val)

    def _update_identify_button_state(self):
        can_run = bool(self.reference_records and self.current_group_bgr is not None)
        self.btn_identify.setEnabled(can_run)

    @Slot()
    def _on_identify_clicked(self):
        if not self.reference_records:
            QMessageBox.information(
                self,
                "Employee Database Required",
                "Please select an employee folder containing reference portraits before identifying people."
            )
            return

        if self.current_group_bgr is None:
            QMessageBox.information(
                self,
                "Group Photograph Required",
                "Please upload a target group photograph first."
            )
            return

        self.btn_identify.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate spinner
        self.status_label.setText("Detecting and recognizing faces offline...")

        # Run recognition in background thread
        self.recog_worker = RecognitionWorker(
            self.processor,
            self.current_group_bgr,
            self.reference_records,
        )
        self.recog_worker.finished.connect(self._on_recognition_finished)
        self.recog_worker.error.connect(self._on_recognition_error)
        self.recog_worker.start()

    @Slot(object, list)
    def _on_recognition_finished(self, annotated_bgr: np.ndarray, results: List[MatchResult]):
        self.btn_identify.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_annotated_bgr = annotated_bgr
        self.current_results = results

        # Update Viewer with annotated image
        pixmap = self._cv2_to_qpixmap(annotated_bgr)
        self.image_viewer.set_image_pixmap(pixmap)

        # Enable export
        self.btn_save_image.setEnabled(True)

        # Populate Results Sidebar
        self._populate_results_panel(results)

        total = len(results)
        known = sum(1 for r in results if r.is_known)
        unknown = total - known
        self.status_label.setText(
            f"Analysis complete: {total} faces detected ({known} identified, {unknown} unknown)."
        )

    @Slot(str)
    def _on_recognition_error(self, err_msg: str):
        self.btn_identify.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Recognition Error", f"Error during face recognition:\n{err_msg}")
        self.status_label.setText("Recognition error occurred.")

    @Slot()
    def _on_save_image_clicked(self):
        if self.current_annotated_bgr is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Photograph",
            "annotated_group_photo.jpg",
            "JPEG Image (*.jpg);;PNG Image (*.png)",
        )
        if not save_path:
            return

        # Unicode safe save
        try:
            ext = Path(save_path).suffix or ".jpg"
            success, encoded_img = cv2.imencode(ext, self.current_annotated_bgr)
            if success:
                with open(save_path, "wb") as f:
                    f.write(encoded_img.tobytes())
                QMessageBox.information(self, "Saved", f"Annotated photograph saved successfully to:\n{save_path}")
            else:
                QMessageBox.critical(self, "Save Error", "Could not encode image format.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    # ------------------------------------------------------------------
    # Results Display Helpers
    # ------------------------------------------------------------------

    def _clear_results_panel(self):
        # Remove all existing result cards
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.stat_total.val_lbl.setText("0")
        self.stat_known.val_lbl.setText("0")
        self.stat_unknown.val_lbl.setText("0")

    def _populate_results_panel(self, results: List[MatchResult]):
        self._clear_results_panel()

        total = len(results)
        known = sum(1 for r in results if r.is_known)
        unknown = total - known

        self.stat_total.val_lbl.setText(str(total))
        self.stat_known.val_lbl.setText(str(known))
        self.stat_unknown.val_lbl.setText(str(unknown))

        if total == 0:
            no_faces_lbl = QLabel("No faces were detected in this photograph.")
            no_faces_lbl.setObjectName("lblSubtext")
            self.results_layout.insertWidget(0, no_faces_lbl)
            return

        for res in results:
            card = QFrame()
            card.setObjectName("resultCard")
            c_layout = QHBoxLayout(card)
            c_layout.setContentsMargins(10, 8, 10, 8)
            c_layout.setSpacing(10)

            # Left Face Index Badge
            idx_badge = QLabel(f"#{res.face_index + 1}")
            idx_badge.setObjectName("faceIndexBadge")
            c_layout.addWidget(idx_badge)

            # Identity details
            details_box = QVBoxLayout()
            details_box.setSpacing(2)

            name_lbl = QLabel(res.identity)
            name_lbl.setObjectName("resultNameKnown" if res.is_known else "resultNameUnknown")

            score_str = f"Match: {res.similarity_score * 100:.1f}% | Detection: {res.detection_score * 100:.0f}%"
            score_lbl = QLabel(score_str)
            score_lbl.setObjectName("lblSubtext")

            details_box.addWidget(name_lbl)
            details_box.addWidget(score_lbl)
            c_layout.addLayout(details_box, 1)

            # Status Pill (Identified / Unknown)
            pill = QLabel("IDENTIFIED" if res.is_known else "UNKNOWN")
            pill.setObjectName("statusPillKnown" if res.is_known else "statusPillUnknown")
            c_layout.addWidget(pill)

            self.results_layout.insertWidget(self.results_layout.count() - 1, card)

    @staticmethod
    def _cv2_to_qpixmap(bgr_img: np.ndarray) -> QPixmap:
        """Converts an OpenCV BGR numpy array to QPixmap for display."""
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def closeEvent(self, event):
        """Safely terminates any background worker threads before window closes."""
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.terminate()
            self.load_worker.wait(1000)
        if self.recog_worker and self.recog_worker.isRunning():
            self.recog_worker.terminate()
            self.recog_worker.wait(1000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Modern Dark Theme Stylesheet
    # ------------------------------------------------------------------

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                font-size: 13px;
            }

            #appTitle {
                font-size: 22px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: -0.5px;
            }

            #appSubtitle {
                font-size: 12px;
                color: #94a3b8;
            }

            #sectionHeader {
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                letter-spacing: 1.2px;
            }

            #controlPanel, #viewerPanel, #resultsPanel {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
            }

            #stepBox {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
            }

            #stepTitle {
                font-size: 12px;
                font-weight: 600;
                color: #e2e8f0;
            }

            #lblSubtext {
                font-size: 11px;
                color: #94a3b8;
            }

            #lblThresholdValue {
                font-size: 12px;
                font-weight: 700;
                color: #38bdf8;
            }

            QPushButton {
                padding: 8px 14px;
                border-radius: 5px;
                font-weight: 600;
                font-size: 12px;
            }

            #btnPrimary {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
            }
            #btnPrimary:hover {
                background-color: #1d4ed8;
            }
            #btnPrimary:pressed {
                background-color: #1e40af;
            }

            #btnSecondary {
                background-color: #334155;
                color: #f8fafc;
                border: 1px solid #475569;
            }
            #btnSecondary:hover {
                background-color: #475569;
            }
            #btnSecondary:pressed {
                background-color: #1e293b;
            }

            #btnIdentify {
                background-color: #059669;
                color: #ffffff;
                border: none;
                font-size: 13px;
                padding: 10px 14px;
            }
            #btnIdentify:hover {
                background-color: #047857;
            }
            #btnIdentify:pressed {
                background-color: #065f46;
            }
            #btnIdentify:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }

            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }

            QSlider::groove:horizontal {
                height: 4px;
                background: #334155;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #38bdf8;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f8fafc;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }

            #photoScrollArea, #resultsScroll {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
            }

            #statsBox {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
            }

            #statBadge {
                background-color: transparent;
            }

            #resultCard {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            #resultCard:hover {
                border-color: #475569;
            }

            #faceIndexBadge {
                background-color: #334155;
                color: #e2e8f0;
                font-weight: 700;
                font-size: 11px;
                padding: 4px 6px;
                border-radius: 4px;
            }

            #resultNameKnown {
                font-weight: 700;
                font-size: 13px;
                color: #34d399;
            }

            #resultNameUnknown {
                font-weight: 700;
                font-size: 13px;
                color: #fbbf24;
            }

            #statusPillKnown {
                background-color: #064e3b;
                color: #34d399;
                font-size: 10px;
                font-weight: 700;
                padding: 3px 7px;
                border-radius: 4px;
            }

            #statusPillUnknown {
                background-color: #451a03;
                color: #fbbf24;
                font-size: 10px;
                font-weight: 700;
                padding: 3px 7px;
                border-radius: 4px;
            }

            #statusLabel {
                font-size: 12px;
                color: #94a3b8;
            }

            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 5px;
            }

            QScrollBar:vertical {
                background: #0f172a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
        """)
