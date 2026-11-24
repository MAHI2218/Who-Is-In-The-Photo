# Who Is in the Photo? - Offline Face Recognition Desktop Application

[![Tests](https://github.com/MAHI2218/Who-Is-In-The-Photo/actions/workflows/ci.yml/badge.svg)](https://github.com/MAHI2218/Who-Is-In-The-Photo/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-PySide6%20(Qt6)-green.svg)](https://pypi.org/project/PySide6/)
[![Engine](https://img.shields.io/badge/CV-OpenCV%20(YuNet%20%2B%20SFace)-orange.svg)](https://opencv.org/)
[![Offline](https://img.shields.io/badge/Runtime-100%25%20Offline-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-lightgrey.svg)]()

A modern, fully offline desktop application built with Python, PySide6, and OpenCV that automatically identifies individuals in a group photograph against an office directory of reference photos. Individuals in the group photo who are not in the reference directory are accurately flagged as **"Unknown"**.

---

## 📌 Problem Overview
An organization wants to quickly identify who is present in a group photograph of approximately 10 individuals using an offline database of 50 single-person reference portraits named in the format `Firstname Lastname.jpg` (e.g. `Priya Sharma.jpg`). Any individuals present in the group photo who are not registered in the directory must be marked as **"Unknown"**.

---

## 🚀 Key Features
- **100% Offline Runtime:** No external internet connectivity, cloud APIs, or hosted services required. Everything runs locally on CPU.
- **Ultra-Fast & Lightweight:** Powered by OpenCV Zoo's native C++ ONNX models:
  - **YuNet** (~227 KB): Sub-nanosecond face detection with 5-point landmark extraction.
  - **SFace** (~36.9 MB): 128-dimensional biometric embedding extraction and face alignment.
- **Strict Reference Validation:** Verifies that every employee portrait contains **exactly one face**. Ambiguous photos (0 faces or multiple faces) are flagged and reported in an ingestion modal.
- **Duplicate Match Prevention:** Greedy global best-match algorithm ensures each employee identity is matched at most once per photo.
- **Adjustable Tolerance Slider:** Live similarity threshold control (`0.20` – `0.70`) to demonstrate precision-recall trade-offs without reprocessing images.
- **Responsive Dark-Slate GUI:** High-DPI scaling, smooth image resizing, and asynchronous `QThread` workers so the UI never freezes during batch processing.
- **High-Resolution Export:** One-click saving of the annotated photo with high-contrast colored bounding boxes (Emerald Green for identified staff, Amber for unknowns).

---

## 📂 Project Architecture & Directory Layout

```text
Who-Is-In-The-Photo/
├── app/
│   ├── employee_loader.py    # Reference directory scanning & single-face validation
│   ├── face_recognizer.py    # SFace cosine similarity & greedy duplicate prevention
│   ├── image_processor.py    # Face detection & resolution-adaptive visual annotation
│   ├── ui_main_window.py     # Modern PySide6 GUI with non-blocking QThread workers
│   └── __init__.py
├── models/
│   ├── face_detection_yunet_2023mar.onnx     # YuNet ONNX weights (~227 KB)
│   ├── face_recognition_sface_2021dec.onnx   # SFace ONNX weights (~36.9 MB)
│   └── README.md
├── sample_data/
│   ├── known_faces/          # 50 sample portraits (named 'Firstname Lastname.jpg')
│   ├── group_photos/         # Test group photos with mixed known and unknown individuals
│   └── README.md
├── scripts/
│   ├── download_models.py    # 1-command model downloader from OpenCV Zoo
│   └── generate_sample_data.py # Sample data generator
├── tests/
│   ├── test_employee_loader.py
│   ├── test_face_recognizer.py
│   ├── test_image_processor.py
│   ├── test_ui.py
│   └── __init__.py
├── .github/workflows/ci.yml  # Automated GitHub Actions test runner
├── run_app.bat               # 1-click double-click desktop launcher (Windows)
├── run_app_debug.bat         # Launcher with console logs attached
├── AI_COLLAB.md              # Technical decision log & AI collaboration history
├── ARCHITECTURE.md           # Mermaid system architecture & design specifications
├── RESEARCH.md               # Model benchmarks, trade-off evaluations & comparisons
├── requirements.txt          # Python dependencies
├── main.py                   # Desktop application entry point
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Windows, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/MAHI2218/Who-Is-In-The-Photo.git
cd Who-Is-In-The-Photo
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Models (If Not Present)
If the ONNX models are not already downloaded in `models/`, run:
```bash
python scripts/download_models.py
```

---

## 🖥️ How to Run

### Option A: Double-Click Launcher (Windows)
Simply double-click **`run_app.bat`** inside the project folder.

### Option B: Terminal Command
```bash
python main.py
```

---

## 🧪 Step-by-Step Usage Guide

1. **Step 1: Select Reference Directory**  
   Click **"Select Reference Folder"** and select `sample_data/known_faces`. The application will index all 50 photos in the background and display a success status badge.
2. **Step 2: Upload Group Photograph**  
   Click **"Upload Group Photo"** and select `sample_data/group_photos/office_team_meeting_1.jpg`.
3. **Step 3: Set Threshold & Identify**  
   Adjust the **Similarity Threshold** slider (default: `0.36` – `0.40`), then click **"🔍 Identify People"**.
4. **Step 4: Inspect Results & Export**  
   - Inspect the green boxes on identified staff and orange boxes on "Unknown" guests.
   - Review the detailed breakdown cards in the results sidebar.
   - Click **"💾 Save Annotated Photo"** to export the annotated image.

---

## 🔬 Running Unit Tests

The test suite covers single-face validation, cosine similarity matching, duplicate prevention, group image processing, and PySide6 UI state management:

```bash
pytest -v
```

All 15 tests pass out-of-the-box:
```text
tests/test_employee_loader.py ....                                       [ 26%]
tests/test_face_recognizer.py ....                                       [ 53%]
tests/test_image_processor.py ....                                       [ 80%]
tests/test_ui.py ...                                                     [100%]
============================= 15 passed ==============================
```

---

## 👤 Author & Acknowledgments
- **Developer:** Mahendra ([@MAHI2218](https://github.com/MAHI2218))
- **Models:** [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet by Shi et al. / SFace by Zhong et al.)
- **GUI Framework:** [PySide6 (Qt for Python)](https://wiki.qt.io/Qt_for_Python)
