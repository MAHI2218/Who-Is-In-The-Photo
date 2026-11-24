# Full AI Collaboration Transcript & Session History

This log records the complete prompt-by-prompt human-AI collaboration trajectory for **Task A: Who Is in the Photo?**, starting from the initial problem breakdown to final deployment and verification.

---

## Session Trajectory

### Prompt 1: Requirements Breakdown & Technology Guidance
* **User:** "Read the assignment carefully. I am building Task A, Who Is in the Photo? I want to create an offline face recognition desktop application using Python. First, explain the assignment in simple language. Tell me the features we need to build and suggest a simple technology stack. Do not write any code yet. I want to understand the project before starting."
* **AI:** Outlined the core constraints (50 reference portraits, ~10 person group photo, offline runtime, Unknown detection), suggested lightweight models (YuNet + SFace or SCRFD + ArcFace), and proposed UI options (PySide6 vs. CustomTkinter).
* **Decision:** Selected PySide6 + OpenCV native DNN classes (`FaceDetectorYN` and `FaceRecognizerSF`).

### Prompt 2: Project Architecture & Phasing
* **User:** "Do not implement the application yet. Just create the structure and explain what each file is used for."
* **AI:** Structured the modular layout (`app/`, `models/`, `sample_data/`, `docs/`, `tests/`) and initialized foundational configuration files.
* **Decision:** Strict modular separation between computer vision core and UI presentation.

### Prompt 3: Environment Setup
* **User:** "Set up a Python virtual environment for this project. I am using Windows. Create a virtual environment named .venv and install the required packages for PySide6, OpenCV, ONNX Runtime, NumPy, Pillow, and Pytest. Add the packages to requirements.txt. Give me the exact terminal commands and explain how to activate the environment. Do not install unnecessary packages."
* **AI:** Created `.venv` on Python 3.11, configured `requirements.txt`, and verified package installations.

### Prompt 4: Pretrained Models & Offline Verification
* **User:** "I want to use YuNet for face detection and SFace for face recognition. Help me set up both pretrained ONNX models locally. Download the correct model files into the models folder. Make sure the models can run on my laptop without internet. Check that both model files load successfully using Python. Also explain where the models came from and check their licenses and usage restrictions. Do not use any online API or cloud service."
* **AI:** Placed `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from OpenCV Zoo into `models/`, verified offline loading, and documented MIT & Apache-2.0 licenses in `RESEARCH.md`.

### Prompt 5: Reference Directory Ingestion & Validation
* **User:** "Create a Python file named employee_loader.py inside the app folder. The program should allow the user to select a folder containing employee photographs. Read all JPG, JPEG, and PNG images from that folder. Extract the person's name from the filename without the file extension. For example, Priya Sharma.jpg should give the name Priya Sharma. Check each image and report if it has no face or more than one face. Store the valid employee face information for later recognition. Handle invalid files properly and explain the code."
* **AI:** Implemented `app/employee_loader.py` with `EmployeeRecord` and `LoaderReport`, strictly enforcing that each photo contains exactly one face. Added 4 unit tests in `tests/test_employee_loader.py`.

### Prompt 6: Biometric Matching & Duplicate Prevention
* **User:** "Now implement face recognition using the SFace model. Create a Python file named face_recognizer.py. The program should convert each employee face into a numerical face embedding, convert each detected group photo face into an embedding, compare the embeddings using cosine similarity, find the closest matching employee, and return the person's name and matching score. If the similarity is below a configurable threshold, return Unknown. Do not force a match when the face does not belong to any employee. Avoid identifying the same face multiple times. Write simple code with comments and explain how the matching works. Keep the threshold configurable so we can test and improve it later."
* **AI:** Implemented `app/face_recognizer.py` with vectorized matrix multiplication, calibrated thresholding, and greedy global best-match assignment to eliminate duplicate identities.

### Prompt 7: Group Photograph Processor & Visual Annotator
* **User:** "Now combine the face detector, employee loader, and face recognizer. Create image_processor.py. It should accept a group photograph, detect all faces, compare every detected face with the employee database, identify matching employees, mark unmatched people as Unknown, draw rectangles around the faces, write the person's name above each rectangle, and return the final annotated image and a list of recognition results. Make sure every detected face is processed separately. Handle cases where no faces are detected. Test the complete process."
* **AI:** Implemented `app/image_processor.py` with resolution-adaptive typography, bounding box offsets, and contrast-preserving pills (Emerald Green for matched staff, Amber for Unknown).

### Prompt 8: Desktop GUI Application
* **User:** "Now create the user interface using PySide6. I want a clean and modern desktop application called 'Who Is in the Photo?' The main screen should have: a button to select the employee folder, a message showing how many employee photographs were loaded, a button to upload a group photograph, a button named Identify People, a large area to display the group photograph, a results section showing each detected person’s name, Unknown labels for people who are not recognized, and a button to save the final annotated photograph. Use a simple professional design with good spacing, readable text, and clear buttons. Do not use any online services. Keep the interface separate from the face recognition logic."
* **AI:** Built `app/ui_main_window.py` with modern dark-slate UI, non-blocking `QThread` workers, threshold slider, responsive image viewport, and detailed results sidebar.

### Prompt 9: Error Dialogs & Pre-condition Guards
* **User:** "Now connect the PySide6 interface with all the face recognition modules. When the user selects an employee folder, load the employee photographs. When the user selects a group photograph, display it on the screen. When the user clicks Identify People, run face detection and face recognition. Display the names on the photograph and show all results in a separate list. Show Unknown for unmatched people. Add proper error messages if the folder or image is invalid. Make sure the application does not freeze while processing a large photograph. Test the complete application."
* **AI:** Added defensive modals for corrupt files, empty directories, and missing faces. Expanded test suite to 15 passing tests (`tests/test_ui.py`).

### Prompt 10: Test Dataset Generation
* **User:** "can u generate 50 images and give each image name and save them in a folder and if u give me some group photos containing some people among 50 persons and some without them for testing can u do it"
* **AI:** Created `scripts/generate_sample_data.py`, downloaded 50 single-face portraits with realistic names into `sample_data/known_faces/`, plus 8 unlisted individuals, and generated realistic group photos with mixed known and unknown individuals into `sample_data/group_photos/`.

### Prompt 11: Calibration & Threshold Clarification
* **User:** "i didnt included the man right side to mahendra in known faces 51 photos but the model verfiying him as david kim what to do"
* **User:** "why wrong detection is happening"
* **AI:** Diagnosed baseline cosine similarity noise (generic human facial structure produces 20%–35% similarity). Explained how the threshold slider acts as a gatekeeper and demonstrated that raising the threshold from 0.30 to 0.40 eliminates the false match while keeping Mahendra (82.7%) identified.

### Prompt 12: Offline Standalone Launchers
* **User:** "how to run this application everytime without opening antigravity and offline"
* **AI:** Created `run_app.bat` and `run_app_debug.bat` for one-click launching via `pythonw.exe` without terminal clutter.

### Prompt 13: GitHub Repository & CI/CD
* **User:** "this is my Github account MAHI2218"
* **AI:** Initialized Git repository, staged all 79 files, added GitHub Actions CI pipeline (`.github/workflows/ci.yml`), and pushed `main` branch to `https://github.com/MAHI2218/Who-Is-In-The-Photo`.

### Prompt 14: Archify Architecture Visualization
* **Requirement:** Install Archify per assignment specification (`https://raw.githubusercontent.com/tt-a1i/archify/main/llms.txt`), author the project architecture candidate, and render standalone interactive HTML and SVG diagrams.
* **Execution:** Installed Archify skill, rendered `docs/architecture/system_architecture.html` and dual-theme SVG diagrams, and linked them into `ARCHITECTURE.md`.
