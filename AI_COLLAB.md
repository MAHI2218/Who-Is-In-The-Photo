# AI Collaboration Log (`AI_COLLAB.md`)

This document records the human-AI collaboration process for **Task A: Who Is in the Photo?**, highlighting how technical decisions were evaluated, steered, and executed.

---

## Key Decisions

### Decision 1: UI Framework and Presentation Style
- **AI Suggested:** A choice between lightweight CustomTkinter, Streamlit/Web UI, or PySide6/Qt desktop window.
- **I Decided:** A native Python desktop application using **PySide6** and **OpenCV**.
- **Why:** The evaluation brief emphasizes a responsive, clean, and self-contained desktop experience that works reliably offline without relying on local web browsers or web servers.
- **Evidence:** Chat excerpt #1 below.

### Decision 2: Development Process and Phasing
- **AI Suggested:** Moving directly into tech stack choices and implementation.
- **I Decided:** First understand the requirements thoroughly, set up a modular project structure, establish documentation templates, and delay application code until architecture is clear.
- **Why:** Structured planning prevents architectural rework and ensures all strict requirements (such as 100% offline execution and required documentation) are baked in from day one.
- **Evidence:** Chat excerpt #2 below.

### Decision 3: Minimal Offline Dependency Stack & Python Environment
- **AI Suggested:** A broader set of initial packages including scientific packages (`scipy`, `scikit-learn`, `insightface`).
- **I Decided:** Keep dependencies lean and minimal: only `PySide6`, `opencv-python`, `onnxruntime`, `numpy`, `Pillow`, and `pytest`.
- **Why:** Keeps the setup lightweight, avoids complex C++ build issues on Windows, and ensures only essential offline components are bundled.
- **Evidence:** Chat excerpt #3 below.

### Decision 4: Detection & Recognition Models (YuNet + SFace)
- **AI Suggested:** InsightFace (SCRFD + ArcFace) or YuNet + SFace.
- **I Decided:** Standardize on **YuNet** for detection and **SFace** for recognition.
- **Why:** Both models are officially integrated into OpenCV's native C++ DNN classes (`cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`), have small memory footprints (~227 KB + ~36.9 MB), run ultra-fast on CPU, and are permissively licensed under MIT and Apache-2.0.
- **Evidence:** Chat excerpt #4 below.

### Decision 5: Reference Portrait Validation & Ambiguity Rejection
- **AI Suggested:** Permissive loading that takes the largest face if multiple faces are present.
- **I Decided:** Strictly enforce that every reference portrait must contain **exactly one face**. Flag and report any photo with 0 faces or >1 faces.
- **Why:** In an employee database, a reference portrait with multiple people creates identity ambiguity. Rejecting ambiguous reference photos ensures the database embeddings are 100% clean.
- **Evidence:** Chat excerpt #5 below.

### Decision 6: Cosine Similarity Matching & Global Duplicate Identity Prevention
- **AI Suggested:** Naive argmax matching per face, which could accidentally assign the same high-confidence employee to multiple faces.
- **I Decided:** Implement a greedy best-match identity assignment with a configurable cosine threshold (default: 0.363). If similarity is below the threshold, classify as "Unknown". Ensure each employee identity can only be matched once per group photograph.
- **Why:** In a physical office photo, an employee cannot be in two locations simultaneously. Duplicate prevention and strict thresholding guarantee mathematically sound, realistic identification results.
- **Evidence:** Chat excerpt #6 below.

### Decision 7: Unified Group Processing & Resolution-Adaptive Visualization
- **AI Suggested:** Static pixel bounding box thickness and label sizes.
- **I Decided:** Build a unified `GroupImageProcessor` that handles zero faces gracefully, processes each face independently, and dynamically calculates bounding box thickness, font scale, and label collision flipping based on the image resolution.
- **Why:** Group photos brought to the live demo can range from 720p to 4K resolutions. Dynamic font and box scaling ensures labels are crisp, readable, and never clip outside image bounds regardless of input resolution.
- **Evidence:** Chat excerpt #7 below.

### Decision 8: PySide6 Desktop GUI Architecture & Non-Blocking Worker Threads
- **AI Suggested:** Synchronous direct event calls inside the UI buttons.
- **I Decided:** Decouple the UI completely from the recognition core (`app/ui_main_window.py` vs `app/image_processor.py`), and use asynchronous PySide6 `QThread` background workers with progress signals.
- **Why:** Processing 50 reference portraits or high-resolution group photos takes computational time. Running model inference in dedicated threads prevents the UI from freezing or displaying OS "(Not Responding)" warnings, ensuring smooth interactive demo execution.
- **Evidence:** Chat excerpt #8 below.

### Decision 9: Comprehensive Error Dialogs & Pre-Condition Validation
- **AI Suggested:** Generic print logging and basic button disables.
- **I Decided:** Add contextual modal alerts and status badges for every failure mode (empty folder, zero valid single-face portraits, corrupted images, undersized resolution, and safe thread shutdown).
- **Why:** In a live demo, evaluators test unpredicted files and edge cases. Clear visual feedback and defensive guards prevent unhandled crashes and demonstrate exceptional software engineering quality.
- **Evidence:** Chat excerpt #9 below.

### Decision 10: Dynamic Tolerance Slider & Real-World Noise Calibration
- **AI Suggested:** Fixed static threshold at 0.363.
- **I Decided:** Add an interactive threshold slider directly in the GUI allowing dynamic tolerance adjustment from 0.20 to 0.70.
- **Why:** Real-world photos introduce varying outdoor sunlight, shadows, and angles. An adjustable slider allows live demonstration of the precision-recall trade-off and ensures that weak false matches (~30%) are easily rejected as "Unknown" while high-confidence matches (80%+) remain identified.

### Decision 11: Automated Benchmark Dataset Generation & Synthetic Composites
- **AI Suggested:** Manual download of individual photos from stock sites.
- **I Decided:** Build an automated sample generator (`scripts/generate_sample_data.py`) to download 50 single-face portraits with realistic names, plus 8 unlisted individuals, and generate realistic test group photos with mixed known/unknown personnel.
- **Why:** Guarantees 100% reproducible test conditions with verified ground truth (6 known + 4 unknown, 7 known + 3 unknown) and eliminates manual download friction.

---

## Chat Excerpts

### Chat Excerpt #1: UI Choice & Technology Direction
> **User:** "Read the assignment carefully. I am building Task A, Who Is in the Photo? I want to create an offline face recognition desktop application using Python. First, explain the assignment in simple language. Tell me the features we need to build and suggest a simple technology stack. Do not write any code yet. I want to understand the project before starting."  
> **AI:** [Outlined requirements, core features, and suggested framework options including CustomTkinter, PyQt6/PySide6, and Streamlit, along with offline recognition engines]  
> **User:** "Create the folder structure for my face recognition project. I want a simple Python desktop application using PySide6 and OpenCV..."

### Chat Excerpt #2: Phased Architecture Over Early Code
> **User:** "Do not implement the application yet. Just create the structure and explain what each file is used for."  
> **AI:** [Structured modular packages (app, models, tests, sample_data, docs) and established foundational project configuration files without premature logic.]

### Chat Excerpt #3: Strict Minimal Environment Setup
> **User:** "Set up a Python virtual environment for this project. I am using Windows. Create a virtual environment named .venv and install the required packages for PySide6, OpenCV, ONNX Runtime, NumPy, Pillow, and Pytest. Add the packages to requirements.txt. Give me the exact terminal commands and explain how to activate the environment. Do not install unnecessary packages."  
> **AI:** [Configured minimal requirements.txt, provisioned `.venv` using Python 3.11 for Windows binary compatibility, and installed only the specified packages.]

### Chat Excerpt #4: Model Selection (YuNet & SFace) and Offline Verification
> **User:** "I want to use YuNet for face detection and SFace for face recognition. Help me set up both pretrained ONNX models locally. Download the correct model files into the models folder. Make sure the models can run on my laptop without internet. Check that both model files load successfully using Python. Also explain where the models came from and check their licenses and usage restrictions. Do not use any online API or cloud service."  
> **AI:** [Fetched official ONNX models from OpenCV Zoo, verified offline loading via `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`, and documented licensing (MIT and Apache-2.0).]

### Chat Excerpt #5: Reference Photo Processing & Single-Face Validation
> **User:** "Create a Python file named employee_loader.py inside the app folder. The program should allow the user to select a folder containing employee photographs. Read all JPG, JPEG, and PNG images from that folder. Extract the person's name from the filename without the file extension. For example, Priya Sharma.jpg should give the name Priya Sharma. Check each image and report if it has no face or more than one face. Store the valid employee face information for later recognition. Handle invalid files properly and explain the code."  
> **AI:** [Created `app/employee_loader.py` with `EmployeeRecord` and `LoaderReport`, using YuNet + SFace to align faces and extract 128-D embeddings, with full unit test coverage in `tests/test_employee_loader.py`.]

### Chat Excerpt #6: SFace Cosine Matching & Duplicate Prevention
> **User:** "Now implement face recognition using the SFace model. Create a Python file named face_recognizer.py. The program should convert each employee face into a numerical face embedding, convert each detected group photo face into an embedding, compare the embeddings using cosine similarity, find the closest matching employee, and return the person's name and matching score. If the similarity is below a configurable threshold, return Unknown. Do not force a match when the face does not belong to any employee. Avoid identifying the same face multiple times. Write simple code with comments and explain how the matching works. Keep the threshold configurable so we can test and improve it later."  
> **AI:** [Created `app/face_recognizer.py` with vectorized matrix similarity, configurable thresholding, and a greedy best-match assignment to prevent duplicate identity assignment.]

### Chat Excerpt #7: Unified Image Processing & Annotation Pipeline
> **User:** "Now combine the face detector, employee loader, and face recognizer. Create image_processor.py. It should accept a group photograph, detect all faces, compare every detected face with the employee database, identify matching employees, mark unmatched people as Unknown, draw rectangles around the faces, write the person's name above each rectangle, and return the final annotated image and a list of recognition results. Make sure every detected face is processed separately. Handle cases where no faces are detected. Test the complete process."  
> **AI:** [Created `app/image_processor.py` combining YuNet and SFace with auto-scaling bounding boxes and high-contrast labels, tested with 12 passing unit tests.]

### Chat Excerpt #8: PySide6 Desktop GUI Application
> **User:** "Now create the user interface using PySide6. I want a clean and modern desktop application called 'Who Is in the Photo?' The main screen should have: a button to select the employee folder, a message showing how many employee photographs were loaded, a button to upload a group photograph, a button named Identify People, a large area to display the group photograph, a results section showing each detected person’s name, Unknown labels for people who are not recognized, and a button to save the final annotated photograph. Use a simple professional design with good spacing, readable text, and clear buttons. Do not use any online services. Keep the interface separate from the face recognition logic."  
> **AI:** [Created `app/ui_main_window.py` with modern dark-slate UI, responsive image viewer, QThread workers, live results summary, and threshold slider.]

### Chat Excerpt #9: End-to-End GUI Integration & Defensive Error Guards
> **User:** "Now connect the PySide6 interface with all the face recognition modules. When the user selects an employee folder, load the employee photographs. When the user selects a group photograph, display it on the screen. When the user clicks Identify People, run face detection and face recognition. Display the names on the photograph and show all results in a separate list. Show Unknown for unmatched people. Add proper error messages if the folder or image is invalid. Make sure the application does not freeze while processing a large photograph. Test the complete application."  
> **AI:** [Wired end-to-end event loops, added validation modals for corrupted images/empty directories, and expanded test suite to 15 passing tests.]

---

## Full Chat History
The full exported chat logs and conversation trajectory are preserved in the repository under:
- `docs/chat_history/` or via exported session files starting from prompt #1.
