# Technical Research & Model Selection (`RESEARCH.md`)

This document details the preliminary research, evaluated technologies, model comparisons, and design decisions conducted for **Task A: Who Is in the Photo?**.

---

## 1. Problem Context & Constraints
* **Dataset Size:** 50 reference portraits (individual single faces) and 1 group photo containing ~10 faces.
* **Environment:** 100% offline execution on local machine. No cloud APIs, no external calls at runtime.
* **Key Technical Challenges:**
  * Accurate face detection in crowded/group scenes (variable lighting, tilt, occlusion).
  * High embedding discriminability to correctly match known identities.
  * Reliable distance thresholding to detect and label "Unknown" persons without false positives.
  * Rapid embedding generation (< 5-10 seconds total on standard laptop CPU).

---

## 2. Face Detection & Recognition Frameworks Compared

| Framework / Model | Detection Method | Recognition / Embeddings | Offline Performance (CPU) | Group Photo Reliability | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenCV DNN (YuNet + SFace)** | YuNet (ONNX) | SFace (128-dim embedding) | Ultra-fast (~15-30ms per face on CPU) | High (fast multi-scale edge CNN) | **Selected Primary Pipeline** |
| **InsightFace (SCRFD + ArcFace)** | SCRFD (ONNX) | ArcFace (512-dim embedding) | Fast (~60-100ms per face on CPU) | High | **Evaluated Alternative** |
| **`face_recognition` (dlib)** | HOG / dlib CNN | ResNet-34 (128-dim embedding) | Slow on CNN; HOG misses small faces | Low-to-Moderate | **Rejected** |
| **DeepFace (RetinaFace + Facenet512)** | RetinaFace | Facenet512 / ArcFace | Moderate-to-Slow | High | **Rejected** (heavy dependency overhead) |

### Detailed Evaluations

### A. OpenCV Zoo: YuNet + SFace (Selected Pipeline)
* **Website / Repo:** [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | [OpenCV Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
* **YuNet Paper:** *A Fast and Compact Face Detector with Nanosecond Speed* (Shi et al.)
* **SFace Paper:** *Accurate and Lightweight Face Recognition with SFace* (Zhong et al., IEEE Transactions on Biometrics, Behavior, and Identity Science 2021)
* **Why Picked:** 
  1. **Zero External Inference Overhead:** Native C++ acceleration via `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF` inside standard OpenCV.
  2. **Ultra-Lightweight:** YuNet model is only ~227 KB and SFace is ~36.9 MB, eliminating huge memory footprints.
  3. **High-Speed CPU Execution:** Detects and embeds a 10-person group photo in under 150ms on a consumer laptop CPU.
  4. **Built-in Face Alignment:** YuNet outputs 5 facial landmarks (eyes, nose, mouth corners) used directly by `recognizer.alignCrop()` for robust pose normalization.
  5. **Licensing:** Permissive open source (MIT for YuNet, Apache-2.0 for SFace).

### C. `face_recognition` (dlib)
* **Website / Repo:** [face_recognition GitHub](https://github.com/ageitgey/face_recognition)
* **Why Rejected:**
  1. Default HOG detector frequently misses faces that are turned sideways or distant in a group photo.
  2. The CNN detector is accurate but computationally too slow on a standard laptop CPU without CUDA.
  3. Windows installation requires compiling `dlib` with CMake and Visual Studio C++ build tools, which introduces installation fragility during assessment.

---

## 3. Desktop UI Framework Comparison

| UI Framework | Appearance / Customization | Native Windows Integration | Packaging / Footprint | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **PySide6 (Qt for Python)** | Professional, full widget set, graphics canvas for bounding boxes | Native, hardware-accelerated rendering | Moderate | **Selected** |
| **CustomTkinter** | Clean modern dark-mode, rounded buttons | Good, wrapper around Tk | Very lightweight | **Alternative Considered** |
| **Streamlit / Local Web UI** | Web-based dashboard | Runs in browser tab | Requires local server process | **Rejected** (User preferred native desktop app) |

* **Why PySide6 was selected:** PySide6 provides robust high-DPI scaling, precise image coordinate transformation (essential for drawing exact bounding boxes over resized images), and multi-threaded background worker signals (`QThread`) so the UI never freezes while indexing 50 reference photos.

---

## 4. Matching & Thresholding Strategy
1. **Metric:** Cosine Similarity between 128-dimensional normalized SFace embeddings.
   $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
2. **Threshold Tuning:**
   * Cosine similarity $\ge 0.363$ (default baseline) $\rightarrow$ Positive match (person identified).
   * Cosine similarity $< 0.363$ $\rightarrow$ Unmatched (**"Unknown"**).
   * The application exposes an interactive threshold slider in the GUI (`0.20` to `0.70`) to demonstrate precision-recall trade-offs during live review.
