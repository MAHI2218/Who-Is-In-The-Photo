# Offline Pretrained Models (`models/`)

This directory stores the official, offline ONNX model weights for face detection and face recognition utilized by the application.

---

## 1. Installed Models

### A. YuNet (Face Detection & Landmarks)
* **File:** `face_detection_yunet_2023mar.onnx`
* **Size:** ~227 KB
* **Architecture:** Lightweight CNN designed for edge real-time face detection
* **Outputs:** 
  * Face bounding boxes: `[x, y, w, h]`
  * 5 facial landmarks: Left eye, right eye, nose tip, right mouth corner, left mouth corner
  * Confidence score
* **Source:** [OpenCV Model Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
* **Paper / Research:** *A Fast and Compact Face Detector with Nanosecond Speed* (Shi et al.)
* **License:** MIT License (Permissive open source)

### B. SFace (Face Feature Extraction & Alignment)
* **File:** `face_recognition_sface_2021dec.onnx`
* **Size:** ~36.9 MB
* **Architecture:** ResNet-based deep feature extractor trained with Sigmoid-Constrained Hypersphere Loss
* **Outputs:** 128-dimensional normalized facial embedding vector
* **Source:** [OpenCV Model Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
* **Paper / Research:** *Accurate and Lightweight Face Recognition with SFace* (Zhong et al., IEEE Transactions on Biometrics, Behavior, and Identity Science 2021)
* **License:** Apache License 2.0 (Permissive open source)

---

## 2. 100% Offline Runtime Verification
Both models run fully locally on CPU without requiring an internet connection, GPU hardware, external accounts, or cloud APIs.

Verification test executed:
```bash
python -c "import cv2; detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar.onnx', '', (320, 320)); recognizer = cv2.FaceRecognizerSF.create('models/face_recognition_sface_2021dec.onnx', ''); print('Models loaded successfully!')"
```
