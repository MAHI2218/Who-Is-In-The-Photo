"""
Pretrained Model Downloader for 'Who Is In The Photo?'
Downloads official OpenCV Zoo YuNet (face detection) and SFace (face recognition)
ONNX models into the models/ directory.
"""

import os
from pathlib import Path
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Checking and downloading required ONNX models...")
    for filename, url in MODELS.items():
        dest = models_dir / filename
        if dest.exists() and dest.stat().st_size > 10000:
            print(f" [OK] Already present: {filename} ({dest.stat().st_size / (1024*1024):.2f} MB)")
            continue

        print(f" Downloading {filename} from OpenCV Zoo...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f" [SUCCESS] Downloaded: {filename} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f" [ERROR] Failed to download {filename}: {e}")

    print("\nModel verification complete.")

if __name__ == "__main__":
    main()
