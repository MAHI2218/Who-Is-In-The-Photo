"""
Sample Data Generation Script for 'Who Is In The Photo?'
Downloads 50 reference portraits with full names into sample_data/known_faces/
and composes realistic group photos into sample_data/group_photos/ containing
a mix of known reference individuals and unknown individuals.
"""

import os
import sys
import time
import random
from pathlib import Path
import urllib.request
import urllib.error
import cv2
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWN_FACES_DIR = PROJECT_ROOT / "sample_data" / "known_faces"
GROUP_PHOTOS_DIR = PROJECT_ROOT / "sample_data" / "group_photos"
UNKNOWN_CACHE_DIR = PROJECT_ROOT / "sample_data" / ".cache_unknown"

# 50 Realistic Names (25 Male, 25 Female)
MALE_NAMES = [
    "Aarav Patel", "Alexander Wright", "Carlos Mendez", "David Kim", "Ethan Davis",
    "James Wilson", "Liam Brown", "Lucas Garcia", "Marcus Johnson", "Michael Chang",
    "Noah Miller", "Oliver Smith", "Rahul Sharma", "Robert Taylor", "Ryan Thomas",
    "Samuel Jackson", "Sebastian Moore", "Vikram Malhotra", "William Anderson", "Zack Martin",
    "Brian Clark", "Daniel Lewis", "George Hall", "Henry White", "Jack Harris"
]

FEMALE_NAMES = [
    "Amara Singh", "Chloe Bennett", "Elena Rostova", "Emily Watson", "Grace Hopper",
    "Hannah Scott", "Isabella Rodriguez", "Jessica Alba", "Katherine Johnson", "Layla Hassan",
    "Maya Lin", "Mia Thompson", "Nadia Petrova", "Olivia Martin", "Pooja Hegde",
    "Priya Nair", "Rachel Green", "Sarah Connor", "Sophia Chen", "Tara Sharma",
    "Victoria Adams", "Zoe Saldana", "Alice Walker", "Clara Oswald", "Diana Prince"
]

UNKNOWN_NAMES = [
    "Unknown Guest 1", "Unknown Guest 2", "Unknown Guest 3", "Unknown Guest 4",
    "Unknown Guest 5", "Unknown Guest 6", "Unknown Guest 7", "Unknown Guest 8"
]


def download_image(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download an image with user-agent headers and retry logic."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 1000:  # Valid image payload
                    with open(dest_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Failed to download {url}: {e}")
    return False


def create_office_backdrop(width: int = 1600, height: int = 900) -> np.ndarray:
    """Creates a clean, realistic modern office/studio gradient backdrop."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Elegant dark-slate modern office backdrop with subtle lighting vignette
    y_coords, x_coords = np.indices((height, width))
    center_x, center_y = width // 2, height // 3
    
    # Radial falloff from center light
    dist = np.sqrt(((x_coords - center_x) / (width * 0.7)) ** 2 + ((y_coords - center_y) / (height * 0.6)) ** 2)
    dist = np.clip(dist, 0.0, 1.0)
    
    # Center color: warm corporate grey-blue (140, 145, 155), edges: deep slate (45, 48, 55)
    center_color = np.array([160, 155, 150], dtype=np.float32)  # BGR
    edge_color = np.array([55, 50, 45], dtype=np.float32)        # BGR
    
    for c in range(3):
        canvas[:, :, c] = (center_color[c] * (1.0 - dist) + edge_color[c] * dist).astype(np.uint8)
        
    # Add subtle wood/floor base bar at the bottom 15%
    floor_y = int(height * 0.85)
    canvas[floor_y:, :] = (canvas[floor_y:, :] * 0.6 + np.array([30, 35, 40], dtype=np.float32) * 0.4).astype(np.uint8)
    
    return canvas


def compose_group_photo(
    output_path: Path,
    people_records: list,  # list of tuples (img_bgr, name, is_known)
    canvas_w: int = 1600,
    canvas_h: int = 900
):
    """
    Composes a natural 2-row group photo containing 10 individuals.
    5 standing in back row, 5 seated/standing in front row.
    """
    canvas = create_office_backdrop(canvas_w, canvas_h)
    
    # 10 positions: 5 back row, 5 front row
    num_people = len(people_records)
    assert num_people == 10, f"Expected 10 people, got {num_people}"
    
    back_row = people_records[:5]
    front_row = people_records[5:]
    
    # Back row specs (slightly higher, slightly smaller)
    back_target_size = (200, 240)  # (w, h)
    back_y = int(canvas_h * 0.28)
    spacing_x = canvas_w // 6
    
    for i, (person_img, name, is_known) in enumerate(back_row):
        h, w = person_img.shape[:2]
        # Resize person image maintaining aspect ratio
        resized = cv2.resize(person_img, back_target_size, interpolation=cv2.INTER_LANCZOS4)
        rw, rh = back_target_size
        
        pos_x = int((i + 1) * spacing_x - rw // 2)
        pos_y = back_y
        
        # Add subtle soft shadow behind
        shadow_mask = np.zeros((rh + 20, rw + 20), dtype=np.float32)
        cv2.ellipse(shadow_mask, ((rw + 20)//2, (rh + 20)//2), (rw//2 + 5, rh//2 + 5), 0, 0, 360, 0.4, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Blend shadow onto canvas
        sy1, sy2 = max(0, pos_y - 10), min(canvas_h, pos_y + rh + 10)
        sx1, sx2 = max(0, pos_x - 10), min(canvas_w, pos_x + rw + 10)
        my1, my2 = 0, sy2 - sy1
        mx1, mx2 = 0, sx2 - sx1
        
        for c in range(3):
            canvas[sy1:sy2, sx1:sx2, c] = (
                canvas[sy1:sy2, sx1:sx2, c] * (1.0 - shadow_mask[my1:my2, mx1:mx2])
            ).astype(np.uint8)
        
        # Place resized person on canvas
        canvas[pos_y:pos_y + rh, pos_x:pos_x + rw] = resized
        
    # Front row specs (lower, slightly larger)
    front_target_size = (230, 270)  # (w, h)
    front_y = int(canvas_h * 0.48)
    front_offsets = [0.6, 1.5, 2.5, 3.5, 4.4]
    
    for i, (person_img, name, is_known) in enumerate(front_row):
        rw, rh = front_target_size
        resized = cv2.resize(person_img, front_target_size, interpolation=cv2.INTER_LANCZOS4)
        
        pos_x = int(front_offsets[i] * spacing_x)
        pos_y = front_y
        
        # Soft shadow behind
        shadow_mask = np.zeros((rh + 25, rw + 25), dtype=np.float32)
        cv2.ellipse(shadow_mask, ((rw + 25)//2, (rh + 25)//2), (rw//2 + 8, rh//2 + 8), 0, 0, 360, 0.45, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 0)
        
        sy1, sy2 = max(0, pos_y - 10), min(canvas_h, pos_y + rh + 15)
        sx1, sx2 = max(0, pos_x - 10), min(canvas_w, pos_x + rw + 15)
        my1, my2 = 0, sy2 - sy1
        mx1, mx2 = 0, sx2 - sx1
        
        for c in range(3):
            canvas[sy1:sy2, sx1:sx2, c] = (
                canvas[sy1:sy2, sx1:sx2, c] * (1.0 - shadow_mask[my1:my2, mx1:mx2])
            ).astype(np.uint8)
            
        canvas[pos_y:pos_y + rh, pos_x:pos_x + rw] = resized
        
    cv2.imwrite(str(output_path), canvas)
    print(f"Created group photo: {output_path} ({canvas_w}x{canvas_h})")


def main():
    print("=" * 65)
    print("WHO IS IN THE PHOTO? - SAMPLE DATA GENERATOR")
    print("=" * 65)
    
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    UNKNOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download 25 Men & 25 Women portraits from RandomUser.me
    print("\n[Step 1/3] Downloading 50 Reference Portraits from RandomUser.me...")
    known_portraits = {}  # name -> file_path
    
    # Men (1 to 25)
    for idx, name in enumerate(MALE_NAMES, start=1):
        file_path = KNOWN_FACES_DIR / f"{name}.jpg"
        if not file_path.exists():
            url = f"https://randomuser.me/api/portraits/men/{idx}.jpg"
            print(f"  Downloading [{idx}/50] {name}...")
            download_image(url, file_path)
            time.sleep(0.05)
        known_portraits[name] = file_path
        
    # Women (1 to 25)
    for idx, name in enumerate(FEMALE_NAMES, start=1):
        file_path = KNOWN_FACES_DIR / f"{name}.jpg"
        if not file_path.exists():
            url = f"https://randomuser.me/api/portraits/women/{idx}.jpg"
            print(f"  Downloading [{25 + idx}/50] {name}...")
            download_image(url, file_path)
            time.sleep(0.05)
        known_portraits[name] = file_path
        
    print(f"Successfully prepared 50 reference portraits in '{KNOWN_FACES_DIR.relative_to(PROJECT_ROOT)}'.")
    
    # 2. Download 8 Unknown Individuals (indices 26 to 29 men, 26 to 29 women)
    print("\n[Step 2/3] Downloading Unlisted / Unknown Individuals...")
    unknown_portraits = {}
    for idx, name in enumerate(UNKNOWN_NAMES[:4], start=26):
        file_path = UNKNOWN_CACHE_DIR / f"{name}.jpg"
        if not file_path.exists():
            url = f"https://randomuser.me/api/portraits/men/{idx}.jpg"
            download_image(url, file_path)
            time.sleep(0.05)
        unknown_portraits[name] = file_path
        
    for idx, name in enumerate(UNKNOWN_NAMES[4:], start=26):
        file_path = UNKNOWN_CACHE_DIR / f"{name}.jpg"
        if not file_path.exists():
            url = f"https://randomuser.me/api/portraits/women/{idx}.jpg"
            download_image(url, file_path)
            time.sleep(0.05)
        unknown_portraits[name] = file_path
        
    print(f"Prepared {len(unknown_portraits)} unknown portraits for test group photos.")
    
    # 3. Compose Group Photos
    print("\n[Step 3/3] Composing Realistic Group Photographs...")
    
    # --- Group Photo 1: 6 Known Employees + 4 Unknown Guests ---
    # Known: Aarav Patel, Alexander Wright, Carlos Mendez, Amara Singh, Chloe Bennett, Elena Rostova
    # Unknown: Unknown Guest 1, Unknown Guest 2, Unknown Guest 5, Unknown Guest 6
    known_selection_1 = [
        "Aarav Patel", "Alexander Wright", "Carlos Mendez",
        "Amara Singh", "Chloe Bennett", "Elena Rostova"
    ]
    unknown_selection_1 = [
        "Unknown Guest 1", "Unknown Guest 2",
        "Unknown Guest 5", "Unknown Guest 6"
    ]
    
    gp1_records = []
    # 5 in back row: 3 known + 2 unknown
    gp1_records.append((cv2.imread(str(known_portraits["Aarav Patel"])), "Aarav Patel", True))
    gp1_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 1"])), "Unknown Guest 1", False))
    gp1_records.append((cv2.imread(str(known_portraits["Carlos Mendez"])), "Carlos Mendez", True))
    gp1_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 5"])), "Unknown Guest 5", False))
    gp1_records.append((cv2.imread(str(known_portraits["Amara Singh"])), "Amara Singh", True))
    
    # 5 in front row: 3 known + 2 unknown
    gp1_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 2"])), "Unknown Guest 2", False))
    gp1_records.append((cv2.imread(str(known_portraits["Alexander Wright"])), "Alexander Wright", True))
    gp1_records.append((cv2.imread(str(known_portraits["Chloe Bennett"])), "Chloe Bennett", True))
    gp1_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 6"])), "Unknown Guest 6", False))
    gp1_records.append((cv2.imread(str(known_portraits["Elena Rostova"])), "Elena Rostova", True))
    
    gp1_path = GROUP_PHOTOS_DIR / "office_team_meeting_1.jpg"
    compose_group_photo(gp1_path, gp1_records)
    
    # --- Group Photo 2: 7 Known Employees + 3 Unknown Guests ---
    # Known: David Kim, Ethan Davis, James Wilson, Emily Watson, Grace Hopper, Hannah Scott, Isabella Rodriguez
    # Unknown: Unknown Guest 3, Unknown Guest 4, Unknown Guest 7
    gp2_records = []
    # 5 in back row: 4 known + 1 unknown
    gp2_records.append((cv2.imread(str(known_portraits["David Kim"])), "David Kim", True))
    gp2_records.append((cv2.imread(str(known_portraits["Emily Watson"])), "Emily Watson", True))
    gp2_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 3"])), "Unknown Guest 3", False))
    gp2_records.append((cv2.imread(str(known_portraits["Ethan Davis"])), "Ethan Davis", True))
    gp2_records.append((cv2.imread(str(known_portraits["Grace Hopper"])), "Grace Hopper", True))
    
    # 5 in front row: 3 known + 2 unknown
    gp2_records.append((cv2.imread(str(known_portraits["James Wilson"])), "James Wilson", True))
    gp2_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 4"])), "Unknown Guest 4", False))
    gp2_records.append((cv2.imread(str(known_portraits["Hannah Scott"])), "Hannah Scott", True))
    gp2_records.append((cv2.imread(str(unknown_portraits["Unknown Guest 7"])), "Unknown Guest 7", False))
    gp2_records.append((cv2.imread(str(known_portraits["Isabella Rodriguez"])), "Isabella Rodriguez", True))
    
    gp2_path = GROUP_PHOTOS_DIR / "office_team_meeting_2.jpg"
    compose_group_photo(gp2_path, gp2_records)
    
    print("\n" + "=" * 65)
    print("ALL SAMPLE DATA GENERATED SUCCESSFULLY!")
    print(f"- 50 Reference Portraits in: {KNOWN_FACES_DIR}")
    print(f"- 2 Group Photos in:       {GROUP_PHOTOS_DIR}")
    print("  1. office_team_meeting_1.jpg (6 Known Employees + 4 Unknown)")
    print("  2. office_team_meeting_2.jpg (7 Known Employees + 3 Unknown)")
    print("=" * 65)


if __name__ == "__main__":
    main()
