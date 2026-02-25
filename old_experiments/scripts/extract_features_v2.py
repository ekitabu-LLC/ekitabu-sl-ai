"""
KSL Enhanced Feature Extraction v2
Extracts comprehensive landmarks including face, lips, and hands.

Features extracted per frame:
- Pose landmarks: 33 points × 3 coords = 99
- Left hand: 21 points × 3 coords = 63
- Right hand: 21 points × 3 coords = 63
- Face (selected key points): 68 points × 3 coords = 204
- Lips: 40 points × 3 coords = 120
Total: 549 features per frame

Key face landmarks selected:
- Lips (inner and outer contour): 40 points
- Eyebrows: 10 points each = 20 points
- Eyes: 12 points each = 24 points
- Nose: 4 points

Usage:
    modal run extract_features_v2.py
"""

import modal
import os

app = modal.App("ksl-feature-extractor-v2")
volume = modal.Volume.from_name("ksl-dataset-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless", "numpy")
)

ALL_CLASSES = [
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444",
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]

# MediaPipe Face Mesh landmark indices for sign language relevant features
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Lips - outer contour (16 points)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
# Lips - inner contour (8 points)
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]
# Upper lip top
LIPS_UPPER = [37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 269, 267, 0, 37]
# Lower lip bottom
LIPS_LOWER = [84, 17, 314, 405, 321, 375, 291, 61, 146, 91, 181]

# Combined unique lip indices (40 points)
LIPS_INDICES = list(set([
    # Outer lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    # Inner lips
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
    # Additional lip corners and midpoints
    37, 39, 40, 185, 76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 292
]))[:40]

# Eyebrows (10 points each)
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]

# Eyes (12 points each)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]

# Nose (key points)
NOSE = [1, 2, 98, 327]

# All face indices for extraction (68 points total)
FACE_INDICES = LIPS_INDICES[:40] + LEFT_EYEBROW[:5] + RIGHT_EYEBROW[:5] + LEFT_EYE[:6] + RIGHT_EYE[:6] + NOSE[:4] + [10, 152]  # + forehead and chin


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def extract_video_features(video_bytes: bytes, filename: str, class_name: str, output_dir: str = "/data/train_v2") -> str:
    """
    Extract comprehensive landmarks from a single video.

    Features per frame (549 total):
    - Pose: 33 × 3 = 99
    - Left hand: 21 × 3 = 63
    - Right hand: 21 × 3 = 63
    - Face key points: 68 × 3 = 204
    - Lips detail: 40 × 3 = 120
    """
    import cv2
    import numpy as np
    import mediapipe as mp

    # Save video temporarily
    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    # Initialize MediaPipe Holistic (includes face mesh)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True  # Enable detailed face mesh
    )

    # Process video
    cap = cv2.VideoCapture(temp_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frame_landmarks = []

        # 1. Pose landmarks (33 points × 3 = 99 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 99)

        # 2. Left hand landmarks (21 points × 3 = 63 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 3. Right hand landmarks (21 points × 3 = 63 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 4. Face key points (68 points × 3 = 204 values)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in FACE_INDICES[:68]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks.extend([0.0, 0.0, 0.0])
        else:
            frame_landmarks.extend([0.0] * 204)

        # 5. Detailed lips (40 points × 3 = 120 values)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in LIPS_INDICES[:40]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks.extend([0.0, 0.0, 0.0])
        else:
            frame_landmarks.extend([0.0] * 120)

        frames_data.append(frame_landmarks)

    cap.release()
    holistic.close()
    os.remove(temp_path)

    # Save features
    out_dir = f"{output_dir}/{class_name}"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/{os.path.splitext(filename)[0]}.npy"
    np.save(output_path, np.array(frames_data, dtype=np.float32))

    return f"Extracted {filename} -> {len(frames_data)} frames, {len(frame_landmarks)} features"


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def extract_val_video_features(video_bytes: bytes, filename: str, class_name: str, signer: str) -> str:
    """Extract features for validation videos."""
    import cv2
    import numpy as np
    import mediapipe as mp

    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )

    cap = cv2.VideoCapture(temp_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frame_landmarks = []

        # Pose (99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 99)

        # Left hand (63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # Right hand (63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # Face key points (204)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in FACE_INDICES[:68]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks.extend([0.0, 0.0, 0.0])
        else:
            frame_landmarks.extend([0.0] * 204)

        # Lips detail (120)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in LIPS_INDICES[:40]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks.extend([0.0, 0.0, 0.0])
        else:
            frame_landmarks.extend([0.0] * 120)

        frames_data.append(frame_landmarks)

    cap.release()
    holistic.close()
    os.remove(temp_path)

    output_dir = f"/data/val_v2/{class_name}"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    output_path = f"{output_dir}/{signer}_{base_name}.npy"
    np.save(output_path, np.array(frames_data, dtype=np.float32))

    return f"Extracted {signer}/{class_name}/{filename} -> {len(frames_data)} frames"


@app.local_entrypoint()
def main(mode: str = "train"):
    """
    Extract features from videos.

    Args:
        mode: "train", "val", or "both"
    """
    import os
    import re

    print("=" * 70)
    print("KSL Enhanced Feature Extraction v2")
    print("Features: Pose + Hands + Face + Lips = 549 dims per frame")
    print("=" * 70)

    if mode in ["train", "both"]:
        print("\n--- Training Data ---")
        dataset_dir = "dataset"
        tasks = []

        for class_name in ALL_CLASSES:
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"  Warning: {class_name} not found")
                continue

            video_count = 0
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith((".mov", ".mp4", ".avi")):
                    video_path = os.path.join(class_path, video_file)
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    tasks.append((video_bytes, video_file, class_name, "/data/train_v2"))
                    video_count += 1
            print(f"  {class_name}: {video_count} videos")

        print(f"\nTotal training videos: {len(tasks)}")
        print("Processing on Modal...")

        results = list(extract_video_features.starmap(tasks))
        for r in results[:5]:
            print(f"  {r}")
        print(f"  ... and {len(results) - 5} more")
        print(f"\nTraining extraction complete! Saved to /data/train_v2/")

    if mode in ["val", "both"]:
        print("\n--- Validation Data ---")
        val_dir = "validation-data"
        tasks = []

        for signer_folder in sorted(os.listdir(val_dir)):
            signer_path = os.path.join(val_dir, signer_folder)
            if not os.path.isdir(signer_path):
                continue

            signer_match = re.match(r"(\d+)\.\s*Signer", signer_folder)
            if not signer_match:
                continue

            signer_id = f"signer{signer_match.group(1)}"
            print(f"\n{signer_folder}:")

            for class_folder in sorted(os.listdir(signer_path)):
                class_path = os.path.join(signer_path, class_folder)
                if not os.path.isdir(class_path):
                    continue

                # Parse class name
                if class_folder.startswith("No."):
                    class_name = class_folder.replace("No.", "").strip()
                else:
                    match = re.match(r"^\d+\.\s*(.+)$", class_folder)
                    class_name = match.group(1).strip() if match else class_folder

                video_count = 0
                for video_file in os.listdir(class_path):
                    if video_file.lower().endswith((".mov", ".mp4", ".avi")):
                        video_path = os.path.join(class_path, video_file)
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        tasks.append((video_bytes, video_file, class_name, signer_id))
                        video_count += 1
                print(f"  {class_folder} -> {class_name}: {video_count} videos")

        print(f"\nTotal validation videos: {len(tasks)}")
        print("Processing on Modal...")

        results = list(extract_val_video_features.starmap(tasks))
        print(f"\nValidation extraction complete! Saved to /data/val_v2/")

    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("New feature dimension: 549 (pose:99 + hands:126 + face:204 + lips:120)")
    print("=" * 70)
