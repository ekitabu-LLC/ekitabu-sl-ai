"""
Extract test video features using EXACT same method as training data.
This ensures train/test consistency.

Usage:
    modal run extract_test_features.py
"""

import modal
import os

app = modal.App("ksl-test-extractor")
volume = modal.Volume.from_name("ksl-dataset-vol")

# EXACT same image as training extraction
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless", "numpy")
)

# Same face/lip indices as extract_features_v2.py
LIPS_INDICES = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
    37, 39, 40, 185, 76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 292
]))[:40]

LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]
NOSE = [1, 2, 98, 327]
FACE_INDICES = LIPS_INDICES[:40] + LEFT_EYEBROW[:5] + RIGHT_EYEBROW[:5] + LEFT_EYE[:6] + RIGHT_EYE[:6] + NOSE[:4] + [10, 152]


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def extract_single_video(video_bytes: bytes, filename: str, class_name: str, category: str) -> str:
    """Extract features from a single test video - SAME as training extraction."""
    import cv2
    import numpy as np
    import mediapipe as mp

    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    # EXACT same holistic config as training
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

        # 1. Pose (99 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 99)

        # 2. Left hand (63 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 3. Right hand (63 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 4. Face key points (204 values)
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

        # 5. Lips (120 values)
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

    # Save to testing_v2 folder
    output_dir = f"/data/testing_v2/{category}/{class_name}"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    output_path = f"{output_dir}/{base_name}.npy"
    np.save(output_path, np.array(frames_data, dtype=np.float32))

    return f"{category}/{class_name}/{filename} -> {len(frames_data)} frames, 549 features"


@app.local_entrypoint()
def main():
    """Extract features from local test videos."""
    import os

    print("=" * 70)
    print("Test Video Feature Extraction")
    print("Using EXACT same method as training data (549 features)")
    print("=" * 70)

    test_dir = "testing"
    tasks = []

    # Numbers
    numbers_dir = os.path.join(test_dir, "Numbers")
    if os.path.exists(numbers_dir):
        print("\nNumbers:")
        for fn in sorted(os.listdir(numbers_dir)):
            if fn.endswith(".mov"):
                class_name = fn.replace("No. ", "").replace(".mov", "")
                video_path = os.path.join(numbers_dir, fn)
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                tasks.append((video_bytes, fn, class_name, "Numbers"))
                print(f"  {fn} -> class: {class_name}")

    # Words
    words_dir = os.path.join(test_dir, "Words - Left Side")
    if os.path.exists(words_dir):
        print("\nWords:")
        for fn in sorted(os.listdir(words_dir)):
            if fn.endswith(".mov"):
                if ". " in fn:
                    class_name = fn.split(". ", 1)[1].replace(".mov", "")
                else:
                    class_name = fn.replace(".mov", "")
                video_path = os.path.join(words_dir, fn)
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                tasks.append((video_bytes, fn, class_name, "Words"))
                print(f"  {fn} -> class: {class_name}")

    print(f"\nTotal videos to extract: {len(tasks)}")
    print("\nProcessing on Modal...")

    results = list(extract_single_video.starmap(tasks))
    for r in results:
        print(f"  {r}")

    # Commit volume changes
    volume.commit()

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("Test features saved to /data/testing_v2/")
    print("=" * 70)
