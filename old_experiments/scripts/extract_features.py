"""
KSL Feature Extraction Script
Extracts MediaPipe Holistic landmarks from training videos.
Runs on Modal with parallel processing.
"""

import modal
import os

# Modal app setup
app = modal.App("ksl-feature-extractor")
volume = modal.Volume.from_name("ksl-dataset-vol", create_if_missing=True)

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless", "numpy")
)

# All 30 classes (15 numbers + 15 words)
ALL_CLASSES = [
    # Numbers
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444",
    # Words
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def extract_video_features(video_bytes: bytes, filename: str, class_name: str) -> str:
    """
    Extract pose and hand landmarks from a single video.

    Features extracted per frame (225 total):
    - Pose landmarks: 33 points × 3 coords = 99
    - Left hand: 21 points × 3 coords = 63
    - Right hand: 21 points × 3 coords = 63
    """
    import cv2
    import numpy as np
    import mediapipe as mp

    # Save video temporarily
    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Process video
    cap = cv2.VideoCapture(temp_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frame_landmarks = []

        # Extract pose landmarks (33 points × 3 = 99 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 99)

        # Extract left hand landmarks (21 points × 3 = 63 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # Extract right hand landmarks (21 points × 3 = 63 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        frames_data.append(frame_landmarks)

    cap.release()
    holistic.close()

    # Clean up temp file
    os.remove(temp_path)

    # Save features
    output_dir = f"/data/train/{class_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{os.path.splitext(filename)[0]}.npy"
    np.save(output_path, np.array(frames_data, dtype=np.float32))

    return f"Extracted {filename} -> {len(frames_data)} frames"


@app.local_entrypoint()
def main():
    """
    Main entry point: scans local dataset and dispatches extraction jobs.
    """
    import os

    dataset_dir = "dataset"
    tasks = []

    print("=" * 60)
    print("KSL Feature Extraction Pipeline")
    print("=" * 60)
    print(f"\nScanning dataset directory: {dataset_dir}")
    print(f"Classes to process: {len(ALL_CLASSES)}")

    # Collect all video files
    for class_name in ALL_CLASSES:
        class_path = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_path):
            print(f"  Warning: Class folder not found: {class_name}")
            continue

        video_count = 0
        for video_file in os.listdir(class_path):
            if video_file.lower().endswith((".mov", ".mp4", ".avi")):
                video_path = os.path.join(class_path, video_file)
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                tasks.append((video_bytes, video_file, class_name))
                video_count += 1

        print(f"  {class_name}: {video_count} videos")

    print(f"\nTotal videos to process: {len(tasks)}")
    print("\nDispatching to Modal for parallel processing...")
    print("-" * 60)

    # Process all videos in parallel on Modal
    results = list(extract_video_features.starmap(tasks))

    for result in results:
        print(f"  {result}")

    print("-" * 60)
    print(f"\nExtraction complete! Processed {len(results)} videos.")
    print("Features saved to Modal volume: ksl-dataset-vol/train/")
