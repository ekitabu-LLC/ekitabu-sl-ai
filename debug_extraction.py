"""
Debug script to compare training data extraction vs test extraction.
Identifies mismatches in landmark detection.

Usage:
    modal run debug_extraction.py
"""

import modal
import os

app = modal.App("ksl-debug")
volume = modal.Volume.from_name("ksl-dataset-vol")

# Use SAME MediaPipe version as training extraction
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libegl1", "libxext6")
    .pip_install("torch", "numpy", "opencv-python-headless", "mediapipe==0.10.14")
)


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def debug_landmarks():
    import numpy as np
    import cv2
    import mediapipe as mp

    print("=" * 70)
    print("DEBUG: Comparing Training Data vs Test Extraction")
    print("=" * 70)
    print(f"MediaPipe version: {mp.__version__}")

    # Initialize same as training extraction
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    def extract_from_video(vpath):
        """Extract landmarks same way as training."""
        cap = cv2.VideoCapture(vpath)
        frames_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            frame_landmarks = []

            # Pose (99 values)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 99)

            # Left hand (63 values)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)

            # Right hand (63 values)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)

            frames_data.append(frame_landmarks)

        cap.release()
        return np.array(frames_data, dtype=np.float32)

    # Check training data format
    print("\n--- TRAINING DATA ANALYSIS ---")
    train_dir = "/data/train_v2"
    sample_classes = ["444", "54", "Friend", "Monday"]

    for cls in sample_classes:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"{cls}: NOT FOUND")
            continue

        files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        if not files:
            continue

        # Load first sample
        data = np.load(os.path.join(cls_dir, files[0]))
        print(f"\n{cls}:")
        print(f"  Shape: {data.shape}")
        print(f"  Frames: {data.shape[0]}")

        # Check hand presence
        lh_present = np.sum(np.abs(data[:, 99:162])) > 0
        rh_present = np.sum(np.abs(data[:, 162:225])) > 0
        pose_present = np.sum(np.abs(data[:, 0:99])) > 0
        print(f"  Pose detected: {pose_present}")
        print(f"  Left hand detected: {lh_present}")
        print(f"  Right hand detected: {rh_present}")

        # Sample values
        mid = data.shape[0] // 2
        print(f"  Sample LH wrist (frame {mid}): {data[mid, 99:102]}")
        print(f"  Sample RH wrist (frame {mid}): {data[mid, 162:165]}")

    # Extract from test videos and compare
    print("\n--- TEST VIDEO EXTRACTION ---")
    test_results = {}

    for folder, prefix in [("/data/testing/Numbers", "No. "), ("/data/testing/Words - Left Side", "")]:
        if not os.path.exists(folder):
            continue

        for fn in sorted(os.listdir(folder))[:3]:  # First 3 only
            if not fn.endswith(".mov"):
                continue

            if prefix:
                gt = fn.replace(prefix, "").replace(".mov", "")
            else:
                gt = fn.split(". ", 1)[1].replace(".mov", "") if ". " in fn else fn.replace(".mov", "")

            print(f"\n{fn} (class: {gt}):")
            vpath = os.path.join(folder, fn)
            data = extract_from_video(vpath)
            print(f"  Shape: {data.shape}")

            if data.shape[0] == 0:
                print("  NO FRAMES EXTRACTED!")
                continue

            # Check hand presence
            lh_present = np.sum(np.abs(data[:, 99:162])) > 0
            rh_present = np.sum(np.abs(data[:, 162:225])) > 0
            pose_present = np.sum(np.abs(data[:, 0:99])) > 0
            print(f"  Pose detected: {pose_present}")
            print(f"  Left hand detected: {lh_present}")
            print(f"  Right hand detected: {rh_present}")

            # Count frames with hands
            lh_frames = sum(1 for i in range(data.shape[0]) if np.sum(np.abs(data[i, 99:162])) > 0)
            rh_frames = sum(1 for i in range(data.shape[0]) if np.sum(np.abs(data[i, 162:225])) > 0)
            print(f"  LH frames: {lh_frames}/{data.shape[0]}")
            print(f"  RH frames: {rh_frames}/{data.shape[0]}")

            # Compare with training data
            train_path = f"/data/train_v2/{gt}"
            if os.path.exists(train_path):
                train_files = [f for f in os.listdir(train_path) if f.endswith(".npy")]
                if train_files:
                    train_data = np.load(os.path.join(train_path, train_files[0]))
                    print(f"  Training sample shape: {train_data.shape}")

                    # Compare hand dominance
                    train_lh = np.sum(np.abs(train_data[:, 99:162]))
                    train_rh = np.sum(np.abs(train_data[:, 162:225]))
                    test_lh = np.sum(np.abs(data[:, 99:162]))
                    test_rh = np.sum(np.abs(data[:, 162:225]))

                    print(f"  Train LH/RH ratio: {train_lh/(train_rh+1e-8):.2f}")
                    print(f"  Test LH/RH ratio: {test_lh/(test_rh+1e-8):.2f}")

            test_results[gt] = data

    holistic.close()
    return {"status": "debug complete"}


@app.local_entrypoint()
def main():
    debug_landmarks.remote()
