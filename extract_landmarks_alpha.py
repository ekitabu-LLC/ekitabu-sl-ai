#!/usr/bin/env python3
"""
Extract MediaPipe Holistic landmarks from ksl-alpha .mov videos.

Reads splits_info.csv to determine train/val/test splits and processes all
videos into .npy files matching the format used by train_ksl_v25.py.

Features extracted per frame (549 total):
- Pose landmarks: 33 points x 3 coords = 99
- Left hand: 21 points x 3 coords = 63
- Right hand: 21 points x 3 coords = 63
- Face (selected key points): 68 points x 3 coords = 204
- Lips: 40 points x 3 coords = 120

Usage:
    python extract_landmarks_alpha.py
    python extract_landmarks_alpha.py --split train
    python extract_landmarks_alpha.py --split val
    python extract_landmarks_alpha.py --workers 128
"""

import argparse
import csv
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (from extract_features_v2.py)
# ---------------------------------------------------------------------------

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

FACE_INDICES = (LIPS_INDICES[:40] + LEFT_EYEBROW[:5] + RIGHT_EYEBROW[:5] +
                LEFT_EYE[:6] + RIGHT_EYE[:6] + NOSE[:4] + [10, 152])


# ---------------------------------------------------------------------------
# Number-to-folder name mapping
# ---------------------------------------------------------------------------

# ksl-alpha uses "No_XX" folder names for numbers
NUMBER_FOLDER_PREFIX = "No_"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_landmarks_from_video(video_path):
    """Extract 549-dim MediaPipe Holistic landmarks from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    )

    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frame_landmarks = []

        # 1. Pose landmarks (33 x 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 99)

        # 2. Left hand (21 x 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 3. Right hand (21 x 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 63)

        # 4. Face key points (68 x 3 = 204)
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

        # 5. Detailed lips (40 x 3 = 120)
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

    if len(frames_data) == 0:
        return None

    arr = np.array(frames_data, dtype=np.float32)
    assert arr.shape[1] == 549, f"Expected 549 features, got {arr.shape[1]}"
    return arr


def class_name_from_folder(folder_name):
    """Convert ksl-alpha folder name to class name used in training."""
    if folder_name.startswith(NUMBER_FOLDER_PREFIX):
        return folder_name[len(NUMBER_FOLDER_PREFIX):]
    return folder_name


def process_single_video(task):
    """Worker function: extract landmarks for one video and save .npy.

    Takes a dict with all info needed. Returns (out_path, shape, success).
    Each worker creates its own MediaPipe instance (not picklable).
    """
    video_path = task["video_path"]
    out_path = task["out_path"]

    # Skip if already extracted
    if os.path.exists(out_path):
        return (out_path, None, True, "skipped")

    # Make output dir (safe for concurrent calls)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    landmarks = extract_landmarks_from_video(video_path)

    if landmarks is None:
        return (out_path, None, False, "no landmarks")

    np.save(out_path, landmarks)
    return (out_path, landmarks.shape, True, "extracted")


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from ksl-alpha videos")
    parser.add_argument("--alpha-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-alpha/cleaned_data",
                        help="Path to ksl-alpha cleaned_data directory")
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data",
                        help="Base output directory")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"[{ts()}] KSL-Alpha Landmark Extraction")
    print(f"  MediaPipe version: {mp.__version__}")
    print(f"  Source: {args.alpha_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    # Read splits_info.csv
    splits_csv = os.path.join(args.alpha_dir, "splits_info.csv")
    if not os.path.exists(splits_csv):
        print(f"ERROR: splits_info.csv not found at {splits_csv}")
        return

    videos = []
    with open(splits_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row)

    print(f"\n[{ts()}] Loaded {len(videos)} video entries from splits_info.csv")

    # Filter by split
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    videos = [v for v in videos if v["split"] in splits_to_process]
    print(f"[{ts()}] Processing {len(videos)} videos for splits: {splits_to_process}")

    # Count per split
    split_counts = Counter(v["split"] for v in videos)
    for s, c in sorted(split_counts.items()):
        print(f"  {s}: {c} videos")

    # Map split name to output directory
    split_dir_map = {
        "train": "train_alpha",
        "val": "val_alpha",
        "test": "test_alpha",
    }

    # Build task list
    tasks = []
    skipped_parse = 0
    skipped_missing = 0

    for video_info in videos:
        filename = video_info["filename"]
        class_folder = video_info["class"]
        split = video_info["split"]
        relative_path = video_info["relative_path"]

        # Parse signer ID from filename (format: Signer_XX_*.mov)
        signer_match = re.match(r"Signer_(\d+)", filename)
        if not signer_match:
            print(f"  WARNING: Cannot parse signer from {filename}, skipping")
            skipped_parse += 1
            continue
        signer_id = int(signer_match.group(1))

        # Convert folder name to class name
        class_name = class_name_from_folder(class_folder)

        # Build paths
        video_path = os.path.join(args.alpha_dir, relative_path)
        if not os.path.exists(video_path):
            print(f"  WARNING: Video not found: {video_path}")
            skipped_missing += 1
            continue

        # Output filename
        instance_match = re.search(r"_(\d+)\.mov$", filename, re.IGNORECASE)
        instance_id = instance_match.group(1) if instance_match else os.path.splitext(filename)[0].replace("Signer_", "").replace(f"{signer_id}_", "")

        out_split_dir = os.path.join(args.output_dir, split_dir_map[split], class_name)
        out_filename = f"{class_name}-{signer_id}-{instance_id}.npy"
        out_path = os.path.join(out_split_dir, out_filename)

        tasks.append({
            "video_path": video_path,
            "out_path": out_path,
        })

    if skipped_parse or skipped_missing:
        print(f"  Skipped: {skipped_parse} parse errors, {skipped_missing} missing files")

    print(f"\n[{ts()}] Starting extraction of {len(tasks)} videos with {args.workers} workers...")

    # Process videos
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    if args.workers <= 1:
        # Sequential mode
        for i, task in enumerate(tasks):
            out_path, shape, ok, status = process_single_video(task)
            if ok:
                success += 1
                if status == "skipped":
                    skipped += 1
            else:
                failed += 1
                print(f"  FAILED: {task['video_path']} ({status})")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(tasks) - i - 1) / rate
                print(f"[{ts()}] {i + 1}/{len(tasks)} "
                      f"({success} ok, {failed} failed, {skipped} skipped) "
                      f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")
    else:
        # Parallel mode
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                try:
                    out_path, shape, ok, status = future.result()
                    if ok:
                        success += 1
                        if status == "skipped":
                            skipped += 1
                    else:
                        failed += 1
                        task = futures[future]
                        print(f"  FAILED: {task['video_path']} ({status})")
                except Exception as e:
                    failed += 1
                    task = futures[future]
                    print(f"  ERROR: {task['video_path']} ({e})")

                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(tasks) - completed) / rate
                    print(f"[{ts()}] {completed}/{len(tasks)} "
                          f"({success} ok, {failed} failed, {skipped} skipped) "
                          f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"[{ts()}] Extraction complete!")
    print(f"  Total: {len(tasks)}, Success: {success}, Failed: {failed}, Skipped existing: {skipped}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output directories:")
    for split_name in splits_to_process:
        out_dir = os.path.join(args.output_dir, split_dir_map[split_name])
        if os.path.exists(out_dir):
            count = sum(len(files) for _, _, files in os.walk(out_dir))
            print(f"    {out_dir}: {count} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
