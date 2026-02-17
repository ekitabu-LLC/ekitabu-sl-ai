#!/usr/bin/env python3
"""
Pre-extract MediaPipe Holistic landmarks from real tester videos.

Saves .npy files alongside original videos so evaluation can skip
the slow MediaPipe extraction step.

Output: <video_path>.landmarks.npy (same dir as video)

Usage:
    python extract_landmarks_real_testers.py
    python extract_landmarks_real_testers.py --workers 32
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (same as extract_landmarks_alpha.py)
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


def process_single_video(video_path):
    """Extract landmarks from one video and save as .npy alongside it."""
    npy_path = video_path + ".landmarks.npy"

    # Skip if already extracted
    if os.path.exists(npy_path):
        return (video_path, None, True, "skipped")

    landmarks = extract_landmarks_from_video(video_path)

    if landmarks is None:
        return (video_path, None, False, "no landmarks")

    np.save(npy_path, landmarks)
    return (video_path, landmarks.shape, True, "extracted")


def discover_videos(base_dir):
    """Find all video files in the real_testers directory."""
    videos = []
    for root, dirs, files in os.walk(base_dir):
        for fn in sorted(files):
            if fn.lower().endswith((".mov", ".mp4", ".avi")):
                videos.append(os.path.join(root, fn))
    return videos


def main():
    parser = argparse.ArgumentParser(description="Pre-extract landmarks from real tester videos")
    parser.add_argument("--base-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-alpha/data/real_testers",
                        help="Path to real_testers directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    args = parser.parse_args()

    print("=" * 70)
    print(f"[{ts()}] Real Tester Landmark Pre-Extraction")
    print(f"  MediaPipe version: {mp.__version__}")
    print(f"  Source: {args.base_dir}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    videos = discover_videos(args.base_dir)
    print(f"\n[{ts()}] Found {len(videos)} videos")

    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    if args.workers <= 1:
        for i, vp in enumerate(videos):
            _, shape, ok, status = process_single_video(vp)
            if ok:
                success += 1
                if status == "skipped":
                    skipped += 1
                else:
                    print(f"  [{i+1}/{len(videos)}] {os.path.basename(vp)} -> {shape}", flush=True)
            else:
                failed += 1
                print(f"  [{i+1}/{len(videos)}] FAILED: {os.path.basename(vp)}", flush=True)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, vp): vp for vp in videos}

            for future in as_completed(futures):
                completed += 1
                try:
                    vp, shape, ok, status = future.result()
                    if ok:
                        success += 1
                        if status == "skipped":
                            skipped += 1
                    else:
                        failed += 1
                        print(f"  FAILED: {os.path.basename(futures[future])}", flush=True)
                except Exception as e:
                    failed += 1
                    print(f"  ERROR: {os.path.basename(futures[future])} ({e})", flush=True)

                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(videos) - completed) / rate if rate > 0 else 0
                    print(f"[{ts()}] {completed}/{len(videos)} "
                          f"({success} ok, {failed} failed, {skipped} skipped) "
                          f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]",
                          flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"[{ts()}] Done!")
    print(f"  Total: {len(videos)}, Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
