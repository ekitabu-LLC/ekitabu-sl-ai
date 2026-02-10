#!/usr/bin/env python3
"""
Evaluate v19 model on real-world test signers.

Extracts MediaPipe landmarks from .mov videos, preprocesses them
using the same pipeline as training, and runs inference with the
v19 numbers and words models.

Usage:
    python evaluate_real_testers.py
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import mediapipe as mp

# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v19.py)
# ---------------------------------------------------------------------------

POSE_INDICES = [11, 12, 13, 14, 15, 16]

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

# Parent map for bone features (from v19)
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

# ---------------------------------------------------------------------------
# Video → Landmarks extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
    """Extract MediaPipe Holistic landmarks from a video file.

    Returns numpy array of shape (num_frames, 225) matching training format:
    - Pose: 33 landmarks * 3 = 99
    - Left hand: 21 landmarks * 3 = 63
    - Right hand: 21 landmarks * 3 = 63
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    )

    all_landmarks = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        landmarks = []

        # Pose (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        # Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        all_landmarks.append(landmarks)
        frame_count += 1

    cap.release()
    holistic.close()

    if frame_count == 0:
        return None

    return np.array(all_landmarks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (matching v19 dataset pipeline exactly, no augmentation)
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors: bone[child] = node[child] - node[parent]."""
    bones = np.zeros_like(h)
    for child in range(48):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def preprocess_landmarks(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (num_frames, 225) landmarks to match v19 input format.

    Returns tensor of shape (9, max_frames, 48) = (C, T, N)
    """
    f = raw.shape[0]

    # Extract and reshape to (f, 48, 3)
    if raw.shape[1] >= 225:
        pose = np.zeros((f, 6, 3), dtype=np.float32)
        for pi, idx_pose in enumerate(POSE_INDICES):
            start = idx_pose * 3
            pose[:, pi, :] = raw[:, start:start + 3]
        lh = raw[:, 99:162].reshape(f, 21, 3)
        rh = raw[:, 162:225].reshape(f, 21, 3)
    else:
        return None

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # Centering
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01

    if np.any(lh_valid):
        h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
    if np.any(rh_valid):
        h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

    mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
    h[:, 42:48, :] -= mid_shoulder

    # Separate normalization (hands vs pose)
    hand_data = h[:, :42, :]
    pose_data = h[:, 42:48, :]
    hand_max = np.abs(hand_data).max()
    pose_max = np.abs(pose_data).max()
    if hand_max > 0.01:
        h[:, :42, :] = hand_data / hand_max
    if pose_max > 0.01:
        h[:, 42:48, :] = pose_data / pose_max
    h = np.clip(h, -1, 1).astype(np.float32)

    # Velocity BEFORE resampling
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]

    # Bone features
    bones = compute_bones(h)

    # Temporal sampling / padding
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h = h[indices]
        velocity = velocity[indices]
        bones = bones[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])

    # Concatenate: (max_frames, 48, 9)
    features = np.concatenate([h, velocity, bones], axis=2)

    # Return (9, max_frames, 48) = (C, T, N)
    return torch.FloatTensor(features).permute(2, 0, 1).unsqueeze(0)  # add batch dim


# ---------------------------------------------------------------------------
# Discover test videos (skip Spelling words)
# ---------------------------------------------------------------------------

def discover_test_videos(base_dir):
    """Walk the real_testers directory and find all Numbers and Words videos.

    Returns list of (video_path, class_name, category, signer) tuples.
    """
    videos = []

    for signer_dir in sorted(os.listdir(base_dir)):
        signer_path = os.path.join(base_dir, signer_dir)
        if not os.path.isdir(signer_path):
            continue

        for subdir in os.listdir(signer_path):
            subdir_lower = subdir.lower()

            # Skip spelling words
            if "spelling" in subdir_lower:
                continue

            subdir_path = os.path.join(signer_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # Determine category
            if "number" in subdir_lower:
                category = "numbers"
            elif "word" in subdir_lower:
                # "Words and numbers- Front" contains both
                category = "mixed"
            else:
                continue

            for fn in sorted(os.listdir(subdir_path)):
                if not fn.lower().endswith((".mov", ".mp4", ".avi")):
                    continue

                video_path = os.path.join(subdir_path, fn)
                # Extract class name from filename
                name = os.path.splitext(fn)[0]
                # Strip numbered prefixes like "2. Teach" -> "Teach"
                # or "No. 35" -> "35"
                name = name.strip()
                if ". " in name:
                    name = name.split(". ", 1)[1]
                name = name.replace("No ", "").replace("No. ", "").strip()

                # Determine if this is a number or word
                if name in [c for c in NUMBER_CLASSES]:
                    item_category = "numbers"
                elif name in WORD_CLASSES:
                    item_category = "words"
                else:
                    # Try matching case-insensitively
                    matched = False
                    for wc in WORD_CLASSES:
                        if name.lower() == wc.lower():
                            name = wc
                            item_category = "words"
                            matched = True
                            break
                    if not matched:
                        print(f"  SKIP: Unknown class '{name}' from {fn}")
                        continue

                videos.append((video_path, name, item_category, signer_dir))

    return videos


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    print("=" * 70)
    print("V19 Real-World Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Discover videos
    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    print(f"Found {len(videos)} test videos (excluding Spelling words)")

    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"  Numbers: {len(numbers_videos)} videos")
    print(f"  Words: {len(words_videos)} videos")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Import model class from v19
    from train_ksl_v19 import KSLGraphNet, build_adj

    results = {}

    # ===== NUMBERS =====
    numbers_ckpt = os.path.join(ckpt_dir, "v19_numbers", "best_model.pt")
    if os.path.exists(numbers_ckpt) and numbers_videos:
        print(f"\n{'='*70}")
        print(f"NUMBERS EVALUATION ({len(numbers_videos)} videos)")
        print(f"{'='*70}")

        ckpt = torch.load(numbers_ckpt, map_location=device, weights_only=False)
        config = ckpt["config"]
        adj = build_adj(config["num_nodes"]).to(device)
        tk = tuple(config.get("temporal_kernels", [9]))

        model = KSLGraphNet(
            len(NUMBER_CLASSES), config["num_nodes"], config["in_channels"],
            config["hidden_dim"], config["num_layers"], tk,
            config["dropout"], adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded numbers model: {numbers_ckpt}")
        print(f"  Val accuracy: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")

        c2i = {c: i for i, c in enumerate(NUMBER_CLASSES)}
        i2c = {i: c for c, i in c2i.items()}

        correct, total = 0, 0
        per_class = {}
        per_signer = {}
        details = []

        for video_path, true_class, signer in numbers_videos:
            print(f"\n  Processing: {os.path.basename(video_path)} (class={true_class}, {signer})")

            # Extract landmarks
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: Could not extract landmarks")
                continue

            print(f"    Extracted {raw.shape[0]} frames, shape {raw.shape}")

            # Preprocess
            tensor = preprocess_landmarks(raw)
            if tensor is None:
                print(f"    FAILED: Preprocessing error")
                continue

            # Inference
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                pred_class = i2c[pred_idx]
                confidence = probs[0, pred_idx].item()

            is_correct = (pred_class == true_class)
            correct += int(is_correct)
            total += 1

            # Track per-class
            if true_class not in per_class:
                per_class[true_class] = {"correct": 0, "total": 0}
            per_class[true_class]["total"] += 1
            per_class[true_class]["correct"] += int(is_correct)

            # Track per-signer
            if signer not in per_signer:
                per_signer[signer] = {"correct": 0, "total": 0}
            per_signer[signer]["total"] += 1
            per_signer[signer]["correct"] += int(is_correct)

            status = "OK" if is_correct else "WRONG"
            print(f"    Pred: {pred_class} (conf={confidence:.3f}) | True: {true_class} [{status}]")

            # Top-3 predictions
            top3 = torch.topk(probs, 3, dim=1)
            top3_str = ", ".join([f"{i2c[idx.item()]}={prob.item():.3f}"
                                  for idx, prob in zip(top3.indices[0], top3.values[0])])
            print(f"    Top-3: {top3_str}")

            details.append({
                "video": os.path.basename(video_path),
                "signer": signer,
                "true": true_class,
                "pred": pred_class,
                "correct": is_correct,
                "confidence": confidence,
            })

        numbers_acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"\n  NUMBERS OVERALL: {correct}/{total} = {numbers_acc:.1f}%")
        print(f"\n  Per-class:")
        for cls in NUMBER_CLASSES:
            if cls in per_class:
                pc = per_class[cls]
                acc = 100.0 * pc["correct"] / pc["total"]
                print(f"    {cls:>5s}: {pc['correct']}/{pc['total']} = {acc:.0f}%")
        print(f"\n  Per-signer:")
        for signer in sorted(per_signer.keys()):
            ps = per_signer[signer]
            acc = 100.0 * ps["correct"] / ps["total"]
            print(f"    {signer}: {ps['correct']}/{ps['total']} = {acc:.0f}%")

        results["numbers"] = {
            "overall": numbers_acc,
            "correct": correct,
            "total": total,
            "per_class": {k: {"acc": 100.0 * v["correct"] / v["total"], **v}
                          for k, v in per_class.items()},
            "per_signer": {k: {"acc": 100.0 * v["correct"] / v["total"], **v}
                           for k, v in per_signer.items()},
            "details": details,
        }
    else:
        print(f"\nSkipping numbers: checkpoint not found or no videos")

    # ===== WORDS =====
    words_ckpt = os.path.join(ckpt_dir, "v19_words", "best_model.pt")
    if os.path.exists(words_ckpt) and words_videos:
        print(f"\n{'='*70}")
        print(f"WORDS EVALUATION ({len(words_videos)} videos)")
        print(f"{'='*70}")

        ckpt = torch.load(words_ckpt, map_location=device, weights_only=False)
        config = ckpt["config"]
        adj = build_adj(config["num_nodes"]).to(device)
        tk = tuple(config.get("temporal_kernels", [9]))

        model = KSLGraphNet(
            len(WORD_CLASSES), config["num_nodes"], config["in_channels"],
            config["hidden_dim"], config["num_layers"], tk,
            config["dropout"], adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded words model: {words_ckpt}")
        print(f"  Val accuracy: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")

        c2i = {c: i for i, c in enumerate(WORD_CLASSES)}
        i2c = {i: c for c, i in c2i.items()}

        correct, total = 0, 0
        per_class = {}
        per_signer = {}
        details = []

        for video_path, true_class, signer in words_videos:
            print(f"\n  Processing: {os.path.basename(video_path)} (class={true_class}, {signer})")

            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: Could not extract landmarks")
                continue

            print(f"    Extracted {raw.shape[0]} frames, shape {raw.shape}")

            tensor = preprocess_landmarks(raw)
            if tensor is None:
                print(f"    FAILED: Preprocessing error")
                continue

            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                pred_class = i2c[pred_idx]
                confidence = probs[0, pred_idx].item()

            is_correct = (pred_class == true_class)
            correct += int(is_correct)
            total += 1

            if true_class not in per_class:
                per_class[true_class] = {"correct": 0, "total": 0}
            per_class[true_class]["total"] += 1
            per_class[true_class]["correct"] += int(is_correct)

            if signer not in per_signer:
                per_signer[signer] = {"correct": 0, "total": 0}
            per_signer[signer]["total"] += 1
            per_signer[signer]["correct"] += int(is_correct)

            status = "OK" if is_correct else "WRONG"
            print(f"    Pred: {pred_class} (conf={confidence:.3f}) | True: {true_class} [{status}]")

            top3 = torch.topk(probs, 3, dim=1)
            top3_str = ", ".join([f"{i2c[idx.item()]}={prob.item():.3f}"
                                  for idx, prob in zip(top3.indices[0], top3.values[0])])
            print(f"    Top-3: {top3_str}")

            details.append({
                "video": os.path.basename(video_path),
                "signer": signer,
                "true": true_class,
                "pred": pred_class,
                "correct": is_correct,
                "confidence": confidence,
            })

        words_acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"\n  WORDS OVERALL: {correct}/{total} = {words_acc:.1f}%")
        print(f"\n  Per-class:")
        for cls in WORD_CLASSES:
            if cls in per_class:
                pc = per_class[cls]
                acc = 100.0 * pc["correct"] / pc["total"]
                print(f"    {cls:>12s}: {pc['correct']}/{pc['total']} = {acc:.0f}%")
        print(f"\n  Per-signer:")
        for signer in sorted(per_signer.keys()):
            ps = per_signer[signer]
            acc = 100.0 * ps["correct"] / ps["total"]
            print(f"    {signer}: {ps['correct']}/{ps['total']} = {acc:.0f}%")

        results["words"] = {
            "overall": words_acc,
            "correct": correct,
            "total": total,
            "per_class": {k: {"acc": 100.0 * v["correct"] / v["total"], **v}
                          for k, v in per_class.items()},
            "per_signer": {k: {"acc": 100.0 * v["correct"] / v["total"], **v}
                           for k, v in per_signer.items()},
            "details": details,
        }
    else:
        print(f"\nSkipping words: checkpoint not found or no videos")

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V19 Real-World Evaluation")
    print(f"{'='*70}")
    if "numbers" in results:
        print(f"  Numbers: {results['numbers']['overall']:.1f}% "
              f"({results['numbers']['correct']}/{results['numbers']['total']})")
    if "words" in results:
        print(f"  Words:   {results['words']['overall']:.1f}% "
              f"({results['words']['correct']}/{results['words']['total']})")
    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
        print(f"  Combined: {combined:.1f}%")
        print(f"  (Val combined was 87.3%)")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v19_real_testers_{ts}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
