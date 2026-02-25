#!/usr/bin/env python3
"""
Evaluate v21 model on real-world test signers.

V21 changes from v20:
- Auxiliary MLP branch: joint angles + fingertip distances
- ArcFace head (not used at inference - uses CE classifier)
- Signer-adversarial training via GRL
- Supervised contrastive loss
- Model returns (logits, signer_logits, embedding) during forward

Usage:
    python evaluate_real_testers_v21.py
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import mediapipe as mp

# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v21.py)
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

# Parent map for bone features (48 nodes)
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]

PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

# Joint angle topology (same as train_ksl_v21.py)
def _build_angle_joints():
    children = defaultdict(list)
    for child_idx, parent_idx in enumerate(PARENT_MAP):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    angle_joints = []
    for node in range(48):
        parent = PARENT_MAP[node]
        if parent < 0:
            continue
        for child in children[node]:
            angle_joints.append((node, parent, child))
    return angle_joints

ANGLE_JOINTS = _build_angle_joints()
NUM_ANGLE_FEATURES = len(ANGLE_JOINTS)

# Fingertip indices
LH_TIPS = [4, 8, 12, 16, 20]
RH_TIPS = [25, 29, 33, 37, 41]
NUM_FINGERTIP_PAIRS = 10

# ---------------------------------------------------------------------------
# Video -> Landmarks extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
    """Extract MediaPipe Holistic landmarks from a video file."""
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        all_landmarks.append(landmarks)

    cap.release()
    holistic.close()

    if len(all_landmarks) == 0:
        return None

    return np.array(all_landmarks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (matching v21 dataset pipeline, no augmentation)
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors."""
    bones = np.zeros_like(h)
    for child in range(48):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def compute_joint_angles(h):
    """Compute joint angles at each node where two bones meet."""
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)

    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        bone_in = h[:, node, :] - h[:, parent, :]
        bone_out = h[:, child, :] - h[:, node, :]

        norm_in = np.linalg.norm(bone_in, axis=-1, keepdims=True)
        norm_out = np.linalg.norm(bone_out, axis=-1, keepdims=True)
        norm_in = np.maximum(norm_in, 1e-8)
        norm_out = np.maximum(norm_out, 1e-8)

        bone_in_n = bone_in / norm_in
        bone_out_n = bone_out / norm_out

        dot = np.sum(bone_in_n * bone_out_n, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        angles[:, idx] = np.arccos(dot)

    return angles


def compute_fingertip_distances(h):
    """Compute pairwise L2 distances between fingertips for each hand."""
    T = h.shape[0]
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)

    col = 0
    for i in range(len(LH_TIPS)):
        for j in range(i + 1, len(LH_TIPS)):
            diff = h[:, LH_TIPS[i], :] - h[:, LH_TIPS[j], :]
            distances[:, col] = np.linalg.norm(diff, axis=-1)
            col += 1

    for i in range(len(RH_TIPS)):
        for j in range(i + 1, len(RH_TIPS)):
            diff = h[:, RH_TIPS[i], :] - h[:, RH_TIPS[j], :]
            distances[:, col] = np.linalg.norm(diff, axis=-1)
            col += 1

    return distances


def normalize_wrist_palm(h):
    """Wrist-centric + palm-size normalization (matching v21 training)."""
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        lh_wrist = lh[:, 0:1, :]
        lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]
        palm_sizes = np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        rh_wrist = rh[:, 0:1, :]
        rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]
        palm_sizes = np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] = pose[pose_valid] - mid_shoulder[pose_valid]
        shoulder_width = np.linalg.norm(
            pose[pose_valid, 0, :] - pose[pose_valid, 1, :],
            axis=-1, keepdims=True
        )
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        pose[pose_valid] = pose[pose_valid] / shoulder_width[:, :, np.newaxis]
    h[:, 42:48, :] = pose

    return h


def preprocess_landmarks(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (num_frames, 225) landmarks to v21 format.

    Returns:
        gcn_tensor: (9, max_frames, 48)
        aux_tensor: (max_frames, D_aux)
    """
    f = raw.shape[0]

    if raw.shape[1] < 225:
        return None, None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # Normalization
    h = normalize_wrist_palm(h)

    # Velocity before resampling
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]

    # Bone features
    bones = compute_bones(h)

    # Auxiliary features
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    # Temporal sampling / padding
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h = h[indices]
        velocity = velocity[indices]
        bones = bones[indices]
        joint_angles = joint_angles[indices]
        fingertip_dists = fingertip_dists[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])

    # GCN features: (max_frames, 48, 9) -> (9, max_frames, 48)
    gcn_features = np.concatenate([h, velocity, bones], axis=2)
    gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)
    gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)

    # Aux features: (max_frames, D_aux)
    aux_features = np.concatenate([joint_angles, fingertip_dists], axis=1)
    aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
    aux_tensor = torch.FloatTensor(aux_features)

    return gcn_tensor, aux_tensor


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA) - adapted for v21
# ---------------------------------------------------------------------------

def apply_tta(raw, max_frames=MAX_FRAMES):
    """Generate 5 augmented versions for TTA.

    Returns list of (gcn_tensor, aux_tensor) tuples.
    """
    augmented = []

    # 1. Original
    gcn, aux = preprocess_landmarks(raw, max_frames)
    if gcn is None:
        return None
    augmented.append((gcn, aux))

    # 2. Spatial jitter
    raw_jitter = raw.copy()
    raw_jitter += np.random.normal(0, 0.01, raw_jitter.shape).astype(np.float32)
    gcn_j, aux_j = preprocess_landmarks(raw_jitter, max_frames)
    if gcn_j is not None:
        augmented.append((gcn_j, aux_j))

    # 3. Speed perturbation 0.9x
    f = raw.shape[0]
    target_len_slow = int(f * 0.9)
    if target_len_slow >= 2:
        indices_slow = np.linspace(0, f - 1, target_len_slow, dtype=int)
        gcn_s, aux_s = preprocess_landmarks(raw[indices_slow], max_frames)
        if gcn_s is not None:
            augmented.append((gcn_s, aux_s))

    # 4. Speed perturbation 1.1x
    target_len_fast = int(f * 1.1)
    if target_len_fast >= 2:
        indices_fast = np.linspace(0, f - 1, target_len_fast, dtype=int)
        gcn_f, aux_f = preprocess_landmarks(raw[indices_fast], max_frames)
        if gcn_f is not None:
            augmented.append((gcn_f, aux_f))

    # 5. Small rotation (+5 degrees Z-axis on hands)
    raw_rot = raw.copy()
    angle_rad = np.radians(5.0)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for frame_idx in range(raw_rot.shape[0]):
        for lm_idx in range(21):
            base = 99 + lm_idx * 3
            x = raw_rot[frame_idx, base]
            y = raw_rot[frame_idx, base + 1]
            raw_rot[frame_idx, base] = cos_a * x - sin_a * y
            raw_rot[frame_idx, base + 1] = sin_a * x + cos_a * y
        for lm_idx in range(21):
            base = 162 + lm_idx * 3
            x = raw_rot[frame_idx, base]
            y = raw_rot[frame_idx, base + 1]
            raw_rot[frame_idx, base] = cos_a * x - sin_a * y
            raw_rot[frame_idx, base + 1] = sin_a * x + cos_a * y
    gcn_r, aux_r = preprocess_landmarks(raw_rot, max_frames)
    if gcn_r is not None:
        augmented.append((gcn_r, aux_r))

    return augmented


# ---------------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------------

def confidence_level(conf):
    if conf > 0.6:
        return "HIGH"
    elif conf >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Discover test videos
# ---------------------------------------------------------------------------

def discover_test_videos(base_dir):
    """Walk the real_testers directory and find all Numbers and Words videos."""
    videos = []

    for signer_dir in sorted(os.listdir(base_dir)):
        signer_path = os.path.join(base_dir, signer_dir)
        if not os.path.isdir(signer_path):
            continue

        for subdir in os.listdir(signer_path):
            subdir_lower = subdir.lower()
            if "spelling" in subdir_lower:
                continue

            subdir_path = os.path.join(signer_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            if "number" in subdir_lower:
                category = "numbers"
            elif "word" in subdir_lower:
                category = "mixed"
            else:
                continue

            for fn in sorted(os.listdir(subdir_path)):
                if not fn.lower().endswith((".mov", ".mp4", ".avi")):
                    continue

                video_path = os.path.join(subdir_path, fn)
                name = os.path.splitext(fn)[0].strip()
                if ". " in name:
                    name = name.split(". ", 1)[1]
                name = name.replace("No ", "").replace("No. ", "").strip()

                if name in NUMBER_CLASSES:
                    item_category = "numbers"
                elif name in WORD_CLASSES:
                    item_category = "words"
                else:
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
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, raw, device, classes, use_tta=True):
    """Run inference with optional TTA. Returns dict with predictions."""
    i2c = {i: c for i, c in enumerate(classes)}

    # No-TTA prediction
    gcn, aux = preprocess_landmarks(raw)
    if gcn is None:
        return None

    with torch.no_grad():
        logits, _, _ = model(
            gcn.unsqueeze(0).to(device),
            aux.unsqueeze(0).to(device),
        )
        probs_noTTA = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx_noTTA = probs_noTTA.argmax().item()
        conf_noTTA = probs_noTTA[pred_idx_noTTA].item()

    result = {
        "pred_class_noTTA": i2c[pred_idx_noTTA],
        "pred_idx_noTTA": pred_idx_noTTA,
        "confidence_noTTA": conf_noTTA,
        "probs_noTTA": probs_noTTA.numpy(),
    }

    if use_tta:
        augmented = apply_tta(raw)
        if augmented is None or len(augmented) == 0:
            result["pred_class"] = result["pred_class_noTTA"]
            result["pred_idx"] = result["pred_idx_noTTA"]
            result["confidence"] = result["confidence_noTTA"]
            result["probs_tta"] = result["probs_noTTA"]
            result["tta_count"] = 0
            return result

        all_probs = []
        with torch.no_grad():
            for gcn_aug, aux_aug in augmented:
                logits, _, _ = model(
                    gcn_aug.unsqueeze(0).to(device),
                    aux_aug.unsqueeze(0).to(device),
                )
                probs = F.softmax(logits, dim=1).cpu().squeeze(0)
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred_idx_tta = avg_probs.argmax().item()
        conf_tta = avg_probs[pred_idx_tta].item()

        result["pred_class"] = i2c[pred_idx_tta]
        result["pred_idx"] = pred_idx_tta
        result["confidence"] = conf_tta
        result["probs_tta"] = avg_probs.numpy()
        result["tta_count"] = len(augmented)
    else:
        result["pred_class"] = result["pred_class_noTTA"]
        result["pred_idx"] = result["pred_idx_noTTA"]
        result["confidence"] = result["confidence_noTTA"]
        result["probs_tta"] = result["probs_noTTA"]
        result["tta_count"] = 1

    return result


# ---------------------------------------------------------------------------
# Evaluate one category
# ---------------------------------------------------------------------------

def evaluate_category(category_name, videos, model, device, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct_tta, correct_noTTA, total = 0, 0, 0
    per_class = {}
    per_signer = {}
    details = []

    nc = len(classes)
    confusion = [[0] * nc for _ in range(nc)]

    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}

    for video_path, true_class, signer in videos:
        print(f"\n  Processing: {os.path.basename(video_path)} (class={true_class}, {signer})")

        raw = extract_landmarks_from_video(video_path)
        if raw is None:
            print(f"    FAILED: Could not extract landmarks")
            continue

        print(f"    Extracted {raw.shape[0]} frames, shape {raw.shape}")

        result = run_inference(model, raw, device, classes, use_tta=True)
        if result is None:
            print(f"    FAILED: Preprocessing error")
            continue

        pred_class = result["pred_class"]
        confidence = result["confidence"]
        pred_noTTA = result["pred_class_noTTA"]
        conf_noTTA = result["confidence_noTTA"]
        conf_lvl = confidence_level(confidence)

        is_correct_tta = (pred_class == true_class)
        is_correct_noTTA = (pred_noTTA == true_class)
        correct_tta += int(is_correct_tta)
        correct_noTTA += int(is_correct_noTTA)
        total += 1

        true_idx = c2i.get(true_class, -1)
        pred_idx = c2i.get(pred_class, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct_tta)

        if true_class not in per_class:
            per_class[true_class] = {"correct_tta": 0, "correct_noTTA": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct_tta"] += int(is_correct_tta)
        per_class[true_class]["correct_noTTA"] += int(is_correct_noTTA)

        if signer not in per_signer:
            per_signer[signer] = {"correct_tta": 0, "correct_noTTA": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct_tta"] += int(is_correct_tta)
        per_signer[signer]["correct_noTTA"] += int(is_correct_noTTA)

        status = "OK" if is_correct_tta else "WRONG"
        tta_delta = ""
        if pred_class != pred_noTTA:
            tta_delta = f" [TTA changed: {pred_noTTA}->{pred_class}]"
        print(f"    TTA Pred: {pred_class} (conf={confidence:.3f}, {conf_lvl}) | "
              f"True: {true_class} [{status}]{tta_delta}")
        print(f"    No-TTA:   {pred_noTTA} (conf={conf_noTTA:.3f}) | "
              f"{'OK' if is_correct_noTTA else 'WRONG'}")

        probs_tta = torch.FloatTensor(result["probs_tta"])
        top3 = torch.topk(probs_tta, min(3, len(classes)))
        top3_str = ", ".join([f"{i2c[idx.item()]}={prob.item():.3f}"
                              for idx, prob in zip(top3.indices, top3.values)])
        print(f"    Top-3 (TTA): {top3_str}")

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred_tta": pred_class,
            "pred_noTTA": pred_noTTA,
            "correct_tta": is_correct_tta,
            "correct_noTTA": is_correct_noTTA,
            "confidence_tta": confidence,
            "confidence_noTTA": conf_noTTA,
            "confidence_level": conf_lvl,
            "tta_changed_prediction": pred_class != pred_noTTA,
            "tta_count": result["tta_count"],
        })

    # Summary
    acc_tta = 100.0 * correct_tta / total if total > 0 else 0.0
    acc_noTTA = 100.0 * correct_noTTA / total if total > 0 else 0.0

    print(f"\n  {category_name.upper()} OVERALL (TTA):   {correct_tta}/{total} = {acc_tta:.1f}%")
    print(f"  {category_name.upper()} OVERALL (no-TTA): {correct_noTTA}/{total} = {acc_noTTA:.1f}%")
    tta_diff = acc_tta - acc_noTTA
    print(f"  TTA effect: {'+' if tta_diff >= 0 else ''}{tta_diff:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'TTA':>7s}  {'no-TTA':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            acc_t = 100.0 * pc["correct_tta"] / pc["total"]
            acc_n = 100.0 * pc["correct_noTTA"] / pc["total"]
            print(f"    {cls:>12s}  {acc_t:5.0f}%   {acc_n:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    print(f"    {'Signer':>20s}  {'TTA':>7s}  {'no-TTA':>7s}  {'N':>3s}")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        acc_t = 100.0 * ps["correct_tta"] / ps["total"]
        acc_n = 100.0 * ps["correct_noTTA"] / ps["total"]
        print(f"    {signer:>20s}  {acc_t:5.0f}%   {acc_n:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence-based analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {acc:.0f}%")
        else:
            print(f"    {lvl:>6s}: 0 predictions")

    high_med_correct = conf_buckets["HIGH"]["correct"] + conf_buckets["MEDIUM"]["correct"]
    high_med_total = conf_buckets["HIGH"]["total"] + conf_buckets["MEDIUM"]["total"]
    if high_med_total > 0:
        filtered_acc = 100.0 * high_med_correct / high_med_total
        print(f"\n    If we reject predictions below 0.4 confidence:")
        print(f"      Accuracy: {high_med_correct}/{high_med_total} = {filtered_acc:.1f}%")
        print(f"      Rejection rate: {conf_buckets['LOW']['total']}/{total} "
              f"= {100.0 * conf_buckets['LOW']['total'] / total:.0f}%")

    print(f"\n  Confusion Matrix ({category_name}):")
    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"    {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{confusion[i][j]:5d}" for j in range(nc))
        print(f"    {row_str}")

    return {
        "overall_tta": acc_tta,
        "overall_noTTA": acc_noTTA,
        "correct_tta": correct_tta,
        "correct_noTTA": correct_noTTA,
        "total": total,
        "tta_effect": tta_diff,
        "per_class": {k: {
            "acc_tta": 100.0 * v["correct_tta"] / v["total"],
            "acc_noTTA": 100.0 * v["correct_noTTA"] / v["total"],
            **v
        } for k, v in per_class.items()},
        "per_signer": {k: {
            "acc_tta": 100.0 * v["correct_tta"] / v["total"],
            "acc_noTTA": 100.0 * v["correct_noTTA"] / v["total"],
            **v
        } for k, v in per_signer.items()},
        "confidence_buckets": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            **v
        } for k, v in conf_buckets.items()},
        "confusion_matrix": confusion,
        "confusion_labels": [i2c[i] for i in range(nc)],
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    print("=" * 70)
    print("V21 Real-World Evaluation (with TTA + Confidence Analysis)")
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

    # Import model class from v21
    from train_ksl_v21 import KSLGraphNetV21, build_adj

    results = {}

    # ===== NUMBERS =====
    numbers_ckpt = os.path.join(ckpt_dir, "v21_numbers", "best_model.pt")
    if os.path.exists(numbers_ckpt) and numbers_videos:
        print(f"\n{'='*70}")
        print(f"NUMBERS EVALUATION ({len(numbers_videos)} videos)")
        print(f"{'='*70}")

        ckpt = torch.load(numbers_ckpt, map_location=device, weights_only=False)
        config = ckpt["config"]
        adj = build_adj(config["num_nodes"]).to(device)
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 10)
        aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS)

        model = KSLGraphNetV21(
            nc=len(NUMBER_CLASSES),
            num_signers=num_signers,
            aux_dim=aux_dim,
            nn_=config["num_nodes"],
            ic=config["in_channels"],
            hd=config["hidden_dim"],
            nl=config["num_layers"],
            tk=tk,
            dr=config["dropout"],
            spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
            arcface_s=config.get("arcface_s", 30.0),
            arcface_m=config.get("arcface_m", 0.3),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded numbers model: {numbers_ckpt}")
        print(f"  Val accuracy: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        print(f"  Aux dim: {aux_dim} (angles={NUM_ANGLE_FEATURES}, fingertip_dists=20)")

        results["numbers"] = evaluate_category(
            "Numbers", numbers_videos, model, device, NUMBER_CLASSES
        )
    else:
        if not os.path.exists(numbers_ckpt):
            print(f"\nSkipping numbers: checkpoint not found at {numbers_ckpt}")
        elif not numbers_videos:
            print(f"\nSkipping numbers: no test videos found")

    # ===== WORDS =====
    words_ckpt = os.path.join(ckpt_dir, "v21_words", "best_model.pt")
    if os.path.exists(words_ckpt) and words_videos:
        print(f"\n{'='*70}")
        print(f"WORDS EVALUATION ({len(words_videos)} videos)")
        print(f"{'='*70}")

        ckpt = torch.load(words_ckpt, map_location=device, weights_only=False)
        config = ckpt["config"]
        adj = build_adj(config["num_nodes"]).to(device)
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 10)
        aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS)

        model = KSLGraphNetV21(
            nc=len(WORD_CLASSES),
            num_signers=num_signers,
            aux_dim=aux_dim,
            nn_=config["num_nodes"],
            ic=config["in_channels"],
            hd=config["hidden_dim"],
            nl=config["num_layers"],
            tk=tk,
            dr=config["dropout"],
            spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
            arcface_s=config.get("arcface_s", 30.0),
            arcface_m=config.get("arcface_m", 0.3),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded words model: {words_ckpt}")
        print(f"  Val accuracy: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        print(f"  Aux dim: {aux_dim} (angles={NUM_ANGLE_FEATURES}, fingertip_dists=20)")

        results["words"] = evaluate_category(
            "Words", words_videos, model, device, WORD_CLASSES
        )
    else:
        if not os.path.exists(words_ckpt):
            print(f"\nSkipping words: checkpoint not found at {words_ckpt}")
        elif not words_videos:
            print(f"\nSkipping words: no test videos found")

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V21 Real-World Evaluation")
    print(f"{'='*70}")

    if "numbers" in results:
        r = results["numbers"]
        print(f"  Numbers (TTA):    {r['overall_tta']:.1f}% ({r['correct_tta']}/{r['total']})")
        print(f"  Numbers (no-TTA): {r['overall_noTTA']:.1f}% ({r['correct_noTTA']}/{r['total']})")

    if "words" in results:
        r = results["words"]
        print(f"  Words (TTA):      {r['overall_tta']:.1f}% ({r['correct_tta']}/{r['total']})")
        print(f"  Words (no-TTA):   {r['overall_noTTA']:.1f}% ({r['correct_noTTA']}/{r['total']})")

    if "numbers" in results and "words" in results:
        combined_tta = (results["numbers"]["overall_tta"] + results["words"]["overall_tta"]) / 2
        combined_noTTA = (results["numbers"]["overall_noTTA"] + results["words"]["overall_noTTA"]) / 2
        print(f"\n  Combined (TTA):    {combined_tta:.1f}%")
        print(f"  Combined (no-TTA): {combined_noTTA:.1f}%")
        print(f"  TTA effect:        {combined_tta - combined_noTTA:+.1f}%")

    # Confidence summary
    print(f"\n  Confidence Summary (all predictions):")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        tot_c, tot_n = 0, 0
        for cat in ["numbers", "words"]:
            if cat in results and lvl in results[cat]["confidence_buckets"]:
                b = results[cat]["confidence_buckets"][lvl]
                tot_c += b["correct"]
                tot_n += b["total"]
        if tot_n > 0:
            print(f"    {lvl:>6s}: {tot_c}/{tot_n} = {100.0 * tot_c / tot_n:.0f}%")

    # Rejection analysis
    all_total = sum(results[c]["total"] for c in results)
    low_total = sum(results[c]["confidence_buckets"]["LOW"]["total"]
                    for c in results if "confidence_buckets" in results[c])
    accepted_correct = sum(
        results[c]["confidence_buckets"]["HIGH"]["correct"] +
        results[c]["confidence_buckets"]["MEDIUM"]["correct"]
        for c in results if "confidence_buckets" in results[c]
    )
    accepted_total = sum(
        results[c]["confidence_buckets"]["HIGH"]["total"] +
        results[c]["confidence_buckets"]["MEDIUM"]["total"]
        for c in results if "confidence_buckets" in results[c]
    )
    if accepted_total > 0:
        print(f"\n  If rejecting predictions below 0.4 confidence:")
        print(f"    Accepted: {accepted_total}/{all_total} "
              f"({100.0 * accepted_total / all_total:.0f}% of predictions)")
        print(f"    Accuracy on accepted: {accepted_correct}/{accepted_total} "
              f"= {100.0 * accepted_correct / accepted_total:.1f}%")
        print(f"    Rejected: {low_total}/{all_total} "
              f"({100.0 * low_total / all_total:.0f}% of predictions)")

    # Previous version comparison
    print(f"\n  Previous versions (for comparison):")
    print(f"    V19: Numbers 20.3% | Words 48.1% | Combined 34.2%")
    print(f"    V20: Numbers 20.3% | Words 40.7% | Combined 30.5%")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v21_real_testers_{ts_str}.json")

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    results_out = {
        "version": "v21",
        "evaluation_type": "real_testers",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tta_enabled": True,
        "tta_augmentations": [
            "original",
            "spatial_jitter_N(0,0.01)",
            "speed_0.9x",
            "speed_1.1x",
            "rotation_+5deg_Z",
        ],
        "normalization": "wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)",
        "features": "GCN: xyz(3)+velocity(3)+bone(3)=9ch | AUX: angles+fingertip_dists",
        "model": "KSLGraphNetV21 (ST-GCN + aux MLP + ArcFace + signer-adversarial)",
        "results": make_serializable(results),
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
