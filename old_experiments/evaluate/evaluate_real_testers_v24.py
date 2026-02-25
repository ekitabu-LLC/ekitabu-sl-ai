#!/usr/bin/env python3
"""
Evaluate v24 model on real-world test signers.

V24 changes from v22 evaluation:
- Prototypical classification (cosine similarity to class prototypes)
- Distance-based confidence (margin between top-1 and top-2 prototype distances)
- No TTA (confirmed to hurt in v22)
- Enrollment simulation (BN adaptation + prototype blending)

Usage:
    python evaluate_real_testers_v24.py
    python evaluate_real_testers_v24.py --enroll   # with enrollment simulation
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
# Constants (matching train_ksl_v24.py)
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

LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

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
LH_TIPS = [4, 8, 12, 16, 20]
RH_TIPS = [25, 29, 33, 37, 41]
NUM_FINGERTIP_PAIRS = 10

# ---------------------------------------------------------------------------
# Video -> Landmarks
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
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
# Preprocessing (matching v24 pipeline, no augmentation)
# ---------------------------------------------------------------------------

def compute_bones(h):
    bones = np.zeros_like(h)
    for child in range(48):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def compute_joint_angles(h):
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)
    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        bone_in = h[:, node, :] - h[:, parent, :]
        bone_out = h[:, child, :] - h[:, node, :]
        norm_in = np.maximum(np.linalg.norm(bone_in, axis=-1, keepdims=True), 1e-8)
        norm_out = np.maximum(np.linalg.norm(bone_out, axis=-1, keepdims=True), 1e-8)
        dot = np.sum((bone_in / norm_in) * (bone_out / norm_out), axis=-1)
        angles[:, idx] = np.arccos(np.clip(dot, -1.0, 1.0))
    return angles


def compute_fingertip_distances(h):
    T = h.shape[0]
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
    col = 0
    for tips in [LH_TIPS, RH_TIPS]:
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                diff = h[:, tips[i], :] - h[:, tips[j], :]
                distances[:, col] = np.linalg.norm(diff, axis=-1)
                col += 1
    return distances


def normalize_wrist_palm(h):
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        lh_wrist = lh[:, 0:1, :]
        lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]
        palm_sizes = np.maximum(np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        rh_wrist = rh[:, 0:1, :]
        rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]
        palm_sizes = np.maximum(np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] = pose[pose_valid] - mid_shoulder[pose_valid]
        sw = np.maximum(np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :],
                                        axis=-1, keepdims=True), 1e-6)
        pose[pose_valid] = pose[pose_valid] / sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


def preprocess_landmarks(raw, max_frames=MAX_FRAMES):
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
    h = normalize_wrist_palm(h)

    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h, velocity, bones = h[indices], velocity[indices], bones[indices]
        joint_angles, fingertip_dists = joint_angles[indices], fingertip_dists[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])

    gcn_features = np.clip(np.concatenate([h, velocity, bones], axis=2), -10, 10).astype(np.float32)
    gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)

    aux_features = np.clip(np.concatenate([joint_angles, fingertip_dists], axis=1), -10, 10).astype(np.float32)
    aux_tensor = torch.FloatTensor(aux_features)

    return gcn_tensor, aux_tensor


# ---------------------------------------------------------------------------
# Confidence (distance-based)
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
# Inference (v24: prototypical, no TTA)
# ---------------------------------------------------------------------------

def run_inference(model, raw, device, classes):
    """Run prototypical inference. No TTA."""
    i2c = {i: c for i, c in enumerate(classes)}

    gcn, aux = preprocess_landmarks(raw)
    if gcn is None:
        return None

    with torch.no_grad():
        gcn_dev = gcn.unsqueeze(0).to(device)
        aux_dev = aux.unsqueeze(0).to(device)

        # Get embedding
        embedding = model.get_embedding(gcn_dev, aux_dev)

        # Prototypical classification
        proto_logits = model.proto_classify(embedding)
        proto_probs = F.softmax(proto_logits, dim=1).cpu().squeeze(0)
        pred_idx = proto_probs.argmax().item()

        # Distance-based confidence
        confidence = model.proto_confidence(embedding).cpu().item()

        # Also get FC prediction for comparison
        logits_fc, _, _ = model(gcn_dev, aux_dev, grl_lambda=0.0)
        fc_probs = F.softmax(logits_fc, dim=1).cpu().squeeze(0)
        fc_pred_idx = fc_probs.argmax().item()

    return {
        "pred_class": i2c[pred_idx],
        "pred_idx": pred_idx,
        "confidence": confidence,
        "probs": proto_probs.numpy(),
        "fc_pred_class": i2c[fc_pred_idx],
        "fc_confidence": fc_probs[fc_pred_idx].item(),
    }


# ---------------------------------------------------------------------------
# Enrollment simulation
# ---------------------------------------------------------------------------

def simulate_enrollment(model, videos_by_signer_class, device, classes, n_enroll=3):
    """
    Simulate user enrollment: for each signer, use n_enroll samples per class
    to update prototypes, evaluate on remaining samples.

    Returns per-signer enrollment results.
    """
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    results = {}

    for signer, class_videos in videos_by_signer_class.items():
        print(f"\n  Enrollment simulation for signer: {signer}")

        # Split into enrollment and test
        enroll_embeddings = defaultdict(list)
        test_samples = []
        n_prior = 5  # prior strength for blending

        for true_class, video_list in class_videos.items():
            cls_idx = c2i.get(true_class, -1)
            if cls_idx < 0:
                continue

            # Process all videos for this class
            processed = []
            for video_path in video_list:
                raw = extract_landmarks_from_video(video_path)
                if raw is None:
                    continue
                gcn, aux = preprocess_landmarks(raw)
                if gcn is None:
                    continue
                processed.append((gcn, aux, video_path))

            if len(processed) <= n_enroll:
                # Not enough for enrollment + test; use all for enrollment, skip test
                for gcn, aux, vp in processed:
                    with torch.no_grad():
                        emb = model.get_embedding(
                            gcn.unsqueeze(0).to(device),
                            aux.unsqueeze(0).to(device)
                        ).cpu().squeeze(0)
                    enroll_embeddings[cls_idx].append(emb)
            else:
                # First n_enroll for enrollment, rest for test
                for gcn, aux, vp in processed[:n_enroll]:
                    with torch.no_grad():
                        emb = model.get_embedding(
                            gcn.unsqueeze(0).to(device),
                            aux.unsqueeze(0).to(device)
                        ).cpu().squeeze(0)
                    enroll_embeddings[cls_idx].append(emb)
                for gcn, aux, vp in processed[n_enroll:]:
                    test_samples.append((gcn, aux, true_class, vp))

        if not test_samples:
            print(f"    No test samples remaining after enrollment split")
            continue

        # Blend prototypes
        blended_prototypes = model.prototypes.cpu().clone()
        enrolled_classes = 0
        for cls_idx, embs in enroll_embeddings.items():
            if embs:
                user_proto = torch.stack(embs).mean(dim=0)
                n_user = len(embs)
                alpha = n_user / (n_user + n_prior)
                blended_prototypes[cls_idx] = (
                    alpha * user_proto + (1 - alpha) * blended_prototypes[cls_idx]
                )
                enrolled_classes += 1

        print(f"    Enrolled {enrolled_classes} classes, "
              f"{sum(len(v) for v in enroll_embeddings.values())} samples")
        print(f"    Testing on {len(test_samples)} samples")

        # Evaluate with blended prototypes
        correct, total = 0, 0
        blended_dev = blended_prototypes.to(device)

        for gcn, aux, true_class, vp in test_samples:
            with torch.no_grad():
                emb = model.get_embedding(
                    gcn.unsqueeze(0).to(device),
                    aux.unsqueeze(0).to(device)
                )
                emb_norm = F.normalize(emb, dim=1)
                proto_norm = F.normalize(blended_dev.unsqueeze(0), dim=2).squeeze(0)
                sims = torch.mm(emb_norm, proto_norm.t())
                pred_idx = sims.argmax(dim=1).item()
                pred_class = i2c[pred_idx]

            is_correct = (pred_class == true_class)
            correct += int(is_correct)
            total += 1

        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"    Accuracy with enrollment: {correct}/{total} = {acc:.1f}%")
        results[signer] = {"correct": correct, "total": total, "accuracy": acc,
                           "enrolled_classes": enrolled_classes}

    return results


# ---------------------------------------------------------------------------
# Evaluate one category
# ---------------------------------------------------------------------------

def evaluate_category(category_name, videos, model, device, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct, total = 0, 0
    correct_fc = 0
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

        print(f"    Extracted {raw.shape[0]} frames")

        result = run_inference(model, raw, device, classes)
        if result is None:
            print(f"    FAILED: Preprocessing error")
            continue

        pred_class = result["pred_class"]
        confidence = result["confidence"]
        conf_lvl = confidence_level(confidence)
        fc_pred = result["fc_pred_class"]

        is_correct = (pred_class == true_class)
        is_correct_fc = (fc_pred == true_class)
        correct += int(is_correct)
        correct_fc += int(is_correct_fc)
        total += 1

        true_idx = c2i.get(true_class, -1)
        pred_idx = c2i.get(pred_class, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct)

        if true_class not in per_class:
            per_class[true_class] = {"correct_proto": 0, "correct_fc": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct_proto"] += int(is_correct)
        per_class[true_class]["correct_fc"] += int(is_correct_fc)

        if signer not in per_signer:
            per_signer[signer] = {"correct_proto": 0, "correct_fc": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct_proto"] += int(is_correct)
        per_signer[signer]["correct_fc"] += int(is_correct_fc)

        status = "OK" if is_correct else "WRONG"
        fc_match = "" if pred_class == fc_pred else f" [FC={fc_pred}]"
        print(f"    Proto: {pred_class} (conf={confidence:.3f}, {conf_lvl}) | "
              f"True: {true_class} [{status}]{fc_match}")

        # Top-3
        probs = torch.FloatTensor(result["probs"])
        top3 = torch.topk(probs, min(3, len(classes)))
        top3_str = ", ".join([f"{i2c[idx.item()]}={prob.item():.3f}"
                              for idx, prob in zip(top3.indices, top3.values)])
        print(f"    Top-3: {top3_str}")

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred_proto": pred_class,
            "pred_fc": fc_pred,
            "correct_proto": is_correct,
            "correct_fc": is_correct_fc,
            "confidence": confidence,
            "confidence_level": conf_lvl,
        })

    acc_proto = 100.0 * correct / total if total > 0 else 0.0
    acc_fc = 100.0 * correct_fc / total if total > 0 else 0.0

    print(f"\n  {category_name.upper()} OVERALL (Proto):  {correct}/{total} = {acc_proto:.1f}%")
    print(f"  {category_name.upper()} OVERALL (FC):     {correct_fc}/{total} = {acc_fc:.1f}%")
    print(f"  Proto vs FC: {acc_proto - acc_fc:+.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'Proto':>7s}  {'FC':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            acc_p = 100.0 * pc["correct_proto"] / pc["total"]
            acc_f = 100.0 * pc["correct_fc"] / pc["total"]
            print(f"    {cls:>12s}  {acc_p:5.0f}%   {acc_f:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    print(f"    {'Signer':>20s}  {'Proto':>7s}  {'FC':>7s}  {'N':>3s}")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        acc_p = 100.0 * ps["correct_proto"] / ps["total"]
        acc_f = 100.0 * ps["correct_fc"] / ps["total"]
        print(f"    {signer:>20s}  {acc_p:5.0f}%   {acc_f:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence analysis (distance-based):")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {acc:.0f}%")
        else:
            print(f"    {lvl:>6s}: 0 predictions")

    # Rejection analysis
    high_med_correct = conf_buckets["HIGH"]["correct"] + conf_buckets["MEDIUM"]["correct"]
    high_med_total = conf_buckets["HIGH"]["total"] + conf_buckets["MEDIUM"]["total"]
    if high_med_total > 0:
        filtered_acc = 100.0 * high_med_correct / high_med_total
        print(f"\n    Rejecting LOW confidence predictions:")
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
        "overall_proto": acc_proto,
        "overall_fc": acc_fc,
        "correct_proto": correct,
        "correct_fc": correct_fc,
        "total": total,
        "proto_vs_fc": acc_proto - acc_fc,
        "per_class": {k: {
            "acc_proto": 100.0 * v["correct_proto"] / v["total"],
            "acc_fc": 100.0 * v["correct_fc"] / v["total"],
            **v
        } for k, v in per_class.items()},
        "per_signer": {k: {
            "acc_proto": 100.0 * v["correct_proto"] / v["total"],
            "acc_fc": 100.0 * v["correct_fc"] / v["total"],
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
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", action="store_true",
                        help="Run enrollment simulation")
    parser.add_argument("--enroll-n", type=int, default=3,
                        help="Number of enrollment samples per class")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    print("=" * 70)
    print("V24 Real-World Evaluation (Prototypical + Distance Confidence)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Enrollment: {'enabled (n={})'.format(args.enroll_n) if args.enroll else 'disabled'}")
    print("=" * 70)

    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    print(f"Found {len(videos)} test videos")

    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"  Numbers: {len(numbers_videos)} videos")
    print(f"  Words: {len(words_videos)} videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from train_ksl_v24 import KSLGraphNetV24, build_adj

    results = {}
    enrollment_results = {}

    # ===== NUMBERS =====
    numbers_ckpt = os.path.join(ckpt_dir, "v24_numbers", "best_model.pt")
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

        model = KSLGraphNetV24(
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
            mixstyle_p=config.get("mixstyle_p", 0.5),
            mixstyle_alpha=config.get("mixstyle_alpha", 0.1),
            mixstyle_layers=config.get("mixstyle_layers", [0, 1]),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded numbers model: {numbers_ckpt}")
        print(f"  Val acc: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        results["numbers"] = evaluate_category(
            "Numbers", numbers_videos, model, device, NUMBER_CLASSES
        )

        if args.enroll:
            # Group videos by signer and class
            vids_by_signer = defaultdict(lambda: defaultdict(list))
            for vp, name, signer in numbers_videos:
                vids_by_signer[signer][name].append(vp)
            enrollment_results["numbers"] = simulate_enrollment(
                model, vids_by_signer, device, NUMBER_CLASSES, n_enroll=args.enroll_n
            )
    else:
        if not os.path.exists(numbers_ckpt):
            print(f"\nSkipping numbers: checkpoint not found at {numbers_ckpt}")

    # ===== WORDS =====
    words_ckpt = os.path.join(ckpt_dir, "v24_words", "best_model.pt")
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

        model = KSLGraphNetV24(
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
            mixstyle_p=config.get("mixstyle_p", 0.5),
            mixstyle_alpha=config.get("mixstyle_alpha", 0.1),
            mixstyle_layers=config.get("mixstyle_layers", [0, 1]),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded words model: {words_ckpt}")
        print(f"  Val acc: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        results["words"] = evaluate_category(
            "Words", words_videos, model, device, WORD_CLASSES
        )

        if args.enroll:
            vids_by_signer = defaultdict(lambda: defaultdict(list))
            for vp, name, signer in words_videos:
                vids_by_signer[signer][name].append(vp)
            enrollment_results["words"] = simulate_enrollment(
                model, vids_by_signer, device, WORD_CLASSES, n_enroll=args.enroll_n
            )
    else:
        if not os.path.exists(words_ckpt):
            print(f"\nSkipping words: checkpoint not found at {words_ckpt}")

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V24 Real-World Evaluation")
    print(f"{'='*70}")

    if "numbers" in results:
        r = results["numbers"]
        print(f"  Numbers (Proto):  {r['overall_proto']:.1f}% ({r['correct_proto']}/{r['total']})")
        print(f"  Numbers (FC):     {r['overall_fc']:.1f}% ({r['correct_fc']}/{r['total']})")

    if "words" in results:
        r = results["words"]
        print(f"  Words (Proto):    {r['overall_proto']:.1f}% ({r['correct_proto']}/{r['total']})")
        print(f"  Words (FC):       {r['overall_fc']:.1f}% ({r['correct_fc']}/{r['total']})")

    if "numbers" in results and "words" in results:
        comb_proto = (results["numbers"]["overall_proto"] + results["words"]["overall_proto"]) / 2
        comb_fc = (results["numbers"]["overall_fc"] + results["words"]["overall_fc"]) / 2
        print(f"\n  Combined (Proto): {comb_proto:.1f}%")
        print(f"  Combined (FC):    {comb_fc:.1f}%")
        print(f"  Proto vs FC:      {comb_proto - comb_fc:+.1f}%")

    # Confidence summary
    print(f"\n  Distance-Based Confidence Summary:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        tot_c, tot_n = 0, 0
        for cat in ["numbers", "words"]:
            if cat in results and lvl in results[cat]["confidence_buckets"]:
                b = results[cat]["confidence_buckets"][lvl]
                tot_c += b["correct"]
                tot_n += b["total"]
        if tot_n > 0:
            print(f"    {lvl:>6s}: {tot_c}/{tot_n} = {100.0 * tot_c / tot_n:.0f}%")

    # Enrollment results
    if enrollment_results:
        print(f"\n  Enrollment Simulation (n={args.enroll_n} per class):")
        for cat in ["numbers", "words"]:
            if cat in enrollment_results:
                er = enrollment_results[cat]
                total_c = sum(v["correct"] for v in er.values())
                total_n = sum(v["total"] for v in er.values())
                if total_n > 0:
                    acc = 100.0 * total_c / total_n
                    print(f"    {cat.capitalize()} with enrollment: {total_c}/{total_n} = {acc:.1f}%")
                    base = results[cat]["overall_proto"]
                    print(f"    Improvement: {acc - base:+.1f}% over base proto")

    # Comparison with previous versions
    print(f"\n  Previous versions:")
    print(f"    V22 (no-TTA): Numbers 33.9% | Words 45.7% | Combined 39.8%")
    print(f"    V23 (no-TTA): Numbers 33.9% | Words 42.0% | Combined 37.9%")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v24_real_testers_{ts_str}.json")

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    results_out = {
        "version": "v24",
        "evaluation_type": "real_testers",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tta_enabled": False,
        "classification": "prototypical (cosine similarity to class prototypes)",
        "confidence": "distance-based (margin between top-1 and top-2 prototype distances)",
        "enrollment": {"enabled": args.enroll, "n_per_class": args.enroll_n} if args.enroll else None,
        "model": "KSLGraphNetV24 (v22 backbone + MixStyle + prototypical inference)",
        "results": make_serializable(results),
        "enrollment_results": make_serializable(enrollment_results) if enrollment_results else None,
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
