#!/usr/bin/env python3
"""
Evaluate v28 multi-stream model on real-world test signers with AdaBN.

V28 changes from v27:
- Multi-stream late fusion: 3 separate models (joint/bone/velocity) per category
- AdaBN: Adapt BatchNorm statistics to real-tester distribution at test time
- Three evaluation modes: baseline (no AdaBN), adabn_global, adabn_per_signer

Usage:
    python evaluate_real_testers_v28.py
"""

import copy
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import mediapipe as mp

# ---------------------------------------------------------------------------
# MediaPipe version check
# ---------------------------------------------------------------------------

def check_mediapipe_version():
    version = mp.__version__
    expected = "0.10.5"
    print(f"  MediaPipe version: {version}")
    if version == expected:
        print(f"  (matches v28 training data extraction version)")
    else:
        print(f"  WARNING: Expected {expected}. Landmark coordinates may differ.")


# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v28.py)
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
NUM_HAND_BODY_FEATURES = 8

# ---------------------------------------------------------------------------
# Video -> Landmarks extraction
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
# Preprocessing (matching v28 multi-stream pipeline)
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


def compute_hand_body_features(h_raw):
    T = h_raw.shape[0]
    features = np.zeros((T, 8), dtype=np.float32)
    mid_shoulder = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2
    shoulder_width = np.maximum(
        np.linalg.norm(h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True), 1e-6
    )
    lh_centroid = h_raw[:, :21, :].mean(axis=1)
    rh_centroid = h_raw[:, 21:42, :].mean(axis=1)
    features[:, 0] = (lh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]
    features[:, 1] = (rh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]
    features[:, 2] = (lh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
    features[:, 3] = (rh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
    features[:, 4] = np.linalg.norm(lh_centroid - rh_centroid, axis=-1) / shoulder_width[:, 0]
    face_approx = mid_shoulder.copy()
    face_approx[:, 1] -= shoulder_width[:, 0] * 0.7
    features[:, 5] = np.linalg.norm(lh_centroid - face_approx, axis=-1) / shoulder_width[:, 0]
    features[:, 6] = np.linalg.norm(rh_centroid - face_approx, axis=-1) / shoulder_width[:, 0]
    features[:, 7] = np.abs(lh_centroid[:, 1] - rh_centroid[:, 1]) / shoulder_width[:, 0]
    return features


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
        sw = np.maximum(np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True), 1e-6)
        pose[pose_valid] = pose[pose_valid] / sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


def preprocess_multistream(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (num_frames, 225) landmarks to multi-stream format.

    Returns:
        streams: dict of {stream_name: (3, max_frames, 48) tensor}
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

    # Hand-body features BEFORE normalization
    hand_body_feats = compute_hand_body_features(h)

    # Normalization
    h = normalize_wrist_palm(h)

    # Velocity
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]

    # Bones
    bones = compute_bones(h)

    # Aux features
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
        hand_body_feats = hand_body_feats[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
        hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

    # Per-stream tensors
    streams = {
        "joint": torch.FloatTensor(np.clip(h, -10, 10)).permute(2, 0, 1),
        "bone": torch.FloatTensor(np.clip(bones, -10, 10)).permute(2, 0, 1),
        "velocity": torch.FloatTensor(np.clip(velocity, -10, 10)).permute(2, 0, 1),
    }

    # Aux
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_tensor = torch.FloatTensor(np.clip(aux_features, -10, 10))

    return streams, aux_tensor


# ---------------------------------------------------------------------------
# AdaBN: Adaptive Batch Normalization (v28 NEW)
# ---------------------------------------------------------------------------

def adapt_bn_stats(model, data_list, device):
    """Reset BN stats and forward-pass target data to collect new statistics.

    Args:
        model: nn.Module with BatchNorm layers
        data_list: list of (streams_dict, aux_tensor) tuples (per-sample, not batched)
        device: torch device
    """
    if len(data_list) == 0:
        return

    # Reset running stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()

    # Forward pass in train mode (BN updates running_mean/running_var)
    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()


# ---------------------------------------------------------------------------
# Multi-stream inference
# ---------------------------------------------------------------------------

def run_multistream_inference(models, fusion_weights, streams, aux, device, classes):
    """Run inference through all stream models and fuse predictions.

    Args:
        models: dict of {stream_name: model}
        fusion_weights: dict of {stream_name: weight}
        streams: dict of {stream_name: (3, T, N) tensor}
        aux: (T, D_aux) tensor
        device: torch device
        classes: list of class names

    Returns:
        pred_class, confidence, fused_probs, per_stream_probs
    """
    i2c = {i: c for i, c in enumerate(classes)}
    per_stream_probs = {}

    with torch.no_grad():
        for sname, model in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, _, _ = model(gcn, aux_t, grl_lambda=0.0)
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
            per_stream_probs[sname] = probs

    # Fuse
    fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
    pred_idx = fused.argmax().item()
    confidence = fused[pred_idx].item()
    pred_class = i2c[pred_idx]

    return pred_class, confidence, fused, per_stream_probs


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
# Evaluate one category with multi-stream fusion
# ---------------------------------------------------------------------------

def evaluate_category_multistream(category_name, videos, models, fusion_weights,
                                   device, classes, mode_name="baseline"):
    """Evaluate with multi-stream fusion.

    Args:
        models: dict of {stream_name: model} (already adapted for AdaBN if needed)
        fusion_weights: dict of {stream_name: weight}
        mode_name: "baseline", "adabn_global", or "adabn_per_signer"
    """
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct, total = 0, 0
    per_class = {}
    per_signer = {}
    nc = len(classes)
    confusion = [[0] * nc for _ in range(nc)]
    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}
    details = []

    for video_path, true_class, signer in videos:
        print(f"\n  [{mode_name}] Processing: {os.path.basename(video_path)} "
              f"(class={true_class}, {signer})")

        # Use pre-extracted landmarks if available
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: Could not extract landmarks")
                continue

        streams, aux = preprocess_multistream(raw)
        if streams is None:
            print(f"    FAILED: Preprocessing error")
            continue

        pred_class, confidence, fused_probs, per_stream_probs = \
            run_multistream_inference(models, fusion_weights, streams, aux, device, classes)

        conf_lvl = confidence_level(confidence)
        is_correct = (pred_class == true_class)
        correct += int(is_correct)
        total += 1

        true_idx = c2i.get(true_class, -1)
        pred_idx = c2i.get(pred_class, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct)

        if true_class not in per_class:
            per_class[true_class] = {"correct": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct"] += int(is_correct)

        if signer not in per_signer:
            per_signer[signer] = {"correct": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct"] += int(is_correct)

        status = "OK" if is_correct else "WRONG"
        print(f"    Pred: {pred_class} (conf={confidence:.3f}, {conf_lvl}) | "
              f"True: {true_class} [{status}]")

        # Per-stream predictions (diagnostic)
        stream_preds = {}
        for sname, sprobs in per_stream_probs.items():
            sidx = sprobs.argmax().item()
            stream_preds[sname] = i2c[sidx]

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred": pred_class,
            "correct": is_correct,
            "confidence": confidence,
            "confidence_level": conf_lvl,
            "stream_preds": stream_preds,
        })

    # Summary
    acc = 100.0 * correct / total if total > 0 else 0.0

    print(f"\n  {category_name.upper()} [{mode_name}]: {correct}/{total} = {acc:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'Acc':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            cls_acc = 100.0 * pc["correct"] / pc["total"]
            print(f"    {cls:>12s}  {cls_acc:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        s_acc = 100.0 * ps["correct"] / ps["total"]
        print(f"    {signer:>20s}  {s_acc:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            b_acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {b_acc:.0f}%")

    # Rejection analysis
    high_med_correct = conf_buckets["HIGH"]["correct"] + conf_buckets["MEDIUM"]["correct"]
    high_med_total = conf_buckets["HIGH"]["total"] + conf_buckets["MEDIUM"]["total"]
    if high_med_total > 0:
        filtered_acc = 100.0 * high_med_correct / high_med_total
        print(f"\n    If rejecting below 0.4 confidence:")
        print(f"      Accuracy: {high_med_correct}/{high_med_total} = {filtered_acc:.1f}%")
        print(f"      Rejection rate: {conf_buckets['LOW']['total']}/{total} "
              f"= {100.0 * conf_buckets['LOW']['total'] / total:.0f}%")

    return {
        "mode": mode_name,
        "overall": acc,
        "correct": correct,
        "total": total,
        "per_class": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"],
            **v
        } for k, v in per_class.items()},
        "per_signer": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"],
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
# Load stream models
# ---------------------------------------------------------------------------

def load_stream_models(ckpt_dir, classes, device, streams=("joint", "bone", "velocity")):
    """Load all stream models and fusion weights for one category."""
    from train_ksl_v28 import KSLGraphNetV25, build_adj

    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in streams:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing checkpoint for {sname}: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV25(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
            nn_=config["num_nodes"],
            ic=config["in_channels"],  # 3
            hd=config["hidden_dim"],
            nl=config["num_layers"],
            tk=tk,
            dr=config["dropout"],
            spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {sname}: val_acc={ckpt['val_acc']:.1f}%, "
              f"epoch={ckpt['epoch']}, params={param_count:,}")

    # Load fusion weights
    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            fusion_data = json.load(f)
        fusion_weights = fusion_data["weights"]
        print(f"  Fusion weights: {fusion_weights} (val_acc={fusion_data['val_accuracy']:.1f}%)")
    else:
        # Default weights
        fusion_weights = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
        print(f"  Using default fusion weights: {fusion_weights}")

    return models, fusion_weights


# ---------------------------------------------------------------------------
# Pre-extract all test data for AdaBN
# ---------------------------------------------------------------------------

def preextract_test_data(videos, stream_name):
    """Pre-extract and preprocess all test videos for one stream.

    Returns list of (stream_tensor, aux_tensor) tuples.
    """
    data_list = []
    for video_path, true_class, signer in videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                continue

        streams, aux = preprocess_multistream(raw)
        if streams is None:
            continue

        data_list.append((streams[stream_name], aux))

    return data_list


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    print("=" * 70)
    print("V28 Real-World Evaluation (Multi-Stream Fusion + AdaBN)")
    print(f"Streams: joint, bone, velocity (ic=3 each)")
    print(f"AdaBN modes: baseline, global, per-signer")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    check_mediapipe_version()

    # Discover videos
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

    all_results = {}

    for cat_name, cat_videos, classes in [
        ("numbers", numbers_videos, NUMBER_CLASSES),
        ("words", words_videos, WORD_CLASSES),
    ]:
        cat_ckpt_dir = os.path.join(ckpt_dir, f"v28_{cat_name}")

        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos")
            continue

        if not os.path.isdir(cat_ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {cat_ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} EVALUATION ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        models, fusion_weights = load_stream_models(cat_ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  WARNING: Only {len(models)}/3 stream models loaded, skipping")
            continue

        # ===== MODE 1: Baseline (no AdaBN) =====
        print(f"\n--- Mode: BASELINE (no AdaBN) ---")
        baseline_result = evaluate_category_multistream(
            cat_name, cat_videos, models, fusion_weights, device, classes,
            mode_name="baseline"
        )

        # ===== MODE 2: AdaBN Global =====
        print(f"\n--- Mode: AdaBN GLOBAL ---")
        # Deep copy models for AdaBN
        adabn_models = {sname: copy.deepcopy(model) for sname, model in models.items()}

        # Pre-extract test data and adapt BN stats for each stream
        for sname in adabn_models:
            print(f"  Adapting BN stats for {sname} stream...")
            stream_data = preextract_test_data(cat_videos, sname)
            adapt_bn_stats(adabn_models[sname], stream_data, device)
            print(f"    Adapted on {len(stream_data)} samples")

        adabn_global_result = evaluate_category_multistream(
            cat_name, cat_videos, adabn_models, fusion_weights, device, classes,
            mode_name="adabn_global"
        )

        # ===== MODE 3: AdaBN Per-Signer =====
        print(f"\n--- Mode: AdaBN PER-SIGNER ---")
        # Group videos by signer
        signer_videos = defaultdict(list)
        for v, n, s in cat_videos:
            signer_videos[s].append((v, n, s))

        per_signer_all_details = []
        per_signer_correct = 0
        per_signer_total = 0

        for signer, signer_vids in sorted(signer_videos.items()):
            print(f"\n  AdaBN for signer: {signer} ({len(signer_vids)} videos)")

            # Deep copy and adapt per signer
            signer_models = {sname: copy.deepcopy(model) for sname, model in models.items()}
            for sname in signer_models:
                signer_data = preextract_test_data(signer_vids, sname)
                adapt_bn_stats(signer_models[sname], signer_data, device)

            # Evaluate this signer's videos
            for video_path, true_class, signer_name in signer_vids:
                npy_cache = video_path + ".landmarks.npy"
                if os.path.exists(npy_cache):
                    raw = np.load(npy_cache)
                else:
                    raw = extract_landmarks_from_video(video_path)
                    if raw is None:
                        continue

                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    continue

                pred_class, confidence, _, _ = run_multistream_inference(
                    signer_models, fusion_weights, streams, aux, device, classes
                )

                is_correct = (pred_class == true_class)
                per_signer_correct += int(is_correct)
                per_signer_total += 1

                status = "OK" if is_correct else "WRONG"
                print(f"    {os.path.basename(video_path)}: {pred_class} "
                      f"(conf={confidence:.3f}) | True: {true_class} [{status}]")

                per_signer_all_details.append({
                    "video": os.path.basename(video_path),
                    "signer": signer_name,
                    "true": true_class,
                    "pred": pred_class,
                    "correct": is_correct,
                    "confidence": confidence,
                })

        adabn_per_signer_acc = 100.0 * per_signer_correct / per_signer_total if per_signer_total > 0 else 0.0
        print(f"\n  {cat_name.upper()} AdaBN per-signer: "
              f"{per_signer_correct}/{per_signer_total} = {adabn_per_signer_acc:.1f}%")

        adabn_per_signer_result = {
            "mode": "adabn_per_signer",
            "overall": adabn_per_signer_acc,
            "correct": per_signer_correct,
            "total": per_signer_total,
            "details": per_signer_all_details,
        }

        # ===== Per-stream diagnostic =====
        print(f"\n--- Per-stream accuracy (diagnostic, no fusion) ---")
        for sname in models:
            stream_correct = 0
            stream_total = 0
            for video_path, true_class, signer in cat_videos:
                npy_cache = video_path + ".landmarks.npy"
                if os.path.exists(npy_cache):
                    raw = np.load(npy_cache)
                else:
                    raw = extract_landmarks_from_video(video_path)
                    if raw is None:
                        continue

                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    continue

                with torch.no_grad():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    aux_t = aux.unsqueeze(0).to(device)
                    logits, _, _ = models[sname](gcn, aux_t, grl_lambda=0.0)
                    pred_idx = logits.argmax(dim=1).item()
                    pred_class = classes[pred_idx]

                stream_correct += int(pred_class == true_class)
                stream_total += 1

            stream_acc = 100.0 * stream_correct / stream_total if stream_total > 0 else 0.0
            print(f"  {sname:>10s}: {stream_correct}/{stream_total} = {stream_acc:.1f}%")

        # ===== Mode comparison =====
        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} MODE COMPARISON:")
        print(f"  Baseline (no AdaBN):  {baseline_result['overall']:.1f}%")
        print(f"  AdaBN Global:         {adabn_global_result['overall']:.1f}%")
        print(f"  AdaBN Per-Signer:     {adabn_per_signer_acc:.1f}%")

        all_results[cat_name] = {
            "baseline": baseline_result,
            "adabn_global": adabn_global_result,
            "adabn_per_signer": adabn_per_signer_result,
            "fusion_weights": fusion_weights,
        }

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V28 Real-World Evaluation")
    print(f"{'='*70}")

    for mode_name in ["baseline", "adabn_global", "adabn_per_signer"]:
        nums = all_results.get("numbers", {}).get(mode_name, {}).get("overall", 0)
        wrds = all_results.get("words", {}).get(mode_name, {}).get("overall", 0)
        combined = (nums + wrds) / 2 if nums and wrds else 0
        print(f"\n  {mode_name:>20s}:  Numbers={nums:.1f}%  Words={wrds:.1f}%  "
              f"Combined={combined:.1f}%")

    # Comparison with previous versions
    print(f"\n  Previous versions (no-TTA):")
    print(f"    V22: Numbers 33.9% | Words 45.7% | Combined 39.8%")
    print(f"    V26: Numbers 45.8% | Words 49.4% | Combined 47.6%")
    print(f"    V27: Numbers 54.2% | Words 53.1% | Combined 53.7%  <-- previous best")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v28_real_testers_{ts_str}.json")

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
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    results_out = {
        "version": "v28",
        "evaluation_type": "real_testers",
        "training_data": "ksl-alpha (12 signers train [1-12], 3 val [13-15])",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "Multi-stream late fusion: 3x KSLGraphNetV25(ic=3)",
        "streams": ["joint", "bone", "velocity"],
        "adabn_modes": ["baseline", "adabn_global", "adabn_per_signer"],
        "v28_changes": [
            "Multi-stream late fusion (3 streams x 3ch instead of 1 x 9ch)",
            "AdaBN: Adaptive BatchNorm at test time (global + per-signer modes)",
            "Fusion weight learning: grid search on val set",
        ],
        "results": make_serializable(all_results),
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
