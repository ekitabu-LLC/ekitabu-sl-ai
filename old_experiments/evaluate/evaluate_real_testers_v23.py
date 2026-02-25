#!/usr/bin/env python3
"""
Evaluate v23 multi-stream ensemble model on real-world test signers.

V23 changes from v22:
- Multi-stream: 4 independent 3-channel models (joint, bone, velocity, bone_velocity)
- Late fusion with learned per-stream weights
- Post-hoc calibration (per-class temperature + bias)
- NO TTA (removed entirely, v22 analysis showed TTA hurts)
- Per-stream agreement analysis
- Top-3 predictions for each sample

Usage:
    python evaluate_real_testers_v23.py
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
# Constants (matching train_ksl_v23.py)
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

STREAM_NAMES = ["joint", "bone", "velocity", "bone_velocity"]

MAX_FRAMES = 90

# Joint angle topology (same as train_ksl_v23.py)
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
# Preprocessing (matching v23 dataset pipeline, no augmentation)
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
    """Wrist-centric + palm-size normalization (matching v23 training)."""
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


def preprocess_landmarks_multistream(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (num_frames, 225) landmarks to v23 multi-stream format.

    Returns:
        streams: dict of {stream_name: (3, max_frames, 48) tensor}
        aux_tensor: (max_frames, D_aux) tensor
    Or (None, None) on failure.
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

    # Compute stream features
    joint_xyz = h.copy()

    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]

    bones = compute_bones(h)

    bone_velocity = np.zeros_like(bones)
    bone_velocity[1:] = bones[1:] - bones[:-1]

    # Auxiliary features
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    # Temporal sampling / padding
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        joint_xyz = joint_xyz[indices]
        velocity = velocity[indices]
        bones = bones[indices]
        bone_velocity = bone_velocity[indices]
        joint_angles = joint_angles[indices]
        fingertip_dists = fingertip_dists[indices]
    else:
        pad_len = max_frames - f
        z48 = np.zeros((pad_len, 48, 3), dtype=np.float32)
        joint_xyz = np.concatenate([joint_xyz, z48])
        velocity = np.concatenate([velocity, z48])
        bones = np.concatenate([bones, z48])
        bone_velocity = np.concatenate([bone_velocity, z48])
        joint_angles = np.concatenate([joint_angles,
                                       np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists,
                                          np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])

    # Clip
    joint_xyz = np.clip(joint_xyz, -10, 10).astype(np.float32)
    velocity = np.clip(velocity, -10, 10).astype(np.float32)
    bones = np.clip(bones, -10, 10).astype(np.float32)
    bone_velocity = np.clip(bone_velocity, -10, 10).astype(np.float32)

    # Build stream tensors: each (C=3, T=max_frames, N=48)
    streams = {
        "joint": torch.FloatTensor(joint_xyz).permute(2, 0, 1),
        "bone": torch.FloatTensor(bones).permute(2, 0, 1),
        "velocity": torch.FloatTensor(velocity).permute(2, 0, 1),
        "bone_velocity": torch.FloatTensor(bone_velocity).permute(2, 0, 1),
    }

    # Aux features: (max_frames, D_aux)
    aux_features = np.concatenate([joint_angles, fingertip_dists], axis=1)
    aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
    aux_tensor = torch.FloatTensor(aux_features)

    return streams, aux_tensor


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
# Load multi-stream models for one category
# ---------------------------------------------------------------------------

def load_stream_models(ckpt_base_dir, classes, device):
    """Load 4 stream models + fusion weights + calibration for a category.

    Returns:
        models: dict {stream_name: model}
        fusion_weights: dict {stream_name: float}
        calibration: dict with temperatures and biases
    """
    from train_ksl_v23 import KSLGraphNetV23, build_adj

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS
    models = {}
    total_params = 0

    for sname in STREAM_NAMES:
        ckpt_path = os.path.join(ckpt_base_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing checkpoint for stream {sname}: {ckpt_path}")
            return None, None, None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        adj = build_adj(config["num_nodes"]).to(device)
        ic = ckpt.get("in_channels", 3)
        num_signers = ckpt.get("num_signers", 4)

        model = KSLGraphNetV23(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
            nn_=config["num_nodes"],
            ic=ic,
            hd=config["hidden_dim"],
            nl=config["num_layers"],
            tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
            dp=config["dropout"],
            sd=config.get("spatial_dropout", 0.1),
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"  Loaded {sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={params:,}")
        models[sname] = model

    # Load fusion weights
    fusion_path = os.path.join(ckpt_base_dir, "fusion_weights.json")
    if os.path.exists(fusion_path):
        with open(fusion_path) as f:
            fusion_data = json.load(f)
        fusion_weights = fusion_data["weights"]
        print(f"  Fusion weights: {fusion_weights} (val_acc={fusion_data.get('val_acc', 'N/A'):.1f}%)")
    else:
        print(f"  WARNING: No fusion weights found at {fusion_path}, using equal weights")
        fusion_weights = {sname: 0.25 for sname in STREAM_NAMES}

    # Load calibration
    cal_path = os.path.join(ckpt_base_dir, "calibration.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            calibration = json.load(f)
        print(f"  Calibration loaded: uncal={calibration.get('val_acc_uncalibrated', 'N/A'):.1f}% "
              f"-> cal={calibration.get('val_acc_calibrated', 'N/A'):.1f}%")
    else:
        print(f"  WARNING: No calibration found at {cal_path}, using no calibration")
        calibration = None

    print(f"  Total parameters: {total_params:,}")

    return models, fusion_weights, calibration


# ---------------------------------------------------------------------------
# Multi-stream inference (NO TTA)
# ---------------------------------------------------------------------------

def run_multistream_inference(models, fusion_weights, calibration, raw, device,
                              classes):
    """Run multi-stream fusion inference on one video. No TTA.

    Returns dict with predictions, per-stream details, top-3, etc.
    """
    i2c = {i: c for i, c in enumerate(classes)}
    c2i = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    streams, aux = preprocess_landmarks_multistream(raw)
    if streams is None:
        return None

    # Get per-stream logits and softmax
    per_stream_logits = {}
    per_stream_probs = {}
    per_stream_preds = {}

    with torch.no_grad():
        for sname in STREAM_NAMES:
            gcn_data = streams[sname].unsqueeze(0).to(device)
            aux_data = aux.unsqueeze(0).to(device)
            logits, _, _ = models[sname](gcn_data, aux_data, grl_lambda=0.0)
            per_stream_logits[sname] = logits.cpu().squeeze(0).numpy()
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
            per_stream_probs[sname] = probs.numpy()
            per_stream_preds[sname] = i2c[probs.argmax().item()]

    # Weighted fusion of softmax outputs
    fused_probs = sum(
        fusion_weights[sname] * per_stream_probs[sname]
        for sname in STREAM_NAMES
    )

    # Apply calibration if available
    if calibration is not None:
        # Fuse logits (not softmax) for calibration
        fused_logits = sum(
            fusion_weights[sname] * per_stream_logits[sname]
            for sname in STREAM_NAMES
        )

        temps = np.array([calibration["temperatures"][c] for c in classes])
        biases = np.array([calibration["biases"][c] for c in classes])
        cal_logits = fused_logits / temps - biases

        # Softmax on calibrated logits
        cal_logits_shifted = cal_logits - cal_logits.max()
        cal_exp = np.exp(cal_logits_shifted)
        cal_probs = cal_exp / (cal_exp.sum() + 1e-10)
    else:
        cal_probs = fused_probs

    pred_idx = int(np.argmax(cal_probs))
    confidence = float(cal_probs[pred_idx])
    pred_class = i2c[pred_idx]

    # Also get uncalibrated prediction for comparison
    uncal_pred_idx = int(np.argmax(fused_probs))
    uncal_confidence = float(fused_probs[uncal_pred_idx])
    uncal_pred_class = i2c[uncal_pred_idx]

    # Per-stream agreement analysis
    stream_pred_list = [per_stream_preds[sname] for sname in STREAM_NAMES]
    all_agree = len(set(stream_pred_list)) == 1
    majority_pred = max(set(stream_pred_list), key=stream_pred_list.count)
    agreement_count = stream_pred_list.count(majority_pred)

    # Top-3 predictions
    top3_indices = np.argsort(cal_probs)[::-1][:3]
    top3 = [(i2c[int(idx)], float(cal_probs[idx])) for idx in top3_indices]

    return {
        "pred_class": pred_class,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "probs": cal_probs,
        "uncal_pred_class": uncal_pred_class,
        "uncal_confidence": uncal_confidence,
        "per_stream_preds": per_stream_preds,
        "per_stream_probs": per_stream_probs,
        "all_streams_agree": all_agree,
        "agreement_count": agreement_count,
        "majority_pred": majority_pred,
        "top3": top3,
    }


# ---------------------------------------------------------------------------
# Evaluate one category
# ---------------------------------------------------------------------------

def evaluate_category(category_name, videos, models, fusion_weights, calibration,
                      device, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct, total = 0, 0
    per_class = {}
    per_signer = {}
    details = []

    nc = len(classes)
    confusion = [[0] * nc for _ in range(nc)]

    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}

    # Stream agreement tracking
    total_all_agree = 0
    total_majority_correct = 0

    # Per-stream individual accuracy (which stream is best alone?)
    per_stream_correct = {sn: 0 for sn in STREAM_NAMES}
    per_stream_total = {sn: 0 for sn in STREAM_NAMES}

    for video_path, true_class, signer in videos:
        print(f"\n  Processing: {os.path.basename(video_path)} (class={true_class}, {signer})")

        raw = extract_landmarks_from_video(video_path)
        if raw is None:
            print(f"    FAILED: Could not extract landmarks")
            continue

        print(f"    Extracted {raw.shape[0]} frames, shape {raw.shape}")

        result = run_multistream_inference(
            models, fusion_weights, calibration, raw, device, classes
        )
        if result is None:
            print(f"    FAILED: Preprocessing error")
            continue

        pred_class = result["pred_class"]
        confidence = result["confidence"]
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

        # Stream agreement
        if result["all_streams_agree"]:
            total_all_agree += 1
        if result["majority_pred"] == true_class:
            total_majority_correct += 1

        # Per-stream individual accuracy
        for sn in STREAM_NAMES:
            per_stream_total[sn] += 1
            if result["per_stream_preds"][sn] == true_class:
                per_stream_correct[sn] += 1

        status = "OK" if is_correct else "WRONG"
        stream_preds_str = ", ".join(
            f"{sn}={result['per_stream_preds'][sn]}"
            for sn in STREAM_NAMES
        )
        agree_count = result["agreement_count"]
        agree_str = "ALL AGREE" if result["all_streams_agree"] else f"{agree_count}/4 agree"
        print(f"    Fused Pred: {pred_class} (conf={confidence:.3f}, {conf_lvl}) | "
              f"True: {true_class} [{status}]")
        print(f"    Streams: {stream_preds_str} [{agree_str}]")

        top3_str = ", ".join([f"{cls}={prob:.3f}" for cls, prob in result["top3"]])
        print(f"    Top-3: {top3_str}")

        if result["uncal_pred_class"] != result["pred_class"]:
            print(f"    Calibration changed: {result['uncal_pred_class']} -> {result['pred_class']}")

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred": pred_class,
            "correct": is_correct,
            "confidence": confidence,
            "confidence_level": conf_lvl,
            "uncal_pred": result["uncal_pred_class"],
            "uncal_confidence": result["uncal_confidence"],
            "per_stream_preds": result["per_stream_preds"],
            "all_streams_agree": result["all_streams_agree"],
            "agreement_count": result["agreement_count"],
            "majority_pred": result["majority_pred"],
            "top3": result["top3"],
        })

    # Summary
    acc = 100.0 * correct / total if total > 0 else 0.0

    print(f"\n  {category_name.upper()} OVERALL: {correct}/{total} = {acc:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'Acc':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            cls_acc = 100.0 * pc["correct"] / pc["total"]
            print(f"    {cls:>12s}  {cls_acc:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    print(f"    {'Signer':>20s}  {'Acc':>7s}  {'N':>3s}")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        signer_acc = 100.0 * ps["correct"] / ps["total"]
        print(f"    {signer:>20s}  {signer_acc:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence-based analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            bucket_acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {bucket_acc:.0f}%")
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

    # Per-stream individual accuracy
    per_stream_acc = {}
    print(f"\n  Per-stream individual accuracy (which stream is best alone?):")
    print(f"    {'Stream':>15s}  {'Acc':>7s}  {'Correct':>7s}  {'Total':>5s}")
    for sn in STREAM_NAMES:
        if per_stream_total[sn] > 0:
            s_acc = 100.0 * per_stream_correct[sn] / per_stream_total[sn]
        else:
            s_acc = 0.0
        per_stream_acc[sn] = s_acc
        print(f"    {sn:>15s}  {s_acc:5.1f}%   {per_stream_correct[sn]:5d}   {per_stream_total[sn]:5d}")

    # Stream agreement analysis
    print(f"\n  Stream agreement analysis:")
    if total > 0:
        print(f"    All 4 streams agree: {total_all_agree}/{total} "
              f"({100.0 * total_all_agree / total:.0f}%)")
        print(f"    Majority stream correct: {total_majority_correct}/{total} "
              f"({100.0 * total_majority_correct / total:.0f}%)")
    else:
        print(f"    N/A (no predictions)")

    print(f"\n  Confusion Matrix ({category_name}):")
    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"    {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{confusion[i][j]:5d}" for j in range(nc))
        print(f"    {row_str}")

    return {
        "overall": acc,
        "correct": correct,
        "total": total,
        "per_stream_accuracy": per_stream_acc,
        "per_class": {k: {
            "acc": 100.0 * v["correct"] / v["total"],
            **v
        } for k, v in per_class.items()},
        "per_signer": {k: {
            "acc": 100.0 * v["correct"] / v["total"],
            **v
        } for k, v in per_signer.items()},
        "confidence_buckets": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            **v
        } for k, v in conf_buckets.items()},
        "stream_agreement": {
            "all_agree_count": total_all_agree,
            "all_agree_pct": 100.0 * total_all_agree / total if total > 0 else 0.0,
            "majority_correct_count": total_majority_correct,
            "majority_correct_pct": 100.0 * total_majority_correct / total if total > 0 else 0.0,
        },
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
    print("V23 Real-World Evaluation (Multi-Stream Fusion, NO TTA, Calibrated)")
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

    results = {}

    # ===== NUMBERS =====
    numbers_ckpt_dir = os.path.join(ckpt_dir, "v23_numbers")
    if os.path.isdir(numbers_ckpt_dir) and numbers_videos:
        print(f"\n{'='*70}")
        print(f"NUMBERS EVALUATION ({len(numbers_videos)} videos)")
        print(f"{'='*70}")

        models, fusion_weights, calibration = load_stream_models(
            numbers_ckpt_dir, NUMBER_CLASSES, device
        )

        if models is not None:
            results["numbers"] = evaluate_category(
                "Numbers", numbers_videos, models, fusion_weights, calibration,
                device, NUMBER_CLASSES
            )
            # Free GPU memory
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        if not os.path.isdir(numbers_ckpt_dir):
            print(f"\nSkipping numbers: checkpoint dir not found at {numbers_ckpt_dir}")
        elif not numbers_videos:
            print(f"\nSkipping numbers: no test videos found")

    # ===== WORDS =====
    words_ckpt_dir = os.path.join(ckpt_dir, "v23_words")
    if os.path.isdir(words_ckpt_dir) and words_videos:
        print(f"\n{'='*70}")
        print(f"WORDS EVALUATION ({len(words_videos)} videos)")
        print(f"{'='*70}")

        models, fusion_weights, calibration = load_stream_models(
            words_ckpt_dir, WORD_CLASSES, device
        )

        if models is not None:
            results["words"] = evaluate_category(
                "Words", words_videos, models, fusion_weights, calibration,
                device, WORD_CLASSES
            )
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        if not os.path.isdir(words_ckpt_dir):
            print(f"\nSkipping words: checkpoint dir not found at {words_ckpt_dir}")
        elif not words_videos:
            print(f"\nSkipping words: no test videos found")

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V23 Real-World Evaluation (Multi-Stream, NO TTA)")
    print(f"{'='*70}")

    if "numbers" in results:
        r = results["numbers"]
        print(f"  Numbers: {r['overall']:.1f}% ({r['correct']}/{r['total']})")
        if "per_stream_accuracy" in r:
            for sn in STREAM_NAMES:
                print(f"    {sn:>15s} alone: {r['per_stream_accuracy'][sn]:.1f}%")

    if "words" in results:
        r = results["words"]
        print(f"  Words:   {r['overall']:.1f}% ({r['correct']}/{r['total']})")
        if "per_stream_accuracy" in r:
            for sn in STREAM_NAMES:
                print(f"    {sn:>15s} alone: {r['per_stream_accuracy'][sn]:.1f}%")

    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
        print(f"\n  Combined: {combined:.1f}%")

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
    if results:
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

    # Stream agreement summary
    print(f"\n  Stream Agreement Summary:")
    for cat in ["numbers", "words"]:
        if cat in results and "stream_agreement" in results[cat]:
            sa = results[cat]["stream_agreement"]
            print(f"    {cat.capitalize()}: all-agree={sa['all_agree_pct']:.0f}%, "
                  f"majority-correct={sa['majority_correct_pct']:.0f}%")

    # Previous version comparison
    print(f"\n  Previous versions (for comparison):")
    print(f"    V19: Numbers 20.3% | Words 48.1% | Combined 34.2%")
    print(f"    V20: Numbers 20.3% | Words 40.7% | Combined 30.5%")
    print(f"    V21: Numbers 25.4% | Words 37.0% | Combined 31.2%")
    print(f"    V22: Numbers 28.8% | Words 46.9% | Combined 37.9% (no-TTA: 33.9%/45.7%/39.8%)")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v23_real_testers_{ts_str}.json")

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
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, tuple):
            return [make_serializable(v) for v in obj]
        return obj

    results_out = {
        "version": "v23",
        "evaluation_type": "real_testers",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tta_enabled": False,
        "normalization": "wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)",
        "features": "4 streams: joint(3ch), bone(3ch), velocity(3ch), bone_velocity(3ch) | AUX: angles+fingertip_dists",
        "model": "KSLGraphNetV23 x4 (multi-stream ST-GCN ensemble + learned fusion + calibration)",
        "fusion": "weighted softmax average with grid-searched weights",
        "calibration": "per-class temperature + bias via L-BFGS-B",
        "results": make_serializable(results),
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
