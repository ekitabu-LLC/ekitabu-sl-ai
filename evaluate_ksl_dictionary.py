#!/usr/bin/env python3
"""
evaluate_ksl_dictionary.py

Evaluate ALL KSL models (v22-v43) on KSL Dictionary reference videos.

 - CPU only (no GPU) — set CUDA_VISIBLE_DEVICES="" or device=cpu
 - Caches MediaPipe landmarks as .npy for fast reuse
 - Words category only (all 14 downloaded videos are word signs)
 - Global AdaBN applied for BatchNorm models (adapts BN stats to the 14 ref videos)

Usage:
    python evaluate_ksl_dictionary.py
    python evaluate_ksl_dictionary.py --videos-dir data/ksl_dictionary_videos
    python evaluate_ksl_dictionary.py --no-adabn
    python evaluate_ksl_dictionary.py --models v27 v31_exp5 v43
"""

import os
import sys
import json
import copy
import argparse
import importlib
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v28.py / evaluate_real_testers_v30.py)
# ---------------------------------------------------------------------------

POSE_INDICES = [11, 12, 13, 14, 15, 16]

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

# Mapping from downloaded filename (stem) to dataset class name
VIDEO_TO_CLASS = {
    "Apple":     "Apple",
    "Agreement": "Agreement",
    "Color":     "Colour",    # KSL website uses "Color", dataset uses "Colour"
    "Friend":    "Friend",
    "Market":    "Market",
    "Monday":    "Monday",
    "Picture":   "Picture",
    "Proud":     "Proud",
    "Sweater":   "Sweater",
    "Teach":     "Teach",
    "Tomato":    "Tomatoes",  # website "Tomato" → dataset "Tomatoes"
    "Tortoise":  "Tortoise",
    "Twins":     "Twin",      # website "Twins" → dataset "Twin"
    "Ugali":     "Ugali",
}

LH_PARENT  = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT  = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP  = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

LH_TIPS = [4, 8, 12, 16, 20]
RH_TIPS  = [25, 29, 33, 37, 41]
NUM_FINGERTIP_PAIRS = 10  # C(5,2) per hand
NUM_HAND_BODY_FEATURES = 8
NUM_NODES = 48


def _build_angle_joints():
    children = defaultdict(list)
    for child_idx, parent_idx in enumerate(PARENT_MAP):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    angle_joints = []
    for node in range(NUM_NODES):
        parent = PARENT_MAP[node]
        if parent < 0:
            continue
        for child in children[node]:
            angle_joints.append((node, parent, child))
    return angle_joints


ANGLE_JOINTS = _build_angle_joints()
NUM_ANGLE_FEATURES = len(ANGLE_JOINTS)          # 33
AUX_DIM_NO_HANDBODY = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS           # 53
AUX_DIM_WITH_HANDBODY = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES  # 61

# ---------------------------------------------------------------------------
# Model Registry
# Each entry describes one model version.
# ckpt_words: path relative to data/checkpoints/ for the words checkpoint.
#   For single-model versions: path to best_model.pt
#   For multi-stream: path to the directory containing {joint,bone,velocity}/
# input_type:
#   "single_9ch"   → single best_model.pt, model takes (9,T,48) input
#   "3stream"      → multi-stream {joint,bone,velocity}, each (3,T,48)
#   "4stream"      → v23 multi-stream {joint,bone,velocity,bone_velocity}
# norm: "BN" or "GN" (BatchNorm vs GroupNorm)
# aux_type: "no_hand_body" (53-dim) or "with_hand_body" (61-dim)
# train_module: name of the Python module to import model class from
# model_class: name of the class in that module
# build_adj_fn: name of the build_adj function (usually "build_adj")
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    # ─── Era 1: Single-stream 9ch, no hand_body ───────────────────────────
    dict(name="v22", ckpt_words="v22_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="no_hand_body",
         train_module="train_ksl_v22", model_class="KSLGraphNetV22",
         build_adj_fn="build_adj"),

    # ─── Era 2: Multi-stream 4-stream, no hand_body ───────────────────────
    dict(name="v23", ckpt_words="v23_words",
         input_type="4stream", norm="BN", aux_type="no_hand_body",
         train_module="train_ksl_v23", model_class="KSLGraphNetV23",
         build_adj_fn="build_adj"),

    # ─── Era 3: Single-stream 9ch, with hand_body ─────────────────────────
    dict(name="v24", ckpt_words="v24_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v24", model_class="KSLGraphNetV24",
         build_adj_fn="build_adj"),

    dict(name="v25", ckpt_words="v25_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v25", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v26", ckpt_words="v26_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v25", model_class="KSLGraphNetV25",  # same arch
         build_adj_fn="build_adj"),

    dict(name="v27", ckpt_words="v27_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v27", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    # ─── Era 4: Multi-stream 3-stream v28 / v29 single ────────────────────
    dict(name="v28", ckpt_words="v28_words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v28", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v29", ckpt_words="v29_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v29", model_class="KSLGraphNetV29",
         build_adj_fn="build_adj"),

    dict(name="v30", ckpt_words="v30_words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v30", model_class="KSLGraphNetV30",
         build_adj_fn="build_adj"),

    # ─── Era 5: v31 experiments ───────────────────────────────────────────
    dict(name="v31_exp1", ckpt_words="v31_exp1/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    # v31_exp2 uses ProtoNetEncoder (no classifier head) — skip
    # dict(name="v31_exp2", ...),

    dict(name="v31_exp3", ckpt_words="v31_exp3/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp3", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v31_exp4", ckpt_words="v31_exp4/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp4", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v31_exp5", ckpt_words="v31_exp5/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp5", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    # ─── Era 6: v32-v43 ───────────────────────────────────────────────────
    dict(name="v32", ckpt_words="v32/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v32", model_class="KSLGraphNetV32",
         build_adj_fn="build_adj"),

    dict(name="v32b", ckpt_words="v32b/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v32", model_class="KSLGraphNetV32",   # same arch
         build_adj_fn="build_adj"),

    dict(name="v33", ckpt_words="v33/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v33", model_class="KSLGraphNetV33",
         build_adj_fn="build_adj"),

    dict(name="v33b", ckpt_words="v33b/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v33", model_class="KSLGraphNetV33",   # same arch
         build_adj_fn="build_adj"),

    dict(name="v34a_seed123", ckpt_words="v34a_seed123/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v34b_seed456", ckpt_words="v34b_seed456/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v35", ckpt_words="v35/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v35", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v36", ckpt_words="v36/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v36", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v37", ckpt_words="v37/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v37", model_class="KSLGraphNetV37",
         build_adj_fn="build_adj"),

    dict(name="v38", ckpt_words="v38/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v38", model_class="KSLGraphNetV38",
         build_adj_fn="build_graph_distance"),  # v38 uses build_graph_distance

    dict(name="v39", ckpt_words="v39/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v39", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v40", ckpt_words="v40/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v40", model_class="KSLGraphNetV40",
         build_adj_fn="build_adj"),

    dict(name="v41", ckpt_words="v41/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v41", model_class="KSLGraphNetV41",
         build_adj_fn="build_adj"),

    dict(name="v42", ckpt_words="v42/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v42", model_class="KSLGraphNetV42",
         build_adj_fn="build_adj"),

    dict(name="v43", ckpt_words="v43/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v43", model_class="KSLGraphNetV43",
         build_adj_fn="build_adj"),
]


# ---------------------------------------------------------------------------
# Landmark extraction (CPU, MediaPipe)
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
    """Extract mediapipe holistic landmarks from a video file.
    Returns (N_frames, 225) float32 array or None on failure.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] Cannot open: {video_path}")
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

        frame_lm = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_lm.extend([lm.x, lm.y, lm.z])
        else:
            frame_lm.extend([0.0] * 99)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_lm.extend([lm.x, lm.y, lm.z])
        else:
            frame_lm.extend([0.0] * 63)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_lm.extend([lm.x, lm.y, lm.z])
        else:
            frame_lm.extend([0.0] * 63)

        all_landmarks.append(frame_lm)

    cap.release()
    holistic.close()

    if not all_landmarks:
        print(f"    [WARN] No frames extracted from: {video_path}")
        return None

    return np.array(all_landmarks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def compute_bones(h):
    bones = np.zeros_like(h)
    for child in range(NUM_NODES):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def compute_joint_angles(h):
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)
    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        bone_in  = h[:, node, :] - h[:, parent, :]
        bone_out = h[:, child, :] - h[:, node, :]
        n_in  = np.maximum(np.linalg.norm(bone_in,  axis=-1, keepdims=True), 1e-8)
        n_out = np.maximum(np.linalg.norm(bone_out, axis=-1, keepdims=True), 1e-8)
        dot   = np.sum((bone_in / n_in) * (bone_out / n_out), axis=-1)
        angles[:, idx] = np.arccos(np.clip(dot, -1.0, 1.0))
    return angles


def compute_fingertip_distances(h):
    T   = h.shape[0]
    col = 0
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
    for tips in [LH_TIPS, RH_TIPS]:
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                diff = h[:, tips[i], :] - h[:, tips[j], :]
                distances[:, col] = np.linalg.norm(diff, axis=-1)
                col += 1
    return distances


def compute_hand_body_features(h_raw):
    T = h_raw.shape[0]
    feats = np.zeros((T, NUM_HAND_BODY_FEATURES), dtype=np.float32)
    mid_shoulder   = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2
    shoulder_width = np.maximum(
        np.linalg.norm(h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True), 1e-6
    )
    lh_centroid = h_raw[:, :21, :].mean(axis=1)
    rh_centroid = h_raw[:, 21:42, :].mean(axis=1)
    face_approx = mid_shoulder.copy()
    face_approx[:, 1] -= shoulder_width[:, 0] * 0.7
    sw = shoulder_width[:, 0]
    feats[:, 0] = (lh_centroid[:, 1] - mid_shoulder[:, 1]) / sw
    feats[:, 1] = (rh_centroid[:, 1] - mid_shoulder[:, 1]) / sw
    feats[:, 2] = (lh_centroid[:, 0] - mid_shoulder[:, 0]) / sw
    feats[:, 3] = (rh_centroid[:, 0] - mid_shoulder[:, 0]) / sw
    feats[:, 4] = np.linalg.norm(lh_centroid - rh_centroid, axis=-1) / sw
    feats[:, 5] = np.linalg.norm(lh_centroid - face_approx, axis=-1) / sw
    feats[:, 6] = np.linalg.norm(rh_centroid - face_approx, axis=-1) / sw
    feats[:, 7] = np.abs(lh_centroid[:, 1] - rh_centroid[:, 1]) / sw
    return feats


def normalize_wrist_palm(h):
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = lh[:, 0:1, :]
        lh[lh_valid] -= wrist[lh_valid]
        psz = np.maximum(np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        lh[lh_valid] /= psz[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = rh[:, 0:1, :]
        rh[rh_valid] -= wrist[rh_valid]
        psz = np.maximum(np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        rh[rh_valid] /= psz[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] -= mid[pose_valid]
        sw = np.maximum(
            np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True), 1e-6
        )
        pose[pose_valid] /= sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


def preprocess_raw(raw, max_frames=MAX_FRAMES, include_hand_body=True):
    """Preprocess raw (N_frames, 225) landmarks.

    Returns:
        streams: {
            "joint":    (3, max_frames, 48) tensor,
            "bone":     (3, max_frames, 48) tensor,
            "velocity": (3, max_frames, 48) tensor,
            "bone_velocity": (3, max_frames, 48) tensor,  # only when include_hand_body=False (v23 era)
        }
        aux: (max_frames, aux_dim) tensor  — aux_dim = 53 or 61
    """
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None, None

    # ── extract skeleton ──────────────────────────────────────────────────
    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)
    h  = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # ── hand-body features BEFORE normalization ───────────────────────────
    hb_feats = compute_hand_body_features(h) if include_hand_body else None

    # ── normalization ─────────────────────────────────────────────────────
    h = normalize_wrist_palm(h)

    # ── stream features ───────────────────────────────────────────────────
    velocity      = np.zeros_like(h)
    velocity[1:]  = h[1:] - h[:-1]
    bones         = compute_bones(h)
    bone_velocity = np.zeros_like(bones)
    bone_velocity[1:] = bones[1:] - bones[:-1]

    # ── aux features ─────────────────────────────────────────────────────
    joint_angles   = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    # ── temporal resampling / padding ─────────────────────────────────────
    def _resamp(arr, nf):
        if f >= max_frames:
            idx = np.linspace(0, f - 1, max_frames, dtype=int)
            return arr[idx]
        pad = max_frames - f
        pad_shape = (pad,) + arr.shape[1:]
        return np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)])

    h             = _resamp(h,             f)
    velocity      = _resamp(velocity,      f)
    bones         = _resamp(bones,         f)
    bone_velocity = _resamp(bone_velocity, f)
    joint_angles  = _resamp(joint_angles,  f)
    fingertip_dists = _resamp(fingertip_dists, f)
    if hb_feats is not None:
        hb_feats = _resamp(hb_feats, f)

    # ── build stream tensors (3, T, N) ────────────────────────────────────
    def _t(x):
        return torch.FloatTensor(np.clip(x, -10, 10)).permute(2, 0, 1)

    streams = {
        "joint":         _t(h),
        "bone":          _t(bones),
        "velocity":      _t(velocity),
        "bone_velocity": _t(bone_velocity),
    }

    # ── aux tensor ────────────────────────────────────────────────────────
    if include_hand_body:
        aux_np = np.concatenate([joint_angles, fingertip_dists, hb_feats], axis=1)
    else:
        aux_np = np.concatenate([joint_angles, fingertip_dists], axis=1)
    aux = torch.FloatTensor(np.clip(aux_np, -10, 10))

    return streams, aux


def stack_9ch(streams):
    """Stack joint + velocity + bone into (9, T, N) for single-model versions."""
    return torch.cat([streams["joint"], streams["velocity"], streams["bone"]], dim=0)


# ---------------------------------------------------------------------------
# AdaBN
# ---------------------------------------------------------------------------

def adapt_bn_stats(model, data_list, device):
    """Reset BN running stats and forward-pass all data to collect new stats."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None  # cumulative moving average — equal weight to all samples

    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            try:
                model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
            except Exception:
                pass
    model.eval()


# ---------------------------------------------------------------------------
# Load fusion weights
# ---------------------------------------------------------------------------

def _load_fusion_weights(ckpt_dir, streams=("joint", "bone", "velocity")):
    """Load fusion_weights.json or return equal weights."""
    w_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(w_path):
        with open(w_path) as fh:
            data = json.load(fh)
        w = data.get("weights", data)
        # Handle nested {"weights": {...}} format
        if isinstance(w, dict) and "weights" in w:
            w = w["weights"]
        return {s: w.get(s, 1.0 / len(streams)) for s in streams}
    return {s: 1.0 / len(streams) for s in streams}


# ---------------------------------------------------------------------------
# Model instantiation helpers
# ---------------------------------------------------------------------------

def _lazy_import(module_name, class_name, build_adj_fn="build_adj"):
    """Import a model class lazily from a training script."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    build_adj = getattr(mod, build_adj_fn)
    return cls, build_adj


def _build_model_from_ckpt(ckpt, cls, build_adj, nc, device):
    """Instantiate a model from checkpoint config using the provided class."""
    config      = ckpt.get("config", {})
    num_signers = ckpt.get("num_signers", 12)
    num_nodes   = config.get("num_nodes", NUM_NODES)
    adj         = build_adj(num_nodes).to(device)
    aux_dim     = ckpt.get("aux_dim", AUX_DIM_WITH_HANDBODY)
    hd          = config.get("hidden_dim", 64)
    nl          = config.get("num_layers", 4)
    ic          = config.get("in_channels", 3)
    tk          = tuple(config.get("temporal_kernels", [3, 5, 7]))
    dr          = config.get("dropout", 0.3)
    sd          = config.get("spatial_dropout", 0.1)

    # Different model classes have slightly different __init__ signatures.
    # Try a common set of kwargs, falling back to simpler ones.
    kwargs_full = dict(
        nc=nc, num_signers=num_signers, aux_dim=aux_dim,
        nn_=num_nodes, ic=ic, hd=hd, nl=nl, tk=tk, dr=dr,
        spatial_dropout=sd, adj=adj,
    )

    # Some classes also take proj_dim (SupCon projection)
    proj_dim = config.get("supcon_proj_dim", 128)
    kwargs_with_proj = dict(**kwargs_full, proj_dim=proj_dim)

    for kwargs in [kwargs_with_proj, kwargs_full]:
        try:
            return cls(**kwargs)
        except TypeError:
            pass

    # Last resort: strip extra kwargs until it works
    essential = dict(nc=nc, aux_dim=aux_dim, nn_=num_nodes, ic=ic, adj=adj)
    for k in ["hd", "nl", "tk", "dr", "spatial_dropout", "num_signers"]:
        try:
            return cls(**essential)
        except TypeError:
            essential[k] = kwargs_full[k]

    raise RuntimeError(f"Cannot instantiate {cls.__name__} with checkpoint config")


# ---------------------------------------------------------------------------
# Run inference on one video with one model
# ---------------------------------------------------------------------------

def run_inference_single(model, gcn_9ch, aux, device):
    """Single-stream model inference → (pred_idx, probs)."""
    with torch.no_grad():
        out = model(gcn_9ch.unsqueeze(0).to(device),
                    aux.unsqueeze(0).to(device),
                    grl_lambda=0.0)
        logits = out[0]
        probs  = F.softmax(logits, dim=1).cpu().squeeze(0)
    return probs.argmax().item(), probs


def run_inference_multistream(models_dict, fusion_weights, streams, aux, device,
                              stream_names=("joint", "bone", "velocity")):
    """Multi-stream fusion inference → (pred_idx, fused_probs)."""
    per_stream = {}
    with torch.no_grad():
        for sname in stream_names:
            if sname not in models_dict:
                continue
            gcn = streams[sname].unsqueeze(0).to(device)
            out = models_dict[sname](gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
            logits = out[0]
            per_stream[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)

    fused = sum(fusion_weights.get(s, 1.0 / len(per_stream)) * per_stream[s]
                for s in per_stream)
    pred_idx = fused.argmax().item()
    return pred_idx, fused


# ---------------------------------------------------------------------------
# Evaluate one model entry
# ---------------------------------------------------------------------------

def evaluate_model_entry(entry, video_data, classes, device, use_adabn, ckpt_base):
    """Load and evaluate one model version on all videos.

    Args:
        entry: MODEL_REGISTRY dict entry
        video_data: list of {name, true_class, raw (np.ndarray)} dicts
        classes: list of class strings
        device: torch device
        use_adabn: bool
        ckpt_base: path to data/checkpoints/

    Returns:
        dict {video_name: (pred_class, confidence, correct)}
        or None on failure
    """
    name        = entry["name"]
    input_type  = entry["input_type"]
    norm        = entry["norm"]
    aux_type    = entry["aux_type"]
    include_hb  = (aux_type == "with_hand_body")

    ckpt_words_rel = entry["ckpt_words"]
    ckpt_dir       = os.path.join(ckpt_base, ckpt_words_rel)

    # ── import model class ────────────────────────────────────────────────
    try:
        cls, build_adj = _lazy_import(
            entry["train_module"], entry["model_class"], entry["build_adj_fn"]
        )
    except Exception as e:
        print(f"  [{name}] Import error: {e}")
        return None

    # ── preprocess all videos ─────────────────────────────────────────────
    preprocessed = {}   # {video_name: (streams_dict, aux_tensor)} or None
    for vd in video_data:
        streams, aux = preprocess_raw(vd["raw"], include_hand_body=include_hb)
        if streams is None:
            preprocessed[vd["name"]] = None
        else:
            preprocessed[vd["name"]] = (streams, aux)

    # ── load models ───────────────────────────────────────────────────────
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = classes

    if input_type == "single_9ch":
        # Single best_model.pt
        ckpt_path = ckpt_dir  # ckpt_dir already ends with best_model.pt for these
        if not os.path.exists(ckpt_path):
            print(f"  [{name}] Checkpoint not found: {ckpt_path}")
            return None
        try:
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            nc    = len(classes)
            model = _build_model_from_ckpt(ckpt, cls, build_adj, nc, device)
            model.load_state_dict(ckpt["model"])
            model.eval()
        except Exception as e:
            print(f"  [{name}] Load error: {e}")
            return None

        val_acc = ckpt.get("val_acc", 0.0)
        print(f"  [{name}] Loaded single model (val_acc={val_acc:.1f}%, norm={norm})")

        # AdaBN
        if use_adabn and norm == "BN":
            adabn_data = []
            for vd in video_data:
                if preprocessed[vd["name"]] is not None:
                    s, a = preprocessed[vd["name"]]
                    adabn_data.append((stack_9ch(s), a))
            if adabn_data:
                adapt_bn_stats(model, adabn_data, device)
                print(f"  [{name}] AdaBN applied ({len(adabn_data)} samples)")

        # Inference
        results = {}
        for vd in video_data:
            if preprocessed[vd["name"]] is None:
                results[vd["name"]] = None
                continue
            s, a = preprocessed[vd["name"]]
            gcn_9ch = stack_9ch(s)
            pred_idx, probs = run_inference_single(model, gcn_9ch, a, device)
            pred_class  = i2c[pred_idx]
            confidence  = probs[pred_idx].item()
            is_correct  = (pred_class == vd["true_class"])
            results[vd["name"]] = {
                "pred": pred_class,
                "conf": confidence,
                "correct": is_correct,
                "top3": [(i2c[i], probs[i].item()) for i in probs.argsort(descending=True)[:3]],
            }

    elif input_type in ("3stream", "4stream"):
        stream_names = (["joint", "bone", "velocity", "bone_velocity"]
                        if input_type == "4stream" else
                        ["joint", "bone", "velocity"])

        # Load each stream model
        models_dict  = {}
        fusion_weights = {}
        for sname in stream_names:
            ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [{name}] Missing stream {sname}: {ckpt_path}")
                continue
            try:
                ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
                nc    = len(classes)
                model = _build_model_from_ckpt(ckpt, cls, build_adj, nc, device)
                model.load_state_dict(ckpt["model"])
                model.eval()
                models_dict[sname] = model
            except Exception as e:
                print(f"  [{name}] Load error ({sname}): {e}")

        if not models_dict:
            print(f"  [{name}] No streams loaded, skipping")
            return None

        # Load fusion weights
        fusion_weights = _load_fusion_weights(ckpt_dir, stream_names)
        val_acc = ckpt.get("val_acc", 0.0)
        print(f"  [{name}] Loaded {len(models_dict)}/{len(stream_names)} streams "
              f"(val_acc={val_acc:.1f}%, norm={norm}, fw={fusion_weights})")

        # AdaBN
        if use_adabn and norm == "BN":
            for sname, model in models_dict.items():
                adabn_data = []
                for vd in video_data:
                    if preprocessed[vd["name"]] is not None:
                        s, a = preprocessed[vd["name"]]
                        adabn_data.append((s[sname], a))
                if adabn_data:
                    adapt_bn_stats(model, adabn_data, device)
            print(f"  [{name}] AdaBN applied to all {len(models_dict)} stream models")

        # Inference
        results = {}
        for vd in video_data:
            if preprocessed[vd["name"]] is None:
                results[vd["name"]] = None
                continue
            s, a = preprocessed[vd["name"]]
            pred_idx, probs = run_inference_multistream(
                models_dict, fusion_weights, s, a, device, stream_names
            )
            pred_class  = i2c[pred_idx]
            confidence  = probs[pred_idx].item()
            is_correct  = (pred_class == vd["true_class"])
            results[vd["name"]] = {
                "pred": pred_class,
                "conf": confidence,
                "correct": is_correct,
                "top3": [(i2c[i], probs[i].item()) for i in probs.argsort(descending=True)[:3]],
            }

    else:
        print(f"  [{name}] Unknown input_type: {input_type}")
        return None

    return results


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results, video_data):
    """Print a compact table: rows = models, columns = videos."""
    video_names = [vd["name"] for vd in video_data]
    true_classes = {vd["name"]: vd["true_class"] for vd in video_data}

    # Header
    col_w = 14
    name_w = 18
    print("\n" + "=" * (name_w + col_w * len(video_names) + 8))
    print("RESULTS — KSL Dictionary Reference Videos (Words)")
    print("=" * (name_w + col_w * len(video_names) + 8))

    hdr = f"{'Model':{name_w}s}"
    for vn in video_names:
        hdr += f"  {vn[:col_w-2]:>{col_w-2}s}"
    hdr += "  Acc"
    print(hdr)

    # Ground truth row
    gt_row = f"{'[TRUE CLASS]':{name_w}s}"
    for vn in video_names:
        gt_row += f"  {true_classes[vn][:col_w-2]:>{col_w-2}s}"
    print(gt_row)
    print("-" * (name_w + col_w * len(video_names) + 8))

    # Model rows
    for model_name, results in all_results.items():
        if results is None:
            print(f"{model_name:{name_w}s}  [FAILED]")
            continue
        correct = 0
        total   = 0
        row     = f"{model_name:{name_w}s}"
        for vn in video_names:
            r = results.get(vn)
            if r is None:
                cell = "ERR"
            else:
                pred = r["pred"]
                mark = "✓" if r["correct"] else "✗"
                cell = f"{pred[:col_w-4]:>{col_w-4}s}{mark}"
                correct += int(r["correct"])
                total   += 1
            row += f"  {cell:>{col_w-2}s}"
        acc = f"{100.0 * correct / total:.0f}%" if total > 0 else "N/A"
        row += f"  {correct}/{total} ({acc})"
        print(row)

    print("=" * (name_w + col_w * len(video_names) + 8))

    # Per-video accuracy across all models
    print("\nPer-video accuracy (fraction of models correct):")
    for vn in video_names:
        n_correct = sum(
            1 for r in all_results.values()
            if r is not None and r.get(vn) is not None and r[vn]["correct"]
        )
        n_total = sum(
            1 for r in all_results.values()
            if r is not None and r.get(vn) is not None
        )
        pct = 100 * n_correct / n_total if n_total > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"  {vn:12s} ({true_classes[vn]:10s}): {n_correct:2d}/{n_total:2d} = {pct:4.0f}%  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate KSL models on dictionary videos")
    parser.add_argument("--videos-dir", default=os.path.join(PROJECT_ROOT, "data/ksl_dictionary_videos"),
                        help="Directory containing .mp4 videos")
    parser.add_argument("--ckpt-base", default=os.path.join(PROJECT_ROOT, "data/checkpoints"),
                        help="Base checkpoint directory")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "data/results/ksl_dictionary"),
                        help="Output directory for JSON results")
    parser.add_argument("--no-adabn", action="store_true",
                        help="Disable AdaBN for BatchNorm models")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to evaluate (default: all). E.g.: --models v27 v41 v43")
    args = parser.parse_args()

    use_adabn = not args.no_adabn
    device    = torch.device("cpu")

    print("=" * 70)
    print("KSL Dictionary Reference Video Evaluation")
    print(f"Videos dir:  {args.videos_dir}")
    print(f"Ckpt base:   {args.ckpt_base}")
    print(f"Device:      {device}")
    print(f"AdaBN:       {'enabled (BN models only)' if use_adabn else 'DISABLED'}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── mediapipe version check ───────────────────────────────────────────
    try:
        import mediapipe as mp
        print(f"\nMediaPipe version: {mp.__version__} (expected 0.10.5 for v28+)")
    except ImportError:
        print("\n[ERROR] mediapipe not found — cannot extract landmarks")
        sys.exit(1)

    # ── discover videos ───────────────────────────────────────────────────
    video_dir = args.videos_dir
    if not os.path.isdir(video_dir):
        print(f"[ERROR] Videos directory not found: {video_dir}")
        sys.exit(1)

    video_data = []
    for fname in sorted(os.listdir(video_dir)):
        stem = os.path.splitext(fname)[0]
        if not fname.lower().endswith(".mp4"):
            continue
        if stem not in VIDEO_TO_CLASS:
            print(f"  [SKIP] Unknown video: {fname}")
            continue
        true_class = VIDEO_TO_CLASS[stem]
        video_path = os.path.join(video_dir, fname)
        video_data.append({"name": stem, "true_class": true_class, "path": video_path})

    print(f"\nFound {len(video_data)} videos:")
    for vd in video_data:
        print(f"  {vd['name']:12s} → class '{vd['true_class']}'")

    # ── extract/load landmarks ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Extracting landmarks (caching as .npy)...")
    for vd in video_data:
        npy_path = vd["path"] + ".landmarks.npy"
        if os.path.exists(npy_path):
            vd["raw"] = np.load(npy_path)
            print(f"  {vd['name']:12s}: loaded from cache ({vd['raw'].shape[0]} frames)")
        else:
            print(f"  {vd['name']:12s}: extracting...", end="", flush=True)
            raw = extract_landmarks_from_video(vd["path"])
            if raw is None:
                print(" FAILED")
                vd["raw"] = None
            else:
                np.save(npy_path, raw)
                vd["raw"] = raw
                print(f" done ({raw.shape[0]} frames) → saved {os.path.basename(npy_path)}")

    video_data = [vd for vd in video_data if vd.get("raw") is not None]
    print(f"\n{len(video_data)} videos ready for evaluation")

    # ── filter registry ───────────────────────────────────────────────────
    if args.models:
        registry = [e for e in MODEL_REGISTRY if e["name"] in args.models]
        print(f"Evaluating {len(registry)} model(s): {[e['name'] for e in registry]}")
    else:
        registry = MODEL_REGISTRY
        print(f"Evaluating all {len(registry)} model versions")

    classes = WORD_CLASSES

    # ── evaluate each model ───────────────────────────────────────────────
    all_results = {}
    for entry in registry:
        print(f"\n{'─'*50}")
        print(f"[{entry['name']}]  input={entry['input_type']}  norm={entry['norm']}  "
              f"aux={entry['aux_type']}")
        try:
            results = evaluate_model_entry(
                entry, video_data, classes, device, use_adabn, args.ckpt_base
            )
        except Exception as e:
            print(f"  [{entry['name']}] EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results = None
        all_results[entry["name"]] = results

        if results is not None:
            correct = sum(1 for r in results.values() if r and r["correct"])
            total   = sum(1 for r in results.values() if r is not None)
            print(f"  → {correct}/{total} correct ({100.0*correct/total:.0f}%)" if total else "  → No results")

    # ── summary table ─────────────────────────────────────────────────────
    print_summary_table(all_results, video_data)

    # ── save JSON ─────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"ksl_dict_eval_{ts}.json")

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_videos": len(video_data),
        "videos": [{"name": vd["name"], "true_class": vd["true_class"]} for vd in video_data],
        "use_adabn": use_adabn,
        "results": make_serializable(all_results),
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
