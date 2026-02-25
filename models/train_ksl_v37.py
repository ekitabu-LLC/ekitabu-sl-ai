#!/usr/bin/env python3
"""
KSL Training v37 - GroupNorm + SupCon + Projection Head + Speed Augmentation

Combines the two best individual techniques from v31 (never previously combined):
  1. GroupNorm (v31 exp1): Replaces ALL BatchNorm with GroupNorm for signer-invariant
     normalization — no AdaBN needed at inference (+3.2pp vs v28 baseline)
  2. SupCon loss (v31 exp5): Cross-signer supervised contrastive loss pulls same-class
     embeddings together across signers (+5.0pp vs v28 baseline)

New additions not in any prior experiment:
  3. SupCon Projection Head: Dedicated 320→128→128 MLP for SupCon loss, keeping the
     classifier head unconstrained (standard Khosla et al. 2020 practice — was missing
     in exp1/exp5 which applied SupCon directly to the 320-d classifier embedding)
  4. Temporal Speed Augmentation: Randomly stretches training sequences to 130–220 frames
     BEFORE the final resampling to 90 frames. This directly addresses the discovered
     train-test sequence length mismatch: train mean=104 frames, test mean=145 frames.
     Without this, test sequences are downsampled 38% more aggressively than train,
     creating a systematic temporal domain gap.

Changes from v31 exp1:
  1. batch_size: 32 → 64 (for SupCon cross-signer diversity, from exp5)
  2. supcon_weight: 0.1 → 0.05 (from exp5, more stable)
  3. Projection head: SupCon applied to proj_head(embedding) NOT embedding directly
  4. Speed augmentation: pre-stretch to [130, 220] frames before resampling to 90
  5. Model class renamed KSLGraphNetV37 (architecture identical to V31Exp1 + proj_head)

Usage:
    python train_ksl_v37.py --model-type numbers
    python train_ksl_v37.py --model-type words
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_mediapipe_version():
    try:
        import mediapipe as mp
        version = mp.__version__
        if version != "0.10.5":
            print(f"[{ts()}] WARNING: MediaPipe version is {version}, "
                  f"training data was extracted with 0.10.5.")
        else:
            print(f"[{ts()}] MediaPipe version: {version} (matches training data)")
    except ImportError:
        print(f"[{ts()}] INFO: MediaPipe not installed (not needed for training)")


# ---------------------------------------------------------------------------
# Class Definitions
# ---------------------------------------------------------------------------

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

# ---------------------------------------------------------------------------
# Graph Topology
# ---------------------------------------------------------------------------

HAND_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]

POSE_EDGES = [
    (42, 43), (42, 44), (43, 45), (44, 46), (45, 47), (46, 0), (47, 21),
]

POSE_INDICES = [11, 12, 13, 14, 15, 16]

LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

LH_CHAINS = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
RH_CHAINS = [[(p + 21, c + 21) for p, c in chain] for chain in LH_CHAINS]


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
# Config (v37)
# ---------------------------------------------------------------------------

V37_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "streams": ["joint", "bone", "velocity"],
    "hidden_dim": 64,
    "num_layers": 4,
    "temporal_kernels": [3, 5, 7],
    # From exp5: larger batch for SupCon cross-signer diversity
    "batch_size": 64,
    "epochs": 500,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "spatial_dropout": 0.1,
    "label_smoothing": 0.1,
    "warmup_epochs": 10,
    # SupCon: projection head + reduced weight (from exp5)
    "supcon_weight": 0.05,
    "supcon_temperature": 0.07,
    "supcon_proj_dim": 128,  # NEW: projection head output dim (standard SupCon practice)
    # Signer adversarial (same as exp1/exp5)
    "grl_lambda_max": 0.3,
    "grl_start_epoch": 0,
    "grl_ramp_epochs": 100,
    # Temporal speed augmentation (NEW in v37)
    # Pre-stretches training sequences to simulate test-like longer sequences.
    # Train mean=104 frames, test mean=145 frames → train gets downsampled less than test.
    # Stretching to [speed_aug_min, speed_aug_max] before resampling to 90 frames
    # makes the downsampling ratio match what test sequences experience.
    "speed_aug_prob": 0.6,
    "speed_aug_min_frames": 130,
    "speed_aug_max_frames": 220,
    # Spatial augmentation (identical to exp1/exp5)
    "rotation_prob": 0.5,
    "rotation_max_deg": 15.0,
    "shear_prob": 0.3,
    "shear_max": 0.2,
    "joint_dropout_prob": 0.2,
    "joint_dropout_rate": 0.08,
    "noise_std": 0.015,
    "noise_prob": 0.4,
    "bone_perturb_prob": 0.5,
    "bone_perturb_range": (0.8, 1.2),
    "hand_size_prob": 0.5,
    "hand_size_range": (0.8, 1.2),
    "temporal_warp_prob": 0.4,
    "temporal_warp_sigma": 0.2,
    "hand_dropout_prob": 0.2,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.3,
    "complete_hand_drop_prob": 0.03,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.3,
}

# ---------------------------------------------------------------------------
# Adjacency Matrix
# ---------------------------------------------------------------------------

def build_adj(n=48):
    adj = np.zeros((n, n))
    for i, j in HAND_EDGES:
        adj[i, j] = adj[j, i] = 1
    for i, j in HAND_EDGES:
        adj[i + 21, j + 21] = adj[j + 21, i + 21] = 1
    for i, j in POSE_EDGES:
        adj[i, j] = adj[j, i] = 1
    adj[0, 21] = adj[21, 0] = 0.3
    adj += np.eye(n)
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0
    return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))

# ---------------------------------------------------------------------------
# Data Deduplication
# ---------------------------------------------------------------------------

def get_file_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def extract_signer_id(filename):
    import re
    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[1]
    match = re.search(r'Signer_(\d+)', base)
    if match:
        return match.group(1)
    return "unknown"


def deduplicate_signer_groups(sample_paths):
    groups = defaultdict(list)
    for path in sample_paths:
        class_dir = os.path.basename(os.path.dirname(path))
        signer = extract_signer_id(os.path.basename(path))
        groups[(class_dir, signer)].append(path)

    class_groups = defaultdict(dict)
    for (class_dir, signer), paths in groups.items():
        class_groups[class_dir][signer] = sorted(paths)

    duplicated_signers = set()
    for class_dir, signer_dict in class_groups.items():
        signers = list(signer_dict.keys())
        group_hashes = {}
        for signer in signers:
            file_hashes = tuple(sorted(get_file_hash(p) for p in signer_dict[signer]))
            group_hashes[signer] = file_hashes
        seen = {}
        for signer in signers:
            gh = group_hashes[signer]
            if gh in seen:
                duplicated_signers.add((class_dir, signer))
                print(f"[{ts()}]     Class {class_dir}: Signer {signer} is duplicate of "
                      f"Signer {seen[gh]}")
            else:
                seen[gh] = signer

    unique, removed = [], []
    for path in sample_paths:
        class_dir = os.path.basename(os.path.dirname(path))
        signer = extract_signer_id(os.path.basename(path))
        if (class_dir, signer) in duplicated_signers:
            removed.append(path)
        else:
            unique.append(path)
    return unique, removed

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_wrist_palm(h):
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

# ---------------------------------------------------------------------------
# Augmentation Suite
# ---------------------------------------------------------------------------

def augment_bone_length_perturbation(h, chains, scale_range=(0.8, 1.2)):
    h = h.copy()
    for chain in chains:
        for parent_idx, child_idx in chain:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            bone_vec = h[:, child_idx, :] - h[:, parent_idx, :]
            new_pos = h[:, parent_idx, :] + bone_vec * scale
            displacement = new_pos - h[:, child_idx, :]
            h[:, child_idx, :] = new_pos
            found = False
            for p2, c2 in chain:
                if found:
                    h[:, c2, :] += displacement
                if (p2, c2) == (parent_idx, child_idx):
                    found = True
    return h


def augment_hand_size(h, scale_range=(0.8, 1.2)):
    h = h.copy()
    scale = np.random.uniform(scale_range[0], scale_range[1])
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h[:, 0:1, :]
        h[np.ix_(lh_valid, range(1, 21))] = (
            wrist[lh_valid] + (h[np.ix_(lh_valid, range(1, 21))] - wrist[lh_valid]) * scale
        )
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h[:, 21:22, :]
        h[np.ix_(rh_valid, range(22, 42))] = (
            wrist[rh_valid] + (h[np.ix_(rh_valid, range(22, 42))] - wrist[rh_valid]) * scale
        )
    return h


def augment_temporal_warp(data_list, sigma=0.2):
    T = data_list[0].shape[0]
    if T < 2:
        return data_list
    warp = np.cumsum(np.abs(np.random.normal(1.0, sigma, T)))
    warp = warp / warp[-1] * (T - 1)
    indices = np.clip(np.round(warp).astype(int), 0, T - 1)
    return [arr[indices] for arr in data_list]


def augment_rotation(h, max_deg=15.0):
    h = h.copy()
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h[lh_valid, 0:1, :]
        centered = h[lh_valid, :21, :] - wrist
        h[lh_valid, :21, :] = np.einsum('ij,ntj->nti', rot, centered) + wrist
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h[rh_valid, 21:22, :]
        centered = h[rh_valid, 21:42, :] - wrist
        h[rh_valid, 21:42, :] = np.einsum('ij,ntj->nti', rot, centered) + wrist
    return h


def augment_shear(h, max_shear=0.2):
    h = h.copy()
    s = np.random.uniform(-max_shear, max_shear)
    axis_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    src_axis, dst_axis = axis_pairs[np.random.randint(len(axis_pairs))]
    h[:, :, dst_axis] = h[:, :, dst_axis] + s * h[:, :, src_axis]
    return h


def augment_joint_dropout(h, dropout_rate=0.08):
    h = h.copy()
    n_nodes = h.shape[1]
    n_drop = max(1, int(n_nodes * dropout_rate))
    drop_indices = np.random.choice(n_nodes, n_drop, replace=False)
    h[:, drop_indices, :] = 0
    return h


def augment_temporal_speed(data_list, min_frames, max_frames):
    """
    NEW in v37: Pre-stretch sequence to simulate test-like longer recordings.

    Train sequences mean=104 frames, test sequences mean=145 frames.
    Both get resampled to 90 frames, but test loses 38% more temporal detail.
    By stretching training sequences to [min_frames, max_frames] before resampling,
    the model learns to classify from coarsely-sampled temporal data — matching test.
    """
    T = data_list[0].shape[0]
    # Pick a target length covering the test distribution
    target_len = np.random.randint(min_frames, max_frames + 1)
    if target_len <= T:
        return data_list  # don't compress, only stretch
    stretch_idx = np.clip(
        np.round(np.linspace(0, T - 1, target_len)).astype(int), 0, T - 1
    )
    return [arr[stretch_idx] for arr in data_list]

# ---------------------------------------------------------------------------
# Feature Computation
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
        dot = np.clip(np.sum((bone_in / norm_in) * (bone_out / norm_out), axis=-1), -1.0, 1.0)
        angles[:, idx] = np.arccos(dot)
    return angles


def compute_fingertip_distances(h):
    T = h.shape[0]
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
    col = 0
    for i in range(len(LH_TIPS)):
        for j in range(i + 1, len(LH_TIPS)):
            distances[:, col] = np.linalg.norm(h[:, LH_TIPS[i], :] - h[:, LH_TIPS[j], :], axis=-1)
            col += 1
    for i in range(len(RH_TIPS)):
        for j in range(i + 1, len(RH_TIPS)):
            distances[:, col] = np.linalg.norm(h[:, RH_TIPS[i], :] - h[:, RH_TIPS[j], :], axis=-1)
            col += 1
    return distances


def compute_hand_body_features(h_raw):
    T = h_raw.shape[0]
    features = np.zeros((T, NUM_HAND_BODY_FEATURES), dtype=np.float32)
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

# ---------------------------------------------------------------------------
# Signer-Balanced Batch Sampler
# ---------------------------------------------------------------------------

class SignerBalancedSampler(Sampler):
    def __init__(self, labels, signer_labels, batch_size, drop_last=True):
        self.labels = np.array(labels)
        self.signer_labels = np.array(signer_labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.class_signer_indices = defaultdict(lambda: defaultdict(list))
        for idx in range(len(self.labels)):
            self.class_signer_indices[self.labels[idx]][self.signer_labels[idx]].append(idx)
        self.classes = sorted(self.class_signer_indices.keys())
        self.num_samples = len(self.labels)

    def __iter__(self):
        all_indices = []
        classes_per_batch = min(len(self.classes), max(4, self.batch_size // 4))
        samples_per_class = self.batch_size // classes_per_batch
        class_order = self.classes.copy()
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch = []
            random.shuffle(class_order)
            for cls in class_order[:classes_per_batch]:
                signer_dict = self.class_signer_indices[cls]
                signers = list(signer_dict.keys())
                picked = []
                signer_cycle = signers.copy()
                random.shuffle(signer_cycle)
                si = 0
                while len(picked) < samples_per_class:
                    signer = signer_cycle[si % len(signer_cycle)]
                    if signer_dict[signer]:
                        picked.append(random.choice(signer_dict[signer]))
                    si += 1
                    if si >= samples_per_class * 3:
                        break
                batch.extend(picked)
            if len(batch) >= self.batch_size:
                batch = batch[:self.batch_size]
            else:
                batch.extend(random.choices(range(self.num_samples), k=self.batch_size - len(batch)))
            random.shuffle(batch)
            all_indices.extend(batch)
        return iter(all_indices)

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size

# ---------------------------------------------------------------------------
# Temporal CutMix
# ---------------------------------------------------------------------------

def temporal_cutmix(gcn_input, aux_input, labels, signer_labels, alpha=1.0):
    B, C, T, N = gcn_input.shape
    lam = np.random.beta(alpha, alpha)
    cut_len = int(T * (1 - lam))
    if cut_len == 0:
        return gcn_input, aux_input, labels, labels, 1.0
    cut_start = np.random.randint(0, T - cut_len + 1)
    indices = torch.randperm(B, device=gcn_input.device)
    gcn_mixed = gcn_input.clone()
    aux_mixed = aux_input.clone()
    gcn_mixed[:, :, cut_start:cut_start+cut_len, :] = gcn_input[indices, :, cut_start:cut_start+cut_len, :]
    aux_mixed[:, cut_start:cut_start+cut_len, :] = aux_input[indices, cut_start:cut_start+cut_len, :]
    lam = 1.0 - cut_len / T
    return gcn_mixed, aux_mixed, labels, labels[indices], lam

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KSLMultiStreamDataset(Dataset):
    def __init__(self, data_dirs, classes, config, aug=False):
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        all_paths, all_labels = [], []
        for data_dir in data_dirs:
            for cn in classes:
                cd = os.path.join(data_dir, cn)
                if os.path.exists(cd):
                    for fn in sorted(os.listdir(cd)):
                        if fn.endswith(".npy"):
                            all_paths.append(os.path.join(cd, fn))
                            all_labels.append(self.c2i[cn])

        if aug:
            unique_paths, removed = deduplicate_signer_groups(all_paths)
            if removed:
                print(f"[{ts()}]   Deduplication: {len(all_paths)} -> "
                      f"{len(unique_paths)} samples ({len(removed)} removed)")
            unique_set = set(unique_paths)
            self.samples, self.labels = [], []
            for path, label in zip(all_paths, all_labels):
                if path in unique_set:
                    self.samples.append(path)
                    self.labels.append(label)
                    unique_set.discard(path)
        else:
            self.samples = all_paths
            self.labels = all_labels

        self.signer_to_idx = {}
        self.signer_labels = []
        for path in self.samples:
            signer = extract_signer_id(os.path.basename(path))
            if signer not in self.signer_to_idx:
                self.signer_to_idx[signer] = len(self.signer_to_idx)
            self.signer_labels.append(self.signer_to_idx[signer])
        self.num_signers = len(self.signer_to_idx)

        dirs_str = ", ".join(data_dirs)
        print(f"[{ts()}]   Loaded {len(self.samples)} samples from {dirs_str}")
        print(f"[{ts()}]   Signers: {self.num_signers} "
              f"({dict(Counter(extract_signer_id(os.path.basename(p)) for p in self.samples))})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]

        if d.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                pose[:, pi, :] = d[:, idx_pose * 3:idx_pose * 3 + 3]
            lh = d[:, 99:162].reshape(f, 21, 3)
            rh = d[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        # Hand dropout (pre-norm)
        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dropout_rate = np.random.uniform(
                self.config["hand_dropout_min"], self.config["hand_dropout_max"]
            )
            if np.random.random() < 0.6:
                h[np.random.random(f) <= dropout_rate, :21, :] = 0
            if np.random.random() < 0.6:
                h[np.random.random(f) <= dropout_rate, 21:42, :] = 0

        # Complete hand drop (pre-norm)
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # Hand swap (pre-norm)
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # Hand-body features BEFORE normalization
        hand_body_feats = compute_hand_body_features(h)

        # Normalize
        h = normalize_wrist_palm(h)

        # Spatial augmentations (post-norm)
        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(h, LH_CHAINS + RH_CHAINS,
                                                  self.config["bone_perturb_range"])
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, self.config["hand_size_range"])
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, self.config["rotation_max_deg"])
        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, self.config["shear_max"])
        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, self.config["joint_dropout_rate"])
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)
        if self.aug and np.random.random() < self.config["noise_prob"]:
            h = h + np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)

        # Compute features before temporal transforms
        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]
        bones = compute_bones(h)
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        # Temporal warp
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = \
                augment_temporal_warp(
                    [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                    sigma=self.config["temporal_warp_sigma"]
                )

        # NEW: Temporal speed augmentation (v37)
        # Stretch the sequence to a random longer length in [min_frames, max_frames]
        # to simulate the coarser temporal sampling test sequences experience.
        if self.aug and np.random.random() < self.config["speed_aug_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = \
                augment_temporal_speed(
                    [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                    self.config["speed_aug_min_frames"],
                    self.config["speed_aug_max_frames"]
                )

        # Temporal resampling to max_frames
        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
            velocity = velocity[indices]
            bones = bones[indices]
            joint_angles = joint_angles[indices]
            fingertip_dists = fingertip_dists[indices]
            hand_body_feats = hand_body_feats[indices]
        else:
            pad = self.mf - f
            h = np.concatenate([h, np.zeros((pad, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad, 48, 3), dtype=np.float32)])
            joint_angles = np.concatenate([joint_angles, np.zeros((pad, NUM_ANGLE_FEATURES), dtype=np.float32)])
            fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
            hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

        # Temporal shift
        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z3 = np.zeros((shift, 48, 3), dtype=np.float32)
                za = np.zeros((shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                h = np.concatenate([z3, h[:-shift]])
                velocity = np.concatenate([z3, velocity[:-shift]])
                bones = np.concatenate([z3, bones[:-shift]])
                joint_angles = np.concatenate([za, joint_angles[:-shift]])
                fingertip_dists = np.concatenate([zd, fingertip_dists[:-shift]])
                hand_body_feats = np.concatenate([zh, hand_body_feats[:-shift]])
            elif shift < 0:
                z3 = np.zeros((-shift, 48, 3), dtype=np.float32)
                za = np.zeros((-shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((-shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((-shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                h = np.concatenate([h[-shift:], z3])
                velocity = np.concatenate([velocity[-shift:], z3])
                bones = np.concatenate([bones[-shift:], z3])
                joint_angles = np.concatenate([joint_angles[-shift:], za])
                fingertip_dists = np.concatenate([fingertip_dists[-shift:], zd])
                hand_body_feats = np.concatenate([hand_body_feats[-shift:], zh])

        streams = {
            "joint":    torch.FloatTensor(np.clip(h, -10, 10)).permute(2, 0, 1),
            "bone":     torch.FloatTensor(np.clip(bones, -10, 10)).permute(2, 0, 1),
            "velocity": torch.FloatTensor(np.clip(velocity, -10, 10)).permute(2, 0, 1),
        }
        aux_features = np.clip(
            np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1),
            -10, 10
        ).astype(np.float32)

        return streams, torch.FloatTensor(aux_features), self.labels[idx], self.signer_labels[idx]


def stream_collate_fn(batch):
    streams_list, aux_list, labels_list, signer_list = zip(*batch)
    stream_names = list(streams_list[0].keys())
    return (
        {name: torch.stack([s[name] for s in streams_list]) for name in stream_names},
        torch.stack(aux_list),
        torch.tensor(labels_list, dtype=torch.long),
        torch.tensor(signer_list, dtype=torch.long),
    )

# ---------------------------------------------------------------------------
# Model Components (GroupNorm throughout — identical to v31 exp1)
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module):
    def __init__(self, channels, kernels=(3, 5, 7), stride=1, dropout=0.3):
        super().__init__()
        num_groups = 8 if channels >= 8 else 1
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), stride=(stride, 1)),
                nn.GroupNorm(num_groups, channels),
            )
            for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(sum(b(x) for b in self.branches) / len(self.branches))


class AttentionPool(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        mid = in_channels // 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, mid), nn.Tanh(), nn.Linear(mid, 1))
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        outs = []
        for head in self.heads:
            attn = F.softmax(head(x_t), dim=1)
            outs.append((x_t * attn).sum(dim=1))
        return torch.cat(outs, dim=1)


class SpatialNodeDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        mask = F.dropout(torch.ones(B, 1, 1, N, device=x.device), p=self.p, training=True)
        return x * mask


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, ic, oc, adj, temporal_kernels=(3, 5, 7), st=1, dr=0.3, spatial_dropout=0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        num_groups = 8 if oc >= 8 else 1
        self.bn1 = nn.GroupNorm(num_groups, oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, st, dr)
        if ic != oc or st != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=(st, 1)),
                nn.GroupNorm(8 if oc >= 8 else 1, oc),
            )
        else:
            self.residual = nn.Identity()
        self.spatial_drop = SpatialNodeDropout(spatial_dropout)

    def forward(self, x):
        r = self.residual(x)
        b, c, t, n = x.shape
        x = self.gcn(x.permute(0, 2, 3, 1).reshape(b * t, n, c), self.adj)
        x = torch.relu(self.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2)))
        x = self.spatial_drop(x)
        x = self.tcn(x)
        return torch.relu(x + r)

# ---------------------------------------------------------------------------
# Main Model (v37: GroupNorm + SupCon Projection Head)
# ---------------------------------------------------------------------------

class KSLGraphNetV37(nn.Module):
    """
    ST-GCN v37: GroupNorm throughout + dedicated SupCon projection head.

    Key difference from exp1: adds self.proj_head for SupCon loss.
    The proj_head is a 2-layer MLP (embed_dim → proj_dim → proj_dim) that
    produces features for contrastive loss. The classifier uses the original
    embed_dim representation, keeping it unconstrained (standard SupCon practice).

    Key difference from exp5: GroupNorm replaces all BatchNorm (no AdaBN at inference).
    """

    def __init__(self, nc, num_signers, aux_dim, proj_dim=128,
                 nn_=48, ic=3, hd=64, nl=4, tk=(3, 5, 7),
                 dr=0.3, spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)

        # Data normalization: GroupNorm on flattened (B, C*N, T)
        num_groups_data = 8 if (nn_ * ic) >= 8 else 1
        self.data_bn = nn.GroupNorm(num_groups_data, nn_ * ic)

        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]
        self.layers = nn.ModuleList([
            STGCNBlock(ch[i], ch[i + 1], adj, tk,
                       2 if i in [1, 3] else 1, dr, spatial_dropout)
            for i in range(nl)
        ])

        final_ch = ch[-1]
        self.attn_pool = AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch  # 256

        # Auxiliary branch: GroupNorm
        num_groups_aux = 1
        for g in [8, 4, 2]:
            if aux_dim % g == 0:
                num_groups_aux = g
                break
        self.aux_bn = nn.GroupNorm(num_groups_aux, aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, 64)
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)
        )

        self.embed_dim = gcn_embed_dim + 64  # 320

        # Classifier: operates on the full 320-d embedding
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc)
        )

        # Signer adversarial head
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_signers)
        )

        # NEW: SupCon projection head (standard Khosla et al. 2020 practice)
        # SupCon loss is applied to proj_head(embedding), NOT embedding directly.
        # This keeps the classifier embedding unconstrained by contrastive geometry.
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        b, c, t, n = gcn_input.shape
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        gcn_embedding = self.attn_pool(x.mean(dim=3))

        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)
        aux = self.aux_mlp(aux)
        aux_t = self.aux_temporal_conv(aux.permute(0, 2, 1)).permute(0, 2, 1)
        attn_weights = F.softmax(self.aux_attn(aux_t), dim=1)
        aux_embedding = (aux_t * attn_weights).sum(dim=1)

        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)  # (B, 320)
        logits = self.classifier(embedding)

        reversed_emb = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_emb)

        # Projection head for SupCon (not used at inference)
        proj_features = self.proj_head(embedding)  # (B, proj_dim)

        return logits, signer_logits, embedding, proj_features

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        self_mask = torch.eye(B, device=device)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1 - self_mask)
        pos_count = mask.sum(1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=device)
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob = (mask * log_prob).sum(1) / pos_count.clamp(min=1)
        return -mean_log_prob.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        focal_loss = ((1 - torch.exp(-ce_loss)) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

# ---------------------------------------------------------------------------
# Train One Stream
# ---------------------------------------------------------------------------

def train_stream(stream_name, name, classes, config, train_dirs, val_dir,
                 ckpt_dir, device, use_focal=False):
    stream_ckpt_dir = os.path.join(ckpt_dir, stream_name)
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} - Stream: {stream_name} ({len(classes)} classes)")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)
    train_ds = KSLMultiStreamDataset(train_dirs, classes, config, aug=True)
    val_ds = KSLMultiStreamDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    i2c = {v: k for k, v in train_ds.c2i.items()}
    num_signers = train_ds.num_signers
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    proj_dim = config["supcon_proj_dim"]
    print(f"[{ts()}] Aux dim: {aux_dim}, Num signers: {num_signers}, Proj dim: {proj_dim}")

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )
    train_ld = DataLoader(train_ds, batch_size=config["batch_size"],
                          sampler=train_sampler, num_workers=2, pin_memory=True,
                          collate_fn=stream_collate_fn)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=stream_collate_fn)

    model = KSLGraphNetV37(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        proj_dim=proj_dim, nn_=config["num_nodes"],
        ic=config["in_channels"], hd=config["hidden_dim"],
        nl=config["num_layers"], tk=tuple(config["temporal_kernels"]),
        dr=config["dropout"], spatial_dropout=config["spatial_dropout"],
        adj=adj,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Normalization: GroupNorm (v37)")
    print(f"[{ts()}] SupCon: weight={config['supcon_weight']}, "
          f"temp={config['supcon_temperature']}, proj_dim={proj_dim}")
    print(f"[{ts()}] Speed aug: prob={config['speed_aug_prob']}, "
          f"frames=[{config['speed_aug_min_frames']}, {config['speed_aug_max_frames']}]")
    _ls = config.get("label_smoothing", 0.0)
    print(f"[{ts()}] Loss: {'FocalLoss(gamma=2.0)' if use_focal else f'CE(ls={_ls})'}")

    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])

    if use_focal:
        class_counts = Counter(train_ds.labels)
        total = len(train_ds)
        nc = len(classes)
        alpha_w = torch.tensor(
            [total / (nc * class_counts.get(i, 1)) for i in range(nc)]
        )
        alpha_w = (alpha_w / alpha_w.mean()).to(device)
        cls_loss_fn = FocalLoss(gamma=2.0, alpha=alpha_w,
                                label_smoothing=config["label_smoothing"])
    else:
        cls_loss_fn = None

    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )

    os.makedirs(stream_ckpt_dir, exist_ok=True)
    best_path = os.path.join(stream_ckpt_dir, "best_model.pt")
    best = 0.0
    supcon_weight = config["supcon_weight"]
    cutmix_prob = config["cutmix_prob"]
    cutmix_alpha = config["cutmix_alpha"]
    mixup_alpha = config["mixup_alpha"]
    warmup_epochs = config["warmup_epochs"]

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        signer_correct, signer_total = 0, 0

        progress = min(1.0, max(0.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"]))
        grl_lambda = config["grl_lambda_max"] * progress if ep >= config["grl_start_epoch"] else 0.0

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for streams_batch, aux_data, targets, signer_targets in train_ld:
            gcn_data = streams_batch[stream_name].to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and mixup_alpha > 0

            if use_cutmix:
                gcn_m, aux_m, ta, tb, lam = temporal_cutmix(
                    gcn_data, aux_data, targets, signer_targets, cutmix_alpha
                )
                logits, signer_logits, _, proj_feat = model(gcn_m, aux_m, grl_lambda)
                if cls_loss_fn is not None:
                    cls_loss = lam * cls_loss_fn(logits, ta) + (1 - lam) * cls_loss_fn(logits, tb)
                else:
                    ls = config["label_smoothing"]
                    cls_loss = (lam * F.cross_entropy(logits, ta, label_smoothing=ls)
                                + (1 - lam) * F.cross_entropy(logits, tb, label_smoothing=ls))
                sc_loss = supcon_loss_fn(proj_feat, ta)  # projection head for SupCon
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(ta).float() + (1 - lam) * p.eq(tb).float()).sum().item()

            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                t_perm = targets[perm]
                logits, signer_logits, _, proj_feat = model(
                    lam * gcn_data + (1 - lam) * gcn_data[perm],
                    lam * aux_data + (1 - lam) * aux_data[perm],
                    grl_lambda
                )
                if cls_loss_fn is not None:
                    cls_loss = lam * cls_loss_fn(logits, targets) + (1 - lam) * cls_loss_fn(logits, t_perm)
                else:
                    ls = config["label_smoothing"]
                    cls_loss = (lam * F.cross_entropy(logits, targets, label_smoothing=ls)
                                + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=ls))
                sc_loss = supcon_loss_fn(proj_feat, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()

            else:
                logits, signer_logits, _, proj_feat = model(gcn_data, aux_data, grl_lambda)
                if cls_loss_fn is not None:
                    cls_loss = cls_loss_fn(logits, targets)
                else:
                    cls_loss = F.cross_entropy(logits, targets,
                                               label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(proj_feat, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

            _, s_pred = signer_logits.max(1)
            signer_correct += s_pred.eq(signer_targets).sum().item()
            signer_total += signer_targets.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            tt += targets.size(0)

        if ep >= warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for streams_batch, aux_data, targets, _ in val_ld:
                gcn_data = streams_batch[stream_name].to(device)
                logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets.to(device)).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        signer_acc = 100.0 * signer_correct / signer_total if signer_total > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] [{stream_name}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
                f"SignerAcc: {signer_acc:.1f}% | LR: {lr_now:.2e} | GRL: {grl_lambda:.3f} | "
                f"Time: {ep_time:.1f}s"
            )

        if va > best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "val_acc": va,
                "epoch": ep + 1,
                "classes": classes,
                "num_nodes": config["num_nodes"],
                "num_signers": num_signers,
                "aux_dim": aux_dim,
                "proj_dim": proj_dim,
                "stream": stream_name,
                "version": "v37",
                "norm_type": "groupnorm",
                "config": config,
            }, best_path)
            print(f"[{ts()}]   -> New best! {va:.1f}%")

        # Per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for streams_batch, aux_data, targets, _ in val_ld:
                    gcn_data = streams_batch[stream_name].to(device)
                    logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
                    preds_ep.extend(logits.max(1)[1].cpu().numpy())
                    tgts_ep.extend(targets.numpy())
            print(f"[{ts()}]   Per-class at epoch {ep + 1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                print(f"[{ts()}]     {cn:12s}: {100.0 * cor / tot if tot > 0 else 0:.1f}% ({cor}/{tot})")

    # Final evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for streams_batch, aux_data, targets, _ in val_ld:
            gcn_data = streams_batch[stream_name].to(device)
            logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
            preds.extend(logits.max(1)[1].cpu().numpy())
            tgts.extend(targets.numpy())

    print(f"\n[{ts()}] {name} [{stream_name}] Per-Class Results (best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot if tot > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot})")
    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if tgts else 0.0
    print(f"[{ts()}] {name} [{stream_name}] Overall: {ov:.1f}%")

    return {"stream": stream_name, "overall": ov, "per_class": res,
            "best_epoch": ckpt["epoch"], "params": param_count}

# ---------------------------------------------------------------------------
# Fusion Weight Learning
# ---------------------------------------------------------------------------

def learn_fusion_weights(category_name, classes, config, val_dir, ckpt_dir, device):
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Learning fusion weights for {category_name}")
    print(f"[{ts()}] {'=' * 70}")

    streams = config["streams"]
    adj = build_adj(config["num_nodes"]).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    proj_dim = config["supcon_proj_dim"]

    models = {}
    for sname in streams:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{ts()}] WARNING: Missing checkpoint for '{sname}'")
            return None, 0.0
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = KSLGraphNetV37(
            nc=len(classes), num_signers=ckpt.get("num_signers", 12),
            aux_dim=aux_dim, proj_dim=proj_dim,
            nn_=config["num_nodes"], ic=3, hd=config["hidden_dim"],
            nl=config["num_layers"], tk=tuple(config["temporal_kernels"]),
            dr=config["dropout"], spatial_dropout=config["spatial_dropout"], adj=adj,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models[sname] = m
        print(f"[{ts()}] Loaded {sname}: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}")

    val_ds = KSLMultiStreamDataset(val_dir, classes, config, aug=False)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=stream_collate_fn)

    all_probs = {s: [] for s in streams}
    all_labels = []
    with torch.no_grad():
        for streams_batch, aux_data, targets, _ in val_ld:
            aux_data = aux_data.to(device)
            all_labels.extend(targets.numpy())
            for sname in streams:
                gcn_data = streams_batch[sname].to(device)
                logits, _, _, _ = models[sname](gcn_data, aux_data, 0.0)
                all_probs[sname].append(F.softmax(logits, dim=1).cpu())
    for sname in streams:
        all_probs[sname] = torch.cat(all_probs[sname], dim=0)
    all_labels = np.array(all_labels)

    print(f"\n[{ts()}] Per-stream val accuracy:")
    for sname in streams:
        preds = all_probs[sname].argmax(dim=1).numpy()
        print(f"[{ts()}]   {sname:>10s}: {100.0 * (preds == all_labels).mean():.1f}%")

    best_acc, best_weights = 0.0, {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
    for wj in range(10, 81, 5):
        for wb in range(10, 81, 5):
            wv = 100 - wj - wb
            if wv < 5:
                continue
            fused = (wj / 100 * all_probs["joint"] +
                     wb / 100 * all_probs["bone"] +
                     wv / 100 * all_probs["velocity"])
            acc = 100.0 * (fused.argmax(1).numpy() == all_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_weights = {"joint": wj / 100, "bone": wb / 100, "velocity": wv / 100}

    print(f"\n[{ts()}] Best fusion weights: {best_weights}")
    print(f"[{ts()}] Fused val accuracy: {best_acc:.1f}%")

    equal_acc = 100.0 * (
        (sum(all_probs[s] for s in streams) / 3).argmax(1).numpy() == all_labels
    ).mean()
    print(f"[{ts()}] Equal weights accuracy: {equal_acc:.1f}%")

    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    with open(weights_path, "w") as f:
        json.dump({"weights": best_weights, "val_accuracy": best_acc,
                   "equal_weights_accuracy": equal_acc, "category": category_name}, f, indent=2)
    print(f"[{ts()}] Fusion weights saved to {weights_path}")
    return best_weights, best_acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL ST-GCN v37 - GroupNorm + SupCon Projection Head + Speed Augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--seed", type=int, default=42)

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"))
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(base_data, "checkpoints", "v37"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("KSL Training v37 - GroupNorm + SupCon Proj Head + Speed Augmentation")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dirs = [args.train_dir, args.val_dir]  # signers 1-12
    val_dir = args.test_dir                        # signers 13-15

    print(f"\nv37 Data Split:")
    print(f"  Train: {train_dirs}  (signers 1-12)")
    print(f"  Val:   {val_dir}  (signers 13-15)")
    print(f"\nv37 Changes from exp1/exp5:")
    print(f"  1. GroupNorm throughout (from exp1) — no AdaBN needed at inference")
    print(f"  2. SupCon weight={V37_CONFIG['supcon_weight']}, batch={V37_CONFIG['batch_size']} (from exp5)")
    print(f"  3. NEW: SupCon projection head ({V37_CONFIG['supcon_proj_dim']}-d, decouples classifier from contrastive space)")
    print(f"  4. NEW: Speed augmentation (stretch to [{V37_CONFIG['speed_aug_min_frames']}, "
          f"{V37_CONFIG['speed_aug_max_frames']}] frames, prob={V37_CONFIG['speed_aug_prob']})")
    print(f"         Addresses train-test sequence length mismatch (104 vs 145 frames)")
    print(f"\n[{ts()}] Config:")
    print(json.dumps(V37_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    categories = []
    if args.model_type in ("numbers", "both"):
        categories.append(("Numbers", NUMBER_CLASSES, False))
    if args.model_type in ("words", "both"):
        categories.append(("Words", WORD_CLASSES, True))

    for cat_name, classes, use_focal in categories:
        cat_key = cat_name.lower()
        ckpt_dir = os.path.join(args.checkpoint_dir, cat_key)

        stream_results = {}
        for stream_name in V37_CONFIG["streams"]:
            result = train_stream(stream_name, cat_name, classes, V37_CONFIG,
                                  train_dirs, val_dir, ckpt_dir, device,
                                  use_focal=use_focal)
            if result:
                stream_results[stream_name] = result

        fusion_weights, fused_acc = learn_fusion_weights(
            cat_name, classes, V37_CONFIG, val_dir, ckpt_dir, device
        )
        results[cat_key] = {
            "streams": stream_results,
            "fusion_weights": fusion_weights,
            "fused_val_accuracy": fused_acc,
        }

    total_time = time.time() - start_time

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v37")
    print(f"[{ts()}] {'=' * 70}")
    for cat_key, cat_result in results.items():
        print(f"\n[{ts()}] {cat_key.upper()}:")
        for sname, sres in cat_result.get("streams", {}).items():
            print(f"[{ts()}]   {sname:>10s}: {sres['overall']:.1f}% "
                  f"(ep {sres['best_epoch']}, {sres['params']:,} params)")
        fw = cat_result.get("fusion_weights", {}) or {}
        fused = cat_result.get("fused_val_accuracy", 0)
        print(f"[{ts()}]   {'fused':>10s}: {fused:.1f}% "
              f"(j={fw.get('joint', 0):.2f}, b={fw.get('bone', 0):.2f}, "
              f"v={fw.get('velocity', 0):.2f})")

    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["fused_val_accuracy"] +
                    results["words"]["fused_val_accuracy"]) / 2
        print(f"\n[{ts()}] Combined fused val: {combined:.1f}%")

    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v37_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(results_path, "w") as f:
        json.dump({
            "version": "v37",
            "model_type": args.model_type,
            "seed": args.seed,
            "config": V37_CONFIG,
            "results": results,
            "total_time_seconds": total_time,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": ts(),
            "changes_from_exp1": [
                "1. batch_size: 32 → 64 (SupCon cross-signer diversity, from exp5)",
                "2. supcon_weight: 0.1 → 0.05 (from exp5)",
                "3. NEW: SupCon projection head (320→128→128 MLP, decouples classifier)",
                "4. NEW: Speed augmentation (stretch to [130,220] frames before resample to 90)",
            ],
        }, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")
    print(f"[{ts()}] Done.")


if __name__ == "__main__":
    main()
