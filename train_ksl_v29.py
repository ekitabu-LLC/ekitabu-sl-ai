#!/usr/bin/env python3
"""
KSL Training v29 (Alpine) - Deeper + Wider + EMA + Fixed Epochs

Based on v27. Key changes:

  1. 8 GCN LAYERS (up from 4): Channel progression ic->64->64->128->...->128,
     matching SOTA depth for skeleton-based action recognition.
  2. DILATED MULTI-SCALE TCN: kernel=3 with dilations=(1,2,4) replaces
     kernels=(3,5,7). Effective receptive fields of 3, 5, 9 with fewer params.
  3. 4 ATTENTION HEADS (up from 2): gcn_embed_dim = 4 * 128 = 512.
  4. WIDER AUX BRANCH: 128-dim output (was 64-dim).
  5. WIDER CLASSIFIER: 640->128->nc (was 320->64->nc).
  6. 300 FIXED EPOCHS with EMA, no early stopping.
  7. REDUCED REGULARIZATION: dropout 0.2, GRL 0.15, CutMix 0.15.
  8. EMA (decay=0.9998): Maintains shadow weights, saves both regular and EMA
     checkpoints. EMA val accuracy tracked alongside regular.
  9. AdamW with weight_decay=0.05.

Target ~1.3M params per model.

Usage:
    python train_ksl_v29.py --model-type numbers
    python train_ksl_v29.py --model-type words
    python train_ksl_v29.py --model-type both --seed 42
    python train_ksl_v29.py --model-type numbers --loso
"""

import argparse
import hashlib
import json
import math
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
    """Log a warning if MediaPipe version != 0.10.5."""
    try:
        import mediapipe as mp
        version = mp.__version__
        if version != "0.10.5":
            print(f"[{ts()}] WARNING: MediaPipe version is {version}, "
                  f"v27 training data was extracted with 0.10.5. "
                  f"Landmark coordinates may differ between versions.")
        else:
            print(f"[{ts()}] MediaPipe version: {version} (matches v27 training data)")
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

# Parent map for bone computation (48 nodes)
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]

PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

# Kinematic chains for bone-length perturbation
LH_CHAINS = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]

RH_CHAINS = [
    [(p + 21, c + 21) for p, c in chain]
    for chain in LH_CHAINS
]

# Joint angle topology
def _build_angle_joints():
    """Build list of (node, parent, child) triples for joint angle computation."""
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
NUM_FINGERTIP_PAIRS = 10  # C(5,2) = 10 per hand, 20 total
NUM_HAND_BODY_FEATURES = 8  # v25: hand-to-body spatial features

# ---------------------------------------------------------------------------
# Config (v29 - deeper/wider + EMA + fixed 300 epochs)
# ---------------------------------------------------------------------------

V29_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,           # GCN input: xyz(3) + velocity(3) + bone(3)
    "hidden_dim": 64,           # base dim (layers go 64->64->128->128->128->128->128->128)
    "num_layers": 8,            # UP from 4
    "temporal_dilations": [1, 2, 4],  # NEW: replaces temporal_kernels
    "batch_size": 32,
    "epochs": 300,              # UP from 200
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 0.05,       # UP from 1e-3 (AdamW-style)
    "dropout": 0.2,             # DOWN from 0.3
    "spatial_dropout": 0.1,     # same
    "label_smoothing": 0.05,    # DOWN from 0.1
    "patience": 999,            # effectively disabled (fixed epochs)
    "warmup_epochs": 20,        # UP from 10
    "ema_decay": 0.9998,        # NEW
    # SupCon
    "supcon_weight": 0.1,
    "supcon_temperature": 0.07,
    # GRL (REDUCED)
    "grl_lambda_max": 0.15,     # DOWN from 0.3
    "grl_start_epoch": 0,
    "grl_ramp_epochs": 100,
    # Augmentation (REDUCED from v27)
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
    "cutmix_prob": 0.15,        # DOWN from 0.3
}

# ---------------------------------------------------------------------------
# Adjacency Matrix Builder
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
# Data Deduplication (signer-group level only, same as v21)
# ---------------------------------------------------------------------------

def get_file_hash(filepath):
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def extract_signer_id(filename):
    """Extract signer ID from filename format CLASS-SIGNER-REP.npy."""
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
    """
    V21/v22 dedup: Only remove truly byte-identical FULL signer duplicates.
    Group files by (class_dir, signer_id). If two signer groups within the
    same class have identical file contents (all files match), remove one.
    """
    groups = defaultdict(list)
    for path in sample_paths:
        class_dir = os.path.basename(os.path.dirname(path))
        signer = extract_signer_id(os.path.basename(path))
        groups[(class_dir, signer)].append(path)

    class_groups = defaultdict(dict)
    for (class_dir, signer), paths in groups.items():
        class_groups[class_dir][signer] = sorted(paths)

    duplicated_signers = set()
    total_removed = 0

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
                total_removed += len(signer_dict[signer])
                print(f"[{ts()}]     Class {class_dir}: Signer {signer} is duplicate of "
                      f"Signer {seen[gh]} ({len(signer_dict[signer])} files)")
            else:
                seen[gh] = signer

    unique = []
    removed = []
    for path in sample_paths:
        class_dir = os.path.basename(os.path.dirname(path))
        signer = extract_signer_id(os.path.basename(path))
        if (class_dir, signer) in duplicated_signers:
            removed.append(path)
        else:
            unique.append(path)

    return unique, removed


# ---------------------------------------------------------------------------
# Wrist-Centric + Palm-Size Normalization
# ---------------------------------------------------------------------------

def normalize_wrist_palm(h):
    """
    Wrist-centric + palm-size normalization.
    Left hand (0-20): center at wrist (node 0), scale by palm size (node 0 -> node 9)
    Right hand (21-41): center at wrist (node 21), scale by palm size (node 21 -> node 30)
    Pose (42-47): center at mid-shoulder (42+43)/2, scale by shoulder width
    """
    T = h.shape[0]

    # -- Left hand --
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01

    if np.any(lh_valid):
        lh_wrist = lh[:, 0:1, :]
        lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]
        palm_sizes = np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]

    h[:, :21, :] = lh

    # -- Right hand --
    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01

    if np.any(rh_valid):
        rh_wrist = rh[:, 0:1, :]
        rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]
        palm_sizes = np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]

    h[:, 21:42, :] = rh

    # -- Pose --
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
# Augmentation Suite (v22 - MODERATE, reduced from v21)
# ---------------------------------------------------------------------------

def augment_bone_length_perturbation(h, chains, scale_range=(0.8, 1.2)):
    """Perturb bone lengths by walking kinematic chains from root outward."""
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
    """Scale all hand joints relative to their wrist."""
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
    """Non-uniform temporal resampling using cumulative random normal."""
    T = data_list[0].shape[0]
    if T < 2:
        return data_list
    warp = np.cumsum(np.abs(np.random.normal(1.0, sigma, T)))
    warp = warp / warp[-1] * (T - 1)
    indices = np.clip(np.round(warp).astype(int), 0, T - 1)
    return [arr[indices] for arr in data_list]


def augment_rotation(h, max_deg=15.0):
    """Rotate hand joints around wrist by random angle (Z-axis rotation)."""
    h = h.copy()
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h[lh_valid, 0:1, :]
        centered = h[lh_valid, :21, :] - wrist
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h[lh_valid, :21, :] = rotated + wrist

    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h[rh_valid, 21:22, :]
        centered = h[rh_valid, 21:42, :] - wrist
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h[rh_valid, 21:42, :] = rotated + wrist

    return h


def augment_shear(h, max_shear=0.2):
    """Apply random shear transformation on a random pair of axes."""
    h = h.copy()
    s = np.random.uniform(-max_shear, max_shear)
    axis_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    src_axis, dst_axis = axis_pairs[np.random.randint(len(axis_pairs))]
    h[:, :, dst_axis] = h[:, :, dst_axis] + s * h[:, :, src_axis]
    return h


def augment_joint_dropout(h, dropout_rate=0.08):
    """Zero out random joints across all frames."""
    h = h.copy()
    n_nodes = h.shape[1]
    n_drop = max(1, int(n_nodes * dropout_rate))
    drop_indices = np.random.choice(n_nodes, n_drop, replace=False)
    h[:, drop_indices, :] = 0
    return h


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors: bone[child] = node[child] - node[parent]."""
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


def compute_hand_body_features(h_raw):
    """
    Compute hand-to-body spatial features BEFORE wrist-centric normalization.
    h_raw: (T, 48, 3) -- raw landmarks, not yet normalized
    Returns: (T, 8) feature vector per frame
    """
    T = h_raw.shape[0]
    features = np.zeros((T, NUM_HAND_BODY_FEATURES), dtype=np.float32)

    mid_shoulder = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2
    shoulder_width = np.linalg.norm(
        h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True
    )
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    lh_centroid = h_raw[:, :21, :].mean(axis=1)
    rh_centroid = h_raw[:, 21:42, :].mean(axis=1)

    features[:, 0] = (lh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]
    features[:, 1] = (rh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]
    features[:, 2] = (lh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
    features[:, 3] = (rh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
    features[:, 4] = np.linalg.norm(
        lh_centroid - rh_centroid, axis=-1
    ) / shoulder_width[:, 0]

    face_approx = mid_shoulder.copy()
    face_approx[:, 1] -= shoulder_width[:, 0] * 0.7
    features[:, 5] = np.linalg.norm(
        lh_centroid - face_approx, axis=-1
    ) / shoulder_width[:, 0]
    features[:, 6] = np.linalg.norm(
        rh_centroid - face_approx, axis=-1
    ) / shoulder_width[:, 0]

    features[:, 7] = np.abs(lh_centroid[:, 1] - rh_centroid[:, 1]) / shoulder_width[:, 0]

    return features


# ---------------------------------------------------------------------------
# Signer-Balanced Batch Sampler (v22 NEW)
# ---------------------------------------------------------------------------

class SignerBalancedSampler(Sampler):
    """
    Custom sampler that ensures each batch contains samples from multiple signers
    for the same class, maximizing cross-signer positive pairs for SupCon loss.
    """

    def __init__(self, labels, signer_labels, batch_size, drop_last=True):
        self.labels = np.array(labels)
        self.signer_labels = np.array(signer_labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.class_signer_indices = defaultdict(lambda: defaultdict(list))
        for idx in range(len(self.labels)):
            cls = self.labels[idx]
            signer = self.signer_labels[idx]
            self.class_signer_indices[cls][signer].append(idx)

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
            selected_classes = class_order[:classes_per_batch]

            for cls in selected_classes:
                signer_dict = self.class_signer_indices[cls]
                signers = list(signer_dict.keys())

                picked = []
                signer_cycle = signers.copy()
                random.shuffle(signer_cycle)

                si = 0
                while len(picked) < samples_per_class:
                    signer = signer_cycle[si % len(signer_cycle)]
                    indices = signer_dict[signer]
                    if indices:
                        chosen = random.choice(indices)
                        picked.append(chosen)
                    si += 1
                    if si >= samples_per_class * 3:
                        break

                batch.extend(picked)

            if len(batch) >= self.batch_size:
                batch = batch[:self.batch_size]
            else:
                remaining = self.batch_size - len(batch)
                extra = random.choices(range(self.num_samples), k=remaining)
                batch.extend(extra)

            random.shuffle(batch)
            all_indices.extend(batch)

        return iter(all_indices)

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size


# ---------------------------------------------------------------------------
# Temporal CutMix (v25 - batch-level augmentation)
# ---------------------------------------------------------------------------

def temporal_cutmix(gcn_input, aux_input, labels, signer_labels, alpha=1.0):
    """
    CutMix at the temporal dimension for skeleton sequences.
    Splices a contiguous temporal segment from shuffled samples.
    Recomputes velocity channels (3:6) at splice boundaries.
    """
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

    if cut_start > 0:
        gcn_mixed[:, 3:6, cut_start, :] = gcn_mixed[:, 0:3, cut_start, :] - gcn_mixed[:, 0:3, cut_start-1, :]
    if cut_start + cut_len < T:
        gcn_mixed[:, 3:6, cut_start+cut_len, :] = gcn_mixed[:, 0:3, cut_start+cut_len, :] - gcn_mixed[:, 0:3, cut_start+cut_len-1, :]

    lam = 1.0 - cut_len / T
    labels_b = labels[indices]

    return gcn_mixed, aux_mixed, labels, labels_b, lam


# ---------------------------------------------------------------------------
# Dataset (v29 - same as v27, multi-directory support)
# ---------------------------------------------------------------------------

class KSLGraphDataset(Dataset):
    def __init__(self, data_dirs, classes, config, aug=False):
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        all_paths = []
        all_labels = []
        for data_dir in data_dirs:
            for cn in classes:
                cd = os.path.join(data_dir, cn)
                if os.path.exists(cd):
                    for fn in sorted(os.listdir(cd)):
                        if fn.endswith(".npy"):
                            all_paths.append(os.path.join(cd, fn))
                            all_labels.append(self.c2i[cn])

        total_before = len(all_paths)

        if aug:
            unique_paths, removed = deduplicate_signer_groups(all_paths)
            if removed:
                print(f"[{ts()}]   Deduplication (signer-group): {total_before} -> "
                      f"{len(unique_paths)} samples ({len(removed)} removed)")
            unique_set = set(unique_paths)
            self.samples = []
            self.labels = []
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
        print(f"[{ts()}]   Signers: {self.num_signers} unique "
              f"({dict(Counter(extract_signer_id(os.path.basename(p)) for p in self.samples))})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]

        if d.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                start = idx_pose * 3
                pose[:, pi, :] = d[:, start:start + 3]
            lh = d[:, 99:162].reshape(f, 21, 3)
            rh = d[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dropout_rate = np.random.uniform(
                self.config["hand_dropout_min"], self.config["hand_dropout_max"]
            )
            if np.random.random() < 0.6:
                lh_mask = np.random.random(f) > dropout_rate
                h[~lh_mask, :21, :] = 0
            if np.random.random() < 0.6:
                rh_mask = np.random.random(f) > dropout_rate
                h[~rh_mask, 21:42, :] = 0

        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        hand_body_feats = compute_hand_body_features(h)
        h = normalize_wrist_palm(h)

        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(
                h, LH_CHAINS + RH_CHAINS,
                scale_range=self.config["bone_perturb_range"]
            )

        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])

        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])

        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, max_shear=self.config["shear_max"])

        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, dropout_rate=self.config["joint_dropout_rate"])

        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)

        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]
        bones = compute_bones(h)
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                sigma=self.config["temporal_warp_sigma"]
            )

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
            pad_len = self.mf - f
            h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
            fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
            hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z3 = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z3, h[:-shift]], axis=0)
                velocity = np.concatenate([z3, velocity[:-shift]], axis=0)
                bones = np.concatenate([z3, bones[:-shift]], axis=0)
                za = np.zeros((shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([za, joint_angles[:-shift]], axis=0)
                fingertip_dists = np.concatenate([zd, fingertip_dists[:-shift]], axis=0)
                hand_body_feats = np.concatenate([zh, hand_body_feats[:-shift]], axis=0)
            elif shift < 0:
                z3 = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z3], axis=0)
                velocity = np.concatenate([velocity[-shift:], z3], axis=0)
                bones = np.concatenate([bones[-shift:], z3], axis=0)
                za = np.zeros((-shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((-shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((-shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([joint_angles[-shift:], za], axis=0)
                fingertip_dists = np.concatenate([fingertip_dists[-shift:], zd], axis=0)
                hand_body_feats = np.concatenate([hand_body_feats[-shift:], zh], axis=0)

        gcn_features = np.concatenate([h, velocity, bones], axis=2)
        gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)

        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)

        gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)
        aux_tensor = torch.FloatTensor(aux_features)

        return gcn_tensor, aux_tensor, self.labels[idx], self.signer_labels[idx]


# ---------------------------------------------------------------------------
# Dilated Multi-Scale TCN (v29 NEW - replaces MultiScaleTCN)
# ---------------------------------------------------------------------------

class DilatedMultiScaleTCN(nn.Module):
    """Dilated multi-scale temporal convolution.

    Uses kernel_size=3 with dilations=(1,2,4) instead of kernels=(3,5,7).
    Effective receptive fields: 3, 5, 9 with fewer parameters.
    """

    def __init__(self, channels, dilations=(1, 2, 4), stride=1, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (3, 1),
                          padding=(d, 0), dilation=(d, 1), stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )
            for d in dilations
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Attention Pooling (v29: 4 heads)
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        mid = in_channels // 2

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, mid),
                nn.Tanh(),
                nn.Linear(mid, 1),
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        b, c, t = x.shape
        x_t = x.permute(0, 2, 1)  # (B, T, C)

        outs = []
        for head in self.heads:
            attn_logits = head(x_t)
            attn_weights = F.softmax(attn_logits, dim=1)
            pooled = (x_t * attn_weights).sum(dim=1)
            outs.append(pooled)

        return torch.cat(outs, dim=1)  # (B, num_heads * C)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
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

        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self_mask = torch.eye(B, device=device)
        mask = mask * (1 - self_mask)

        pos_count = mask.sum(1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=device)

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob = (mask * log_prob).sum(1) / pos_count.clamp(min=1)

        return -mean_log_prob.mean()


# ---------------------------------------------------------------------------
# Focal Loss (v27 - for words model to address Market sink class)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss with optional per-class alpha weighting and label smoothing."""

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Spatial Dropout
# ---------------------------------------------------------------------------

class SpatialNodeDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        mask = torch.ones(B, 1, 1, N, device=x.device)
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) - v29 NEW
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ---------------------------------------------------------------------------
# Model (v29: 8-layer ST-GCN + Dilated TCN + 4-head Attn + Wider Aux/Classifier)
# ---------------------------------------------------------------------------

class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, ic, oc, adj, temporal_dilations=(1, 2, 4), st=1, dr=0.2,
                 spatial_dropout=0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = DilatedMultiScaleTCN(oc, temporal_dilations, st, dr)
        self.residual = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc))
            if ic != oc or st != 1
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dr)
        self.spatial_drop = SpatialNodeDropout(spatial_dropout)

    def forward(self, x):
        r = self.residual(x)
        b, c, t, n = x.shape
        x = self.gcn(x.permute(0, 2, 3, 1).reshape(b * t, n, c), self.adj)
        x = x.reshape(b, t, n, -1).permute(0, 3, 1, 2)
        x = torch.relu(self.bn1(x))
        x = self.spatial_drop(x)
        x = self.tcn(x)
        return torch.relu(x + r)


class KSLGraphNetV29(nn.Module):
    """
    V29 ST-GCN with:
    - 8 GCN layers: ic->64->64->128->128->128->128->128->128
    - Dilated multi-scale TCN (k=3, d={1,2,4})
    - 4-head attention pooling (gcn_embed_dim = 4*128 = 512)
    - Wider aux branch (128-dim output with temporal conv1d)
    - Wider classifier: 640->128->nc
    - Signer-adversarial head with GRL
    - Embedding output for SupCon loss
    """

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=9, hd=64, nl=8,
                 td=(1, 2, 4), dr=0.2, spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        # Channel progression: ic->64->64->128->128->128->128->128->128
        ch = [ic, 64, 64, 128, 128, 128, 128, 128, 128]
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, td,
                        2 if i in [1, 3] else 1,
                        dr, spatial_dropout)
             for i in range(nl)]
        )

        final_ch = ch[-1]

        # Attention pooling for GCN output: 4 heads
        self.attn_pool = AttentionPool(final_ch, num_heads=4)
        gcn_embed_dim = 4 * final_ch  # 4 heads * 128 = 512

        # v29: WIDER Auxiliary MLP for angle/distance features
        # aux_mlp: aux_dim -> 256 -> 128 (was aux_dim -> 128 -> 64)
        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 256),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 128),
        )
        # v29: Wider temporal conv (128 channels, was 64)
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Temporal attention pooling for aux features (over 128-dim)
        self.aux_attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Combined embedding dimension: 512 + 128 = 640 (was 256 + 64 = 320)
        self.embed_dim = gcn_embed_dim + 128

        # v29: Wider classifier: 640->128->nc (was 320->64->nc)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(128, nc),
        )

        # v29: Wider signer adversarial head: 640->128->num_signers (was 320->64->num_signers)
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_signers),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        b, c, t, n = gcn_input.shape

        # GCN branch
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        x_node_avg = x.mean(dim=3)  # (B, C_final, T')
        gcn_embedding = self.attn_pool(x_node_avg)  # (B, gcn_embed_dim)

        # Wider auxiliary branch with temporal conv
        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)
        aux = self.aux_mlp(aux)  # (B, T, 128)

        aux_t = aux.permute(0, 2, 1)  # (B, 128, T)
        aux_t = self.aux_temporal_conv(aux_t)  # (B, 128, T)
        aux = aux_t.permute(0, 2, 1)  # (B, T, 128)

        attn_logits = self.aux_attn(aux)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        aux_embedding = (aux * attn_weights).sum(dim=1)  # (B, 128)

        # Combine embeddings
        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)  # (B, 640)

        # Classification
        logits = self.classifier(embedding)

        # Signer adversarial
        reversed_embedding = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_embedding)

        return logits, signer_logits, embedding


# ---------------------------------------------------------------------------
# Training loop for one split (numbers or words) - v29
# ---------------------------------------------------------------------------
def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device):
    """Train one split (numbers or words) with v29.

    V29 changes vs v27:
    - KSLGraphNetV29 with 8-layer dilated TCN + 4-head attention + wider aux (128-dim)
    - EMA (Exponential Moving Average) of model weights
    - No focal loss - both splits use CE with label_smoothing
    """

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v29")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLGraphDataset(train_dir, classes, config, aug=True)
    val_ds = KSLGraphDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found in {train_dir}")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers

    # Compute auxiliary feature dimension from constants (v25: +8 hand-body features)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    print(f"[{ts()}] Auxiliary features: {NUM_ANGLE_FEATURES} joint angles + "
          f"{2 * NUM_FINGERTIP_PAIRS} fingertip distances + "
          f"{NUM_HAND_BODY_FEATURES} hand-body features = {aux_dim} total")
    print(f"[{ts()}] Num signers in training: {num_signers}")

    # v22: Signer-balanced batch sampler for training
    train_sampler = SignerBalancedSampler(
        train_ds.labels,
        train_ds.signer_labels,
        batch_size=config["batch_size"],
        drop_last=True,
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,  # sampler handles this
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model (v29 - 8-layer dilated TCN, 4-head attention, wider aux)
    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)

    # EMA (Exponential Moving Average)
    ema = EMA(model, decay=config["ema_decay"])

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Config: 8 layers (hardcoded), dilated TCN (1,2,4), "
          f"dropout={config['dropout']}, wd={config['weight_decay']}, "
          f"ls={config['label_smoothing']}, in_ch={config['in_channels']}")
    print(f"[{ts()}] Single-stage: CE + SupCon(weight={config['supcon_weight']})")
    print(f"[{ts()}] GRL: lambda_max={config['grl_lambda_max']}, "
          f"start_epoch={config['grl_start_epoch']}, ramp={config['grl_ramp_epochs']}")
    print(f"[{ts()}] Aux branch: 128-dim output with temporal conv1d "
          f"(~{100 * 128 / model.embed_dim:.0f}% of embedding)")
    print(f"[{ts()}] Signer-balanced batch sampling: ENABLED")
    print(f"[{ts()}] EMA: decay={config['ema_decay']}")
    print(f"[{ts()}] Augmentation: rotation_deg={config['rotation_max_deg']}, "
          f"shear={config['shear_max']}, joint_dropout={config['joint_dropout_rate']}, "
          f"noise={config['noise_std']}, temporal_warp_sigma={config['temporal_warp_sigma']}, "
          f"NO axis_mask")

    # Losses
    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])
    print(f"[{ts()}] Loss: CE(ls={config['label_smoothing']})")

    # Optimizer + scheduler
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    warmup_epochs = config["warmup_epochs"]

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    ema_best_path = os.path.join(ckpt_dir, "best_model_ema.pt")

    best, patience_counter = 0.0, 0
    ema_best = 0.0
    mixup_alpha = config.get("mixup_alpha", 0)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    cutmix_prob = config.get("cutmix_prob", 0.3)
    supcon_weight = config["supcon_weight"]

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        signer_correct, signer_total = 0, 0

        # Compute GRL lambda for this epoch (from epoch 0)
        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        # Warmup LR
        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for gcn_data, aux_data, targets, signer_targets in train_ld:
            gcn_data = gcn_data.to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            opt.zero_grad()

            # --- v25: CutMix or Mixup (mutually exclusive per batch) ---
            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and (mixup_alpha > 0)

            if use_cutmix:
                gcn_mixed, aux_mixed, targets_a, targets_b, lam = temporal_cutmix(
                    gcn_data, aux_data, targets, signer_targets,
                    alpha=cutmix_alpha,
                )

                logits, signer_logits, embedding = model(
                    gcn_mixed, aux_mixed, grl_lambda=grl_lambda,
                )

                cls_loss = (
                    lam * F.cross_entropy(logits, targets_a, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, targets_b, label_smoothing=config["label_smoothing"])
                )

                sc_loss = supcon_loss_fn(embedding, targets_a)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss

                _, p = logits.max(1)
                tc += (lam * p.eq(targets_a).float() + (1 - lam) * p.eq(targets_b).float()).sum().item()

            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_mixed = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_mixed = lam * aux_data + (1 - lam) * aux_data[perm]
                t_perm = targets[perm]

                logits, signer_logits, embedding = model(
                    gcn_mixed, aux_mixed, grl_lambda=grl_lambda,
                )

                cls_loss = (
                    lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                )

                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss

                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()

            else:
                logits, signer_logits, embedding = model(
                    gcn_data, aux_data, grl_lambda=grl_lambda,
                )

                cls_loss = F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)

                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss

                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

            # Track signer classifier accuracy (to monitor GRL effectiveness)
            _, s_pred = signer_logits.max(1)
            signer_correct += s_pred.eq(signer_targets).sum().item()
            signer_total += signer_targets.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # EMA update after each optimizer step
            ema.update()

            tl += loss.item()
            tt += targets.size(0)

        # Step scheduler (after warmup)
        if ep >= warmup_epochs:
            scheduler.step()

        # Validation (regular model)
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0

        # EMA validation
        ema.apply_shadow()
        model.eval()
        ema_vc, ema_vt = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                ema_vt += targets.size(0)
                ema_vc += p.eq(targets).sum().item()
        ema_va = 100.0 * ema_vc / ema_vt if ema_vt > 0 else 0.0
        ema.restore()

        ta = 100.0 * tc / tt if tt > 0 else 0.0
        signer_acc = 100.0 * signer_correct / signer_total if signer_total > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best or ema_va > ema_best:
            print(
                f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
                f"EMA-Val: {ema_va:.1f}% | "
                f"SignerAcc: {signer_acc:.1f}% | "
                f"LR: {lr_now:.2e} | GRL: {grl_lambda:.3f} | Time: {ep_time:.1f}s"
            )

        if va > best:
            best, patience_counter = va, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": va,
                    "epoch": ep + 1,
                    "classes": classes,
                    "num_nodes": config["num_nodes"],
                    "num_signers": num_signers,
                    "aux_dim": aux_dim,
                    "version": "v29",
                    "config": config,
                },
                best_path,
            )
            print(f"[{ts()}]   -> New best! {va:.1f}%")

            # Save EMA model checkpoint
            ema.apply_shadow()
            torch.save({
                "model": model.state_dict(),
                "val_acc": va,
                "epoch": ep + 1,
                "classes": classes,
                "num_nodes": config["num_nodes"],
                "num_signers": num_signers,
                "aux_dim": aux_dim,
                "version": "v29_ema",
                "config": config,
            }, ema_best_path)
            ema.restore()
        else:
            patience_counter += 1

        # Track best EMA val accuracy separately
        if ema_va > ema_best:
            ema_best = ema_va
            print(f"[{ts()}]   -> New best EMA! {ema_va:.1f}%")

        if patience_counter >= config["patience"]:
            print(f"[{ts()}] Early stopping at epoch {ep + 1}")
            break

        # Print per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for gcn_data, aux_data, targets, _ in val_ld:
                    logits, _, _ = model(gcn_data.to(device), aux_data.to(device),
                                         grl_lambda=0.0)
                    _, p = logits.max(1)
                    preds_ep.extend(p.cpu().numpy())
                    tgts_ep.extend(targets.cpu().numpy())
            print(f"[{ts()}]   Per-class at epoch {ep + 1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot_cls = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                acc = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
                print(f"[{ts()}]     {cn:12s}: {acc:5.1f}% ({cor}/{tot_cls})")

    # Final evaluation
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in val_ld:
            logits, _, _ = model(gcn_data.to(device), aux_data.to(device),
                                 grl_lambda=0.0)
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    print(f"\n[{ts()}] {name} Per-Class Results (final, best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_cls})")

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0
    print(f"[{ts()}] {name} Overall: {ov:.1f}%")

    # EMA final evaluation
    ema_ckpt = torch.load(ema_best_path, map_location=device, weights_only=False)
    model.load_state_dict(ema_ckpt["model"])
    model.eval()

    ema_preds, ema_tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in val_ld:
            logits, _, _ = model(gcn_data.to(device), aux_data.to(device),
                                 grl_lambda=0.0)
            _, p = logits.max(1)
            ema_preds.extend(p.cpu().numpy())
            ema_tgts.extend(targets.cpu().numpy())

    ema_ov = 100.0 * sum(1 for t, p in zip(ema_tgts, ema_preds) if t == p) / len(ema_tgts) if len(ema_tgts) > 0 else 0.0
    print(f"[{ts()}] {name} EMA Overall: {ema_ov:.1f}% (best EMA val: {ema_best:.1f}%)")

    # Confusion matrix
    print(f"\n[{ts()}] {name} Confusion Matrix:")
    nc = len(classes)
    cm = [[0] * nc for _ in range(nc)]
    for t, p in zip(tgts, preds):
        cm[t][p] += 1

    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"[{ts()}] {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{cm[i][j]:5d}" for j in range(nc))
        print(f"[{ts()}] {row_str}")

    return {
        "overall": ov,
        "per_class": res,
        "best_epoch": ckpt["epoch"],
        "params": param_count,
        "ema_overall": ema_ov,
        "ema_best_val": ema_best,
    }

# Part 2 - appended to train_ksl_v29.py


# ---------------------------------------------------------------------------
# LOSO (Leave-One-Signer-Out) Cross-Validation
# ---------------------------------------------------------------------------

def collect_all_samples(data_dir, classes):
    """Collect all sample paths and labels from a data directory."""
    c2i = {c: i for i, c in enumerate(classes)}
    all_paths = []
    all_labels = []
    for cn in classes:
        cd = os.path.join(data_dir, cn)
        if os.path.exists(cd):
            for fn in sorted(os.listdir(cd)):
                if fn.endswith(".npy"):
                    all_paths.append(os.path.join(cd, fn))
                    all_labels.append(c2i[cn])
    return all_paths, all_labels


def run_loso(name, classes, config, train_dir, val_dir, ckpt_dir, results_dir, device):
    """
    Run leave-one-signer-out cross-validation.
    Combines train+val data, deduplicates, then holds out one signer per fold.
    After dedup, signers 2 & 3 collapse, giving ~4 unique signers for numbers.
    """
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] LOSO Cross-Validation: {name} ({len(classes)} classes) - v29")
    print(f"[{ts()}] {'=' * 70}")

    # Collect all data from both train and val dirs
    train_paths, train_labels = collect_all_samples(train_dir, classes)
    val_paths, val_labels = collect_all_samples(val_dir, classes)
    all_paths = train_paths + val_paths
    all_labels = train_labels + val_labels

    print(f"[{ts()}] Total samples before dedup: {len(all_paths)}")

    # Deduplicate
    unique_paths, removed = deduplicate_signer_groups(all_paths)
    if removed:
        print(f"[{ts()}] Deduplication: {len(all_paths)} -> {len(unique_paths)} "
              f"({len(removed)} removed)")

    unique_set = set(unique_paths)
    dedup_paths = []
    dedup_labels = []
    for path, label in zip(all_paths, all_labels):
        if path in unique_set:
            dedup_paths.append(path)
            dedup_labels.append(label)
            unique_set.discard(path)

    # Identify unique signers
    signer_ids = [extract_signer_id(os.path.basename(p)) for p in dedup_paths]
    unique_signers = sorted(set(signer_ids))
    print(f"[{ts()}] Unique signers after dedup: {unique_signers}")
    print(f"[{ts()}] Samples per signer: "
          f"{dict(Counter(signer_ids))}")

    fold_results = []
    for fold_idx, held_out_signer in enumerate(unique_signers):
        print(f"\n[{ts()}] --- LOSO Fold {fold_idx + 1}/{len(unique_signers)}: "
              f"Hold out signer {held_out_signer} ---")

        # Split into train/val for this fold
        fold_train_paths = []
        fold_train_labels = []
        fold_val_paths = []
        fold_val_labels = []

        for path, label, signer in zip(dedup_paths, dedup_labels, signer_ids):
            if signer == held_out_signer:
                fold_val_paths.append(path)
                fold_val_labels.append(label)
            else:
                fold_train_paths.append(path)
                fold_train_labels.append(label)

        print(f"[{ts()}] Train: {len(fold_train_paths)}, Val: {len(fold_val_paths)}")

        # Create datasets directly from paths (no dedup needed, already done)
        fold_ckpt_dir = os.path.join(ckpt_dir, f"loso_fold{fold_idx}")
        fold_result = train_split_from_paths(
            f"{name}_Fold{fold_idx+1}(held={held_out_signer})",
            classes, config,
            fold_train_paths, fold_train_labels,
            fold_val_paths, fold_val_labels,
            fold_ckpt_dir, device,
        )

        if fold_result:
            fold_result["held_out_signer"] = held_out_signer
            fold_results.append(fold_result)
            print(f"[{ts()}] Fold {fold_idx + 1} result: {fold_result['overall']:.1f}%")

    # Summary
    if fold_results:
        accs = [r["overall"] for r in fold_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"\n[{ts()}] LOSO Summary for {name}:")
        for r in fold_results:
            print(f"[{ts()}]   Signer {r['held_out_signer']} held out: {r['overall']:.1f}%")
        print(f"[{ts()}]   Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%")
        return {
            "folds": fold_results,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
        }
    return None


def train_split_from_paths(name, classes, config,
                            train_paths, train_labels,
                            val_paths, val_labels,
                            ckpt_dir, device):
    """
    Train one split from pre-split path lists (for LOSO).
    Similar to train_split but takes paths directly instead of directories.
    v29: Uses KSLGraphNetV29 with EMA support.
    """
    adj = build_adj(config["num_nodes"]).to(device)

    # Create datasets from paths
    train_ds = KSLGraphDatasetFromPaths(train_paths, train_labels, classes, config, aug=True)
    val_ds = KSLGraphDatasetFromPaths(val_paths, val_labels, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples")
        return None

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )

    train_ld = DataLoader(
        train_ds, batch_size=config["batch_size"],
        sampler=train_sampler, num_workers=2, pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        adj=adj,
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
    ).to(device)

    # EMA
    ema = EMA(model, decay=config.get("ema_decay", 0.9998))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}, signers: {num_signers}")

    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    warmup_epochs = config["warmup_epochs"]

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")

    best, best_ema, patience_counter = 0.0, 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    cutmix_prob = config.get("cutmix_prob", 0.3)
    supcon_weight = config["supcon_weight"]

    for ep in range(config["epochs"]):
        model.train()
        tl, tc, tt = 0.0, 0, 0

        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for gcn_data, aux_data, targets, signer_targets in train_ld:
            gcn_data = gcn_data.to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and (mixup_alpha > 0)

            if use_cutmix:
                gcn_mixed, aux_mixed, targets_a, targets_b, lam = temporal_cutmix(
                    gcn_data, aux_data, targets, signer_targets, alpha=cutmix_alpha,
                )
                logits, signer_logits, embedding = model(gcn_mixed, aux_mixed, grl_lambda=grl_lambda)
                cls_loss = (
                    lam * F.cross_entropy(logits, targets_a, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, targets_b, label_smoothing=config["label_smoothing"])
                )
                sc_loss = supcon_loss_fn(embedding, targets_a)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets_a).float() + (1 - lam) * p.eq(targets_b).float()).sum().item()
            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_mixed = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_mixed = lam * aux_data + (1 - lam) * aux_data[perm]
                t_perm = targets[perm]
                logits, signer_logits, embedding = model(gcn_mixed, aux_mixed, grl_lambda=grl_lambda)
                cls_loss = (
                    lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                )
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()
            else:
                logits, signer_logits, embedding = model(gcn_data, aux_data, grl_lambda=grl_lambda)
                cls_loss = F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()
            tl += loss.item()
            tt += targets.size(0)

        if ep >= warmup_epochs:
            scheduler.step()

        # Validation (regular model)
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0

        if va > best:
            best, patience_counter = va, 0
            torch.save({
                "model": model.state_dict(), "val_acc": va, "epoch": ep + 1,
                "classes": classes, "num_nodes": config["num_nodes"],
                "num_signers": num_signers, "aux_dim": aux_dim,
                "version": "v29", "config": config,
            }, best_path)
        else:
            patience_counter += 1

        # Validation (EMA model)
        ema.apply_shadow()
        vc_ema, vt_ema = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt_ema += targets.size(0)
                vc_ema += p.eq(targets).sum().item()

        va_ema = 100.0 * vc_ema / vt_ema if vt_ema > 0 else 0.0

        if va_ema > best_ema:
            best_ema = va_ema
            torch.save({
                "model": model.state_dict(), "val_acc": va_ema, "epoch": ep + 1,
                "classes": classes, "num_nodes": config["num_nodes"],
                "num_signers": num_signers, "aux_dim": aux_dim,
                "version": "v29_ema", "config": config,
            }, best_ema_path)

        ema.restore()

        if patience_counter >= config["patience"]:
            break

        if ep < 5 or (ep + 1) % 20 == 0:
            ta = 100.0 * tc / tt if tt > 0 else 0.0
            print(f"[{ts()}]   Ep {ep+1:3d} | Loss: {tl/max(len(train_ld),1):.4f} | "
                  f"Train: {ta:.1f}% | Val: {va:.1f}% | EMA: {va_ema:.1f}%")

    # Final evaluation with best model (use EMA if it's better)
    if best_ema >= best and os.path.isfile(best_ema_path):
        ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
        print(f"[{ts()}] Using EMA checkpoint (EMA {best_ema:.1f}% >= regular {best:.1f}%)")
    else:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        print(f"[{ts()}] Using regular checkpoint (regular {best:.1f}% > EMA {best_ema:.1f}%)")
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in val_ld:
            logits, _, _ = model(gcn_data.to(device), aux_data.to(device), grl_lambda=0.0)
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0

    return {
        "overall": ov,
        "best_epoch": ckpt["epoch"],
        "params": param_count,
    }


class KSLGraphDatasetFromPaths(Dataset):
    """Dataset that takes pre-split paths directly (for LOSO mode)."""

    def __init__(self, sample_paths, labels, classes, config, aug=False):
        self.samples = sample_paths
        self.labels = labels
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        # Build signer label mapping
        self.signer_to_idx = {}
        self.signer_labels = []
        for path in self.samples:
            signer = extract_signer_id(os.path.basename(path))
            if signer not in self.signer_to_idx:
                self.signer_to_idx[signer] = len(self.signer_to_idx)
            self.signer_labels.append(self.signer_to_idx[signer])

        self.num_signers = len(self.signer_to_idx)
        print(f"[{ts()}]   LOSO dataset: {len(self.samples)} samples, "
              f"{self.num_signers} signers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]

        if d.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                start = idx_pose * 3
                pose[:, pi, :] = d[:, start:start + 3]
            lh = d[:, 99:162].reshape(f, 21, 3)
            rh = d[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dropout_rate = np.random.uniform(
                self.config["hand_dropout_min"], self.config["hand_dropout_max"])
            if np.random.random() < 0.6:
                lh_mask = np.random.random(f) > dropout_rate
                h[~lh_mask, :21, :] = 0
            if np.random.random() < 0.6:
                rh_mask = np.random.random(f) > dropout_rate
                h[~rh_mask, 21:42, :] = 0

        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        hand_body_feats = compute_hand_body_features(h)
        h = normalize_wrist_palm(h)

        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(h, LH_CHAINS + RH_CHAINS,
                                                  scale_range=self.config["bone_perturb_range"])
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])
        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, max_shear=self.config["shear_max"])
        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, dropout_rate=self.config["joint_dropout_rate"])
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]
        bones = compute_bones(h)
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                sigma=self.config["temporal_warp_sigma"])

        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]; velocity = velocity[indices]; bones = bones[indices]
            joint_angles = joint_angles[indices]; fingertip_dists = fingertip_dists[indices]
            hand_body_feats = hand_body_feats[indices]
        else:
            pad_len = self.mf - f
            h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
            fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
            hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z3 = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z3, h[:-shift]], axis=0)
                velocity = np.concatenate([z3, velocity[:-shift]], axis=0)
                bones = np.concatenate([z3, bones[:-shift]], axis=0)
                za = np.zeros((shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([za, joint_angles[:-shift]], axis=0)
                fingertip_dists = np.concatenate([zd, fingertip_dists[:-shift]], axis=0)
                hand_body_feats = np.concatenate([zh, hand_body_feats[:-shift]], axis=0)
            elif shift < 0:
                z3 = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z3], axis=0)
                velocity = np.concatenate([velocity[-shift:], z3], axis=0)
                bones = np.concatenate([bones[-shift:], z3], axis=0)
                za = np.zeros((-shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((-shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((-shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([joint_angles[-shift:], za], axis=0)
                fingertip_dists = np.concatenate([fingertip_dists[-shift:], zd], axis=0)
                hand_body_feats = np.concatenate([hand_body_feats[-shift:], zh], axis=0)

        gcn_features = np.concatenate([h, velocity, bones], axis=2)
        gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)
        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)

        gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)
        aux_tensor = torch.FloatTensor(aux_features)

        return gcn_tensor, aux_tensor, self.labels[idx], self.signer_labels[idx]


# ---------------------------------------------------------------------------
# Test Set Evaluation (held-out signer evaluation)
# ---------------------------------------------------------------------------

def evaluate_test_set(name, classes, config, test_dir, ckpt_path, device):
    """
    Evaluate a trained model on the held-out test set (ksl-alpha signers 13-15).
    Returns accuracy dict or None if test dir doesn't exist.
    """
    if not os.path.isdir(test_dir):
        print(f"[{ts()}] Test dir not found: {test_dir}, skipping test evaluation for {name}")
        return None

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Test Set Evaluation: {name} ({len(classes)} classes) - v29")
    print(f"[{ts()}] Test dir: {test_dir}")
    print(f"[{ts()}] {'=' * 70}")

    # Load test data (no augmentation)
    test_ds = KSLGraphDataset(test_dir, classes, config, aug=False)
    if len(test_ds) == 0:
        print(f"[{ts()}] WARNING: No test samples found in {test_dir}")
        return None

    test_ld = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load trained model
    if not os.path.isfile(ckpt_path):
        print(f"[{ts()}] WARNING: Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    adj = build_adj(config["num_nodes"]).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    num_signers = ckpt.get("num_signers", test_ds.num_signers)

    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        adj=adj,
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {v: k for k, v in c2i.items()}

    preds, tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in test_ld:
            logits, _, _ = model(
                gcn_data.to(device), aux_data.to(device), grl_lambda=0.0
            )
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    # Per-class results
    print(f"\n[{ts()}] {name} Test Set Per-Class Results:")
    per_class = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        per_class[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {per_class[cn]:5.1f}% ({cor}/{tot_cls})")

    overall = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0
    print(f"[{ts()}] {name} Test Overall: {overall:.1f}% ({sum(1 for t,p in zip(tgts,preds) if t==p)}/{len(tgts)})")

    # Per-signer accuracy
    signer_preds = defaultdict(list)
    signer_tgts = defaultdict(list)
    for idx, path in enumerate(test_ds.samples):
        signer = extract_signer_id(os.path.basename(path))
        if idx < len(preds):
            signer_preds[signer].append(preds[idx])
            signer_tgts[signer].append(tgts[idx])

    print(f"[{ts()}] {name} Test Per-Signer Results:")
    per_signer = {}
    for signer in sorted(signer_preds.keys()):
        sp = signer_preds[signer]
        st = signer_tgts[signer]
        acc = 100.0 * sum(1 for t, p in zip(st, sp) if t == p) / len(st) if len(st) > 0 else 0.0
        per_signer[signer] = acc
        print(f"[{ts()}]   Signer {signer}: {acc:.1f}% ({sum(1 for t,p in zip(st,sp) if t==p)}/{len(st)})")

    # Confusion matrix
    print(f"\n[{ts()}] {name} Test Confusion Matrix:")
    nc = len(classes)
    cm = [[0] * nc for _ in range(nc)]
    for t, p in zip(tgts, preds):
        cm[t][p] += 1

    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"[{ts()}] {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{cm[i][j]:5d}" for j in range(nc))
        print(f"[{ts()}] {row_str}")

    return {
        "overall": overall,
        "per_class": per_class,
        "per_signer": per_signer,
        "num_test_samples": len(tgts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL Training v29 - Deeper+Wider+EMA+FixedEpochs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="both",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loso", action="store_true",
                        help="Run leave-one-signer-out cross-validation instead of normal training")

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"),
                        help="Held-out test set directory (ksl-alpha signers 13-15)")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    args = parser.parse_args()

    set_seed(args.seed)

    mode_str = "LOSO" if args.loso else "Normal"
    print("=" * 70)
    print(f"KSL Training v29 - Deeper+Wider+EMA+FixedEpochs [{mode_str}]")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # v29: Train on 12 signers (train_alpha + val_alpha), validate on 3 (test_alpha)
    train_dirs = [args.train_dir, args.val_dir]  # signers 1-12
    val_dir = args.test_dir                       # signers 13-15

    print(f"\nv29 Data Split:")
    print(f"  Train dirs: {train_dirs}  (signers 1-12, ~895 samples)")
    print(f"  Val dir:    {val_dir}  (signers 13-15, ~225 samples)")
    print(f"  Ckpt dir:   {args.checkpoint_dir}")

    print(f"\n[{ts()}] v29 Config:")
    print(json.dumps(V29_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    if args.loso:
        # LOSO cross-validation mode (uses original train/val dirs)
        if args.model_type in ("numbers", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_loso_numbers")
            results["numbers_loso"] = run_loso(
                "Numbers", NUMBER_CLASSES, V29_CONFIG,
                args.train_dir, args.val_dir, ckpt_dir, args.results_dir, device,
            )

        if args.model_type in ("words", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_loso_words")
            results["words_loso"] = run_loso(
                "Words", WORD_CLASSES, V29_CONFIG,
                args.train_dir, args.val_dir, ckpt_dir, args.results_dir, device,
            )
    else:
        # Normal training mode — v29: 12 signers train, 3 val, no focal loss
        if args.model_type in ("numbers", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_numbers")
            results["numbers"] = train_split(
                "Numbers", NUMBER_CLASSES, V29_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
            )
            # Evaluate on test set after training
            if results["numbers"]:
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")
                # Prefer EMA checkpoint if it exists
                eval_path = best_ema_path if os.path.isfile(best_ema_path) else best_path
                results["numbers_test"] = evaluate_test_set(
                    "Numbers", NUMBER_CLASSES, V29_CONFIG,
                    args.test_dir, eval_path, device,
                )

        if args.model_type in ("words", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_words")
            results["words"] = train_split(
                "Words", WORD_CLASSES, V29_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
            )
            # Evaluate on test set after training
            if results["words"]:
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")
                # Prefer EMA checkpoint if it exists
                eval_path = best_ema_path if os.path.isfile(best_ema_path) else best_path
                results["words_test"] = evaluate_test_set(
                    "Words", WORD_CLASSES, V29_CONFIG,
                    args.test_dir, eval_path, device,
                )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v29 [{mode_str}]")
    print(f"[{ts()}] {'=' * 70}")

    if args.loso:
        if results.get("numbers_loso"):
            r = results["numbers_loso"]
            print(f"[{ts()}] Numbers LOSO: {r['mean_accuracy']:.1f}% +/- {r['std_accuracy']:.1f}%")
        if results.get("words_loso"):
            r = results["words_loso"]
            print(f"[{ts()}] Words LOSO:   {r['mean_accuracy']:.1f}% +/- {r['std_accuracy']:.1f}%")
        if results.get("numbers_loso") and results.get("words_loso"):
            combined = (results["numbers_loso"]["mean_accuracy"] + results["words_loso"]["mean_accuracy"]) / 2
            print(f"[{ts()}] Combined LOSO: {combined:.1f}%")
    else:
        if results.get("numbers"):
            r = results["numbers"]
            print(f"[{ts()}] Numbers: {r['overall']:.1f}% "
                  f"(best epoch {r['best_epoch']}, params {r['params']:,})")
        if results.get("words"):
            r = results["words"]
            print(f"[{ts()}] Words:   {r['overall']:.1f}% "
                  f"(best epoch {r['best_epoch']}, params {r['params']:,})")
        if results.get("numbers") and results.get("words"):
            combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
            print(f"[{ts()}] Combined: {combined:.1f}%")
        # Test set results
        if results.get("numbers_test"):
            r = results["numbers_test"]
            print(f"[{ts()}] Numbers Test: {r['overall']:.1f}% ({r['num_test_samples']} samples)")
        if results.get("words_test"):
            r = results["words_test"]
            print(f"[{ts()}] Words Test:   {r['overall']:.1f}% ({r['num_test_samples']} samples)")
        if results.get("numbers_test") and results.get("words_test"):
            combined_test = (results["numbers_test"]["overall"] + results["words_test"]["overall"]) / 2
            print(f"[{ts()}] Combined Test: {combined_test:.1f}%")

    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    loso_tag = "_loso" if args.loso else ""
    results_path = os.path.join(
        args.results_dir,
        f"v29{loso_tag}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v29",
        "mode": mode_str,
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V29_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v27": [
            "1. 8 GCN LAYERS (up from 4): Channel progression ic->64->64->128->...->128.",
            "2. DILATED MULTI-SCALE TCN: kernel=3 with dilations=(1,2,4) replaces kernels=(3,5,7).",
            "3. 4 ATTENTION HEADS (up from 2): gcn_embed_dim = 4 * 128 = 512.",
            "4. WIDER AUX BRANCH: 128-dim output (was 64-dim).",
            "5. WIDER CLASSIFIER: 640->128->nc (was 320->64->nc).",
            "6. 300 FIXED EPOCHS with EMA, no early stopping.",
            "7. REDUCED REGULARIZATION: dropout 0.2, GRL 0.15, CutMix 0.15.",
            "8. EMA (decay=0.9998): Saves both regular and EMA checkpoints.",
            "9. AdamW with weight_decay=0.05.",
            "10. NO FOCAL LOSS for either numbers or words.",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
