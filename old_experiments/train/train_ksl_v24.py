#!/usr/bin/env python3
"""
KSL Training v24 (Alpine) - Prototypical Classification + MixStyle + LOSO

Changes from v22:
  1. MIXSTYLE: Feature-level signer style mixing after GCN blocks 0 and 1
  2. PROTOTYPICAL INFERENCE: After training, compute class prototypes for
     cosine-similarity-based classification (replaces FC head at inference)
  3. LOSO DIAGNOSTIC: Optional leave-one-signer-out cross-validation
  4. WIDER AUGMENTATION: bone_perturb (0.7-1.3), hand_size (0.7-1.3)
  5. SPEED WARP: Uniform temporal speed perturbation (0.8-1.2x)
  6. DISTANCE CONFIDENCE: Prototype-based confidence replaces softmax

Training uses CE+SupCon+GRL (same as v22) for stability.
Prototypical classification is applied at inference only.

Usage:
    python train_ksl_v24.py --model-type both
    python train_ksl_v24.py --model-type both --loso
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
    try:
        import mediapipe as mp
        version = mp.__version__
        if version != "0.10.14":
            print(f"[{ts()}] WARNING: MediaPipe version is {version}, "
                  f"training data was processed with 0.10.14.")
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

RH_CHAINS = [
    [(p + 21, c + 21) for p, c in chain]
    for chain in LH_CHAINS
]

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
# Config (v24)
# ---------------------------------------------------------------------------

V24_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,
    "hidden_dim": 64,
    "num_layers": 4,
    "temporal_kernels": [3, 5, 7],
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "spatial_dropout": 0.1,
    "label_smoothing": 0.1,
    "patience": 40,
    "warmup_epochs": 10,
    # SupCon
    "supcon_weight": 0.1,
    "supcon_temperature": 0.07,
    # GRL
    "grl_lambda_max": 0.3,
    "grl_start_epoch": 0,
    "grl_ramp_epochs": 100,
    # Augmentation (WIDENED from v22)
    "rotation_prob": 0.5,
    "rotation_max_deg": 15.0,
    "shear_prob": 0.3,
    "shear_max": 0.2,
    "joint_dropout_prob": 0.2,
    "joint_dropout_rate": 0.08,
    "noise_std": 0.015,
    "noise_prob": 0.4,
    "bone_perturb_prob": 0.5,
    "bone_perturb_range": (0.7, 1.3),       # v22: (0.8, 1.2)
    "hand_size_prob": 0.5,
    "hand_size_range": (0.7, 1.3),           # v22: (0.8, 1.2)
    "temporal_warp_prob": 0.4,
    "temporal_warp_sigma": 0.2,
    "hand_dropout_prob": 0.2,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.3,
    "complete_hand_drop_prob": 0.03,
    "mixup_alpha": 0.2,
    # NEW: Speed warp (uniform temporal)
    "speed_warp_prob": 0.3,
    "speed_warp_range": (0.8, 1.2),
    # NEW: MixStyle
    "mixstyle_p": 0.5,
    "mixstyle_alpha": 0.1,
    "mixstyle_layers": [0, 1],
    # Prototypical inference
    "proto_temperature": 0.1,
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
# Data Deduplication
# ---------------------------------------------------------------------------

def get_file_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def extract_signer_id(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[1]
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
# Normalization
# ---------------------------------------------------------------------------

def normalize_wrist_palm(h):
    T = h.shape[0]
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
# Augmentation Suite (v24 - widened from v22)
# ---------------------------------------------------------------------------

def augment_bone_length_perturbation(h, chains, scale_range=(0.7, 1.3)):
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


def augment_hand_size(h, scale_range=(0.7, 1.3)):
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


def augment_speed_warp(data_list, speed_range=(0.8, 1.2)):
    """Uniform temporal speed perturbation. Resample all arrays to new length."""
    speed = np.random.uniform(speed_range[0], speed_range[1])
    T = data_list[0].shape[0]
    new_len = max(2, int(T * speed))
    if new_len == T:
        return data_list
    indices = np.linspace(0, T - 1, new_len, dtype=int)
    return [arr[indices] for arr in data_list]


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
# Dataset (v24 - supports file lists for LOSO + new augmentations)
# ---------------------------------------------------------------------------

class KSLGraphDataset(Dataset):
    """
    V24 dataset. Supports two modes:
    - Directory mode: data_dir + classes (like v22)
    - File-list mode: file_list of (path, class_name) tuples (for LOSO)
    """

    def __init__(self, data_dir, classes, config, aug=False, file_list=None):
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        if file_list is not None:
            # File-list mode (LOSO)
            self.samples = [f for f, c in file_list]
            self.labels = [self.c2i[c] for f, c in file_list]
        else:
            # Directory mode (standard)
            all_paths = []
            all_labels = []
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
                    print(f"[{ts()}]   Dedup: {total_before} -> "
                          f"{len(unique_paths)} ({len(removed)} removed)")
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

        # Build signer labels
        self.signer_to_idx = {}
        self.signer_labels = []
        for path in self.samples:
            signer = extract_signer_id(os.path.basename(path))
            if signer not in self.signer_to_idx:
                self.signer_to_idx[signer] = len(self.signer_to_idx)
            self.signer_labels.append(self.signer_to_idx[signer])
        self.num_signers = len(self.signer_to_idx)

        print(f"[{ts()}]   Loaded {len(self.samples)} samples, "
              f"{self.num_signers} signers "
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

        # --- Hand dropout ---
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

        # --- Complete hand drop ---
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # --- Hand swap ---
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # --- Normalization ---
        h = normalize_wrist_palm(h)

        # --- Bone-length perturbation (WIDENED range) ---
        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(
                h, LH_CHAINS + RH_CHAINS,
                scale_range=self.config["bone_perturb_range"]
            )

        # --- Hand-size randomization (WIDENED range) ---
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])

        # --- Rotation ---
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])

        # --- Shear ---
        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, max_shear=self.config["shear_max"])

        # --- Joint dropout ---
        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, dropout_rate=self.config["joint_dropout_rate"])

        # --- Scale jitter ---
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)

        # --- Spatial jitter ---
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        # --- Compute velocity ---
        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]

        # --- Compute bone features ---
        bones = compute_bones(h)

        # --- Compute auxiliary features ---
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        # --- Temporal warping (non-uniform) ---
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists],
                sigma=self.config["temporal_warp_sigma"]
            )

        # --- v24 NEW: Speed warp (uniform) ---
        if self.aug and np.random.random() < self.config.get("speed_warp_prob", 0):
            h, velocity, bones, joint_angles, fingertip_dists = augment_speed_warp(
                [h, velocity, bones, joint_angles, fingertip_dists],
                speed_range=self.config["speed_warp_range"]
            )

        # --- Temporal sampling / padding ---
        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
            velocity = velocity[indices]
            bones = bones[indices]
            joint_angles = joint_angles[indices]
            fingertip_dists = fingertip_dists[indices]
        else:
            pad_len = self.mf - f
            h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
            fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])

        # --- Temporal shift ---
        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z3 = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z3, h[:-shift]], axis=0)
                velocity = np.concatenate([z3, velocity[:-shift]], axis=0)
                bones = np.concatenate([z3, bones[:-shift]], axis=0)
                za = np.zeros((shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                joint_angles = np.concatenate([za, joint_angles[:-shift]], axis=0)
                fingertip_dists = np.concatenate([zd, fingertip_dists[:-shift]], axis=0)
            elif shift < 0:
                z3 = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z3], axis=0)
                velocity = np.concatenate([velocity[-shift:], z3], axis=0)
                bones = np.concatenate([bones[-shift:], z3], axis=0)
                za = np.zeros((-shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((-shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                joint_angles = np.concatenate([joint_angles[-shift:], za], axis=0)
                fingertip_dists = np.concatenate([fingertip_dists[-shift:], zd], axis=0)

        # --- Build tensors ---
        gcn_features = np.concatenate([h, velocity, bones], axis=2)
        gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)
        aux_features = np.concatenate([joint_angles, fingertip_dists], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)

        gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)
        aux_tensor = torch.FloatTensor(aux_features)

        return gcn_tensor, aux_tensor, self.labels[idx], self.signer_labels[idx]


# ---------------------------------------------------------------------------
# Multi-Scale Temporal Convolution
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module):
    def __init__(self, channels, kernels=(3, 5, 7), stride=1, dropout=0.3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )
            for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, in_channels, num_heads=2):
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
        x_t = x.permute(0, 2, 1)
        outs = []
        for head in self.heads:
            attn_logits = head(x_t)
            attn_weights = F.softmax(attn_logits, dim=1)
            pooled = (x_t * attn_weights).sum(dim=1)
            outs.append(pooled)
        return torch.cat(outs, dim=1)


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
# v24 NEW: MixStyle (Zhou et al., ICLR 2021)
# ---------------------------------------------------------------------------

class MixStyle(nn.Module):
    """Mix feature statistics across samples for domain generalization."""

    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B <= 1:
            return x

        # Compute per-sample, per-channel statistics: x is (B, C, T, N)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + 1e-6).sqrt()
        x_normed = (x - mu) / sigma

        # Shuffle to get mixing partners
        perm = torch.randperm(B)
        mu2, sigma2 = mu[perm], sigma[perm]

        # Mix statistics with Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)
        ).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sigma_mix = lam * sigma + (1 - lam) * sigma2

        return x_normed * sigma_mix + mu_mix


# ---------------------------------------------------------------------------
# Model (v24: v22 backbone + MixStyle + prototypical inference)
# ---------------------------------------------------------------------------

class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, ic, oc, adj, temporal_kernels=(3, 5, 7), st=1, dr=0.3,
                 spatial_dropout=0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, st, dr)
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


class KSLGraphNetV24(nn.Module):
    """
    V24: V22 backbone + MixStyle + prototypical inference support.

    Training uses standard FC classifier + CE loss (proven stable).
    After training, prototypes are computed and used for inference.
    """

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=9, hd=64, nl=4,
                 tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=None,
                 mixstyle_p=0.5, mixstyle_alpha=0.1, mixstyle_layers=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk,
                        2 if i in [1, 3] else 1,
                        dr, spatial_dropout)
             for i in range(nl)]
        )

        # v24: MixStyle modules
        self.mixstyle_layers = set(mixstyle_layers or [0, 1])
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        final_ch = ch[-1]
        self.attn_pool = AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch

        # Auxiliary branch (same as v22)
        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(128, 64),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.embed_dim = gcn_embed_dim + 64

        # FC classifier (used during training)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

        # Signer adversarial head
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_signers),
        )

        # Prototypical inference (populated after training)
        self.register_buffer(
            "prototypes",
            torch.zeros(nc, self.embed_dim),
        )
        self.proto_temperature = nn.Parameter(torch.tensor(0.1))

    def get_embedding(self, gcn_input, aux_input):
        """Extract the 320-dim embedding without classification."""
        b, c, t, n = gcn_input.shape

        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.mixstyle_layers:
                x = self.mixstyle(x)

        x_node_avg = x.mean(dim=3)
        gcn_embedding = self.attn_pool(x_node_avg)

        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)
        aux = self.aux_mlp(aux)
        aux_t = aux.permute(0, 2, 1)
        aux_t = self.aux_temporal_conv(aux_t)
        aux = aux_t.permute(0, 2, 1)
        attn_logits = self.aux_attn(aux)
        attn_weights = F.softmax(attn_logits, dim=1)
        aux_embedding = (aux * attn_weights).sum(dim=1)

        return torch.cat([gcn_embedding, aux_embedding], dim=1)

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        embedding = self.get_embedding(gcn_input, aux_input)

        # FC classification (training)
        logits = self.classifier(embedding)

        # Signer adversarial
        reversed_embedding = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_embedding)

        return logits, signer_logits, embedding

    def proto_classify(self, embedding):
        """Classify via cosine similarity to prototypes (inference only)."""
        emb_norm = F.normalize(embedding, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        logits = torch.mm(emb_norm, proto_norm.t()) / self.proto_temperature.clamp(min=0.01)
        return logits

    def proto_confidence(self, embedding):
        """Compute distance-based confidence: margin between top-1 and top-2."""
        emb_norm = F.normalize(embedding, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        cosine_sims = torch.mm(emb_norm, proto_norm.t())
        top2 = cosine_sims.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        confidence = torch.sigmoid(margin * 5.0)
        return confidence


# ---------------------------------------------------------------------------
# Prototype Computation
# ---------------------------------------------------------------------------

def compute_prototypes(model, dataloader, device, num_classes):
    """Compute per-class mean embeddings (prototypes) from a dataset."""
    model.eval()
    class_embeddings = defaultdict(list)

    with torch.no_grad():
        for gcn, aux, labels, _ in dataloader:
            gcn, aux = gcn.to(device), aux.to(device)
            embedding = model.get_embedding(gcn, aux)
            for i in range(len(labels)):
                class_embeddings[labels[i].item()].append(embedding[i].cpu())

    prototypes = torch.zeros(num_classes, model.embed_dim)
    for cls_idx, embs in class_embeddings.items():
        prototypes[cls_idx] = torch.stack(embs).mean(dim=0)

    print(f"[{ts()}]   Computed prototypes for {len(class_embeddings)} classes "
          f"(avg {np.mean([len(v) for v in class_embeddings.values()]):.0f} samples/class)")

    return prototypes


# ---------------------------------------------------------------------------
# LOSO Helpers
# ---------------------------------------------------------------------------

def collect_all_samples(data_dirs, classes):
    """Collect all .npy files from multiple directories."""
    all_files = []
    seen = set()  # deduplicate by filename
    for data_dir in data_dirs:
        for cn in classes:
            cd = os.path.join(data_dir, cn)
            if not os.path.exists(cd):
                continue
            for fn in sorted(os.listdir(cd)):
                if fn.endswith(".npy"):
                    filepath = os.path.join(cd, fn)
                    signer = extract_signer_id(fn)
                    key = (cn, fn)  # unique by class + filename
                    if key not in seen:
                        seen.add(key)
                        all_files.append((filepath, cn, signer))
    return all_files


def get_loso_splits(all_files):
    """Generate leave-one-signer-out splits."""
    signers = sorted(set(s for _, _, s in all_files))
    for held_out in signers:
        train_files = [(f, c) for f, c, s in all_files if s != held_out]
        val_files = [(f, c) for f, c, s in all_files if s == held_out]
        yield held_out, train_files, val_files


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device,
                train_file_list=None, val_file_list=None):
    """Train one split with v24 improvements."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v24")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLGraphDataset(
        train_dir, classes, config, aug=True, file_list=train_file_list
    )
    val_ds = KSLGraphDataset(
        val_dir, classes, config, aug=False, file_list=val_file_list
    )

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS

    print(f"[{ts()}] Aux features: {aux_dim} total")
    print(f"[{ts()}] Num signers in training: {num_signers}")

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

    # Also create a non-augmented train loader for prototype computation
    train_ds_noaug = KSLGraphDataset(
        train_dir, classes, config, aug=False, file_list=train_file_list
    )
    train_ld_noaug = DataLoader(
        train_ds_noaug, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    model = KSLGraphNetV24(
        nc=len(classes),
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

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] v24 changes: MixStyle(p={config.get('mixstyle_p', 0.5)}, "
          f"layers={config.get('mixstyle_layers', [0, 1])}), "
          f"bone_perturb={config['bone_perturb_range']}, "
          f"hand_size={config['hand_size_range']}, "
          f"speed_warp={config.get('speed_warp_range', 'off')}")

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

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    supcon_weight = config["supcon_weight"]

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        signer_correct, signer_total = 0, 0

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

            if mixup_alpha > 0:
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

        # Validation (FC-based during training)
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
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        signer_acc = 100.0 * signer_correct / signer_total if signer_total > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
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
                    "version": "v24",
                    "config": config,
                },
                best_path,
            )
            print(f"[{ts()}]   -> New best! {va:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"[{ts()}] Early stopping at epoch {ep + 1}")
            break

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

    # --- Load best model and compute prototypes ---
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n[{ts()}] Computing prototypes from training data...")
    prototypes = compute_prototypes(model, train_ld_noaug, device, len(classes))
    model.prototypes.copy_(prototypes.to(device))

    # Save prototypes in checkpoint
    ckpt["prototypes"] = prototypes
    ckpt["model"] = model.state_dict()  # includes prototypes buffer
    torch.save(ckpt, best_path)
    print(f"[{ts()}] Prototypes saved to checkpoint")

    # --- Final evaluation (FC + prototypical) ---
    preds_fc, preds_proto, tgts = [], [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in val_ld:
            gcn_data = gcn_data.to(device)
            aux_data = aux_data.to(device)
            targets_cpu = targets.numpy()

            # FC prediction
            logits_fc, _, embedding = model(gcn_data, aux_data, grl_lambda=0.0)
            _, p_fc = logits_fc.max(1)
            preds_fc.extend(p_fc.cpu().numpy())

            # Prototypical prediction
            logits_proto = model.proto_classify(embedding)
            _, p_proto = logits_proto.max(1)
            preds_proto.extend(p_proto.cpu().numpy())

            tgts.extend(targets_cpu)

    fc_acc = 100.0 * sum(1 for t, p in zip(tgts, preds_fc) if t == p) / len(tgts)
    proto_acc = 100.0 * sum(1 for t, p in zip(tgts, preds_proto) if t == p) / len(tgts)

    print(f"\n[{ts()}] {name} Final Results (best epoch {ckpt['epoch']}):")
    print(f"[{ts()}]   FC accuracy:    {fc_acc:.1f}%")
    print(f"[{ts()}]   Proto accuracy: {proto_acc:.1f}%")

    # Per-class breakdown (prototypical)
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds_proto) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_cls})")

    print(f"[{ts()}] {name} Overall (proto): {proto_acc:.1f}%")

    return {
        "overall_fc": fc_acc,
        "overall_proto": proto_acc,
        "per_class": res,
        "best_epoch": ckpt["epoch"],
        "params": param_count,
    }


# ---------------------------------------------------------------------------
# LOSO Cross-Validation
# ---------------------------------------------------------------------------

def train_loso(name, classes, config, train_dir, val_dir, ckpt_dir, device):
    """Run leave-one-signer-out cross-validation."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] LOSO Cross-Validation: {name} ({len(classes)} classes)")
    print(f"[{ts()}] {'=' * 70}")

    all_files = collect_all_samples([train_dir, val_dir], classes)
    signers = sorted(set(s for _, _, s in all_files))
    print(f"[{ts()}] Total samples: {len(all_files)}")
    print(f"[{ts()}] Signers: {signers}")

    signer_counts = Counter(s for _, _, s in all_files)
    for s, c in sorted(signer_counts.items()):
        print(f"[{ts()}]   Signer {s}: {c} samples")

    fold_results = []

    for fold_idx, (held_out, train_files, val_files) in enumerate(get_loso_splits(all_files)):
        print(f"\n[{ts()}] --- LOSO Fold {fold_idx + 1}/4: Hold out signer {held_out} ---")
        print(f"[{ts()}]   Train: {len(train_files)} samples, Val: {len(val_files)} samples")

        fold_ckpt_dir = os.path.join(ckpt_dir, f"v24_loso_{name.lower()}_fold{fold_idx}")

        # Use shorter patience for LOSO folds
        loso_config = dict(config)
        loso_config["patience"] = 30
        loso_config["epochs"] = 150

        result = train_split(
            f"{name} (fold {fold_idx + 1}, hold-out={held_out})",
            classes, loso_config,
            train_dir, val_dir, fold_ckpt_dir, device,
            train_file_list=train_files,
            val_file_list=val_files,
        )

        if result:
            fold_results.append({
                "fold": fold_idx + 1,
                "held_out_signer": held_out,
                **result,
            })

    # Aggregate LOSO results
    if fold_results:
        avg_fc = np.mean([r["overall_fc"] for r in fold_results])
        avg_proto = np.mean([r["overall_proto"] for r in fold_results])
        avg_epoch = np.mean([r["best_epoch"] for r in fold_results])

        print(f"\n[{ts()}] {'=' * 70}")
        print(f"[{ts()}] LOSO Summary: {name}")
        print(f"[{ts()}] {'=' * 70}")
        for r in fold_results:
            print(f"[{ts()}]   Fold {r['fold']} (signer {r['held_out_signer']}): "
                  f"FC={r['overall_fc']:.1f}%, Proto={r['overall_proto']:.1f}%, "
                  f"Epoch={r['best_epoch']}")
        print(f"[{ts()}]   Average FC:    {avg_fc:.1f}%")
        print(f"[{ts()}]   Average Proto: {avg_proto:.1f}%")
        print(f"[{ts()}]   Average Epoch: {avg_epoch:.0f}")

        return {
            "folds": fold_results,
            "avg_fc": avg_fc,
            "avg_proto": avg_proto,
            "avg_epoch": avg_epoch,
        }
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL ST-GCN Training v24 (Alpine HPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="both",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loso", action="store_true",
                        help="Run LOSO cross-validation (diagnostic)")

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_v2"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_v2"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("KSL Training v24 - Prototypical + MixStyle + Wider Aug")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print(f"LOSO: {'enabled' if args.loso else 'disabled'}")
    print("=" * 70)

    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nTrain dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Ckpt dir:  {args.checkpoint_dir}")

    print(f"\n[{ts()}] v24 Config:")
    print(json.dumps(V24_CONFIG, indent=2, default=str))

    results = {}
    loso_results = {}
    start_time = time.time()

    # --- Optional LOSO ---
    if args.loso:
        if args.model_type in ("numbers", "both"):
            loso_results["numbers"] = train_loso(
                "Numbers", NUMBER_CLASSES, V24_CONFIG,
                args.train_dir, args.val_dir,
                args.checkpoint_dir, device,
            )
        if args.model_type in ("words", "both"):
            loso_results["words"] = train_loso(
                "Words", WORD_CLASSES, V24_CONFIG,
                args.train_dir, args.val_dir,
                args.checkpoint_dir, device,
            )

    # --- Standard training (train on train_v2, validate on val_v2) ---
    if args.model_type in ("numbers", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v24_numbers")
        results["numbers"] = train_split(
            "Numbers", NUMBER_CLASSES, V24_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    if args.model_type in ("words", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v24_words")
        results["words"] = train_split(
            "Words", WORD_CLASSES, V24_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    total_time = time.time() - start_time

    # --- Summary ---
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v24")
    print(f"[{ts()}] {'=' * 70}")

    if results.get("numbers"):
        r = results["numbers"]
        print(f"[{ts()}] Numbers: FC={r['overall_fc']:.1f}%, Proto={r['overall_proto']:.1f}% "
              f"(epoch {r['best_epoch']}, params {r['params']:,})")
    if results.get("words"):
        r = results["words"]
        print(f"[{ts()}] Words:   FC={r['overall_fc']:.1f}%, Proto={r['overall_proto']:.1f}% "
              f"(epoch {r['best_epoch']}, params {r['params']:,})")
    if results.get("numbers") and results.get("words"):
        combined_fc = (results["numbers"]["overall_fc"] + results["words"]["overall_fc"]) / 2
        combined_proto = (results["numbers"]["overall_proto"] + results["words"]["overall_proto"]) / 2
        print(f"[{ts()}] Combined: FC={combined_fc:.1f}%, Proto={combined_proto:.1f}%")

    if loso_results:
        print(f"\n[{ts()}] LOSO Results:")
        for cat, lr in loso_results.items():
            if lr:
                print(f"[{ts()}]   {cat}: FC={lr['avg_fc']:.1f}%, Proto={lr['avg_proto']:.1f}%")

    print(f"[{ts()}] Total time: {total_time / 60:.1f} minutes")

    # --- Save results ---
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v24_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v24",
        "model_type": args.model_type,
        "seed": args.seed,
        "loso_enabled": args.loso,
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in V24_CONFIG.items()},
        "results": results,
        "loso_results": loso_results if loso_results else None,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v22": [
            "1. MIXSTYLE: Feature-level signer style mixing after GCN blocks 0,1 (p=0.5, alpha=0.1)",
            "2. PROTOTYPICAL INFERENCE: Cosine similarity to class prototypes (replaces FC at inference)",
            "3. WIDER AUGMENTATION: bone_perturb (0.7-1.3 vs 0.8-1.2), hand_size (0.7-1.3 vs 0.8-1.2)",
            "4. SPEED WARP: Uniform temporal speed perturbation (0.8-1.2x, p=0.3)",
            "5. DISTANCE CONFIDENCE: Prototype margin-based confidence (replaces softmax)",
            "6. LOSO DIAGNOSTIC: Optional leave-one-signer-out cross-validation",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")
    print(f"[{ts()}] Done.")

    return results


if __name__ == "__main__":
    main()
