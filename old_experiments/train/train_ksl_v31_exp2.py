#!/usr/bin/env python3
"""
KSL Training v31 Exp2 - Prototypical Networks on v28 Architecture

Based on v28 (multi-stream late fusion), but replaces standard cross-entropy
training with prototypical network training (episodic learning).

Key changes from v28:
  1. PROTOTYPICAL LOSS: Train with N-way K-shot episodes. Each episode samples
     N classes with K_support support samples and K_query query samples per class.
     Compute class prototypes as mean of support embeddings, classify query samples
     by negative Euclidean distance to prototypes.
  2. EPISODIC SAMPLER: Custom sampler that yields batches with exactly N classes
     and K_support + K_query samples per class.
  3. EVALUATION: Use all training data as support set to compute prototypes,
     classify val samples by nearest prototype (standard prototypical eval).

Reference: "Prototypical Networks for Few-shot Learning" (Snell et al., NeurIPS 2017)
           "Data-Efficient ASL Recognition via Prototypical Networks" (arXiv 2512.10562)

Usage:
    python train_ksl_v31_exp2.py --model-type numbers
    python train_ksl_v31_exp2.py --model-type words
"""

import argparse
import copy
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
                  f"v28 training data was extracted with 0.10.5.")
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
# Graph Topology (from v28)
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
NUM_HAND_BODY_FEATURES = 8

# ---------------------------------------------------------------------------
# Config (v31 exp2 - prototypical networks)
# ---------------------------------------------------------------------------

V31_EXP2_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,           # per-stream
    "streams": ["joint", "bone", "velocity"],
    "hidden_dim": 64,
    "num_layers": 4,
    "temporal_kernels": [3, 5, 7],
    "batch_size": 32,  # Will be overridden by episodic sampler
    "epochs": 500,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "spatial_dropout": 0.1,
    "patience": 80,
    "warmup_epochs": 10,

    # Prototypical network specific
    "n_way": 10,           # Number of classes per episode (10 out of 15 for numbers)
    "k_support": 3,        # Support samples per class
    "k_query": 2,          # Query samples per class

    # Augmentation (same as v28, but lighter for prototypical)
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
# Augmentation Suite (from v28)
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


def compute_hand_body_features(h_raw):
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
# Episodic Sampler for Prototypical Networks
# ---------------------------------------------------------------------------

class EpisodicSampler(Sampler):
    """Sampler that yields N-way K-shot episodes.

    Each batch contains:
    - N_way classes (randomly sampled from all classes)
    - K_support + K_query samples per class
    - Total batch size = N_way * (K_support + K_query)
    """

    def __init__(self, labels, n_way, k_support, k_query, n_episodes):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.n_episodes = n_episodes

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        self.classes = list(self.class_indices.keys())
        self.n_classes = len(self.classes)

        # Validate
        min_samples = k_support + k_query
        for cls in self.classes:
            if len(self.class_indices[cls]) < min_samples:
                print(f"[{ts()}] WARNING: Class {cls} has only "
                      f"{len(self.class_indices[cls])} samples, need >= {min_samples}")

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []

            # Sample N classes
            episode_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )

            # Sample K_support + K_query samples per class
            for cls in episode_classes:
                indices = self.class_indices[cls]
                n_samples = self.k_support + self.k_query

                if len(indices) < n_samples:
                    # If not enough samples, sample with replacement
                    sampled = np.random.choice(indices, n_samples, replace=True)
                else:
                    # Sample without replacement
                    sampled = np.random.choice(indices, n_samples, replace=False)

                batch.extend(sampled.tolist())

            # Yield in random order (will be re-organized by training loop)
            yield batch

    def __len__(self):
        return self.n_episodes


# ---------------------------------------------------------------------------
# Multi-Stream Dataset (from v28)
# ---------------------------------------------------------------------------

class KSLMultiStreamDataset(Dataset):
    """Returns per-stream tensors (joint/bone/velocity) instead of concatenated 9-ch."""

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
                print(f"[{ts()}]   Deduplication: {total_before} -> "
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

        # Augmentation: hand dropout
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

        # Augmentation: complete hand drop
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # Augmentation: hand swap
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # Hand-body features BEFORE normalization
        hand_body_feats = compute_hand_body_features(h)

        # Normalization
        h = normalize_wrist_palm(h)

        # Augmentation: bone-length perturbation
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

        # Compute velocity BEFORE resampling
        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]

        # Compute bone features
        bones = compute_bones(h)

        # Compute auxiliary features
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        # Augmentation: temporal warping
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                sigma=self.config["temporal_warp_sigma"]
            )

        # Temporal sampling / padding
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

        # Augmentation: temporal shift
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

        # --- Build per-stream GCN tensors (3, T, N) ---
        h_clipped = np.clip(h, -10, 10).astype(np.float32)
        bones_clipped = np.clip(bones, -10, 10).astype(np.float32)
        velocity_clipped = np.clip(velocity, -10, 10).astype(np.float32)

        streams = {
            "joint": torch.FloatTensor(h_clipped).permute(2, 0, 1),        # (3, T, 48)
            "bone": torch.FloatTensor(bones_clipped).permute(2, 0, 1),     # (3, T, 48)
            "velocity": torch.FloatTensor(velocity_clipped).permute(2, 0, 1),  # (3, T, 48)
        }

        # Auxiliary features: (T, D_aux)
        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
        aux_tensor = torch.FloatTensor(aux_features)

        return streams, aux_tensor, self.labels[idx], self.signer_labels[idx]


def stream_collate_fn(batch):
    """Custom collate for multi-stream dataset. Stacks per-stream tensors."""
    streams_list, aux_list, labels_list, signer_list = zip(*batch)
    stream_names = list(streams_list[0].keys())
    streams_batch = {
        name: torch.stack([s[name] for s in streams_list])
        for name in stream_names
    }
    aux_batch = torch.stack(aux_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    signers = torch.tensor(signer_list, dtype=torch.long)
    return streams_batch, aux_batch, labels, signers


# ---------------------------------------------------------------------------
# Multi-Scale Temporal Convolution (from v28)
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
# Attention Pooling (from v28)
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
# Spatial Dropout (from v28)
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
# Model Architecture (from v28, but without classifier head)
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


class ProtoNetEncoder(nn.Module):
    """ST-GCN encoder for prototypical networks (no classifier, outputs embeddings)."""

    def __init__(self, aux_dim, nn_=48, ic=3, hd=64, nl=4, tk=(3, 5, 7),
                 dr=0.3, spatial_dropout=0.1, adj=None):
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

        final_ch = ch[-1]
        self.attn_pool = AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch

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

    def forward(self, gcn_input, aux_input):
        """Forward pass that returns embedding vectors."""
        b, c, t, n = gcn_input.shape
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

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

        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)
        return embedding


# ---------------------------------------------------------------------------
# Prototypical Loss
# ---------------------------------------------------------------------------

def prototypical_loss(embeddings, labels, n_way, k_support, k_query):
    """Compute prototypical loss.

    Args:
        embeddings: (B, D) tensor of embeddings
        labels: (B,) tensor of labels
        n_way: number of classes in episode
        k_support: support samples per class
        k_query: query samples per class

    Returns:
        loss: scalar prototypical loss
        acc: accuracy on query set
    """
    B, D = embeddings.shape

    # Reshape embeddings: (n_way, k_support + k_query, D)
    embeddings = embeddings.view(n_way, k_support + k_query, D)

    # Split support and query
    support_embeddings = embeddings[:, :k_support, :]  # (n_way, k_support, D)
    query_embeddings = embeddings[:, k_support:, :]    # (n_way, k_query, D)

    # Compute prototypes (mean of support embeddings per class)
    prototypes = support_embeddings.mean(dim=1)  # (n_way, D)

    # Flatten query embeddings
    query_embeddings = query_embeddings.reshape(n_way * k_query, D)  # (n_way * k_query, D)

    # Compute distances from queries to prototypes (negative Euclidean distance)
    # (n_way * k_query, n_way)
    dists = torch.cdist(query_embeddings, prototypes, p=2)  # Euclidean distance
    logits = -dists  # Negative distance (higher is better)

    # Query labels: [0, 0, ..., 1, 1, ..., n_way-1, n_way-1, ...]
    query_labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(k_query)

    # Cross-entropy loss on distance-based logits
    loss = F.cross_entropy(logits, query_labels)

    # Accuracy
    _, preds = logits.max(1)
    acc = (preds == query_labels).float().mean()

    return loss, acc


# ---------------------------------------------------------------------------
# Train One Stream with Prototypical Networks
# ---------------------------------------------------------------------------

def train_stream_prototypical(stream_name, name, classes, config, train_dir, val_dir,
                               ckpt_dir, device):
    """Train one stream model with prototypical networks.

    Args:
        stream_name: "joint", "bone", or "velocity"
        name: category name ("Numbers" or "Words")
        classes: list of class names
        config: V31_EXP2_CONFIG
        train_dir: list of training data dirs
        val_dir: validation data dir
        ckpt_dir: base checkpoint dir (stream subdir will be created)
        device: torch device
    """
    stream_ckpt_dir = os.path.join(ckpt_dir, stream_name)

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} - Stream: {stream_name} ({len(classes)} classes)")
    print(f"[{ts()}] Method: Prototypical Networks")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLMultiStreamDataset(train_dir, classes, config, aug=True)
    val_ds = KSLMultiStreamDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    print(f"[{ts()}] Aux dim: {aux_dim}")

    # Episodic parameters
    n_way = min(config["n_way"], len(classes))  # Can't have more ways than classes
    k_support = config["k_support"]
    k_query = config["k_query"]
    episodes_per_epoch = len(train_ds) // (n_way * (k_support + k_query))

    print(f"[{ts()}] Episodic training: {n_way}-way {k_support}-shot + {k_query} query")
    print(f"[{ts()}] Episodes per epoch: {episodes_per_epoch}")

    # Episodic sampler
    train_sampler = EpisodicSampler(
        train_ds.labels, n_way, k_support, k_query, episodes_per_epoch
    )

    train_ld = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=2, pin_memory=True,
        collate_fn=stream_collate_fn,
    )
    val_ld = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=stream_collate_fn,
    )

    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    model = ProtoNetEncoder(
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

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Config: ic={config['in_channels']}, hd={config['hidden_dim']}, "
          f"nl={config['num_layers']}, dr={config['dropout']}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    warmup_epochs = config["warmup_epochs"]

    os.makedirs(stream_ckpt_dir, exist_ok=True)
    best_path = os.path.join(stream_ckpt_dir, "best_model.pt")

    best, patience_counter = 0.0, 0

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, ta_sum, tt = 0.0, 0.0, 0

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for streams_batch, aux_data, targets, _ in train_ld:
            # Extract this stream's data
            gcn_data = streams_batch[stream_name].to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)

            opt.zero_grad()

            # Forward: get embeddings
            embeddings = model(gcn_data, aux_data)  # (B, D)

            # Prototypical loss
            loss, acc = prototypical_loss(
                embeddings, targets, n_way, k_support, k_query
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += loss.item()
            ta_sum += acc.item()
            tt += 1

        if ep >= warmup_epochs:
            scheduler.step()

        # Validation: use all training data as support set
        model.eval()
        with torch.no_grad():
            # Collect all training embeddings and labels
            train_embeddings = []
            train_labels = []
            for streams_batch, aux_data, targets, _ in DataLoader(
                train_ds, batch_size=config["batch_size"], shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=stream_collate_fn
            ):
                gcn_data = streams_batch[stream_name].to(device)
                aux_data = aux_data.to(device)
                embeddings = model(gcn_data, aux_data)
                train_embeddings.append(embeddings.cpu())
                train_labels.append(targets)

            train_embeddings = torch.cat(train_embeddings, dim=0)  # (N_train, D)
            train_labels = torch.cat(train_labels, dim=0)  # (N_train,)

            # Compute prototypes from all training data
            prototypes = []
            for cls_idx in range(len(classes)):
                cls_mask = train_labels == cls_idx
                if cls_mask.sum() > 0:
                    cls_embeddings = train_embeddings[cls_mask]
                    prototype = cls_embeddings.mean(dim=0)
                    prototypes.append(prototype)
                else:
                    # No training samples for this class
                    prototypes.append(torch.zeros(model.embed_dim))

            prototypes = torch.stack(prototypes).to(device)  # (num_classes, D)

            # Classify val samples by nearest prototype
            vc, vt = 0, 0
            for streams_batch, aux_data, targets, _ in val_ld:
                gcn_data = streams_batch[stream_name].to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)

                embeddings = model(gcn_data, aux_data)  # (B, D)

                # Compute distances to prototypes
                dists = torch.cdist(embeddings, prototypes, p=2)  # (B, num_classes)
                preds = dists.argmin(dim=1)  # Nearest prototype

                vt += targets.size(0)
                vc += (preds == targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * ta_sum / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] [{stream_name}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / max(tt, 1):.4f} | TrainAcc: {ta:.1f}% | Val: {va:.1f}% | "
                f"LR: {lr_now:.2e} | Time: {ep_time:.1f}s"
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
                    "aux_dim": aux_dim,
                    "stream": stream_name,
                    "version": "v31_exp2",
                    "training_type": "prototypical",
                    "config": config,
                    "embed_dim": model.embed_dim,
                },
                best_path,
            )
            print(f"[{ts()}]   -> New best! {va:.1f}%")
        else:
            patience_counter += 1

        # No early stopping - run all 500 epochs, best model already saved

        # Per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                # Re-collect prototypes
                train_embeddings = []
                train_labels = []
                for streams_batch, aux_data, targets, _ in DataLoader(
                    train_ds, batch_size=config["batch_size"], shuffle=False,
                    num_workers=2, pin_memory=True, collate_fn=stream_collate_fn
                ):
                    gcn_data = streams_batch[stream_name].to(device)
                    embeddings = model(gcn_data.to(device), aux_data.to(device))
                    train_embeddings.append(embeddings.cpu())
                    train_labels.append(targets)

                train_embeddings = torch.cat(train_embeddings, dim=0)
                train_labels = torch.cat(train_labels, dim=0)

                prototypes = []
                for cls_idx in range(len(classes)):
                    cls_mask = train_labels == cls_idx
                    if cls_mask.sum() > 0:
                        prototypes.append(train_embeddings[cls_mask].mean(dim=0))
                    else:
                        prototypes.append(torch.zeros(model.embed_dim))
                prototypes = torch.stack(prototypes).to(device)

                for streams_batch, aux_data, targets, _ in val_ld:
                    gcn_data = streams_batch[stream_name].to(device)
                    embeddings = model(gcn_data.to(device), aux_data.to(device))
                    dists = torch.cdist(embeddings, prototypes, p=2)
                    preds = dists.argmin(dim=1)
                    preds_ep.extend(preds.cpu().numpy())
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

    with torch.no_grad():
        # Collect training prototypes
        train_embeddings = []
        train_labels = []
        for streams_batch, aux_data, targets, _ in DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=stream_collate_fn
        ):
            gcn_data = streams_batch[stream_name].to(device)
            embeddings = model(gcn_data.to(device), aux_data.to(device))
            train_embeddings.append(embeddings.cpu())
            train_labels.append(targets)

        train_embeddings = torch.cat(train_embeddings, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        prototypes = []
        for cls_idx in range(len(classes)):
            cls_mask = train_labels == cls_idx
            if cls_mask.sum() > 0:
                prototypes.append(train_embeddings[cls_mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(model.embed_dim))
        prototypes = torch.stack(prototypes).to(device)

        preds, tgts = [], []
        for streams_batch, aux_data, targets, _ in val_ld:
            gcn_data = streams_batch[stream_name].to(device)
            embeddings = model(gcn_data.to(device), aux_data.to(device))
            dists = torch.cdist(embeddings, prototypes, p=2)
            p = dists.argmin(dim=1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    print(f"\n[{ts()}] {name} [{stream_name}] Per-Class Results (best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_cls})")

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0
    print(f"[{ts()}] {name} [{stream_name}] Overall: {ov:.1f}%")

    return {
        "stream": stream_name,
        "overall": ov,
        "per_class": res,
        "best_epoch": ckpt["epoch"],
        "params": param_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL v31 Exp2 - Prototypical Networks on v28 Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="numbers",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints", "v31_exp2"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print(f"KSL v31 Exp2 - Prototypical Networks")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # v31 exp2: Same data split as v28 (12 train, 3 val)
    train_dirs = [args.train_dir, args.val_dir]  # signers 1-12
    val_dir = args.test_dir                       # signers 13-15

    print(f"\nv31 Exp2 Data Split:")
    print(f"  Train dirs: {train_dirs}  (signers 1-12)")
    print(f"  Val dir:    {val_dir}  (signers 13-15)")
    print(f"  Streams: {V31_EXP2_CONFIG['streams']}")
    print(f"  Episodic: {V31_EXP2_CONFIG['n_way']}-way "
          f"{V31_EXP2_CONFIG['k_support']}-shot + {V31_EXP2_CONFIG['k_query']} query")

    print(f"\n[{ts()}] v31 Exp2 Config:")
    print(json.dumps(V31_EXP2_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    categories = []
    if args.model_type in ("numbers", "both"):
        categories.append(("Numbers", NUMBER_CLASSES))
    if args.model_type in ("words", "both"):
        categories.append(("Words", WORD_CLASSES))

    for cat_name, classes in categories:
        cat_key = cat_name.lower()
        ckpt_dir = os.path.join(args.checkpoint_dir, cat_key)

        # Train each stream with prototypical networks
        stream_results = {}
        for stream_name in V31_EXP2_CONFIG["streams"]:
            result = train_stream_prototypical(
                stream_name, cat_name, classes, V31_EXP2_CONFIG,
                train_dirs, val_dir, ckpt_dir, device
            )
            if result:
                stream_results[stream_name] = result

        results[cat_key] = {
            "streams": stream_results,
        }

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v31 Exp2 - Prototypical Networks")
    print(f"[{ts()}] {'=' * 70}")

    for cat_key, cat_result in results.items():
        print(f"\n[{ts()}] {cat_key.upper()}:")
        for sname, sres in cat_result.get("streams", {}).items():
            print(f"[{ts()}]   {sname:>10s}: {sres['overall']:.1f}% "
                  f"(ep {sres['best_epoch']}, {sres['params']:,} params)")

    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v31_exp2_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v31_exp2",
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V31_EXP2_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v28": [
            "1. PROTOTYPICAL LOSS: Replace CE with episodic N-way K-shot training. "
               "Compute class prototypes as mean of support embeddings, classify by "
               "negative Euclidean distance.",
            "2. EPISODIC SAMPLER: Custom sampler yields N classes with K_support + K_query "
               "samples per class per batch.",
            "3. EVALUATION: Use all training data as support set to compute prototypes, "
               "classify val by nearest prototype.",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")
    print(f"[{ts()}] Done.")

    return results


if __name__ == "__main__":
    main()
