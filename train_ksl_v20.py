#!/usr/bin/env python3
"""
KSL Training v20 (Alpine) - Real-Tester Generalization Fix

Changes from v19 (based on real-tester gap analysis):

P0 Critical:
  - P0.1: Deduplicate training data (Signers 2 & 3 are byte-identical)
  - P0.2: Wrist-centric + palm-size normalization (replaces per-sample max(abs))
  - P0.3: MediaPipe version warning

P1 High Priority:
  - P1.1: Shrink model to ~300K params (hidden_dim=48, num_layers=4)
  - P1.2: Bone features kept from v19 (9-channel input)
  - P1.3: Full signer-mimicking augmentation suite
  - P1.4: Velocity before resampling (kept from v19)

P2 Medium Priority:
  - P2.1: Multi-stream (bone-only auxiliary head, 0.3 weight)
  - P2.2: Supervised contrastive loss (auxiliary, 0.1 weight)
  - P2.3: Label smoothing=0.1
  - P2.4: Stronger regularization (weight_decay=1e-3, dropout=0.3, spatial dropout)

Usage:
    python train_ksl_v20.py --model-type numbers
    python train_ksl_v20.py --model-type words
    python train_ksl_v20.py --model-type both --seed 42
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    """P0.3: Log a warning if MediaPipe version != 0.10.14."""
    try:
        import mediapipe as mp
        version = mp.__version__
        if version != "0.10.14":
            print(f"[{ts()}] WARNING: MediaPipe version is {version}, "
                  f"training data was processed with 0.10.14. "
                  f"Landmark coordinates may differ between versions.")
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

# Parent map for bone computation (48 nodes)
# Left hand (0-20): standard hand skeleton tree
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
# Right hand (21-41): same structure, offset by 21
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
# Pose (42-47): shoulder-elbow-wrist chain
POSE_PARENT = [-1, 42, 42, 43, 44, 45]

PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

# Kinematic chains for bone-length perturbation (from root outward)
# Each chain: list of (parent_idx, child_idx) pairs to walk from root
LH_CHAINS = [
    # Thumb
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    # Index
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    # Middle
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    # Ring
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    # Pinky
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]

# Right hand chains are left hand + 21
RH_CHAINS = [
    [(p + 21, c + 21) for p, c in chain]
    for chain in LH_CHAINS
]

# ---------------------------------------------------------------------------
# Config (v20 - unified lightweight model)
# ---------------------------------------------------------------------------

V20_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,          # xyz(3) + velocity(3) + bone(3)
    "hidden_dim": 48,          # P1.1: REDUCED from 96/128 to ~300K params
    "num_layers": 4,           # P1.1: REDUCED from 5/6
    "temporal_kernels": [3, 5, 7],  # Multi-scale TCN (smaller max kernel)
    "batch_size": 32,
    "epochs": 300,
    "learning_rate": 1e-3,     # Higher than v19
    "min_lr": 1e-6,
    "weight_decay": 1e-3,      # P2.4: Stronger regularization
    "dropout": 0.3,            # P2.4
    "spatial_dropout": 0.1,    # P2.4: Node-level dropout
    "label_smoothing": 0.1,    # P2.3
    "patience": 50,
    "warmup_epochs": 10,
    # Augmentation (P1.3)
    "hand_dropout_prob": 0.3,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.4,
    "complete_hand_drop_prob": 0.05,
    "noise_std": 0.02,         # P1.3: spatial jitter
    "noise_prob": 0.5,         # Higher prob for jitter
    "bone_perturb_prob": 0.5,  # P1.3: bone-length perturbation
    "bone_perturb_range": (0.8, 1.2),
    "hand_size_prob": 0.4,     # P1.3: hand-size randomization
    "hand_size_range": (0.85, 1.15),
    "temporal_warp_prob": 0.4, # P1.3: temporal warping
    "temporal_warp_sigma": 0.2,
    "rotation_prob": 0.3,      # P1.3: random rotation
    "rotation_max_deg": 15.0,
    # Mixup
    "mixup_alpha": 0.2,
    # Multi-stream (P2.1)
    "bone_head_weight": 0.3,
    # Supervised contrastive (P2.2)
    "supcon_weight": 0.1,
    "supcon_temperature": 0.07,
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
# P0.1: Data Deduplication
# ---------------------------------------------------------------------------

def get_file_hash(filepath):
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def extract_signer_id(filename):
    """Extract signer ID from filename format CLASS-SIGNER-REP.npy."""
    base = os.path.splitext(filename)[0]  # remove .npy
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def deduplicate_samples(sample_paths):
    """Remove byte-identical duplicate samples. Returns unique paths and stats."""
    seen_hashes = {}
    unique = []
    duplicates = []

    for path in sample_paths:
        h = get_file_hash(path)
        if h not in seen_hashes:
            seen_hashes[h] = path
            unique.append(path)
        else:
            duplicates.append((path, seen_hashes[h]))

    return unique, duplicates

# ---------------------------------------------------------------------------
# P0.2: Wrist-Centric + Palm-Size Normalization
# ---------------------------------------------------------------------------

def normalize_wrist_palm(h):
    """
    Wrist-centric + palm-size normalization (replaces per-sample max(abs)).

    Left hand (0-20): center at wrist (node 0), scale by palm size (node 0 -> node 9)
    Right hand (21-41): center at wrist (node 21), scale by palm size (node 21 -> node 30)
    Pose (42-47): center at mid-shoulder (42+43)/2, scale by shoulder width
    """
    T = h.shape[0]

    # -- Left hand --
    lh = h[:, :21, :]  # (T, 21, 3)
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01  # (T,)

    if np.any(lh_valid):
        # Center at wrist (node 0)
        lh_wrist = lh[:, 0:1, :]  # (T, 1, 3)
        lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]

        # Scale by palm size: norm(node 0 -> node 9 = middle finger MCP)
        # After centering, node 0 is at origin, so palm_size = norm(node 9)
        palm_sizes = np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True)  # (n_valid, 1)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        # Reshape for broadcasting: (n_valid, 1, 1)
        lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]

    h[:, :21, :] = lh

    # -- Right hand --
    rh = h[:, 21:42, :]  # (T, 21, 3)
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01

    if np.any(rh_valid):
        # Center at wrist (node 0 of right hand = node 21 overall, but index 0 in rh slice)
        rh_wrist = rh[:, 0:1, :]  # (T, 1, 3)
        rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]

        # Scale by palm size: norm(node 9 of right hand = MCP)
        palm_sizes = np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True)
        palm_sizes = np.maximum(palm_sizes, 1e-6)
        rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]

    h[:, 21:42, :] = rh

    # -- Pose --
    pose = h[:, 42:48, :]  # (T, 6, 3)
    # Pose nodes: 42=left_shoulder, 43=right_shoulder, 44=left_elbow,
    #             45=right_elbow, 46=left_wrist, 47=right_wrist
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01

    if np.any(pose_valid):
        # Center at mid-shoulder: (node 42 + node 43) / 2 = (pose[0] + pose[1]) / 2
        mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2  # (T, 1, 3)
        pose[pose_valid] = pose[pose_valid] - mid_shoulder[pose_valid]

        # Scale by shoulder width: norm(left_shoulder - right_shoulder)
        shoulder_width = np.linalg.norm(
            pose[pose_valid, 0, :] - pose[pose_valid, 1, :],
            axis=-1, keepdims=True
        )  # (n_valid, 1)
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        pose[pose_valid] = pose[pose_valid] / shoulder_width[:, :, np.newaxis]

    h[:, 42:48, :] = pose

    return h

# ---------------------------------------------------------------------------
# P1.3: Signer-Mimicking Augmentation
# ---------------------------------------------------------------------------

def augment_bone_length_perturbation(h, chains, scale_range=(0.8, 1.2)):
    """
    Perturb bone lengths by walking kinematic chains from root outward.
    Propagates position changes to all downstream joints.
    """
    h = h.copy()
    T = h.shape[0]

    for chain in chains:
        for parent_idx, child_idx in chain:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            # bone vector from parent to child
            bone_vec = h[:, child_idx, :] - h[:, parent_idx, :]
            # Scale the bone
            new_pos = h[:, parent_idx, :] + bone_vec * scale
            # Compute displacement
            displacement = new_pos - h[:, child_idx, :]
            h[:, child_idx, :] = new_pos

            # Propagate to all downstream joints in this chain
            # Find remaining children in the chain after this edge
            found = False
            for p2, c2 in chain:
                if found:
                    h[:, c2, :] += displacement
                if (p2, c2) == (parent_idx, child_idx):
                    found = True

    return h


def augment_hand_size(h, scale_range=(0.85, 1.15)):
    """Scale all hand joints relative to their wrist."""
    h = h.copy()
    scale = np.random.uniform(scale_range[0], scale_range[1])

    # Left hand: scale relative to wrist (node 0)
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h[:, 0:1, :]
        h[np.ix_(lh_valid, range(1, 21))] = (
            wrist[lh_valid] + (h[np.ix_(lh_valid, range(1, 21))] - wrist[lh_valid]) * scale
        )

    # Right hand: scale relative to wrist (node 21)
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h[:, 21:22, :]
        h[np.ix_(rh_valid, range(22, 42))] = (
            wrist[rh_valid] + (h[np.ix_(rh_valid, range(22, 42))] - wrist[rh_valid]) * scale
        )

    return h


def augment_temporal_warp(data_list, sigma=0.2):
    """
    Non-uniform temporal resampling using cumulative random normal.
    data_list: list of arrays each with shape (T, ...).
    Returns list of warped arrays with same T.
    """
    T = data_list[0].shape[0]
    if T < 2:
        return data_list

    # Generate warp field
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

    # Left hand: rotate around wrist (node 0)
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h[lh_valid, 0:1, :]  # (n, 1, 3)
        centered = h[lh_valid, :21, :] - wrist  # (n, 21, 3)
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h[lh_valid, :21, :] = rotated + wrist

    # Right hand: rotate around wrist (node 21)
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h[rh_valid, 21:22, :]
        centered = h[rh_valid, 21:42, :] - wrist
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h[rh_valid, 21:42, :] = rotated + wrist

    return h

# ---------------------------------------------------------------------------
# Bone Features
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors: bone[child] = node[child] - node[parent]."""
    bones = np.zeros_like(h)  # (T, 48, 3)
    for child in range(48):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones

# ---------------------------------------------------------------------------
# Dataset (v20 - dedup, wrist-norm, signer-mimicking augmentation)
# ---------------------------------------------------------------------------

class KSLGraphDataset(Dataset):
    def __init__(self, data_dir, classes, config, aug=False):
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        # Collect all sample paths
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

        # P0.1: Deduplicate training data
        if aug:  # Only dedup training set
            unique_paths, duplicates = deduplicate_samples(all_paths)

            if duplicates:
                # Log signer info for duplicates
                dup_signers = Counter()
                for dup_path, orig_path in duplicates:
                    signer = extract_signer_id(os.path.basename(dup_path))
                    dup_signers[signer] += 1
                print(f"[{ts()}]   Deduplication: {total_before} -> {len(unique_paths)} samples "
                      f"({len(duplicates)} duplicates removed)")
                for signer, count in sorted(dup_signers.items()):
                    print(f"[{ts()}]     Signer {signer}: {count} duplicates removed")

            # Rebuild labels for unique paths only
            unique_set = set(unique_paths)
            self.samples = []
            self.labels = []
            for path, label in zip(all_paths, all_labels):
                if path in unique_set:
                    self.samples.append(path)
                    self.labels.append(label)
                    unique_set.discard(path)  # Handle only first occurrence
        else:
            # Validation set: no dedup needed (different signers)
            self.samples = all_paths
            self.labels = all_labels

        print(f"[{ts()}]   Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]

        # Extract landmarks
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
        # h shape: (f, 48, 3)

        # --- Augmentation: Hand dropout ---
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

        # --- Augmentation: Complete hand drop ---
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # --- Augmentation: Hand swap (flips pose X too) ---
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # --- P0.2: Wrist-centric + palm-size normalization ---
        h = normalize_wrist_palm(h)

        # --- Augmentation P1.3: Bone-length perturbation ---
        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(
                h, LH_CHAINS + RH_CHAINS,
                scale_range=self.config["bone_perturb_range"]
            )

        # --- Augmentation P1.3: Hand-size randomization ---
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])

        # --- Augmentation P1.3: Random rotation around wrist ---
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])

        # --- Augmentation: Scale jitter ---
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)

        # --- Augmentation P1.3: Spatial jitter (Gaussian noise) ---
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        # --- P1.4: Compute velocity BEFORE resampling ---
        velocity = np.zeros_like(h)  # (f, 48, 3)
        velocity[1:] = h[1:] - h[:-1]

        # --- Compute bone features (signer-invariant) ---
        bones = compute_bones(h)  # (f, 48, 3)

        # --- Augmentation P1.3: Temporal warping (before resampling) ---
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones = augment_temporal_warp(
                [h, velocity, bones],
                sigma=self.config["temporal_warp_sigma"]
            )

        # --- Temporal sampling / padding ---
        f = h.shape[0]  # may have changed from temporal warp
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
            velocity = velocity[indices]
            bones = bones[indices]
        else:
            pad_len = self.mf - f
            h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])

        # --- Augmentation: Temporal shift ---
        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z, h[:-shift]], axis=0)
                velocity = np.concatenate([z, velocity[:-shift]], axis=0)
                bones = np.concatenate([z, bones[:-shift]], axis=0)
            elif shift < 0:
                z = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z], axis=0)
                velocity = np.concatenate([velocity[-shift:], z], axis=0)
                bones = np.concatenate([bones[-shift:], z], axis=0)

        # --- Concatenate all features ---
        # h: (mf, 48, 3) position
        # velocity: (mf, 48, 3) temporal differences
        # bones: (mf, 48, 3) structural (child-parent)
        features = np.concatenate([h, velocity, bones], axis=2)  # (mf, 48, 9)

        # Clip for numerical stability
        features = np.clip(features, -10, 10).astype(np.float32)

        # Return (9, mf, 48) = (C, T, N)
        return torch.FloatTensor(features).permute(2, 0, 1), self.labels[idx]

# ---------------------------------------------------------------------------
# Multi-Scale Temporal Convolution
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module):
    """Parallel temporal conv branches with different kernel sizes."""

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
    """Multi-head attention pooling over temporal dimension."""

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
        x_t = x.permute(0, 2, 1)  # (B, T, C)

        outs = []
        for head in self.heads:
            attn_logits = head(x_t)                        # (B, T, 1)
            attn_weights = F.softmax(attn_logits, dim=1)   # (B, T, 1)
            pooled = (x_t * attn_weights).sum(dim=1)       # (B, C)
            outs.append(pooled)

        return torch.cat(outs, dim=1)  # (B, num_heads * C)

# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (P2.2)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) - L2-normalized embeddings
        labels: (B,) - class labels
        """
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: same class = positive
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B, B)
        # Remove self-pairs
        self_mask = torch.eye(B, device=device)
        mask = mask * (1 - self_mask)

        # If no positive pairs exist, return 0
        pos_count = mask.sum(1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=device)

        # Log-sum-exp for numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Denominator: all pairs except self
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-prob for positive pairs
        mean_log_prob = (mask * log_prob).sum(1) / pos_count.clamp(min=1)

        return -mean_log_prob.mean()

# ---------------------------------------------------------------------------
# Spatial Dropout (P2.4)
# ---------------------------------------------------------------------------

class SpatialNodeDropout(nn.Module):
    """Randomly zero out entire nodes with probability p during training."""

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        """x: (B, C, T, N)"""
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        # Create node-level mask: same across C and T dimensions
        mask = torch.ones(B, 1, 1, N, device=x.device)
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask

# ---------------------------------------------------------------------------
# Model (P1.1: Lightweight ST-GCN + Multi-Scale TCN + Dual Head)
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


class KSLGraphNet(nn.Module):
    """
    Lightweight ST-GCN with dual classification heads (P2.1).

    Main head: full features -> class logits
    Bone head: bone-only features -> class logits (auxiliary)
    Embedding: extracted before final FC for SupCon loss (P2.2)
    """

    def __init__(self, nc, nn_=48, ic=9, hd=48, nl=4, tk=(3, 5, 7), dr=0.3,
                 spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        # Channel progression: keep it simple for small model
        # 4 layers: ic -> hd -> hd -> hd*2 -> hd*2
        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk,
                        2 if i in [1, 3] else 1,  # Downsample at layers 1, 3
                        dr, spatial_dropout)
             for i in range(nl)]
        )

        final_ch = ch[-1]

        # Attention pooling
        self.attn_pool = AttentionPool(final_ch, num_heads=2)

        # Embedding dimension (before classifier)
        self.embed_dim = 2 * final_ch

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

        # P2.1: Bone-only auxiliary head
        # Bone features are channels 6:9 of input (indices in the 9-ch input)
        self.bone_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

    def forward(self, x, return_embedding=False):
        b, c, t, n = x.shape
        x = self.data_bn(
            x.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        # x: (B, C_final, T', N)
        x_node_avg = x.mean(dim=3)  # (B, C_final, T')
        embedding = self.attn_pool(x_node_avg)  # (B, embed_dim)

        logits = self.classifier(embedding)

        if return_embedding:
            bone_logits = self.bone_classifier(embedding)
            return logits, bone_logits, embedding
        return logits

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device):
    """Train one split (numbers or words) with v20 improvements."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v20")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLGraphDataset(train_dir, classes, config, aug=True)
    val_ds = KSLGraphDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found in {train_dir}")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}

    train_ld = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Required for mixup
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    model = KSLGraphNet(
        len(classes),
        config["num_nodes"],
        config["in_channels"],
        config["hidden_dim"],
        config["num_layers"],
        tk,
        config["dropout"],
        config.get("spatial_dropout", 0.1),
        adj,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Config: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, "
          f"dropout={config['dropout']}, wd={config['weight_decay']}, "
          f"ls={config['label_smoothing']}, in_ch={config['in_channels']}")
    print(f"[{ts()}] Temporal kernels: {tk}")
    print(f"[{ts()}] Spatial dropout: {config.get('spatial_dropout', 0.1)}")
    print(f"[{ts()}] Mixup alpha: {config.get('mixup_alpha', 0)}")
    print(f"[{ts()}] Bone head weight: {config.get('bone_head_weight', 0)}")
    print(f"[{ts()}] SupCon weight: {config.get('supcon_weight', 0)}, "
          f"temp={config.get('supcon_temperature', 0.07)}")
    print(f"[{ts()}] Signer augmentation: bone_perturb={config.get('bone_perturb_prob', 0)}, "
          f"hand_size={config.get('hand_size_prob', 0)}, "
          f"temporal_warp={config.get('temporal_warp_prob', 0)}, "
          f"rotation={config.get('rotation_prob', 0)}")

    # Losses
    supcon_loss_fn = SupConLoss(temperature=config.get("supcon_temperature", 0.07))
    supcon_weight = config.get("supcon_weight", 0.1)
    bone_head_weight = config.get("bone_head_weight", 0.3)

    # Optimizer + Cosine schedule with warmup
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )

    # Warmup: linear warmup for first warmup_epochs
    warmup_epochs = config["warmup_epochs"]

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        # Warmup LR
        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for d, t in train_ld:
            d, t = d.to(device), t.to(device)
            opt.zero_grad()

            # --- Mixup augmentation ---
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(d.size(0), device=device)
                d_mixed = lam * d + (1 - lam) * d[perm]
                t_perm = t[perm]

                logits, bone_logits, embedding = model(d_mixed, return_embedding=True)

                # Main CE loss (mixed)
                ce_loss = (
                    lam * F.cross_entropy(logits, t, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                )

                # Bone head CE loss (P2.1)
                bone_loss = (
                    lam * F.cross_entropy(bone_logits, t, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(bone_logits, t_perm, label_smoothing=config["label_smoothing"])
                )

                # SupCon loss (P2.2) - use dominant label
                sc_loss = supcon_loss_fn(embedding, t)

                loss = (
                    (1 - bone_head_weight - supcon_weight) * ce_loss
                    + bone_head_weight * bone_loss
                    + supcon_weight * sc_loss
                )

                # For accuracy tracking, use the dominant label
                _, p = logits.max(1)
                tc += (lam * p.eq(t).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()
            else:
                logits, bone_logits, embedding = model(d, return_embedding=True)

                ce_loss = F.cross_entropy(logits, t, label_smoothing=config["label_smoothing"])
                bone_loss = F.cross_entropy(bone_logits, t, label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(embedding, t)

                loss = (
                    (1 - bone_head_weight - supcon_weight) * ce_loss
                    + bone_head_weight * bone_loss
                    + supcon_weight * sc_loss
                )

                _, p = logits.max(1)
                tc += p.eq(t).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += loss.item()
            tt += t.size(0)

        # Step scheduler (after warmup)
        if ep >= warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for d, t in val_ld:
                d, t = d.to(device), t.to(device)
                _, p = model(d).max(1)
                vt += t.size(0)
                vc += p.eq(t).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        # Print every epoch for first 10, then every 10 epochs, plus whenever we beat best
        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / len(train_ld):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
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
                    "version": "v20",
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

        # Print per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for d, t in val_ld:
                    _, p = model(d.to(device)).max(1)
                    preds_ep.extend(p.cpu().numpy())
                    tgts_ep.extend(t.cpu().numpy())
            print(f"[{ts()}]   Per-class at epoch {ep + 1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot_cls = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                acc = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
                print(f"[{ts()}]     {cn:12s}: {acc:5.1f}% ({cor}/{tot_cls})")

    # Final evaluation with per-class breakdown
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for d, t in val_ld:
            _, p = model(d.to(device)).max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(t.cpu().numpy())

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
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL ST-GCN Training v20 (Alpine HPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="both",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_v2"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_v2"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print(f"KSL Training v20 - Real-Tester Generalization Fix (Alpine HPC)")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # P0.3: Check MediaPipe version
    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nTrain dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Ckpt dir:  {args.checkpoint_dir}")

    print(f"\n[{ts()}] v20 Config:")
    print(json.dumps(V20_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    if args.model_type in ("numbers", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v20_numbers")
        results["numbers"] = train_split(
            "Numbers", NUMBER_CLASSES, V20_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    if args.model_type in ("words", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v20_words")
        results["words"] = train_split(
            "Words", WORD_CLASSES, V20_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v20")
    print(f"[{ts()}] {'=' * 70}")
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
    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v20_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v20",
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V20_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v19": [
            "P0.1: Deduplicate training data (remove byte-identical Signers 2&3)",
            "P0.2: Wrist-centric + palm-size normalization (replaces per-sample max(abs))",
            "P0.3: MediaPipe version check/warning",
            "P1.1: Shrink model to ~300K params (hidden_dim=48, layers=4)",
            "P1.2: Keep bone features from v19 (9ch input)",
            "P1.3: Full signer-mimicking augmentation (bone perturb, hand size, temporal warp, rotation)",
            "P1.4: Keep velocity before resampling from v19",
            "P2.1: Bone-only auxiliary classification head (weight=0.3)",
            "P2.2: Supervised contrastive loss (weight=0.1, temp=0.07)",
            "P2.3: Label smoothing=0.1",
            "P2.4: Stronger regularization (wd=1e-3, dropout=0.3, spatial node dropout=0.1)",
            "UNIFIED: Same architecture for numbers and words",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
