#!/usr/bin/env python3
"""
KSL Training v21 (Alpine) - Multi-Stream + Contrastive + ArcFace + Signer-Adversarial

Changes from v20:
  1. DEDUP FIX: Only remove byte-identical full signer duplicates (~600-650 samples kept)
  2. NEW FEATURES: Joint angles + fingertip distances as auxiliary MLP branch
  3. ArcFace HEAD: Angular margin classifier (replaces FC in stage 2)
  4. Supervised Contrastive Loss: Higher weight, two-stage training
  5. Signer-Adversarial Training: GRL + signer classifier head
  6. MUCH WIDER Augmentation: shear, joint dropout, axis masking, wider ranges
  7. MODEL: hidden_dim=64, 4 layers, auxiliary MLP for angle/distance features
  8. Two-stage training: Stage 1 (CE+SupCon), Stage 2 (ArcFace+SupCon+Adversarial)

Usage:
    python train_ksl_v21.py --model-type numbers
    python train_ksl_v21.py --model-type words
    python train_ksl_v21.py --model-type both --seed 42
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
    """Log a warning if MediaPipe version != 0.10.14."""
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

# Joint angle topology: for each node, which two parent-child bones meet there
# Format: (node_index, parent_bone_child, child_bone_child)
# A "joint angle" is the angle between the bone coming INTO a node and the bone going OUT.
# For hand nodes that have exactly one parent bone in and one child bone out:
#   angle at node X = angle between bone(parent(X)->X) and bone(X->child(X))
# We precompute which nodes have valid angles (nodes with both a parent and at least one child).

def _build_angle_joints():
    """Build list of (node, parent, child) triples for joint angle computation."""
    # Build children map from PARENT_MAP
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
            # angle at 'node' between bone(parent->node) and bone(node->child)
            angle_joints.append((node, parent, child))
    return angle_joints

ANGLE_JOINTS = _build_angle_joints()
NUM_ANGLE_FEATURES = len(ANGLE_JOINTS)

# Fingertip indices
LH_TIPS = [4, 8, 12, 16, 20]       # Left hand fingertips
RH_TIPS = [25, 29, 33, 37, 41]     # Right hand fingertips (LH + 21)
NUM_FINGERTIP_PAIRS = 10  # C(5,2) = 10 per hand, 20 total

# ---------------------------------------------------------------------------
# Config (v21)
# ---------------------------------------------------------------------------

V21_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,           # GCN input: xyz(3) + velocity(3) + bone(3)
    "hidden_dim": 64,           # Slightly larger than v20's 48
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
    # ArcFace
    "arcface_s": 30.0,
    "arcface_m": 0.3,
    # SupCon
    "supcon_weight_stage1": 0.5,
    "supcon_weight_stage2": 0.3,
    "supcon_temperature": 0.07,
    # Signer adversarial
    "grl_lambda_max": 0.1,
    "grl_start_epoch": 50,
    "grl_ramp_epochs": 100,
    # Stage transition
    "stage2_start_epoch": 50,
    # Augmentation (AGGRESSIVE)
    "rotation_prob": 0.5,
    "rotation_max_deg": 30.0,
    "shear_prob": 0.4,
    "shear_max": 0.5,
    "joint_dropout_prob": 0.3,
    "joint_dropout_rate": 0.15,
    "axis_mask_prob": 0.1,
    "noise_std": 0.03,
    "noise_prob": 0.5,
    "bone_perturb_prob": 0.5,
    "bone_perturb_range": (0.7, 1.3),
    "hand_size_prob": 0.5,
    "hand_size_range": (0.7, 1.3),
    "temporal_warp_prob": 0.5,
    "temporal_warp_sigma": 0.3,
    "hand_dropout_prob": 0.3,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.4,
    "complete_hand_drop_prob": 0.05,
    "mixup_alpha": 0.2,
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
# Data Deduplication (v21: signer-group level only)
# ---------------------------------------------------------------------------

def get_file_hash(filepath):
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def extract_signer_id(filename):
    """Extract signer ID from filename format CLASS-SIGNER-REP.npy."""
    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def deduplicate_signer_groups(sample_paths):
    """
    V21 dedup: Only remove truly byte-identical FULL signer duplicates.
    Group files by (class_dir, signer_id). If two signer groups within the
    same class have identical file contents (all files match), remove one.
    This is much less aggressive than v20's per-file dedup.
    Should keep ~600-650 samples instead of v20's 240.
    """
    # Group paths by (class_dir, signer_id)
    groups = defaultdict(list)
    for path in sample_paths:
        class_dir = os.path.basename(os.path.dirname(path))
        signer = extract_signer_id(os.path.basename(path))
        groups[(class_dir, signer)].append(path)

    # For each class, check if any two signer groups are byte-identical
    # Group by class
    class_groups = defaultdict(dict)
    for (class_dir, signer), paths in groups.items():
        class_groups[class_dir][signer] = sorted(paths)

    duplicated_signers = set()  # (class_dir, signer) pairs to remove
    total_removed = 0

    for class_dir, signer_dict in class_groups.items():
        signers = list(signer_dict.keys())
        # Compute a "group hash" for each signer: concatenation of sorted file hashes
        group_hashes = {}
        for signer in signers:
            file_hashes = tuple(sorted(get_file_hash(p) for p in signer_dict[signer]))
            group_hashes[signer] = file_hashes

        # Find duplicate groups
        seen = {}
        for signer in signers:
            gh = group_hashes[signer]
            if gh in seen:
                # This signer group is a duplicate of seen[gh]
                duplicated_signers.add((class_dir, signer))
                total_removed += len(signer_dict[signer])
                print(f"[{ts()}]     Class {class_dir}: Signer {signer} is duplicate of "
                      f"Signer {seen[gh]} ({len(signer_dict[signer])} files)")
            else:
                seen[gh] = signer

    # Build unique paths list
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
# Augmentation Suite (v21 - AGGRESSIVE)
# ---------------------------------------------------------------------------

def augment_bone_length_perturbation(h, chains, scale_range=(0.7, 1.3)):
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


def augment_hand_size(h, scale_range=(0.7, 1.3)):
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


def augment_temporal_warp(data_list, sigma=0.3):
    """Non-uniform temporal resampling using cumulative random normal."""
    T = data_list[0].shape[0]
    if T < 2:
        return data_list
    warp = np.cumsum(np.abs(np.random.normal(1.0, sigma, T)))
    warp = warp / warp[-1] * (T - 1)
    indices = np.clip(np.round(warp).astype(int), 0, T - 1)
    return [arr[indices] for arr in data_list]


def augment_rotation(h, max_deg=30.0):
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


def augment_shear(h, max_shear=0.5):
    """v21 NEW: Apply random shear transformation on a random pair of axes."""
    h = h.copy()
    s = np.random.uniform(-max_shear, max_shear)
    # Pick a random pair of axes to shear
    axis_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    src_axis, dst_axis = axis_pairs[np.random.randint(len(axis_pairs))]

    # Apply: h[..., dst_axis] += s * h[..., src_axis]
    h[:, :, dst_axis] = h[:, :, dst_axis] + s * h[:, :, src_axis]
    return h


def augment_joint_dropout(h, dropout_rate=0.15):
    """v21 NEW: Zero out random joints across all frames."""
    h = h.copy()
    n_nodes = h.shape[1]
    n_drop = max(1, int(n_nodes * dropout_rate))
    drop_indices = np.random.choice(n_nodes, n_drop, replace=False)
    h[:, drop_indices, :] = 0
    return h


def augment_axis_mask(h):
    """v21 NEW: Zero out an entire coordinate axis (X, Y, or Z)."""
    h = h.copy()
    axis = np.random.randint(3)
    h[:, :, axis] = 0
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
    """
    Compute joint angles at each node where two bones meet.
    Returns: (T, num_angles) array of angles in radians.
    """
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)

    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        # Bone into node: parent -> node
        bone_in = h[:, node, :] - h[:, parent, :]
        # Bone out of node: node -> child
        bone_out = h[:, child, :] - h[:, node, :]

        # Normalize
        norm_in = np.linalg.norm(bone_in, axis=-1, keepdims=True)
        norm_out = np.linalg.norm(bone_out, axis=-1, keepdims=True)
        norm_in = np.maximum(norm_in, 1e-8)
        norm_out = np.maximum(norm_out, 1e-8)

        bone_in_n = bone_in / norm_in
        bone_out_n = bone_out / norm_out

        # Dot product -> angle
        dot = np.sum(bone_in_n * bone_out_n, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        angles[:, idx] = np.arccos(dot)

    return angles


def compute_fingertip_distances(h):
    """
    Compute pairwise L2 distances between fingertips for each hand.
    Returns: (T, 20) array -- 10 pairs for left hand + 10 pairs for right hand.
    """
    T = h.shape[0]
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)

    col = 0
    # Left hand
    for i in range(len(LH_TIPS)):
        for j in range(i + 1, len(LH_TIPS)):
            diff = h[:, LH_TIPS[i], :] - h[:, LH_TIPS[j], :]
            distances[:, col] = np.linalg.norm(diff, axis=-1)
            col += 1

    # Right hand
    for i in range(len(RH_TIPS)):
        for j in range(i + 1, len(RH_TIPS)):
            diff = h[:, RH_TIPS[i], :] - h[:, RH_TIPS[j], :]
            distances[:, col] = np.linalg.norm(diff, axis=-1)
            col += 1

    return distances


# ---------------------------------------------------------------------------
# Dataset (v21 - signer-group dedup, signer labels, auxiliary features)
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

        # v21 DEDUP: Only remove byte-identical signer group duplicates
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

        # Build signer label mapping
        self.signer_to_idx = {}
        self.signer_labels = []
        for path in self.samples:
            signer = extract_signer_id(os.path.basename(path))
            if signer not in self.signer_to_idx:
                self.signer_to_idx[signer] = len(self.signer_to_idx)
            self.signer_labels.append(self.signer_to_idx[signer])

        self.num_signers = len(self.signer_to_idx)

        print(f"[{ts()}]   Loaded {len(self.samples)} samples from {data_dir}")
        print(f"[{ts()}]   Signers: {self.num_signers} unique "
              f"({dict(Counter(extract_signer_id(os.path.basename(p)) for p in self.samples))})")

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

        # --- Normalization ---
        h = normalize_wrist_palm(h)

        # --- v21 Augmentation: Bone-length perturbation ---
        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(
                h, LH_CHAINS + RH_CHAINS,
                scale_range=self.config["bone_perturb_range"]
            )

        # --- v21 Augmentation: Hand-size randomization ---
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])

        # --- v21 Augmentation: Random rotation (WIDER range) ---
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])

        # --- v21 NEW Augmentation: Shear ---
        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, max_shear=self.config["shear_max"])

        # --- v21 NEW Augmentation: Joint dropout ---
        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, dropout_rate=self.config["joint_dropout_rate"])

        # --- v21 NEW Augmentation: Axis masking ---
        if self.aug and np.random.random() < self.config["axis_mask_prob"]:
            h = augment_axis_mask(h)

        # --- Augmentation: Scale jitter ---
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)

        # --- Augmentation: Spatial jitter (Gaussian noise, WIDER sigma) ---
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        # --- Compute velocity BEFORE resampling ---
        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]

        # --- Compute bone features ---
        bones = compute_bones(h)

        # --- Compute auxiliary features: joint angles + fingertip distances ---
        joint_angles = compute_joint_angles(h)       # (f, NUM_ANGLE_FEATURES)
        fingertip_dists = compute_fingertip_distances(h)  # (f, 20)

        # --- Augmentation: Temporal warping (WIDER sigma) ---
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists],
                sigma=self.config["temporal_warp_sigma"]
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

        # --- Augmentation: Temporal shift ---
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

        # --- Build GCN features: (mf, 48, 9) = xyz + velocity + bone ---
        gcn_features = np.concatenate([h, velocity, bones], axis=2)
        gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)

        # --- Build auxiliary features: (mf, num_angles + 20) ---
        aux_features = np.concatenate([joint_angles, fingertip_dists], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)

        # GCN features: (C=9, T=mf, N=48)
        gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)
        # Aux features: (T=mf, D_aux)
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
# ArcFace Head (v21 NEW)
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """ArcFace angular margin classifier."""

    def __init__(self, in_features, num_classes, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:  # inference mode
            return self.s * cosine
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = torch.cos(theta + self.m * one_hot)
        return self.s * output


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (v21 NEW)
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
# Model (v21: ST-GCN + Auxiliary MLP + ArcFace + Signer Adversarial)
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


class KSLGraphNetV21(nn.Module):
    """
    V21 ST-GCN with:
    - Auxiliary MLP branch for joint angles + fingertip distances
    - ArcFace classification head (used in stage 2)
    - Standard CE classifier (used in stage 1)
    - Signer-adversarial head with GRL
    - Embedding output for SupCon loss
    """

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=9, hd=64, nl=4,
                 tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=None,
                 arcface_s=30.0, arcface_m=0.3):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        # Channel progression: ic -> hd -> hd -> hd*2 -> hd*2
        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk,
                        2 if i in [1, 3] else 1,
                        dr, spatial_dropout)
             for i in range(nl)]
        )

        final_ch = ch[-1]

        # Attention pooling for GCN output
        self.attn_pool = AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch  # 2 heads * final_ch

        # Auxiliary MLP for angle/distance features
        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(64, 32),
        )
        # Temporal pooling for aux features
        self.aux_attn = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        # Combined embedding dimension
        self.embed_dim = gcn_embed_dim + 32

        # Stage 1: Standard CE classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

        # Stage 2: ArcFace head
        self.arcface = ArcFaceHead(self.embed_dim, nc, s=arcface_s, m=arcface_m)

        # Signer adversarial head
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_signers),
        )

    def forward(self, gcn_input, aux_input, labels=None, grl_lambda=0.0,
                use_arcface=False):
        """
        Args:
            gcn_input: (B, C=9, T, N=48) - GCN features
            aux_input: (B, T, D_aux) - auxiliary features (angles + distances)
            labels: (B,) class labels (needed for ArcFace during training)
            grl_lambda: float - gradient reversal strength
            use_arcface: bool - whether to use ArcFace head (stage 2) or CE head (stage 1)

        Returns:
            logits: (B, nc) class logits
            signer_logits: (B, num_signers) signer logits
            embedding: (B, embed_dim) for SupCon loss
        """
        b, c, t, n = gcn_input.shape

        # GCN branch
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        x_node_avg = x.mean(dim=3)  # (B, C_final, T')
        gcn_embedding = self.attn_pool(x_node_avg)  # (B, gcn_embed_dim)

        # Auxiliary branch
        # aux_input: (B, T, D_aux)
        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)  # BN over feature dim
        aux = self.aux_mlp(aux)  # (B, T, 32)

        # Attention pooling over time
        attn_logits = self.aux_attn(aux)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        aux_embedding = (aux * attn_weights).sum(dim=1)  # (B, 32)

        # Combine embeddings
        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)  # (B, embed_dim)

        # Classification
        if use_arcface:
            logits = self.arcface(embedding, labels)
        else:
            logits = self.classifier(embedding)

        # Signer adversarial
        reversed_embedding = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_embedding)

        return logits, signer_logits, embedding


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device):
    """Train one split (numbers or words) with v21 improvements."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v21")
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

    # Compute auxiliary feature dimension from constants
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS
    print(f"[{ts()}] Auxiliary features: {NUM_ANGLE_FEATURES} joint angles + "
          f"{2 * NUM_FINGERTIP_PAIRS} fingertip distances = {aux_dim} total")
    print(f"[{ts()}] Num signers in training: {num_signers}")

    train_ld = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
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
    model = KSLGraphNetV21(
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
        arcface_s=config["arcface_s"],
        arcface_m=config["arcface_m"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Config: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, "
          f"dropout={config['dropout']}, wd={config['weight_decay']}, "
          f"ls={config['label_smoothing']}, in_ch={config['in_channels']}")
    print(f"[{ts()}] Temporal kernels: {tk}")
    print(f"[{ts()}] ArcFace: s={config['arcface_s']}, m={config['arcface_m']}")
    print(f"[{ts()}] SupCon: stage1_w={config['supcon_weight_stage1']}, "
          f"stage2_w={config['supcon_weight_stage2']}, temp={config['supcon_temperature']}")
    print(f"[{ts()}] GRL: lambda_max={config['grl_lambda_max']}, "
          f"start_epoch={config['grl_start_epoch']}, ramp={config['grl_ramp_epochs']}")
    print(f"[{ts()}] Stage2 starts at epoch {config['stage2_start_epoch']}")
    print(f"[{ts()}] Augmentation: rotation_deg={config['rotation_max_deg']}, "
          f"shear={config['shear_max']}, joint_dropout={config['joint_dropout_rate']}, "
          f"noise={config['noise_std']}, temporal_warp_sigma={config['temporal_warp_sigma']}")

    # Losses
    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])

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

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    stage2_start = config["stage2_start_epoch"]

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        # Determine training stage
        use_arcface = (ep >= stage2_start)
        stage_name = "Stage2(ArcFace)" if use_arcface else "Stage1(CE)"

        # Compute GRL lambda for this epoch
        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        # SupCon weight for this epoch
        supcon_weight = (config["supcon_weight_stage1"] if not use_arcface
                         else config["supcon_weight_stage2"])

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

            # --- Mixup augmentation ---
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_mixed = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_mixed = lam * aux_data + (1 - lam) * aux_data[perm]
                t_perm = targets[perm]

                # For ArcFace, use the dominant label
                af_labels = targets if lam >= 0.5 else t_perm

                logits, signer_logits, embedding = model(
                    gcn_mixed, aux_mixed,
                    labels=af_labels if use_arcface else None,
                    grl_lambda=grl_lambda,
                    use_arcface=use_arcface,
                )

                # Classification loss
                if use_arcface:
                    # ArcFace logits already incorporate margin for af_labels
                    cls_loss = (
                        lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                        + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                    )
                else:
                    cls_loss = (
                        lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                        + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                    )

                # SupCon loss (use dominant label)
                sc_loss = supcon_loss_fn(embedding, targets)

                # Signer adversarial loss
                signer_loss = F.cross_entropy(signer_logits, signer_targets)

                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss

                # Accuracy tracking with dominant label
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()
            else:
                logits, signer_logits, embedding = model(
                    gcn_data, aux_data,
                    labels=targets if use_arcface else None,
                    grl_lambda=grl_lambda,
                    use_arcface=use_arcface,
                )

                cls_loss = F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)

                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss

                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += loss.item()
            tt += targets.size(0)

        # Step scheduler (after warmup)
        if ep >= warmup_epochs:
            scheduler.step()

        # Validation (always use CE classifier for consistent evaluation)
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                # Use classifier (CE) for validation for consistency
                logits, _, _ = model(gcn_data, aux_data, labels=None,
                                     grl_lambda=0.0, use_arcface=False)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} [{stage_name}] | "
                f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
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
                    "version": "v21",
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
                for gcn_data, aux_data, targets, _ in val_ld:
                    logits, _, _ = model(gcn_data.to(device), aux_data.to(device),
                                         labels=None, grl_lambda=0.0, use_arcface=False)
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
                                 labels=None, grl_lambda=0.0, use_arcface=False)
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
        description="KSL ST-GCN Training v21 (Alpine HPC)",
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
    print(f"KSL Training v21 - Multi-Stream + Contrastive + ArcFace + Adversarial")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
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

    print(f"\n[{ts()}] v21 Config:")
    print(json.dumps(V21_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    if args.model_type in ("numbers", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v21_numbers")
        results["numbers"] = train_split(
            "Numbers", NUMBER_CLASSES, V21_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    if args.model_type in ("words", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, "v21_words")
        results["words"] = train_split(
            "Words", WORD_CLASSES, V21_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v21")
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
        f"v21_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v21",
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V21_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v20": [
            "1. DEDUP FIX: Signer-group dedup only (keep ~600-650 samples vs v20's 240)",
            "2. NEW FEATURES: Joint angles + fingertip distances as auxiliary MLP branch",
            "3. ArcFace HEAD: Angular margin classifier (stage 2, s=30, m=0.3)",
            "4. SupCon INCREASED: stage1=0.5, stage2=0.3 (was 0.1 in v20)",
            "5. SIGNER-ADVERSARIAL: GRL + signer classifier, lambda ramp 0->0.1",
            "6. WIDER AUGMENTATION: rotation +/-30deg, shear, joint dropout, axis mask, "
               "noise 0.03, temporal warp sigma=0.3, bone perturb (0.7-1.3), hand size (0.7-1.3)",
            "7. MODEL: hidden_dim=64 (was 48), aux MLP branch for angles/distances",
            "8. TWO-STAGE: Stage1 (ep 1-50) CE+SupCon, Stage2 (ep 51-200) ArcFace+SupCon+GRL",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
