#!/usr/bin/env python3
"""
KSL Training v33 (Alpine) - Knowledge Distillation: Ensemble Teacher → GroupNorm Student

Distills the 5-model ensemble's knowledge into a single GroupNorm student model.
The student uses GroupNorm (no BN adaptation needed at inference) and is trained
with a KD loss that combines soft teacher targets with hard labels.

Architecture: Same as v31_exp1 (GroupNorm multi-stream late fusion)
Loss: α * T² * KL(student_soft, teacher_soft) + (1-α) * CE(logits, hard_labels) + GRL signer loss
Teacher: v27 + v28 + v29 + v31_exp1 + v31_exp5 (5-model ensemble, 70.7% combined)

Changes from v31_exp1:
  1. KD loss replaces CE + SupCon
  2. Extended dataset returns teacher soft logits
  3. CutMix/MixUp mixes teacher soft labels the same way as hard labels
  4. Checkpoint metadata includes teacher info

Usage:
    python train_ksl_v33.py --model-type numbers
    python train_ksl_v33.py --model-type words
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
# Graph Topology (same as v31_exp1)
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
NUM_HAND_BODY_FEATURES = 8

# ---------------------------------------------------------------------------
# Config (v33 KD)
# ---------------------------------------------------------------------------

V33_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "streams": ["joint", "bone", "velocity"],
    "hidden_dim": 64,
    "num_layers": 4,
    "temporal_kernels": [3, 5, 7],
    "batch_size": 32,
    "epochs": 300,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "spatial_dropout": 0.1,
    "label_smoothing": 0.1,
    "patience": 80,
    "warmup_epochs": 10,
    # KD parameters
    "kd_temperature": 20.0,
    "kd_alpha": 0.3,       # 30% KD, 70% CE
    # SupCon disabled (harmful with GroupNorm)
    "supcon_weight": 0.0,
    # Signer adversarial
    "grl_lambda_max": 0.3,
    "grl_start_epoch": 0,
    "grl_ramp_epochs": 100,
    # Augmentation (same as v28/v31)
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
# Data Deduplication (same as v31)
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
# Normalization & Augmentation (same as v31_exp1)
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
            pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True
        )
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        pose[pose_valid] = pose[pose_valid] / shoulder_width[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


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
# Feature Computation (same as v31_exp1)
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
# Signer-Balanced Batch Sampler (same as v31)
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
# Temporal CutMix (extended for KD)
# ---------------------------------------------------------------------------

def temporal_cutmix_kd(gcn_input, aux_input, labels, signer_labels, teacher_logits, alpha=1.0):
    """CutMix at temporal dimension, also mixing teacher soft labels."""
    B, C, T, N = gcn_input.shape
    lam = np.random.beta(alpha, alpha)
    cut_len = int(T * (1 - lam))
    if cut_len == 0:
        return gcn_input, aux_input, labels, labels, teacher_logits, teacher_logits, 1.0

    cut_start = np.random.randint(0, T - cut_len + 1)
    indices = torch.randperm(B, device=gcn_input.device)

    gcn_mixed = gcn_input.clone()
    aux_mixed = aux_input.clone()

    gcn_mixed[:, :, cut_start:cut_start+cut_len, :] = gcn_input[indices, :, cut_start:cut_start+cut_len, :]
    aux_mixed[:, cut_start:cut_start+cut_len, :] = aux_input[indices, cut_start:cut_start+cut_len, :]

    lam = 1.0 - cut_len / T
    labels_b = labels[indices]
    teacher_b = teacher_logits[indices]

    return gcn_mixed, aux_mixed, labels, labels_b, teacher_logits, teacher_b, lam


# ---------------------------------------------------------------------------
# KD Dataset (extends KSLMultiStreamDataset with teacher logits)
# ---------------------------------------------------------------------------

class KSLMultiStreamDatasetKD(Dataset):
    """Returns per-stream tensors + teacher soft logits for KD training."""

    def __init__(self, data_dirs, classes, config, teacher_logits_dict, aug=False):
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)
        self.teacher_logits = teacher_logits_dict  # {filename: tensor(num_classes,)}

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

        # Count how many samples have teacher logits
        matched = sum(1 for p in self.samples if os.path.basename(p) in self.teacher_logits)
        dirs_str = ", ".join(data_dirs)
        print(f"[{ts()}]   Loaded {len(self.samples)} samples from {dirs_str}")
        print(f"[{ts()}]   Teacher logits matched: {matched}/{len(self.samples)}")
        print(f"[{ts()}]   Signers: {self.num_signers} unique")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]
        filename = os.path.basename(self.samples[idx])

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

        h_clipped = np.clip(h, -10, 10).astype(np.float32)
        bones_clipped = np.clip(bones, -10, 10).astype(np.float32)
        velocity_clipped = np.clip(velocity, -10, 10).astype(np.float32)

        streams = {
            "joint": torch.FloatTensor(h_clipped).permute(2, 0, 1),
            "bone": torch.FloatTensor(bones_clipped).permute(2, 0, 1),
            "velocity": torch.FloatTensor(velocity_clipped).permute(2, 0, 1),
        }

        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
        aux_tensor = torch.FloatTensor(aux_features)

        # Teacher logits: use stored or uniform fallback
        if filename in self.teacher_logits:
            t_logits = self.teacher_logits[filename]
        else:
            # Uniform distribution as fallback (no teacher signal)
            t_logits = torch.zeros(self.num_classes)

        return streams, aux_tensor, self.labels[idx], self.signer_labels[idx], t_logits


def stream_collate_fn_kd(batch):
    """Custom collate for KD dataset. Stacks per-stream tensors + teacher logits."""
    streams_list, aux_list, labels_list, signer_list, teacher_list = zip(*batch)
    stream_names = list(streams_list[0].keys())
    streams_batch = {
        name: torch.stack([s[name] for s in streams_list])
        for name in stream_names
    }
    aux_batch = torch.stack(aux_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    signers = torch.tensor(signer_list, dtype=torch.long)
    teachers = torch.stack(teacher_list)
    return streams_batch, aux_batch, labels, signers, teachers


# ---------------------------------------------------------------------------
# Model (same as v31_exp1 — GroupNorm)
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
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        return self.dropout(out)


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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


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
        num_groups = 8 if oc >= 8 else 1
        self.bn1 = nn.GroupNorm(num_groups, oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, st, dr)
        if ic != oc or st != 1:
            num_groups_res = 8 if oc >= 8 else 1
            self.residual = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=(st, 1)),
                nn.GroupNorm(num_groups_res, oc)
            )
        else:
            self.residual = nn.Identity()
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


class KSLGraphNetV33(nn.Module):
    """V33 KD Student: GroupNorm ST-GCN (same architecture as v31_exp1)."""

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=3, hd=64, nl=4,
                 tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        num_groups_data = 8 if (nn_ * ic) >= 8 else 1
        self.data_bn = nn.GroupNorm(num_groups_data, nn_ * ic)

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

        num_groups_aux = 1
        for g in [8, 4, 2]:
            if aux_dim % g == 0:
                num_groups_aux = g
                break
        self.aux_bn = nn.GroupNorm(num_groups_aux, aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(128, 64),
        )
        num_groups_aux_conv = 8 if 64 >= 8 else 1
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups_aux_conv, 64),
            nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.embed_dim = gcn_embed_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_signers),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
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
        logits = self.classifier(embedding)
        reversed_embedding = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_embedding)

        return logits, signer_logits, embedding


# ---------------------------------------------------------------------------
# KD Loss
# ---------------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, hard_labels, T=4.0, alpha=0.7, label_smoothing=0.1):
    """Knowledge Distillation loss combining soft and hard targets.

    Args:
        student_logits: (B, C) raw logits from student
        teacher_logits: (B, C) pseudo-logits (log-probs) from teacher
        hard_labels: (B,) integer class labels
        T: temperature for softening
        alpha: weight for KD loss (1-alpha for CE)
        label_smoothing: for hard label CE
    """
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    ce = F.cross_entropy(student_logits, hard_labels, label_smoothing=label_smoothing)
    return alpha * kd + (1 - alpha) * ce


# ---------------------------------------------------------------------------
# Train One Stream (modified for KD)
# ---------------------------------------------------------------------------

def train_stream(stream_name, name, classes, config, train_dir, val_dir,
                 ckpt_dir, device, teacher_logits, use_focal=False):
    """Train one stream model with KD."""
    stream_ckpt_dir = os.path.join(ckpt_dir, stream_name)

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} - Stream: {stream_name} ({len(classes)} classes)")
    print(f"[{ts()}] KD: T={config['kd_temperature']}, alpha={config['kd_alpha']}")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLMultiStreamDatasetKD(train_dir, classes, config, teacher_logits, aug=True)
    # Val set: use base dataset without KD (no teacher logits needed)
    val_ds = KSLMultiStreamDatasetKD(val_dir, classes, config, {}, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    print(f"[{ts()}] Aux dim: {aux_dim}, Num signers: {num_signers}")

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )

    train_ld = DataLoader(
        train_ds, batch_size=config["batch_size"],
        sampler=train_sampler, num_workers=2, pin_memory=True,
        collate_fn=stream_collate_fn_kd,
    )
    val_ld = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=stream_collate_fn_kd,
    )

    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    model = KSLGraphNetV33(
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
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Normalization: GroupNorm (v33 KD student)")
    print(f"[{ts()}] Loss: KD(T={config['kd_temperature']}, alpha={config['kd_alpha']}) + GRL")

    if use_focal:
        class_counts = Counter(train_ds.labels)
        num_classes = len(classes)
        total_samples = len(train_ds)
        alpha_weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 1)
            alpha_weights[cls_idx] = total_samples / (num_classes * count)
        alpha_weights = alpha_weights / alpha_weights.mean()
        alpha_weights = alpha_weights.to(device)
        focal_loss_fn = FocalLoss(
            gamma=2.0, alpha=alpha_weights,
            label_smoothing=config["label_smoothing"],
        )
    else:
        focal_loss_fn = None

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
    mixup_alpha = config.get("mixup_alpha", 0)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    cutmix_prob = config.get("cutmix_prob", 0.3)
    kd_T = config["kd_temperature"]
    kd_alpha = config["kd_alpha"]

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        signer_correct, signer_total = 0, 0
        kd_loss_sum, ce_loss_sum = 0.0, 0.0

        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for streams_batch, aux_data, targets, signer_targets, teacher_batch in train_ld:
            gcn_data = streams_batch[stream_name].to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            teacher_batch = teacher_batch.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and (mixup_alpha > 0)

            if use_cutmix:
                gcn_mixed, aux_mixed, targets_a, targets_b, teacher_a, teacher_b, lam = \
                    temporal_cutmix_kd(gcn_data, aux_data, targets, signer_targets,
                                       teacher_batch, alpha=cutmix_alpha)
                logits, signer_logits, embedding = model(
                    gcn_mixed, aux_mixed, grl_lambda=grl_lambda,
                )
                # Mixed KD loss
                loss_a = kd_loss(logits, teacher_a, targets_a, T=kd_T,
                                 alpha=kd_alpha, label_smoothing=config["label_smoothing"])
                loss_b = kd_loss(logits, teacher_b, targets_b, T=kd_T,
                                 alpha=kd_alpha, label_smoothing=config["label_smoothing"])
                cls_loss = lam * loss_a + (1 - lam) * loss_b
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets_a).float() + (1 - lam) * p.eq(targets_b).float()).sum().item()

            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_mixed = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_mixed = lam * aux_data + (1 - lam) * aux_data[perm]
                t_perm = targets[perm]
                teacher_perm = teacher_batch[perm]
                logits, signer_logits, embedding = model(
                    gcn_mixed, aux_mixed, grl_lambda=grl_lambda,
                )
                loss_a = kd_loss(logits, teacher_batch, targets, T=kd_T,
                                 alpha=kd_alpha, label_smoothing=config["label_smoothing"])
                loss_b = kd_loss(logits, teacher_perm, t_perm, T=kd_T,
                                 alpha=kd_alpha, label_smoothing=config["label_smoothing"])
                cls_loss = lam * loss_a + (1 - lam) * loss_b
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()

            else:
                logits, signer_logits, embedding = model(
                    gcn_data, aux_data, grl_lambda=grl_lambda,
                )
                cls_loss = kd_loss(logits, teacher_batch, targets, T=kd_T,
                                   alpha=kd_alpha, label_smoothing=config["label_smoothing"])
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + grl_lambda * signer_loss
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
            for streams_batch, aux_data, targets, _, _ in val_ld:
                gcn_data = streams_batch[stream_name].to(device)
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
                f"[{ts()}] [{stream_name}] Ep {ep + 1:3d}/{config['epochs']} | "
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
                    "stream": stream_name,
                    "version": "v33_kd",
                    "norm_type": "groupnorm",
                    "teacher": "v27+v28+v29+exp1+exp5",
                    "kd_temperature": kd_T,
                    "kd_alpha": kd_alpha,
                    "config": config,
                },
                best_path,
            )
            print(f"[{ts()}]   -> New best! {va:.1f}%")
        else:
            patience_counter += 1

        # Per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for streams_batch, aux_data, targets, _, _ in val_ld:
                    gcn_data = streams_batch[stream_name].to(device)
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
        for streams_batch, aux_data, targets, _, _ in val_ld:
            gcn_data = streams_batch[stream_name].to(device)
            logits, _, _ = model(gcn_data.to(device), aux_data.to(device),
                                 grl_lambda=0.0)
            _, p = logits.max(1)
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
# Fusion Weight Learning (same as v31)
# ---------------------------------------------------------------------------

def learn_fusion_weights(category_name, classes, config, val_dir, ckpt_dir, device):
    """Grid search optimal fusion weights for 3 streams on val set."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Learning fusion weights for {category_name}")
    print(f"[{ts()}] {'=' * 70}")

    streams = config["streams"]
    adj = build_adj(config["num_nodes"]).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))

    models = {}
    for sname in streams:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{ts()}] WARNING: Missing checkpoint for stream '{sname}': {ckpt_path}")
            return None, 0.0

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV33(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config["num_nodes"], ic=3, hd=config["hidden_dim"],
            nl=config["num_layers"], tk=tk, dr=config["dropout"],
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"[{ts()}] Loaded {sname} model: val_acc={ckpt['val_acc']:.1f}%, "
              f"epoch={ckpt['epoch']}")

    val_ds = KSLMultiStreamDatasetKD(val_dir, classes, config, {}, aug=False)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=stream_collate_fn_kd)

    all_probs = {sname: [] for sname in streams}
    all_labels = []

    with torch.no_grad():
        for streams_batch, aux_data, targets, _, _ in val_ld:
            aux_data = aux_data.to(device)
            all_labels.extend(targets.numpy())

            for sname in streams:
                gcn_data = streams_batch[sname].to(device)
                logits, _, _ = models[sname](gcn_data, aux_data, grl_lambda=0.0)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs[sname].append(probs)

    for sname in streams:
        all_probs[sname] = torch.cat(all_probs[sname], dim=0)
    all_labels = np.array(all_labels)

    print(f"\n[{ts()}] Per-stream val accuracy:")
    for sname in streams:
        preds = all_probs[sname].argmax(dim=1).numpy()
        acc = 100.0 * (preds == all_labels).mean()
        print(f"[{ts()}]   {sname:>10s}: {acc:.1f}%")

    best_acc = 0.0
    best_weights = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}

    for w_joint_100 in range(10, 81, 5):
        for w_bone_100 in range(10, 81, 5):
            w_vel_100 = 100 - w_joint_100 - w_bone_100
            if w_vel_100 < 5:
                continue
            w_joint = w_joint_100 / 100.0
            w_bone = w_bone_100 / 100.0
            w_vel = w_vel_100 / 100.0
            fused = (w_joint * all_probs["joint"] +
                     w_bone * all_probs["bone"] +
                     w_vel * all_probs["velocity"])
            preds = fused.argmax(dim=1).numpy()
            acc = 100.0 * (preds == all_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_weights = {"joint": float(w_joint), "bone": float(w_bone), "velocity": float(w_vel)}

    print(f"\n[{ts()}] Best fusion weights: {best_weights}")
    print(f"[{ts()}] Fused val accuracy: {best_acc:.1f}%")

    equal_fused = sum(all_probs[s] for s in streams) / len(streams)
    equal_preds = equal_fused.argmax(dim=1).numpy()
    equal_acc = 100.0 * (equal_preds == all_labels).mean()
    print(f"[{ts()}] Equal weights val accuracy: {equal_acc:.1f}%")

    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    with open(weights_path, "w") as f:
        json.dump({
            "weights": best_weights,
            "val_accuracy": best_acc,
            "equal_weights_accuracy": equal_acc,
            "category": category_name,
        }, f, indent=2)

    print(f"[{ts()}] Fusion weights saved to {weights_path}")

    return best_weights, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL v33 KD Training: Ensemble Teacher → GroupNorm Student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="both",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints/v33"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))
    parser.add_argument("--kd-labels-dir", type=str, default=os.path.join(base_data, "kd_labels"))

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print(f"KSL Training v33 - Knowledge Distillation")
    print(f"Teacher: v27+v28+v29+exp1+exp5 ensemble (70.7% combined)")
    print(f"Student: GroupNorm ST-GCN (signer-agnostic)")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dirs = [args.train_dir, args.val_dir]  # signers 1-12
    val_dir = args.test_dir                       # signers 13-15

    print(f"\nv33 Data Split:")
    print(f"  Train dirs: {train_dirs}  (signers 1-12)")
    print(f"  Val dir:    {val_dir}  (signers 13-15)")
    print(f"  KD labels:  {args.kd_labels_dir}")

    print(f"\n[{ts()}] v33 Config:")
    print(json.dumps(V33_CONFIG, indent=2, default=str))

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

        # Load teacher soft labels
        kd_path = os.path.join(args.kd_labels_dir, f"{cat_key}_ensemble_logits.pt")
        if os.path.exists(kd_path):
            teacher_logits = torch.load(kd_path, map_location="cpu", weights_only=False)
            print(f"\n[{ts()}] Loaded {len(teacher_logits)} teacher logits from {kd_path}")
        else:
            print(f"\n[{ts()}] ERROR: Teacher logits not found at {kd_path}")
            print(f"[{ts()}] Run generate_kd_labels.py first!")
            continue

        # Train each stream
        stream_results = {}
        for stream_name in V33_CONFIG["streams"]:
            result = train_stream(
                stream_name, cat_name, classes, V33_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
                teacher_logits, use_focal=use_focal,
            )
            if result:
                stream_results[stream_name] = result

        # Learn fusion weights
        fusion_weights, fused_acc = learn_fusion_weights(
            cat_name, classes, V33_CONFIG, val_dir, ckpt_dir, device
        )

        results[cat_key] = {
            "streams": stream_results,
            "fusion_weights": fusion_weights,
            "fused_val_accuracy": fused_acc,
        }

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v33 - Knowledge Distillation")
    print(f"[{ts()}] {'=' * 70}")

    for cat_key, cat_result in results.items():
        print(f"\n[{ts()}] {cat_key.upper()}:")
        for sname, sres in cat_result.get("streams", {}).items():
            print(f"[{ts()}]   {sname:>10s}: {sres['overall']:.1f}% "
                  f"(ep {sres['best_epoch']}, {sres['params']:,} params)")
        fw = cat_result.get("fusion_weights", {})
        fused = cat_result.get("fused_val_accuracy", 0)
        print(f"[{ts()}]   {'fused':>10s}: {fused:.1f}% "
              f"(weights: j={fw.get('joint', 0):.2f}, b={fw.get('bone', 0):.2f}, "
              f"v={fw.get('velocity', 0):.2f})")

    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["fused_val_accuracy"] +
                    results["words"]["fused_val_accuracy"]) / 2
        print(f"\n[{ts()}] Combined fused val: {combined:.1f}%")
        print(f"[{ts()}] Reference: v31 exp1 (GroupNorm, no KD) = 61.4%")
        print(f"[{ts()}] Reference: v31 ensemble (5 models) = 70.7%")

    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v33_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v33_kd",
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V33_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v31_exp1": [
            "1. KD LOSS: α*T²*KL(student,teacher) + (1-α)*CE instead of CE+SupCon",
            "2. TEACHER: 5-model ensemble (v27+v28+v29+exp1+exp5) soft labels",
            "3. GROUPNORM: Same architecture as v31_exp1 (signer-agnostic)",
            "4. NO SUPCON: Disabled (harmful with GroupNorm per v32 results)",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")
    print(f"[{ts()}] Done.")

    return results


if __name__ == "__main__":
    main()
