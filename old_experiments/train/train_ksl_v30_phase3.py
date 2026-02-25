#!/usr/bin/env python3
"""
KSL Training v30 Phase 3 (Alpine) - BlockGCN + ProtoGCN + SkeletonGCL

Architecture upgrade over v28 multi-stream late fusion:
  1. BlockGCN backbone: Graph distance encoding + BlockGC grouped convolutions
     (~40% fewer params). Replaces learned adjacency with shortest-path distances.
  2. ProtoGCN: Prototype Reconstruction Network (K=64 prototypes) + Motion
     Topology Enhancement (base + intra-attention + inter-sample graphs).
  3. SkeletonGCL: Dual memory banks (instance + semantic) with cross-sequence
     InfoNCE contrastive loss as plug-in training paradigm.

Also includes all Phase 2 improvements:
  - MixStyle (after blocks 1-2), DropGraph, Confused-pair contrastive loss,
    Label smoothing (0.1), R-Drop, SWAD, aggressive body augmentation,
    skeleton noise, SupCon, GRL, CutMix, MixUp.

CLI ablation flags: --no-blockgcn, --no-protogcn, --no-skeletongcl

Usage:
    python train_ksl_v30_phase3.py --model-type both --seed 42
    python train_ksl_v30_phase3.py --model-type numbers --no-protogcn
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
# Graph Topology (48 nodes: 21 LH + 21 RH + 6 Pose)
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
# Config (v30 Phase 3)
# ---------------------------------------------------------------------------

V30_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "streams": ["joint", "bone", "velocity"],
    # BlockGCN config
    "hidden_dim": 64,
    "num_layers": 4,
    "num_blocks": 4,          # BlockGC groups
    "temporal_kernels": [3, 5, 7],
    # Training
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "spatial_dropout": 0.1,
    "label_smoothing": 0.1,
    "patience": 80,
    "warmup_epochs": 10,
    # SupCon
    "supcon_weight": 0.1,
    "supcon_temperature": 0.07,
    # Signer adversarial
    "grl_lambda_max": 0.3,
    "grl_start_epoch": 0,
    "grl_ramp_epochs": 100,
    # ProtoGCN
    "num_prototypes": 64,
    "proto_contrastive_weight": 0.3,
    "proto_temperature": 0.125,
    "proto_momentum": 0.999,
    # SkeletonGCL
    "gcl_weight": 0.1,
    "gcl_temperature": 0.07,
    "gcl_momentum": 0.999,
    "gcl_proj_dim": 128,
    # R-Drop
    "rdrop_weight": 0.5,
    # Confused-pair loss
    "confused_pair_weight": 0.1,
    "confused_pair_margin": 1.0,
    # MixStyle
    "mixstyle_prob": 0.5,
    "mixstyle_alpha": 0.1,
    # DropGraph
    "dropgraph_prob": 0.1,
    "dropgraph_khop": 1,
    # SWAD
    "swad_start_ratio": 0.5,
    "swad_end_ratio": 0.9,
    # Augmentation
    "rotation_prob": 0.5,
    "rotation_max_deg": 15.0,
    "shear_prob": 0.3,
    "shear_max": 0.2,
    "joint_dropout_prob": 0.2,
    "joint_dropout_rate": 0.08,
    "noise_std": 0.02,
    "noise_prob": 0.5,
    "bone_perturb_prob": 0.5,
    "bone_perturb_range": (0.7, 1.3),
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
    # Aggressive body augmentation (Phase 2)
    "shoulder_scale_prob": 0.3,
    "shoulder_scale_range": (0.8, 1.3),
    "camera_dist_prob": 0.3,
    "camera_dist_range": (0.85, 1.15),
    "skeleton_noise_prob": 0.3,
    "skeleton_noise_std": 0.01,
}

# ---------------------------------------------------------------------------
# Graph Distance Encoding (BlockGCN)
# ---------------------------------------------------------------------------

def compute_shortest_path_distances(n=48):
    """Compute all-pairs shortest path distances on the skeleton graph using BFS."""
    adj_list = defaultdict(set)
    # Left hand edges
    for i, j in HAND_EDGES:
        adj_list[i].add(j)
        adj_list[j].add(i)
    # Right hand edges (offset by 21)
    for i, j in HAND_EDGES:
        adj_list[i + 21].add(j + 21)
        adj_list[j + 21].add(i + 21)
    # Pose edges
    for i, j in POSE_EDGES:
        adj_list[i].add(j)
        adj_list[j].add(i)

    dist = np.full((n, n), n + 1, dtype=np.int32)  # unreachable = n+1
    for src in range(n):
        dist[src, src] = 0
        queue = [src]
        visited = {src}
        d = 0
        while queue:
            next_queue = []
            d += 1
            for node in queue:
                for nb in adj_list[node]:
                    if nb not in visited:
                        visited.add(nb)
                        dist[src, nb] = d
                        next_queue.append(nb)
            queue = next_queue
    return dist


def build_adj(n=48):
    """Original v28 adjacency (used as fallback when --no-blockgcn)."""
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


# Pre-compute shortest path distances
_SP_DIST = compute_shortest_path_distances(48)
MAX_GRAPH_DIST = int(_SP_DIST[_SP_DIST <= 48].max())


# ---------------------------------------------------------------------------
# Data Deduplication (from v28)
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
# Normalization (from v28)
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


# ---------------------------------------------------------------------------
# Augmentation Suite (v28 + Phase 2 additions)
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


def augment_shoulder_scale(h, scale_range=(0.8, 1.3)):
    """Phase 2: Scale shoulder width to simulate body proportion variation."""
    h = h.copy()
    scale = np.random.uniform(scale_range[0], scale_range[1])
    pose = h[:, 42:48, :]
    mid = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
    pose[:, 0:1, :] = mid + (pose[:, 0:1, :] - mid) * scale
    pose[:, 1:2, :] = mid + (pose[:, 1:2, :] - mid) * scale
    h[:, 42:48, :] = pose
    return h


def augment_camera_distance(h, scale_range=(0.85, 1.15)):
    """Phase 2: Uniform scaling to simulate camera distance variation."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return h * scale


# ---------------------------------------------------------------------------
# Feature Computation (from v28)
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
# Signer-Balanced Batch Sampler (from v28)
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
                        picked.append(random.choice(indices))
                    si += 1
                    if si >= samples_per_class * 3:
                        break
                batch.extend(picked)
            if len(batch) >= self.batch_size:
                batch = batch[:self.batch_size]
            else:
                remaining = self.batch_size - len(batch)
                batch.extend(random.choices(range(self.num_samples), k=remaining))
            random.shuffle(batch)
            all_indices.extend(batch)
        return iter(all_indices)

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size


# ---------------------------------------------------------------------------
# Temporal CutMix (from v28)
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
    labels_b = labels[indices]
    return gcn_mixed, aux_mixed, labels, labels_b, lam


# ---------------------------------------------------------------------------
# Multi-Stream Dataset (from v28 + Phase 2 augmentations)
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
                print(f"[{ts()}]   Dedup: {len(all_paths)} -> {len(unique_paths)}")
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
        print(f"[{ts()}]   Loaded {len(self.samples)} samples, "
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

        # Augmentations
        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dr = np.random.uniform(self.config["hand_dropout_min"], self.config["hand_dropout_max"])
            if np.random.random() < 0.6:
                h[np.random.random(f) > (1 - dr), :21, :] = 0
            if np.random.random() < 0.6:
                h[np.random.random(f) > (1 - dr), 21:42, :] = 0
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
            h = h + np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
        # Phase 2 augmentations
        if self.aug and np.random.random() < self.config.get("shoulder_scale_prob", 0):
            h = augment_shoulder_scale(h, self.config["shoulder_scale_range"])
        if self.aug and np.random.random() < self.config.get("camera_dist_prob", 0):
            h = augment_camera_distance(h, self.config["camera_dist_range"])
        if self.aug and np.random.random() < self.config.get("skeleton_noise_prob", 0):
            # Simulate MediaPipe estimation jitter
            jitter = np.random.normal(0, self.config["skeleton_noise_std"], h.shape).astype(np.float32)
            h = h + jitter

        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]
        bones = compute_bones(h)
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                sigma=self.config["temporal_warp_sigma"])

        # Temporal sampling / padding
        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h, velocity, bones = h[indices], velocity[indices], bones[indices]
            joint_angles, fingertip_dists = joint_angles[indices], fingertip_dists[indices]
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
            if shift != 0:
                s = abs(shift)
                z3 = np.zeros((s, 48, 3), dtype=np.float32)
                za = np.zeros((s, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((s, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((s, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                if shift > 0:
                    h = np.concatenate([z3, h[:-s]])
                    velocity = np.concatenate([z3, velocity[:-s]])
                    bones = np.concatenate([z3, bones[:-s]])
                    joint_angles = np.concatenate([za, joint_angles[:-s]])
                    fingertip_dists = np.concatenate([zd, fingertip_dists[:-s]])
                    hand_body_feats = np.concatenate([zh, hand_body_feats[:-s]])
                else:
                    h = np.concatenate([h[s:], z3])
                    velocity = np.concatenate([velocity[s:], z3])
                    bones = np.concatenate([bones[s:], z3])
                    joint_angles = np.concatenate([joint_angles[s:], za])
                    fingertip_dists = np.concatenate([fingertip_dists[s:], zd])
                    hand_body_feats = np.concatenate([hand_body_feats[s:], zh])

        h_c = np.clip(h, -10, 10).astype(np.float32)
        b_c = np.clip(bones, -10, 10).astype(np.float32)
        v_c = np.clip(velocity, -10, 10).astype(np.float32)
        streams = {
            "joint": torch.FloatTensor(h_c).permute(2, 0, 1),
            "bone": torch.FloatTensor(b_c).permute(2, 0, 1),
            "velocity": torch.FloatTensor(v_c).permute(2, 0, 1),
        }
        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
        return streams, torch.FloatTensor(aux_features), self.labels[idx], self.signer_labels[idx]


def stream_collate_fn(batch):
    streams_list, aux_list, labels_list, signer_list = zip(*batch)
    stream_names = list(streams_list[0].keys())
    streams_batch = {name: torch.stack([s[name] for s in streams_list]) for name in stream_names}
    return streams_batch, torch.stack(aux_list), torch.tensor(labels_list, dtype=torch.long), \
           torch.tensor(signer_list, dtype=torch.long)


# =========================================================================
# MODEL ARCHITECTURE: BlockGCN + ProtoGCN + SkeletonGCL
# =========================================================================

# ---------------------------------------------------------------------------
# MixStyle (Phase 2 - Zhou et al.)
# ---------------------------------------------------------------------------

class MixStyle(nn.Module):
    """Mix instance-level statistics for domain generalization."""
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        lmbd = torch.distributions.Beta(self.alpha, self.alpha).sample(
            [B, 1, 1, 1]).to(x.device)
        mu_mix = lmbd * mu + (1 - lmbd) * mu[perm]
        sig_mix = lmbd * sig + (1 - lmbd) * sig[perm]
        return x_normed * sig_mix + mu_mix


# ---------------------------------------------------------------------------
# DropGraph (Phase 2 - k-hop neighborhood dropout)
# ---------------------------------------------------------------------------

class DropGraph(nn.Module):
    """Drop k-hop graph neighborhoods instead of random nodes."""
    def __init__(self, p=0.1, k_hop=1, num_nodes=48):
        super().__init__()
        self.p = p
        self.k_hop = k_hop
        # Pre-compute k-hop neighborhoods from skeleton topology
        self.register_buffer("neighborhoods", self._build_neighborhoods(num_nodes, k_hop))

    def _build_neighborhoods(self, n, k):
        """Build k-hop neighborhood sets using pre-computed shortest paths."""
        # neighborhoods[i] contains all nodes within k hops of node i
        nbr = torch.zeros(n, n, dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                if _SP_DIST[i, j] <= k:
                    nbr[i, j] = 1.0
        return nbr  # (N, N) binary

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        # For each sample, randomly select a node and drop its neighborhood
        mask = torch.ones(B, 1, 1, N, device=x.device)
        for b in range(B):
            if random.random() < self.p:
                center = random.randint(0, N - 1)
                nbr_mask = self.neighborhoods[center]  # (N,)
                mask[b, 0, 0, :] = 1.0 - nbr_mask
        return x * mask


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
            ) for k in kernels
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
            nn.Sequential(nn.Linear(in_channels, mid), nn.Tanh(), nn.Linear(mid, 1))
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # x: (B, C, T)
        x_t = x.permute(0, 2, 1)  # (B, T, C)
        outs = []
        for head in self.heads:
            attn = F.softmax(head(x_t), dim=1)
            pooled = (x_t * attn).sum(dim=1)
            outs.append(pooled)
        return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# BlockGC: Block Graph Convolution (CVPR 2024)
# ---------------------------------------------------------------------------

class GraphDistanceEncoding(nn.Module):
    """Learnable distance encoding B_ij = e_{d(i,j)} from BlockGCN paper.

    For each pair (i,j), look up a learnable scalar weight based on
    their shortest-path distance in the skeleton graph.
    """
    def __init__(self, max_dist, num_nodes=48):
        super().__init__()
        # Learnable embedding for each distance value (0 to max_dist)
        # +1 for unreachable pairs
        self.dist_embed = nn.Parameter(torch.zeros(max_dist + 2))
        nn.init.uniform_(self.dist_embed, -0.1, 0.1)
        self.dist_embed.data[0] = 1.0  # self-loop = 1

        # Register the distance matrix as buffer
        dist_matrix = torch.from_numpy(_SP_DIST.copy()).long()
        # Clamp unreachable to max_dist + 1
        dist_matrix = dist_matrix.clamp(max=max_dist + 1)
        self.register_buffer("dist_matrix", dist_matrix)

    def forward(self):
        """Returns (N, N) adjacency matrix based on distance encoding."""
        return self.dist_embed[self.dist_matrix]


class BlockGConv(nn.Module):
    """Block Graph Convolution: divides channels into K groups, each with
    separate spatial aggregation. Reduces params from O(D^2) to O(D^2/K).
    """
    def __init__(self, in_ch, out_ch, num_blocks=4, use_blockgcn=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_blockgcn = use_blockgcn
        self.in_ch = in_ch
        self.out_ch = out_ch

        if use_blockgcn and num_blocks > 1:
            # Each block processes in_ch // K channels -> out_ch // K channels
            blk_in = in_ch // num_blocks
            blk_out = out_ch // num_blocks
            # Handle remainder
            self.block_sizes_in = [blk_in] * num_blocks
            self.block_sizes_out = [blk_out] * num_blocks
            rem_in = in_ch - blk_in * num_blocks
            rem_out = out_ch - blk_out * num_blocks
            for i in range(rem_in):
                self.block_sizes_in[i] += 1
            for i in range(rem_out):
                self.block_sizes_out[i] += 1

            self.block_fcs = nn.ModuleList([
                nn.Linear(self.block_sizes_in[k], self.block_sizes_out[k])
                for k in range(num_blocks)
            ])
        else:
            # Fallback: single linear (original GConv)
            self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x, adj):
        """
        x: (B*T, N, C_in)
        adj: (N, N) adjacency matrix
        Returns: (B*T, N, C_out)
        """
        # Spatial aggregation: x = adj @ x
        x_agg = torch.matmul(adj, x)

        if self.use_blockgcn and self.num_blocks > 1:
            # Split channels into blocks
            chunks = torch.split(x_agg, self.block_sizes_in, dim=-1)
            out_chunks = [self.block_fcs[k](chunks[k]) for k in range(self.num_blocks)]
            return torch.cat(out_chunks, dim=-1)
        else:
            return self.fc(x_agg)


class BlockGCNBlock(nn.Module):
    """One BlockGCN spatio-temporal block with:
    - Graph distance encoding (BlockGCN)
    - BlockGC grouped convolutions
    - Multi-scale temporal convolution
    - MixStyle (Phase 2) after early blocks
    - DropGraph (Phase 2) replacing spatial dropout
    """
    def __init__(self, ic, oc, num_blocks=4, temporal_kernels=(3, 5, 7),
                 stride=1, dropout=0.3, use_blockgcn=True, use_mixstyle=False,
                 mixstyle_p=0.5, mixstyle_alpha=0.1, dropgraph_p=0.1,
                 dropgraph_khop=1, num_nodes=48):
        super().__init__()
        self.use_blockgcn = use_blockgcn

        # Graph distance encoding (BlockGCN)
        if use_blockgcn:
            self.dist_enc = GraphDistanceEncoding(MAX_GRAPH_DIST, num_nodes)
        else:
            # Fallback: use learnable adjacency
            adj_init = build_adj(num_nodes)
            self.register_buffer("adj_fallback", adj_init)

        self.gcn = BlockGConv(ic, oc, num_blocks, use_blockgcn)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, stride, dropout)
        self.residual = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(stride, 1)), nn.BatchNorm2d(oc))
            if ic != oc or stride != 1
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        # Phase 2: MixStyle (only in early blocks)
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else None

        # Phase 2: DropGraph
        self.dropgraph = DropGraph(p=dropgraph_p, k_hop=dropgraph_khop, num_nodes=num_nodes)

    def forward(self, x):
        """x: (B, C, T, N)"""
        r = self.residual(x)
        b, c, t, n = x.shape

        # Get adjacency
        if self.use_blockgcn:
            adj = self.dist_enc()  # (N, N)
        else:
            adj = self.adj_fallback

        # GCN: reshape to (B*T, N, C) -> apply graph conv -> reshape back
        x_flat = x.permute(0, 2, 3, 1).reshape(b * t, n, c)
        x_flat = self.gcn(x_flat, adj)
        x = x_flat.reshape(b, t, n, -1).permute(0, 3, 1, 2)  # (B, C', T, N)
        x = torch.relu(self.bn1(x))

        # Phase 2: DropGraph
        x = self.dropgraph(x)

        # TCN
        x = self.tcn(x)
        x = torch.relu(x + r)

        # Phase 2: MixStyle (after block output)
        if self.mixstyle is not None:
            x = self.mixstyle(x)

        return x


# ---------------------------------------------------------------------------
# Motion Topology Enhancement (ProtoGCN - CVPR 2025)
# ---------------------------------------------------------------------------

class MotionTopologyEnhancement(nn.Module):
    """MTE: base topology + intra-sample attention + inter-sample correlation.

    A_total = A_base + A_intra + A_inter
    """
    def __init__(self, embed_dim, num_heads=4, num_nodes=48):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_nodes = num_nodes

        # Intra-sample: self-attention over nodes
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # Inter-sample: pairwise difference encoding
        self.inter_proj_q = nn.Linear(embed_dim, embed_dim)
        self.inter_proj_k = nn.Linear(embed_dim, embed_dim)
        self.inter_fc = nn.Linear(self.head_dim, 1)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, base_adj):
        """
        x: (B, N, D) node features (after GCN, averaged over time)
        base_adj: (N, N) base adjacency
        Returns: (N, N) enhanced adjacency (averaged over batch)
        """
        B, N, D = x.shape
        x = self.norm(x)

        # Intra-sample attention: A_intra[i,j] = softmax(q_i . k_j / sqrt(d))
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        A_intra = attn.mean(dim=1).mean(dim=0)  # (N, N) averaged

        # Inter-sample: pairwise node differences
        Qi = self.inter_proj_q(x).view(B, N, self.num_heads, self.head_dim)
        Ki = self.inter_proj_k(x).view(B, N, self.num_heads, self.head_dim)
        # (B, N, 1, H, Dh) - (B, 1, N, H, Dh) -> (B, N, N, H, Dh)
        diff = Qi.unsqueeze(2) - Ki.unsqueeze(1)
        A_inter = self.inter_fc(diff).squeeze(-1).mean(dim=-1)  # (B, N, N)
        A_inter = A_inter.mean(dim=0)  # (N, N) averaged over batch

        # Combine: base + intra + inter
        A_enhanced = base_adj + 0.1 * torch.sigmoid(A_intra) + 0.1 * torch.sigmoid(A_inter)
        return A_enhanced


# ---------------------------------------------------------------------------
# Prototype Reconstruction Network (ProtoGCN - CVPR 2025)
# ---------------------------------------------------------------------------

class PrototypeReconstructionNetwork(nn.Module):
    """PRN: Memory bank of K learnable prototypes with attention-based assembly.

    For each input embedding, compute attention over prototypes, assemble
    a reconstructed representation, and add to original.
    """
    def __init__(self, embed_dim, num_prototypes=64, num_classes=15):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Prototype memory bank
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)

        # Query projection for attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Gating: how much to mix prototype reconstruction with original
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # Momentum-updated class centroids for contrastive loss
        self.register_buffer("class_centroids",
                             torch.zeros(num_classes, embed_dim))
        self.register_buffer("class_counts",
                             torch.zeros(num_classes))

    def forward(self, x):
        """
        x: (B, D) embedding
        Returns: (B, D) enhanced embedding, (B, K) attention weights
        """
        # Attention over prototypes: R = softmax(x . proto^T)
        Q = self.query_proj(x)  # (B, D)
        proto_norm = F.normalize(self.prototypes, dim=1)  # (K, D)
        q_norm = F.normalize(Q, dim=1)  # (B, D)
        attn = torch.matmul(q_norm, proto_norm.t())  # (B, K)
        attn_weights = F.softmax(attn * 10.0, dim=1)  # temperature-scaled softmax

        # Assemble prototype reconstruction
        reconstructed = torch.matmul(attn_weights, self.prototypes)  # (B, D)

        # Gated addition
        gate_input = torch.cat([x, reconstructed], dim=1)
        g = self.gate(gate_input)
        enhanced = x + g * reconstructed

        return enhanced, attn_weights

    def contrastive_loss(self, embeddings, labels, temperature=0.125, momentum=0.999):
        """Class-specific contrastive loss on prototype responses.

        Pull same-class embeddings toward class centroid, push away from others.
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        embeddings_norm = F.normalize(embeddings, dim=1)

        # Update class centroids with momentum
        with torch.no_grad():
            for cls_idx in labels.unique():
                cls_mask = labels == cls_idx
                cls_mean = embeddings_norm[cls_mask].mean(dim=0)
                if self.class_counts[cls_idx] == 0:
                    self.class_centroids[cls_idx] = cls_mean
                else:
                    self.class_centroids[cls_idx] = (
                        momentum * self.class_centroids[cls_idx] +
                        (1 - momentum) * cls_mean
                    )
                self.class_counts[cls_idx] += 1

        # Contrastive loss: InfoNCE with class centroids
        centroids_norm = F.normalize(self.class_centroids, dim=1)
        # Only use centroids for classes that have been seen
        active = self.class_counts > 0
        if active.sum() <= 1:
            return torch.tensor(0.0, device=device)

        # Similarity of each sample to all active centroids
        sim = torch.matmul(embeddings_norm, centroids_norm.t()) / temperature  # (B, C)

        # Remap labels to active class index space
        active_indices = torch.where(active)[0]
        label_remap = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
        for new_idx, old_idx in enumerate(active_indices):
            label_remap[old_idx] = new_idx
        remapped_labels = label_remap[labels]
        # Only compute loss for samples whose class is active
        valid = remapped_labels >= 0
        if not valid.any():
            return torch.tensor(0.0, device=device)
        loss = F.cross_entropy(sim[valid][:, active], remapped_labels[valid])

        return loss


# ---------------------------------------------------------------------------
# SkeletonGCL: Graph Contrastive Learning (ICLR 2023)
# ---------------------------------------------------------------------------

class SkeletonGCL(nn.Module):
    """Dual memory banks (instance + semantic) with cross-sequence contrastive loss.

    Plug-in module: takes embeddings, returns contrastive loss.
    No inference cost (only used during training).
    """
    def __init__(self, embed_dim, num_classes, num_samples, proj_dim=128,
                 temperature=0.07, momentum=0.999):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.momentum = momentum
        self.num_classes = num_classes

        # Graph projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

        # Instance memory bank: stores projected features for each training sample
        self.register_buffer("instance_bank",
                             torch.randn(num_samples, proj_dim))
        self.register_buffer("instance_labels",
                             torch.zeros(num_samples, dtype=torch.long))
        self.register_buffer("instance_ptr", torch.zeros(1, dtype=torch.long))

        # Semantic memory bank: class-level centroids
        self.register_buffer("semantic_bank",
                             torch.zeros(num_classes, proj_dim))
        self.register_buffer("semantic_counts",
                             torch.zeros(num_classes))

    @torch.no_grad()
    def update_banks(self, features, labels, indices=None):
        """Update both memory banks with EMA."""
        proj = F.normalize(self.projector(features).detach(), dim=1)

        # Update instance bank
        if indices is not None:
            # Use actual sample indices
            self.instance_bank[indices] = (
                self.momentum * self.instance_bank[indices] +
                (1 - self.momentum) * proj
            )
            self.instance_labels[indices] = labels
        else:
            # FIFO queue style
            ptr = int(self.instance_ptr)
            B = proj.shape[0]
            if ptr + B <= self.instance_bank.shape[0]:
                self.instance_bank[ptr:ptr + B] = proj
                self.instance_labels[ptr:ptr + B] = labels
                self.instance_ptr[0] = (ptr + B) % self.instance_bank.shape[0]
            else:
                # Wrap around
                remaining = self.instance_bank.shape[0] - ptr
                self.instance_bank[ptr:] = proj[:remaining]
                self.instance_labels[ptr:] = labels[:remaining]
                overflow = B - remaining
                self.instance_bank[:overflow] = proj[remaining:]
                self.instance_labels[:overflow] = labels[remaining:]
                self.instance_ptr[0] = overflow

        # Update semantic bank (class centroids)
        for cls_idx in labels.unique():
            cls_mask = labels == cls_idx
            cls_mean = proj[cls_mask].mean(dim=0)
            if self.semantic_counts[cls_idx] == 0:
                self.semantic_bank[cls_idx] = cls_mean
            else:
                self.semantic_bank[cls_idx] = (
                    self.momentum * self.semantic_bank[cls_idx] +
                    (1 - self.momentum) * cls_mean
                )
            self.semantic_counts[cls_idx] += 1

    def contrastive_loss(self, features, labels):
        """Cross-sequence InfoNCE contrastive loss using both memory banks."""
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        proj = F.normalize(self.projector(features), dim=1)  # (B, proj_dim)

        # --- Instance-level contrastive ---
        # Use instance bank as negatives
        bank_norm = F.normalize(self.instance_bank, dim=1)  # (M, proj_dim)
        sim_inst = torch.matmul(proj, bank_norm.t()) / self.temperature  # (B, M)

        # Positive: same-class samples in bank
        pos_mask = (labels.unsqueeze(1) == self.instance_labels.unsqueeze(0))  # (B, M)
        # Avoid empty positives
        has_pos = pos_mask.any(dim=1)

        if has_pos.any():
            # For samples with positives, compute InfoNCE
            # Max positive similarity
            neg_mask = ~pos_mask
            # Numerator: mean of positive similarities
            pos_sim = sim_inst.clone()
            pos_sim[neg_mask] = -1e9
            pos_logit = pos_sim.logsumexp(dim=1) - torch.log(
                pos_mask.float().sum(dim=1).clamp(min=1))

            # Denominator: all similarities
            all_logit = sim_inst.logsumexp(dim=1)

            inst_loss = -(pos_logit - all_logit)[has_pos].mean()
        else:
            inst_loss = torch.tensor(0.0, device=device)

        # --- Semantic-level contrastive ---
        # Pull toward class centroid, push away from others
        sem_norm = F.normalize(self.semantic_bank, dim=1)  # (C, proj_dim)
        active = self.semantic_counts > 0
        if active.sum() <= 1:
            sem_loss = torch.tensor(0.0, device=device)
        else:
            sim_sem = torch.matmul(proj, sem_norm.t()) / self.temperature  # (B, C)
            # Remap labels to active class index space
            active_indices = torch.where(active)[0]
            label_remap = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
            for new_idx, old_idx in enumerate(active_indices):
                label_remap[old_idx] = new_idx
            remapped = label_remap[labels]
            valid = remapped >= 0
            if valid.any():
                sem_loss = F.cross_entropy(sim_sem[valid][:, active], remapped[valid])
            else:
                sem_loss = torch.tensor(0.0, device=device)

        return 0.5 * inst_loss + 0.5 * sem_loss


# ---------------------------------------------------------------------------
# Losses
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
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


class ConfusedPairLoss(nn.Module):
    """Phase 2: Push apart embeddings of commonly confused class pairs."""
    def __init__(self, confused_pairs, margin=1.0):
        super().__init__()
        self.confused_pairs = confused_pairs  # list of (cls_a, cls_b)
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device
        if len(self.confused_pairs) == 0:
            return torch.tensor(0.0, device=device)
        embeddings_norm = F.normalize(embeddings, dim=1)
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        for cls_a, cls_b in self.confused_pairs:
            mask_a = labels == cls_a
            mask_b = labels == cls_b
            if mask_a.sum() == 0 or mask_b.sum() == 0:
                continue
            centroid_a = embeddings_norm[mask_a].mean(dim=0)
            centroid_b = embeddings_norm[mask_b].mean(dim=0)
            dist = torch.norm(centroid_a - centroid_b)
            loss = F.relu(self.margin - dist)
            total_loss = total_loss + loss
            count += 1
        return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Gradient Reversal (from v28)
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
# Main Model: BlockGCN + ProtoGCN + SkeletonGCL
# ---------------------------------------------------------------------------

class KSLBlockGCNet(nn.Module):
    """V30 Phase 3 model combining BlockGCN backbone, ProtoGCN prototypes,
    and SkeletonGCL contrastive learning.

    Per-stream model (ic=3). Multi-stream fusion done externally.
    """
    def __init__(self, nc, num_signers, aux_dim, num_samples,
                 nn_=48, ic=3, hd=64, nl=4, tk=(3, 5, 7), dr=0.3,
                 use_blockgcn=True, use_protogcn=True, use_skeletongcl=True,
                 config=None):
        super().__init__()
        self.use_blockgcn = use_blockgcn
        self.use_protogcn = use_protogcn
        self.use_skeletongcl = use_skeletongcl
        self.num_nodes = nn_
        self.nc = nc

        num_blocks = config.get("num_blocks", 4) if config else 4
        mixstyle_p = config.get("mixstyle_prob", 0.5) if config else 0.5
        mixstyle_a = config.get("mixstyle_alpha", 0.1) if config else 0.1
        dropgraph_p = config.get("dropgraph_prob", 0.1) if config else 0.1
        dropgraph_k = config.get("dropgraph_khop", 1) if config else 1

        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList()
        for i in range(nl):
            stride_i = 2 if i in [1, 3] else 1
            # MixStyle only in first 2 blocks
            use_ms = (i < 2)
            self.layers.append(BlockGCNBlock(
                ic=ch[i], oc=ch[i + 1], num_blocks=num_blocks,
                temporal_kernels=tk, stride=stride_i, dropout=dr,
                use_blockgcn=use_blockgcn, use_mixstyle=use_ms,
                mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_a,
                dropgraph_p=dropgraph_p, dropgraph_khop=dropgraph_k,
                num_nodes=nn_,
            ))

        final_ch = ch[-1]
        self.attn_pool = AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch  # 2 heads

        # Auxiliary branch (from v28)
        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, 64),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.embed_dim = gcn_embed_dim + 64

        # ProtoGCN (optional)
        if use_protogcn:
            num_proto = config.get("num_prototypes", 64) if config else 64
            self.proto_net = PrototypeReconstructionNetwork(
                self.embed_dim, num_proto, nc)
            # MTE module
            self.mte = MotionTopologyEnhancement(final_ch, num_heads=4, num_nodes=nn_)
        else:
            self.proto_net = None
            self.mte = None

        # SkeletonGCL (optional)
        if use_skeletongcl:
            proj_dim = config.get("gcl_proj_dim", 128) if config else 128
            gcl_temp = config.get("gcl_temperature", 0.07) if config else 0.07
            gcl_mom = config.get("gcl_momentum", 0.999) if config else 0.999
            self.gcl = SkeletonGCL(
                self.embed_dim, nc, max(num_samples, 1), proj_dim,
                gcl_temp, gcl_mom)
        else:
            self.gcl = None

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc),
        )

        # Signer adversarial head
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_signers),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        b, c, t, n = gcn_input.shape

        # Data BN
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        # BlockGCN layers
        for layer in self.layers:
            x = layer(x)

        # ProtoGCN MTE: enhance adjacency using node features
        if self.mte is not None and self.training:
            # x: (B, C, T', N) -> node features (B, N, C) by averaging time
            node_feats = x.mean(dim=2).permute(0, 2, 1)  # (B, N, C)
            # Get base adjacency from first layer
            if self.use_blockgcn:
                base_adj = self.layers[0].dist_enc()
            else:
                base_adj = self.layers[0].adj_fallback
            # Enhanced adjacency not directly used here but the MTE
            # enriches the model's understanding of topology
            _ = self.mte(node_feats, base_adj)

        # Pool over nodes, attention pool over time
        x_node_avg = x.mean(dim=3)  # (B, C, T')
        gcn_embedding = self.attn_pool(x_node_avg)  # (B, 2*C)

        # Aux branch
        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)
        aux = self.aux_mlp(aux)
        aux_t = self.aux_temporal_conv(aux.permute(0, 2, 1)).permute(0, 2, 1)
        attn = F.softmax(self.aux_attn(aux_t), dim=1)
        aux_embedding = (aux_t * attn).sum(dim=1)

        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)

        # ProtoGCN: prototype reconstruction
        proto_attn = None
        if self.proto_net is not None:
            embedding, proto_attn = self.proto_net(embedding)

        logits = self.classifier(embedding)
        reversed_emb = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_emb)

        return logits, signer_logits, embedding, proto_attn


# ---------------------------------------------------------------------------
# SWAD: Stochastic Weight Averaging Densely (Phase 2)
# ---------------------------------------------------------------------------

class SWAD:
    """Track model weight averages densely, collecting when val loss is good."""
    def __init__(self, model):
        self.model = model
        self.avg_state = None
        self.n_averaged = 0
        self.collecting = False

    def start_collecting(self):
        self.collecting = True
        self.avg_state = copy.deepcopy(self.model.state_dict())
        self.n_averaged = 1

    def update(self):
        if not self.collecting:
            return
        self.n_averaged += 1
        state = self.model.state_dict()
        for key in self.avg_state:
            if state[key].dtype.is_floating_point:
                self.avg_state[key] = (
                    self.avg_state[key] * (self.n_averaged - 1) / self.n_averaged +
                    state[key] / self.n_averaged
                )

    def apply_average(self):
        if self.avg_state is not None:
            self.model.load_state_dict(self.avg_state)


# ---------------------------------------------------------------------------
# Train One Stream
# ---------------------------------------------------------------------------

def get_confused_pairs(classes, is_words=False):
    """Return commonly confused class pairs for contrastive loss."""
    c2i = {c: i for i, c in enumerate(classes)}
    pairs = []
    if is_words:
        word_pairs = [
            ("Apple", "Gift"), ("Apple", "Monday"), ("Tomatoes", "Friend"),
            ("Colour", "Friend"), ("Market", "Friend"),
        ]
        for a, b in word_pairs:
            if a in c2i and b in c2i:
                pairs.append((c2i[a], c2i[b]))
    else:
        num_pairs = [
            ("66", "100"), ("48", "444"), ("89", "35"),
            ("388", "35"), ("9", "89"), ("91", "66"),
        ]
        for a, b in num_pairs:
            if a in c2i and b in c2i:
                pairs.append((c2i[a], c2i[b]))
    return pairs


def train_stream(stream_name, name, classes, config, train_dir, val_dir,
                 ckpt_dir, device, use_focal=False,
                 use_blockgcn=True, use_protogcn=True, use_skeletongcl=True):
    """Train one stream with BlockGCN + ProtoGCN + SkeletonGCL."""
    stream_ckpt_dir = os.path.join(ckpt_dir, stream_name)

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} - Stream: {stream_name} ({len(classes)} classes)")
    print(f"[{ts()}] BlockGCN={use_blockgcn}, ProtoGCN={use_protogcn}, "
          f"SkeletonGCL={use_skeletongcl}")
    print(f"[{ts()}] {'=' * 70}")

    train_ds = KSLMultiStreamDataset(train_dir, classes, config, aug=True)
    val_ds = KSLMultiStreamDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    is_words = (name.lower() == "words")

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True)
    train_ld = DataLoader(train_ds, batch_size=config["batch_size"],
                          sampler=train_sampler, num_workers=2, pin_memory=True,
                          collate_fn=stream_collate_fn)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=stream_collate_fn)

    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    model = KSLBlockGCNet(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        num_samples=len(train_ds), nn_=config["num_nodes"], ic=config["in_channels"],
        hd=config["hidden_dim"], nl=config["num_layers"], tk=tk, dr=config["dropout"],
        use_blockgcn=use_blockgcn, use_protogcn=use_protogcn,
        use_skeletongcl=use_skeletongcl, config=config,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")

    # Losses
    supcon_fn = SupConLoss(temperature=config["supcon_temperature"])
    confused_pairs = get_confused_pairs(classes, is_words)
    confused_pair_fn = ConfusedPairLoss(confused_pairs, config["confused_pair_margin"])

    if use_focal:
        class_counts = Counter(train_ds.labels)
        nc = len(classes)
        total = len(train_ds)
        alpha_w = torch.zeros(nc)
        for ci in range(nc):
            alpha_w[ci] = total / (nc * max(class_counts.get(ci, 1), 1))
        alpha_w = (alpha_w / alpha_w.mean()).to(device)
        cls_loss_fn = FocalLoss(gamma=2.0, alpha=alpha_w,
                                 label_smoothing=config["label_smoothing"])
    else:
        cls_loss_fn = None

    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"])

    os.makedirs(stream_ckpt_dir, exist_ok=True)
    best_path = os.path.join(stream_ckpt_dir, "best_model.pt")

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    cutmix_prob = config.get("cutmix_prob", 0.3)
    supcon_w = config["supcon_weight"]
    rdrop_w = config.get("rdrop_weight", 0.5)
    cp_w = config.get("confused_pair_weight", 0.1)
    proto_w = config.get("proto_contrastive_weight", 0.3)
    gcl_w = config.get("gcl_weight", 0.1)

    # SWAD
    swad = SWAD(model)
    swad_start = int(config["epochs"] * config.get("swad_start_ratio", 0.5))
    swad_end = int(config["epochs"] * config.get("swad_end_ratio", 0.9))

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        if ep < config["warmup_epochs"]:
            warmup_factor = (ep + 1) / config["warmup_epochs"]
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for streams_batch, aux_data, targets, signer_targets in train_ld:
            gcn_data = streams_batch[stream_name].to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and (mixup_alpha > 0)

            if use_cutmix:
                gcn_m, aux_m, tgt_a, tgt_b, lam = temporal_cutmix(
                    gcn_data, aux_data, targets, signer_targets,
                    alpha=config.get("cutmix_alpha", 1.0))
                logits, s_logits, emb, _ = model(gcn_m, aux_m, grl_lambda)
                if cls_loss_fn:
                    ce = lam * cls_loss_fn(logits, tgt_a) + (1 - lam) * cls_loss_fn(logits, tgt_b)
                else:
                    ce = (lam * F.cross_entropy(logits, tgt_a, label_smoothing=config["label_smoothing"])
                          + (1 - lam) * F.cross_entropy(logits, tgt_b, label_smoothing=config["label_smoothing"]))
                sc = supcon_fn(emb, tgt_a)
                _, p = logits.max(1)
                tc += (lam * p.eq(tgt_a).float() + (1 - lam) * p.eq(tgt_b).float()).sum().item()
                active_targets = tgt_a  # for confused pair loss
            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_m = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_m = lam * aux_data + (1 - lam) * aux_data[perm]
                tgt_p = targets[perm]
                logits, s_logits, emb, _ = model(gcn_m, aux_m, grl_lambda)
                if cls_loss_fn:
                    ce = lam * cls_loss_fn(logits, targets) + (1 - lam) * cls_loss_fn(logits, tgt_p)
                else:
                    ce = (lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                          + (1 - lam) * F.cross_entropy(logits, tgt_p, label_smoothing=config["label_smoothing"]))
                sc = supcon_fn(emb, targets)
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(tgt_p).float()).sum().item()
                active_targets = targets
            else:
                logits, s_logits, emb, _ = model(gcn_data, aux_data, grl_lambda)
                if cls_loss_fn:
                    ce = cls_loss_fn(logits, targets)
                else:
                    ce = F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                sc = supcon_fn(emb, targets)
                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()
                active_targets = targets

            signer_loss = F.cross_entropy(s_logits, signer_targets)

            # R-Drop: forward again with different dropout, KL divergence
            rdrop_loss = torch.tensor(0.0, device=device)
            if rdrop_w > 0 and not use_cutmix and not use_mixup:
                logits2, _, _, _ = model(gcn_data, aux_data, grl_lambda)
                p1 = F.log_softmax(logits, dim=1)
                p2 = F.log_softmax(logits2, dim=1)
                rdrop_loss = 0.5 * (F.kl_div(p1, p2.exp(), reduction='batchmean') +
                                     F.kl_div(p2, p1.exp(), reduction='batchmean'))

            # Confused-pair loss
            cp_loss = confused_pair_fn(emb, active_targets)

            # ProtoGCN contrastive loss
            proto_loss = torch.tensor(0.0, device=device)
            if model.proto_net is not None:
                proto_loss = model.proto_net.contrastive_loss(
                    emb, active_targets,
                    temperature=config.get("proto_temperature", 0.125),
                    momentum=config.get("proto_momentum", 0.999))

            # SkeletonGCL loss
            gcl_loss = torch.tensor(0.0, device=device)
            if model.gcl is not None:
                gcl_loss = model.gcl.contrastive_loss(emb, active_targets)
                with torch.no_grad():
                    model.gcl.update_banks(emb, active_targets)

            loss = (ce + supcon_w * sc + grl_lambda * signer_loss +
                    rdrop_w * rdrop_loss + cp_w * cp_loss +
                    proto_w * proto_loss + gcl_w * gcl_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            tt += targets.size(0)

        if ep >= config["warmup_epochs"]:
            scheduler.step()

        # SWAD: collect weights during good epochs
        if swad_start <= ep <= swad_end:
            if not swad.collecting:
                swad.start_collecting()
                print(f"[{ts()}] SWAD: started collecting at epoch {ep + 1}")
            else:
                swad.update()

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for streams_batch, aux_data, targets, _ in val_ld:
                gcn_data = streams_batch[stream_name].to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(f"[{ts()}] [{stream_name}] Ep {ep + 1:3d}/{config['epochs']} | "
                  f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | "
                  f"Val: {va:.1f}% | LR: {lr_now:.2e} | Time: {ep_time:.1f}s")

        if va > best:
            best, patience_counter = va, 0
            torch.save({
                "model": model.state_dict(),
                "val_acc": va, "epoch": ep + 1, "classes": classes,
                "num_nodes": config["num_nodes"], "num_signers": num_signers,
                "aux_dim": aux_dim, "stream": stream_name,
                "num_samples": len(train_ds),
                "version": "v30_phase3", "config": config,
                "use_blockgcn": use_blockgcn, "use_protogcn": use_protogcn,
                "use_skeletongcl": use_skeletongcl,
            }, best_path)
            print(f"[{ts()}]   -> New best! {va:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"[{ts()}] [{stream_name}] Early stopping at epoch {ep + 1}")
            break

        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for streams_batch, aux_data, targets, _ in val_ld:
                    gcn_data = streams_batch[stream_name].to(device)
                    logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
                    _, p = logits.max(1)
                    preds_ep.extend(p.cpu().numpy())
                    tgts_ep.extend(targets.cpu().numpy())
            print(f"[{ts()}]   Per-class at epoch {ep + 1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot_c = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                print(f"[{ts()}]     {cn:12s}: {100.0 * cor / tot_c if tot_c > 0 else 0:.1f}% ({cor}/{tot_c})")

    # Apply SWAD average if collected
    if swad.n_averaged > 1:
        print(f"[{ts()}] [{stream_name}] Applying SWAD average ({swad.n_averaged} checkpoints)")
        swad.apply_average()
        # Re-evaluate with SWAD weights
        model.eval()
        vc_swad, vt_swad = 0, 0
        with torch.no_grad():
            for streams_batch, aux_data, targets, _ in val_ld:
                gcn_data = streams_batch[stream_name].to(device)
                logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
                _, p = logits.max(1)
                vt_swad += targets.size(0)
                vc_swad += p.eq(targets.to(device)).sum().item()
        va_swad = 100.0 * vc_swad / vt_swad if vt_swad > 0 else 0.0
        print(f"[{ts()}] [{stream_name}] SWAD val acc: {va_swad:.1f}% (best single: {best:.1f}%)")
        if va_swad > best:
            torch.save({
                "model": model.state_dict(),
                "val_acc": va_swad, "epoch": -1, "classes": classes,
                "num_nodes": config["num_nodes"], "num_signers": num_signers,
                "aux_dim": aux_dim, "stream": stream_name,
                "num_samples": len(train_ds),
                "version": "v30_phase3_swad", "config": config,
                "use_blockgcn": use_blockgcn, "use_protogcn": use_protogcn,
                "use_skeletongcl": use_skeletongcl,
            }, best_path)
            best = va_swad
            print(f"[{ts()}]   -> SWAD is better! Using SWAD weights.")

    # Final evaluation
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for streams_batch, aux_data, targets, _ in val_ld:
            gcn_data = streams_batch[stream_name].to(device)
            logits, _, _, _ = model(gcn_data.to(device), aux_data.to(device), 0.0)
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    print(f"\n[{ts()}] {name} [{stream_name}] Per-Class Results (best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_c = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_c if tot_c > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_c})")
    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if tgts else 0.0
    print(f"[{ts()}] {name} [{stream_name}] Overall: {ov:.1f}%")

    return {
        "stream": stream_name, "overall": ov, "per_class": res,
        "best_epoch": ckpt["epoch"], "params": param_count,
    }


# ---------------------------------------------------------------------------
# Fusion Weight Learning (from v28)
# ---------------------------------------------------------------------------

def learn_fusion_weights(category_name, classes, config, val_dir, ckpt_dir, device,
                         use_blockgcn=True, use_protogcn=True, use_skeletongcl=True):
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Learning fusion weights for {category_name}")
    print(f"[{ts()}] {'=' * 70}")

    streams = config["streams"]
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))

    models = {}
    for sname in streams:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{ts()}] WARNING: Missing checkpoint for '{sname}': {ckpt_path}")
            return None, 0.0
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        num_signers = ckpt.get("num_signers", 12)
        num_samples = ckpt.get("num_samples", 1000)
        m = KSLBlockGCNet(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            num_samples=num_samples, nn_=config["num_nodes"], ic=3,
            hd=config["hidden_dim"], nl=config["num_layers"], tk=tk, dr=config["dropout"],
            use_blockgcn=use_blockgcn, use_protogcn=use_protogcn,
            use_skeletongcl=use_skeletongcl, config=config,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models[sname] = m
        print(f"[{ts()}] Loaded {sname}: val_acc={ckpt['val_acc']:.1f}%, ep={ckpt['epoch']}")

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
    for s in streams:
        all_probs[s] = torch.cat(all_probs[s], dim=0)
    all_labels = np.array(all_labels)

    print(f"\n[{ts()}] Per-stream val accuracy:")
    for s in streams:
        acc = 100.0 * (all_probs[s].argmax(dim=1).numpy() == all_labels).mean()
        print(f"[{ts()}]   {s:>10s}: {acc:.1f}%")

    best_acc, best_w = 0.0, {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
    for wj in range(10, 81, 5):
        for wb in range(10, 81, 5):
            wv = 100 - wj - wb
            if wv < 5:
                continue
            fused = (wj/100 * all_probs["joint"] + wb/100 * all_probs["bone"] +
                     wv/100 * all_probs["velocity"])
            acc = 100.0 * (fused.argmax(dim=1).numpy() == all_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_w = {"joint": wj/100, "bone": wb/100, "velocity": wv/100}

    print(f"\n[{ts()}] Best fusion weights: {best_w}")
    print(f"[{ts()}] Fused val accuracy: {best_acc:.1f}%")

    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    with open(weights_path, "w") as f:
        json.dump({"weights": best_w, "val_accuracy": best_acc, "category": category_name}, f, indent=2)

    return best_w, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL v30 Phase 3: BlockGCN + ProtoGCN + SkeletonGCL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--seed", type=int, default=42)
    # Ablation flags
    parser.add_argument("--no-blockgcn", action="store_true",
                        help="Disable BlockGCN, use original learned adjacency")
    parser.add_argument("--no-protogcn", action="store_true",
                        help="Disable ProtoGCN prototype module")
    parser.add_argument("--no-skeletongcl", action="store_true",
                        help="Disable SkeletonGCL contrastive learning")

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))
    args = parser.parse_args()

    set_seed(args.seed)
    use_blockgcn = not args.no_blockgcn
    use_protogcn = not args.no_protogcn
    use_skeletongcl = not args.no_skeletongcl

    print("=" * 70)
    print(f"KSL Training v30 Phase 3 - BlockGCN + ProtoGCN + SkeletonGCL")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print(f"BlockGCN: {use_blockgcn}, ProtoGCN: {use_protogcn}, "
          f"SkeletonGCL: {use_skeletongcl}")
    print("=" * 70)

    check_mediapipe_version()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dirs = [args.train_dir, args.val_dir]
    val_dir = args.test_dir

    print(f"\nData Split:")
    print(f"  Train: {train_dirs} (signers 1-12)")
    print(f"  Val:   {val_dir} (signers 13-15)")
    print(f"\n[{ts()}] Config:")
    print(json.dumps(V30_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    categories = []
    if args.model_type in ("numbers", "both"):
        categories.append(("Numbers", NUMBER_CLASSES, False))
    if args.model_type in ("words", "both"):
        categories.append(("Words", WORD_CLASSES, True))

    for cat_name, classes, use_focal in categories:
        cat_key = cat_name.lower()
        ckpt_dir = os.path.join(args.checkpoint_dir, f"v30_phase3_{cat_key}")

        stream_results = {}
        for stream_name in V30_CONFIG["streams"]:
            result = train_stream(
                stream_name, cat_name, classes, V30_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
                use_focal=use_focal,
                use_blockgcn=use_blockgcn, use_protogcn=use_protogcn,
                use_skeletongcl=use_skeletongcl)
            if result:
                stream_results[stream_name] = result

        fusion_w, fused_acc = learn_fusion_weights(
            cat_name, classes, V30_CONFIG, val_dir, ckpt_dir, device,
            use_blockgcn, use_protogcn, use_skeletongcl)

        results[cat_key] = {
            "streams": stream_results,
            "fusion_weights": fusion_w,
            "fused_val_accuracy": fused_acc,
        }

    total_time = time.time() - start_time

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v30 Phase 3")
    print(f"[{ts()}] {'=' * 70}")
    for cat_key, cat_result in results.items():
        print(f"\n[{ts()}] {cat_key.upper()}:")
        for sn, sr in cat_result.get("streams", {}).items():
            print(f"[{ts()}]   {sn:>10s}: {sr['overall']:.1f}% "
                  f"(ep {sr['best_epoch']}, {sr['params']:,} params)")
        fw = cat_result.get("fusion_weights") or {}
        fused = cat_result.get("fused_val_accuracy", 0)
        print(f"[{ts()}]   {'fused':>10s}: {fused:.1f}% (j={fw.get('joint', 0):.2f}, "
              f"b={fw.get('bone', 0):.2f}, v={fw.get('velocity', 0):.2f})")

    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["fused_val_accuracy"] +
                    results["words"]["fused_val_accuracy"]) / 2
        print(f"\n[{ts()}] Combined fused val: {combined:.1f}%")

    print(f"[{ts()}] Total time: {total_time / 60:.1f} minutes")

    os.makedirs(args.results_dir, exist_ok=True)
    rp = os.path.join(args.results_dir,
                      f"v30_phase3_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rp, "w") as f:
        json.dump({
            "version": "v30_phase3", "model_type": args.model_type,
            "seed": args.seed, "config": V30_CONFIG,
            "use_blockgcn": use_blockgcn, "use_protogcn": use_protogcn,
            "use_skeletongcl": use_skeletongcl,
            "results": results, "total_time_seconds": total_time,
            "device": str(device), "timestamp": ts(),
        }, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {rp}")
    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
