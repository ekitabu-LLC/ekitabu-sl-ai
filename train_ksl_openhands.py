#!/usr/bin/env python3
"""
KSL Training — OpenHands DecoupledGCN Fine-Tuning

Fine-tunes OpenHands' DecoupledGCN (SL-GCN) pretrained on WLASL (American SL).
Uses 27-joint minimal MediaPipe skeleton with 2D coordinates.

Joint mapping (OpenHands 27 → MediaPipe Holistic → Our 48):
  OH 0:  nose (holistic 0)       → approx from our shoulders (midpoint above)
  OH 1:  l_eye_outer (holistic 2) → zero (we don't have face landmarks)
  OH 2:  r_eye_outer (holistic 5) → zero (we don't have face landmarks)
  OH 3:  l_shoulder (holistic 11) → our joint 42
  OH 4:  r_shoulder (holistic 12) → our joint 43
  OH 5:  l_elbow (holistic 13)    → our joint 44
  OH 6:  r_elbow (holistic 14)    → our joint 45
  OH 7:  lh_wrist (hand 0)        → our joint 0
  OH 8:  lh_thumb_tip (hand 4)    → our joint 4
  OH 9:  lh_index_mcp (hand 5)    → our joint 5
  OH 10: lh_index_tip (hand 8)    → our joint 8
  OH 11: lh_middle_mcp (hand 9)   → our joint 9
  OH 12: lh_middle_tip (hand 12)  → our joint 12
  OH 13: lh_ring_mcp (hand 13)    → our joint 13
  OH 14: lh_ring_tip (hand 16)    → our joint 16
  OH 15: lh_pinky_mcp (hand 17)   → our joint 17
  OH 16: lh_pinky_tip (hand 20)   → our joint 20
  OH 17: rh_wrist (hand 0)        → our joint 21
  OH 18: rh_thumb_tip (hand 4)    → our joint 25
  OH 19: rh_index_mcp (hand 5)    → our joint 26
  OH 20: rh_index_tip (hand 8)    → our joint 29
  OH 21: rh_middle_mcp (hand 9)   → our joint 30
  OH 22: rh_middle_tip (hand 12)  → our joint 33
  OH 23: rh_ring_mcp (hand 13)    → our joint 34
  OH 24: rh_ring_tip (hand 16)    → our joint 37
  OH 25: rh_pinky_mcp (hand 17)   → our joint 38
  OH 26: rh_pinky_tip (hand 20)   → our joint 41

Architecture: DecoupledGCN (10 GCN-TCN blocks) → 256-d embedding → FC classifier
Input: (B, 2, T, 27) — 2D coords, T frames, 27 joints

Usage:
    python train_ksl_openhands.py --model-type numbers
    python train_ksl_openhands.py --model-type words
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
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


NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

POSE_INDICES = [11, 12, 13, 14, 15, 16]


# ---------------------------------------------------------------------------
# OpenHands 27-joint mapping
# ---------------------------------------------------------------------------

# Our 48 joints → OpenHands 27 joints
# Maps: openhands_index → our_48_joint_index (or -1 for missing)
OH27_TO_OUR48 = {
    0: -1,   # nose → approximate
    1: -2,   # l_eye_outer → zero
    2: -2,   # r_eye_outer → zero
    3: 42,   # l_shoulder
    4: 43,   # r_shoulder
    5: 44,   # l_elbow
    6: 45,   # r_elbow
    # LH: wrist + thumb_tip + 4*(MCP, tip)
    7: 0,    # lh_wrist
    8: 4,    # lh_thumb_tip
    9: 5,    # lh_index_mcp
    10: 8,   # lh_index_tip
    11: 9,   # lh_middle_mcp
    12: 12,  # lh_middle_tip
    13: 13,  # lh_ring_mcp
    14: 16,  # lh_ring_tip
    15: 17,  # lh_pinky_mcp
    16: 20,  # lh_pinky_tip
    # RH: same pattern, offset by 21
    17: 21,  # rh_wrist
    18: 25,  # rh_thumb_tip
    19: 26,  # rh_index_mcp
    20: 29,  # rh_index_tip
    21: 30,  # rh_middle_mcp
    22: 33,  # rh_middle_tip
    23: 34,  # rh_ring_mcp
    24: 37,  # rh_ring_tip
    25: 38,  # rh_pinky_mcp
    26: 41,  # rh_pinky_tip
}


def adapt_48_to_27(h):
    """Convert our 48-joint (T, 48, 3) to OpenHands 27-joint (T, 27, 2).

    Returns x,y coordinates only (no z).
    """
    T = h.shape[0]
    out = np.zeros((T, 27, 2), dtype=np.float32)

    for oh_idx, our_idx in OH27_TO_OUR48.items():
        if our_idx == -1:
            # Nose: approximate from shoulders
            mid_shoulder = (h[:, 42, :2] + h[:, 43, :2]) / 2
            sw = np.linalg.norm(h[:, 42, :2] - h[:, 43, :2], axis=-1, keepdims=True)
            sw = np.maximum(sw, 1e-6)
            out[:, oh_idx, 0] = mid_shoulder[:, 0]
            out[:, oh_idx, 1] = mid_shoulder[:, 1] - 0.7 * sw[:, 0]
        elif our_idx == -2:
            # Face landmarks we don't have → zero
            pass
        else:
            out[:, oh_idx, :] = h[:, our_idx, :2]

    return out


# ---------------------------------------------------------------------------
# OpenHands DecoupledGCN Architecture (reconstructed from source)
# ---------------------------------------------------------------------------

# Graph definition from WLASL decoupled_gcn.yaml
OH_INWARD_EDGES = [
    [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6],
    [5, 7], [6, 17],
    [7, 8], [7, 9], [9, 10], [7, 11], [11, 12],
    [7, 13], [13, 14], [7, 15], [15, 16],
    [17, 18], [17, 19], [19, 20], [17, 21], [21, 22],
    [17, 23], [23, 24], [17, 25], [25, 26],
]


def build_oh_adjacency(num_nodes=27, inward_edges=None):
    """Build 3-part adjacency (inward, outward, self) from OpenHands graph definition."""
    if inward_edges is None:
        inward_edges = OH_INWARD_EDGES

    A_inward = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_outward = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_self = np.eye(num_nodes, dtype=np.float32)

    for src, tgt in inward_edges:
        A_inward[src, tgt] = 1
        A_outward[tgt, src] = 1

    # Normalize each adjacency
    def normalize(A):
        d = A.sum(axis=1)
        d_inv = np.where(d > 0, 1.0 / d, 0)
        return np.diag(d_inv) @ A

    A_in = normalize(A_inward + A_self)
    A_out = normalize(A_outward + A_self)
    A_s = normalize(A_self)

    # Stack: (3, V, V)
    return torch.FloatTensor(np.stack([A_in, A_out, A_s], axis=0))


class DecoupledGCNUnit(nn.Module):
    """Decoupled graph convolution with learnable adjacency."""
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4):
        super().__init__()
        self.num_subsets = A.shape[0]  # 3 (in, out, self)
        inter_channels = out_channels // coff_embedding
        self.inter_channels = inter_channels

        self.register_buffer('A', A)

        # Per-subset convolution
        self.conv_down = nn.ModuleList([
            nn.Conv2d(in_channels, inter_channels, 1) for _ in range(self.num_subsets)
        ])
        self.conv_up = nn.ModuleList([
            nn.Conv2d(inter_channels, out_channels, 1) for _ in range(self.num_subsets)
        ])

        # Learnable adjacency
        self.PA = nn.ParameterList([
            nn.Parameter(torch.zeros_like(A[i])) for i in range(self.num_subsets)
        ])

        # Group conv for efficient decoupled operation
        self.conv_group = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.shape

        # Decoupled group convolution
        out = self.conv_group(x)  # (N, out_ch*3, T, V)
        out_channels = out.shape[1] // self.num_subsets

        total = 0
        for i in range(self.num_subsets):
            A_i = self.A[i] + self.PA[i]

            # Attention-weighted adjacency
            z = self.conv_down[i](x)  # (N, inter_ch, T, V)
            z = z.mean(dim=2)  # (N, inter_ch, V) - average over time
            z = self.conv_up[i](z.unsqueeze(2)).squeeze(2)  # (N, out_ch, V)

            # Subset output
            subset_out = out[:, i*out_channels:(i+1)*out_channels]  # (N, out_ch, T, V)
            subset_out = torch.einsum('nctv,vw->nctw', subset_out, A_i)

            total = total + subset_out

        return self.bn(total)


class TCNUnit(nn.Module):
    """Temporal convolution unit."""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttention(nn.Module):
    """Spatial attention from DecoupledGCN."""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        # x: (N, C, T, V)
        se = self.fc(x).sigmoid()  # (N, 1, T, V)
        return x * se


class TemporalAttention(nn.Module):
    """Temporal attention."""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        se = self.fc(x).mean(dim=3, keepdim=True).sigmoid()  # (N, 1, T, 1)
        return x * se


class ChannelAttention(nn.Module):
    """Channel attention (squeeze-excitation)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(2).unsqueeze(3)
        return x * w


class DecoupledGCN_TCN_Unit(nn.Module):
    """Combined GCN + TCN block with attention."""
    def __init__(self, in_channels, out_channels, A, stride=1, groups=8, dropout=0.0):
        super().__init__()
        self.gcn = DecoupledGCNUnit(in_channels, out_channels, A, groups=groups)
        self.tcn = TCNUnit(out_channels, out_channels, kernel_size=9, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        # Attention
        self.att_s = SpatialAttention(out_channels)
        self.att_t = TemporalAttention(out_channels)
        self.att_c = ChannelAttention(out_channels)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.gcn(x))
        x = self.tcn(x)
        x = self.att_s(x)
        x = self.att_t(x)
        x = self.att_c(x)
        x = self.dropout(x)
        return self.relu(x + res)


class DecoupledGCN(nn.Module):
    """OpenHands DecoupledGCN encoder.

    10 GCN-TCN blocks: 64×4 → 128×3 → 256×3
    Input: (B, in_channels, T, V)
    Output: (B, n_out_features)
    """
    def __init__(self, in_channels=2, num_nodes=27, inward_edges=None,
                 groups=8, n_out_features=256):
        super().__init__()
        A = build_oh_adjacency(num_nodes, inward_edges)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 10 blocks
        self.layers = nn.ModuleList([
            DecoupledGCN_TCN_Unit(in_channels, 64, A, groups=min(groups, in_channels)),
            DecoupledGCN_TCN_Unit(64, 64, A, groups=groups),
            DecoupledGCN_TCN_Unit(64, 64, A, groups=groups),
            DecoupledGCN_TCN_Unit(64, 64, A, groups=groups),
            DecoupledGCN_TCN_Unit(64, 128, A, stride=2, groups=groups),
            DecoupledGCN_TCN_Unit(128, 128, A, groups=groups),
            DecoupledGCN_TCN_Unit(128, 128, A, groups=groups),
            DecoupledGCN_TCN_Unit(128, 256, A, stride=2, groups=groups),
            DecoupledGCN_TCN_Unit(256, 256, A, groups=groups),
            DecoupledGCN_TCN_Unit(256, n_out_features, A, groups=groups),
        ])

    def forward(self, x):
        """
        x: (B, C, T, V) where C=2 (x,y), V=27
        Returns: (B, n_out_features)
        """
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)  # (N, C, T, V)

        for layer in self.layers:
            x = layer(x)

        # Global average pool
        x = x.mean(dim=(2, 3))  # (N, n_out_features)
        return x


class OpenHandsClassifier(nn.Module):
    """DecoupledGCN encoder + classification head for KSL."""
    def __init__(self, num_classes, in_channels=2, num_nodes=27,
                 n_out_features=256, cls_dropout=0.3):
        super().__init__()
        self.encoder = DecoupledGCN(
            in_channels=in_channels,
            num_nodes=num_nodes,
            n_out_features=n_out_features,
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_out_features, 128),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 2, T, 27)
        Returns: logits (B, num_classes), embedding (B, 256)
        """
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OH_CONFIG = {
    "max_frames": 90,
    "num_nodes": 27,
    "in_channels": 2,
    "n_out_features": 256,
    "epochs": 200,
    "batch_size": 64,
    "encoder_lr": 1e-5,
    "head_lr": 1e-4,
    "min_lr": 1e-7,
    "weight_decay": 1e-3,
    "warmup_epochs": 10,
    "cls_dropout": 0.3,
    "label_smoothing": 0.1,
    # Augmentation
    "speed_aug_prob": 0.6,
    "speed_aug_min_frames": 130,
    "speed_aug_max_frames": 220,
    "rotation_prob": 0.5,
    "rotation_max_deg": 15.0,
    "shear_prob": 0.3,
    "shear_max": 0.2,
    "joint_dropout_prob": 0.2,
    "joint_dropout_rate": 0.08,
    "noise_std": 0.015,
    "noise_prob": 0.4,
    "hand_dropout_prob": 0.2,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.3,
    "complete_hand_drop_prob": 0.03,
    "temporal_warp_prob": 0.4,
    "temporal_warp_sigma": 0.2,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.3,
}


# ---------------------------------------------------------------------------
# Data loading (adapted from v37, simplified)
# ---------------------------------------------------------------------------

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


def get_file_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


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
        seen = {}
        for signer in signer_dict:
            gh = tuple(sorted(get_file_hash(p) for p in signer_dict[signer]))
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


def normalize_wrist_palm(h):
    """Normalize hands wrist-centric, body shoulder-centric."""
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
        sw = np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True)
        sw = np.maximum(sw, 1e-6)
        pose[pose_valid] = pose[pose_valid] / sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


# Augmentation functions
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
    T = data_list[0].shape[0]
    target_len = np.random.randint(min_frames, max_frames + 1)
    if target_len <= T:
        return data_list
    stretch_idx = np.clip(
        np.round(np.linspace(0, T - 1, target_len)).astype(int), 0, T - 1
    )
    return [arr[stretch_idx] for arr in data_list]


class SignerBalancedSampler(Sampler):
    def __init__(self, labels, signer_labels, batch_size, drop_last=True):
        self.labels = np.array(labels)
        self.signer_labels = np.array(signer_labels)
        self.batch_size = batch_size
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
# Dataset
# ---------------------------------------------------------------------------

class KSLOpenHandsDataset(Dataset):
    """Dataset producing OpenHands 27-joint 2D tensors."""

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

        print(f"[{ts()}]   Loaded {len(self.samples)} samples, {self.num_signers} signers")

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

        # Augmentations (on 48-joint format before conversion)
        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dr = np.random.uniform(self.config["hand_dropout_min"], self.config["hand_dropout_max"])
            if np.random.random() < 0.6:
                h[np.random.random(f) <= dr, :21, :] = 0
            if np.random.random() < 0.6:
                h[np.random.random(f) <= dr, 21:42, :] = 0

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

        h = normalize_wrist_palm(h)

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

        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            [h] = augment_temporal_warp([h], sigma=self.config["temporal_warp_sigma"])

        if self.aug and np.random.random() < self.config["speed_aug_prob"]:
            [h] = augment_temporal_speed(
                [h], self.config["speed_aug_min_frames"], self.config["speed_aug_max_frames"]
            )

        # Temporal resampling
        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
        else:
            pad = self.mf - f
            h = np.concatenate([h, np.zeros((pad, 48, 3), dtype=np.float32)])

        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z, h[:-shift]])
            elif shift < 0:
                z = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z])

        # Convert to 27-joint 2D
        h = np.clip(h, -10, 10)
        oh27 = adapt_48_to_27(h)  # (T, 27, 2)
        oh27 = np.clip(oh27, -10, 10)

        # Format: (C=2, T, V=27) matching DecoupledGCN input
        x = torch.FloatTensor(oh27).permute(2, 0, 1)  # (2, T, 27)

        return x, self.labels[idx], self.signer_labels[idx]


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------

def load_openhands_pretrained(model, checkpoint_path, device):
    """Load OpenHands pretrained DecoupledGCN weights.

    OpenHands checkpoint is a zip containing a .ckpt (PyTorch Lightning format).
    """
    print(f"[{ts()}] Loading pretrained OpenHands checkpoint from {checkpoint_path}")

    if checkpoint_path.endswith('.zip'):
        import zipfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(checkpoint_path, 'r') as z:
                z.extractall(tmpdir)
            # Find .ckpt file
            ckpt_files = []
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.ckpt'):
                        ckpt_files.append(os.path.join(root, f))
            if not ckpt_files:
                print(f"[{ts()}] ERROR: No .ckpt file found in zip")
                return 0
            ckpt_path = ckpt_files[0]
            print(f"[{ts()}]   Found: {os.path.basename(ckpt_path)}")
            # Shim: checkpoint was saved with old 'pytorch_lightning' package name
            import sys
            if 'pytorch_lightning' not in sys.modules:
                try:
                    import lightning as _pl_compat
                    sys.modules['pytorch_lightning'] = _pl_compat
                    sys.modules['pytorch_lightning.trainer'] = _pl_compat.trainer
                except Exception:
                    pass
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # PyTorch Lightning format
    if "state_dict" in ckpt:
        pretrained = ckpt["state_dict"]
    elif "model" in ckpt:
        pretrained = ckpt["model"]
    else:
        pretrained = ckpt

    our_state = model.state_dict()
    matched, skipped = 0, 0

    # Print prefixes
    prefixes = set()
    for k in pretrained.keys():
        parts = k.split(".")
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
    print(f"[{ts()}]   Pretrained prefixes: {sorted(prefixes)[:15]}")

    new_state = {}
    for pk, pt in pretrained.items():
        # OpenHands uses model.encoder.* in Lightning
        # Try various prefix mappings
        candidates = [pk]
        if pk.startswith("model.encoder."):
            candidates.append("encoder." + pk[len("model.encoder."):])
        if pk.startswith("model."):
            candidates.append(pk[len("model."):])
        if pk.startswith("encoder."):
            candidates.append(pk)

        mapped = False
        for candidate in candidates:
            if candidate in our_state and pt.shape == our_state[candidate].shape:
                new_state[candidate] = pt
                matched += 1
                mapped = True
                break

        if not mapped:
            skipped += 1

    missing = sum(1 for k in our_state if k not in new_state)
    print(f"[{ts()}]   Loaded {matched}, skipped {skipped}, {missing} random init")

    model.load_state_dict(new_state, strict=False)
    return matched


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(name, classes, config, train_dirs, val_dir, ckpt_dir,
                device, pretrained_path=None):
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training OpenHands - {name} ({len(classes)} classes)")
    print(f"[{ts()}] {'=' * 70}")

    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = KSLOpenHandsDataset(train_dirs, classes, config, aug=True)
    val_ds = KSLOpenHandsDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples")
        return None

    i2c = {v: k for k, v in train_ds.c2i.items()}

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )
    train_ld = DataLoader(train_ds, batch_size=config["batch_size"],
                          sampler=train_sampler, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True)

    model = OpenHandsClassifier(
        num_classes=len(classes),
        in_channels=config["in_channels"],
        num_nodes=config["num_nodes"],
        n_out_features=config["n_out_features"],
        cls_dropout=config["cls_dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")

    # Load pretrained
    encoder_lr = config["encoder_lr"]
    head_lr = config["head_lr"]
    if pretrained_path and os.path.exists(pretrained_path):
        matched = load_openhands_pretrained(model, pretrained_path, device)
        if matched > 0:
            print(f"[{ts()}] Pretrained encoder loaded, differential LR")
        else:
            encoder_lr = head_lr
            print(f"[{ts()}] No weights matched, uniform LR={head_lr}")
    else:
        encoder_lr = head_lr
        print(f"[{ts()}] No pretrained checkpoint, uniform LR={head_lr}")

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )

    ls = config["label_smoothing"]
    warmup_epochs = config["warmup_epochs"]
    cutmix_prob = config["cutmix_prob"]
    cutmix_alpha = config["cutmix_alpha"]
    mixup_alpha = config["mixup_alpha"]

    best_path = os.path.join(ckpt_dir, "best_model.pt")
    best = 0.0

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        if ep < warmup_epochs:
            wf = (ep + 1) / warmup_epochs
            opt.param_groups[0]['lr'] = encoder_lr * wf
            opt.param_groups[1]['lr'] = head_lr * wf

        for x, targets, _ in train_ld:
            x = x.to(device)
            targets = targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            if use_cutmix:
                B, C, T, V = x.shape
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                cut_len = int(T * (1 - lam))
                if cut_len > 0:
                    cut_start = np.random.randint(0, T - cut_len + 1)
                    perm = torch.randperm(B, device=x.device)
                    x_m = x.clone()
                    x_m[:, :, cut_start:cut_start+cut_len] = x[perm, :, cut_start:cut_start+cut_len]
                    lam = 1.0 - cut_len / T
                    logits, _ = model(x_m)
                    loss = lam * F.cross_entropy(logits, targets, label_smoothing=ls) + \
                           (1 - lam) * F.cross_entropy(logits, targets[perm], label_smoothing=ls)
                    _, p = logits.max(1)
                    tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(targets[perm]).float()).sum().item()
                else:
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits, targets, label_smoothing=ls)
                    _, p = logits.max(1)
                    tc += p.eq(targets).sum().item()
            elif mixup_alpha > 0 and np.random.random() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(x.size(0), device=x.device)
                logits, _ = model(lam * x + (1 - lam) * x[perm])
                loss = lam * F.cross_entropy(logits, targets, label_smoothing=ls) + \
                       (1 - lam) * F.cross_entropy(logits, targets[perm], label_smoothing=ls)
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(targets[perm]).float()).sum().item()
            else:
                logits, _ = model(x)
                loss = F.cross_entropy(logits, targets, label_smoothing=ls)
                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

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
            for x, targets, _ in val_ld:
                logits, _ = model(x.to(device))
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets.to(device)).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(f"[{ts()}] Ep {ep+1:3d}/{config['epochs']} | "
                  f"Loss: {tl/max(len(train_ld),1):.4f} | Train: {ta:.1f}% | "
                  f"Val: {va:.1f}% | Time: {ep_time:.1f}s")

        if va > best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "val_acc": va,
                "epoch": ep + 1,
                "classes": classes,
                "version": "openhands",
                "config": config,
            }, best_path)
            print(f"[{ts()}]   -> New best! {va:.1f}%")

        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for x, targets, _ in val_ld:
                    logits, _ = model(x.to(device))
                    preds_ep.extend(logits.max(1)[1].cpu().numpy())
                    tgts_ep.extend(targets.numpy())
            print(f"[{ts()}]   Per-class at epoch {ep+1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                print(f"[{ts()}]     {cn:12s}: {100.0*cor/tot if tot>0 else 0:.1f}%")

    # Final
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, targets, _ in val_ld:
            logits, _ = model(x.to(device))
            preds.extend(logits.max(1)[1].cpu().numpy())
            tgts.extend(targets.numpy())

    print(f"\n[{ts()}] {name} Per-Class (best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot if tot > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot})")
    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if tgts else 0.0
    print(f"[{ts()}] {name} Overall: {ov:.1f}%")

    return {"overall": ov, "per_class": res, "best_epoch": ckpt["epoch"], "params": param_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KSL OpenHands Fine-Tuning")
    parser.add_argument("--model-type", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--seed", type=int, default=42)

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"))
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(base_data, "checkpoints", "openhands"))
    parser.add_argument("--pretrained", type=str,
                        default=os.path.join(base_data, "checkpoints", "openhands_pretrained", "wlasl_slgcn.zip"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("KSL OpenHands DecoupledGCN Fine-Tuning")
    print(f"Started: {ts()}")
    print(f"Pretrained: {args.pretrained}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dirs = [args.train_dir, args.val_dir]
    val_dir = args.test_dir

    print(f"\nData Split:")
    print(f"  Train: {train_dirs}")
    print(f"  Val:   {val_dir}")
    print(f"\nConfig:")
    print(json.dumps(OH_CONFIG, indent=2, default=str))

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
        result = train_model(cat_name, classes, OH_CONFIG, train_dirs, val_dir,
                            ckpt_dir, device, args.pretrained)
        if result:
            results[cat_key] = result

    total_time = time.time() - start_time

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY - OpenHands DecoupledGCN")
    print(f"[{ts()}] {'=' * 70}")
    for cat_key, result in results.items():
        print(f"  {cat_key.upper()}: {result['overall']:.1f}% "
              f"(ep {result['best_epoch']}, {result['params']:,} params)")
    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
        print(f"  Combined: {combined:.1f}%")
    print(f"  Total time: {total_time / 60:.1f} minutes")

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"openhands_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(results_path, "w") as f:
        json.dump({
            "version": "openhands",
            "model_type": args.model_type,
            "seed": args.seed,
            "config": OH_CONFIG,
            "results": results,
            "total_time_seconds": total_time,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": ts(),
            "pretrained": args.pretrained,
        }, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")


if __name__ == "__main__":
    main()
