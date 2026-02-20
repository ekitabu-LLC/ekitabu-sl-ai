#!/usr/bin/env python3
"""
KSL Training — MASA Fine-Tuning

Fine-tunes MASA (Motion-Aware Masked Autoencoder) pretrained encoder on KSL data.
MASA was pretrained on 161K sign language samples (WLASL+MSASL+NMFs-CSL+SLR500).

Architecture:
  - Encoder: 3 ST-GCN branches (right hand, left hand, body) -> 512 each -> 1536
  - Transformer: 3 blocks, 8 heads, d_ff=2048 -> 1536
  - Classification head: 1536 -> num_classes (replaces original ProjectHead)

Joint Adapter (MediaPipe 48 -> MASA 49):
  - LH: our joints 0-20 -> MASA lh (21 joints, x,y only)
  - RH: our joints 21-41 -> MASA rh (21 joints, x,y only)
  - Body: our 6 joints -> MASA 7 joints (nose=zero, rest mapped)
    MASA body: [nose, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist]
    Our body:  [l_shoulder(42), r_shoulder(43), l_elbow(44), r_elbow(45), l_wrist(46), r_wrist(47)]

Fine-tuning strategy:
  - Differential LR: encoder 1e-5, classification head 1e-4
  - GroupNorm classification head (no AdaBN needed)
  - 200 epochs, cosine schedule
  - Same augmentation pipeline as v37

Usage:
    python train_ksl_masa.py --model-type numbers
    python train_ksl_masa.py --model-type words
"""

import argparse
import copy
import hashlib
import json
import math
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

# Body joint indices from MediaPipe
POSE_INDICES = [11, 12, 13, 14, 15, 16]

# ---------------------------------------------------------------------------
# MASA Graph Definitions (reconstructed from source)
# ---------------------------------------------------------------------------

def build_masa_stb_graph():
    """Build the hand (STB) graph adjacency for MASA's ST-GCN.
    21 joints: wrist=0, then 4 joints per finger (thumb, index, middle, ring, little).
    """
    num_node = 21
    # 1-indexed edges from MASA source, converted to 0-indexed
    neighbor_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # little
    ]
    symmetry_edges = [
        (17, 13), (13, 9), (9, 5), (5, 1),
        (18, 14), (14, 10), (10, 6), (6, 2),
        (19, 15), (15, 11), (11, 7), (7, 3),
        (20, 16), (16, 12), (12, 8), (8, 4),
    ]
    # Build adjacency
    adj = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in neighbor_edges + symmetry_edges:
        adj[i, j] = 1
        adj[j, i] = 1
    adj += np.eye(num_node, dtype=np.float32)
    # Normalize
    d = adj.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    return torch.FloatTensor(np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt))


def build_masa_body_graph():
    """Build the body graph adjacency for MASA's ST-GCN.
    7 joints: nose=0, l_shoulder=1, l_elbow=2, l_wrist=3,
              r_shoulder=4, r_elbow=5, r_wrist=6
    """
    num_node = 7
    # 1-indexed edges converted to 0-indexed
    neighbor_edges = [
        (0, 1), (1, 2), (2, 3),  # nose -> l_shoulder -> l_elbow -> l_wrist
        (0, 4), (4, 5), (5, 6),  # nose -> r_shoulder -> r_elbow -> r_wrist
    ]
    symmetry_edges = [
        (1, 4), (2, 5), (3, 6),  # left-right symmetry
    ]
    adj = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in neighbor_edges + symmetry_edges:
        adj[i, j] = 1
        adj[j, i] = 1
    adj += np.eye(num_node, dtype=np.float32)
    d = adj.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    return torch.FloatTensor(np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt))


# Part definitions for graph pooling (from MASA source)
STB_PART = {
    "thumb_finger": [2, 3, 4],
    "index_finger": [6, 7, 8],
    "mid_finger": [10, 11, 12],
    "ring_finger": [14, 15, 16],
    "little_finger": [18, 19, 20],
    "cb": [0, 1, 5, 9, 13, 17],
}

BODY_PART = {
    "nose": [0],
    "left_arm": [1, 2, 3],
    "right_arm": [4, 5, 6],
}


# ---------------------------------------------------------------------------
# MASA ST-GCN Encoder (reconstructed from source)
# ---------------------------------------------------------------------------

class ConvTemporalGraphical(nn.Module):
    """Graph convolution for single-frame processing."""
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, 1, V)
        x = torch.matmul(x.squeeze(2), self.adj).unsqueeze(2)  # (B, C, 1, V)
        return self.conv(x)


class STGCNBlock_MASA(nn.Module):
    """Single ST-GCN block from MASA."""
    def __init__(self, in_channels, out_channels, adj, stride=1, dropout=0.05, residual=True):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, adj)
        self.bn_gcn = nn.BatchNorm2d(out_channels, momentum=0.1)

        # TCN branch
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=0.1),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn_gcn(self.gcn(x)))
        x = self.dropout(x)
        x = self.tcn(x)
        return self.relu(x + res)


class TemporalConvBlock(nn.Module):
    """Temporal conv block with parallel branches (from MASA)."""
    def __init__(self, num_inputs, num_outputs, dropout=0.25):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(num_inputs, num_outputs, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_outputs, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(num_inputs, num_outputs, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_outputs, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_outputs, num_outputs, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_outputs, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Merge two branches back to num_outputs
        self.merge = nn.Sequential(
            nn.Conv1d(num_outputs * 2, num_outputs, kernel_size=1),
            nn.BatchNorm1d(num_outputs, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv1d(num_inputs, num_outputs, kernel_size=1),
            nn.BatchNorm1d(num_outputs, momentum=0.1),
        ) if num_inputs != num_outputs else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = self.merge(torch.cat([b1, b2], dim=1))
        return self.relu(out + res)


class MASASTGCNEncoder(nn.Module):
    """ST-GCN encoder for one body part (hand or body) from MASA.

    Processes single-frame graph convolutions followed by temporal convolutions.
    Input: (B, T, V, 2) where V=21 (hand) or V=7 (body)
    Output: (B, T, 512)
    """
    def __init__(self, adj, part_dict, in_channels=2, gcn_channels=[32, 64, 128],
                 tcn_channels=[256, 512], dropout_gcn=0.05, dropout_tcn=0.25):
        super().__init__()
        num_node = adj.shape[0]
        self.in_channels = in_channels

        # Data normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_node, momentum=0.1)

        # ST-GCN layers (process each frame independently, then pool joints)
        self.gcn_layers = nn.ModuleList()
        ch = in_channels
        for out_ch in gcn_channels:
            self.gcn_layers.append(STGCNBlock_MASA(ch, out_ch, adj, dropout=dropout_gcn))
            ch = out_ch

        # Graph pooling: average joints within each part
        self.part_indices = []
        for part_name in sorted(part_dict.keys()):
            self.part_indices.append(part_dict[part_name])
        num_parts = len(self.part_indices)

        # After pooling: (B*T, gcn_channels[-1], 1, num_parts) -> flatten to (B, T, gcn_channels[-1]*num_parts)
        tcn_input_dim = gcn_channels[-1] * num_parts

        # Temporal convolution blocks
        self.tcn_layers = nn.ModuleList()
        ch = tcn_input_dim
        for out_ch in tcn_channels:
            self.tcn_layers.append(TemporalConvBlock(ch, out_ch, dropout=dropout_tcn))
            ch = out_ch

        self.output_dim = tcn_channels[-1]

    def forward(self, x):
        """
        x: (B, T, V, 2)
        Returns: (B, T, 512)
        """
        B, T, V, C = x.shape

        # Data normalization: (B*T, V*C) -> BN -> reshape
        x_flat = x.reshape(B * T, V * C)
        x_flat = self.data_bn(x_flat)
        x_flat = x_flat.reshape(B * T, 1, V, C).permute(0, 3, 1, 2)  # (B*T, C, 1, V)

        # GCN layers (per-frame)
        for gcn in self.gcn_layers:
            x_flat = gcn(x_flat)  # (B*T, ch, 1, V)

        # Graph pooling: average over each part
        pooled = []
        for indices in self.part_indices:
            pooled.append(x_flat[:, :, :, indices].mean(dim=3, keepdim=True))
        x_pooled = torch.cat(pooled, dim=3)  # (B*T, ch, 1, num_parts)

        # Reshape for temporal processing: (B, T, ch*num_parts)
        ch = x_pooled.shape[1]
        num_parts = x_pooled.shape[3]
        x_temporal = x_pooled.reshape(B, T, ch * num_parts).permute(0, 2, 1)  # (B, ch*num_parts, T)

        # TCN layers
        for tcn in self.tcn_layers:
            x_temporal = tcn(x_temporal)  # (B, out_ch, T)

        return x_temporal.permute(0, 2, 1)  # (B, T, 512)


# ---------------------------------------------------------------------------
# MASA Transformer
# ---------------------------------------------------------------------------

class PositionEncoding(nn.Module):
    """Sinusoidal position encoding from MASA."""
    def __init__(self, d_model=1536, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MASATransformerBlock(nn.Module):
    """Single transformer block from MASA."""
    def __init__(self, d_model=1536, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class MASAEmbed(nn.Module):
    """MASA Embed module: 3 ST-GCN encoders -> 1536-d with position encoding."""
    def __init__(self, d_model=1536, dropout=0.1):
        super().__init__()
        hand_adj = build_masa_stb_graph()
        body_adj = build_masa_body_graph()

        # Shared hand encoder for both left and right hands
        self.st_gcn_hand = MASASTGCNEncoder(
            adj=hand_adj, part_dict=STB_PART, in_channels=2,
            gcn_channels=[32, 64, 128], tcn_channels=[256, 512],
        )
        self.st_gcn_body = MASASTGCNEncoder(
            adj=body_adj, part_dict=BODY_PART, in_channels=2,
            gcn_channels=[32, 64, 128], tcn_channels=[256, 512],
        )
        self.pe = PositionEncoding(d_model=d_model, dropout=dropout)

    def forward(self, rh, lh, body):
        """
        rh: (B, T, 21, 2)
        lh: (B, T, 21, 2)
        body: (B, T, 7, 2)
        Returns: (B, T, 1536)
        """
        rh_feat = self.st_gcn_hand(rh)      # (B, T, 512)
        lh_feat = self.st_gcn_hand(lh)      # (B, T, 512)
        body_feat = self.st_gcn_body(body)   # (B, T, 512)
        pose_feat = torch.cat([rh_feat, lh_feat, body_feat], dim=2)  # (B, T, 1536)
        return self.pe(pose_feat)


class MASAEncoder(nn.Module):
    """Full MASA encoder: Embed + Transformer blocks."""
    def __init__(self, d_model=1536, nhead=8, d_ff=2048, num_blocks=3, dropout=0.1):
        super().__init__()
        self.embed = MASAEmbed(d_model=d_model, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([
            MASATransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_blocks)
        ])
        self.d_model = d_model

    def forward(self, rh, lh, body):
        """Returns: (B, T, 1536)"""
        x = self.embed(rh, lh, body)
        for block in self.transformer_blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Joint Adapter: MediaPipe 48 -> MASA 49
# ---------------------------------------------------------------------------

# Mapping: our body joints -> MASA body joints
# Our body (48-joint system): [l_shoulder(42), r_shoulder(43), l_elbow(44),
#                               r_elbow(45), l_wrist(46), r_wrist(47)]
# MASA body: [nose(0), l_shoulder(1), l_elbow(2), l_wrist(3),
#             r_shoulder(4), r_elbow(5), r_wrist(6)]
BODY_MAP = {
    # masa_idx: our_joint_idx
    1: 42,  # l_shoulder
    2: 44,  # l_elbow
    3: 46,  # l_wrist
    4: 43,  # r_shoulder
    5: 45,  # r_elbow
    6: 47,  # r_wrist
    # 0 (nose): not available, stays zero
}


def adapt_joints_to_masa(h):
    """Convert MediaPipe 48-joint array to MASA's 3-part dict format.

    Args:
        h: numpy array (T, 48, 3) - MediaPipe joints

    Returns:
        rh: (T, 21, 2) - right hand x,y
        lh: (T, 21, 2) - left hand x,y
        body: (T, 7, 2) - body x,y
    """
    T = h.shape[0]

    # Left hand: joints 0-20, take x,y only
    lh = h[:, :21, :2].copy()

    # Right hand: joints 21-41, take x,y only
    rh = h[:, 21:42, :2].copy()

    # Body: 7 joints, map from our 6
    body = np.zeros((T, 7, 2), dtype=np.float32)
    for masa_idx, our_idx in BODY_MAP.items():
        body[:, masa_idx, :] = h[:, our_idx, :2]
    # nose (index 0): approximate as midpoint above shoulders
    mid_shoulder = (h[:, 42, :2] + h[:, 43, :2]) / 2
    shoulder_width = np.linalg.norm(h[:, 42, :2] - h[:, 43, :2], axis=-1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    # Nose is roughly 0.7 * shoulder_width above the midpoint
    body[:, 0, 0] = mid_shoulder[:, 0]  # same x
    body[:, 0, 1] = mid_shoulder[:, 1] - 0.7 * shoulder_width[:, 0]  # above (y decreases upward)

    return rh, lh, body


# ---------------------------------------------------------------------------
# MASA KSL Model (for fine-tuning)
# ---------------------------------------------------------------------------

class MASAClassifier(nn.Module):
    """MASA encoder + classification head for KSL.

    Replaces MASA's ProjectHead with a simple GroupNorm-based classifier.
    """
    def __init__(self, num_classes, d_model=1536, nhead=8, d_ff=2048,
                 num_blocks=3, dropout=0.1, cls_dropout=0.3):
        super().__init__()
        self.encoder = MASAEncoder(d_model, nhead, d_ff, num_blocks, dropout)

        # Classification head with GroupNorm (signer-invariant)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(512, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, rh, lh, body):
        """
        rh: (B, T, 21, 2)
        lh: (B, T, 21, 2)
        body: (B, T, 7, 2)
        Returns: logits (B, num_classes), embedding (B, 1536)
        """
        # Encoder: (B, T, 1536)
        features = self.encoder(rh, lh, body)

        # Temporal average pooling
        embedding = features.mean(dim=1)  # (B, 1536)

        # Classification
        logits = self.classifier(embedding)

        return logits, embedding


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MASA_CONFIG = {
    "max_frames": 90,
    "epochs": 200,
    "batch_size": 64,
    # Differential learning rates
    "encoder_lr": 1e-5,
    "head_lr": 1e-4,
    "min_lr": 1e-7,
    "weight_decay": 1e-3,
    "warmup_epochs": 10,
    "dropout": 0.1,
    "cls_dropout": 0.3,
    "label_smoothing": 0.1,
    # Augmentation (same as v37)
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
# Data Loading (adapted from v37)
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


# Augmentation functions (from v37)
LH_CHAINS = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
RH_CHAINS = [[(p + 21, c + 21) for p, c in chain] for chain in LH_CHAINS]


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
    # 2D rotation only (x,y)
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
# Dataset
# ---------------------------------------------------------------------------

class KSLMASADataset(Dataset):
    """Dataset that outputs MASA-formatted tensors (rh, lh, body as 2D coords)."""

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
        print(f"[{ts()}]   Signers: {self.num_signers}")

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

        # Complete hand drop
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # Hand swap
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # Normalize
        h = normalize_wrist_palm(h)

        # Spatial augmentations
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

        # Temporal warp
        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            [h] = augment_temporal_warp([h], sigma=self.config["temporal_warp_sigma"])

        # Speed augmentation
        if self.aug and np.random.random() < self.config["speed_aug_prob"]:
            [h] = augment_temporal_speed(
                [h], self.config["speed_aug_min_frames"], self.config["speed_aug_max_frames"]
            )

        # Temporal resampling to max_frames
        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
        else:
            pad = self.mf - f
            h = np.concatenate([h, np.zeros((pad, 48, 3), dtype=np.float32)])

        # Temporal shift
        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z, h[:-shift]])
            elif shift < 0:
                z = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z])

        # Convert to MASA format: 3 separate tensors (rh, lh, body) with 2D coords
        h = np.clip(h, -10, 10)
        rh, lh, body = adapt_joints_to_masa(h)

        return (
            torch.FloatTensor(rh),
            torch.FloatTensor(lh),
            torch.FloatTensor(body),
            self.labels[idx],
            self.signer_labels[idx],
        )


def masa_collate_fn(batch):
    rh_list, lh_list, body_list, labels_list, signer_list = zip(*batch)
    return (
        torch.stack(rh_list),
        torch.stack(lh_list),
        torch.stack(body_list),
        torch.tensor(labels_list, dtype=torch.long),
        torch.tensor(signer_list, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Temporal CutMix (adapted for MASA format)
# ---------------------------------------------------------------------------

def temporal_cutmix_masa(rh, lh, body, labels, alpha=1.0):
    B, T = rh.shape[0], rh.shape[1]
    lam = np.random.beta(alpha, alpha)
    cut_len = int(T * (1 - lam))
    if cut_len == 0:
        return rh, lh, body, labels, labels, 1.0
    cut_start = np.random.randint(0, T - cut_len + 1)
    indices = torch.randperm(B, device=rh.device)
    rh_m = rh.clone()
    lh_m = lh.clone()
    body_m = body.clone()
    rh_m[:, cut_start:cut_start+cut_len] = rh[indices, cut_start:cut_start+cut_len]
    lh_m[:, cut_start:cut_start+cut_len] = lh[indices, cut_start:cut_start+cut_len]
    body_m[:, cut_start:cut_start+cut_len] = body[indices, cut_start:cut_start+cut_len]
    lam = 1.0 - cut_len / T
    return rh_m, lh_m, body_m, labels, labels[indices], lam


# ---------------------------------------------------------------------------
# Pretrained Checkpoint Loading
# ---------------------------------------------------------------------------

def load_masa_pretrained(model, checkpoint_path, device):
    """Load MASA pretrained weights, mapping from their state dict to ours.

    The pretrained model has keys like:
      - encoder_q.embed.st_gcn_hand.* -> our encoder.embed.st_gcn_hand.*
      - encoder_q.embed.st_gcn_body.* -> our encoder.embed.st_gcn_body.*
      - encoder_q.embed.pe.* -> our encoder.embed.pe.*
      - encoder_q.transformer_encoder.layers.N.* -> our encoder.transformer_blocks.N.*

    We skip: encoder_k.*, queue, queue_ptr, project_head.*, decoder.*
    """
    print(f"[{ts()}] Loading pretrained MASA checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint may have different structures
    if "state_dict" in ckpt:
        pretrained = ckpt["state_dict"]
    elif "model" in ckpt:
        pretrained = ckpt["model"]
    else:
        pretrained = ckpt

    # Try to map pretrained keys to our model
    our_state = model.state_dict()
    matched, skipped, missing = 0, 0, 0

    # Print available prefixes in pretrained
    prefixes = set()
    for k in pretrained.keys():
        parts = k.split(".")
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
    print(f"[{ts()}] Pretrained key prefixes: {sorted(prefixes)[:20]}")
    print(f"[{ts()}] Total pretrained keys: {len(pretrained)}")
    print(f"[{ts()}] Our model keys: {len(our_state)}")

    # Build mapping: pretrained -> our keys
    key_mapping = {}

    for pretrained_key in pretrained.keys():
        # Try direct match first
        if pretrained_key in our_state:
            key_mapping[pretrained_key] = pretrained_key
            continue

        # Try stripping 'encoder_q.' prefix (MoCo style)
        if pretrained_key.startswith("encoder_q."):
            stripped = pretrained_key[len("encoder_q."):]
            # Map to our encoder.*
            our_key = f"encoder.{stripped}"
            if our_key in our_state:
                key_mapping[pretrained_key] = our_key
                continue

            # Try transformer mapping:
            # encoder_q.transformer_encoder.layers.N.* -> encoder.transformer_blocks.N.*
            if "transformer_encoder.layers" in stripped:
                our_key = our_key.replace("transformer_encoder.layers", "transformer_blocks")
                # Map self_attn_layer_norm -> norm1, final_layer_norm -> norm2
                our_key = our_key.replace("self_attn_layer_norm", "norm1")
                our_key = our_key.replace("final_layer_norm", "norm2")
                # Map fc1/fc2 -> ffn.0/ffn.3
                our_key = our_key.replace(".fc1.", ".ffn.0.")
                our_key = our_key.replace(".fc2.", ".ffn.3.")
                if our_key in our_state:
                    key_mapping[pretrained_key] = our_key
                    continue

    # Apply mapping
    new_state = {}
    for pretrained_key, our_key in key_mapping.items():
        pt_tensor = pretrained[pretrained_key]
        our_tensor = our_state[our_key]
        if pt_tensor.shape == our_tensor.shape:
            new_state[our_key] = pt_tensor
            matched += 1
        else:
            print(f"[{ts()}]   Shape mismatch: {pretrained_key} {pt_tensor.shape} "
                  f"vs {our_key} {our_tensor.shape}")
            skipped += 1

    # Count unmapped keys
    mapped_our_keys = set(new_state.keys())
    for k in our_state:
        if k not in mapped_our_keys:
            missing += 1

    print(f"[{ts()}] Loaded {matched} pretrained parameters, "
          f"skipped {skipped} (shape mismatch), "
          f"{missing} initialized randomly")

    # Load matched weights
    model.load_state_dict(new_state, strict=False)
    return matched


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(name, classes, config, train_dirs, val_dir, ckpt_dir,
                device, pretrained_path=None, use_focal=False):
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training MASA - {name} ({len(classes)} classes)")
    print(f"[{ts()}] {'=' * 70}")

    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = KSLMASADataset(train_dirs, classes, config, aug=True)
    val_ds = KSLMASADataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found")
        return None

    num_signers = train_ds.num_signers
    i2c = {v: k for k, v in train_ds.c2i.items()}

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )
    train_ld = DataLoader(train_ds, batch_size=config["batch_size"],
                          sampler=train_sampler, num_workers=2, pin_memory=True,
                          collate_fn=masa_collate_fn)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=masa_collate_fn)

    model = MASAClassifier(
        num_classes=len(classes),
        d_model=1536, nhead=8, d_ff=2048,
        num_blocks=3, dropout=config["dropout"],
        cls_dropout=config["cls_dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Total parameters: {param_count:,}")

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        matched = load_masa_pretrained(model, pretrained_path, device)
        if matched > 0:
            print(f"[{ts()}] Pretrained MASA encoder loaded successfully")
        else:
            print(f"[{ts()}] WARNING: No pretrained weights matched, training from scratch")
    else:
        print(f"[{ts()}] No pretrained checkpoint found, training from scratch")

    # Differential learning rates
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.temporal_pool.parameters()) + list(model.classifier.parameters())
    encoder_lr = config["encoder_lr"]
    head_lr = config["head_lr"]

    # If no pretrained weights, use same LR for both
    if not pretrained_path or not os.path.exists(pretrained_path):
        encoder_lr = head_lr
        print(f"[{ts()}] No pretrained weights: using uniform LR={head_lr}")
    else:
        print(f"[{ts()}] Differential LR: encoder={encoder_lr}, head={head_lr}")

    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )

    ls = config["label_smoothing"]
    cutmix_prob = config["cutmix_prob"]
    cutmix_alpha = config["cutmix_alpha"]
    mixup_alpha = config["mixup_alpha"]
    warmup_epochs = config["warmup_epochs"]

    best_path = os.path.join(ckpt_dir, "best_model.pt")
    best = 0.0

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            # param_groups[0] = encoder, param_groups[1] = head
            opt.param_groups[0]['lr'] = encoder_lr * warmup_factor
            opt.param_groups[1]['lr'] = head_lr * warmup_factor

        for rh, lh, body, targets, signer_targets in train_ld:
            rh = rh.to(device)
            lh = lh.to(device)
            body = body.to(device)
            targets = targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob

            if use_cutmix:
                rh_m, lh_m, body_m, ta, tb, lam = temporal_cutmix_masa(
                    rh, lh, body, targets, cutmix_alpha
                )
                logits, _ = model(rh_m, lh_m, body_m)
                loss = (lam * F.cross_entropy(logits, ta, label_smoothing=ls)
                        + (1 - lam) * F.cross_entropy(logits, tb, label_smoothing=ls))
                _, p = logits.max(1)
                tc += (lam * p.eq(ta).float() + (1 - lam) * p.eq(tb).float()).sum().item()

            elif mixup_alpha > 0 and np.random.random() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(rh.size(0), device=device)
                t_perm = targets[perm]
                logits, _ = model(
                    lam * rh + (1 - lam) * rh[perm],
                    lam * lh + (1 - lam) * lh[perm],
                    lam * body + (1 - lam) * body[perm],
                )
                loss = (lam * F.cross_entropy(logits, targets, label_smoothing=ls)
                        + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=ls))
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()

            else:
                logits, _ = model(rh, lh, body)
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
            for rh, lh, body, targets, _ in val_ld:
                logits, _ = model(rh.to(device), lh.to(device), body.to(device))
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets.to(device)).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0
        ta = 100.0 * tc / tt if tt > 0 else 0.0
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

        if ep < 10 or (ep + 1) % 10 == 0 or va > best:
            print(
                f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} | "
                f"Loss: {tl / max(len(train_ld), 1):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
                f"LR: {lr_now:.2e} | Time: {ep_time:.1f}s"
            )

        if va > best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "val_acc": va,
                "epoch": ep + 1,
                "classes": classes,
                "num_classes": len(classes),
                "version": "masa",
                "config": config,
            }, best_path)
            print(f"[{ts()}]   -> New best! {va:.1f}%")

        # Per-class accuracy every 50 epochs
        if (ep + 1) % 50 == 0:
            model.eval()
            preds_ep, tgts_ep = [], []
            with torch.no_grad():
                for rh, lh, body, targets, _ in val_ld:
                    logits, _ = model(rh.to(device), lh.to(device), body.to(device))
                    preds_ep.extend(logits.max(1)[1].cpu().numpy())
                    tgts_ep.extend(targets.numpy())
            print(f"[{ts()}]   Per-class at epoch {ep + 1}:")
            for i in range(len(classes)):
                cn = i2c[i]
                tot = sum(1 for t in tgts_ep if t == i)
                cor = sum(1 for t, p in zip(tgts_ep, preds_ep) if t == i and t == p)
                print(f"[{ts()}]     {cn:12s}: {100.0 * cor / tot if tot > 0 else 0:.1f}% ({cor}/{tot})")

    # Final evaluation
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for rh, lh, body, targets, _ in val_ld:
            logits, _ = model(rh.to(device), lh.to(device), body.to(device))
            preds.extend(logits.max(1)[1].cpu().numpy())
            tgts.extend(targets.numpy())

    print(f"\n[{ts()}] {name} Per-Class Results (best epoch {ckpt['epoch']}):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot if tot > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot})")
    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if tgts else 0.0
    print(f"[{ts()}] {name} Overall: {ov:.1f}%")

    return {"overall": ov, "per_class": res, "best_epoch": ckpt["epoch"],
            "params": param_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL MASA Fine-Tuning",
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
                        default=os.path.join(base_data, "checkpoints", "masa"))
    parser.add_argument("--pretrained", type=str,
                        default=os.path.join(base_data, "checkpoints", "masa_pretrained", "checkpoint.pth"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("KSL MASA Fine-Tuning")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
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
    print(f"  Train: {train_dirs}  (signers 1-12)")
    print(f"  Val:   {val_dir}  (signers 13-15)")
    print(f"\nConfig:")
    print(json.dumps(MASA_CONFIG, indent=2, default=str))

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
        result = train_model(cat_name, classes, MASA_CONFIG, train_dirs, val_dir,
                            ckpt_dir, device, args.pretrained, use_focal)
        if result:
            results[cat_key] = result

    total_time = time.time() - start_time

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY - MASA Fine-Tuning")
    print(f"[{ts()}] {'=' * 70}")
    for cat_key, result in results.items():
        print(f"\n[{ts()}] {cat_key.upper()}: {result['overall']:.1f}% "
              f"(ep {result['best_epoch']}, {result['params']:,} params)")
    if "numbers" in results and "words" in results:
        combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
        print(f"\n[{ts()}] Combined: {combined:.1f}%")
    print(f"[{ts()}] Total time: {total_time / 60:.1f} minutes")

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"masa_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(results_path, "w") as f:
        json.dump({
            "version": "masa",
            "model_type": args.model_type,
            "seed": args.seed,
            "config": MASA_CONFIG,
            "results": results,
            "total_time_seconds": total_time,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": ts(),
            "pretrained": args.pretrained,
        }, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")
    print(f"[{ts()}] Done.")


if __name__ == "__main__":
    main()
