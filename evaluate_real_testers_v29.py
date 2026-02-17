#!/usr/bin/env python3
"""
Evaluate v29 model on real-world test signers.

V29 changes from v27:
- 8 GCN layers (not 4), channels: ic->64->64->128->128->128->128->128->128
- Dilated TCN with kernel=3, dilations=(1,2,4) instead of multi-scale kernels (3,5,7)
- 4 attention heads (not 2) => gcn_embed_dim = 512
- Wider aux branch: 128-dim output (not 64)
- embed_dim = 640 (512 + 128), classifier: 640->128->nc
- dropout=0.2 (not 0.3), spatial_dropout=0.1

Evaluation modes:
- baseline: standard model (best_model.pt)
- adabn_global: AdaBN on all test data (best_model.pt)
- ema: EMA model (best_model_ema.pt)

Usage:
    python evaluate_real_testers_v29.py
"""

import copy
import math
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import mediapipe as mp


# ---------------------------------------------------------------------------
# MediaPipe version check
# ---------------------------------------------------------------------------

def check_mediapipe_version():
    """Check MediaPipe version. V29 uses same 0.10.5 extracted data as v27."""
    version = mp.__version__
    expected = "0.10.5"
    print(f"  MediaPipe version: {version}")
    if version == expected:
        print(f"  (matches v29 training data extraction version)")
    else:
        print(f"  WARNING: Expected {expected} (v29 training data version). "
              f"Landmark coordinates may differ.")


# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v29.py)
# ---------------------------------------------------------------------------

POSE_INDICES = [11, 12, 13, 14, 15, 16]

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

# Parent map for bone features (48 nodes)
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]

PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

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

# Hand-body spatial features count
NUM_HAND_BODY_FEATURES = 8


# ---------------------------------------------------------------------------
# Graph Topology (matching train_ksl_v29.py exactly)
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


# ---------------------------------------------------------------------------
# Adjacency Matrix Builder (matching train_ksl_v29.py exactly)
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
# V29 Model Architecture (copied EXACTLY from train_ksl_v29.py)
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
# Video -> Landmarks extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
    """Extract MediaPipe Holistic landmarks from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    )

    all_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        all_landmarks.append(landmarks)

    cap.release()
    holistic.close()

    if len(all_landmarks) == 0:
        return None

    return np.array(all_landmarks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (matching v25/v27 dataset pipeline, no augmentation)
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors."""
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
    features = np.zeros((T, 8), dtype=np.float32)

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


def normalize_wrist_palm(h):
    """Wrist-centric + palm-size normalization (matching v25 training)."""
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


def preprocess_landmarks(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (num_frames, 225) landmarks to v29 format.

    Returns:
        gcn_tensor: (9, max_frames, 48)
        aux_tensor: (max_frames, D_aux)
    """
    f = raw.shape[0]

    if raw.shape[1] < 225:
        return None, None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # Compute hand-body features BEFORE normalization
    hand_body_feats = compute_hand_body_features(h)

    # Normalization
    h = normalize_wrist_palm(h)

    # Velocity before resampling
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]

    # Bone features
    bones = compute_bones(h)

    # Auxiliary features
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    # Temporal sampling / padding
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h = h[indices]
        velocity = velocity[indices]
        bones = bones[indices]
        joint_angles = joint_angles[indices]
        fingertip_dists = fingertip_dists[indices]
        hand_body_feats = hand_body_feats[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
        hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

    # GCN features: (max_frames, 48, 9) -> (9, max_frames, 48)
    gcn_features = np.concatenate([h, velocity, bones], axis=2)
    gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)
    gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)

    # Aux features: (max_frames, D_aux) = angles + fingertip_dists + hand_body_feats
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_features = np.clip(aux_features, -10, 10).astype(np.float32)
    aux_tensor = torch.FloatTensor(aux_features)

    return gcn_tensor, aux_tensor


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

def apply_tta(raw, max_frames=MAX_FRAMES):
    """Generate 5 augmented versions for TTA.

    Returns list of (gcn_tensor, aux_tensor) tuples.
    """
    augmented = []

    # 1. Original
    gcn, aux = preprocess_landmarks(raw, max_frames)
    if gcn is None:
        return None
    augmented.append((gcn, aux))

    # 2. Spatial jitter
    raw_jitter = raw.copy()
    raw_jitter += np.random.normal(0, 0.01, raw_jitter.shape).astype(np.float32)
    gcn_j, aux_j = preprocess_landmarks(raw_jitter, max_frames)
    if gcn_j is not None:
        augmented.append((gcn_j, aux_j))

    # 3. Speed perturbation 0.9x
    f = raw.shape[0]
    target_len_slow = int(f * 0.9)
    if target_len_slow >= 2:
        indices_slow = np.linspace(0, f - 1, target_len_slow, dtype=int)
        gcn_s, aux_s = preprocess_landmarks(raw[indices_slow], max_frames)
        if gcn_s is not None:
            augmented.append((gcn_s, aux_s))

    # 4. Speed perturbation 1.1x
    target_len_fast = int(f * 1.1)
    if target_len_fast >= 2:
        indices_fast = np.linspace(0, f - 1, target_len_fast, dtype=int)
        gcn_f, aux_f = preprocess_landmarks(raw[indices_fast], max_frames)
        if gcn_f is not None:
            augmented.append((gcn_f, aux_f))

    # 5. Small rotation (+5 degrees Z-axis on hands)
    raw_rot = raw.copy()
    angle_rad = np.radians(5.0)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for frame_idx in range(raw_rot.shape[0]):
        for lm_idx in range(21):
            base = 99 + lm_idx * 3
            x = raw_rot[frame_idx, base]
            y = raw_rot[frame_idx, base + 1]
            raw_rot[frame_idx, base] = cos_a * x - sin_a * y
            raw_rot[frame_idx, base + 1] = sin_a * x + cos_a * y
        for lm_idx in range(21):
            base = 162 + lm_idx * 3
            x = raw_rot[frame_idx, base]
            y = raw_rot[frame_idx, base + 1]
            raw_rot[frame_idx, base] = cos_a * x - sin_a * y
            raw_rot[frame_idx, base + 1] = sin_a * x + cos_a * y
    gcn_r, aux_r = preprocess_landmarks(raw_rot, max_frames)
    if gcn_r is not None:
        augmented.append((gcn_r, aux_r))

    return augmented


# ---------------------------------------------------------------------------
# AdaBN: Adaptive Batch Normalization
# ---------------------------------------------------------------------------

def adapt_bn_stats(model, data_list, device):
    """Reset BN stats and recompute from test data.

    Args:
        model: nn.Module with BatchNorm layers
        data_list: list of (gcn_tensor, aux_tensor) tuples (unbatched)
        device: torch device
    """
    if len(data_list) == 0:
        return

    # Reset all BN layers
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None  # use cumulative moving average

    # Forward pass in train mode to accumulate stats
    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()


# ---------------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------------

def confidence_level(conf):
    if conf > 0.6:
        return "HIGH"
    elif conf >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Discover test videos
# ---------------------------------------------------------------------------

def discover_test_videos(base_dir):
    """Walk the real_testers directory and find all Numbers and Words videos."""
    videos = []

    for signer_dir in sorted(os.listdir(base_dir)):
        signer_path = os.path.join(base_dir, signer_dir)
        if not os.path.isdir(signer_path):
            continue

        for subdir in os.listdir(signer_path):
            subdir_lower = subdir.lower()
            if "spelling" in subdir_lower:
                continue

            subdir_path = os.path.join(signer_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            if "number" in subdir_lower:
                category = "numbers"
            elif "word" in subdir_lower:
                category = "mixed"
            else:
                continue

            for fn in sorted(os.listdir(subdir_path)):
                if not fn.lower().endswith((".mov", ".mp4", ".avi")):
                    continue

                video_path = os.path.join(subdir_path, fn)
                name = os.path.splitext(fn)[0].strip()
                if ". " in name:
                    name = name.split(". ", 1)[1]
                name = name.replace("No ", "").replace("No. ", "").strip()

                if name in NUMBER_CLASSES:
                    item_category = "numbers"
                elif name in WORD_CLASSES:
                    item_category = "words"
                else:
                    matched = False
                    for wc in WORD_CLASSES:
                        if name.lower() == wc.lower():
                            name = wc
                            item_category = "words"
                            matched = True
                            break
                    if not matched:
                        print(f"  SKIP: Unknown class '{name}' from {fn}")
                        continue

                videos.append((video_path, name, item_category, signer_dir))

    return videos


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, raw, device, classes, use_tta=True):
    """Run inference with optional TTA. Returns dict with predictions."""
    i2c = {i: c for i, c in enumerate(classes)}

    # No-TTA prediction
    gcn, aux = preprocess_landmarks(raw)
    if gcn is None:
        return None

    with torch.no_grad():
        logits, _, _ = model(
            gcn.unsqueeze(0).to(device),
            aux.unsqueeze(0).to(device),
            grl_lambda=0.0,
        )
        probs_noTTA = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx_noTTA = probs_noTTA.argmax().item()
        conf_noTTA = probs_noTTA[pred_idx_noTTA].item()

    result = {
        "pred_class_noTTA": i2c[pred_idx_noTTA],
        "pred_idx_noTTA": pred_idx_noTTA,
        "confidence_noTTA": conf_noTTA,
        "probs_noTTA": probs_noTTA.numpy(),
    }

    if use_tta:
        augmented = apply_tta(raw)
        if augmented is None or len(augmented) == 0:
            result["pred_class"] = result["pred_class_noTTA"]
            result["pred_idx"] = result["pred_idx_noTTA"]
            result["confidence"] = result["confidence_noTTA"]
            result["probs_tta"] = result["probs_noTTA"]
            result["tta_count"] = 0
            return result

        all_probs = []
        with torch.no_grad():
            for gcn_aug, aux_aug in augmented:
                logits, _, _ = model(
                    gcn_aug.unsqueeze(0).to(device),
                    aux_aug.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                probs = F.softmax(logits, dim=1).cpu().squeeze(0)
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred_idx_tta = avg_probs.argmax().item()
        conf_tta = avg_probs[pred_idx_tta].item()

        result["pred_class"] = i2c[pred_idx_tta]
        result["pred_idx"] = pred_idx_tta
        result["confidence"] = conf_tta
        result["probs_tta"] = avg_probs.numpy()
        result["tta_count"] = len(augmented)
    else:
        result["pred_class"] = result["pred_class_noTTA"]
        result["pred_idx"] = result["pred_idx_noTTA"]
        result["confidence"] = result["confidence_noTTA"]
        result["probs_tta"] = result["probs_noTTA"]
        result["tta_count"] = 1

    return result


# ---------------------------------------------------------------------------
# Evaluate one category (single model, with TTA and no-TTA)
# ---------------------------------------------------------------------------

def evaluate_category(category_name, videos, model, device, classes, mode_name="baseline"):
    """Evaluate a single model variant on a category.

    Tracks both TTA and no-TTA accuracy.
    """
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct_tta, correct_noTTA, total = 0, 0, 0
    per_class = {}
    per_signer = {}
    details = []

    nc = len(classes)
    confusion = [[0] * nc for _ in range(nc)]

    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}

    for video_path, true_class, signer in videos:
        print(f"\n  [{mode_name}] Processing: {os.path.basename(video_path)} "
              f"(class={true_class}, {signer})")

        # Use pre-extracted landmarks if available
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
            print(f"    Loaded cached {raw.shape[0]} frames, shape {raw.shape}")
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: Could not extract landmarks")
                continue
            print(f"    Extracted {raw.shape[0]} frames, shape {raw.shape}")

        result = run_inference(model, raw, device, classes, use_tta=True)
        if result is None:
            print(f"    FAILED: Preprocessing error")
            continue

        pred_class = result["pred_class"]
        confidence = result["confidence"]
        pred_noTTA = result["pred_class_noTTA"]
        conf_noTTA = result["confidence_noTTA"]
        conf_lvl = confidence_level(confidence)

        is_correct_tta = (pred_class == true_class)
        is_correct_noTTA = (pred_noTTA == true_class)
        correct_tta += int(is_correct_tta)
        correct_noTTA += int(is_correct_noTTA)
        total += 1

        true_idx = c2i.get(true_class, -1)
        pred_idx = c2i.get(pred_class, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct_tta)

        if true_class not in per_class:
            per_class[true_class] = {"correct_tta": 0, "correct_noTTA": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct_tta"] += int(is_correct_tta)
        per_class[true_class]["correct_noTTA"] += int(is_correct_noTTA)

        if signer not in per_signer:
            per_signer[signer] = {"correct_tta": 0, "correct_noTTA": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct_tta"] += int(is_correct_tta)
        per_signer[signer]["correct_noTTA"] += int(is_correct_noTTA)

        status = "OK" if is_correct_tta else "WRONG"
        tta_delta = ""
        if pred_class != pred_noTTA:
            tta_delta = f" [TTA changed: {pred_noTTA}->{pred_class}]"
        print(f"    TTA Pred: {pred_class} (conf={confidence:.3f}, {conf_lvl}) | "
              f"True: {true_class} [{status}]{tta_delta}")
        print(f"    No-TTA:   {pred_noTTA} (conf={conf_noTTA:.3f}) | "
              f"{'OK' if is_correct_noTTA else 'WRONG'}")

        probs_tta = torch.FloatTensor(result["probs_tta"])
        top3 = torch.topk(probs_tta, min(3, len(classes)))
        top3_str = ", ".join([f"{i2c[idx.item()]}={prob.item():.3f}"
                              for idx, prob in zip(top3.indices, top3.values)])
        print(f"    Top-3 (TTA): {top3_str}")

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred_tta": pred_class,
            "pred_noTTA": pred_noTTA,
            "correct_tta": is_correct_tta,
            "correct_noTTA": is_correct_noTTA,
            "confidence_tta": confidence,
            "confidence_noTTA": conf_noTTA,
            "confidence_level": conf_lvl,
            "tta_changed_prediction": pred_class != pred_noTTA,
            "tta_count": result["tta_count"],
        })

    # Summary
    acc_tta = 100.0 * correct_tta / total if total > 0 else 0.0
    acc_noTTA = 100.0 * correct_noTTA / total if total > 0 else 0.0

    print(f"\n  {category_name.upper()} [{mode_name}] (TTA):   {correct_tta}/{total} = {acc_tta:.1f}%")
    print(f"  {category_name.upper()} [{mode_name}] (no-TTA): {correct_noTTA}/{total} = {acc_noTTA:.1f}%")
    tta_diff = acc_tta - acc_noTTA
    print(f"  TTA effect: {'+' if tta_diff >= 0 else ''}{tta_diff:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'TTA':>7s}  {'no-TTA':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            acc_t = 100.0 * pc["correct_tta"] / pc["total"]
            acc_n = 100.0 * pc["correct_noTTA"] / pc["total"]
            print(f"    {cls:>12s}  {acc_t:5.0f}%   {acc_n:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    print(f"    {'Signer':>20s}  {'TTA':>7s}  {'no-TTA':>7s}  {'N':>3s}")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        acc_t = 100.0 * ps["correct_tta"] / ps["total"]
        acc_n = 100.0 * ps["correct_noTTA"] / ps["total"]
        print(f"    {signer:>20s}  {acc_t:5.0f}%   {acc_n:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence-based analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {acc:.0f}%")
        else:
            print(f"    {lvl:>6s}: 0 predictions")

    high_med_correct = conf_buckets["HIGH"]["correct"] + conf_buckets["MEDIUM"]["correct"]
    high_med_total = conf_buckets["HIGH"]["total"] + conf_buckets["MEDIUM"]["total"]
    if high_med_total > 0:
        filtered_acc = 100.0 * high_med_correct / high_med_total
        print(f"\n    If we reject predictions below 0.4 confidence:")
        print(f"      Accuracy: {high_med_correct}/{high_med_total} = {filtered_acc:.1f}%")
        print(f"      Rejection rate: {conf_buckets['LOW']['total']}/{total} "
              f"= {100.0 * conf_buckets['LOW']['total'] / total:.0f}%")

    print(f"\n  Confusion Matrix ({category_name}, {mode_name}):")
    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"    {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{confusion[i][j]:5d}" for j in range(nc))
        print(f"    {row_str}")

    return {
        "mode": mode_name,
        "overall_tta": acc_tta,
        "overall_noTTA": acc_noTTA,
        "correct_tta": correct_tta,
        "correct_noTTA": correct_noTTA,
        "total": total,
        "tta_effect": tta_diff,
        "per_class": {k: {
            "acc_tta": 100.0 * v["correct_tta"] / v["total"],
            "acc_noTTA": 100.0 * v["correct_noTTA"] / v["total"],
            **v
        } for k, v in per_class.items()},
        "per_signer": {k: {
            "acc_tta": 100.0 * v["correct_tta"] / v["total"],
            "acc_noTTA": 100.0 * v["correct_noTTA"] / v["total"],
            **v
        } for k, v in per_signer.items()},
        "confidence_buckets": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            **v
        } for k, v in conf_buckets.items()},
        "confusion_matrix": confusion,
        "confusion_labels": [i2c[i] for i in range(nc)],
        "details": details,
    }


# ---------------------------------------------------------------------------
# Pre-extract all test data for AdaBN
# ---------------------------------------------------------------------------

def preextract_test_data(videos):
    """Pre-extract and preprocess all test videos.

    Returns list of (gcn_tensor, aux_tensor) tuples.
    """
    data_list = []
    for video_path, true_class, signer in videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                continue

        gcn, aux = preprocess_landmarks(raw)
        if gcn is None:
            continue

        data_list.append((gcn, aux))

    return data_list


# ---------------------------------------------------------------------------
# Build v29 model from checkpoint
# ---------------------------------------------------------------------------

def build_model_from_ckpt(ckpt, device, nc, num_nodes=48):
    """Create a KSLGraphNetV29 model from checkpoint config."""
    config = ckpt["config"]
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES)
    adj = build_adj(num_nodes).to(device)

    model = KSLGraphNetV29(
        nc=nc,
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=num_nodes,
        ic=config.get("in_channels", 9),
        hd=config.get("hidden_dim", 64),
        nl=config.get("num_layers", 8),
        td=tuple(config.get("temporal_dilations", [1, 2, 4])),
        dr=config.get("dropout", 0.2),
        spatial_dropout=config.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)

    return model, aux_dim


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    print("=" * 70)
    print("V29 Real-World Evaluation (Deeper ST-GCN + AdaBN + EMA)")
    print(f"Architecture: 8 GCN layers, dilated TCN (k=3, d=1,2,4), 4-head attn")
    print(f"embed_dim=640 (512 gcn + 128 aux), dropout=0.2")
    print(f"Modes: baseline, adabn_global, EMA")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # MediaPipe version check
    check_mediapipe_version()

    # Discover videos
    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    print(f"Found {len(videos)} test videos (excluding Spelling words)")

    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"  Numbers: {len(numbers_videos)} videos")
    print(f"  Words: {len(words_videos)} videos")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    for cat_name, cat_videos, classes in [
        ("numbers", numbers_videos, NUMBER_CLASSES),
        ("words", words_videos, WORD_CLASSES),
    ]:
        ckpt_path = os.path.join(ckpt_dir, f"v29_{cat_name}", "best_model.pt")
        ema_ckpt_path = os.path.join(ckpt_dir, f"v29_{cat_name}", "best_model_ema.pt")

        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos")
            continue

        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {cat_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} EVALUATION ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        # ===== Load baseline model =====
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, aux_dim = build_model_from_ckpt(ckpt, device, len(classes))
        model.load_state_dict(ckpt["model"])
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cat_name} model: {ckpt_path}")
        print(f"  Val accuracy: {ckpt['val_acc']:.1f}%, epoch {ckpt['epoch']}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Aux dim: {aux_dim} (angles={NUM_ANGLE_FEATURES}, "
              f"fingertip_dists=20, hand_body=8)")

        cat_results = {}

        # ===== MODE 1: Baseline (no AdaBN, standard model) =====
        print(f"\n--- Mode: BASELINE (standard model, no AdaBN) ---")
        baseline_result = evaluate_category(
            cat_name, cat_videos, model, device, classes, mode_name="baseline"
        )
        cat_results["baseline"] = baseline_result

        # ===== MODE 2: AdaBN Global =====
        print(f"\n--- Mode: AdaBN GLOBAL ---")
        adabn_model = copy.deepcopy(model)

        # Pre-extract test data and adapt BN stats
        print(f"  Pre-extracting test data for AdaBN...")
        test_data = preextract_test_data(cat_videos)
        print(f"  Adapting BN stats on {len(test_data)} samples...")
        adapt_bn_stats(adabn_model, test_data, device)

        adabn_result = evaluate_category(
            cat_name, cat_videos, adabn_model, device, classes, mode_name="adabn_global"
        )
        cat_results["adabn_global"] = adabn_result

        # ===== MODE 3: EMA model =====
        if os.path.exists(ema_ckpt_path):
            print(f"\n--- Mode: EMA MODEL ---")
            ema_ckpt = torch.load(ema_ckpt_path, map_location=device, weights_only=False)
            ema_model, _ = build_model_from_ckpt(ema_ckpt, device, len(classes))
            ema_model.load_state_dict(ema_ckpt["model"])
            ema_model.eval()

            ema_param_count = sum(p.numel() for p in ema_model.parameters())
            print(f"Loaded EMA model: {ema_ckpt_path}")
            print(f"  Val accuracy: {ema_ckpt['val_acc']:.1f}%, epoch {ema_ckpt['epoch']}")
            print(f"  Parameters: {ema_param_count:,}")

            ema_result = evaluate_category(
                cat_name, cat_videos, ema_model, device, classes, mode_name="ema"
            )
            cat_results["ema"] = ema_result
        else:
            print(f"\n  Skipping EMA: checkpoint not found at {ema_ckpt_path}")
            cat_results["ema"] = None

        # ===== Mode comparison =====
        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} MODE COMPARISON:")
        print(f"  {'Mode':<20s}  {'TTA':>7s}  {'no-TTA':>7s}")
        print(f"  {'Baseline':<20s}  {baseline_result['overall_tta']:5.1f}%  "
              f"{baseline_result['overall_noTTA']:5.1f}%")
        print(f"  {'AdaBN Global':<20s}  {adabn_result['overall_tta']:5.1f}%  "
              f"{adabn_result['overall_noTTA']:5.1f}%")
        if cat_results["ema"] is not None:
            ema_r = cat_results["ema"]
            print(f"  {'EMA':<20s}  {ema_r['overall_tta']:5.1f}%  "
                  f"{ema_r['overall_noTTA']:5.1f}%")

        # Determine best mode
        mode_accs = {
            "baseline": baseline_result["overall_noTTA"],
            "adabn_global": adabn_result["overall_noTTA"],
        }
        if cat_results["ema"] is not None:
            mode_accs["ema"] = cat_results["ema"]["overall_noTTA"]

        best_mode = max(mode_accs, key=mode_accs.get)
        print(f"\n  BEST mode (no-TTA): {best_mode} ({mode_accs[best_mode]:.1f}%)")

        cat_results["best_mode"] = best_mode
        all_results[cat_name] = cat_results

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V29 Real-World Evaluation")
    print(f"{'='*70}")

    for mode_name in ["baseline", "adabn_global", "ema"]:
        nums_r = all_results.get("numbers", {}).get(mode_name)
        wrds_r = all_results.get("words", {}).get(mode_name)

        if nums_r is None and wrds_r is None:
            continue

        nums_tta = nums_r["overall_tta"] if nums_r else 0
        nums_noTTA = nums_r["overall_noTTA"] if nums_r else 0
        wrds_tta = wrds_r["overall_tta"] if wrds_r else 0
        wrds_noTTA = wrds_r["overall_noTTA"] if wrds_r else 0

        combined_tta = (nums_tta + wrds_tta) / 2 if nums_r and wrds_r else 0
        combined_noTTA = (nums_noTTA + wrds_noTTA) / 2 if nums_r and wrds_r else 0

        print(f"\n  {mode_name:>20s}:")
        print(f"    Numbers  TTA={nums_tta:.1f}%  no-TTA={nums_noTTA:.1f}%")
        print(f"    Words    TTA={wrds_tta:.1f}%  no-TTA={wrds_noTTA:.1f}%")
        print(f"    Combined TTA={combined_tta:.1f}%  no-TTA={combined_noTTA:.1f}%")

    # Confidence summary across best modes
    print(f"\n  Confidence Summary (baseline, all predictions):")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        tot_c, tot_n = 0, 0
        for cat in ["numbers", "words"]:
            if cat in all_results and "baseline" in all_results[cat]:
                r = all_results[cat]["baseline"]
                if lvl in r.get("confidence_buckets", {}):
                    b = r["confidence_buckets"][lvl]
                    tot_c += b["correct"]
                    tot_n += b["total"]
        if tot_n > 0:
            print(f"    {lvl:>6s}: {tot_c}/{tot_n} = {100.0 * tot_c / tot_n:.0f}%")

    # Rejection analysis
    all_total = 0
    low_total = 0
    accepted_correct = 0
    accepted_total = 0
    for cat in ["numbers", "words"]:
        if cat in all_results and "baseline" in all_results[cat]:
            r = all_results[cat]["baseline"]
            all_total += r["total"]
            if "confidence_buckets" in r:
                low_total += r["confidence_buckets"]["LOW"]["total"]
                accepted_correct += (r["confidence_buckets"]["HIGH"]["correct"] +
                                     r["confidence_buckets"]["MEDIUM"]["correct"])
                accepted_total += (r["confidence_buckets"]["HIGH"]["total"] +
                                   r["confidence_buckets"]["MEDIUM"]["total"])

    if accepted_total > 0:
        print(f"\n  If rejecting predictions below 0.4 confidence:")
        print(f"    Accepted: {accepted_total}/{all_total} "
              f"({100.0 * accepted_total / all_total:.0f}% of predictions)")
        print(f"    Accuracy on accepted: {accepted_correct}/{accepted_total} "
              f"= {100.0 * accepted_correct / accepted_total:.1f}%")
        print(f"    Rejected: {low_total}/{all_total} "
              f"({100.0 * low_total / all_total:.0f}% of predictions)")

    # Previous version comparison
    print(f"\n  Previous versions (for comparison, no-TTA):")
    print(f"    V19: Numbers 20.3% | Words 48.1% | Combined 34.2%")
    print(f"    V20: Numbers 20.3% | Words 40.7% | Combined 30.5%")
    print(f"    V21: Numbers 25.4% | Words 37.0% | Combined 31.2%")
    print(f"    V22: Numbers 33.9% | Words 45.7% | Combined 39.8%")
    print(f"    V25: Numbers 27.1% | Words 49.4% | Combined 38.3%")
    print(f"    V26: Numbers 45.8% | Words 49.4% | Combined 47.6%")
    print(f"    V27: Numbers 54.2% | Words 53.1% | Combined 53.7%  <-- previous best")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v29_real_testers_{ts_str}.json")

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    results_out = {
        "version": "v29",
        "evaluation_type": "real_testers",
        "training_data": "ksl-alpha (12 signers train [1-12], 3 val [13-15])",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tta_enabled": True,
        "tta_augmentations": [
            "original",
            "spatial_jitter_N(0,0.01)",
            "speed_0.9x",
            "speed_1.1x",
            "rotation_+5deg_Z",
        ],
        "normalization": "wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)",
        "features": (
            "GCN: xyz(3)+velocity(3)+bone(3)=9ch | "
            "AUX: angles+fingertip_dists+hand_body_features "
            f"({NUM_ANGLE_FEATURES}+20+8={NUM_ANGLE_FEATURES + 20 + 8} features, "
            "128-dim with temporal conv1d)"
        ),
        "model": "KSLGraphNetV29 (8-layer ST-GCN + dilated TCN + 4-head attn + wider aux)",
        "v29_changes": [
            "8 GCN layers (ic->64->64->128->128->128->128->128->128), stride=2 at layers 1,3",
            "Dilated TCN (k=3, d=1,2,4) replaces multi-kernel (3,5,7)",
            "4 attention heads -> gcn_embed_dim=512",
            "Wider aux branch: 128-dim output (not 64), aux_mlp 256->128",
            "embed_dim=640 (512+128), classifier 640->128->nc",
            "dropout=0.2 (not 0.3)",
            "EMA model checkpoint evaluated alongside standard",
            "AdaBN global mode: reset BN stats from test data",
        ],
        "evaluation_modes": ["baseline", "adabn_global", "ema"],
        "results": make_serializable(all_results),
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
