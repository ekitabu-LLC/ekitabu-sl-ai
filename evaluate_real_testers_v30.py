#!/usr/bin/env python3
"""
Evaluate v30 Phase 2 and Phase 3 models on real-world test signers.

Supports:
  - Phase 2 model (KSLGraphNetV30): v28-style multi-stream + MixStyle + DropGraph
  - Phase 3 model (BlockGCN + ProtoGCN): graph distance encoding + prototype assembly
  - All Phase 1 inference-time improvements as CLI flags
  - AdaBN Global (default) + baseline comparison
  - Per-class, per-signer, confidence, confusion matrix analysis

Usage:
    python evaluate_real_testers_v30.py --model-type phase2 --tent 3 --alpha-bn 0.3
    python evaluate_real_testers_v30.py --model-type phase3 --temp-scale --conf-fusion
    python evaluate_real_testers_v30.py --ablation
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
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
    version = mp.__version__
    expected = "0.10.5"
    print(f"  MediaPipe version: {version}")
    if version != expected:
        print(f"  WARNING: Expected {expected}. Landmark coordinates may differ.")


# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v30.py)
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

LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

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


# Neighbor map for DropGraph (used by v30 training, no-op at eval)
def _build_neighbor_map(n=48):
    neighbors = defaultdict(set)
    all_edges = []
    for i, j in HAND_EDGES:
        all_edges.append((i, j))
    for i, j in HAND_EDGES:
        all_edges.append((i + 21, j + 21))
    for i, j in POSE_EDGES:
        all_edges.append((i, j))
    for i, j in all_edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    for i in range(n):
        neighbors[i].add(i)
    return dict(neighbors)

NEIGHBOR_MAP = _build_neighbor_map()


# ---------------------------------------------------------------------------
# Video -> Landmarks extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
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
# Preprocessing helpers
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
    for tips in [LH_TIPS, RH_TIPS]:
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                diff = h[:, tips[i], :] - h[:, tips[j], :]
                distances[:, col] = np.linalg.norm(diff, axis=-1)
                col += 1
    return distances


def compute_hand_body_features(h_raw):
    T = h_raw.shape[0]
    features = np.zeros((T, 8), dtype=np.float32)
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


def normalize_wrist_palm(h):
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        lh_wrist = lh[:, 0:1, :]
        lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]
        palm_sizes = np.maximum(np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        rh_wrist = rh[:, 0:1, :]
        rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]
        palm_sizes = np.maximum(np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] = pose[pose_valid] - mid_shoulder[pose_valid]
        sw = np.maximum(np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :],
                                        axis=-1, keepdims=True), 1e-6)
        pose[pose_valid] = pose[pose_valid] / sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


# ---------------------------------------------------------------------------
# Preprocessing: v27 format (9ch single model, for ensemble)
# ---------------------------------------------------------------------------

def preprocess_v27(raw, max_frames=MAX_FRAMES):
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
    hand_body_feats = compute_hand_body_features(h)
    h = normalize_wrist_palm(h)
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h, velocity, bones = h[indices], velocity[indices], bones[indices]
        joint_angles, fingertip_dists = joint_angles[indices], fingertip_dists[indices]
        hand_body_feats = hand_body_feats[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
        hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

    gcn_features = np.concatenate([h, velocity, bones], axis=2)
    gcn_tensor = torch.FloatTensor(np.clip(gcn_features, -10, 10)).permute(2, 0, 1)
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_tensor = torch.FloatTensor(np.clip(aux_features, -10, 10))
    return gcn_tensor, aux_tensor


# ---------------------------------------------------------------------------
# Preprocessing: v28/v30 multi-stream format
# ---------------------------------------------------------------------------

def preprocess_multistream(raw, max_frames=MAX_FRAMES):
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
    hand_body_feats = compute_hand_body_features(h)
    h = normalize_wrist_palm(h)
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h, velocity, bones = h[indices], velocity[indices], bones[indices]
        joint_angles, fingertip_dists = joint_angles[indices], fingertip_dists[indices]
        hand_body_feats = hand_body_feats[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
        hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

    streams = {
        "joint": torch.FloatTensor(np.clip(h, -10, 10)).permute(2, 0, 1),
        "bone": torch.FloatTensor(np.clip(bones, -10, 10)).permute(2, 0, 1),
        "velocity": torch.FloatTensor(np.clip(velocity, -10, 10)).permute(2, 0, 1),
    }
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_tensor = torch.FloatTensor(np.clip(aux_features, -10, 10))
    return streams, aux_tensor


# ===========================================================================
# MODEL ARCHITECTURES
# ===========================================================================

# --- Shared components ---

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


class AttentionPoolV25(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        mid = in_channels // 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, mid), nn.Tanh(), nn.Linear(mid, 1))
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


class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


# --- Phase 2: KSLGraphNetV30 (v28 architecture + MixStyle + DropGraph) ---
# At eval time, MixStyle is no-op (only active during training)
# DropGraph is no-op (only active during training)
# So the architecture is functionally identical to v28's KSLGraphNetV25

class MixStyle(nn.Module):
    """MixStyle: no-op at eval time."""
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B)
        lmbd = self.beta.sample([B, 1, 1, 1]).to(x.device)
        mu_mix = lmbd * mu + (1 - lmbd) * mu[perm]
        sig_mix = lmbd * sig + (1 - lmbd) * sig[perm]
        return x_normed * sig_mix + mu_mix


class DropGraph(nn.Module):
    """DropGraph: no-op at eval time."""
    def __init__(self, p=0.1, num_nodes=48):
        super().__init__()
        self.p = p
        self.num_nodes = num_nodes

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        mask = torch.ones(B, 1, 1, N, device=x.device)
        for b in range(B):
            if random.random() < self.p:
                seed = random.randint(0, N - 1)
                neighbors = NEIGHBOR_MAP.get(seed, {seed})
                for n in neighbors:
                    if n < N:
                        mask[b, 0, 0, n] = 0
        keep_ratio = mask.mean()
        if keep_ratio > 0:
            return x * mask / keep_ratio
        return x * mask


class STGCNBlockV30(nn.Module):
    def __init__(self, ic, oc, adj, temporal_kernels=(3, 5, 7), st=1, dr=0.3,
                 spatial_dropout=0.1, use_dropgraph=False):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, st, dr)
        self.residual = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc))
            if ic != oc or st != 1 else nn.Identity()
        )
        self.dropout = nn.Dropout(dr)
        if use_dropgraph:
            self.spatial_drop = DropGraph(p=spatial_dropout, num_nodes=48)
        else:
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


class KSLGraphNetV30(nn.Module):
    """Phase 2 model: ST-GCN + MixStyle + DropGraph (matching train_ksl_v30.py)."""

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=3, hd=64, nl=4,
                 tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=None,
                 use_mixstyle=True, use_dropgraph=True):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)
        self.use_mixstyle = use_mixstyle

        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlockV30(ch[i], ch[i + 1], adj, tk,
                           2 if i in [1, 3] else 1,
                           dr, spatial_dropout, use_dropgraph=use_dropgraph)
             for i in range(nl)]
        )

        if use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        final_ch = ch[-1]
        self.attn_pool = AttentionPoolV25(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch

        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, 64),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.embed_dim = gcn_embed_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc),
        )
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_signers),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        b, c, t, n = gcn_input.shape
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_mixstyle and i in (0, 1):
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

        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)
        logits = self.classifier(embedding)
        reversed_embedding = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_embedding)
        return logits, signer_logits, embedding


# --- V27/V28 model (for ensemble loading) ---

class STGCNBlockV25(nn.Module):
    def __init__(self, ic, oc, adj, temporal_kernels=(3, 5, 7), st=1, dr=0.3,
                 spatial_dropout=0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = MultiScaleTCN(oc, temporal_kernels, st, dr)
        self.residual = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc))
            if ic != oc or st != 1 else nn.Identity()
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


class KSLGraphNetV25(nn.Module):
    """V25/V26/V27 (ic=9) and V28 per-stream (ic=3)."""

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=9, hd=64, nl=4,
                 tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        ch = [ic] + [hd] * 2 + [hd * 2] * 2
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlockV25(ch[i], ch[i + 1], adj, tk,
                           2 if i in [1, 3] else 1, dr, spatial_dropout)
             for i in range(nl)]
        )

        final_ch = ch[-1]
        self.attn_pool = AttentionPoolV25(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch

        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, 64),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.embed_dim = gcn_embed_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc),
        )
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_signers),
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


# --- V29 model (for ensemble loading) ---

class DilatedMultiScaleTCN(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 4), stride=1, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (3, 1),
                          padding=(d, 0), dilation=(d, 1), stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            ) for d in dilations
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        return self.dropout(out)


class AttentionPoolV29(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        mid = in_channels // 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, mid), nn.Tanh(), nn.Linear(mid, 1))
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


class STGCNBlockV29(nn.Module):
    def __init__(self, ic, oc, adj, temporal_dilations=(1, 2, 4), st=1, dr=0.2,
                 spatial_dropout=0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = DilatedMultiScaleTCN(oc, temporal_dilations, st, dr)
        self.residual = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc))
            if ic != oc or st != 1 else nn.Identity()
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
    """V29: 8 GCN layers, dilated TCN, 4-head attn, wider aux."""

    def __init__(self, nc, num_signers, aux_dim, nn_=48, ic=9, hd=64, nl=8,
                 td=(1, 2, 4), dr=0.2, spatial_dropout=0.1, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        ch = [ic, 64, 64, 128, 128, 128, 128, 128, 128]
        ch = ch[:nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlockV29(ch[i], ch[i + 1], adj, td,
                           2 if i in [1, 3] else 1, dr, spatial_dropout)
             for i in range(nl)]
        )

        final_ch = ch[-1]
        self.attn_pool = AttentionPoolV29(final_ch, num_heads=4)
        gcn_embed_dim = 4 * final_ch

        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 256), nn.ReLU(), nn.Dropout(dr), nn.Linear(256, 128),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

        self.embed_dim = gcn_embed_dim + 128

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, nc),
        )
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128), nn.ReLU(), nn.Linear(128, num_signers),
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


# --- Phase 3: BlockGCN + ProtoGCN (matching train_ksl_v30_phase3.py) ---

# Pre-compute shortest path distances for graph topology
def _compute_shortest_path_distances(n=48):
    """BFS all-pairs shortest path on skeleton graph."""
    adj_list = defaultdict(set)
    for i, j in HAND_EDGES:
        adj_list[i].add(j)
        adj_list[j].add(i)
    for i, j in HAND_EDGES:
        adj_list[i + 21].add(j + 21)
        adj_list[j + 21].add(i + 21)
    for i, j in POSE_EDGES:
        adj_list[i].add(j)
        adj_list[j].add(i)

    dist = np.full((n, n), n + 1, dtype=np.int32)
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

_SP_DIST = _compute_shortest_path_distances(48)
MAX_GRAPH_DIST = int(_SP_DIST[_SP_DIST <= 48].max())


class GraphDistanceEncoding(nn.Module):
    """Learnable distance encoding B_ij = e_{d(i,j)} from BlockGCN paper."""
    def __init__(self, max_dist, num_nodes=48):
        super().__init__()
        self.dist_embed = nn.Parameter(torch.zeros(max_dist + 2))
        nn.init.uniform_(self.dist_embed, -0.1, 0.1)
        self.dist_embed.data[0] = 1.0
        dist_matrix = torch.from_numpy(_SP_DIST.copy()).long()
        dist_matrix = dist_matrix.clamp(max=max_dist + 1)
        self.register_buffer("dist_matrix", dist_matrix)

    def forward(self):
        return self.dist_embed[self.dist_matrix]


class BlockGConv(nn.Module):
    """Block Graph Convolution: divides channels into K groups."""
    def __init__(self, in_ch, out_ch, num_blocks=4, use_blockgcn=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_blockgcn = use_blockgcn
        self.in_ch = in_ch
        self.out_ch = out_ch

        if use_blockgcn and num_blocks > 1:
            blk_in = in_ch // num_blocks
            blk_out = out_ch // num_blocks
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
            self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x, adj):
        x_agg = torch.matmul(adj, x)
        if self.use_blockgcn and self.num_blocks > 1:
            chunks = torch.split(x_agg, self.block_sizes_in, dim=-1)
            out_chunks = [self.block_fcs[k](chunks[k]) for k in range(self.num_blocks)]
            return torch.cat(out_chunks, dim=-1)
        else:
            return self.fc(x_agg)


class Phase3MixStyle(nn.Module):
    """MixStyle for Phase 3: no-op at eval time."""
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


class Phase3DropGraph(nn.Module):
    """DropGraph for Phase 3: no-op at eval time. Uses pre-computed shortest paths."""
    def __init__(self, p=0.1, k_hop=1, num_nodes=48):
        super().__init__()
        self.p = p
        self.k_hop = k_hop
        self.register_buffer("neighborhoods", self._build_neighborhoods(num_nodes, k_hop))

    def _build_neighborhoods(self, n, k):
        nbr = torch.zeros(n, n, dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                if _SP_DIST[i, j] <= k:
                    nbr[i, j] = 1.0
        return nbr

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C, T, N = x.shape
        mask = torch.ones(B, 1, 1, N, device=x.device)
        for b in range(B):
            if random.random() < self.p:
                center = random.randint(0, N - 1)
                nbr_mask = self.neighborhoods[center]
                mask[b, 0, 0, :] = 1.0 - nbr_mask
        return x * mask


class Phase3AttentionPool(nn.Module):
    """Attention pooling for Phase 3 model."""
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        mid = in_channels // 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, mid), nn.Tanh(), nn.Linear(mid, 1))
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        outs = []
        for head in self.heads:
            attn = F.softmax(head(x_t), dim=1)
            pooled = (x_t * attn).sum(dim=1)
            outs.append(pooled)
        return torch.cat(outs, dim=1)


class Phase3BlockGCNBlock(nn.Module):
    """BlockGCN block matching train_ksl_v30_phase3.py exactly."""
    def __init__(self, ic, oc, num_blocks=4, temporal_kernels=(3, 5, 7),
                 stride=1, dropout=0.3, use_blockgcn=True, use_mixstyle=False,
                 mixstyle_p=0.5, mixstyle_alpha=0.1, dropgraph_p=0.1,
                 dropgraph_khop=1, num_nodes=48):
        super().__init__()
        self.use_blockgcn = use_blockgcn

        if use_blockgcn:
            self.dist_enc = GraphDistanceEncoding(MAX_GRAPH_DIST, num_nodes)
        else:
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

        self.mixstyle = Phase3MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else None
        self.dropgraph = Phase3DropGraph(p=dropgraph_p, k_hop=dropgraph_khop, num_nodes=num_nodes)

    def forward(self, x):
        r = self.residual(x)
        b, c, t, n = x.shape

        if self.use_blockgcn:
            adj = self.dist_enc()
        else:
            adj = self.adj_fallback

        x_flat = x.permute(0, 2, 3, 1).reshape(b * t, n, c)
        x_flat = self.gcn(x_flat, adj)
        x = x_flat.reshape(b, t, n, -1).permute(0, 3, 1, 2)
        x = torch.relu(self.bn1(x))
        x = self.dropgraph(x)
        x = self.tcn(x)
        x = torch.relu(x + r)

        if self.mixstyle is not None:
            x = self.mixstyle(x)
        return x


class MotionTopologyEnhancement(nn.Module):
    """MTE: base topology + intra-sample attention + inter-sample correlation.
    Only active during training; included for checkpoint compatibility.
    """
    def __init__(self, embed_dim, num_heads=4, num_nodes=48):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_nodes = num_nodes
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.inter_proj_q = nn.Linear(embed_dim, embed_dim)
        self.inter_proj_k = nn.Linear(embed_dim, embed_dim)
        self.inter_fc = nn.Linear(self.head_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, base_adj):
        B, N, D = x.shape
        x = self.norm(x)
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        A_intra = attn.mean(dim=1).mean(dim=0)
        Qi = self.inter_proj_q(x).view(B, N, self.num_heads, self.head_dim)
        Ki = self.inter_proj_k(x).view(B, N, self.num_heads, self.head_dim)
        diff = Qi.unsqueeze(2) - Ki.unsqueeze(1)
        A_inter = self.inter_fc(diff).squeeze(-1).mean(dim=-1)
        A_inter = A_inter.mean(dim=0)
        A_enhanced = base_adj + 0.1 * torch.sigmoid(A_intra) + 0.1 * torch.sigmoid(A_inter)
        return A_enhanced


class PrototypeReconstructionNetwork(nn.Module):
    """PRN: Memory bank of K learnable prototypes with attention-based gated assembly."""
    def __init__(self, embed_dim, num_prototypes=64, num_classes=15):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.register_buffer("class_centroids", torch.zeros(num_classes, embed_dim))
        self.register_buffer("class_counts", torch.zeros(num_classes))

    def forward(self, x):
        Q = self.query_proj(x)
        proto_norm = F.normalize(self.prototypes, dim=1)
        q_norm = F.normalize(Q, dim=1)
        attn = torch.matmul(q_norm, proto_norm.t())
        attn_weights = F.softmax(attn * 10.0, dim=1)
        reconstructed = torch.matmul(attn_weights, self.prototypes)
        gate_input = torch.cat([x, reconstructed], dim=1)
        g = self.gate(gate_input)
        enhanced = x + g * reconstructed
        return enhanced, attn_weights


class SkeletonGCL(nn.Module):
    """Skeleton Graph Contrastive Learning (training-only, zero inference cost).
    Included for checkpoint compatibility.
    """
    def __init__(self, embed_dim, num_classes, num_samples, proj_dim=128,
                 temperature=0.07, momentum=0.999):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, proj_dim),
        )
        self.register_buffer("instance_bank", torch.randn(num_samples, proj_dim))
        self.register_buffer("instance_labels", torch.zeros(num_samples, dtype=torch.long))
        self.register_buffer("instance_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("semantic_bank", torch.zeros(num_classes, proj_dim))
        self.register_buffer("semantic_counts", torch.zeros(num_classes))


class KSLBlockGCNet(nn.Module):
    """V30 Phase 3 model: BlockGCN + ProtoGCN + SkeletonGCL.
    Matches train_ksl_v30_phase3.py exactly.
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
            use_ms = (i < 2)
            self.layers.append(Phase3BlockGCNBlock(
                ic=ch[i], oc=ch[i + 1], num_blocks=num_blocks,
                temporal_kernels=tk, stride=stride_i, dropout=dr,
                use_blockgcn=use_blockgcn, use_mixstyle=use_ms,
                mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_a,
                dropgraph_p=dropgraph_p, dropgraph_khop=dropgraph_k,
                num_nodes=nn_,
            ))

        final_ch = ch[-1]
        self.attn_pool = Phase3AttentionPool(final_ch, num_heads=2)
        gcn_embed_dim = 2 * final_ch

        self.aux_bn = nn.BatchNorm1d(aux_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128), nn.ReLU(), nn.Dropout(dr), nn.Linear(128, 64),
        )
        self.aux_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.aux_attn = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.embed_dim = gcn_embed_dim + 64

        if use_protogcn:
            num_proto = config.get("num_prototypes", 64) if config else 64
            self.proto_net = PrototypeReconstructionNetwork(self.embed_dim, num_proto, nc)
            self.mte = MotionTopologyEnhancement(final_ch, num_heads=4, num_nodes=nn_)
        else:
            self.proto_net = None
            self.mte = None

        if use_skeletongcl:
            proj_dim = config.get("gcl_proj_dim", 128) if config else 128
            gcl_temp = config.get("gcl_temperature", 0.07) if config else 0.07
            gcl_mom = config.get("gcl_momentum", 0.999) if config else 0.999
            self.gcl = SkeletonGCL(
                self.embed_dim, nc, max(num_samples, 1), proj_dim, gcl_temp, gcl_mom)
        else:
            self.gcl = None

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc),
        )
        self.signer_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_signers),
        )

    def forward(self, gcn_input, aux_input, grl_lambda=0.0):
        b, c, t, n = gcn_input.shape
        x = self.data_bn(
            gcn_input.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        # MTE: training only
        if self.mte is not None and self.training:
            node_feats = x.mean(dim=2).permute(0, 2, 1)
            if self.use_blockgcn:
                base_adj = self.layers[0].dist_enc()
            else:
                base_adj = self.layers[0].adj_fallback
            _ = self.mte(node_feats, base_adj)

        x_node_avg = x.mean(dim=3)
        gcn_embedding = self.attn_pool(x_node_avg)

        aux = self.aux_bn(aux_input.permute(0, 2, 1)).permute(0, 2, 1)
        aux = self.aux_mlp(aux)
        aux_t = self.aux_temporal_conv(aux.permute(0, 2, 1)).permute(0, 2, 1)
        attn = F.softmax(self.aux_attn(aux_t), dim=1)
        aux_embedding = (aux_t * attn).sum(dim=1)

        embedding = torch.cat([gcn_embedding, aux_embedding], dim=1)

        proto_attn = None
        if self.proto_net is not None:
            embedding, proto_attn = self.proto_net(embedding)

        logits = self.classifier(embedding)
        reversed_emb = GradientReversalFunction.apply(embedding, grl_lambda)
        signer_logits = self.signer_head(reversed_emb)
        return logits, signer_logits, embedding, proto_attn


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_v30_phase2_stream_models(ckpt_dir, classes, device):
    """Load Phase 2 (v30) multi-stream models + fusion weights."""
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing v30 phase2 {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV30(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
            use_mixstyle=config.get("use_mixstyle", True),
            use_dropgraph=config.get("use_dropgraph", True),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded v30 phase2/{sname}: val_acc={ckpt['val_acc']:.1f}%, "
              f"epoch={ckpt['epoch']}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v30_phase3_stream_models(ckpt_dir, classes, device):
    """Load Phase 3 (KSLBlockGCNet) multi-stream models + fusion weights."""
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing v30 phase3 {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)
        num_samples = ckpt.get("num_samples", 900)

        model = KSLBlockGCNet(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            num_samples=num_samples,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            use_blockgcn=config.get("use_blockgcn", True),
            use_protogcn=config.get("use_protogcn", True),
            use_skeletongcl=config.get("use_skeletongcl", True),
            config=config,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded v30 phase3/{sname}: val_acc={ckpt['val_acc']:.1f}%, "
              f"epoch={ckpt['epoch']}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def _load_fusion_weights(ckpt_dir):
    """Load fusion weights from checkpoint directory."""
    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            fusion_data = json.load(f)
        fw = fusion_data["weights"]
        val_acc = fusion_data.get("val_accuracy", "N/A")
        print(f"  Fusion weights: {fw} (val_acc={val_acc})")
        return fw
    else:
        fw = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
        print(f"  Using default fusion weights: {fw}")
        return fw


def load_v27_model(ckpt_path, classes, device):
    """Load v27 single-model checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(config["num_nodes"]).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES)

    model = KSLGraphNetV25(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config["num_nodes"], ic=config["in_channels"], hd=config["hidden_dim"],
        nl=config["num_layers"], tk=tk, dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded v27: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


def load_v28_stream_models(ckpt_dir, classes, device):
    """Load v28 multi-stream models + fusion weights."""
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV25(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config["num_nodes"], ic=config["in_channels"],
            hd=config["hidden_dim"], nl=config["num_layers"], tk=tk,
            dr=config["dropout"], spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v28/{sname}: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v29_model(ckpt_path, classes, device):
    """Load v29 single-model checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(48).to(device)
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES)

    model = KSLGraphNetV29(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=48, ic=config.get("in_channels", 9),
        hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 8),
        td=tuple(config.get("temporal_dilations", [1, 2, 4])),
        dr=config.get("dropout", 0.2),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded v29: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


# ===========================================================================
# PHASE 1 TECHNIQUES
# ===========================================================================

# --- Technique 1: Temperature Scaling ---

def learn_temperature(stream_models, fusion_weights, val_data, device):
    """Learn optimal temperature T on val set via grid search over fused logits."""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for streams, aux, label in val_data:
            per_stream_logits = {}
            for sname, smodel in stream_models.items():
                gcn = streams[sname].unsqueeze(0).to(device)
                aux_t = aux.unsqueeze(0).to(device)
                logits, *_ = smodel(gcn, aux_t, grl_lambda=0.0)
                per_stream_logits[sname] = logits
            fused_logits = sum(fusion_weights[s] * per_stream_logits[s] for s in stream_models)
            all_logits.append(fused_logits.cpu().squeeze(0))
            all_labels.append(label)

    all_logits = torch.stack(all_logits)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    best_T = 1.0
    best_nll = float('inf')
    for T_100 in range(10, 1001, 5):
        T = T_100 / 100.0
        nll = F.cross_entropy(all_logits / T, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T

    print(f"  Temperature scaling: T={best_T:.2f} (NLL={best_nll:.4f})")
    return best_T


# --- Technique 2: Alpha-BN ---

def save_bn_stats(model):
    source_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                "running_mean": m.running_mean.clone(),
                "running_var": m.running_var.clone(),
            }
    return source_stats


def compute_target_bn_stats(model, data_list, device, stream_name):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None

    model.train()
    with torch.no_grad():
        for streams, aux in data_list:
            gcn = streams[stream_name].unsqueeze(0).to(device)
            model(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()

    target_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            target_stats[name] = {
                "running_mean": m.running_mean.clone(),
                "running_var": m.running_var.clone(),
            }
    return target_stats


def apply_alpha_bn(model, source_stats, target_stats, alpha):
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if name in source_stats and name in target_stats:
                m.running_mean.copy_(
                    alpha * source_stats[name]["running_mean"] +
                    (1 - alpha) * target_stats[name]["running_mean"]
                )
                m.running_var.copy_(
                    alpha * source_stats[name]["running_var"] +
                    (1 - alpha) * target_stats[name]["running_var"]
                )


# --- Technique 3: TENT ---

def softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_tent(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def run_tent_adaptation(model, data_list, device, n_steps=1, lr=1e-3, stream_name=None):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)

    if not params:
        print("    TENT: No trainable BN params found")
        return

    optimizer = torch.optim.Adam(params, lr=lr)
    model = configure_tent(model)

    for step in range(n_steps):
        total_entropy = 0.0
        n_samples = 0
        for streams, aux in data_list:
            gcn = streams[stream_name].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, *_ = model(gcn, aux_t, grl_lambda=0.0)
            loss = softmax_entropy(logits).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_entropy += loss.item()
            n_samples += 1

        avg_entropy = total_entropy / max(n_samples, 1)
        print(f"    TENT step {step+1}/{n_steps}: avg_entropy={avg_entropy:.4f}")

    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False


# --- Technique 4: Ensemble ---

def ensemble_predict(raw, v27_model, v28_models, v28_fusion_weights, v29_model,
                     v30_models, v30_fusion_weights, device, classes, temperature=1.0):
    """Ensemble prediction combining v27+v28+v29+v30."""
    i2c = {i: c for i, c in enumerate(classes)}
    all_probs = []

    # V27: single model (9ch)
    if v27_model is not None:
        gcn_v27, aux_v27 = preprocess_v27(raw)
        if gcn_v27 is not None:
            with torch.no_grad():
                logits, *_ = v27_model(
                    gcn_v27.unsqueeze(0).to(device), aux_v27.unsqueeze(0).to(device), grl_lambda=0.0)
                all_probs.append(F.softmax(logits / temperature, dim=1).cpu().squeeze(0))

    # V28: multi-stream
    if v28_models and len(v28_models) == 3:
        streams, aux_v28 = preprocess_multistream(raw)
        if streams is not None:
            per_stream_probs = {}
            with torch.no_grad():
                for sname, smodel in v28_models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    logits, *_ = smodel(gcn, aux_v28.unsqueeze(0).to(device), grl_lambda=0.0)
                    per_stream_probs[sname] = F.softmax(logits / temperature, dim=1).cpu().squeeze(0)
            all_probs.append(sum(v28_fusion_weights[s] * per_stream_probs[s] for s in v28_models))

    # V29: single model (9ch)
    if v29_model is not None:
        gcn_v29, aux_v29 = preprocess_v27(raw)
        if gcn_v29 is not None:
            with torch.no_grad():
                logits, *_ = v29_model(
                    gcn_v29.unsqueeze(0).to(device), aux_v29.unsqueeze(0).to(device), grl_lambda=0.0)
                all_probs.append(F.softmax(logits / temperature, dim=1).cpu().squeeze(0))

    # V30: multi-stream
    if v30_models and len(v30_models) == 3:
        streams, aux_v30 = preprocess_multistream(raw)
        if streams is not None:
            per_stream_probs = {}
            with torch.no_grad():
                for sname, smodel in v30_models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    logits, *_ = smodel(gcn, aux_v30.unsqueeze(0).to(device), grl_lambda=0.0)
                    per_stream_probs[sname] = F.softmax(logits / temperature, dim=1).cpu().squeeze(0)
            all_probs.append(sum(v30_fusion_weights[s] * per_stream_probs[s] for s in v30_models))

    if not all_probs:
        return None, 0.0, None

    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    pred_idx = ensemble_probs.argmax().item()
    return i2c[pred_idx], ensemble_probs[pred_idx].item(), ensemble_probs


# --- Technique 5: Confidence-Weighted Fusion ---

def confidence_weighted_fusion(streams, aux, models, device, classes, temperature=1.0):
    i2c = {i: c for i, c in enumerate(classes)}
    per_stream_probs = {}
    per_stream_entropy = {}

    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, *_ = smodel(gcn, aux_t, grl_lambda=0.0)
            probs = F.softmax(logits / temperature, dim=1).cpu().squeeze(0)
            per_stream_probs[sname] = probs
            per_stream_entropy[sname] = softmax_entropy(logits).item()

    inv_entropies = {s: 1.0 / (e + 1e-8) for s, e in per_stream_entropy.items()}
    total_inv = sum(inv_entropies.values())
    conf_weights = {s: v / total_inv for s, v in inv_entropies.items()}

    fused = sum(conf_weights[s] * per_stream_probs[s] for s in models)
    pred_idx = fused.argmax().item()
    return i2c[pred_idx], fused[pred_idx].item(), fused, per_stream_probs


# --- Technique 6: T3A ---

class T3AClassifier:
    def __init__(self, classifier_weight, classifier_bias, confidence_threshold=0.9):
        self.prototypes = F.normalize(classifier_weight.clone(), dim=1)
        self.counts = torch.ones(classifier_weight.shape[0])
        self.threshold = confidence_threshold

    def predict(self, embedding):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        emb_norm = F.normalize(embedding, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        logits = torch.mm(emb_norm, proto_norm.t()) * 10.0
        return logits

    def update(self, embedding, logits):
        probs = F.softmax(logits, dim=1).squeeze(0)
        confidence = probs.max().item()
        pred_idx = probs.argmax().item()
        if confidence >= self.threshold:
            emb_norm = F.normalize(embedding, dim=1).squeeze(0)
            count = self.counts[pred_idx]
            self.prototypes[pred_idx] = (
                self.prototypes[pred_idx] * count + emb_norm
            ) / (count + 1)
            self.counts[pred_idx] += 1


def get_final_classifier_weights(model):
    last_linear = None
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear.weight.data.cpu(), last_linear.bias.data.cpu()
    return None, None


def get_embedding(model, gcn, aux, device):
    with torch.no_grad():
        result = model(
            gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
    return result[2].cpu()  # embedding is always 3rd return value


# ===========================================================================
# ADABN (from v28)
# ===========================================================================

def adapt_bn_stats(model, data_list, device, stream_name=None):
    """Standard AdaBN: reset and recompute BN stats from test data."""
    if len(data_list) == 0:
        return
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None

    model.train()
    with torch.no_grad():
        for item in data_list:
            streams, aux = item
            if stream_name is not None:
                gcn = streams[stream_name].unsqueeze(0).to(device)
            else:
                gcn = streams.unsqueeze(0).to(device)
            model(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()


# ===========================================================================
# CONFIDENCE & VIDEO DISCOVERY
# ===========================================================================

def confidence_level(conf):
    if conf > 0.6:
        return "HIGH"
    elif conf >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def discover_test_videos(base_dir):
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


# ===========================================================================
# PRE-EXTRACT DATA
# ===========================================================================

def preextract_test_data(videos):
    """Pre-extract and preprocess all test videos as multistream."""
    data_list = []
    for video_path, true_class, signer in videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                continue
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            continue
        data_list.append((streams, aux))
    return data_list


def preextract_val_data(val_dir, classes):
    """Pre-extract validation .npy files for temperature scaling."""
    c2i = {c: i for i, c in enumerate(classes)}
    data_list = []
    for cls_name in sorted(os.listdir(val_dir)):
        cls_path = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in c2i:
            continue
        for fn in sorted(os.listdir(cls_path)):
            if not fn.endswith(".npy"):
                continue
            raw = np.load(os.path.join(cls_path, fn)).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] < 225:
                continue
            streams, aux = preprocess_multistream(raw)
            if streams is None:
                continue
            data_list.append((streams, aux, c2i[cls_name]))
    return data_list


# ===========================================================================
# UNIFIED EVALUATION FUNCTION
# ===========================================================================

def evaluate_method(method_name, videos, predict_fn, classes):
    """Generic evaluation: calls predict_fn(raw, true_class) -> (pred, conf)."""
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}

    correct, total = 0, 0
    per_class = {}
    per_signer = {}
    nc = len(classes)
    confusion = [[0] * nc for _ in range(nc)]
    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}
    details = []

    for video_path, true_class, signer in videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: {os.path.basename(video_path)}")
                continue

        result = predict_fn(raw, true_class)
        if result is None:
            continue

        pred_class, confidence = result
        conf_lvl = confidence_level(confidence)
        is_correct = (pred_class == true_class)
        correct += int(is_correct)
        total += 1

        true_idx = c2i.get(true_class, -1)
        pred_idx = c2i.get(pred_class, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct)

        if true_class not in per_class:
            per_class[true_class] = {"correct": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct"] += int(is_correct)

        if signer not in per_signer:
            per_signer[signer] = {"correct": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct"] += int(is_correct)

        status = "OK" if is_correct else "WRONG"
        print(f"    [{method_name}] {os.path.basename(video_path)}: "
              f"{pred_class} (conf={confidence:.3f}) | True: {true_class} [{status}]")

        details.append({
            "video": os.path.basename(video_path),
            "signer": signer, "true": true_class, "pred": pred_class,
            "correct": is_correct, "confidence": confidence,
            "confidence_level": conf_lvl,
        })

    acc = 100.0 * correct / total if total > 0 else 0.0

    # Print summary
    print(f"\n  {method_name}: {correct}/{total} = {acc:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'Acc':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            cls_acc = 100.0 * pc["correct"] / pc["total"]
            print(f"    {cls:>12s}  {cls_acc:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        s_acc = 100.0 * ps["correct"] / ps["total"]
        print(f"    {signer:>20s}  {s_acc:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            b_acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {b_acc:.0f}%")

    # Rejection analysis
    high_med_correct = conf_buckets["HIGH"]["correct"] + conf_buckets["MEDIUM"]["correct"]
    high_med_total = conf_buckets["HIGH"]["total"] + conf_buckets["MEDIUM"]["total"]
    if high_med_total > 0:
        filtered_acc = 100.0 * high_med_correct / high_med_total
        print(f"\n    If rejecting below 0.4 confidence:")
        print(f"      Accuracy: {high_med_correct}/{high_med_total} = {filtered_acc:.1f}%")
        print(f"      Rejection rate: {conf_buckets['LOW']['total']}/{total} "
              f"= {100.0 * conf_buckets['LOW']['total'] / total:.0f}%")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"    {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{confusion[i][j]:5d}" for j in range(nc))
        print(f"    {row_str}")

    return {
        "method": method_name,
        "overall": acc,
        "correct": correct,
        "total": total,
        "per_class": {k: {"accuracy": 100.0 * v["correct"] / v["total"], **v}
                      for k, v in per_class.items()},
        "per_signer": {k: {"accuracy": 100.0 * v["correct"] / v["total"], **v}
                       for k, v in per_signer.items()},
        "confidence_buckets": {k: {
            "accuracy": 100.0 * v["correct"] / v["total"] if v["total"] > 0 else 0.0, **v
        } for k, v in conf_buckets.items()},
        "confusion_matrix": confusion,
        "confusion_labels": [i2c[i] for i in range(nc)],
        "details": details,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V30 Real-World Evaluation (Phase 2 + Phase 3 models)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", type=str, default="phase2",
                        choices=["phase2", "phase3"],
                        help="Model architecture to evaluate")
    # Phase 1 inference techniques
    parser.add_argument("--temp-scale", action="store_true",
                        help="Apply temperature scaling (learn T on val set)")
    parser.add_argument("--alpha-bn", type=float, default=None,
                        help="Alpha-BN: interpolation factor (0=target, 1=source)")
    parser.add_argument("--tent", type=int, default=0,
                        help="TENT: number of entropy minimization steps")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble v27+v28+v29+v30 models")
    parser.add_argument("--conf-fusion", action="store_true",
                        help="Confidence-weighted stream fusion")
    parser.add_argument("--t3a", action="store_true",
                        help="T3A prototype classifier")
    parser.add_argument("--t3a-threshold", type=float, default=0.9,
                        help="T3A confidence threshold for prototype updates")
    parser.add_argument("--ablation", action="store_true",
                        help="Run all techniques separately for comparison")
    parser.add_argument("--no-adabn", action="store_true",
                        help="Skip AdaBN (baseline only)")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    ckpt_base = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"
    val_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/test_alpha"

    print("=" * 70)
    print(f"V30 Real-World Evaluation (model={args.model_type})")
    print(f"Phase 1 techniques: temp_scale={args.temp_scale}, alpha_bn={args.alpha_bn}, "
          f"tent={args.tent}, ensemble={args.ensemble}, conf_fusion={args.conf_fusion}, "
          f"t3a={args.t3a}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    check_mediapipe_version()

    # Discover videos
    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    print(f"Found {len(videos)} test videos")

    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"  Numbers: {len(numbers_videos)} videos")
    print(f"  Words: {len(words_videos)} videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    for cat_name, cat_videos, classes in [
        ("numbers", numbers_videos, NUMBER_CLASSES),
        ("words", words_videos, WORD_CLASSES),
    ]:
        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos")
            continue

        # Determine checkpoint directory
        if args.model_type == "phase2":
            cat_ckpt_dir = os.path.join(ckpt_base, f"v30_{cat_name}")
        else:
            cat_ckpt_dir = os.path.join(ckpt_base, f"v30_phase3_{cat_name}")

        if not os.path.isdir(cat_ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {cat_ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} EVALUATION ({len(cat_videos)} videos, model={args.model_type})")
        print(f"{'='*70}")

        cat_results = {}

        # Load v30 models
        print(f"\n  Loading v30 {args.model_type} models...")
        if args.model_type == "phase2":
            v30_models, v30_fusion_weights = load_v30_phase2_stream_models(
                cat_ckpt_dir, classes, device)
        else:
            v30_models, v30_fusion_weights = load_v30_phase3_stream_models(
                cat_ckpt_dir, classes, device)

        if len(v30_models) < 3:
            print(f"  WARNING: Only {len(v30_models)}/3 stream models loaded, skipping")
            continue

        # Pre-extract test data
        stream_data = preextract_test_data(cat_videos)

        # ================================================================
        # BASELINE (no AdaBN)
        # ================================================================
        print(f"\n--- BASELINE (no AdaBN) ---")
        def baseline_predict(raw, true_class, _models=v30_models, _fw=v30_fusion_weights):
            streams, aux = preprocess_multistream(raw)
            if streams is None:
                return None
            i2c = {i: c for i, c in enumerate(classes)}
            per_stream_probs = {}
            with torch.no_grad():
                for sname, smodel in _models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                    per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
            fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
            pred_idx = fused.argmax().item()
            return i2c[pred_idx], fused[pred_idx].item()

        cat_results["baseline"] = evaluate_method(
            f"v30_{args.model_type}_baseline", cat_videos, baseline_predict, classes
        )

        # ================================================================
        # AdaBN GLOBAL
        # ================================================================
        if not args.no_adabn:
            print(f"\n--- AdaBN GLOBAL ---")
            adabn_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in adabn_models:
                print(f"  Adapting BN stats for {sname} stream...")
                adapt_bn_stats(adabn_models[sname], stream_data, device, stream_name=sname)
                print(f"    Adapted on {len(stream_data)} samples")

            def adabn_predict(raw, true_class, _models=adabn_models, _fw=v30_fusion_weights):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(classes)}
                per_stream_probs = {}
                with torch.no_grad():
                    for sname, smodel in _models.items():
                        gcn = streams[sname].unsqueeze(0).to(device)
                        logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                        per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results["adabn_global"] = evaluate_method(
                f"v30_{args.model_type}_adabn_global", cat_videos, adabn_predict, classes
            )

        # ================================================================
        # TECHNIQUE 1: Temperature Scaling
        # ================================================================
        if args.temp_scale or args.ablation:
            print(f"\n--- TEMPERATURE SCALING ---")
            ts_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in ts_models:
                adapt_bn_stats(ts_models[sname], stream_data, device, stream_name=sname)

            # Learn T on val set
            val_data_dir = os.path.join(val_dir, cat_name) if os.path.isdir(
                os.path.join(val_dir, cat_name)) else val_dir
            val_data = preextract_val_data(val_data_dir, classes)
            if not val_data:
                val_data = preextract_val_data(val_dir, classes)

            if val_data:
                T = learn_temperature(ts_models, v30_fusion_weights, val_data, device)
            else:
                T = 1.5
                print(f"  No val data found, using default T={T}")

            def temp_predict(raw, true_class, _models=ts_models, _fw=v30_fusion_weights, _T=T):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(classes)}
                per_stream_probs = {}
                with torch.no_grad():
                    for sname, smodel in _models.items():
                        gcn = streams[sname].unsqueeze(0).to(device)
                        logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                        per_stream_probs[sname] = F.softmax(logits / _T, dim=1).cpu().squeeze(0)
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results["temp_scale"] = evaluate_method(
                f"v30_adabn+tempscale(T={T:.2f})", cat_videos, temp_predict, classes
            )
            del ts_models

        # ================================================================
        # TECHNIQUE 2: Alpha-BN
        # ================================================================
        if args.alpha_bn is not None or args.ablation:
            alphas = [args.alpha_bn] if args.alpha_bn is not None else [0.3, 0.5, 0.7]
            for alpha in alphas:
                print(f"\n--- ALPHA-BN (alpha={alpha}) ---")
                abn_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
                for sname in abn_models:
                    source_stats = save_bn_stats(abn_models[sname])
                    target_stats = compute_target_bn_stats(
                        abn_models[sname], stream_data, device, stream_name=sname)
                    apply_alpha_bn(abn_models[sname], source_stats, target_stats, alpha)

                def alpha_predict(raw, true_class, _models=abn_models, _fw=v30_fusion_weights):
                    streams, aux = preprocess_multistream(raw)
                    if streams is None:
                        return None
                    i2c = {i: c for i, c in enumerate(classes)}
                    per_stream_probs = {}
                    with torch.no_grad():
                        for sname, smodel in _models.items():
                            gcn = streams[sname].unsqueeze(0).to(device)
                            logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
                    fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                    pred_idx = fused.argmax().item()
                    return i2c[pred_idx], fused[pred_idx].item()

                cat_results[f"alpha_bn_{alpha}"] = evaluate_method(
                    f"v30_alpha_bn(alpha={alpha})", cat_videos, alpha_predict, classes
                )
                del abn_models

        # ================================================================
        # TECHNIQUE 3: TENT
        # ================================================================
        if args.tent > 0 or args.ablation:
            n_steps = args.tent if args.tent > 0 else 3
            print(f"\n--- TENT ({n_steps} steps) ---")
            tent_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in tent_models:
                adapt_bn_stats(tent_models[sname], stream_data, device, stream_name=sname)
                print(f"  Running TENT on {sname} stream...")
                run_tent_adaptation(
                    tent_models[sname], stream_data, device,
                    n_steps=n_steps, lr=1e-3, stream_name=sname)

            def tent_predict(raw, true_class, _models=tent_models, _fw=v30_fusion_weights):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(classes)}
                per_stream_probs = {}
                with torch.no_grad():
                    for sname, smodel in _models.items():
                        gcn = streams[sname].unsqueeze(0).to(device)
                        logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                        per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results[f"tent_{n_steps}"] = evaluate_method(
                f"v30_adabn+tent({n_steps}steps)", cat_videos, tent_predict, classes
            )
            del tent_models

        # ================================================================
        # TECHNIQUE 4: ENSEMBLE (v27+v28+v29+v30)
        # ================================================================
        if args.ensemble or args.ablation:
            print(f"\n--- ENSEMBLE (v27+v28+v29+v30) ---")
            v27_ckpt = os.path.join(ckpt_base, f"v27_{cat_name}", "best_model.pt")
            v28_ckpt_dir = os.path.join(ckpt_base, f"v28_{cat_name}")
            v29_ckpt = os.path.join(ckpt_base, f"v29_{cat_name}", "best_model.pt")

            ens_v27 = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None
            ens_v28, ens_v28_fw = (load_v28_stream_models(v28_ckpt_dir, classes, device)
                                   if os.path.isdir(v28_ckpt_dir) else (None, None))
            ens_v29 = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

            # AdaBN on all
            if ens_v27:
                v27_data = []
                for video_path, true_class, signer in cat_videos:
                    npy_cache = video_path + ".landmarks.npy"
                    raw = np.load(npy_cache) if os.path.exists(npy_cache) else extract_landmarks_from_video(video_path)
                    if raw is None:
                        continue
                    gcn, aux = preprocess_v27(raw)
                    if gcn is not None:
                        # Wrap as dict-like for adapt_bn_stats
                        v27_data.append((gcn, aux))
                # Direct AdaBN for v27 (non-multistream)
                for m_mod in ens_v27.modules():
                    if isinstance(m_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        m_mod.running_mean.zero_()
                        m_mod.running_var.fill_(1.0)
                        m_mod.num_batches_tracked.zero_()
                        m_mod.momentum = None
                ens_v27.train()
                with torch.no_grad():
                    for gcn, aux in v27_data:
                        ens_v27(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
                ens_v27.eval()

            if ens_v28 and len(ens_v28) == 3:
                for sname in ens_v28:
                    adapt_bn_stats(ens_v28[sname], stream_data, device, stream_name=sname)

            if ens_v29:
                v29_data = []
                for video_path, true_class, signer in cat_videos:
                    npy_cache = video_path + ".landmarks.npy"
                    raw = np.load(npy_cache) if os.path.exists(npy_cache) else extract_landmarks_from_video(video_path)
                    if raw is None:
                        continue
                    gcn, aux = preprocess_v27(raw)
                    if gcn is not None:
                        v29_data.append((gcn, aux))
                for m_mod in ens_v29.modules():
                    if isinstance(m_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        m_mod.running_mean.zero_()
                        m_mod.running_var.fill_(1.0)
                        m_mod.num_batches_tracked.zero_()
                        m_mod.momentum = None
                ens_v29.train()
                with torch.no_grad():
                    for gcn, aux in v29_data:
                        ens_v29(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
                ens_v29.eval()

            # AdaBN on v30
            ens_v30 = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in ens_v30:
                adapt_bn_stats(ens_v30[sname], stream_data, device, stream_name=sname)

            def ensemble_predict_fn(raw, true_class):
                pred, conf, _ = ensemble_predict(
                    raw, ens_v27, ens_v28, ens_v28_fw, ens_v29,
                    ens_v30, v30_fusion_weights, device, classes)
                if pred is None:
                    return None
                return pred, conf

            cat_results["ensemble"] = evaluate_method(
                "ensemble_v27+v28+v29+v30", cat_videos, ensemble_predict_fn, classes
            )
            del ens_v27, ens_v28, ens_v29, ens_v30

        # ================================================================
        # TECHNIQUE 5: Confidence-Weighted Fusion
        # ================================================================
        if args.conf_fusion or args.ablation:
            print(f"\n--- CONFIDENCE-WEIGHTED FUSION ---")
            cf_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in cf_models:
                adapt_bn_stats(cf_models[sname], stream_data, device, stream_name=sname)

            def conf_fusion_predict(raw, true_class, _models=cf_models):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                pred, conf, _, _ = confidence_weighted_fusion(
                    streams, aux, _models, device, classes)
                return pred, conf

            cat_results["conf_fusion"] = evaluate_method(
                "v30_adabn+conf_weighted_fusion", cat_videos, conf_fusion_predict, classes
            )
            del cf_models

        # ================================================================
        # TECHNIQUE 6: T3A
        # ================================================================
        if args.t3a or args.ablation:
            print(f"\n--- T3A PROTOTYPE CLASSIFIER ---")
            t3a_models = {s: copy.deepcopy(m) for s, m in v30_models.items()}
            for sname in t3a_models:
                adapt_bn_stats(t3a_models[sname], stream_data, device, stream_name=sname)

            t3a_classifiers = {}
            for sname, smodel in t3a_models.items():
                w, b = get_final_classifier_weights(smodel)
                if w is not None:
                    t3a_classifiers[sname] = T3AClassifier(w, b, args.t3a_threshold)

            def t3a_predict(raw, true_class, _models=t3a_models, _fw=v30_fusion_weights):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(classes)}
                per_stream_probs = {}
                for sname, smodel in _models.items():
                    embedding = get_embedding(smodel, streams[sname], aux, device)
                    if sname in t3a_classifiers:
                        t3a = t3a_classifiers[sname]
                        logits = t3a.predict(embedding.squeeze(0))
                        probs = F.softmax(logits, dim=1).squeeze(0)
                        t3a.update(embedding, logits)
                    else:
                        with torch.no_grad():
                            gcn = streams[sname].unsqueeze(0).to(device)
                            logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
                    per_stream_probs[sname] = probs
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results["t3a"] = evaluate_method(
                f"v30_adabn+t3a(thr={args.t3a_threshold})", cat_videos, t3a_predict, classes
            )
            del t3a_models, t3a_classifiers

        # ================================================================
        # MODE COMPARISON
        # ================================================================
        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} METHOD COMPARISON:")
        print(f"  {'Method':<45s}  {'Accuracy':>8s}")
        print(f"  {'-'*45}  {'-'*8}")
        for method_key in sorted(cat_results.keys()):
            r = cat_results[method_key]
            print(f"  {r['method']:<45s}  {r['overall']:5.1f}%")

        # Delta vs v28
        v28_ref = {"numbers": 55.9, "words": 60.5}
        if cat_name in v28_ref:
            ref = v28_ref[cat_name]
            print(f"\n  Delta vs v28 AdaBN Global ({ref:.1f}%):")
            for method_key in sorted(cat_results.keys()):
                r = cat_results[method_key]
                delta = r["overall"] - ref
                sign = "+" if delta >= 0 else ""
                print(f"    {r['method']:<45s}  {sign}{delta:.1f}pp")

        all_results[cat_name] = cat_results

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - V30 {args.model_type.upper()} Real-World Evaluation")
    print(f"{'='*70}")

    all_methods = set()
    for cat_name in ["numbers", "words"]:
        if cat_name in all_results:
            all_methods.update(all_results[cat_name].keys())

    for method_key in sorted(all_methods):
        nums = all_results.get("numbers", {}).get(method_key, {}).get("overall", 0)
        wrds = all_results.get("words", {}).get(method_key, {}).get("overall", 0)
        method_name = (all_results.get("numbers", {}).get(method_key, {}).get("method", method_key)
                       or all_results.get("words", {}).get(method_key, {}).get("method", method_key))
        combined = (nums + wrds) / 2 if nums and wrds else 0
        print(f"\n  {method_name}:")
        print(f"    Numbers={nums:.1f}%  Words={wrds:.1f}%  Combined={combined:.1f}%")

    print(f"\n  Previous versions (for comparison, best mode):")
    print(f"    V22: Numbers 33.9% | Words 45.7% | Combined 39.8%")
    print(f"    V26: Numbers 45.8% | Words 49.4% | Combined 47.6%")
    print(f"    V27: Numbers 54.2% | Words 53.1% | Combined 53.7%")
    print(f"    V28 AdaBN Global: Numbers 55.9% | Words 60.5% | Combined 58.2%")
    print(f"    V29 AdaBN Global: Numbers 57.6% | Words 59.3% | Combined 58.4%")

    # Save results
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"v30_{args.model_type}_{ts_str}.json")

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
        "version": "v30",
        "model_type": args.model_type,
        "evaluation_type": "real_testers",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "techniques": {
            "temp_scale": args.temp_scale,
            "alpha_bn": args.alpha_bn,
            "tent": args.tent,
            "ensemble": args.ensemble,
            "conf_fusion": args.conf_fusion,
            "t3a": args.t3a,
            "ablation": args.ablation,
        },
        "results": make_serializable(all_results),
    }

    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
