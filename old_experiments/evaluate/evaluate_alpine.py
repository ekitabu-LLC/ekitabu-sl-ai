#!/usr/bin/env python3
"""
Unified KSL Model Evaluation Script for Alpine HPC.

Evaluates any model version (v8-v14) with comprehensive metrics including
per-class accuracy, confusion matrix, F1 scores, and class 444 vs 54 tracking.

Usage:
    python evaluate_alpine.py --version v8 --val-dir /path/to/val_v2
    python evaluate_alpine.py --version v14 --model-type numbers --checkpoint-dir /path/to/checkpoints
    python evaluate_alpine.py --version v14 --model-type both --baseline
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Class Definitions
# ============================================================================

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])
WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

HARD_CLASSES = ["444", "35", "91", "100", "125"]
PROTECTED_CLASSES = ["Tortoise", "388", "9", "268"]
ATTRACTOR_VICTIMS = ["22", "444", "100", "125"]

V8_BASELINE = 0.8365  # 83.65%

# Graph structure for ST-GCN models (v10-v14)
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

# Hand landmark indices for feature engineering (v8/v9)
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17


# ============================================================================
# ST-GCN Model Architecture (v10-v14)
# ============================================================================

def build_adjacency_matrix(n=48):
    """Build normalized adjacency matrix for ST-GCN."""
    adj = np.zeros((n, n))
    for i, j in HAND_EDGES:
        adj[i, j] = adj[j, i] = 1
        adj[i + 21, j + 21] = adj[j + 21, i + 21] = 1
    for i, j in POSE_EDGES:
        adj[i, j] = adj[j, i] = 1
    adj[0, 21] = adj[21, 0] = 0.3
    adj += np.eye(n)
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0
    return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))


class GConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj, kernel_size=9, stride=1, dropout=0.3):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1),
                      padding=(kernel_size // 2, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        r = self.residual(x)
        b, c, t, n = x.shape
        x = self.gcn(x.permute(0, 2, 3, 1).reshape(b * t, n, c), self.adj)
        x = self.dropout(self.tcn(torch.relu(self.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2)))))
        return torch.relu(x + r)


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for v10-v14."""
    def __init__(self, num_classes, num_nodes=48, in_channels=3,
                 hidden_dim=128, num_layers=6, temporal_kernel=9,
                 dropout=0.3, adj=None):
        super().__init__()
        if adj is None:
            adj = build_adjacency_matrix(num_nodes)
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        channels = [in_channels] + [hidden_dim] * 2 + [hidden_dim * 2] * 2 + [hidden_dim * 4] * 2
        channels = channels[:num_layers + 1]
        self.layers = nn.ModuleList([
            STGCNBlock(channels[i], channels[i + 1], adj, temporal_kernel,
                       stride=2 if i in [2, 4] else 1, dropout=dropout)
            for i in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        b, c, t, n = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * n, t))
        x = x.reshape(b, c, n, t).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(self.pool(x).view(b, -1))


# ============================================================================
# TemporalPyramid Model Architecture (v8/v9)
# ============================================================================

class TemporalPyramid(nn.Module):
    """Temporal Pyramid model for v8/v9."""
    def __init__(self, num_classes, feature_dim=649, hidden_dim=320, dropout=0.35):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.temporal_scales = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                          nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3),
                          nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7),
                          nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
        ])
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attention_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, 1))
            for _ in range(4)
        ])
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        # Anti-attractor head (present in saved checkpoints)
        self.anti_attractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(self.input_norm(x))
        x_conv = x.transpose(1, 2)
        multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
        x = self.temporal_fusion(multi_scale)
        lstm_out, _ = self.lstm(x)
        contexts = [(lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1)
                     for head in self.attention_heads]
        return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))


# ============================================================================
# Feature Engineering (v8/v9)
# ============================================================================

def compute_hand_features(hand_landmarks):
    """Compute hand geometry features per frame."""
    frames = hand_landmarks.shape[0]
    features = []
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    for f in range(frames):
        hand = hand_landmarks[f].reshape(21, 3)
        frame_features = []
        wrist = hand[WRIST]
        hand_normalized = hand - wrist
        for tip in tips:
            frame_features.append(np.linalg.norm(hand_normalized[tip]))
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                frame_features.append(np.linalg.norm(hand[tips[i]] - hand[tips[j]]))
        for tip, mcp in zip(tips, mcps):
            frame_features.append(np.linalg.norm(hand[tip] - hand[mcp]))
        max_spread = max(
            np.linalg.norm(hand[tips[i]] - hand[tips[j]])
            for i in range(len(tips)) for j in range(i + 1, len(tips))
        )
        frame_features.append(max_spread)
        thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
        index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
        norm_product = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
        cos_angle = (np.clip(np.dot(thumb_vec, index_vec) / norm_product, -1, 1)
                     if norm_product > 1e-8 else 0)
        frame_features.append(cos_angle)
        v1 = hand[INDEX_MCP] - hand[WRIST]
        v2 = hand[PINKY_MCP] - hand[WRIST]
        palm_normal = np.cross(v1, v2)
        norm = np.linalg.norm(palm_normal)
        palm_normal = palm_normal / norm if norm > 1e-8 else np.zeros(3)
        frame_features.extend(palm_normal.tolist())
        features.append(frame_features)
    return np.array(features, dtype=np.float32)


def compute_temporal_features(data):
    """Compute temporal discriminative features (v9 only)."""
    frames = data.shape[0]
    features = []
    features.append(frames / 90.0)
    segment_size = frames // 3
    if segment_size >= 5:
        segments = [data[i * segment_size:(i + 1) * segment_size].mean(axis=0) for i in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                norm_i, norm_j = np.linalg.norm(segments[i]), np.linalg.norm(segments[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    features.append(np.dot(segments[i], segments[j]) / (norm_i * norm_j))
                else:
                    features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])
    right_hand = data[:, 162:225]
    velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
    features.append(velocities.mean() if len(velocities) > 0 else 0.0)
    features.append(velocities.std() if len(velocities) > 0 else 0.0)
    dip_ratio = (np.sum(velocities < velocities.mean() * 0.3) / len(velocities)
                 if len(velocities) > 0 else 0.0)
    features.append(dip_ratio)
    right_hand_3d = right_hand.reshape(-1, 21, 3)
    spread = np.linalg.norm(right_hand_3d[:, 4] - right_hand_3d[:, 20], axis=1)
    peaks = 0
    for i in range(1, len(spread) - 1):
        if spread[i] > spread[i - 1] and spread[i] > spread[i + 1]:
            if spread[i] > spread.mean() * 0.8:
                peaks += 1
    features.append(peaks / 10.0)
    return np.array(features, dtype=np.float32)


def engineer_features_v8(data):
    """Feature engineering for v8 (feature_dim=649)."""
    left_hand, right_hand = data[:, 99:162], data[:, 162:225]
    left_features = compute_hand_features(left_hand)
    right_features = compute_hand_features(right_hand)
    left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
    left_vel[1:] = left_features[1:] - left_features[:-1]
    right_vel[1:] = right_features[1:] - right_features[:-1]
    return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)


def engineer_features_v9(data):
    """Feature engineering for v9 (feature_dim=657)."""
    left_hand, right_hand = data[:, 99:162], data[:, 162:225]
    left_features = compute_hand_features(left_hand)
    right_features = compute_hand_features(right_hand)
    left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
    left_vel[1:] = left_features[1:] - left_features[:-1]
    right_vel[1:] = right_features[1:] - right_features[:-1]
    temporal = compute_temporal_features(data)
    temporal_broadcast = np.tile(temporal, (data.shape[0], 1))
    return np.concatenate([data, left_features, right_features, left_vel, right_vel,
                           temporal_broadcast], axis=1)


# ============================================================================
# Preprocessing for ST-GCN (v10-v14)
# ============================================================================

def preprocess_stgcn(data, max_frames=90):
    """Preprocess raw landmark data into ST-GCN input format: (1, 3, T, 48)."""
    f = data.shape[0]
    if data.shape[1] >= 225:
        pose = np.zeros((f, 6, 3), dtype=np.float32)
        for pi, idx_pose in enumerate(POSE_INDICES):
            start = idx_pose * 3
            pose[:, pi, :] = data[:, start:start + 3]
        lh = data[:, 99:162].reshape(f, 21, 3)
        rh = data[:, 162:225].reshape(f, 21, 3)
    else:
        pose = np.zeros((f, 6, 3), dtype=np.float32)
        lh = np.zeros((f, 21, 3), dtype=np.float32)
        rh = np.zeros((f, 21, 3), dtype=np.float32)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
    if np.any(rh_valid):
        h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

    mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
    h[:, 42:48, :] -= mid_shoulder

    max_val = np.abs(h).max()
    if max_val > 0.01:
        h = np.clip(h / max_val, -1, 1).astype(np.float32)

    if f >= max_frames:
        h = h[np.linspace(0, f - 1, max_frames, dtype=int)]
    else:
        h = np.concatenate([h, np.zeros((max_frames - f, 48, 3), dtype=np.float32)])

    return torch.FloatTensor(h).permute(2, 0, 1).unsqueeze(0)


# ============================================================================
# Checkpoint Resolution
# ============================================================================

def get_checkpoint_paths(version, model_type, checkpoint_dir):
    """Return list of (checkpoint_path, classes, model_type_label) tuples."""
    version = version.lower()
    paths = []

    if version in ("v8", "v9"):
        # Single combined model
        if version == "v8":
            ckpt = os.path.join(checkpoint_dir, f"checkpoints_{version}", f"ksl_{version}_best.pth")
        else:
            ckpt = os.path.join(checkpoint_dir, f"checkpoints_{version}", f"ksl_{version}_best.pth")
        paths.append((ckpt, ALL_CLASSES, "both"))

    elif version == "v10":
        # Single combined model
        ckpt = os.path.join(checkpoint_dir, "checkpoints_v10", "best_model.pt")
        paths.append((ckpt, ALL_CLASSES, "both"))

    else:
        # v11-v14: split models
        if model_type in ("numbers", "both"):
            ckpt = os.path.join(checkpoint_dir, f"checkpoints_{version}_numbers", "best_model.pt")
            paths.append((ckpt, NUMBER_CLASSES, "numbers"))
        if model_type in ("words", "both"):
            ckpt = os.path.join(checkpoint_dir, f"checkpoints_{version}_words", "best_model.pt")
            paths.append((ckpt, WORD_CLASSES, "words"))

    return paths


def load_model(version, checkpoint_path, classes, device):
    """Load a model from checkpoint."""
    version = version.lower()
    num_classes = len(classes)

    if version in ("v8", "v9"):
        feature_dim = 657 if version == "v9" else 649
        model = TemporalPyramid(num_classes=num_classes, feature_dim=feature_dim,
                                hidden_dim=320, dropout=0.35)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {
            "saved_acc": ckpt.get("val_acc", 0),
            "epoch": ckpt.get("epoch", "N/A"),
        }
    else:
        # v10-v14: STGCN
        adj = build_adjacency_matrix(48).to(device)
        dropout = 0.5 if version == "v14" else 0.3
        model = STGCN(num_classes=num_classes, num_nodes=48, in_channels=3,
                       hidden_dim=128, num_layers=6, temporal_kernel=9,
                       dropout=dropout, adj=adj)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        meta = {
            "saved_acc": ckpt.get("val_acc", 0),
            "epoch": ckpt.get("epoch", "N/A"),
        }

    model = model.to(device)
    model.eval()
    return model, meta


# ============================================================================
# Data Loading
# ============================================================================

def load_val_samples(val_dir, classes):
    """Load validation sample paths."""
    samples = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for class_name in classes:
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            if fn.endswith(".npy"):
                samples.append({
                    "path": os.path.join(class_dir, fn),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                })
    return samples, class_to_idx


def compute_normalization(samples, max_samples=200):
    """Compute mean/std from a subset of samples (for v8/v9)."""
    subset = samples[:max_samples]
    all_data = np.concatenate([np.load(s["path"]).flatten() for s in subset])
    return all_data.mean(), all_data.std() + 1e-8


# ============================================================================
# Inference
# ============================================================================

def prepare_input_v8v9(data, mean, std, version, max_frames=90):
    """Prepare input tensor for v8/v9 TemporalPyramid models."""
    data = (data - mean) / std
    if data.shape[0] > max_frames:
        data = data[np.linspace(0, data.shape[0] - 1, max_frames, dtype=int)]
    elif data.shape[0] < max_frames:
        data = np.vstack([data, np.zeros((max_frames - data.shape[0], data.shape[1]),
                                          dtype=np.float32)])
    if version == "v9":
        features = engineer_features_v9(data)
    else:
        features = engineer_features_v8(data)
    return torch.from_numpy(features).unsqueeze(0)


def run_evaluation(model, samples, classes, version, device, mean=None, std=None):
    """Run evaluation on all samples and return detailed results."""
    correct = 0
    top3_correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sample in samples:
            data = np.load(sample["path"]).astype(np.float32)
            label = sample["label"]
            class_name = sample["class_name"]

            if data.shape[0] < 5:
                per_class_total[class_name] += 1
                confusion[class_name]["__skip__"] += 1
                total += 1
                all_preds.append(-1)
                all_labels.append(label)
                continue

            if version in ("v8", "v9"):
                inp = prepare_input_v8v9(data, mean, std, version).to(device)
            else:
                inp = preprocess_stgcn(data, max_frames=90).to(device)

            logits = model(inp)
            pred = logits.argmax(dim=1).item()
            pred_class = classes[pred]

            per_class_total[class_name] += 1
            confusion[class_name][pred_class] += 1
            total += 1
            all_preds.append(pred)
            all_labels.append(label)

            if pred == label:
                correct += 1
                per_class_correct[class_name] += 1

            _, top3 = logits.topk(min(3, logits.shape[1]), dim=1)
            if label in top3.squeeze().cpu().numpy():
                top3_correct += 1

    return {
        "correct": correct,
        "top3_correct": top3_correct,
        "total": total,
        "per_class_correct": dict(per_class_correct),
        "per_class_total": dict(per_class_total),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(results, classes):
    """Compute comprehensive metrics from raw results."""
    metrics = {}
    total = results["total"]
    if total == 0:
        return {"error": "No samples evaluated"}

    metrics["overall_accuracy"] = results["correct"] / total
    metrics["top3_accuracy"] = results["top3_correct"] / total

    # Per-class accuracy and F1
    per_class = {}
    for c in classes:
        tp = results["per_class_correct"].get(c, 0)
        total_c = results["per_class_total"].get(c, 0)
        acc = tp / total_c if total_c > 0 else 0.0

        # F1 score: need precision and recall
        # Recall = TP / (TP + FN) = TP / total_true
        recall = tp / total_c if total_c > 0 else 0.0

        # Precision = TP / (TP + FP)
        # FP = times predicted as c but true class was different
        fp = 0
        for true_c, pred_dict in results["confusion"].items():
            if true_c != c:
                fp += pred_dict.get(c, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_class[c] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": tp,
            "total": total_c,
        }
    metrics["per_class"] = per_class

    # Category accuracy
    num_correct = sum(results["per_class_correct"].get(c, 0) for c in NUMBER_CLASSES if c in classes)
    num_total = sum(results["per_class_total"].get(c, 0) for c in NUMBER_CLASSES if c in classes)
    word_correct = sum(results["per_class_correct"].get(c, 0) for c in WORD_CLASSES if c in classes)
    word_total = sum(results["per_class_total"].get(c, 0) for c in WORD_CLASSES if c in classes)
    metrics["number_accuracy"] = num_correct / num_total if num_total > 0 else 0.0
    metrics["word_accuracy"] = word_correct / word_total if word_total > 0 else 0.0

    # Hard classes accuracy
    hard_correct = sum(results["per_class_correct"].get(c, 0) for c in HARD_CLASSES if c in classes)
    hard_total = sum(results["per_class_total"].get(c, 0) for c in HARD_CLASSES if c in classes)
    metrics["hard_class_accuracy"] = hard_correct / hard_total if hard_total > 0 else 0.0

    # Protected classes accuracy
    prot_correct = sum(results["per_class_correct"].get(c, 0) for c in PROTECTED_CLASSES if c in classes)
    prot_total = sum(results["per_class_total"].get(c, 0) for c in PROTECTED_CLASSES if c in classes)
    metrics["protected_class_accuracy"] = prot_correct / prot_total if prot_total > 0 else 0.0

    # Attractor analysis (444 vs 54)
    attractor_to_54 = 0
    attractor_total = 0
    for c in ATTRACTOR_VICTIMS:
        if c in results["confusion"]:
            attractor_total += results["per_class_total"].get(c, 0)
            attractor_to_54 += results["confusion"].get(c, {}).get("54", 0)
    metrics["attractor_to_54_count"] = attractor_to_54
    metrics["attractor_total"] = attractor_total
    metrics["attractor_confusion_rate"] = (attractor_to_54 / attractor_total
                                            if attractor_total > 0 else 0.0)

    # Class 444 specific analysis
    if "444" in results["per_class_total"]:
        c444_total = results["per_class_total"]["444"]
        c444_correct = results["per_class_correct"].get("444", 0)
        c444_to_54 = results["confusion"].get("444", {}).get("54", 0)
        metrics["class_444"] = {
            "accuracy": c444_correct / c444_total if c444_total > 0 else 0.0,
            "correct": c444_correct,
            "total": c444_total,
            "confused_as_54": c444_to_54,
            "confusion_rate_with_54": c444_to_54 / c444_total if c444_total > 0 else 0.0,
        }

    # Top confusion pairs
    pairs = []
    for true_c in classes:
        for pred_c in classes:
            if true_c != pred_c:
                count = results["confusion"].get(true_c, {}).get(pred_c, 0)
                if count > 0:
                    pairs.append({
                        "true": true_c,
                        "predicted": pred_c,
                        "count": count,
                        "is_attractor": pred_c == "54",
                    })
    pairs.sort(key=lambda x: -x["count"])
    metrics["top_confusions"] = pairs[:20]

    # Macro F1
    f1_values = [per_class[c]["f1"] for c in classes if per_class[c]["total"] > 0]
    metrics["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0

    # Confusion matrix as 2D list
    metrics["confusion_matrix"] = {
        "classes": classes,
        "matrix": [[results["confusion"].get(t, {}).get(p, 0) for p in classes]
                    for t in classes],
    }

    return metrics


# ============================================================================
# Output Formatting
# ============================================================================

def print_summary(metrics, version, model_type_label, show_baseline=False):
    """Print human-readable summary to stdout."""
    print()
    print("=" * 74)
    print(f"  KSL {version.upper()} Evaluation Results ({model_type_label})")
    print("=" * 74)

    overall = metrics["overall_accuracy"] * 100
    top3 = metrics["top3_accuracy"] * 100
    print(f"\n  Overall Accuracy:  {overall:6.2f}%")
    print(f"  Top-3 Accuracy:    {top3:6.2f}%")
    print(f"  Macro F1 Score:    {metrics['macro_f1']:.4f}")

    if show_baseline:
        baseline = V8_BASELINE * 100
        delta = overall - baseline
        indicator = "+" if delta >= 0 else ""
        print(f"  vs v8 Baseline:    {indicator}{delta:.2f}%  (baseline: {baseline:.2f}%)")

    print(f"\n  Numbers Accuracy:  {metrics['number_accuracy']*100:6.2f}%")
    print(f"  Words Accuracy:    {metrics['word_accuracy']*100:6.2f}%")
    print(f"  Hard Classes:      {metrics['hard_class_accuracy']*100:6.2f}%")
    print(f"  Protected Classes: {metrics['protected_class_accuracy']*100:6.2f}%")

    # Attractor analysis
    if metrics["attractor_total"] > 0:
        print(f"\n  Attractor Analysis (class 54):")
        print(f"    Victims -> 54:  {metrics['attractor_to_54_count']}/{metrics['attractor_total']}"
              f" ({metrics['attractor_confusion_rate']*100:.1f}%)")

    if "class_444" in metrics:
        c444 = metrics["class_444"]
        print(f"    Class 444:      {c444['correct']}/{c444['total']}"
              f" ({c444['accuracy']*100:.1f}% acc,"
              f" {c444['confusion_rate_with_54']*100:.1f}% confused as 54)")

    # Per-class table
    per_class = metrics["per_class"]
    classes = list(per_class.keys())
    number_classes = [c for c in classes if c in NUMBER_CLASSES]
    word_classes = [c for c in classes if c in WORD_CLASSES]

    print(f"\n{'='*74}")
    print(f"  Per-Class Results")
    print(f"{'='*74}")
    print(f"  {'Class':>12}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'N':>4}  Notes")
    print(f"  {'-'*66}")

    if number_classes:
        print(f"  NUMBERS:")
        for c in sorted(number_classes):
            pc = per_class[c]
            notes = []
            if c in HARD_CLASSES:
                notes.append("HARD")
            if c in PROTECTED_CLASSES:
                notes.append("PROTECTED")
            if c in ATTRACTOR_VICTIMS:
                notes.append("ATTRACTOR_VICTIM")
            if c == "54":
                notes.append("ATTRACTOR")
            note_str = ", ".join(notes)
            print(f"  {c:>12}  {pc['accuracy']*100:6.1f}%  {pc['precision']*100:6.1f}%"
                  f"  {pc['recall']*100:6.1f}%  {pc['f1']:.4f}  {pc['total']:4d}  {note_str}")

    if word_classes:
        print(f"  WORDS:")
        for c in sorted(word_classes):
            pc = per_class[c]
            notes = []
            if c in HARD_CLASSES:
                notes.append("HARD")
            if c in PROTECTED_CLASSES:
                notes.append("PROTECTED")
            note_str = ", ".join(notes)
            print(f"  {c:>12}  {pc['accuracy']*100:6.1f}%  {pc['precision']*100:6.1f}%"
                  f"  {pc['recall']*100:6.1f}%  {pc['f1']:.4f}  {pc['total']:4d}  {note_str}")

    # Zero accuracy classes
    zero_classes = [c for c in classes if per_class[c]["total"] > 0 and per_class[c]["accuracy"] == 0]
    if zero_classes:
        print(f"\n  WARNING: {len(zero_classes)} class(es) at 0% accuracy: {', '.join(zero_classes)}")

    # Top confusions
    if metrics["top_confusions"]:
        print(f"\n{'='*74}")
        print(f"  Top Confusion Pairs")
        print(f"{'='*74}")
        for pair in metrics["top_confusions"][:15]:
            marker = "  *** ATTRACTOR" if pair["is_attractor"] else ""
            print(f"  {pair['true']:>12} -> {pair['predicted']:<12}  {pair['count']:3d}{marker}")

    print(f"\n{'='*74}\n")


def save_results(all_metrics, version, model_type, output_dir):
    """Save all metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"eval_{version}_{model_type}.json")

    output = {
        "version": version,
        "model_type": model_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": all_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    return output_path


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def evaluate(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    version = args.version.lower()

    checkpoint_paths = get_checkpoint_paths(version, args.model_type, args.checkpoint_dir)
    if not checkpoint_paths:
        print(f"ERROR: No checkpoint paths resolved for {version} / {args.model_type}")
        sys.exit(1)

    all_metrics = {}

    for ckpt_path, classes, model_type_label in checkpoint_paths:
        print(f"\n{'='*74}")
        print(f"Evaluating {version} - {model_type_label}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Classes: {len(classes)}")
        print(f"{'='*74}")

        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            all_metrics[model_type_label] = {"error": f"Checkpoint not found: {ckpt_path}"}
            continue

        # Load model
        model, meta = load_model(version, ckpt_path, classes, device)
        print(f"Loaded model (saved acc: {meta['saved_acc']:.2f}%, epoch: {meta['epoch']})"
              if isinstance(meta["saved_acc"], float) and meta["saved_acc"] < 1
              else f"Loaded model (saved acc: {meta['saved_acc']}%, epoch: {meta['epoch']})")

        # Load validation data
        samples, class_to_idx = load_val_samples(args.val_dir, classes)
        print(f"Validation samples: {len(samples)}")

        if len(samples) == 0:
            print(f"ERROR: No validation samples found in {args.val_dir}")
            all_metrics[model_type_label] = {"error": "No validation samples"}
            continue

        # Compute normalization for v8/v9
        mean, std = None, None
        if version in ("v8", "v9"):
            mean, std = compute_normalization(samples)

        # Run evaluation
        results = run_evaluation(model, samples, classes, version, device, mean, std)
        metrics = compute_metrics(results, classes)
        metrics["checkpoint_meta"] = meta

        # Print summary
        print_summary(metrics, version, model_type_label, show_baseline=args.baseline)

        all_metrics[model_type_label] = metrics

    # For split models (v11-v14 with --model-type both), compute combined accuracy
    if len(checkpoint_paths) > 1 and args.model_type == "both":
        total_correct = sum(
            m.get("per_class", {}).get(c, {}).get("correct", 0)
            for m in all_metrics.values() if "per_class" in m
            for c in m["per_class"]
        )
        total_samples = sum(
            m.get("per_class", {}).get(c, {}).get("total", 0)
            for m in all_metrics.values() if "per_class" in m
            for c in m["per_class"]
        )
        if total_samples > 0:
            combined_acc = total_correct / total_samples
            print(f"{'='*74}")
            print(f"  Combined Accuracy ({version}): {combined_acc*100:.2f}%"
                  f" ({total_correct}/{total_samples})")
            if args.baseline:
                delta = combined_acc - V8_BASELINE
                indicator = "+" if delta >= 0 else ""
                print(f"  vs v8 Baseline: {indicator}{delta*100:.2f}%")
            print(f"{'='*74}\n")
            all_metrics["combined"] = {
                "overall_accuracy": combined_acc,
                "total_correct": total_correct,
                "total_samples": total_samples,
            }

    # Save results
    save_results(all_metrics, version, args.model_type, args.output_dir)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified KSL Model Evaluation for Alpine HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate v8 (single model, 30 classes)
  python evaluate_alpine.py --version v8 --val-dir /path/to/val_v2

  # Evaluate v14 numbers model only
  python evaluate_alpine.py --version v14 --model-type numbers

  # Evaluate v14 both models with baseline comparison
  python evaluate_alpine.py --version v14 --model-type both --baseline

  # Custom checkpoint and output directories
  python evaluate_alpine.py --version v12 --model-type both \\
      --checkpoint-dir /scratch/alpine/user/checkpoints \\
      --val-dir /scratch/alpine/user/data/val_v2 \\
      --output-dir /scratch/alpine/user/results
""",
    )
    parser.add_argument("--version", type=str, required=True,
                        choices=["v8", "v9", "v10", "v11", "v12", "v13", "v14"],
                        help="Model version to evaluate")
    parser.add_argument("--model-type", type=str, default="both",
                        choices=["numbers", "words", "both"],
                        help="Which model type to evaluate (default: both)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data",
                        help="Base directory containing checkpoint subdirectories")
    parser.add_argument("--val-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/val_v2",
                        help="Validation data directory with per-class subdirectories")
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/results",
                        help="Directory to save JSON evaluation results")
    parser.add_argument("--baseline", action="store_true",
                        help="Compare against v8 baseline (83.65%%)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
