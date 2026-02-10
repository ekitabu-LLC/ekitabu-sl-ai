#!/usr/bin/env python3
"""
KSL Training v19 (Alpine) - Bone Features + Anti-Overfitting + Multi-Scale TCN

Changes from v18 (based on confusion analysis 2026-02-09):
- ADDED: Bone features (child-parent vectors) -> in_channels 6->9 (xyz+vel+bone)
- FIXED: Velocity computed BEFORE temporal resampling (preserves original dynamics)
- FIXED: Numbers model reduced (hidden_dim 128->96, layers 6->5, ~3.5M params)
- ADDED: Mixup augmentation (beta=0.2) for regularization
- ADDED: Multi-scale temporal convolution (kernels 3, 5, 9)
- ADDED: Confusion-pair margin loss (replaces single anti-attractor)
- REMOVED: SWA (consistently hurt: 53.1%->47.5% over epochs)
- UPDATED: Confusion pairs based on v18 actual confusion matrix

Root causes addressed:
1. Classes 35/22, 388/48 nearly identical in xyz (cosine>0.999) -> bone features
2. 7.3M params for 375 samples -> overfit 99.7%/53.6% -> smaller model + mixup
3. Signer10 domain shift (hand Y offset 0.4) -> bones are position-invariant
4. SWA degraded accuracy -> removed

Usage:
    python train_ksl_v19.py --model-type numbers
    python train_ksl_v19.py --model-type words
    python train_ksl_v19.py --model-type both --seed 42
"""

import argparse
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

# Updated confusion pairs from v18 actual confusion matrix
NUMBERS_CONFUSION_PAIRS = [
    ("35", "22", 4.0),    # 64% confused - signer-dependent, nearly identical in xyz
    ("22", "66", 3.5),    # 60% confused
    ("388", "48", 3.5),   # 55% confused - cosine sim 0.9998
    ("444", "22", 3.0),   # 35% confused
    ("444", "100", 3.0),  # 25% confused
    ("73", "91", 3.0),    # 40% confused
    ("100", "54", 3.0),   # 32% confused
    ("125", "100", 3.0),  # 36% confused
    ("89", "9", 2.5),     # 16% confused
]

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

# ---------------------------------------------------------------------------
# Configs (v19 - bone features, smaller numbers model, no SWA)
# ---------------------------------------------------------------------------

NUMBERS_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,          # xyz(3) + velocity(3) + bone(3)
    "hidden_dim": 96,          # REDUCED from 128 (overfitting: 99.7% train vs 53.6% val)
    "num_layers": 5,           # REDUCED from 6
    "temporal_kernels": [3, 5, 9],  # NEW: multi-scale TCN
    "batch_size": 32,
    "epochs": 250,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.002,     # Slightly higher for smaller model
    "dropout": 0.35,           # Slightly lower for smaller model
    "label_smoothing": 0.1,
    "patience": 60,
    "warmup_epochs": 10,
    # Augmentation
    "hand_dropout_prob": 0.3,
    "hand_dropout_min": 0.1,
    "hand_dropout_max": 0.4,
    "complete_hand_drop_prob": 0.05,
    "noise_std": 0.01,
    "noise_prob": 0.3,
    # Mixup (NEW - strongest regularization for small datasets)
    "mixup_alpha": 0.2,
    # Confusion-pair margin loss (NEW - replaces single anti-attractor)
    "confusion_penalty_weight": 0.1,
    "confusion_penalty_start_epoch": 15,  # after warmup
}

WORDS_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 9,          # xyz(3) + velocity(3) + bone(3)
    "hidden_dim": 128,         # Keep for words (not overfitting as badly)
    "num_layers": 6,
    "temporal_kernels": [3, 5, 9],
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.001,
    "dropout": 0.4,
    "label_smoothing": 0.15,
    "patience": 50,
    "warmup_epochs": 10,
    "hand_dropout_prob": 0.5,
    "hand_dropout_min": 0.3,
    "hand_dropout_max": 0.7,
    "complete_hand_drop_prob": 0.1,
    "noise_std": 0.015,
    "noise_prob": 0.4,
    "mixup_alpha": 0.0,       # Words don't need mixup (87.5% already)
    "confusion_penalty_weight": 0.0,
    "confusion_penalty_start_epoch": 999,
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
# Dataset (bone features, velocity before resampling)
# ---------------------------------------------------------------------------

def compute_bones(h):
    """Compute bone vectors: bone[child] = node[child] - node[parent]."""
    bones = np.zeros_like(h)  # (T, 48, 3)
    for child in range(48):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


class KSLGraphDataset(Dataset):
    def __init__(self, data_dir, classes, config, aug=False):
        self.samples = []
        self.labels = []
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        for cn in classes:
            cd = os.path.join(data_dir, cn)
            if os.path.exists(cd):
                for fn in os.listdir(cd):
                    if fn.endswith(".npy"):
                        self.samples.append(os.path.join(cd, fn))
                        self.labels.append(self.c2i[cn])
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

        # --- Augmentation Step 1: Hand dropout ---
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

        # --- Augmentation Step 2: Complete hand drop ---
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # --- Augmentation Step 3: Hand swap (flips pose X too) ---
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        # --- Step 4: Centering ---
        lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
        rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01

        if np.any(lh_valid):
            h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
        if np.any(rh_valid):
            h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

        mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
        h[:, 42:48, :] -= mid_shoulder

        # --- Step 5: Separate normalization (hands vs pose) ---
        hand_data = h[:, :42, :]
        pose_data = h[:, 42:48, :]
        hand_max = np.abs(hand_data).max()
        pose_max = np.abs(pose_data).max()
        if hand_max > 0.01:
            h[:, :42, :] = hand_data / hand_max
        if pose_max > 0.01:
            h[:, 42:48, :] = pose_data / pose_max
        h = np.clip(h, -1, 1).astype(np.float32)

        # --- Augmentation Step 6: Scale jitter ---
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)

        # --- Augmentation Step 7: Gaussian noise ---
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        # --- Step 8: Compute velocity BEFORE resampling (preserves temporal detail) ---
        velocity = np.zeros_like(h)  # (f, 48, 3)
        velocity[1:] = h[1:] - h[:-1]

        # --- Step 9: Compute bone features (signer-invariant) ---
        bones = compute_bones(h)  # (f, 48, 3)

        # --- Step 10: Temporal sampling / padding ---
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

        # --- Augmentation Step 11: Temporal shift (pad+slice, no wrap) ---
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

        # --- Step 12: Concatenate all features ---
        # h: (mf, 48, 3) position
        # velocity: (mf, 48, 3) temporal differences
        # bones: (mf, 48, 3) structural (child-parent)
        features = np.concatenate([h, velocity, bones], axis=2)  # (mf, 48, 9)

        # Return (9, mf, 48) = (C, T, N)
        return torch.FloatTensor(features).permute(2, 0, 1), self.labels[idx]

# ---------------------------------------------------------------------------
# Multi-Scale Temporal Convolution (captures fast + slow patterns)
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module):
    """Parallel temporal conv branches with different kernel sizes."""

    def __init__(self, channels, kernels=(3, 5, 9), stride=1, dropout=0.3):
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
# Attention Pooling (from v18)
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
# Model (ST-GCN + Multi-Scale TCN + AttentionPool)
# ---------------------------------------------------------------------------

class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, ic, oc, adj, temporal_kernels=(3, 5, 9), st=1, dr=0.3):
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

    def forward(self, x):
        r = self.residual(x)
        b, c, t, n = x.shape
        x = self.gcn(x.permute(0, 2, 3, 1).reshape(b * t, n, c), self.adj)
        x = self.tcn(
            torch.relu(self.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2)))
        )
        return torch.relu(x + r)


class KSLGraphNet(nn.Module):
    def __init__(self, nc, nn_=48, ic=9, hd=96, nl=5, tk=(3, 5, 9), dr=0.3, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        ch = [ic] + [hd] * 2 + [hd * 2] * 2 + [hd * 4] * 2
        ch = ch[: nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk, 2 if i in [2, 4] else 1, dr)
             for i in range(nl)]
        )

        final_ch = ch[-1]

        # Attention pooling: average over nodes, then attend over time
        self.attn_pool = AttentionPool(final_ch, num_heads=2)

        # Classifier takes 2 * final_ch (from 2 attention heads)
        self.classifier = nn.Sequential(
            nn.Linear(2 * final_ch, hd),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hd, nc),
        )

    def forward(self, x):
        b, c, t, n = x.shape
        x = self.data_bn(
            x.permute(0, 1, 3, 2).reshape(b, c * n, t)
        ).reshape(b, c, n, t).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        # x: (B, C_final, T', N)
        x_node_avg = x.mean(dim=3)  # (B, C_final, T')
        features = self.attn_pool(x_node_avg)  # (B, 2 * C_final)
        logits = self.classifier(features)
        return logits

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device,
                confusion_pairs=None):
    """Train one split (numbers or words) with v19 improvements."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v19")
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
    tk = tuple(config.get("temporal_kernels", [9]))
    model = KSLGraphNet(
        len(classes),
        config["num_nodes"],
        config["in_channels"],
        config["hidden_dim"],
        config["num_layers"],
        tk,
        config["dropout"],
        adj,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Config: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, "
          f"dropout={config['dropout']}, wd={config['weight_decay']}, "
          f"ls={config['label_smoothing']}, in_ch={config['in_channels']}")
    print(f"[{ts()}] Temporal kernels: {tk}")
    print(f"[{ts()}] Mixup alpha: {config.get('mixup_alpha', 0)}")
    print(f"[{ts()}] Confusion penalty: {config.get('confusion_penalty_weight', 0)}")

    # Confusion pair setup
    conf_pairs = []
    conf_penalty_w = config.get("confusion_penalty_weight", 0)
    conf_start_ep = config.get("confusion_penalty_start_epoch", 999)
    if confusion_pairs and conf_penalty_w > 0:
        for true_name, pred_name, weight in confusion_pairs:
            ti = c2i.get(true_name, -1)
            pi = c2i.get(pred_name, -1)
            if ti >= 0 and pi >= 0:
                conf_pairs.append((ti, pi, weight))
                print(f"[{ts()}]   Confusion pair: {true_name}->{pred_name} "
                      f"(idx {ti}->{pi}, weight={weight})")
        print(f"[{ts()}]   {len(conf_pairs)} pairs, starts epoch {conf_start_ep}")

    # Optimizer + Cosine schedule with warmup
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    def sched_fn(ep):
        if ep < config["warmup_epochs"]:
            return ep / config["warmup_epochs"]
        p = (ep - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
        return config["min_lr"] / config["learning_rate"] + (
            1 - config["min_lr"] / config["learning_rate"]
        ) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        for d, t in train_ld:
            d, t = d.to(device), t.to(device)
            opt.zero_grad()

            # --- Mixup augmentation ---
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(d.size(0), device=device)
                d_mixed = lam * d + (1 - lam) * d[perm]
                t_perm = t[perm]

                logits = model(d_mixed)
                loss = (lam * F.cross_entropy(logits, t, label_smoothing=config["label_smoothing"])
                        + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"]))

                # For accuracy tracking, use the dominant label
                _, p = logits.max(1)
                tc += (lam * p.eq(t).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()
            else:
                logits = model(d)
                loss = F.cross_entropy(logits, t, label_smoothing=config["label_smoothing"])
                _, p = logits.max(1)
                tc += p.eq(t).sum().item()

            # --- Confusion-pair margin loss ---
            if conf_pairs and ep >= conf_start_ep:
                conf_loss = 0.0
                for ti, pi, weight in conf_pairs:
                    mask = (t == ti)
                    if mask.any():
                        # Penalize when confused-class logit exceeds true-class logit
                        margin = logits[mask, pi] - logits[mask, ti]
                        conf_loss += weight * F.softplus(margin).mean()
                loss = loss + conf_penalty_w * conf_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += loss.item()
            tt += t.size(0)

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

        va = 100.0 * vc / vt
        ta = 100.0 * tc / tt
        ep_time = time.time() - ep_start
        lr_now = opt.param_groups[0]["lr"]

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
                    "version": "v19",
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

    # Final evaluation with per-class breakdown
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for d, t in val_ld:
            _, p = model(d.to(device)).max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(t.cpu().numpy())

    print(f"\n[{ts()}] {name} Per-Class Results (final):")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_cls})")

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts)
    print(f"[{ts()}] {name} Overall: {ov:.1f}%")

    # Confusion matrix for numbers
    if name == "Numbers":
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

        # Report on tracked confusion pairs
        if confusion_pairs:
            print(f"\n[{ts()}] Tracked Confusion Pairs:")
            for true_name, pred_name, penalty in confusion_pairs:
                ti = c2i.get(true_name, -1)
                pi = c2i.get(pred_name, -1)
                if ti >= 0 and pi >= 0:
                    tot_cls = sum(1 for t in tgts if t == ti)
                    confused = cm[ti][pi]
                    rate = 100.0 * confused / tot_cls if tot_cls > 0 else 0.0
                    print(f"[{ts()}]   {true_name:>5s} -> {pred_name:<5s}: "
                          f"{confused}/{tot_cls} ({rate:.1f}%) [penalty={penalty}]")

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
        description="KSL ST-GCN Training v19 (Alpine HPC)",
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
    print(f"KSL Training v19 - Bone Features + Anti-Overfitting (Alpine HPC)")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nTrain dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Ckpt dir:  {args.checkpoint_dir}")

    results = {}
    start_time = time.time()

    if args.model_type in ("numbers", "both"):
        print(f"\n[{ts()}] Numbers Config:")
        print(json.dumps(NUMBERS_CONFIG, indent=2))
        ckpt_dir = os.path.join(args.checkpoint_dir, "v19_numbers")
        results["numbers"] = train_split(
            "Numbers", NUMBER_CLASSES, NUMBERS_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
            confusion_pairs=NUMBERS_CONFUSION_PAIRS,
        )

    if args.model_type in ("words", "both"):
        print(f"\n[{ts()}] Words Config:")
        print(json.dumps(WORDS_CONFIG, indent=2))
        ckpt_dir = os.path.join(args.checkpoint_dir, "v19_words")
        results["words"] = train_split(
            "Words", WORD_CLASSES, WORDS_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v19")
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
        f"v19_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v19",
        "model_type": args.model_type,
        "seed": args.seed,
        "numbers_config": NUMBERS_CONFIG if args.model_type in ("numbers", "both") else None,
        "words_config": WORDS_CONFIG if args.model_type in ("words", "both") else None,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v18": [
            "ADD: Bone features (child-parent vectors) -> in_channels 6->9",
            "FIX: Velocity computed BEFORE temporal resampling",
            "FIX: Numbers model reduced (hidden_dim 128->96, layers 6->5)",
            "ADD: Mixup augmentation (alpha=0.2) for numbers",
            "ADD: Multi-scale temporal conv (kernels 3, 5, 9)",
            "ADD: Confusion-pair margin loss (9 pairs from v18 confusion matrix)",
            "REMOVE: SWA (consistently degraded accuracy)",
            "UPDATE: Confusion pairs updated from v18 actual confusion matrix",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
