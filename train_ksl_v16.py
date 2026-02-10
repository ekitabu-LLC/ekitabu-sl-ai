#!/usr/bin/env python3
"""
KSL Training v16 (Alpine) - ST-GCN with Confusion-Aware Training

Combines v15's ST-GCN architecture with v8's proven training tricks:
- Separate configs for Numbers (gentle augmentation) vs Words (stronger augmentation)
- ConfusionAwareFocalLoss with pair penalties (numbers only)
- Anti-attractor head targeting class 54 (numbers only)
- Mixup augmentation (numbers only)
- Supervised contrastive loss for confusable pairs (numbers only)
- Hard-class boosting in sampler weights (numbers only)
- Reduced model capacity for numbers (hidden_dim=96, 5 layers)

Usage:
    python train_ksl_v16.py --model-type numbers
    python train_ksl_v16.py --model-type words
    python train_ksl_v16.py --model-type both
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Timestamp helper for SLURM log readability
# ---------------------------------------------------------------------------

def ts():
    """Return formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

# ---------------------------------------------------------------------------
# Confusion Pairs for Numbers (true_class, wrong_prediction, penalty)
# These are class NAME strings; mapped to indices at runtime.
# ---------------------------------------------------------------------------

NUMBERS_CONFUSION_PAIRS = [
    ("444", "54", 3.5),
    ("35", "54", 3.0),
    ("89", "9", 3.0),
    ("73", "91", 2.5),
    ("35", "100", 2.0),
    ("100", "54", 2.0),
    ("125", "54", 2.0),
]

# Hard classes that get boosted sampling weight (numbers only)
HARD_CLASSES = {
    "444": 2.0,
    "35": 2.0,
    "89": 1.8,
    "91": 1.8,
    "73": 1.8,
    "100": 1.5,
    "125": 1.3,
}

# ---------------------------------------------------------------------------
# Separate Configs for Numbers vs Words
# ---------------------------------------------------------------------------

NUMBERS_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "hidden_dim": 96,
    "num_layers": 5,
    "temporal_kernel": 9,
    "batch_size": 32,
    "epochs": 250,
    "learning_rate": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "dropout": 0.35,
    "label_smoothing": 0.1,
    "patience": 50,
    "warmup_epochs": 15,
    "hand_dropout_prob": 0.15,
    "hand_dropout_min": 0.05,
    "hand_dropout_max": 0.2,
    "complete_hand_drop_prob": 0.0,
    "noise_std": 0.006,
    "noise_prob": 0.3,
    # Focal loss
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    # Mixup
    "mixup_alpha": 0.2,
    "mixup_prob": 0.3,
    # Contrastive loss
    "contrastive_loss_weight": 0.3,
    # Anti-attractor
    "anti_attractor_weight": 0.1,
    "anti_attractor_class": "54",
    # Hard class boosting
    "use_hard_class_boosting": True,
}

WORDS_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "hidden_dim": 128,
    "num_layers": 6,
    "temporal_kernel": 9,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.001,
    "dropout": 0.4,
    "label_smoothing": 0.15,
    "patience": 40,
    "warmup_epochs": 10,
    "hand_dropout_prob": 0.5,
    "hand_dropout_min": 0.3,
    "hand_dropout_max": 0.7,
    "complete_hand_drop_prob": 0.1,
    "noise_std": 0.015,
    "noise_prob": 0.4,
    # No focal loss for words
    "use_focal_loss": False,
    "focal_gamma": 0.0,
    # No mixup for words
    "mixup_alpha": 0.0,
    "mixup_prob": 0.0,
    # No contrastive loss for words
    "contrastive_loss_weight": 0.0,
    # No anti-attractor for words
    "anti_attractor_weight": 0.0,
    "anti_attractor_class": None,
    # No hard class boosting
    "use_hard_class_boosting": False,
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
# Dataset
# ---------------------------------------------------------------------------

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

        # Hand dropout augmentation (config-driven)
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

        # Complete hand drop
        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        # Hand swap augmentation
        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]

        # Normalization
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

        # Temporal padding / sampling
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]
        else:
            h = np.concatenate([h, np.zeros((self.mf - f, 48, 3), dtype=np.float32)])

        # Additional augmentations (config-driven noise)
        if self.aug:
            if np.random.random() < 0.5:
                h = h * np.random.uniform(0.8, 1.2)
            if np.random.random() < self.config.get("noise_prob", 0.4):
                noise = np.random.normal(0, self.config.get("noise_std", 0.03), h.shape).astype(np.float32)
                h = h + noise
            if np.random.random() < 0.3:
                shift = np.random.randint(-8, 9)
                h = np.roll(h, shift, axis=0)

        return torch.FloatTensor(h).permute(2, 0, 1), self.labels[idx]

# ---------------------------------------------------------------------------
# Model Architecture (ST-GCN)
# ---------------------------------------------------------------------------

class GConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    def __init__(self, ic, oc, adj, ks=9, st=1, dr=0.3):
        super().__init__()
        self.register_buffer("adj", adj)
        self.gcn = GConv(ic, oc)
        self.bn1 = nn.BatchNorm2d(oc)
        self.tcn = nn.Sequential(
            nn.Conv2d(oc, oc, (ks, 1), padding=(ks // 2, 0), stride=(st, 1)),
            nn.BatchNorm2d(oc),
        )
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
        x = self.dropout(
            self.tcn(torch.relu(self.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2))))
        )
        return torch.relu(x + r)


class KSLGraphNet(nn.Module):
    """ST-GCN model with optional anti-attractor head."""

    def __init__(self, nc, nn_=48, ic=3, hd=128, nl=6, tk=9, dr=0.3, adj=None,
                 use_anti_attractor=False):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)

        # Channel progression: adapts to num_layers
        ch = [ic] + [hd] * 2 + [hd * 2] * 2 + [hd * 4] * 2
        ch = ch[: nl + 1]

        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk, 2 if i in [2, 4] else 1, dr) for i in range(nl)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        final_ch = ch[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_ch, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc)
        )

        # Anti-attractor head (optional, for numbers model)
        self.use_anti_attractor = use_anti_attractor
        if use_anti_attractor:
            self.anti_attractor = nn.Sequential(
                nn.Linear(final_ch, hd // 2),
                nn.GELU(),
                nn.Linear(hd // 2, 1),
            )

    def forward(self, x, return_features=False):
        b, c, t, n = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * n, t)).reshape(b, c, n, t).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        features = self.pool(x).view(b, -1)
        logits = self.classifier(features)

        if return_features:
            anti_score = None
            if self.use_anti_attractor:
                anti_score = self.anti_attractor(features)
            return logits, features, anti_score

        return logits

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class ConfusionAwareFocalLoss(nn.Module):
    """Focal loss with label smoothing, class weights, and confusion pair penalties."""

    def __init__(self, num_classes, class_weights, confusion_pairs, class_to_idx,
                 gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("class_weights", class_weights)

        # Build confusion penalty matrix
        penalty_matrix = torch.ones(num_classes, num_classes)
        for true_name, pred_name, penalty in confusion_pairs:
            if true_name in class_to_idx and pred_name in class_to_idx:
                ti = class_to_idx[true_name]
                pi = class_to_idx[pred_name]
                penalty_matrix[ti, pi] = penalty
        self.register_buffer("penalty_matrix", penalty_matrix)

    def forward(self, inputs, targets):
        # Label smoothing targets
        smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / self.num_classes

        # Focal loss
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        focal_weights = (1 - probs) ** self.gamma

        base_loss = -focal_weights * smooth_targets * log_probs
        base_loss = base_loss.sum(dim=-1)

        # Class weights
        if self.class_weights is not None:
            base_loss = base_loss * self.class_weights[targets]

        # Differentiable confusion penalty using softmax probabilities
        confusion_penalty = torch.zeros_like(base_loss)
        softmax_probs = F.softmax(inputs, dim=1)
        for i in range(self.num_classes):
            mask = (targets == i)
            if mask.sum() == 0:
                continue
            for j in range(self.num_classes):
                if i == j:
                    continue
                p = self.penalty_matrix[i, j].item()
                if p > 1.0:
                    confusion_penalty[mask] += (p - 1.0) * softmax_probs[mask, j]

        return (base_loss + confusion_penalty).mean()


class SupervisedContrastiveLoss(nn.Module):
    """Contrastive loss that pushes apart embeddings of confusable class pairs."""

    def __init__(self, confusion_pairs, class_to_idx, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # Build set of confusable pair indices
        self.confusable_pairs = set()
        for true_name, pred_name, _ in confusion_pairs:
            if true_name in class_to_idx and pred_name in class_to_idx:
                ti = class_to_idx[true_name]
                pi = class_to_idx[pred_name]
                self.confusable_pairs.add((ti, pi))
                self.confusable_pairs.add((pi, ti))

        # Collect all involved class indices
        self.involved_classes = set()
        for a, b in self.confusable_pairs:
            self.involved_classes.add(a)
            self.involved_classes.add(b)

    def forward(self, features, targets):
        """
        Supervised contrastive: for samples from confusable classes,
        pull same-class together, push different-class apart.
        """
        # Only operate on samples from involved classes
        mask = torch.zeros(len(targets), dtype=torch.bool, device=targets.device)
        for c in self.involved_classes:
            mask |= (targets == c)

        if mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)

        feat = F.normalize(features[mask], dim=1)
        labs = targets[mask]

        # Pairwise similarity
        sim = torch.mm(feat, feat.t()) / self.temperature

        # Same-class mask (excluding diagonal)
        same_class = labs.unsqueeze(0) == labs.unsqueeze(1)
        same_class.fill_diagonal_(False)

        if same_class.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # Log-softmax over all pairs (excluding self)
        diag_mask = torch.eye(len(labs), dtype=torch.bool, device=features.device)
        sim.masked_fill_(diag_mask, float('-inf'))
        log_prob = F.log_softmax(sim, dim=1)

        # Mean of log-prob for same-class pairs
        loss = -(log_prob * same_class.float()).sum(dim=1) / same_class.float().sum(dim=1).clamp(min=1)
        return loss.mean()

# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_split(name, classes, config, train_dir, val_dir, ckpt_dir, device,
                confusion_pairs=None, writer=None):
    """Train one split (numbers or words) using the provided config."""

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - v16")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLGraphDataset(train_dir, classes, config, aug=True)
    val_ds = KSLGraphDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found in {train_dir}")
        return None

    c2i = train_ds.c2i
    i2c = {v: k for k, v in c2i.items()}

    # Class weights for loss
    counts = Counter(train_ds.labels)
    tot = len(train_ds)
    w = torch.ones(len(classes))
    for i, c in counts.items():
        w[i] = min(2.0, tot / (len(counts) * c))
    w = w.to(device)

    # Sampler weights with optional hard-class boosting
    if config.get("use_hard_class_boosting", False):
        sw = []
        for lbl in train_ds.labels:
            cn = i2c[lbl]
            base_w = w[lbl].item()
            if cn in HARD_CLASSES:
                base_w *= HARD_CLASSES[cn]
            sw.append(base_w)
    else:
        sw = [w[l].item() for l in train_ds.labels]

    train_ld = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=WeightedRandomSampler(sw, len(sw), replacement=True),
        num_workers=2,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    use_anti_attractor = config.get("anti_attractor_weight", 0) > 0
    model = KSLGraphNet(
        len(classes),
        config["num_nodes"],
        config["in_channels"],
        config["hidden_dim"],
        config["num_layers"],
        config["temporal_kernel"],
        config["dropout"],
        adj,
        use_anti_attractor=use_anti_attractor,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")
    print(f"[{ts()}] Anti-attractor head: {use_anti_attractor}")
    print(f"[{ts()}] Focal loss: {config.get('use_focal_loss', False)}")
    print(f"[{ts()}] Mixup: alpha={config.get('mixup_alpha', 0)}, prob={config.get('mixup_prob', 0)}")
    print(f"[{ts()}] Contrastive loss weight: {config.get('contrastive_loss_weight', 0)}")

    # Loss functions
    if config.get("use_focal_loss", False) and confusion_pairs:
        criterion = ConfusionAwareFocalLoss(
            num_classes=len(classes),
            class_weights=w,
            confusion_pairs=confusion_pairs,
            class_to_idx=c2i,
            gamma=config["focal_gamma"],
            smoothing=config["label_smoothing"],
        ).to(device)
        print(f"[{ts()}] Using ConfusionAwareFocalLoss (gamma={config['focal_gamma']})")
    else:
        criterion = None  # will use F.cross_entropy
        print(f"[{ts()}] Using standard CrossEntropyLoss")

    # Contrastive loss
    contrastive_loss_fn = None
    contrastive_w = config.get("contrastive_loss_weight", 0)
    if contrastive_w > 0 and confusion_pairs:
        contrastive_loss_fn = SupervisedContrastiveLoss(
            confusion_pairs=confusion_pairs,
            class_to_idx=c2i,
        )
        print(f"[{ts()}] Using SupervisedContrastiveLoss (weight={contrastive_w})")

    # Anti-attractor setup
    anti_attractor_w = config.get("anti_attractor_weight", 0)
    attractor_class = config.get("anti_attractor_class", None)
    attractor_idx = c2i.get(attractor_class, -1) if attractor_class else -1
    anti_attractor_criterion = nn.BCEWithLogitsLoss() if anti_attractor_w > 0 and attractor_idx >= 0 else None
    if anti_attractor_criterion:
        print(f"[{ts()}] Anti-attractor targeting class '{attractor_class}' (idx={attractor_idx}, weight={anti_attractor_w})")

    # Optimizer
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

    best, patience_counter = 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    mixup_prob = config.get("mixup_prob", 0)
    use_aux = (contrastive_loss_fn is not None) or (anti_attractor_criterion is not None)

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        for d, t in train_ld:
            d, t = d.to(device), t.to(device)
            opt.zero_grad()

            # Decide if we use auxiliary outputs this batch
            if use_aux:
                logits, features, anti_score = model(d, return_features=True)
            else:
                logits = model(d)
                features = None
                anti_score = None

            # Mixup (applied probabilistically)
            if mixup_alpha > 0 and np.random.random() < mixup_prob:
                d_mix, y_a, y_b, lam = mixup_data(d, t, mixup_alpha)
                if use_aux:
                    logits_mix, features_mix, anti_score_mix = model(d_mix, return_features=True)
                else:
                    logits_mix = model(d_mix)

                if criterion is not None:
                    cls_loss = lam * criterion(logits_mix, y_a) + (1 - lam) * criterion(logits_mix, y_b)
                else:
                    cls_loss = lam * F.cross_entropy(logits_mix, y_a, weight=w, label_smoothing=config["label_smoothing"]) + \
                               (1 - lam) * F.cross_entropy(logits_mix, y_b, weight=w, label_smoothing=config["label_smoothing"])

                # For accuracy tracking, use non-mixed logits
                _, p = logits.max(1)
            else:
                # Standard forward
                if criterion is not None:
                    cls_loss = criterion(logits, t)
                else:
                    cls_loss = F.cross_entropy(logits, t, weight=w, label_smoothing=config["label_smoothing"])
                _, p = logits.max(1)

            loss = cls_loss

            # Contrastive loss
            if contrastive_loss_fn is not None and features is not None:
                c_loss = contrastive_loss_fn(features, t)
                loss = loss + contrastive_w * c_loss

            # Anti-attractor loss
            if anti_attractor_criterion is not None and anti_score is not None and attractor_idx >= 0:
                anti_targets = (t != attractor_idx).float().unsqueeze(1)
                anti_loss = anti_attractor_criterion(anti_score, anti_targets) * anti_attractor_w
                loss = loss + anti_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            tt += t.size(0)
            tc += p.eq(t).sum().item()

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

        print(
            f"[{ts()}] Ep {ep + 1:3d}/{config['epochs']} | "
            f"Loss: {tl / len(train_ld):.4f} | Train: {ta:.1f}% | Val: {va:.1f}% | "
            f"Time: {ep_time:.1f}s"
        )

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar(f"{name}/train_loss", tl / len(train_ld), ep)
            writer.add_scalar(f"{name}/train_acc", ta, ep)
            writer.add_scalar(f"{name}/val_acc", va, ep)
            writer.add_scalar(f"{name}/lr", scheduler.get_last_lr()[0], ep)

        if va > best:
            best, patience_counter = va, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": va,
                    "epoch": ep + 1,
                    "classes": classes,
                    "num_nodes": config["num_nodes"],
                    "version": "v16",
                    "config": config,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            print(f"[{ts()}]   -> New best! {va:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"[{ts()}] Early stopping at epoch {ep + 1}")
            break

    # Final evaluation with per-class breakdown
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for d, t in val_ld:
            _, p = model(d.to(device)).max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(t.numpy())

    print(f"\n[{ts()}] {name} Per-Class Results:")
    res = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        res[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {res[cn]:5.1f}% ({cor}/{tot_cls})")

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts)
    print(f"[{ts()}] {name} Overall: {ov:.1f}%")

    return {"overall": ov, "per_class": res, "best_epoch": ckpt["epoch"], "params": param_count}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL ST-GCN Training v16 (Alpine HPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="both",
        choices=["numbers", "words", "both"],
        help="Which model(s) to train",
    )

    # Paths
    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_v2"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_v2"))
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb-logdir", type=str, default=os.path.join(base_data, "runs"))

    args = parser.parse_args()

    # Header
    print("=" * 70)
    print(f"KSL Training v16 - ST-GCN + Confusion-Aware Training (Alpine HPC)")
    print(f"Started: {ts()}")
    print("=" * 70)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow")

    print(f"\nTrain dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Ckpt dir:  {args.checkpoint_dir}")

    # TensorBoard setup
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.tb_logdir, f"v16_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard: {tb_dir}")
        except ImportError:
            print("WARNING: tensorboard not installed, skipping TensorBoard logging")

    # Run training
    results = {}
    start_time = time.time()

    if args.model_type in ("numbers", "both"):
        print(f"\n[{ts()}] Numbers Config:")
        print(json.dumps({k: v for k, v in NUMBERS_CONFIG.items()}, indent=2))
        ckpt_dir = os.path.join(args.checkpoint_dir, "v16_numbers")
        results["numbers"] = train_split(
            "Numbers", NUMBER_CLASSES, NUMBERS_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
            confusion_pairs=NUMBERS_CONFUSION_PAIRS, writer=writer,
        )

    if args.model_type in ("words", "both"):
        print(f"\n[{ts()}] Words Config:")
        print(json.dumps({k: v for k, v in WORDS_CONFIG.items()}, indent=2))
        ckpt_dir = os.path.join(args.checkpoint_dir, "v16_words")
        results["words"] = train_split(
            "Words", WORD_CLASSES, WORDS_CONFIG,
            args.train_dir, args.val_dir, ckpt_dir, device,
            confusion_pairs=None, writer=writer,
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v16")
    print(f"[{ts()}] {'=' * 70}")
    if results.get("numbers"):
        print(f"[{ts()}] Numbers: {results['numbers']['overall']:.1f}% (best epoch {results['numbers']['best_epoch']}, params {results['numbers']['params']:,})")
    if results.get("words"):
        print(f"[{ts()}] Words:   {results['words']['overall']:.1f}% (best epoch {results['words']['best_epoch']}, params {results['words']['params']:,})")
    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"v16_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v16",
        "model_type": args.model_type,
        "numbers_config": NUMBERS_CONFIG if args.model_type in ("numbers", "both") else None,
        "words_config": WORDS_CONFIG if args.model_type in ("words", "both") else None,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    if writer is not None:
        writer.close()

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
