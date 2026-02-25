#!/usr/bin/env python3
"""
KSL Training v14 (Alpine) - Extreme Dropout for Domain Shift

Converted from Modal-based train_ksl_v14.py for Alpine HPC cluster.
All model architecture, dataset, augmentation, and training logic preserved exactly.

Key Features (from v14):
- Much more aggressive hand dropout (50-90%)
- Higher probability of applying dropout
- Simulates severe sparse detection like in test data

Usage:
    python train_ksl_alpine.py --version v14 --model-type numbers
    python train_ksl_alpine.py --version v14 --model-type words
    python train_ksl_alpine.py --version v14 --model-type both
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
# Default Config (matches v14 exactly)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "max_frames": 90,
    "num_nodes": 48,
    "in_channels": 3,
    "hidden_dim": 128,
    "num_layers": 6,
    "temporal_kernel": 9,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "label_smoothing": 0.2,
    "patience": 40,
    "warmup_epochs": 10,
    "hand_dropout_prob": 0.7,
    "hand_dropout_min": 0.5,
    "hand_dropout_max": 0.9,
    "complete_hand_drop_prob": 0.2,
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

        # Aggressive hand dropout (v14 key feature)
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

        # Additional augmentations
        if self.aug:
            if np.random.random() < 0.5:
                h = h * np.random.uniform(0.8, 1.2)
            if np.random.random() < 0.4:
                noise = np.random.normal(0, 0.03, h.shape).astype(np.float32)
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


class STGCNModel(nn.Module):
    def __init__(self, nc, nn_=48, ic=3, hd=128, nl=6, tk=9, dr=0.3, adj=None):
        super().__init__()
        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(nn_ * ic)
        ch = [ic] + [hd] * 2 + [hd * 2] * 2 + [hd * 4] * 2
        ch = ch[: nl + 1]
        self.layers = nn.ModuleList(
            [STGCNBlock(ch[i], ch[i + 1], adj, tk, 2 if i in [2, 4] else 1, dr) for i in range(nl)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ch[-1], hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc)
        )

    def forward(self, x):
        b, c, t, n = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * n, t)).reshape(b, c, n, t).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(self.pool(x).view(b, -1))

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_one(name, classes, config, train_dir, val_dir, ckpt_dir, device, writer=None):
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Training {name} ({len(classes)} classes) - Extreme Dropout v14")
    print(f"[{ts()}] {'=' * 70}")

    adj = build_adj(config["num_nodes"]).to(device)

    train_ds = KSLGraphDataset(train_dir, classes, config, aug=True)
    val_ds = KSLGraphDataset(val_dir, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples found in {train_dir}")
        return None

    # Class weights
    counts = Counter(train_ds.labels)
    tot = len(train_ds)
    w = torch.ones(len(classes))
    for i, c in counts.items():
        w[i] = min(2.0, tot / (len(counts) * c))
    w = w.to(device)

    # Weighted sampler
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

    model = STGCNModel(
        len(classes),
        config["num_nodes"],
        config["in_channels"],
        config["hidden_dim"],
        config["num_layers"],
        config["temporal_kernel"],
        config["dropout"],
        adj,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}")

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
    i2c = {v: k for k, v in train_ds.c2i.items()}

    for ep in range(config["epochs"]):
        ep_start = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        for d, t in train_ld:
            d, t = d.to(device), t.to(device)
            opt.zero_grad()
            o = model(d)
            loss = F.cross_entropy(o, t, weight=w, label_smoothing=config["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            _, p = o.max(1)
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
                    "version": "v14",
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
        description="KSL ST-GCN Training v14 (Alpine HPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--version", type=str, default="v14", help="Training version tag")
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

    # Hyperparameters (defaults match v14 CONFIG exactly)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG["min_lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--num-nodes", type=int, default=DEFAULT_CONFIG["num_nodes"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_CONFIG["label_smoothing"])
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_CONFIG["warmup_epochs"])
    parser.add_argument("--max-frames", type=int, default=DEFAULT_CONFIG["max_frames"])

    # Augmentation
    parser.add_argument("--hand-dropout-prob", type=float, default=DEFAULT_CONFIG["hand_dropout_prob"])
    parser.add_argument("--hand-dropout-min", type=float, default=DEFAULT_CONFIG["hand_dropout_min"])
    parser.add_argument("--hand-dropout-max", type=float, default=DEFAULT_CONFIG["hand_dropout_max"])
    parser.add_argument("--complete-hand-drop-prob", type=float, default=DEFAULT_CONFIG["complete_hand_drop_prob"])

    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb-logdir", type=str, default=os.path.join(base_data, "runs"))

    args = parser.parse_args()

    # Build runtime config from args
    config = {
        "max_frames": args.max_frames,
        "num_nodes": args.num_nodes,
        "in_channels": 3,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "temporal_kernel": 9,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "warmup_epochs": args.warmup_epochs,
        "hand_dropout_prob": args.hand_dropout_prob,
        "hand_dropout_min": args.hand_dropout_min,
        "hand_dropout_max": args.hand_dropout_max,
        "complete_hand_drop_prob": args.complete_hand_drop_prob,
    }

    # Header
    print("=" * 70)
    print(f"KSL Training {args.version} - Extreme Dropout (Alpine HPC)")
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

    print(f"\nConfig: {json.dumps(config, indent=2)}")
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Ckpt dir:  {args.checkpoint_dir}")

    # TensorBoard setup
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.tb_logdir, f"{args.version}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard: {tb_dir}")
        except ImportError:
            print("WARNING: tensorboard not installed, skipping TensorBoard logging")

    # Run training
    results = {}
    start_time = time.time()

    if args.model_type in ("numbers", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.version}_numbers")
        results["numbers"] = train_one(
            "Numbers", NUMBER_CLASSES, config, args.train_dir, args.val_dir, ckpt_dir, device, writer
        )

    if args.model_type in ("words", "both"):
        ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.version}_words")
        results["words"] = train_one(
            "Words", WORD_CLASSES, config, args.train_dir, args.val_dir, ckpt_dir, device, writer
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY {args.version}")
    print(f"[{ts()}] {'=' * 70}")
    if results.get("numbers"):
        print(f"[{ts()}] Numbers: {results['numbers']['overall']:.1f}% (best epoch {results['numbers']['best_epoch']})")
    if results.get("words"):
        print(f"[{ts()}] Words:   {results['words']['overall']:.1f}% (best epoch {results['words']['best_epoch']})")
    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"{args.version}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": args.version,
        "model_type": args.model_type,
        "config": config,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"[{ts()}] Results saved to {results_path}")

    if writer is not None:
        writer.close()

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
