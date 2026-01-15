"""
KSL Training v10 - Spatial-Temporal Graph Convolutional Network (ST-GCN)

Architecture Change from v9:
- Instead of temporal pyramid CNN, use Graph Neural Network
- Hand landmarks (21 per hand) become graph nodes
- Edges follow anatomical hand structure
- Spatial convolution aggregates neighbor joint features
- Temporal convolution captures motion patterns

Why GNN for 444 vs 54:
- 444 = three "4" gestures (repeated hand shape)
- 54 = one "5" then one "4" (different hand shapes)
- GNN can learn the structural difference in hand configurations
- Temporal GCN captures the repetition pattern

Usage:
    modal run train_ksl_v10.py
"""

import modal
import os

app = modal.App("ksl-trainer-v10")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

# Classes
NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Hand graph edges (anatomical connections)
HAND_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

HARD_CLASSES = {"444": 2.5, "35": 2.0, "91": 1.8, "89": 1.8, "66": 1.5}

CONFUSION_PAIRS = [
    ("444", "54", 5.0), ("444", "35", 3.0),
    ("22", "54", 2.5), ("125", "54", 2.5), ("100", "54", 2.0),
    ("66", "17", 2.5), ("91", "73", 2.0), ("89", "73", 2.0), ("89", "Teach", 2.0),
]

CONFIG = {
    "max_frames": 90,
    "num_nodes": 42,
    "in_channels": 3,
    "hidden_dim": 128,
    "num_layers": 6,
    "temporal_kernel": 9,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "patience": 35,
    "warmup_epochs": 10,
    "focal_gamma": 2.0,
    "checkpoint_dir": "/data/checkpoints_v10",
    "train_dir": "/data/train_v2",
    "val_dir": "/data/val_v2",
}


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=21600, image=image)
def train_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from collections import Counter, defaultdict
    import math
    import json

    print("=" * 70)
    print("KSL Training v10 - ST-GCN Architecture")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def build_adjacency_matrix(num_nodes=42):
        adj = np.zeros((num_nodes, num_nodes))
        for i, j in HAND_EDGES:
            adj[i, j] = 1
            adj[j, i] = 1
        for i, j in HAND_EDGES:
            adj[i + 21, j + 21] = 1
            adj[j + 21, i + 21] = 1
        adj[0, 21] = 0.5
        adj[21, 0] = 0.5
        adj += np.eye(num_nodes)
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat = np.diag(d_inv_sqrt)
        adj_norm = d_mat @ adj @ d_mat
        return torch.FloatTensor(adj_norm)

    class KSLGraphDataset(Dataset):
        def __init__(self, data_dir, classes, max_frames=90, augment=False):
            self.samples = []
            self.labels = []
            self.max_frames = max_frames
            self.augment = augment
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for fn in os.listdir(class_dir):
                    if fn.endswith(".npy"):
                        self.samples.append(os.path.join(class_dir, fn))
                        self.labels.append(self.class_to_idx[class_name])
            print(f"Loaded {len(self.samples)} samples from {data_dir}")

        def __len__(self):
            return len(self.samples)

        def _extract_hands(self, data):
            frames = data.shape[0]
            feature_dim = data.shape[1]
            if feature_dim >= 225:
                left_hand = data[:, 99:162].reshape(frames, 21, 3)
                right_hand = data[:, 162:225].reshape(frames, 21, 3)
            else:
                left_hand = np.zeros((frames, 21, 3))
                right_hand = np.zeros((frames, 21, 3))
            hands = np.concatenate([left_hand, right_hand], axis=1)
            left_wrist = hands[:, 0:1, :].copy()
            right_wrist = hands[:, 21:22, :].copy()
            hands[:, :21, :] = hands[:, :21, :] - left_wrist
            hands[:, 21:, :] = hands[:, 21:, :] - right_wrist
            scale = np.abs(hands).max() + 1e-8
            hands = hands / scale
            return hands.astype(np.float32)

        def _pad_or_truncate(self, data):
            frames = data.shape[0]
            if frames >= self.max_frames:
                indices = np.linspace(0, frames - 1, self.max_frames, dtype=int)
                return data[indices]
            else:
                pad_size = self.max_frames - frames
                padding = np.zeros((pad_size, *data.shape[1:]), dtype=data.dtype)
                return np.concatenate([data, padding], axis=0)

        def _augment(self, data):
            if not self.augment:
                return data
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                data = data * scale
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.01, data.shape)
                data = data + noise
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 6)
                if shift > 0:
                    data = np.concatenate([np.zeros((shift, *data.shape[1:])), data[:-shift]], axis=0)
                elif shift < 0:
                    data = np.concatenate([data[-shift:], np.zeros((-shift, *data.shape[1:]))], axis=0)
            if np.random.random() < 0.3:
                left = data[:, :21, :].copy()
                right = data[:, 21:, :].copy()
                left[:, :, 0] = -left[:, :, 0]
                right[:, :, 0] = -right[:, :, 0]
                data = np.concatenate([right, left], axis=1)
            return data

        def __getitem__(self, idx):
            data = np.load(self.samples[idx])
            hands = self._extract_hands(data)
            hands = self._pad_or_truncate(hands)
            hands = self._augment(hands)
            hands = torch.FloatTensor(hands).permute(2, 0, 1)
            return hands, self.labels[idx]

    class GraphConvolution(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x, adj):
            x = torch.matmul(adj, x)
            x = self.fc(x)
            return x

    class STGCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, adj, kernel_size=9, stride=1, dropout=0.3):
            super().__init__()
            self.register_buffer('adj', adj)
            self.gcn = GraphConvolution(in_ch, out_ch)
            self.bn1 = nn.BatchNorm2d(out_ch)
            padding = (kernel_size - 1) // 2
            self.tcn = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size, 1), padding=(padding, 0), stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.residual = nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            res = self.residual(x)
            b, c, t, n = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b * t, n, c)
            x = self.gcn(x, self.adj)
            x = x.reshape(b, t, n, -1).permute(0, 3, 1, 2)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.tcn(x)
            x = self.dropout(x)
            x = x + res
            x = self.relu(x)
            return x

    class STGCN(nn.Module):
        def __init__(self, num_classes, num_nodes=42, in_ch=3, hidden_dim=128, num_layers=6, temporal_kernel=9, dropout=0.3, adj=None):
            super().__init__()
            self.register_buffer('adj', adj)
            self.data_bn = nn.BatchNorm1d(num_nodes * in_ch)
            channels = [in_ch] + [hidden_dim] * 2 + [hidden_dim * 2] * 2 + [hidden_dim * 4] * 2
            channels = channels[:num_layers + 1]
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                stride = 2 if i in [2, 4] else 1
                self.layers.append(STGCNBlock(channels[i], channels[i + 1], adj, kernel_size=temporal_kernel, stride=stride, dropout=dropout))
            self.pool = nn.AdaptiveAvgPool2d(1)
            final_ch = channels[-1]
            self.classifier = nn.Sequential(
                nn.Linear(final_ch, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            b, c, t, n = x.shape
            x = x.permute(0, 1, 3, 2).reshape(b, c * n, t)
            x = self.data_bn(x)
            x = x.reshape(b, c, n, t).permute(0, 1, 3, 2)
            for layer in self.layers:
                x = layer(x)
            x = self.pool(x).view(b, -1)
            x = self.classifier(x)
            return x

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.label_smoothing = label_smoothing

        def forward(self, inputs, targets):
            ce = F.cross_entropy(inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='none')
            pt = torch.exp(-ce)
            return (((1 - pt) ** self.gamma) * ce).mean()

    class ConfusionPenaltyLoss(nn.Module):
        def __init__(self, pairs, class_to_idx, weight=1.0):
            super().__init__()
            self.penalties = []
            for tc, wc, p in pairs:
                if tc in class_to_idx and wc in class_to_idx:
                    self.penalties.append((class_to_idx[tc], class_to_idx[wc], p))
            self.weight = weight

        def forward(self, logits, targets):
            if not self.penalties:
                return torch.tensor(0.0, device=logits.device)
            probs = F.softmax(logits, dim=1)
            loss = torch.tensor(0.0, device=logits.device)
            for ti, wi, p in self.penalties:
                mask = (targets == ti)
                if mask.sum() > 0:
                    loss = loss + p * probs[mask, wi].mean()
            return self.weight * loss

    adj = build_adjacency_matrix(CONFIG["num_nodes"]).to(device)
    print(f"Adjacency matrix: {adj.shape}")

    train_ds = KSLGraphDataset(CONFIG["train_dir"], ALL_CLASSES, CONFIG["max_frames"], augment=True)
    val_ds = KSLGraphDataset(CONFIG["val_dir"], ALL_CLASSES, CONFIG["max_frames"], augment=False)

    counts = Counter(train_ds.labels)
    total = len(train_ds)
    weights = torch.zeros(len(ALL_CLASSES))
    for idx, cnt in counts.items():
        weights[idx] = total / (len(counts) * cnt)
    for cn, boost in HARD_CLASSES.items():
        if cn in train_ds.class_to_idx:
            weights[train_ds.class_to_idx[cn]] *= boost
    weights = weights.to(device)

    sample_w = [weights[l].item() for l in train_ds.labels]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = STGCN(len(ALL_CLASSES), CONFIG["num_nodes"], CONFIG["in_channels"], CONFIG["hidden_dim"],
                  CONFIG["num_layers"], CONFIG["temporal_kernel"], CONFIG["dropout"], adj).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    focal = FocalLoss(CONFIG["focal_gamma"], weights, CONFIG["label_smoothing"])
    conf_loss = ConfusionPenaltyLoss(CONFUSION_PAIRS, train_ds.class_to_idx, 1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    def schedule(ep):
        if ep < CONFIG["warmup_epochs"]:
            return ep / CONFIG["warmup_epochs"]
        prog = (ep - CONFIG["warmup_epochs"]) / (CONFIG["epochs"] - CONFIG["warmup_epochs"])
        return CONFIG["min_lr"] / CONFIG["learning_rate"] + (1 - CONFIG["min_lr"] / CONFIG["learning_rate"]) * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    best_acc, best_444, patience_cnt = 0.0, 0.0, 0
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    idx_444 = train_ds.class_to_idx.get("444", -1)

    print("\n" + "=" * 70 + "\nTraining\n" + "=" * 70)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        tloss, tcorr, ttot = 0.0, 0, 0
        for data, tgt in train_loader:
            data, tgt = data.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = focal(out, tgt) + conf_loss(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item()
            _, pred = out.max(1)
            ttot += tgt.size(0)
            tcorr += pred.eq(tgt).sum().item()
        scheduler.step()
        tacc = 100.0 * tcorr / ttot

        model.eval()
        vcorr, vtot = 0, 0
        cc, ct = defaultdict(int), defaultdict(int)
        with torch.no_grad():
            for data, tgt in val_loader:
                data, tgt = data.to(device), tgt.to(device)
                out = model(data)
                _, pred = out.max(1)
                vtot += tgt.size(0)
                vcorr += pred.eq(tgt).sum().item()
                for t, p in zip(tgt.cpu().numpy(), pred.cpu().numpy()):
                    ct[t] += 1
                    if t == p:
                        cc[t] += 1

        vacc = 100.0 * vcorr / vtot
        v444 = 100.0 * cc[idx_444] / ct[idx_444] if ct[idx_444] > 0 else 0.0

        lr = optimizer.param_groups[0]["lr"]
        print(f"Ep {epoch+1:3d}/{CONFIG['epochs']} | Loss: {tloss/len(train_loader):.4f} | Train: {tacc:.1f}% | Val: {vacc:.1f}% | 444: {v444:.1f}% | LR: {lr:.6f}")

        improved = False
        if vacc > best_acc:
            best_acc = vacc
            improved = True
        if v444 > best_444:
            best_444 = v444
            improved = True

        if improved:
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": vacc, "val_444": v444}, os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
            print(f"  -> Best! Val: {vacc:.1f}%, 444: {v444:.1f}%")
        else:
            patience_cnt += 1

        if patience_cnt >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 25 == 0:
            print("\n" + "-" * 40 + "\nBottom 10 classes:")
            accs = [(idx_to_class[i], 100.0 * cc[i] / ct[i] if ct[i] > 0 else 0, ct[i]) for i in range(len(ALL_CLASSES))]
            accs.sort(key=lambda x: x[1])
            for nm, ac, tot in accs[:10]:
                print(f"  {nm:12s}: {ac:5.1f}% ({tot})")
            print("-" * 40 + "\n")

    print("\n" + "=" * 70 + "\nFinal Evaluation\n" + "=" * 70)
    ckpt = torch.load(os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for data, tgt in val_loader:
            out = model(data.to(device))
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())
            tgts.extend(tgt.numpy())

    print("\nPer-class accuracy:")
    res = {}
    for i in range(len(ALL_CLASSES)):
        cn = idx_to_class[i]
        tot = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        acc = 100.0 * cor / tot if tot > 0 else 0.0
        res[cn] = {"accuracy": acc, "total": tot, "correct": cor}
        mark = " <-- CRITICAL" if cn == "444" else ""
        print(f"  {cn:12s}: {acc:5.1f}% ({cor}/{tot}){mark}")

    overall = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts)
    num_acc = np.mean([res[c]["accuracy"] for c in NUMBER_CLASSES])
    word_acc = np.mean([res[c]["accuracy"] for c in WORD_CLASSES])

    print("\n" + "=" * 70)
    print(f"Overall: {overall:.2f}% | Numbers: {num_acc:.2f}% | Words: {word_acc:.2f}%")
    print(f"444: {res.get('444', {}).get('accuracy', 0):.2f}% | Best Val: {best_acc:.2f}% | Best 444: {best_444:.2f}%")
    print("=" * 70)

    with open(os.path.join(CONFIG["checkpoint_dir"], "results.json"), "w") as f:
        json.dump({"overall": overall, "numbers": num_acc, "words": word_acc, "class_444": res.get("444", {}).get("accuracy", 0), "classes": res}, f, indent=2)

    volume.commit()
    return {"overall": overall, "class_444": res.get("444", {}).get("accuracy", 0)}


@app.local_entrypoint()
def main():
    print("Starting KSL v10 (ST-GCN)")
    results = train_model.remote()
    print(f"\nDone! Overall: {results['overall']:.2f}%, 444: {results['class_444']:.2f}%")
