"""
KSL Training v14 - Extreme Dropout for Domain Shift

Key Changes from v13:
- Much more aggressive hand dropout (50-90%)
- Higher probability of applying dropout
- Simulates severe sparse detection like in test data

Usage:
    modal run train_ksl_v14.py
"""

import modal
import os

app = modal.App("ksl-trainer-v14")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])

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

CONFIG = {
    "max_frames": 90, "num_nodes": 48, "in_channels": 3, "hidden_dim": 128,
    "num_layers": 6, "temporal_kernel": 9, "batch_size": 32, "epochs": 200,
    "learning_rate": 1e-3, "min_lr": 1e-6, "weight_decay": 1e-4, "dropout": 0.5,
    "label_smoothing": 0.2, "patience": 40, "warmup_epochs": 10,
    "train_dir": "/data/train_v2", "val_dir": "/data/val_v2",
    "hand_dropout_prob": 0.7,
    "hand_dropout_min": 0.5,
    "hand_dropout_max": 0.9,
    "complete_hand_drop_prob": 0.2,
}


@app.function(gpu="T4", volumes={"/data": volume}, timeout=21600, image=image)
def train_extreme_dropout():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from collections import Counter
    import math
    import json

    print("=" * 70)
    print("KSL Training v14 - Extreme Dropout (50-90%)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def build_adj(n=48):
        adj = np.zeros((n, n))
        for i, j in HAND_EDGES:
            adj[i, j] = adj[j, i] = 1
        for i, j in HAND_EDGES:
            adj[i+21, j+21] = adj[j+21, i+21] = 1
        for i, j in POSE_EDGES:
            adj[i, j] = adj[j, i] = 1
        adj[0, 21] = adj[21, 0] = 0.3
        adj += np.eye(n)
        d = np.sum(adj, axis=1)
        d_inv = np.power(d, -0.5)
        d_inv[np.isinf(d_inv)] = 0
        return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))

    class Dataset_(Dataset):
        def __init__(s, data_dir, classes, mf=90, aug=False):
            s.samples, s.labels, s.mf, s.aug = [], [], mf, aug
            s.c2i = {c: i for i, c in enumerate(classes)}
            for cn in classes:
                cd = os.path.join(data_dir, cn)
                if os.path.exists(cd):
                    for fn in os.listdir(cd):
                        if fn.endswith(".npy"):
                            s.samples.append(os.path.join(cd, fn))
                            s.labels.append(s.c2i[cn])
            print(f"  Loaded {len(s.samples)} samples")

        def __len__(s):
            return len(s.samples)

        def __getitem__(s, idx):
            d = np.load(s.samples[idx])
            f = d.shape[0]

            if d.shape[1] >= 225:
                pose = np.zeros((f, 6, 3), dtype=np.float32)
                for pi, idx_pose in enumerate(POSE_INDICES):
                    start = idx_pose * 3
                    pose[:, pi, :] = d[:, start:start+3]
                lh = d[:, 99:162].reshape(f, 21, 3)
                rh = d[:, 162:225].reshape(f, 21, 3)
            else:
                pose = np.zeros((f, 6, 3), dtype=np.float32)
                lh = np.zeros((f, 21, 3), dtype=np.float32)
                rh = np.zeros((f, 21, 3), dtype=np.float32)

            h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

            if s.aug and np.random.random() < CONFIG["hand_dropout_prob"]:
                dropout_rate = np.random.uniform(CONFIG["hand_dropout_min"], CONFIG["hand_dropout_max"])
                if np.random.random() < 0.6:
                    lh_mask = np.random.random(f) > dropout_rate
                    h[~lh_mask, :21, :] = 0
                if np.random.random() < 0.6:
                    rh_mask = np.random.random(f) > dropout_rate
                    h[~rh_mask, 21:42, :] = 0

            if s.aug and np.random.random() < CONFIG["complete_hand_drop_prob"]:
                if np.random.random() < 0.5:
                    h[:, :21, :] = 0
                else:
                    h[:, 21:42, :] = 0

            if s.aug and np.random.random() < 0.3:
                h_copy = h.copy()
                h[:, :21, :] = h_copy[:, 21:42, :]
                h[:, 21:42, :] = h_copy[:, :21, :]
                h[:, :42, 0] = -h[:, :42, 0]

            lh_valid = np.abs(h[:, :21, :]).sum(axis=(1,2)) > 0.01
            rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1,2)) > 0.01

            if np.any(lh_valid):
                h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
            if np.any(rh_valid):
                h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

            mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
            h[:, 42:48, :] -= mid_shoulder

            max_val = np.abs(h).max()
            if max_val > 0.01:
                h = np.clip(h / max_val, -1, 1).astype(np.float32)

            if f >= s.mf:
                indices = np.linspace(0, f-1, s.mf, dtype=int)
                h = h[indices]
            else:
                h = np.concatenate([h, np.zeros((s.mf-f, 48, 3), dtype=np.float32)])

            if s.aug:
                if np.random.random() < 0.5:
                    h = h * np.random.uniform(0.8, 1.2)
                if np.random.random() < 0.4:
                    noise = np.random.normal(0, 0.03, h.shape).astype(np.float32)
                    h = h + noise
                if np.random.random() < 0.3:
                    shift = np.random.randint(-8, 9)
                    h = np.roll(h, shift, axis=0)

            return torch.FloatTensor(h).permute(2, 0, 1), s.labels[idx]

    class GConv(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.fc = nn.Linear(i, o)
        def forward(s, x, adj):
            return s.fc(torch.matmul(adj, x))

    class Block(nn.Module):
        def __init__(s, ic, oc, adj, ks=9, st=1, dr=0.3):
            super().__init__()
            s.register_buffer('adj', adj)
            s.gcn = GConv(ic, oc)
            s.bn1 = nn.BatchNorm2d(oc)
            s.tcn = nn.Sequential(nn.Conv2d(oc, oc, (ks, 1), padding=(ks//2, 0), stride=(st, 1)), nn.BatchNorm2d(oc))
            s.residual = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc)) if ic != oc or st != 1 else nn.Identity()
            s.dropout = nn.Dropout(dr)
        def forward(s, x):
            r = s.residual(x)
            b, c, t, n = x.shape
            x = s.gcn(x.permute(0, 2, 3, 1).reshape(b*t, n, c), s.adj)
            x = s.dropout(s.tcn(torch.relu(s.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2)))))
            return torch.relu(x + r)

    class Model(nn.Module):
        def __init__(s, nc, nn_=48, ic=3, hd=128, nl=6, tk=9, dr=0.3, adj=None):
            super().__init__()
            s.register_buffer('adj', adj)
            s.data_bn = nn.BatchNorm1d(nn_ * ic)
            ch = [ic] + [hd]*2 + [hd*2]*2 + [hd*4]*2
            ch = ch[:nl+1]
            s.layers = nn.ModuleList([Block(ch[i], ch[i+1], adj, tk, 2 if i in [2,4] else 1, dr) for i in range(nl)])
            s.pool = nn.AdaptiveAvgPool2d(1)
            s.classifier = nn.Sequential(nn.Linear(ch[-1], hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc))
        def forward(s, x):
            b, c, t, n = x.shape
            x = s.data_bn(x.permute(0, 1, 3, 2).reshape(b, c*n, t)).reshape(b, c, n, t).permute(0, 1, 3, 2)
            for l in s.layers:
                x = l(x)
            return s.classifier(s.pool(x).view(b, -1))

    def train_one(name, classes, ckpt_dir):
        print(f"\n{'='*70}\nTraining {name} ({len(classes)} classes) - Extreme Dropout v14\n{'='*70}")
        adj = build_adj(48).to(device)
        train_ds = Dataset_(CONFIG["train_dir"], classes, CONFIG["max_frames"], aug=True)
        val_ds = Dataset_(CONFIG["val_dir"], classes, CONFIG["max_frames"], aug=False)
        if len(train_ds) == 0:
            return None

        counts = Counter(train_ds.labels)
        tot = len(train_ds)
        w = torch.ones(len(classes))
        for i, c in counts.items():
            w[i] = min(2.0, tot / (len(counts) * c))
        w = w.to(device)

        sw = [w[l].item() for l in train_ds.labels]
        train_ld = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=WeightedRandomSampler(sw, len(sw), replacement=True))
        val_ld = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

        model = Model(len(classes), 48, 3, CONFIG["hidden_dim"], CONFIG["num_layers"], CONFIG["temporal_kernel"], CONFIG["dropout"], adj).to(device)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

        def sched(ep):
            if ep < CONFIG["warmup_epochs"]:
                return ep / CONFIG["warmup_epochs"]
            p = (ep - CONFIG["warmup_epochs"]) / (CONFIG["epochs"] - CONFIG["warmup_epochs"])
            return CONFIG["min_lr"] / CONFIG["learning_rate"] + (1 - CONFIG["min_lr"] / CONFIG["learning_rate"]) * 0.5 * (1 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, sched)
        os.makedirs(ckpt_dir, exist_ok=True)
        best, patience = 0.0, 0
        i2c = {v: k for k, v in train_ds.c2i.items()}

        for ep in range(CONFIG["epochs"]):
            model.train()
            tl, tc, tt = 0.0, 0, 0
            for d, t in train_ld:
                d, t = d.to(device), t.to(device)
                opt.zero_grad()
                o = model(d)
                loss = F.cross_entropy(o, t, weight=w, label_smoothing=CONFIG["label_smoothing"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()
                _, p = o.max(1)
                tt += t.size(0)
                tc += p.eq(t).sum().item()
            scheduler.step()

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
            print(f"Ep {ep+1:3d} | Loss: {tl/len(train_ld):.4f} | Train: {ta:.1f}% | Val: {va:.1f}%")

            if va > best:
                best, patience = va, 0
                torch.save({"model": model.state_dict(), "val_acc": va, "classes": classes, "num_nodes": 48, "version": "v14"}, os.path.join(ckpt_dir, "best_model.pt"))
                print(f"  -> Best! {va:.1f}%")
            else:
                patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stopping")
                break

        ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"))
        model.load_state_dict(ckpt["model"])
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for d, t in val_ld:
                _, p = model(d.to(device)).max(1)
                preds.extend(p.cpu().numpy())
                tgts.extend(t.numpy())

        print(f"\n{name} Per-Class Results:")
        res = {}
        for i in range(len(classes)):
            cn = i2c[i]
            tot = sum(1 for t in tgts if t == i)
            cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
            res[cn] = 100.0 * cor / tot if tot > 0 else 0.0
            print(f"  {cn:12s}: {res[cn]:5.1f}%")
        ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts)
        print(f"{name} Overall: {ov:.1f}%")
        return {"overall": ov, "classes": res}

    results = {}
    results["numbers"] = train_one("Numbers", NUMBER_CLASSES, "/data/checkpoints_v14_numbers")
    results["words"] = train_one("Words", WORD_CLASSES, "/data/checkpoints_v14_words")

    print("\n" + "=" * 70)
    print("SUMMARY v14")
    if results["numbers"]:
        print(f"Numbers: {results['numbers']['overall']:.1f}%")
    if results["words"]:
        print(f"Words: {results['words']['overall']:.1f}%")

    with open("/data/checkpoints_v14_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    r = train_extreme_dropout.remote()
    print(f"\nNumbers: {r['numbers']['overall']:.1f}%" if r.get("numbers") else "")
    print(f"Words: {r['words']['overall']:.1f}%" if r.get("words") else "")
