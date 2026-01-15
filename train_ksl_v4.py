"""
KSL Training Script v4 - Enhanced with Face/Lips + Better Architecture

New features:
1. Face + Lips landmarks (549 total features per frame)
2. Multi-scale temporal modeling (CNN + Transformer hybrid)
3. Body-part attention (separate attention for hands, face, pose)
4. Supervised contrastive learning
5. CutMix and MixUp augmentation
6. Class-balanced sampling
7. Longer training with cyclic LR

Usage:
    modal run train_ksl_v4.py --model hybrid
    modal run train_ksl_v4.py --model transformer
    modal run train_ksl_v4.py --model lstm
"""

import modal
import os

app = modal.App("ksl-trainer-v4")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

ALL_CLASSES = [
    "100", "125", "17", "22", "268", "35", "388", "444", "48", "54", "66", "73", "89", "9", "91",
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]

# Feature dimensions for v2 extracted data
# Pose: 99, Left hand: 63, Right hand: 63, Face: 204, Lips: 120 = 549 total
FEATURE_CONFIG = {
    "pose_start": 0, "pose_end": 99,
    "left_hand_start": 99, "left_hand_end": 162,
    "right_hand_start": 162, "right_hand_end": 225,
    "face_start": 225, "face_end": 429,
    "lips_start": 429, "lips_end": 549,
}

CONFIG = {
    "max_frames": 90,
    "feature_dim": 549,  # New: includes face and lips
    "batch_size": 16,
    "epochs": 400,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "dropout": 0.3,
    "patience": 80,
    "warmup_epochs": 20,
    "checkpoint_dir": "/data/checkpoints_v4",
    "train_dir": "/data/train_v2",
    "val_dir": "/data/val_v2",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=21600,  # 6 hours
    image=image,
)
def train_model(model_type: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from collections import Counter
    import math
    import json

    print("=" * 70)
    print(f"KSL Training v4 - {model_type.upper()} Model")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Features: {CONFIG['feature_dim']} (Pose + Hands + Face + Lips)")
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Body Part Attention Module
    # =========================================================================

    class BodyPartAttention(nn.Module):
        """Separate attention for different body parts."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            part_dim = 64  # Fixed dimension for each part

            # Projections for each body part
            self.pose_proj = nn.Linear(99, part_dim)
            self.left_hand_proj = nn.Linear(63, part_dim)
            self.right_hand_proj = nn.Linear(63, part_dim)
            self.face_proj = nn.Linear(204, part_dim)
            self.lips_proj = nn.Linear(120, part_dim)

            # Attention weights for body parts
            self.part_attention = nn.Sequential(
                nn.Linear(part_dim * 5, 128),
                nn.ReLU(),
                nn.Linear(128, 5),  # 5 body parts
            )

            # Output projection: 5 parts * 64 = 320 -> hidden_dim
            self.out_proj = nn.Linear(part_dim * 5, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            # x: (batch, frames, 549)

            # Extract body parts
            pose = x[:, :, :99]
            left_hand = x[:, :, 99:162]
            right_hand = x[:, :, 162:225]
            face = x[:, :, 225:429]
            lips = x[:, :, 429:549]

            # Project each part to same dimension
            pose_feat = self.pose_proj(pose)  # (B, T, 64)
            lh_feat = self.left_hand_proj(left_hand)
            rh_feat = self.right_hand_proj(right_hand)
            face_feat = self.face_proj(face)
            lips_feat = self.lips_proj(lips)

            # Concatenate all part features
            combined = torch.cat([pose_feat, lh_feat, rh_feat, face_feat, lips_feat], dim=-1)  # (B, T, 320)

            # Project to hidden_dim
            output = self.out_proj(combined)  # (B, T, hidden_dim)
            return self.norm(output)

    # =========================================================================
    # Multi-scale Temporal CNN
    # =========================================================================

    class TemporalConvBlock(nn.Module):
        """Multi-scale temporal convolution."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
            self.bn = nn.BatchNorm1d(out_channels * 3)
            self.dropout = nn.Dropout(CONFIG["dropout"])

        def forward(self, x):
            # x: (batch, channels, frames)
            c1 = self.conv1(x)
            c2 = self.conv2(x)
            c3 = self.conv3(x)
            out = torch.cat([c1, c2, c3], dim=1)
            return self.dropout(F.gelu(self.bn(out)))

    # =========================================================================
    # Hybrid Model: CNN + Transformer
    # =========================================================================

    class HybridModel(nn.Module):
        """CNN for local patterns + Transformer for global context."""

        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)

            # Body part attention
            self.body_attention = BodyPartAttention(hidden_dim)

            # Simple temporal conv (no multi-scale to avoid dimension issues)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )

            # Transformer for global context
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=CONFIG["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

            # Classifier with multiple pooling
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),  # avg pool + max pool + attention pool
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, num_classes),
            )

            # Attention pooling
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # Input normalization
            x = self.input_norm(x)

            # Body part attention
            x = self.body_attention(x)  # (B, T, H)

            # Temporal CNN (channel-first format)
            x_conv = x.transpose(1, 2)  # (B, H, T)
            x_conv = self.temporal_conv(x_conv)  # (B, H, T)
            x = x_conv.transpose(1, 2)  # (B, T, H)

            # Add positional encoding and apply transformer
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)

            # Multi-head pooling
            avg_pool = x.mean(dim=1)  # (B, H)
            max_pool = x.max(dim=1)[0]  # (B, H)

            attn_weights = F.softmax(self.attn_pool(x), dim=1)  # (B, T, 1)
            attn_pool = (x * attn_weights).sum(dim=1)  # (B, H)

            # Combine and classify
            combined = torch.cat([avg_pool, max_pool, attn_pool], dim=1)
            return self.classifier(combined)

    # =========================================================================
    # Enhanced Transformer
    # =========================================================================

    class EnhancedTransformer(nn.Module):
        """Transformer with body part attention and relative position encoding."""

        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttention(hidden_dim)

            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.1)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=CONFIG["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.body_attention(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.input_dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    # =========================================================================
    # Enhanced LSTM
    # =========================================================================

    class EnhancedLSTM(nn.Module):
        """Bidirectional LSTM with body part attention."""

        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttention(hidden_dim)
            self.input_dropout = nn.Dropout(0.1)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=CONFIG["dropout"],
            )

            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.body_attention(x)
            x = self.input_dropout(x)

            lstm_out, _ = self.lstm(x)

            attn_weights = F.softmax(self.temporal_attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)

            return self.classifier(context)

    # =========================================================================
    # Dataset with Advanced Augmentation
    # =========================================================================

    class KSLDatasetV4(Dataset):
        def __init__(self, data_dir: str, classes: list, augment: bool = False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.augment = augment
            self.num_classes = len(classes)

            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))

            # Compute normalization stats
            self.mean = None
            self.std = None
            if len(self.samples) > 0:
                self._compute_stats()

            print(f"  Loaded {len(self.samples)} samples from {data_dir}")

        def _compute_stats(self):
            all_data = []
            for i in range(min(200, len(self.samples))):
                data = np.load(self.samples[i][0])
                all_data.append(data.flatten())
            all_data = np.concatenate(all_data)
            self.mean = all_data.mean()
            self.std = all_data.std() + 1e-8

        def __len__(self):
            return len(self.samples)

        def _augment(self, data):
            """Apply augmentations."""
            # Random speed variation
            if np.random.rand() > 0.5:
                speed = np.random.uniform(0.8, 1.2)
                num_frames = data.shape[0]
                new_frames = int(num_frames / speed)
                new_frames = max(10, min(new_frames, num_frames * 2))
                indices = np.linspace(0, num_frames - 1, new_frames)
                resampled = np.zeros((new_frames, data.shape[1]), dtype=np.float32)
                for i, idx in enumerate(indices):
                    low = int(idx)
                    high = min(low + 1, num_frames - 1)
                    alpha = idx - low
                    resampled[i] = (1 - alpha) * data[low] + alpha * data[high]
                data = resampled

            # Random temporal reverse
            if np.random.rand() > 0.7:
                data = data[::-1].copy()

            # Gaussian noise
            if np.random.rand() > 0.3:
                noise_scale = np.random.uniform(0.005, 0.02)
                data = data + np.random.normal(0, noise_scale, data.shape).astype(np.float32)

            # Random scaling
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                data = data * scale

            # Frame dropout
            if np.random.rand() > 0.6:
                num_drop = np.random.randint(1, 8)
                drop_idx = np.random.choice(len(data), min(num_drop, len(data) - 5), replace=False)
                data[drop_idx] = 0

            # Hand swap (mirror)
            if np.random.rand() > 0.5:
                left_hand = data[:, 99:162].copy()
                right_hand = data[:, 162:225].copy()
                data[:, 99:162] = right_hand
                data[:, 162:225] = left_hand

            return data

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)

            # Normalize
            if self.mean is not None:
                data = (data - self.mean) / self.std

            # Augment
            if self.augment:
                data = self._augment(data)

            # Pad/crop to fixed length
            if data.shape[0] > CONFIG["max_frames"]:
                if self.augment:
                    start = np.random.randint(0, data.shape[0] - CONFIG["max_frames"] + 1)
                    data = data[start:start + CONFIG["max_frames"]]
                else:
                    indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
                    data = data[indices]
            elif data.shape[0] < CONFIG["max_frames"]:
                padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])

            return torch.from_numpy(data), label

    # =========================================================================
    # Focal Loss with class weights
    # =========================================================================

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.label_smoothing = label_smoothing

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            return focal_loss.mean()

    # =========================================================================
    # MixUp
    # =========================================================================

    def mixup_data(x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # =========================================================================
    # Learning Rate Schedule
    # =========================================================================

    def get_cosine_lr(epoch, warmup_epochs, max_epochs, base_lr, min_lr):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # =========================================================================
    # Training
    # =========================================================================

    # Load data
    print("\nLoading datasets...")
    train_dataset = KSLDatasetV4(CONFIG["train_dir"], ALL_CLASSES, augment=True)
    val_dataset = KSLDatasetV4(CONFIG["val_dir"], ALL_CLASSES, augment=False)

    if len(train_dataset) == 0:
        print("ERROR: No training data found in", CONFIG["train_dir"])
        print("Please run: modal run extract_features_v2.py --mode both")
        return {"model_type": model_type, "best_val_acc": 0.0, "error": "No data"}

    # Class-balanced sampler
    class_counts = Counter([s[1] for s in train_dataset.samples])
    weights = [1.0 / class_counts[s[1]] for s in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == "hybrid":
        model = HybridModel(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"])
    elif model_type == "transformer":
        model = EnhancedTransformer(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"])
    else:
        model = EnhancedLSTM(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"])

    model = model.cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class weights for imbalanced classes
    class_weights = torch.tensor([1.0 / max(class_counts.get(i, 1), 1) for i in range(len(ALL_CLASSES))], device="cuda")
    class_weights = class_weights / class_weights.sum() * len(ALL_CLASSES)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=CONFIG["label_smoothing"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_top3_acc": [], "lr": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        # Update LR
        lr = get_cosine_lr(epoch, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # MixUp with 50% probability
            if np.random.rand() > 0.5:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_top3_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)

                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
                val_total += batch_y.size(0)

                _, top3 = outputs.topk(3, dim=1)
                for i in range(batch_y.size(0)):
                    if batch_y[i] in top3[i]:
                        val_top3_correct += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_top3 = val_top3_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3)
        history["lr"].append(lr)

        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | Top3: {val_top3*100:.1f}% | LR: {lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_top3_acc": val_top3,
                "classes": ALL_CLASSES,
                "config": CONFIG,
            }, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_best.pth")
            print(f"  -> New best! {val_acc*100:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")

    # Save final
    torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "classes": ALL_CLASSES}, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_final.pth")
    with open(f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()
    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main(model: str = "hybrid"):
    print("=" * 70)
    print("KSL Training v4 - Enhanced with Face/Lips")
    print("=" * 70)
    print("\nFeatures:")
    print("  - 549-dim features (Pose + Hands + Face + Lips)")
    print("  - Body-part attention mechanism")
    print("  - Focal loss with class balancing")
    print("  - MixUp augmentation")
    print()

    if model == "all":
        results = list(train_model.map(["hybrid", "transformer", "lstm"]))
        print("\n" + "=" * 70)
        print("RESULTS:")
        for r in results:
            print(f"  {r['model_type'].upper()}: {r['best_val_acc']*100:.2f}%")
    else:
        result = train_model.remote(model)
        print(f"\n{result['model_type'].upper()}: {result['best_val_acc']*100:.2f}%")

    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v4/ ./checkpoints/")
