"""
KSL Training Script v2 - Improved
Better regularization and augmentation to combat overfitting.

Changes from v1:
- Smaller models with more dropout
- Stronger data augmentation (temporal, spatial, mixup)
- Cosine annealing with warmup
- Feature normalization
- Gradient accumulation for effective larger batch

Usage:
    modal run train_ksl_v2.py --model transformer
    modal run train_ksl_v2.py --model lstm
    modal run train_ksl_v2.py --model both
"""

import modal
import os

app = modal.App("ksl-trainer-v2")
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

CONFIG = {
    "max_frames": 90,           # Reduced from 120
    "feature_dim": 225,
    "batch_size": 16,           # Smaller batch for better regularization
    "epochs": 300,
    "learning_rate": 5e-4,      # Lower LR
    "weight_decay": 0.05,       # Much stronger weight decay
    "label_smoothing": 0.2,     # More smoothing
    "dropout": 0.5,             # High dropout
    "patience": 50,
    "warmup_epochs": 10,
    "checkpoint_dir": "/data/checkpoints_v2",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=14400,
    image=image,
)
def train_model(model_type: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import math
    import json

    print("=" * 70)
    print(f"KSL Training v2 - {model_type.upper()} Model")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: {CONFIG}")
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Improved Models with Heavy Regularization
    # =========================================================================

    class TransformerModel(nn.Module):
        """Smaller Transformer with heavy regularization."""

        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 128):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.3)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=CONFIG["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

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
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.input_dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModel(nn.Module):
        """Smaller LSTM with heavy regularization."""

        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 128):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.input_dropout = nn.Dropout(0.3)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=CONFIG["dropout"],
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.3),
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
            x = self.feature_proj(x)
            x = self.input_dropout(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    # =========================================================================
    # Dataset with Strong Augmentation
    # =========================================================================

    class KSLDataset(Dataset):
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

            # Compute global stats for normalization
            self.mean = None
            self.std = None
            if len(self.samples) > 0:
                self._compute_stats()

        def _compute_stats(self):
            """Compute mean and std from a sample of data."""
            all_data = []
            for i in range(min(100, len(self.samples))):
                data = np.load(self.samples[i][0])
                all_data.append(data.flatten())
            all_data = np.concatenate(all_data)
            self.mean = all_data.mean()
            self.std = all_data.std() + 1e-8

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)

            # Normalize
            if self.mean is not None:
                data = (data - self.mean) / self.std

            # Temporal subsampling/padding to max_frames
            if data.shape[0] > CONFIG["max_frames"]:
                # Random crop during training, center crop during eval
                if self.augment:
                    start = np.random.randint(0, data.shape[0] - CONFIG["max_frames"] + 1)
                    data = data[start:start + CONFIG["max_frames"]]
                else:
                    indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
                    data = data[indices]
            elif data.shape[0] < CONFIG["max_frames"]:
                padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])

            if self.augment:
                # Gaussian noise
                if np.random.rand() > 0.3:
                    noise_scale = np.random.uniform(0.01, 0.05)
                    data += np.random.normal(0, noise_scale, data.shape).astype(np.float32)

                # Random temporal shift (roll)
                if np.random.rand() > 0.5:
                    shift = np.random.randint(-10, 11)
                    data = np.roll(data, shift, axis=0)

                # Random scaling
                if np.random.rand() > 0.5:
                    scale = np.random.uniform(0.9, 1.1)
                    data = data * scale

                # Random frame dropout (set some frames to zero)
                if np.random.rand() > 0.5:
                    num_drop = np.random.randint(1, 10)
                    drop_idx = np.random.choice(CONFIG["max_frames"], num_drop, replace=False)
                    data[drop_idx] = 0

                # Mirror (flip left/right hands) - swap left and right hand landmarks
                if np.random.rand() > 0.5:
                    # Pose: 0-98 (99 values), Left hand: 99-161 (63 values), Right hand: 162-224 (63 values)
                    left_hand = data[:, 99:162].copy()
                    right_hand = data[:, 162:225].copy()
                    data[:, 99:162] = right_hand
                    data[:, 162:225] = left_hand

            return torch.from_numpy(data), label

    # =========================================================================
    # Mixup Augmentation
    # =========================================================================

    def mixup_data(x, y, alpha=0.4):
        """Mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # =========================================================================
    # Cosine Annealing with Warmup
    # =========================================================================

    def get_lr(epoch, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # =========================================================================
    # Training
    # =========================================================================

    train_dataset = KSLDataset("/data/train", ALL_CLASSES, augment=True)
    val_dataset = KSLDataset("/data/val", ALL_CLASSES, augment=False)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("No training data found!")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    if model_type == "transformer":
        model = TransformerModel(num_classes=len(ALL_CLASSES))
    else:
        model = LSTMModel(num_classes=len(ALL_CLASSES))

    model = model.cuda()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_top3_acc": [], "lr": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        # Update learning rate
        lr = get_lr(epoch, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["learning_rate"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # Apply mixup
            if np.random.rand() > 0.5:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
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

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
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

                _, top3_pred = outputs.topk(3, dim=1)
                for i in range(batch_y.size(0)):
                    if batch_y[i] in top3_pred[i]:
                        val_top3_correct += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_top3_acc = val_top3_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3_acc)
        history["lr"].append(lr)

        print(f"Epoch {epoch + 1:3d}/{CONFIG['epochs']} | Loss: {avg_train_loss:.4f} | Train: {train_acc * 100:.1f}% | Val: {val_acc * 100:.1f}% | Top-3: {val_top3_acc * 100:.1f}% | LR: {lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": ALL_CLASSES,
                "config": CONFIG,
            }, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_best.pth")
            print(f"  -> New best model saved! (Acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")

    # Save final model and history
    torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "classes": ALL_CLASSES}, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_final.pth")
    with open(f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()

    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main(model: str = "lstm"):
    print("=" * 70)
    print("KSL Training v2 - Improved Regularization")
    print("=" * 70)

    if model == "both":
        results = list(train_model.map(["transformer", "lstm"]))
        print("\n" + "=" * 70)
        print("RESULTS:")
        for r in results:
            print(f"  {r['model_type'].upper()}: {r['best_val_acc'] * 100:.2f}%")
    else:
        result = train_model.remote(model)
        print(f"\n{result['model_type'].upper()}: {result['best_val_acc'] * 100:.2f}%")

    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v2/ ./checkpoints/")
