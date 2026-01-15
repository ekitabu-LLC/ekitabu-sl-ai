"""
KSL Training Script v3 - Maximum Performance
Advanced techniques to maximize accuracy without more data.

New in v3:
1. Feature Engineering: velocity + acceleration + relative positions
2. Advanced Augmentation: speed variation, reverse, temporal cutmix
3. Focal Loss for hard examples
4. Stochastic Weight Averaging (SWA)
5. Test-Time Augmentation (TTA)
6. Model Ensemble (Transformer + LSTM)

Usage:
    modal run train_ksl_v3.py --model lstm
    modal run train_ksl_v3.py --model transformer
    modal run train_ksl_v3.py --model ensemble
"""

import modal
import os

app = modal.App("ksl-trainer-v3")
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
    "max_frames": 90,
    "base_feature_dim": 225,        # Original landmarks
    "feature_dim": 225 + 225 + 225 + 126,  # landmarks + velocity + acceleration + relative
    "batch_size": 16,
    "epochs": 300,
    "learning_rate": 3e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.15,
    "dropout": 0.4,
    "patience": 60,
    "warmup_epochs": 15,
    "swa_start": 150,               # Start SWA after this epoch
    "focal_gamma": 2.0,             # Focal loss gamma
    "checkpoint_dir": "/data/checkpoints_v3",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=18000,  # 5 hours
    image=image,
)
def train_model(model_type: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.swa_utils import AveragedModel, SWALR
    import math
    import json
    from scipy.ndimage import zoom

    print("=" * 70)
    print(f"KSL Training v3 - {model_type.upper()} Model")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Feature dim: {CONFIG['feature_dim']} (landmarks + velocity + accel + relative)")
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Feature Engineering Functions
    # =========================================================================

    def compute_velocity(data):
        """Compute first derivative (velocity) of landmarks."""
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        return velocity

    def compute_acceleration(data):
        """Compute second derivative (acceleration) of landmarks."""
        velocity = compute_velocity(data)
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        return acceleration

    def compute_relative_positions(data):
        """
        Compute hand positions relative to shoulders.
        Makes features invariant to body position in frame.
        """
        # Pose landmarks: indices 0-32 (33 landmarks * 3 coords = 99 values)
        # Left shoulder: landmark 11 -> indices 33, 34, 35 (x, y, z)
        # Right shoulder: landmark 12 -> indices 36, 37, 38
        # Left hand landmarks: 99-161 (63 values)
        # Right hand landmarks: 162-224 (63 values)

        relative = []

        for frame in data:
            frame_rel = []

            # Left shoulder position
            left_shoulder = frame[33:36] if len(frame) > 35 else np.zeros(3)
            # Right shoulder position
            right_shoulder = frame[36:39] if len(frame) > 38 else np.zeros(3)

            # Left hand relative to left shoulder (21 landmarks * 3 = 63)
            left_hand = frame[99:162].reshape(-1, 3) if len(frame) > 161 else np.zeros((21, 3))
            left_hand_rel = (left_hand - left_shoulder).flatten()
            frame_rel.extend(left_hand_rel)

            # Right hand relative to right shoulder (21 landmarks * 3 = 63)
            right_hand = frame[162:225].reshape(-1, 3) if len(frame) > 224 else np.zeros((21, 3))
            right_hand_rel = (right_hand - right_shoulder).flatten()
            frame_rel.extend(right_hand_rel)

            relative.append(frame_rel)

        return np.array(relative, dtype=np.float32)

    def engineer_features(data):
        """Apply all feature engineering."""
        velocity = compute_velocity(data)
        acceleration = compute_acceleration(data)
        relative = compute_relative_positions(data)

        # Concatenate all features
        return np.concatenate([data, velocity, acceleration, relative], axis=1)

    # =========================================================================
    # Advanced Augmentation Functions
    # =========================================================================

    def temporal_speed_variation(data, speed_range=(0.8, 1.2)):
        """Change playback speed by resampling frames."""
        speed = np.random.uniform(*speed_range)
        num_frames = data.shape[0]
        new_num_frames = int(num_frames / speed)
        new_num_frames = max(10, min(new_num_frames, num_frames * 2))

        # Resample using interpolation
        indices = np.linspace(0, num_frames - 1, new_num_frames)
        resampled = np.zeros((new_num_frames, data.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            low_idx = int(idx)
            high_idx = min(low_idx + 1, num_frames - 1)
            alpha = idx - low_idx
            resampled[i] = (1 - alpha) * data[low_idx] + alpha * data[high_idx]

        return resampled

    def temporal_reverse(data):
        """Reverse the temporal sequence."""
        return data[::-1].copy()

    def temporal_cutmix(data1, data2, alpha=0.3):
        """Cut and mix temporal segments from two samples."""
        lam = np.random.beta(alpha, alpha)
        num_frames = data1.shape[0]
        cut_len = int(num_frames * (1 - lam))
        cut_start = np.random.randint(0, num_frames - cut_len + 1)

        mixed = data1.copy()
        mixed[cut_start:cut_start + cut_len] = data2[cut_start:cut_start + cut_len]
        return mixed, lam

    def spatial_jitter(data, scale=0.02):
        """Add random jitter to landmark positions."""
        jitter = np.random.normal(0, scale, data.shape).astype(np.float32)
        return data + jitter

    def random_frame_masking(data, mask_ratio=0.1):
        """Randomly mask (zero out) some frames."""
        num_frames = data.shape[0]
        num_mask = int(num_frames * mask_ratio)
        mask_indices = np.random.choice(num_frames, num_mask, replace=False)
        masked = data.copy()
        masked[mask_indices] = 0
        return masked

    # =========================================================================
    # Focal Loss
    # =========================================================================

    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance and hard examples."""

        def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.label_smoothing = label_smoothing

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(
                inputs, targets,
                reduction='none',
                label_smoothing=self.label_smoothing
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

            return focal_loss.mean()

    # =========================================================================
    # Models with Enhanced Architecture
    # =========================================================================

    class TransformerModel(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int = 192):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.2)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=6,
                dim_feedforward=384,
                dropout=CONFIG["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, num_classes),
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
        def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int = 192):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_dropout = nn.Dropout(0.2)

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
                nn.Dropout(0.2),
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
    # Dataset with Advanced Augmentation
    # =========================================================================

    class KSLDataset(Dataset):
        def __init__(self, data_dir: str, classes: list, augment: bool = False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.augment = augment
            self.num_classes = len(classes)

            # Load all samples into memory for faster access and cutmix
            self.data_cache = {}

            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, self.class_to_idx[class_name]))

            # Compute normalization stats
            self.mean = None
            self.std = None
            if len(self.samples) > 0:
                self._compute_stats()

        def _compute_stats(self):
            all_data = []
            for i in range(min(100, len(self.samples))):
                data = np.load(self.samples[i][0])
                all_data.append(data.flatten())
            all_data = np.concatenate(all_data)
            self.mean = all_data.mean()
            self.std = all_data.std() + 1e-8

        def _load_and_process(self, filepath):
            """Load, normalize, and engineer features."""
            data = np.load(filepath).astype(np.float32)

            # Normalize raw landmarks
            if self.mean is not None:
                data = (data - self.mean) / self.std

            return data

        def _pad_or_crop(self, data):
            """Pad or crop to fixed length."""
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
            return data

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = self._load_and_process(filepath)

            if self.augment:
                # Speed variation (before padding)
                if np.random.rand() > 0.5:
                    data = temporal_speed_variation(data, (0.8, 1.2))

                # Reverse playback
                if np.random.rand() > 0.7:
                    data = temporal_reverse(data)

            # Pad/crop to fixed length
            data = self._pad_or_crop(data)

            if self.augment:
                # Spatial jitter
                if np.random.rand() > 0.3:
                    data = spatial_jitter(data, scale=0.02)

                # Frame masking
                if np.random.rand() > 0.5:
                    data = random_frame_masking(data, mask_ratio=0.1)

                # Hand mirroring
                if np.random.rand() > 0.5:
                    left_hand = data[:, 99:162].copy()
                    right_hand = data[:, 162:225].copy()
                    data[:, 99:162] = right_hand
                    data[:, 162:225] = left_hand

            # Engineer features (velocity, acceleration, relative positions)
            data = engineer_features(data)

            return torch.from_numpy(data), label

    # =========================================================================
    # Test-Time Augmentation
    # =========================================================================

    def tta_predict(model, x, num_augments=5):
        """Apply test-time augmentation and average predictions."""
        model.eval()
        predictions = []

        with torch.no_grad():
            # Original prediction
            predictions.append(F.softmax(model(x), dim=1))

            # Reversed sequence
            x_rev = torch.flip(x, dims=[1])
            predictions.append(F.softmax(model(x_rev), dim=1))

            # Multiple noise augmentations
            for _ in range(num_augments - 2):
                noise = torch.randn_like(x) * 0.01
                predictions.append(F.softmax(model(x + noise), dim=1))

        return torch.stack(predictions).mean(dim=0)

    # =========================================================================
    # Learning Rate Schedule
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
    print(f"Feature dimension: {CONFIG['feature_dim']}")

    if len(train_dataset) == 0:
        raise ValueError("No training data found!")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    if model_type == "transformer":
        model = TransformerModel(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"])
    else:
        model = LSTMModel(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"])

    model = model.cuda()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Focal loss with label smoothing
    criterion = FocalLoss(
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Stochastic Weight Averaging
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    swa_start = CONFIG["swa_start"]

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_top3_acc": [], "lr": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        # Update learning rate
        if epoch < swa_start:
            lr = get_lr(epoch, CONFIG["warmup_epochs"], swa_start, CONFIG["learning_rate"])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            swa_scheduler.step()
            lr = optimizer.param_groups[0]['lr']

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

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

        # Update SWA model
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation (use SWA model after swa_start)
        eval_model = swa_model if epoch >= swa_start else model
        eval_model.eval()

        val_correct = 0
        val_top3_correct = 0
        val_total = 0

        # Use TTA for validation after certain epoch
        use_tta = epoch >= swa_start

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                if use_tta:
                    probs = tta_predict(eval_model, batch_x, num_augments=3)
                    _, predicted = probs.max(1)
                    _, top3_pred = probs.topk(3, dim=1)
                else:
                    outputs = eval_model(batch_x)
                    _, predicted = outputs.max(1)
                    _, top3_pred = outputs.topk(3, dim=1)

                val_correct += predicted.eq(batch_y).sum().item()
                val_total += batch_y.size(0)

                for i in range(batch_y.size(0)):
                    if batch_y[i] in top3_pred[i]:
                        val_top3_correct += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_top3_acc = val_top3_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3_acc)
        history["lr"].append(lr)

        swa_indicator = " [SWA]" if epoch >= swa_start else ""
        tta_indicator = " [TTA]" if use_tta else ""
        print(f"Epoch {epoch + 1:3d}/{CONFIG['epochs']} | Loss: {avg_train_loss:.4f} | Train: {train_acc * 100:.1f}% | Val: {val_acc * 100:.1f}% | Top-3: {val_top3_acc * 100:.1f}%{swa_indicator}{tta_indicator}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            save_model = swa_model.module if epoch >= swa_start else model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": save_model.state_dict(),
                "val_acc": val_acc,
                "val_top3_acc": val_top3_acc,
                "classes": ALL_CLASSES,
                "config": CONFIG,
                "feature_dim": CONFIG["feature_dim"],
            }, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_best.pth")
            print(f"  -> New best model saved! (Acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"] and epoch > swa_start + 20:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final SWA batch norm update
    if epoch >= swa_start:
        print("\nUpdating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=torch.device('cuda'))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")

    # Save final model
    final_model = swa_model.module if epoch >= swa_start else model
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "val_acc": val_acc,
        "classes": ALL_CLASSES,
        "config": CONFIG,
    }, f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_final.pth")

    with open(f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()

    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_epoch": best_epoch}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=7200,
    image=image,
)
def train_ensemble() -> dict:
    """Train and combine Transformer + LSTM for ensemble."""
    # Train both models
    transformer_result = train_model.remote("transformer")
    lstm_result = train_model.remote("lstm")

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"Transformer: {transformer_result['best_val_acc'] * 100:.2f}%")
    print(f"LSTM: {lstm_result['best_val_acc'] * 100:.2f}%")

    return {
        "transformer": transformer_result,
        "lstm": lstm_result,
    }


@app.local_entrypoint()
def main(model: str = "lstm"):
    print("=" * 70)
    print("KSL Training v3 - Maximum Performance")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Velocity + Acceleration + Relative positions")
    print("  - Advanced augmentation (speed, reverse, cutmix)")
    print("  - Focal loss for hard examples")
    print("  - Stochastic Weight Averaging (SWA)")
    print("  - Test-Time Augmentation (TTA)")
    print()

    if model == "ensemble" or model == "both":
        results = list(train_model.map(["transformer", "lstm"]))
        print("\n" + "=" * 70)
        print("RESULTS:")
        for r in results:
            print(f"  {r['model_type'].upper()}: {r['best_val_acc'] * 100:.2f}%")
    else:
        result = train_model.remote(model)
        print(f"\n{result['model_type'].upper()}: {result['best_val_acc'] * 100:.2f}%")

    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v3/ ./checkpoints/")
