"""
KSL Unified Training Script
Trains Transformer or LSTM models for Kenyan Sign Language recognition.
Supports all 30 classes (15 numbers + 15 words) in a single unified model.

Usage:
    modal run train_ksl.py --model transformer
    modal run train_ksl.py --model lstm
    modal run train_ksl.py --model both
"""

import modal
import os

# Modal app setup
app = modal.App("ksl-trainer")
volume = modal.Volume.from_name("ksl-dataset-vol")

# Container image with PyTorch and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    )
)

# All 30 classes
ALL_CLASSES = [
    # Numbers (sorted for consistency)
    "100", "125", "17", "22", "268", "35", "388", "444", "48", "54", "66", "73", "89", "9", "91",
    # Words (sorted for consistency)
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]

# Training configuration
CONFIG = {
    "max_frames": 120,          # Pad/truncate sequences to this length
    "feature_dim": 225,         # 33*3 + 21*3 + 21*3 (pose + hands)
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "patience": 30,             # Early stopping patience
    "checkpoint_dir": "/data/checkpoints",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=14400,  # 4 hours
    image=image,
)
def train_model(model_type: str) -> dict:
    """
    Train a KSL recognition model.

    Args:
        model_type: "transformer" or "lstm"

    Returns:
        Dictionary with training results
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import json
    from datetime import datetime

    print("=" * 70)
    print(f"KSL Training - {model_type.upper()} Model")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Classes: {len(ALL_CLASSES)}")
    print(f"Config: {CONFIG}")
    print("=" * 70)

    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Model Definitions
    # =========================================================================

    class TransformerModel(nn.Module):
        """Transformer-based sign language classifier."""

        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.3,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            # x: (batch, frames, features)
            batch_size, seq_len, _ = x.shape

            # Project features and add positional encoding
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :seq_len, :]

            # Transformer encoding
            x = self.transformer(x)

            # Global average pooling + classification
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModel(nn.Module):
        """Bidirectional LSTM with attention for sign language classification."""

        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, num_classes),
            )

        def forward(self, x):
            # x: (batch, frames, features)
            x = self.feature_proj(x)

            # LSTM encoding
            lstm_out, _ = self.lstm(x)  # (batch, frames, hidden*2)

            # Attention weights
            attn_weights = self.attention(lstm_out)  # (batch, frames, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)

            # Weighted sum
            context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

            return self.classifier(context)

    # =========================================================================
    # Dataset
    # =========================================================================

    class KSLDataset(Dataset):
        """Dataset for KSL landmark sequences."""

        def __init__(self, data_dir: str, classes: list, augment: bool = False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.augment = augment

            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"  Warning: {class_dir} not found")
                    continue

                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, self.class_to_idx[class_name]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)

            # Normalize sequence length
            if data.shape[0] > CONFIG["max_frames"]:
                # Uniformly sample frames if too long
                indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
                data = data[indices]
            elif data.shape[0] < CONFIG["max_frames"]:
                # Pad with zeros if too short
                padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])

            # Data augmentation
            if self.augment:
                # Random noise
                if np.random.rand() > 0.5:
                    data += np.random.normal(0, 0.01, data.shape).astype(np.float32)

                # Random time shift
                if np.random.rand() > 0.5:
                    shift = np.random.randint(-5, 6)
                    data = np.roll(data, shift, axis=0)

                # Random scaling
                if np.random.rand() > 0.5:
                    scale = np.random.uniform(0.95, 1.05)
                    data = data * scale

            return torch.from_numpy(data), label

    # =========================================================================
    # Training Setup
    # =========================================================================

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KSLDataset("/data/train", ALL_CLASSES, augment=True)
    val_dataset = KSLDataset("/data/val", ALL_CLASSES, augment=False)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("No training data found! Run extract_features.py first.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == "transformer":
        model = TransformerModel(num_classes=len(ALL_CLASSES))
    else:
        model = LSTMModel(num_classes=len(ALL_CLASSES))

    model = model.cuda()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # =========================================================================
    # Training Loop
    # =========================================================================

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_top3_acc": [], "lr": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
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

            # Gradient clipping
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
        per_class_correct = {i: 0 for i in range(len(ALL_CLASSES))}
        per_class_total = {i: 0 for i in range(len(ALL_CLASSES))}

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)

                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
                val_total += batch_y.size(0)

                # Top-3 accuracy
                _, top3_pred = outputs.topk(3, dim=1)
                for i in range(batch_y.size(0)):
                    if batch_y[i] in top3_pred[i]:
                        val_top3_correct += 1

                # Per-class accuracy
                for i in range(len(ALL_CLASSES)):
                    mask = batch_y == i
                    per_class_correct[i] += (predicted[mask] == i).sum().item()
                    per_class_total[i] += mask.sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_top3_acc = val_top3_correct / val_total if val_total > 0 else 0

        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(val_top3_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{CONFIG['epochs']} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Train: {train_acc * 100:.1f}% | "
            f"Val: {val_acc * 100:.1f}% | "
            f"Top-3: {val_top3_acc * 100:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": ALL_CLASSES,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  -> New best model saved! (Acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # =========================================================================
    # Final Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")

    # Per-class accuracy summary
    print("\nPer-class accuracy (final epoch):")
    for i, class_name in enumerate(ALL_CLASSES):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i] * 100
            print(f"  {class_name:12s}: {acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")

    # Save final model
    final_path = f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_final.pth"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "classes": ALL_CLASSES,
        "config": CONFIG,
    }, final_path)

    # Save training history
    history_path = f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)

    # Commit volume changes
    volume.commit()

    print(f"\nCheckpoints saved to: {CONFIG['checkpoint_dir']}/")
    print(f"  - ksl_{model_type}_best.pth (best validation)")
    print(f"  - ksl_{model_type}_final.pth (last epoch)")
    print(f"  - ksl_{model_type}_history.json (training history)")

    return {
        "model_type": model_type,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_epoch": epoch + 1,
    }


@app.local_entrypoint()
def main(model: str = "transformer"):
    """
    Main entry point for training.

    Args:
        model: "transformer", "lstm", or "both"
    """
    print("=" * 70)
    print("KSL Recognition Model Training")
    print("=" * 70)
    print(f"\nModel(s) to train: {model}")
    print(f"GPU: NVIDIA A10G")
    print(f"Classes: {len(ALL_CLASSES)} (15 numbers + 15 words)")

    if model == "both":
        # Train both models in parallel
        print("\nTraining both Transformer and LSTM models in parallel...")
        results = list(train_model.map(["transformer", "lstm"]))

        print("\n" + "=" * 70)
        print("ALL TRAINING COMPLETE")
        print("=" * 70)
        for result in results:
            print(f"\n{result['model_type'].upper()}:")
            print(f"  Best Accuracy: {result['best_val_acc'] * 100:.2f}%")
            print(f"  Best Epoch: {result['best_epoch']}")
    else:
        # Train single model
        result = train_model.remote(model)
        print(f"\n{result['model_type'].upper()} Training Complete!")
        print(f"Best Accuracy: {result['best_val_acc'] * 100:.2f}%")

    print("\nTo download models:")
    print("  modal volume get ksl-dataset-vol /data/checkpoints/ksl_transformer_best.pth .")
    print("  modal volume get ksl-dataset-vol /data/checkpoints/ksl_lstm_best.pth .")
