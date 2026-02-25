"""
KSL v6 Training - Contrastive Learning for Signer-Independent Recognition

Key improvements:
1. Supervised Contrastive Loss - pulls same-class samples together regardless of signing style
2. Strong temporal augmentation - handles speed variations
3. Multi-head prototype learning - learns multiple prototypes per class
4. Domain-invariant features through feature normalization

Usage:
    modal run train_ksl_v6.py
"""

import modal
import os

app = modal.App("ksl-train-v6")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn", "tqdm")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

CONFIG = {
    "max_frames": 90,
    "feature_dim": 649,  # v5 features with hand engineering
    "hidden_dim": 256,
    "batch_size": 32,
    "epochs": 100,  # Reduced - model plateaus around epoch 100
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "temperature": 0.1,  # For contrastive loss
    "contrastive_weight": 0.5,  # Balance between CE and contrastive
    "early_stop_patience": 10,  # Stop if no improvement for 10 epochs
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=7200,
    image=image,
)
def train_contrastive():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import autocast, GradScaler
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from collections import defaultdict
    import random

    print("=" * 70)
    print("KSL v6 - Contrastive Learning for Signer-Independent Recognition")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Hand Feature Engineering (same as v5)
    # =========================================================================
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
    THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17

    def compute_hand_features(hand_landmarks):
        frames = hand_landmarks.shape[0]
        features = []
        for f in range(frames):
            hand = hand_landmarks[f].reshape(21, 3)
            frame_features = []
            wrist = hand[WRIST]
            hand_normalized = hand - wrist
            tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
            for tip in tips:
                frame_features.append(np.linalg.norm(hand_normalized[tip]))
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    frame_features.append(np.linalg.norm(hand[tips[i]] - hand[tips[j]]))
            for tip, mcp in zip(tips, mcps):
                frame_features.append(np.linalg.norm(hand[tip] - hand[mcp]))
            max_spread = max(np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                           for i in range(len(tips)) for j in range(i+1, len(tips)))
            frame_features.append(max_spread)
            thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
            index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
            cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
            frame_features.append(cos_angle)
            v1, v2 = hand[INDEX_MCP] - hand[WRIST], hand[PINKY_MCP] - hand[WRIST]
            palm_normal = np.cross(v1, v2)
            palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
            frame_features.extend(palm_normal.tolist())
            features.append(frame_features)
        return np.array(features, dtype=np.float32)

    def engineer_features(data):
        left_hand = data[:, 99:162]
        right_hand = data[:, 162:225]
        left_features = compute_hand_features(left_hand)
        right_features = compute_hand_features(right_hand)
        left_vel = np.zeros_like(left_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel = np.zeros_like(right_features)
        right_vel[1:] = right_features[1:] - right_features[:-1]
        return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)

    # =========================================================================
    # Strong Augmentation for Variation Robustness
    # =========================================================================

    def temporal_warp(data, sigma=0.2):
        """Randomly warp time dimension to simulate speed variations."""
        n_frames = len(data)
        if n_frames < 5:
            return data

        # Create warped time indices
        orig_steps = np.arange(n_frames)
        warp_steps = orig_steps + np.random.normal(0, sigma, n_frames).cumsum()
        warp_steps = np.clip(warp_steps, 0, n_frames - 1)
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min() + 1e-8) * (n_frames - 1)

        # Interpolate
        warped = np.zeros_like(data)
        for i in range(data.shape[1]):
            warped[:, i] = np.interp(orig_steps, warp_steps, data[:, i])
        return warped

    def temporal_crop(data, crop_ratio=0.1):
        """Randomly crop start/end frames."""
        n_frames = len(data)
        crop_frames = int(n_frames * crop_ratio)
        if crop_frames < 1:
            return data

        start_crop = random.randint(0, crop_frames)
        end_crop = random.randint(0, crop_frames)

        if start_crop + end_crop >= n_frames - 5:
            return data

        return data[start_crop:n_frames-end_crop] if end_crop > 0 else data[start_crop:]

    def add_noise(data, noise_level=0.02):
        """Add Gaussian noise."""
        noise = np.random.randn(*data.shape) * noise_level
        return data + noise

    def scale_features(data, scale_range=(0.9, 1.1)):
        """Random scaling of features."""
        scale = np.random.uniform(*scale_range)
        return data * scale

    def augment_sample(data, strong=False):
        """Apply augmentation pipeline."""
        # Always apply some augmentation
        if random.random() < 0.5:
            data = temporal_warp(data, sigma=0.3 if strong else 0.15)

        if random.random() < 0.5:
            data = temporal_crop(data, crop_ratio=0.15 if strong else 0.08)

        if random.random() < 0.5:
            data = add_noise(data, noise_level=0.03 if strong else 0.015)

        if random.random() < 0.3:
            data = scale_features(data)

        # Randomly reverse time (signing in reverse)
        if strong and random.random() < 0.2:
            data = data[::-1].copy()

        return data

    # =========================================================================
    # Dataset with Contrastive Sampling
    # =========================================================================

    class ContrastiveKSLDataset(Dataset):
        def __init__(self, data_dir, classes, max_frames, mean, std, augment=True):
            self.samples = []
            self.class_to_samples = defaultdict(list)
            self.max_frames = max_frames
            self.mean = mean
            self.std = std
            self.augment = augment

            for class_name in classes:
                class_dir = f"{data_dir}/{class_name}"
                if not os.path.exists(class_dir):
                    continue
                label = classes.index(class_name)
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(class_dir, filename)
                        idx = len(self.samples)
                        self.samples.append((filepath, label))
                        self.class_to_samples[label].append(idx)

            print(f"Loaded {len(self.samples)} samples from {data_dir}")

        def __len__(self):
            return len(self.samples)

        def prepare_sample(self, data, augment=True):
            """Prepare a single sample with optional augmentation."""
            if augment and self.augment:
                data = augment_sample(data, strong=True)

            data = (data - self.mean) / self.std

            # Resize to max_frames
            if data.shape[0] > self.max_frames:
                indices = np.linspace(0, data.shape[0] - 1, self.max_frames, dtype=int)
                data = data[indices]
            elif data.shape[0] < self.max_frames:
                padding = np.zeros((self.max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])

            # Add hand features
            data = engineer_features(data)

            return torch.from_numpy(data.astype(np.float32))

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)

            # Return two augmented views for contrastive learning
            view1 = self.prepare_sample(data, augment=True)
            view2 = self.prepare_sample(data, augment=True)

            return view1, view2, label

    # =========================================================================
    # Model with Projection Head for Contrastive Learning
    # =========================================================================

    class ContrastiveSignModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=256, proj_dim=128):
            super().__init__()

            # Hand-focused encoder
            self.hand_encoder = nn.Sequential(
                nn.Linear(226, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            )

            self.body_encoder = nn.Sequential(
                nn.Linear(423, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

            # Temporal modeling
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim, 3,
                batch_first=True, bidirectional=True, dropout=0.3
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

            # Feature normalization layer (helps with domain shift)
            self.feature_norm = nn.LayerNorm(hidden_dim * 2)

            # Projection head for contrastive learning
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim)
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )

        def encode(self, x):
            """Extract features before classification."""
            # Split features
            pose = x[:, :, :99]
            left_hand_base = x[:, :, 99:162]
            right_hand_base = x[:, :, 162:225]
            face = x[:, :, 225:549]
            left_hand_extra = x[:, :, 549:599]
            right_hand_extra = x[:, :, 599:649]

            left_hand = torch.cat([left_hand_base, left_hand_extra], dim=-1)
            right_hand = torch.cat([right_hand_base, right_hand_extra], dim=-1)
            hands = torch.cat([left_hand, right_hand], dim=-1)
            body = torch.cat([pose, face], dim=-1)

            hand_feat = self.hand_encoder(hands)
            body_feat = self.body_encoder(body)
            fused = self.fusion(torch.cat([hand_feat, body_feat], dim=-1))

            lstm_out, _ = self.lstm(fused)
            attn = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn).sum(dim=1)

            # Normalize features (helps with domain shift)
            context = self.feature_norm(context)

            return context

        def forward(self, x, return_features=False):
            features = self.encode(x)
            logits = self.classifier(features)

            if return_features:
                proj = F.normalize(self.projector(features), dim=1)
                return logits, proj
            return logits

    # =========================================================================
    # Supervised Contrastive Loss
    # =========================================================================

    class SupConLoss(nn.Module):
        """Supervised Contrastive Learning loss."""
        def __init__(self, temperature=0.1):
            super().__init__()
            self.temperature = temperature

        def forward(self, features, labels):
            """
            Args:
                features: [batch_size, proj_dim] - L2 normalized projections
                labels: [batch_size] - class labels
            """
            device = features.device
            batch_size = features.shape[0]

            # Compute similarity matrix
            similarity = torch.matmul(features, features.T) / self.temperature

            # Create mask for positive pairs (same class)
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            # Remove self-similarity
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
            mask = mask * logits_mask

            # Compute log-softmax
            exp_logits = torch.exp(similarity) * logits_mask
            log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

            # Compute mean of log-likelihood over positive pairs
            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

            # Loss
            loss = -mean_log_prob_pos.mean()

            return loss

    # =========================================================================
    # Load Data
    # =========================================================================

    print("\nLoading training data...")

    # Compute normalization stats
    all_data = []
    train_dir = "/data/train_v2"
    for class_name in ALL_CLASSES:
        class_dir = f"{train_dir}/{class_name}"
        if not os.path.exists(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        for f in files[:5]:
            data = np.load(os.path.join(class_dir, f))
            all_data.append(data.flatten())

    all_data = np.concatenate(all_data)
    mean = all_data.mean()
    std = all_data.std() + 1e-8
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")

    # Create datasets
    train_dataset = ContrastiveKSLDataset(
        train_dir, ALL_CLASSES, CONFIG["max_frames"], mean, std, augment=True
    )

    val_dataset = ContrastiveKSLDataset(
        "/data/val_v2", ALL_CLASSES, CONFIG["max_frames"], mean, std, augment=False
    )

    # Class weights for balanced sampling
    class_counts = defaultdict(int)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        sampler=sampler, num_workers=0, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )

    # =========================================================================
    # Training
    # =========================================================================

    model = ContrastiveSignModel(
        num_classes=len(ALL_CLASSES),
        feature_dim=CONFIG["feature_dim"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    contrastive_criterion = SupConLoss(temperature=CONFIG["temperature"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # Mixed precision for faster training
    scaler = GradScaler()

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for {CONFIG['epochs']} epochs (early stop patience: {CONFIG['early_stop_patience']})...")
    print(f"Contrastive weight: {CONFIG['contrastive_weight']}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        total_ce = 0
        total_con = 0
        correct = 0
        total = 0

        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                # Forward pass for both views
                logits1, proj1 = model(view1, return_features=True)
                logits2, proj2 = model(view2, return_features=True)

                # Classification loss (average of both views)
                ce_loss = (ce_criterion(logits1, labels) + ce_criterion(logits2, labels)) / 2

                # Contrastive loss (combine projections from both views)
                projections = torch.cat([proj1, proj2], dim=0)
                con_labels = torch.cat([labels, labels], dim=0)
                con_loss = contrastive_criterion(projections, con_labels)

                # Combined loss
                loss = ce_loss + CONFIG["contrastive_weight"] * con_loss

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_con += con_loss.item()

            # Accuracy from first view
            pred = logits1.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_con = total_con / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for view1, _, labels in val_loader:
                view1, labels = view1.to(device), labels.to(device)
                logits = model(view1)
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": CONFIG,
                "mean": mean,
                "std": std,
            }, "/data/checkpoints_v6/ksl_contrastive_best.pth")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or val_acc > best_val_acc - 0.001:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f} (CE={avg_ce:.4f}, Con={avg_con:.4f}) "
                  f"Train={train_acc*100:.1f}% Val={val_acc*100:.1f}% "
                  f"{'*BEST*' if val_acc >= best_val_acc else ''}")

        # Early stopping
        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {CONFIG['early_stop_patience']} epochs)")
            break

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Per-class Evaluation
    # =========================================================================

    print("\n" + "=" * 70)
    print("Per-class Evaluation")
    print("=" * 70)

    checkpoint = torch.load("/data/checkpoints_v6/ksl_contrastive_best.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for view1, _, labels in val_loader:
            view1, labels = view1.to(device), labels.to(device)
            logits = model(view1)
            pred = logits.argmax(dim=1)

            for p, l in zip(pred.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1

    print("\nNumbers:")
    for i, class_name in enumerate(NUMBER_CLASSES):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i] * 100
            print(f"  {class_name:8s}: {acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")

    print("\nWords:")
    for i, class_name in enumerate(WORD_CLASSES):
        idx = i + len(NUMBER_CLASSES)
        if per_class_total[idx] > 0:
            acc = per_class_correct[idx] / per_class_total[idx] * 100
            print(f"  {class_name:12s}: {acc:5.1f}% ({per_class_correct[idx]}/{per_class_total[idx]})")

    volume.commit()

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main():
    # Create checkpoint directory
    import subprocess
    subprocess.run(["modal", "volume", "put", "ksl-dataset-vol", "/dev/null", "checkpoints_v6/.keep"],
                   capture_output=True)

    result = train_contrastive.remote()
    print(f"\nFinal: Best validation accuracy = {result['best_val_acc']*100:.2f}%")
