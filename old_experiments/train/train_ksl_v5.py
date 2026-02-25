"""
KSL Training v5 - Specialized for Numbers
Numbers are finger-based signs requiring fine-grained hand features.

Key improvements:
1. Hand-focused features: finger angles, distances, curls
2. Two-head classifier: separate heads for numbers vs words
3. Hand landmark normalization (relative to wrist)
4. Heavy augmentation on hand positions

Usage:
    modal run train_ksl_v5.py --model lstm
"""

import modal
import os

app = modal.App("ksl-trainer-v5")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

# Separate classes
NUMBER_CLASSES = ["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"]
WORD_CLASSES = ["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"]
ALL_CLASSES = sorted(NUMBER_CLASSES) + sorted(WORD_CLASSES)  # Keep consistent ordering

CONFIG = {
    "max_frames": 90,
    "base_feature_dim": 549,  # Original v2 features
    "batch_size": 32,  # Larger batch for speed
    "epochs": 200,     # Reduced
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "patience": 40,    # Reduced
    "warmup_epochs": 10,
    "checkpoint_dir": "/data/checkpoints_v5",
    "train_dir": "/data/train_v2",
    "val_dir": "/data/val_v2",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=21600,
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
    print(f"KSL Training v5 - {model_type.upper()} (Number-Focused)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Numbers: {len(NUMBER_CLASSES)}, Words: {len(WORD_CLASSES)}")
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # Hand Feature Engineering
    # =========================================================================

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    def compute_hand_features(hand_landmarks):
        """
        Compute specialized hand features for number recognition.
        Input: (frames, 63) - 21 landmarks * 3 coords
        Output: additional features
        """
        frames = hand_landmarks.shape[0]
        features = []

        for f in range(frames):
            hand = hand_landmarks[f].reshape(21, 3)
            frame_features = []

            # Normalize to wrist
            wrist = hand[WRIST]
            hand_normalized = hand - wrist

            # 1. Fingertip distances from wrist (5 values)
            for tip in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
                dist = np.linalg.norm(hand_normalized[tip])
                frame_features.append(dist)

            # 2. Fingertip-to-fingertip distances (10 values - all pairs)
            tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    dist = np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                    frame_features.append(dist)

            # 3. Finger curl (tip to MCP distance) (5 values)
            mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
            for tip, mcp in zip(tips, mcps):
                curl = np.linalg.norm(hand[tip] - hand[mcp])
                frame_features.append(curl)

            # 4. Hand spread (max distance between fingertips)
            max_spread = 0
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    spread = np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                    max_spread = max(max_spread, spread)
            frame_features.append(max_spread)

            # 5. Thumb-index angle (important for many numbers)
            thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
            index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
            cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
            frame_features.append(cos_angle)

            # 6. Palm normal (cross product of two palm vectors)
            v1 = hand[INDEX_MCP] - hand[WRIST]
            v2 = hand[PINKY_MCP] - hand[WRIST]
            palm_normal = np.cross(v1, v2)
            palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
            frame_features.extend(palm_normal.tolist())

            features.append(frame_features)

        return np.array(features, dtype=np.float32)  # (frames, 25)

    def engineer_features(data):
        """Add hand-specific features to the base features."""
        # data: (frames, 549)
        # Left hand: 99:162, Right hand: 162:225

        left_hand = data[:, 99:162]
        right_hand = data[:, 162:225]

        left_features = compute_hand_features(left_hand)   # (frames, 25)
        right_features = compute_hand_features(right_hand)  # (frames, 25)

        # Velocity of hand features
        left_vel = np.zeros_like(left_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel = np.zeros_like(right_features)
        right_vel[1:] = right_features[1:] - right_features[:-1]

        # Concatenate: base (549) + left hand (25) + right hand (25) + velocities (50) = 649
        enhanced = np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)
        return enhanced

    # =========================================================================
    # Dataset
    # =========================================================================

    class KSLDatasetV5(Dataset):
        def __init__(self, data_dir: str, classes: list, augment: bool = False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.augment = augment
            self.num_classes = len(classes)
            self.number_indices = set(self.class_to_idx[c] for c in NUMBER_CLASSES if c in self.class_to_idx)

            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))

            self.mean = None
            self.std = None
            if len(self.samples) > 0:
                self._compute_stats()

            print(f"  Loaded {len(self.samples)} samples")

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

        def _augment(self, data, is_number):
            # Speed variation
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

            # For numbers: more aggressive hand augmentation
            if is_number:
                # Add noise specifically to hand landmarks
                if np.random.rand() > 0.3:
                    hand_noise = np.random.normal(0, 0.02, (data.shape[0], 126)).astype(np.float32)
                    data[:, 99:225] += hand_noise

                # Random hand scaling (simulates different hand sizes)
                if np.random.rand() > 0.5:
                    scale = np.random.uniform(0.9, 1.1)
                    data[:, 99:225] *= scale
            else:
                # Standard noise for words
                if np.random.rand() > 0.3:
                    noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
                    data += noise

            # Hand swap (mirror) - useful for both
            if np.random.rand() > 0.5:
                left_hand = data[:, 99:162].copy()
                right_hand = data[:, 162:225].copy()
                data[:, 99:162] = right_hand
                data[:, 162:225] = left_hand

            # Frame dropout
            if np.random.rand() > 0.6:
                num_drop = np.random.randint(1, 8)
                drop_idx = np.random.choice(len(data), min(num_drop, len(data) - 5), replace=False)
                data[drop_idx] = 0

            return data

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)
            is_number = label in self.number_indices

            # Normalize base features
            if self.mean is not None:
                data = (data - self.mean) / self.std

            # Augment
            if self.augment:
                data = self._augment(data, is_number)

            # Pad/crop
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

            # Add hand features
            data = engineer_features(data)

            return torch.from_numpy(data), label

    # =========================================================================
    # Model with Separate Number/Word Attention
    # =========================================================================

    class HandFocusedModel(nn.Module):
        """Model with separate processing for hands (numbers) vs full body (words)."""

        def __init__(self, num_classes: int, feature_dim: int = 649, hidden_dim: int = 256):
            super().__init__()

            # Feature dimensions
            # Base: 549, Hand features: 50 (25 each), Hand velocities: 50
            self.pose_dim = 99
            self.left_hand_dim = 63 + 25 + 25  # landmarks + features + velocity
            self.right_hand_dim = 63 + 25 + 25
            self.face_dim = 324  # face + lips

            # Separate encoders
            self.hand_encoder = nn.Sequential(
                nn.Linear(self.left_hand_dim + self.right_hand_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )

            self.body_encoder = nn.Sequential(
                nn.Linear(self.pose_dim + self.face_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )

            # Fusion
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

            # LSTM for temporal modeling
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=CONFIG["dropout"],
            )

            # Attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            # Separate classifiers for numbers and words
            self.number_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, len(NUMBER_CLASSES)),
            )

            self.word_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, len(WORD_CLASSES)),
            )

            # Type classifier (number vs word)
            self.type_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 2, 2),  # 0=number, 1=word
            )

        def forward(self, x, return_type_logits=False):
            batch, frames, _ = x.shape

            # Split features
            pose = x[:, :, :99]
            left_hand_base = x[:, :, 99:162]
            right_hand_base = x[:, :, 162:225]
            face = x[:, :, 225:549]  # face + lips
            left_hand_extra = x[:, :, 549:599]  # hand features + velocity
            right_hand_extra = x[:, :, 599:649]

            # Combine hand features
            left_hand = torch.cat([left_hand_base, left_hand_extra], dim=-1)
            right_hand = torch.cat([right_hand_base, right_hand_extra], dim=-1)
            hands = torch.cat([left_hand, right_hand], dim=-1)

            # Combine body features
            body = torch.cat([pose, face], dim=-1)

            # Encode
            hand_feat = self.hand_encoder(hands)
            body_feat = self.body_encoder(body)

            # Fuse
            combined = torch.cat([hand_feat, body_feat], dim=-1)
            fused = self.fusion(combined)

            # Temporal modeling
            lstm_out, _ = self.lstm(fused)

            # Attention pooling
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)

            # Type prediction
            type_logits = self.type_classifier(context)

            # Get separate predictions
            number_logits = self.number_classifier(context)
            word_logits = self.word_classifier(context)

            # Combine into full class logits
            # Numbers are first 15 classes, words are next 15
            full_logits = torch.cat([number_logits, word_logits], dim=-1)

            if return_type_logits:
                return full_logits, type_logits
            return full_logits

    # =========================================================================
    # Simple LSTM (baseline with hand features)
    # =========================================================================

    class SimpleLSTM(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 649, hidden_dim: int = 256):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=CONFIG["dropout"],
            )

            self.attention = nn.Sequential(
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
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    # =========================================================================
    # Training
    # =========================================================================

    # Reorder classes to match NUMBER_CLASSES + WORD_CLASSES order
    ordered_classes = sorted(NUMBER_CLASSES) + sorted(WORD_CLASSES)

    train_dataset = KSLDatasetV5(CONFIG["train_dir"], ordered_classes, augment=True)
    val_dataset = KSLDatasetV5(CONFIG["val_dir"], ordered_classes, augment=False)

    if len(train_dataset) == 0:
        return {"model_type": model_type, "best_val_acc": 0.0, "error": "No data"}

    # Class-balanced sampling
    class_counts = Counter([s[1] for s in train_dataset.samples])
    weights = [1.0 / class_counts[s[1]] for s in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    feature_dim = 649  # 549 base + 100 hand features
    if model_type == "handfocused":
        model = HandFocusedModel(num_classes=len(ordered_classes), feature_dim=feature_dim)
    else:
        model = SimpleLSTM(num_classes=len(ordered_classes), feature_dim=feature_dim)

    model = model.cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Feature dim: {feature_dim}")

    # Loss with class weights
    class_weights = torch.tensor([1.0 / max(class_counts.get(i, 1), 1) for i in range(len(ordered_classes))], device="cuda")
    class_weights = class_weights / class_weights.sum() * len(ordered_classes)

    # Extra weight for numbers (they're harder)
    for i in range(len(NUMBER_CLASSES)):
        class_weights[i] *= 1.5

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    def get_lr(epoch):
        if epoch < CONFIG["warmup_epochs"]:
            return CONFIG["learning_rate"] * (epoch + 1) / CONFIG["warmup_epochs"]
        progress = (epoch - CONFIG["warmup_epochs"]) / (CONFIG["epochs"] - CONFIG["warmup_epochs"])
        return 1e-6 + 0.5 * (CONFIG["learning_rate"] - 1e-6) * (1 + math.cos(math.pi * progress))

    best_val_acc = 0.0
    best_number_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "number_acc": [], "word_acc": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        number_correct = 0
        number_total = 0
        word_correct = 0
        word_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

                # Separate accuracy for numbers and words
                for i in range(batch_y.size(0)):
                    if batch_y[i] < len(NUMBER_CLASSES):
                        number_total += 1
                        if predicted[i] == batch_y[i]:
                            number_correct += 1
                    else:
                        word_total += 1
                        if predicted[i] == batch_y[i]:
                            word_correct += 1

        val_acc = correct / total if total > 0 else 0
        number_acc = number_correct / number_total if number_total > 0 else 0
        word_acc = word_correct / word_total if word_total > 0 else 0

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["number_acc"].append(number_acc)
        history["word_acc"].append(word_acc)

        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Val: {val_acc*100:.1f}% | Numbers: {number_acc*100:.1f}% | Words: {word_acc*100:.1f}%")

        # Save best model (prioritize overall accuracy but track number accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_number_acc = number_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "number_acc": number_acc,
                "word_acc": word_acc,
                "classes": ordered_classes,
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
    print(f"Number accuracy: {best_number_acc*100:.2f}%")

    with open(f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()
    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_number_acc": best_number_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main(model: str = "lstm"):
    print("=" * 70)
    print("KSL Training v5 - Number-Focused")
    print("=" * 70)
    print("\nImprovements:")
    print("  - Hand-specific features (finger angles, distances, curls)")
    print("  - Extra weight on number classes")
    print("  - Hand-focused augmentation")
    print()

    if model == "both":
        results = list(train_model.map(["lstm", "handfocused"]))
        print("\n" + "=" * 70)
        print("RESULTS:")
        for r in results:
            print(f"  {r['model_type'].upper()}: {r['best_val_acc']*100:.2f}% (Numbers: {r['best_number_acc']*100:.2f}%)")
    else:
        result = train_model.remote(model)
        print(f"\n{result['model_type'].upper()}: {result['best_val_acc']*100:.2f}%")
        print(f"Number accuracy: {result['best_number_acc']*100:.2f}%")

    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v5/ ./checkpoints/")
