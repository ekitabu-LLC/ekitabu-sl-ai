"""
KSL Training v7 - Fixing Critical Performance Issues

Problems identified from v5/v6:
1. Complete failures: 100, 35, 444 (0%), Tomatoes (4%), Proud (16%)
2. Severe overfitting: Train 91.8% vs Val 41.3%
3. Numbers underperform (~39% avg) vs Words (~53% avg)
4. Contrastive loss hurting classification

Solutions in v7:
1. Remove contrastive loss - use focal loss only
2. Hard class boosting - extra weight for failing classes
3. Temporal pyramid - multi-scale for multi-digit numbers (100, 444)
4. Reduce augmentation strength - was destroying features
5. Add mixup (gentler than contrastive)
6. Better hand feature normalization

Usage:
    modal run train_ksl_v7.py
    modal run train_ksl_v7.py --model pyramid
"""

import modal
import os

app = modal.App("ksl-trainer-v7")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

HARD_CLASSES = {
    "100": 3.0, "35": 3.0, "444": 3.0, "125": 2.5, "54": 2.0, "268": 2.0,
    "Tomatoes": 3.0, "Proud": 2.5, "Apple": 2.0, "Gift": 1.5, "Market": 1.5,
}

CONFIG = {
    "max_frames": 90,
    "base_feature_dim": 549,
    "feature_dim": 649,
    "hidden_dim": 320,
    "batch_size": 24,
    "epochs": 300,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.02,
    "dropout": 0.4,
    "label_smoothing": 0.15,
    "patience": 50,
    "warmup_epochs": 15,
    "focal_gamma": 2.5,
    "mixup_alpha": 0.3,
    "checkpoint_dir": "/data/checkpoints_v7",
    "train_dir": "/data/train_v2",
    "val_dir": "/data/val_v2",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=21600,
    image=image,
)
def train_model(model_type: str = "pyramid") -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from collections import Counter, defaultdict
    import math
    import json

    print("=" * 70)
    print("KSL Training v7 - " + model_type.upper())
    print("=" * 70)
    print("Device: " + torch.cuda.get_device_name(0))
    print("Hard classes: " + str(list(HARD_CLASSES.keys())))
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

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
                dist = np.linalg.norm(hand_normalized[tip])
                frame_features.append(dist)
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    dist = np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                    frame_features.append(dist)
            for tip, mcp in zip(tips, mcps):
                curl = np.linalg.norm(hand[tip] - hand[mcp])
                frame_features.append(curl)
            max_spread = 0
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    spread = np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                    max_spread = max(max_spread, spread)
            frame_features.append(max_spread)
            thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
            index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
            norm_product = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
            if norm_product > 1e-8:
                cos_angle = np.clip(np.dot(thumb_vec, index_vec) / norm_product, -1, 1)
            else:
                cos_angle = 0
            frame_features.append(cos_angle)
            v1 = hand[INDEX_MCP] - hand[WRIST]
            v2 = hand[PINKY_MCP] - hand[WRIST]
            palm_normal = np.cross(v1, v2)
            norm = np.linalg.norm(palm_normal)
            if norm > 1e-8:
                palm_normal = palm_normal / norm
            else:
                palm_normal = np.zeros(3)
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
        enhanced = np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)
        return enhanced

    class KSLDatasetV7(Dataset):
        def __init__(self, data_dir, classes, augment=False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.augment = augment
            self.num_classes = len(classes)
            self.hard_class_indices = set()
            for cls_name in HARD_CLASSES.keys():
                if cls_name in self.class_to_idx:
                    self.hard_class_indices.add(self.class_to_idx[cls_name])
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
            print("  Loaded " + str(len(self.samples)) + " samples")

        def _compute_stats(self):
            all_data = []
            for i in range(min(300, len(self.samples))):
                data = np.load(self.samples[i][0])
                all_data.append(data.flatten())
            all_data = np.concatenate(all_data)
            self.mean = all_data.mean()
            self.std = all_data.std() + 1e-8

        def __len__(self):
            return len(self.samples)

        def _augment(self, data, is_hard_class):
            if np.random.rand() > 0.6:
                speed = np.random.uniform(0.9, 1.1)
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
            noise_scale = 0.005 if is_hard_class else 0.01
            if np.random.rand() > 0.5:
                data = data + np.random.normal(0, noise_scale, data.shape).astype(np.float32)
            if np.random.rand() > 0.7:
                scale = np.random.uniform(0.98, 1.02)
                data = data * scale
            if not is_hard_class and np.random.rand() > 0.7:
                num_drop = np.random.randint(1, 4)
                drop_idx = np.random.choice(len(data), min(num_drop, len(data) - 10), replace=False)
                data[drop_idx] = 0
            if np.random.rand() > 0.5:
                left_hand = data[:, 99:162].copy()
                right_hand = data[:, 162:225].copy()
                data[:, 99:162] = right_hand
                data[:, 162:225] = left_hand
            return data

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)
            is_hard_class = label in self.hard_class_indices
            if self.mean is not None:
                data = (data - self.mean) / self.std
            if self.augment:
                data = self._augment(data, is_hard_class)
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
            data = engineer_features(data)
            return torch.from_numpy(data), label

    class TemporalPyramidModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.GELU(),
                ),
            ])
            self.temporal_fusion = nn.Sequential(
                nn.Linear(hidden_dim // 2 * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
            )
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=CONFIG["dropout"],
            )
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                ) for _ in range(4)
            ])
            self.pre_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 * 4),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
            )
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            batch, frames, features = x.shape
            x = self.input_norm(x)
            x = self.input_proj(x)
            x_conv = x.transpose(1, 2)
            scale_outputs = []
            for scale_conv in self.temporal_scales:
                scale_out = scale_conv(x_conv)
                scale_outputs.append(scale_out)
            multi_scale = torch.cat(scale_outputs, dim=1)
            multi_scale = multi_scale.transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = []
            for attn_head in self.attention_heads:
                attn_weights = F.softmax(attn_head(lstm_out), dim=1)
                context = (lstm_out * attn_weights).sum(dim=1)
                contexts.append(context)
            combined = torch.cat(contexts, dim=-1)
            x = self.pre_classifier(combined)
            logits = self.classifier(x)
            return logits

    class EnhancedHandFocusedModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.hand_encoder = nn.Sequential(
                nn.Linear(226, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.body_encoder = nn.Sequential(
                nn.Linear(423, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
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
            x = self.input_norm(x)
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
            hand_feat_attended, _ = self.cross_attention(hand_feat, body_feat, body_feat)
            combined = torch.cat([hand_feat_attended, body_feat], dim=-1)
            lstm_out, _ = self.lstm(combined)
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    class FocalLossWithBoost(nn.Module):
        def __init__(self, gamma=2.5, class_weights=None, hard_boost=None, smoothing=0.1):
            super().__init__()
            self.gamma = gamma
            self.class_weights = class_weights
            self.hard_boost = hard_boost
            self.smoothing = smoothing

        def forward(self, inputs, targets):
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / num_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            focal_weights = (1 - probs) ** self.gamma
            loss = -focal_weights * smooth_targets * log_probs
            loss = loss.sum(dim=-1)
            if self.class_weights is not None:
                weights = self.class_weights[targets]
                loss = loss * weights
            if self.hard_boost is not None:
                boost = self.hard_boost[targets]
                loss = loss * boost
            return loss.mean()

    def mixup_data(x, y, alpha=0.3):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def get_lr_schedule(epoch, warmup, total, base_lr, min_lr):
        if epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / (total - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    print("\nLoading datasets...")
    train_dataset = KSLDatasetV7(CONFIG["train_dir"], ALL_CLASSES, augment=True)
    val_dataset = KSLDatasetV7(CONFIG["val_dir"], ALL_CLASSES, augment=False)

    if len(train_dataset) == 0:
        return {"model_type": model_type, "best_val_acc": 0.0, "error": "No data"}

    class_counts = Counter([s[1] for s in train_dataset.samples])
    weights = []
    for filepath, label in train_dataset.samples:
        base_weight = 1.0 / class_counts[label]
        class_name = train_dataset.idx_to_class[label]
        if class_name in HARD_CLASSES:
            base_weight *= HARD_CLASSES[class_name]
        weights.append(base_weight)

    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print("\nCreating " + model_type + " model...")
    if model_type == "pyramid":
        model = TemporalPyramidModel(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"], hidden_dim=CONFIG["hidden_dim"])
    else:
        model = EnhancedHandFocusedModel(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"], hidden_dim=CONFIG["hidden_dim"])

    model = model.cuda()
    param_count = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(param_count))

    class_weights = torch.tensor([1.0 / max(class_counts.get(i, 1), 1) for i in range(len(ALL_CLASSES))], device="cuda")
    class_weights = class_weights / class_weights.sum() * len(ALL_CLASSES)

    hard_boost = torch.ones(len(ALL_CLASSES), device="cuda")
    for class_name, boost in HARD_CLASSES.items():
        if class_name in train_dataset.class_to_idx:
            idx = train_dataset.class_to_idx[class_name]
            hard_boost[idx] = boost

    criterion = FocalLossWithBoost(gamma=CONFIG["focal_gamma"], class_weights=class_weights, hard_boost=hard_boost, smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    best_val_acc = 0.0
    best_hard_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "number_acc": [], "word_acc": [], "hard_acc": [], "lr": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        lr = get_lr_schedule(epoch, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            if np.random.rand() > 0.6:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, CONFIG["mixup_alpha"])
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        num_correct, num_total = 0, 0
        word_correct, word_total = 0, 0
        hard_correct, hard_total = 0, 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                for i in range(batch_y.size(0)):
                    label = batch_y[i].item()
                    pred = predicted[i].item()
                    class_name = val_dataset.idx_to_class[label]
                    per_class_total[label] += 1
                    total += 1
                    if pred == label:
                        correct += 1
                        per_class_correct[label] += 1
                    if label < len(NUMBER_CLASSES):
                        num_total += 1
                        if pred == label:
                            num_correct += 1
                    else:
                        word_total += 1
                        if pred == label:
                            word_correct += 1
                    if class_name in HARD_CLASSES:
                        hard_total += 1
                        if pred == label:
                            hard_correct += 1

        val_acc = correct / total if total > 0 else 0
        num_acc = num_correct / num_total if num_total > 0 else 0
        word_acc = word_correct / word_total if word_total > 0 else 0
        hard_acc = hard_correct / hard_total if hard_total > 0 else 0

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["number_acc"].append(num_acc)
        history["word_acc"].append(word_acc)
        history["hard_acc"].append(hard_acc)
        history["lr"].append(lr)

        if epoch % 5 == 0 or val_acc > best_val_acc:
            print("Epoch " + str(epoch+1) + "/" + str(CONFIG["epochs"]) + " | Loss: " + str(round(train_loss/len(train_loader), 4)) +
                  " | Val: " + str(round(val_acc*100, 1)) + "% | Num: " + str(round(num_acc*100, 1)) +
                  "% | Word: " + str(round(word_acc*100, 1)) + "% | Hard: " + str(round(hard_acc*100, 1)) + "%")

        if val_acc > best_val_acc or (val_acc >= best_val_acc - 0.01 and hard_acc > best_hard_acc):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            best_hard_acc = hard_acc
            best_epoch = epoch + 1
            per_class_acc = {}
            for label, count in per_class_total.items():
                if count > 0:
                    class_name = val_dataset.idx_to_class[label]
                    per_class_acc[class_name] = per_class_correct[label] / count
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "number_acc": num_acc,
                "word_acc": word_acc,
                "hard_acc": hard_acc,
                "per_class_acc": per_class_acc,
                "classes": ALL_CLASSES,
                "epoch": epoch + 1,
            }, CONFIG["checkpoint_dir"] + "/ksl_" + model_type + "_best.pth")
            print("  -> Saved! Val: " + str(round(val_acc*100, 2)) + "%, Hard: " + str(round(hard_acc*100, 2)) + "%")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print("\nEarly stopping at epoch " + str(epoch+1))
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Best: " + str(round(best_val_acc*100, 2)) + "% @ Epoch " + str(best_epoch))
    print("Best Hard Class Acc: " + str(round(best_hard_acc*100, 2)) + "%")

    checkpoint = torch.load(CONFIG["checkpoint_dir"] + "/ksl_" + model_type + "_best.pth")
    print("\nPer-class accuracy (best model):")
    print("\n  HARD CLASSES:")
    for class_name in HARD_CLASSES.keys():
        if class_name in checkpoint.get("per_class_acc", {}):
            acc = checkpoint["per_class_acc"][class_name]
            print("    " + class_name + ": " + str(round(acc*100, 1)) + "%")

    with open(CONFIG["checkpoint_dir"] + "/ksl_" + model_type + "_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()
    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_hard_acc": best_hard_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main(model: str = "pyramid"):
    print("=" * 70)
    print("KSL Training v7 - Fixing Critical Performance Issues")
    print("=" * 70)
    print("\nImprovements over v5/v6:")
    print("  - Focal loss with hard class boosting (no contrastive)")
    print("  - Multi-scale temporal pyramid for multi-digit numbers")
    print("  - Gentler augmentation (was destroying features)")
    print("  - Hard class oversampling in training")
    print("  - Cross-attention between hands and body")
    print()

    if model == "both":
        results = list(train_model.map(["pyramid", "handfocused"]))
        print("\n" + "=" * 70)
        print("RESULTS:")
        for r in results:
            print("  " + r["model_type"].upper() + ": " + str(round(r["best_val_acc"]*100, 2)) + "% (Hard: " + str(round(r["best_hard_acc"]*100, 2)) + "%)")
    else:
        result = train_model.remote(model)
        print("\n" + result["model_type"].upper() + ": " + str(round(result["best_val_acc"]*100, 2)) + "%")
        print("Hard class accuracy: " + str(round(result["best_hard_acc"]*100, 2)) + "%")

    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v7/ ./checkpoints/")
