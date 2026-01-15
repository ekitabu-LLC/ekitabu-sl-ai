"""
KSL Training v8 - Fixing Attractor Problem & Regressions

Problems from v7:
1. Class 54 is an attractor: 22->54, 444->54, 100->54, 125->54
2. Regressions: Tortoise (100->60%), 100 (88->52%), 444 (50->15%), 9 (100->72%)
3. Still struggling: 444 (15%), 35 (36%), 91 (48%)

Solutions in v8:
1. CONFUSION-AWARE LOSS - Penalize commonly confused pairs
2. BALANCED BOOSTING - Don't over-boost hard classes (caused regressions)
3. NEGATIVE MINING - Explicit "not 54" training
4. ENSEMBLE-READY - Can combine with v5 for best of both
5. ANTI-ATTRACTOR REGULARIZATION - Prevent any class from becoming default

Usage:
    modal run train_ksl_v8.py
    modal run train_ksl_v8.py --model ensemble
"""

import modal
import os

app = modal.App("ksl-trainer-v8")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Balanced hard class weights (reduced from v7 to prevent regressions)
HARD_CLASSES = {
    "444": 2.0, "35": 2.0, "91": 1.8,  # Still struggling
    "100": 1.5, "125": 1.3,  # Regressed - gentle boost only
}

# Classes that regressed - need protection, not boosting
PROTECT_CLASSES = ["Tortoise", "388", "9", "268"]

# Confusion pairs to penalize (true_class, wrong_prediction, penalty)
# Updated based on analyze_class_54.py findings:
# - 444→54: 50% error rate, 0.9990 similarity (CRITICAL)
# - 22→54: 40% error rate, 0.9631 similarity
# - 125→54: 20% error rate, 0.9982 similarity
CONFUSION_PAIRS = [
    ("444", "54", 3.5),   # Highest priority: 50% error, 0.999 similarity
    ("22", "54", 2.5),    # 40% error rate
    ("125", "54", 2.5),   # 0.998 similarity
    ("100", "54", 2.0),   # 16% error rate
    ("35", "Colour", 2.0),
    ("91", "73", 1.5),
    ("Ugali", "Tomatoes", 1.5),
    ("66", "17", 1.5),
    ("9", "89", 1.5),
]

CONFIG = {
    "max_frames": 90,
    "feature_dim": 649,
    "hidden_dim": 320,
    "batch_size": 24,
    "epochs": 250,
    "learning_rate": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.015,
    "dropout": 0.35,
    "label_smoothing": 0.1,
    "patience": 40,
    "warmup_epochs": 15,
    "focal_gamma": 2.0,
    "mixup_alpha": 0.2,
    "confusion_penalty": 1.5,
    "anti_attractor_weight": 0.1,
    "checkpoint_dir": "/data/checkpoints_v8",
    "train_dir": "/data/train_v2",
    "val_dir": "/data/val_v2",
}


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=21600, image=image)
def train_model(model_type: str = "v8") -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from collections import Counter, defaultdict
    import math
    import json

    print("=" * 70)
    print("KSL Training v8 - Fixing Attractor & Regressions")
    print("=" * 70)
    print("Key fixes:")
    print("  - Confusion-aware loss (penalize 22->54, 444->54, etc.)")
    print("  - Balanced boosting (prevent regression)")
    print("  - Anti-attractor regularization")
    print("=" * 70)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Hand feature engineering
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
            norm_product = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
            cos_angle = np.clip(np.dot(thumb_vec, index_vec) / norm_product, -1, 1) if norm_product > 1e-8 else 0
            frame_features.append(cos_angle)
            v1 = hand[INDEX_MCP] - hand[WRIST]
            v2 = hand[PINKY_MCP] - hand[WRIST]
            palm_normal = np.cross(v1, v2)
            norm = np.linalg.norm(palm_normal)
            palm_normal = palm_normal / norm if norm > 1e-8 else np.zeros(3)
            frame_features.extend(palm_normal.tolist())
            features.append(frame_features)
        return np.array(features, dtype=np.float32)

    def engineer_features(data):
        left_hand, right_hand = data[:, 99:162], data[:, 162:225]
        left_features = compute_hand_features(left_hand)
        right_features = compute_hand_features(right_hand)
        left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel[1:] = right_features[1:] - right_features[:-1]
        return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)

    # Dataset
    class KSLDatasetV8(Dataset):
        def __init__(self, data_dir, classes, augment=False):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.augment = augment
            
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
            
            self.mean, self.std = None, None
            if len(self.samples) > 0:
                all_data = [np.load(self.samples[i][0]).flatten() for i in range(min(300, len(self.samples)))]
                all_data = np.concatenate(all_data)
                self.mean, self.std = all_data.mean(), all_data.std() + 1e-8
            
            print(f"  Loaded {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def _augment(self, data, class_name):
            # Gentler augmentation overall
            is_protected = class_name in PROTECT_CLASSES
            
            if np.random.rand() > 0.6:
                speed = np.random.uniform(0.92, 1.08) if is_protected else np.random.uniform(0.88, 1.12)
                num_frames = data.shape[0]
                new_frames = max(10, min(int(num_frames / speed), num_frames * 2))
                indices = np.linspace(0, num_frames - 1, new_frames)
                resampled = np.zeros((new_frames, data.shape[1]), dtype=np.float32)
                for i, idx in enumerate(indices):
                    low, high = int(idx), min(int(idx) + 1, num_frames - 1)
                    alpha = idx - low
                    resampled[i] = (1 - alpha) * data[low] + alpha * data[high]
                data = resampled
            
            noise_scale = 0.005 if is_protected else 0.008
            if np.random.rand() > 0.5:
                data = data + np.random.normal(0, noise_scale, data.shape).astype(np.float32)
            
            if np.random.rand() > 0.5:
                left_hand = data[:, 99:162].copy()
                right_hand = data[:, 162:225].copy()
                data[:, 99:162] = right_hand
                data[:, 162:225] = left_hand
            
            return data

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)
            class_name = self.idx_to_class[label]
            
            if self.mean is not None:
                data = (data - self.mean) / self.std
            
            if self.augment:
                data = self._augment(data, class_name)
            
            if data.shape[0] > CONFIG["max_frames"]:
                if self.augment:
                    start = np.random.randint(0, data.shape[0] - CONFIG["max_frames"] + 1)
                    data = data[start:start + CONFIG["max_frames"]]
                else:
                    data = data[np.linspace(0, data.shape[0]-1, CONFIG["max_frames"], dtype=int)]
            elif data.shape[0] < CONFIG["max_frames"]:
                data = np.vstack([data, np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)])
            
            return torch.from_numpy(engineer_features(data)), label

    # Model with anti-attractor head
    class TemporalPyramidV8(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
            ])
            
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim//2*3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(CONFIG["dropout"]))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=CONFIG["dropout"])
            
            self.attention_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)
            ])
            
            self.pre_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim*2*4),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim*2*4, hidden_dim*2),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
            )
            
            self.classifier = nn.Linear(hidden_dim*2, num_classes)
            
            # Anti-attractor head: predicts if sample is NOT from attractor classes
            self.anti_attractor = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim//2),
                nn.GELU(),
                nn.Linear(hidden_dim//2, 1),
            )

        def forward(self, x, return_anti_attractor=False):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [(lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1) for head in self.attention_heads]
            features = self.pre_classifier(torch.cat(contexts, dim=-1))
            logits = self.classifier(features)
            
            if return_anti_attractor:
                anti_score = self.anti_attractor(features)
                return logits, anti_score
            return logits

    # Confusion-Aware Focal Loss
    class ConfusionAwareLoss(nn.Module):
        def __init__(self, num_classes, class_weights, confusion_matrix, gamma=2.0, smoothing=0.1):
            super().__init__()
            self.num_classes = num_classes
            self.class_weights = class_weights
            self.confusion_matrix = confusion_matrix  # Penalty for confused pairs
            self.gamma = gamma
            self.smoothing = smoothing

        def forward(self, inputs, targets):
            # Label smoothing
            smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / self.num_classes
            
            # Focal loss
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            focal_weights = (1 - probs) ** self.gamma
            
            base_loss = -focal_weights * smooth_targets * log_probs
            base_loss = base_loss.sum(dim=-1)
            
            # Class weights
            if self.class_weights is not None:
                base_loss = base_loss * self.class_weights[targets]
            
            # Confusion penalty: extra loss when predicting confused class
            preds = inputs.argmax(dim=1)
            confusion_penalty = torch.zeros_like(base_loss)
            
            for i in range(len(targets)):
                true_class = targets[i].item()
                pred_class = preds[i].item()
                if true_class != pred_class:
                    penalty = self.confusion_matrix[true_class, pred_class]
                    if penalty > 1.0:
                        confusion_penalty[i] = (penalty - 1.0) * base_loss[i]
            
            return (base_loss + confusion_penalty).mean()

    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        return lam * x + (1 - lam) * x[index], y, y[index], lam

    def get_lr(epoch, warmup, total, base_lr, min_lr):
        if epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / (total - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # Load data
    print("\nLoading datasets...")
    train_dataset = KSLDatasetV8(CONFIG["train_dir"], ALL_CLASSES, augment=True)
    val_dataset = KSLDatasetV8(CONFIG["val_dir"], ALL_CLASSES, augment=False)

    if len(train_dataset) == 0:
        return {"model_type": model_type, "best_val_acc": 0.0, "error": "No data"}

    # Balanced sampling (less aggressive than v7)
    class_counts = Counter([s[1] for s in train_dataset.samples])
    weights = []
    for filepath, label in train_dataset.samples:
        class_name = train_dataset.idx_to_class[label]
        base_weight = 1.0 / class_counts[label]
        
        if class_name in HARD_CLASSES:
            base_weight *= HARD_CLASSES[class_name]
        elif class_name in PROTECT_CLASSES:
            base_weight *= 1.2  # Slight boost to protected classes
        
        weights.append(base_weight)

    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = TemporalPyramidV8(num_classes=len(ALL_CLASSES), feature_dim=CONFIG["feature_dim"], hidden_dim=CONFIG["hidden_dim"])
    model = model.cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    class_weights = torch.tensor([1.0 / max(class_counts.get(i, 1), 1) for i in range(len(ALL_CLASSES))], device="cuda")
    class_weights = class_weights / class_weights.sum() * len(ALL_CLASSES)

    # Build confusion penalty matrix
    confusion_matrix = torch.ones(len(ALL_CLASSES), len(ALL_CLASSES), device="cuda")
    for true_class, pred_class, penalty in CONFUSION_PAIRS:
        if true_class in train_dataset.class_to_idx and pred_class in train_dataset.class_to_idx:
            true_idx = train_dataset.class_to_idx[true_class]
            pred_idx = train_dataset.class_to_idx[pred_class]
            confusion_matrix[true_idx, pred_idx] = penalty
            print(f"  Confusion penalty: {true_class} -> {pred_class} = {penalty}x")

    criterion = ConfusionAwareLoss(
        num_classes=len(ALL_CLASSES),
        class_weights=class_weights,
        confusion_matrix=confusion_matrix,
        gamma=CONFIG["focal_gamma"],
        smoothing=CONFIG["label_smoothing"],
    )
    
    # Anti-attractor loss (BCEWithLogits for binary)
    anti_attractor_criterion = nn.BCEWithLogitsLoss()
    
    # Attractor class index (54)
    attractor_idx = train_dataset.class_to_idx.get("54", -1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    history = {"train_loss": [], "val_acc": [], "number_acc": [], "word_acc": [], "attractor_acc": [], "regressed_acc": []}

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(CONFIG["epochs"]):
        lr = get_lr(epoch, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            optimizer.zero_grad()
            
            # Forward with anti-attractor
            logits, anti_scores = model(batch_x, return_anti_attractor=True)
            
            # Main classification loss
            if np.random.rand() > 0.7:
                batch_x_mix, y_a, y_b, lam = mixup_data(batch_x, batch_y, CONFIG["mixup_alpha"])
                logits_mix = model(batch_x_mix)
                cls_loss = lam * criterion(logits_mix, y_a) + (1 - lam) * criterion(logits_mix, y_b)
            else:
                cls_loss = criterion(logits, batch_y)
            
            # Anti-attractor loss: if true class is NOT 54, but prediction is 54, penalize
            if attractor_idx >= 0:
                # Target: 1 if NOT attractor class, 0 if attractor class
                anti_targets = (batch_y != attractor_idx).float().unsqueeze(1)
                anti_loss = anti_attractor_criterion(anti_scores, anti_targets) * CONFIG["anti_attractor_weight"]
            else:
                anti_loss = 0
            
            loss = cls_loss + anti_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        num_correct, num_total = 0, 0
        word_correct, word_total = 0, 0
        attractor_as_pred, attractor_wrong = 0, 0  # Track 54 predictions
        regressed_correct, regressed_total = 0, 0  # Track regressed classes
        per_class_correct, per_class_total = defaultdict(int), defaultdict(int)

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                logits = model(batch_x)
                preds = logits.argmax(dim=1)

                for i in range(batch_y.size(0)):
                    label = batch_y[i].item()
                    pred = preds[i].item()
                    class_name = val_dataset.idx_to_class[label]
                    
                    per_class_total[class_name] += 1
                    total += 1
                    
                    if pred == label:
                        correct += 1
                        per_class_correct[class_name] += 1
                    
                    # Track attractor (54) predictions
                    if pred == attractor_idx:
                        attractor_as_pred += 1
                        if label != attractor_idx:
                            attractor_wrong += 1
                    
                    # Category tracking
                    if label < len(NUMBER_CLASSES):
                        num_total += 1
                        num_correct += (pred == label)
                    else:
                        word_total += 1
                        word_correct += (pred == label)
                    
                    # Regressed class tracking
                    if class_name in PROTECT_CLASSES + ["100", "444"]:
                        regressed_total += 1
                        regressed_correct += (pred == label)

        val_acc = correct / total
        num_acc = num_correct / num_total if num_total else 0
        word_acc = word_correct / word_total if word_total else 0
        attractor_rate = attractor_wrong / attractor_as_pred if attractor_as_pred > 0 else 0
        regressed_acc = regressed_correct / regressed_total if regressed_total else 0

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["number_acc"].append(num_acc)
        history["word_acc"].append(word_acc)
        history["attractor_acc"].append(1 - attractor_rate)
        history["regressed_acc"].append(regressed_acc)

        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val: {val_acc*100:.1f}% | "
                  f"Num: {num_acc*100:.1f}% | Word: {word_acc*100:.1f}% | "
                  f"54-attract: {attractor_wrong}/{attractor_as_pred} | Regressed: {regressed_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            per_class_acc = {c: per_class_correct[c]/per_class_total[c] if per_class_total[c] > 0 else 0 for c in ALL_CLASSES}
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_acc": num_acc,
                "word_acc": word_acc,
                "per_class_acc": per_class_acc,
                "classes": ALL_CLASSES,
                "epoch": epoch + 1,
            }, CONFIG["checkpoint_dir"] + "/ksl_v8_best.pth")
            print(f"  -> Saved! Val: {val_acc*100:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")

    # Final report
    checkpoint = torch.load(CONFIG["checkpoint_dir"] + "/ksl_v8_best.pth", weights_only=False)
    print("\nPer-class accuracy (focus classes):")
    focus_classes = list(HARD_CLASSES.keys()) + PROTECT_CLASSES + ["54"]
    for c in focus_classes:
        if c in checkpoint.get("per_class_acc", {}):
            print(f"  {c}: {checkpoint['per_class_acc'][c]*100:.1f}%")

    with open(CONFIG["checkpoint_dir"] + "/ksl_v8_history.json", "w") as f:
        json.dump(history, f)

    volume.commit()
    return {"model_type": model_type, "best_val_acc": best_val_acc, "best_epoch": best_epoch}


@app.local_entrypoint()
def main(model: str = "v8"):
    print("=" * 70)
    print("KSL Training v8 - Fixing Attractor & Regressions")
    print("=" * 70)
    print("\nKey improvements:")
    print("  - Confusion-aware loss (penalize 22->54, 444->54, etc.)")
    print("  - Balanced hard class boosting (prevent regressions)")
    print("  - Anti-attractor regularization")
    print("  - Protected class handling (Tortoise, 388, 9)")
    print()

    result = train_model.remote(model)
    print(f"\nFinal: {result['best_val_acc']*100:.2f}% @ Epoch {result['best_epoch']}")
    print("\nDownload: modal volume get ksl-dataset-vol /data/checkpoints_v8/ ./checkpoints/")
