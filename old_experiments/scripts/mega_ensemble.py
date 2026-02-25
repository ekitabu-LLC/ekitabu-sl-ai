"""
KSL Mega Ensemble
Combines best models from all versions (v1-v5) for maximum accuracy.

Usage:
    modal run mega_ensemble.py
"""

import modal
import os

app = modal.App("ksl-mega-ensemble")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES_V5 = NUMBER_CLASSES + WORD_CLASSES  # v5 ordering

ALL_CLASSES_V1_V4 = [
    "100", "125", "17", "22", "268", "35", "388", "444", "48", "54", "66", "73", "89", "9", "91",
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def evaluate_mega_ensemble() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL MEGA ENSEMBLE - All Best Models")
    print("=" * 70)

    # =========================================================================
    # Model Definitions
    # =========================================================================

    # V1/V2 Models (225 features)
    class LSTMModelV1(nn.Module):
        def __init__(self, num_classes=30, feature_dim=225, hidden_dim=256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, num_classes))

        def forward(self, x):
            x = self.feature_proj(x)
            lstm_out, _ = self.lstm(x)
            attn = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn).sum(dim=1)
            return self.classifier(context)

    # V4 Models (549 features)
    class BodyPartAttentionV4(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            part_dim = 64
            self.pose_proj = nn.Linear(99, part_dim)
            self.left_hand_proj = nn.Linear(63, part_dim)
            self.right_hand_proj = nn.Linear(63, part_dim)
            self.face_proj = nn.Linear(204, part_dim)
            self.lips_proj = nn.Linear(120, part_dim)
            self.part_attention = nn.Sequential(
                nn.Linear(part_dim * 5, 128),
                nn.ReLU(),
                nn.Linear(128, 5),
            )
            self.out_proj = nn.Linear(part_dim * 5, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            pose = self.pose_proj(x[:, :, :99])
            lh = self.left_hand_proj(x[:, :, 99:162])
            rh = self.right_hand_proj(x[:, :, 162:225])
            face = self.face_proj(x[:, :, 225:429])
            lips = self.lips_proj(x[:, :, 429:549])
            combined = torch.cat([pose, lh, rh, face, lips], dim=-1)
            return self.norm(self.out_proj(combined))

    class EnhancedLSTMV4(nn.Module):
        def __init__(self, num_classes=30, feature_dim=549, hidden_dim=256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttentionV4(hidden_dim)
            self.input_dropout = nn.Dropout(0.1)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.temporal_attention = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes))

        def forward(self, x):
            x = self.input_norm(x)
            x = self.body_attention(x)
            x = self.input_dropout(x)
            lstm_out, _ = self.lstm(x)
            attn = F.softmax(self.temporal_attention(lstm_out), dim=1)
            context = (lstm_out * attn).sum(dim=1)
            return self.classifier(context)

    # V5 Models (649 features with hand features)
    class HandFocusedModelV5(nn.Module):
        def __init__(self, num_classes=30, feature_dim=649, hidden_dim=256):
            super().__init__()
            self.hand_encoder = nn.Sequential(nn.Linear(226, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.body_encoder = nn.Sequential(nn.Linear(423, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.number_classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim, 15))
            self.word_classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim, 15))
            self.type_classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, 2))

        def forward(self, x):
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
            return torch.cat([self.number_classifier(context), self.word_classifier(context)], dim=-1)

    class SimpleLSTMV5(nn.Module):
        def __init__(self, num_classes=30, feature_dim=649, hidden_dim=256):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2), nn.Dropout(0.3), nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes))

        def forward(self, x):
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)
            attn = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn).sum(dim=1)
            return self.classifier(context)

    # =========================================================================
    # Hand Feature Engineering (for v5 models)
    # =========================================================================

    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
    THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17

    def compute_hand_features(hand_landmarks):
        frames = hand_landmarks.shape[0]
        features = []
        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        for f in range(frames):
            hand = hand_landmarks[f].reshape(21, 3)
            frame_features = []
            wrist = hand[WRIST]
            hand_norm = hand - wrist
            for tip in tips:
                frame_features.append(np.linalg.norm(hand_norm[tip]))
            for i in range(len(tips)):
                for j in range(i+1, len(tips)):
                    frame_features.append(np.linalg.norm(hand[tips[i]] - hand[tips[j]]))
            for tip, mcp in zip(tips, mcps):
                frame_features.append(np.linalg.norm(hand[tip] - hand[mcp]))
            max_spread = max(np.linalg.norm(hand[tips[i]] - hand[tips[j]]) for i in range(len(tips)) for j in range(i+1, len(tips)))
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

    def engineer_features_v5(data):
        left_hand = data[:, 99:162]
        right_hand = data[:, 162:225]
        left_feat = compute_hand_features(left_hand)
        right_feat = compute_hand_features(right_hand)
        left_vel = np.zeros_like(left_feat)
        left_vel[1:] = left_feat[1:] - left_feat[:-1]
        right_vel = np.zeros_like(right_feat)
        right_vel[1:] = right_feat[1:] - right_feat[:-1]
        return np.concatenate([data, left_feat, right_feat, left_vel, right_vel], axis=1)

    # =========================================================================
    # Load Models
    # =========================================================================

    models = []

    def load_model(path, model_class, version, weight=1.0):
        if not os.path.exists(path):
            print(f"  Not found: {path}")
            return None
        try:
            checkpoint = torch.load(path, map_location="cuda")
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.cuda().eval()
            print(f"  Loaded: {path} (weight={weight})")
            return {"model": model, "version": version, "weight": weight}
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            return None

    print("\nLoading models...")

    # V1 models (225 features, 120 frames, old class order)
    m = load_model("/data/checkpoints/ksl_lstm_best.pth", LSTMModelV1, "v1", 0.5)
    if m: models.append(m)

    # V4 models (549 features, 90 frames, old class order)
    m = load_model("/data/checkpoints_v4/ksl_lstm_best.pth", EnhancedLSTMV4, "v4", 1.0)
    if m: models.append(m)

    # V5 models (649 features, 90 frames, new class order)
    m = load_model("/data/checkpoints_v5/ksl_handfocused_best.pth", HandFocusedModelV5, "v5", 2.0)
    if m: models.append(m)

    m = load_model("/data/checkpoints_v5/ksl_lstm_best.pth", SimpleLSTMV5, "v5", 1.5)
    if m: models.append(m)

    print(f"\nTotal models: {len(models)}")
    total_weight = sum(m["weight"] for m in models)

    # =========================================================================
    # Load Validation Data
    # =========================================================================

    print("\nLoading validation data...")

    # We need both v1 (225) and v2 (549) features
    val_samples_v1 = []
    val_samples_v2 = []

    for i, class_name in enumerate(ALL_CLASSES_V5):
        # V2 features (for v4, v5 models)
        class_dir_v2 = f"/data/val_v2/{class_name}"
        if os.path.exists(class_dir_v2):
            for filename in os.listdir(class_dir_v2):
                if filename.endswith(".npy"):
                    val_samples_v2.append((os.path.join(class_dir_v2, filename), i, class_name))

        # V1 features (for v1 models)
        class_dir_v1 = f"/data/val/{class_name}"
        if os.path.exists(class_dir_v1):
            for filename in os.listdir(class_dir_v1):
                if filename.endswith(".npy"):
                    val_samples_v1.append((os.path.join(class_dir_v1, filename), i, class_name))

    print(f"  V2 samples: {len(val_samples_v2)}")
    print(f"  V1 samples: {len(val_samples_v1)}")

    # Compute normalization stats
    def compute_stats(samples):
        all_data = []
        for i in range(min(200, len(samples))):
            data = np.load(samples[i][0])
            all_data.append(data.flatten())
        all_data = np.concatenate(all_data)
        return all_data.mean(), all_data.std() + 1e-8

    mean_v1, std_v1 = compute_stats(val_samples_v1) if val_samples_v1 else (0, 1)
    mean_v2, std_v2 = compute_stats(val_samples_v2) if val_samples_v2 else (0, 1)

    def prepare_v1(data, max_frames=120):
        data = (data - mean_v1) / std_v1
        if data.shape[0] > max_frames:
            indices = np.linspace(0, data.shape[0]-1, max_frames, dtype=int)
            data = data[indices]
        elif data.shape[0] < max_frames:
            padding = np.zeros((max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        return torch.from_numpy(data).unsqueeze(0).cuda()

    def prepare_v4(data, max_frames=90):
        data = (data - mean_v2) / std_v2
        if data.shape[0] > max_frames:
            indices = np.linspace(0, data.shape[0]-1, max_frames, dtype=int)
            data = data[indices]
        elif data.shape[0] < max_frames:
            padding = np.zeros((max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        return torch.from_numpy(data).unsqueeze(0).cuda()

    def prepare_v5(data, max_frames=90):
        data = (data - mean_v2) / std_v2
        if data.shape[0] > max_frames:
            indices = np.linspace(0, data.shape[0]-1, max_frames, dtype=int)
            data = data[indices]
        elif data.shape[0] < max_frames:
            padding = np.zeros((max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        data = engineer_features_v5(data)
        return torch.from_numpy(data).unsqueeze(0).cuda()

    # Class index mapping from v1/v4 order to v5 order
    v1_to_v5_idx = {ALL_CLASSES_V1_V4.index(c): ALL_CLASSES_V5.index(c) for c in ALL_CLASSES_V5}
    v5_to_v1_idx = {v: k for k, v in v1_to_v5_idx.items()}

    def remap_logits_v1_to_v5(logits):
        """Remap logits from v1/v4 class order to v5 class order."""
        new_logits = torch.zeros_like(logits)
        for v1_idx, v5_idx in v1_to_v5_idx.items():
            new_logits[:, v5_idx] = logits[:, v1_idx]
        return new_logits

    # =========================================================================
    # Evaluate Ensemble
    # =========================================================================

    print("\nEvaluating mega ensemble...")

    correct = 0
    top3_correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    # Use v2 samples as reference (has all classes)
    for filepath_v2, label_v5, class_name in val_samples_v2:
        # Find corresponding v1 file
        filename = os.path.basename(filepath_v2)
        filepath_v1 = f"/data/val/{class_name}/{filename}"

        data_v2 = np.load(filepath_v2).astype(np.float32)

        ensemble_logits = None

        for m in models:
            model = m["model"]
            version = m["version"]
            weight = m["weight"]

            with torch.no_grad():
                if version == "v1":
                    if os.path.exists(filepath_v1):
                        data_v1 = np.load(filepath_v1).astype(np.float32)
                        input_tensor = prepare_v1(data_v1)
                        logits = model(input_tensor)
                        logits = remap_logits_v1_to_v5(logits)
                    else:
                        continue
                elif version == "v4":
                    input_tensor = prepare_v4(data_v2)
                    logits = model(input_tensor)
                    logits = remap_logits_v1_to_v5(logits)
                else:  # v5
                    input_tensor = prepare_v5(data_v2)
                    logits = model(input_tensor)

            if ensemble_logits is None:
                ensemble_logits = logits * weight
            else:
                ensemble_logits = ensemble_logits + logits * weight

        if ensemble_logits is None:
            continue

        ensemble_logits = ensemble_logits / total_weight

        pred = ensemble_logits.argmax(dim=1).item()
        _, top3 = ensemble_logits.topk(3, dim=1)
        top3 = top3.squeeze().cpu().numpy()

        if pred == label_v5:
            correct += 1
            per_class_correct[label_v5] += 1
        if label_v5 in top3:
            top3_correct += 1
        per_class_total[label_v5] += 1
        total += 1

    acc = correct / total if total > 0 else 0
    top3_acc = top3_correct / total if total > 0 else 0

    # Compute number/word accuracy
    num_correct = sum(per_class_correct[i] for i in range(15))
    num_total = sum(per_class_total[i] for i in range(15))
    word_correct = sum(per_class_correct[i] for i in range(15, 30))
    word_total = sum(per_class_total[i] for i in range(15, 30))

    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"Top-1 Accuracy: {acc*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Numbers: {num_correct/num_total*100:.2f}% ({num_correct}/{num_total})")
    print(f"Words: {word_correct/word_total*100:.2f}% ({word_correct}/{word_total})")

    print("\nPer-class accuracy:")
    print("\n  NUMBERS:")
    for i, class_name in enumerate(NUMBER_CLASSES):
        if per_class_total[i] > 0:
            class_acc = per_class_correct[i] / per_class_total[i] * 100
            print(f"    {class_name:>6}: {class_acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")

    print("\n  WORDS:")
    for i, class_name in enumerate(WORD_CLASSES):
        idx = i + 15
        if per_class_total[idx] > 0:
            class_acc = per_class_correct[idx] / per_class_total[idx] * 100
            print(f"    {class_name:>12}: {class_acc:5.1f}% ({per_class_correct[idx]}/{per_class_total[idx]})")

    return {"accuracy": acc, "top3": top3_acc}


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("KSL Mega Ensemble")
    print("=" * 70)
    result = evaluate_mega_ensemble.remote()
    print(f"\nFinal: Top-1={result['accuracy']*100:.2f}%, Top-3={result['top3']*100:.2f}%")
