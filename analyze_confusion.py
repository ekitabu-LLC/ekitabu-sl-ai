"""
KSL Confusion Analysis
Analyzes what hard classes are being confused with.

Usage:
    modal run analyze_confusion.py
"""

import modal
import os

app = modal.App("ksl-confusion-analysis")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def analyze_confusion() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL Confusion Analysis")
    print("=" * 70)

    # Model definitions (same as evaluate_v5.py)
    CONFIG = {"max_frames": 90, "dropout": 0.3}

    # Hand feature computation
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

    class SimpleLSTM(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=256):
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

    # Load model
    checkpoint = torch.load("/data/checkpoints_v5/ksl_handfocused_best.pth", map_location="cuda")

    # Need to use HandFocused model - let me define it
    class HandFocusedModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=256):
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
            number_logits = self.number_classifier(context)
            word_logits = self.word_classifier(context)
            return torch.cat([number_logits, word_logits], dim=-1)

    model = HandFocusedModel(num_classes=30)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    print("Loaded HandFocused model")

    # Load validation data
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                val_samples.append((os.path.join(class_dir, filename), ALL_CLASSES.index(class_name)))

    # Compute normalization
    all_data = []
    for i in range(min(200, len(val_samples))):
        data = np.load(val_samples[i][0])
        all_data.append(data.flatten())
    all_data = np.concatenate(all_data)
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input(data):
        data = (data - mean) / std
        if data.shape[0] > 90:
            indices = np.linspace(0, data.shape[0]-1, 90, dtype=int)
            data = data[indices]
        elif data.shape[0] < 90:
            padding = np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        data = engineer_features(data)
        return torch.from_numpy(data).unsqueeze(0).cuda()

    # Build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    predictions_by_class = defaultdict(list)

    print("\nAnalyzing predictions...")
    for filepath, label in val_samples:
        data = np.load(filepath).astype(np.float32)
        input_tensor = prepare_input(data)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)

        pred = logits.argmax(dim=1).item()
        true_class = ALL_CLASSES[label]
        pred_class = ALL_CLASSES[pred]

        confusion[true_class][pred_class] += 1
        predictions_by_class[true_class].append((pred_class, probs[0, pred].item()))

    # Print confusion analysis for hard classes
    hard_classes = ["22", "444", "89", "Market", "Teach", "35"]

    print("\n" + "=" * 70)
    print("CONFUSION ANALYSIS FOR HARD CLASSES")
    print("=" * 70)

    for cls in hard_classes:
        if cls not in confusion:
            continue

        print(f"\n{cls}:")
        total = sum(confusion[cls].values())
        correct = confusion[cls][cls]
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

        print(f"  Confused with:")
        sorted_confusions = sorted(confusion[cls].items(), key=lambda x: -x[1])
        for pred_cls, count in sorted_confusions[:5]:
            if pred_cls != cls and count > 0:
                print(f"    -> {pred_cls}: {count} times ({count/total*100:.1f}%)")

    # Print full confusion matrix for numbers
    print("\n" + "=" * 70)
    print("NUMBER CONFUSION MATRIX")
    print("=" * 70)

    print("\n" + " " * 8 + " ".join(f"{c:>5}" for c in NUMBER_CLASSES))
    for true_cls in NUMBER_CLASSES:
        row = [confusion[true_cls].get(pred_cls, 0) for pred_cls in NUMBER_CLASSES]
        row_str = " ".join(f"{v:>5}" for v in row)
        total = sum(row)
        acc = confusion[true_cls].get(true_cls, 0) / total * 100 if total > 0 else 0
        print(f"{true_cls:>6}: {row_str} | {acc:.0f}%")

    # Identify similar pairs
    print("\n" + "=" * 70)
    print("MOST CONFUSED PAIRS")
    print("=" * 70)

    pairs = []
    for true_cls in ALL_CLASSES:
        for pred_cls in ALL_CLASSES:
            if true_cls != pred_cls:
                count = confusion[true_cls][pred_cls]
                if count > 0:
                    pairs.append((true_cls, pred_cls, count))

    pairs.sort(key=lambda x: -x[2])
    print("\nTop 15 confusions:")
    for true_cls, pred_cls, count in pairs[:15]:
        print(f"  {true_cls} -> {pred_cls}: {count} times")

    return {"hard_classes_analyzed": hard_classes}


@app.local_entrypoint()
def main():
    analyze_confusion.remote()
