"""
KSL v7 Model Evaluation
Comprehensive evaluation with confusion matrix and per-class analysis.

Usage:
    modal run evaluate_v7.py
"""

import modal
import os

app = modal.App("ksl-evaluate-v7")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES
HARD_CLASSES = ["100", "35", "444", "125", "54", "268", "Tomatoes", "Proud", "Apple", "Gift", "Market"]

CONFIG = {"max_frames": 90, "feature_dim": 649, "hidden_dim": 320, "dropout": 0.4}


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def evaluate_v7():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL v7 Evaluation")
    print("=" * 70)

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
        left_features, right_features = compute_hand_features(left_hand), compute_hand_features(right_hand)
        left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel[1:] = right_features[1:] - right_features[:-1]
        return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)

    class TemporalPyramidModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            ])
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim // 2 * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.4))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.4)
            self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)])
            self.pre_classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2 * 4), nn.Dropout(0.4), nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2), nn.GELU(), nn.Dropout(0.4))
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [((lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1)) for head in self.attention_heads]
            return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))

    checkpoint_path = "/data/checkpoints_v7/ksl_pyramid_best.pth"
    if not os.path.exists(checkpoint_path):
        print("ERROR: Checkpoint not found")
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = TemporalPyramidModel(num_classes=len(ALL_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()
    print(f"Loaded model, saved acc: {checkpoint.get('val_acc', 0)*100:.2f}%")

    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if os.path.exists(class_dir):
            for fn in os.listdir(class_dir):
                if fn.endswith(".npy"):
                    val_samples.append((os.path.join(class_dir, fn), ALL_CLASSES.index(class_name), class_name))
    print(f"Validation samples: {len(val_samples)}")

    all_data = np.concatenate([np.load(val_samples[i][0]).flatten() for i in range(min(200, len(val_samples)))])
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input(data):
        data = (data - mean) / std
        if data.shape[0] > 90:
            data = data[np.linspace(0, data.shape[0]-1, 90, dtype=int)]
        elif data.shape[0] < 90:
            data = np.vstack([data, np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features(data)).unsqueeze(0).cuda()

    correct, top3_correct, total = 0, 0, 0
    num_correct, num_total, word_correct, word_total, hard_correct, hard_total = 0, 0, 0, 0, 0, 0
    per_class_correct, per_class_total = defaultdict(int), defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for filepath, label, class_name in val_samples:
            data = np.load(filepath).astype(np.float32)
            logits = model(prepare_input(data))
            pred = logits.argmax(dim=1).item()
            pred_class = ALL_CLASSES[pred]
            _, top3 = logits.topk(3, dim=1)

            per_class_total[class_name] += 1
            confusion[class_name][pred_class] += 1
            total += 1
            if pred == label: correct += 1; per_class_correct[class_name] += 1
            if label in top3.squeeze().cpu().numpy(): top3_correct += 1
            if label < len(NUMBER_CLASSES): num_total += 1; num_correct += (pred == label)
            else: word_total += 1; word_correct += (pred == label)
            if class_name in HARD_CLASSES: hard_total += 1; hard_correct += (pred == label)

    val_acc, top3_acc = correct/total, top3_correct/total
    num_acc = num_correct/num_total if num_total else 0
    word_acc = word_correct/word_total if word_total else 0
    hard_acc = hard_correct/hard_total if hard_total else 0

    print(f"\n{'='*70}\nRESULTS\n{'='*70}")
    print(f"Top-1: {val_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}%")
    print(f"Numbers: {num_acc*100:.2f}% | Words: {word_acc*100:.2f}% | Hard: {hard_acc*100:.2f}%")

    print(f"\n{'='*70}\nPER-CLASS ACCURACY\n{'='*70}\n\nNUMBERS:")
    for c in NUMBER_CLASSES:
        if per_class_total[c]: print(f"  {c:>6}: {per_class_correct[c]/per_class_total[c]*100:5.1f}% ({per_class_correct[c]}/{per_class_total[c]}){' <-- HARD' if c in HARD_CLASSES else ''}")
    print("\nWORDS:")
    for c in WORD_CLASSES:
        if per_class_total[c]: print(f"  {c:>12}: {per_class_correct[c]/per_class_total[c]*100:5.1f}% ({per_class_correct[c]}/{per_class_total[c]}){' <-- HARD' if c in HARD_CLASSES else ''}")

    print(f"\n{'='*70}\nTOP 15 CONFUSIONS\n{'='*70}")
    pairs = sorted([(t, p, confusion[t][p]) for t in ALL_CLASSES for p in ALL_CLASSES if t != p and confusion[t][p] > 0], key=lambda x: -x[2])
    for t, p, cnt in pairs[:15]: print(f"  {t} -> {p}: {cnt}")

    return {"val_acc": val_acc, "top3_acc": top3_acc, "num_acc": num_acc, "word_acc": word_acc, "hard_acc": hard_acc}


@app.local_entrypoint()
def main():
    results = evaluate_v7.remote()
    print(f"\nFinal: {results['val_acc']*100:.2f}% (Hard: {results['hard_acc']*100:.2f}%)")
