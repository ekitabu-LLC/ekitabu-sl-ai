"""
KSL Evaluation with Test-Time Augmentation (TTA)

Averages predictions over multiple augmented versions of each sample.
This helps with signer variations by considering multiple interpretations.

Usage:
    modal run evaluate_tta.py
"""

import modal
import os

app = modal.App("ksl-evaluate-tta")
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
def evaluate_with_tta():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict
    import random

    print("=" * 70)
    print("KSL Evaluation with Test-Time Augmentation")
    print("=" * 70)

    device = torch.device("cuda")

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
    # TTA Augmentations
    # =========================================================================

    def temporal_warp(data, sigma=0.2):
        n_frames = len(data)
        if n_frames < 5:
            return data
        orig_steps = np.arange(n_frames)
        warp_steps = orig_steps + np.random.normal(0, sigma, n_frames).cumsum()
        warp_steps = np.clip(warp_steps, 0, n_frames - 1)
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min() + 1e-8) * (n_frames - 1)
        warped = np.zeros_like(data)
        for i in range(data.shape[1]):
            warped[:, i] = np.interp(orig_steps, warp_steps, data[:, i])
        return warped

    def temporal_shift(data, shift_ratio=0.1):
        n_frames = len(data)
        shift = int(n_frames * shift_ratio * (random.random() * 2 - 1))
        if shift == 0:
            return data
        elif shift > 0:
            return np.vstack([np.zeros((shift, data.shape[1])), data[:-shift]])
        else:
            return np.vstack([data[-shift:], np.zeros((-shift, data.shape[1]))])

    def speed_change(data, speed_factor):
        """Change playback speed."""
        n_frames = len(data)
        new_n_frames = int(n_frames / speed_factor)
        if new_n_frames < 5:
            return data

        new_indices = np.linspace(0, n_frames - 1, new_n_frames)
        new_data = np.zeros((new_n_frames, data.shape[1]))
        for i in range(data.shape[1]):
            new_data[:, i] = np.interp(new_indices, np.arange(n_frames), data[:, i])
        return new_data

    def get_tta_augmentations():
        """Return list of augmentation functions for TTA."""
        return [
            lambda x: x,  # Original
            lambda x: temporal_warp(x, sigma=0.15),  # Slight warp
            lambda x: temporal_warp(x, sigma=0.25),  # More warp
            lambda x: speed_change(x, 0.9),  # Slower
            lambda x: speed_change(x, 1.1),  # Faster
            lambda x: x[::-1].copy(),  # Reversed (might help with mirror signing)
        ]

    # =========================================================================
    # Model Definition (HandFocused from v5)
    # =========================================================================

    class HandFocusedModel(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=256):
            super().__init__()
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
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.number_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 15)
            )
            self.word_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 15)
            )
            self.type_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, 2)
            )

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

    # =========================================================================
    # Load Model and Data
    # =========================================================================

    # Load best model
    checkpoint = torch.load("/data/checkpoints_v5/ksl_handfocused_best.pth", map_location=device)
    model = HandFocusedModel(num_classes=30)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print("Loaded HandFocused model from v5")

    # Load validation samples
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                val_samples.append((os.path.join(class_dir, filename), ALL_CLASSES.index(class_name)))

    print(f"Validation samples: {len(val_samples)}")

    # Compute normalization
    all_data = []
    for i in range(min(200, len(val_samples))):
        data = np.load(val_samples[i][0])
        all_data.append(data.flatten())
    all_data = np.concatenate(all_data)
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input(data, max_frames=90):
        data = (data - mean) / std
        if data.shape[0] > max_frames:
            indices = np.linspace(0, data.shape[0] - 1, max_frames, dtype=int)
            data = data[indices]
        elif data.shape[0] < max_frames:
            padding = np.zeros((max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        data = engineer_features(data)
        return torch.from_numpy(data).unsqueeze(0).to(device)

    # =========================================================================
    # Evaluation Without TTA (baseline)
    # =========================================================================

    print("\n" + "=" * 70)
    print("Evaluation WITHOUT TTA (baseline)")
    print("=" * 70)

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for filepath, label in val_samples:
        data = np.load(filepath).astype(np.float32)
        input_tensor = prepare_input(data)

        with torch.no_grad():
            logits = model(input_tensor)

        pred = logits.argmax(dim=1).item()
        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        per_class_total[label] += 1
        total += 1

    baseline_acc = correct / total
    print(f"Accuracy: {baseline_acc*100:.2f}%")

    # =========================================================================
    # Evaluation WITH TTA
    # =========================================================================

    print("\n" + "=" * 70)
    print("Evaluation WITH TTA (6 augmentations)")
    print("=" * 70)

    tta_augmentations = get_tta_augmentations()

    correct = 0
    top3_correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    hard_classes = ["22", "444", "89", "35", "Market", "Teach"]
    hard_class_results = {c: {"correct": 0, "total": 0} for c in hard_classes}

    for filepath, label in val_samples:
        raw_data = np.load(filepath).astype(np.float32)

        # Collect predictions from all augmentations
        all_logits = []

        for aug_fn in tta_augmentations:
            try:
                aug_data = aug_fn(raw_data.copy())
                input_tensor = prepare_input(aug_data)

                with torch.no_grad():
                    logits = model(input_tensor)
                    all_logits.append(F.softmax(logits, dim=1))
            except:
                continue

        # Average predictions
        avg_probs = torch.stack(all_logits).mean(dim=0)
        pred = avg_probs.argmax(dim=1).item()
        _, top3 = avg_probs.topk(3, dim=1)
        top3 = top3.squeeze().cpu().numpy()

        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        if label in top3:
            top3_correct += 1
        per_class_total[label] += 1
        total += 1

        # Track hard classes
        class_name = ALL_CLASSES[label]
        if class_name in hard_classes:
            hard_class_results[class_name]["total"] += 1
            if pred == label:
                hard_class_results[class_name]["correct"] += 1

    tta_acc = correct / total
    tta_top3 = top3_correct / total

    print(f"Top-1 Accuracy: {tta_acc*100:.2f}% (baseline: {baseline_acc*100:.2f}%)")
    print(f"Top-3 Accuracy: {tta_top3*100:.2f}%")
    print(f"Improvement: +{(tta_acc - baseline_acc)*100:.2f}%")

    print("\nHard classes with TTA:")
    for class_name in hard_classes:
        r = hard_class_results[class_name]
        if r["total"] > 0:
            acc = r["correct"] / r["total"] * 100
            print(f"  {class_name:8s}: {acc:.1f}% ({r['correct']}/{r['total']})")

    print("\nPer-class accuracy:")
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

    return {
        "baseline_acc": baseline_acc,
        "tta_acc": tta_acc,
        "tta_top3": tta_top3,
        "improvement": tta_acc - baseline_acc
    }


@app.local_entrypoint()
def main():
    result = evaluate_with_tta.remote()
    print(f"\nSummary:")
    print(f"  Baseline: {result['baseline_acc']*100:.2f}%")
    print(f"  With TTA: {result['tta_acc']*100:.2f}%")
    print(f"  Top-3: {result['tta_top3']*100:.2f}%")
