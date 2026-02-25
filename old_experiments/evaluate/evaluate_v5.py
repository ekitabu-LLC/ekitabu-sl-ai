"""
KSL v5 Model Evaluation
Evaluates v5 models with hand-focused features for number recognition.

Usage:
    modal run evaluate_v5.py
"""

import modal
import os

app = modal.App("ksl-evaluate-v5")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

# Separate classes (must match train_ksl_v5.py)
NUMBER_CLASSES = ["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"]
WORD_CLASSES = ["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"]
ALL_CLASSES = sorted(NUMBER_CLASSES) + sorted(WORD_CLASSES)

CONFIG = {
    "max_frames": 90,
    "base_feature_dim": 549,
    "feature_dim": 649,  # 549 base + 100 hand features
    "dropout": 0.3,
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def evaluate_models() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL v5 Evaluation - Hand-Focused Features")
    print("=" * 70)

    # =========================================================================
    # Hand Feature Engineering (must match train_ksl_v5.py)
    # =========================================================================

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
    # Model Definitions (must match train_ksl_v5.py)
    # =========================================================================

    class HandFocusedModel(nn.Module):
        """Model with separate processing for hands (numbers) vs full body (words)."""

        def __init__(self, num_classes: int, feature_dim: int = 649, hidden_dim: int = 256):
            super().__init__()

            # Feature dimensions
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
                nn.Linear(hidden_dim * 2, 2),
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
            full_logits = torch.cat([number_logits, word_logits], dim=-1)

            if return_type_logits:
                return full_logits, type_logits
            return full_logits

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
    # Load Models
    # =========================================================================

    def load_model(path, model_class):
        if not os.path.exists(path):
            return None
        checkpoint = torch.load(path, map_location="cuda")
        model = model_class(num_classes=len(ALL_CLASSES))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda().eval()
        return model

    models = []
    model_names = []

    # Load v5 models
    lstm = load_model("/data/checkpoints_v5/ksl_lstm_best.pth", SimpleLSTM)
    if lstm:
        models.append(lstm)
        model_names.append("LSTM-v5")
        print("Loaded: LSTM-v5")

    handfocused = load_model("/data/checkpoints_v5/ksl_handfocused_best.pth", HandFocusedModel)
    if handfocused:
        models.append(handfocused)
        model_names.append("HandFocused-v5")
        print("Loaded: HandFocused-v5")

    print(f"\nTotal models: {len(models)}")

    # =========================================================================
    # Load Validation Data
    # =========================================================================

    print("\nLoading validation data (v2 features)...")
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                val_samples.append((os.path.join(class_dir, filename), ALL_CLASSES.index(class_name)))

    print(f"Validation samples: {len(val_samples)}")

    # Compute normalization stats
    all_data = []
    for i in range(min(200, len(val_samples))):
        data = np.load(val_samples[i][0])
        all_data.append(data.flatten())
    all_data = np.concatenate(all_data)
    mean = all_data.mean()
    std = all_data.std() + 1e-8

    def prepare_input(data):
        # Normalize base features
        data = (data - mean) / std

        # Pad/crop
        if data.shape[0] > CONFIG["max_frames"]:
            indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
            data = data[indices]
        elif data.shape[0] < CONFIG["max_frames"]:
            padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])

        # Add hand features (crucial for v5!)
        data = engineer_features(data)

        return torch.from_numpy(data).unsqueeze(0).cuda()

    # =========================================================================
    # Evaluate Each Model
    # =========================================================================

    results = {}
    num_number_classes = len(NUMBER_CLASSES)

    for model, name in zip(models, model_names):
        correct = 0
        top3_correct = 0
        total = 0
        number_correct = 0
        number_total = 0
        word_correct = 0
        word_total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        for filepath, label in val_samples:
            data = np.load(filepath).astype(np.float32)
            input_tensor = prepare_input(data)

            with torch.no_grad():
                logits = model(input_tensor)

            pred = logits.argmax(dim=1).item()
            _, top3 = logits.topk(3, dim=1)
            top3 = top3.squeeze().cpu().numpy()

            if pred == label:
                correct += 1
                per_class_correct[label] += 1
            if label in top3:
                top3_correct += 1
            per_class_total[label] += 1
            total += 1

            # Separate accuracy for numbers (first 15 classes) and words
            if label < num_number_classes:
                number_total += 1
                if pred == label:
                    number_correct += 1
            else:
                word_total += 1
                if pred == label:
                    word_correct += 1

        acc = correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        number_acc = number_correct / number_total if number_total > 0 else 0
        word_acc = word_correct / word_total if word_total > 0 else 0

        results[name] = {
            "accuracy": acc,
            "top3": top3_acc,
            "number_acc": number_acc,
            "word_acc": word_acc,
        }

        print(f"\n{name}:")
        print(f"  Top-1: {acc*100:.2f}%")
        print(f"  Top-3: {top3_acc*100:.2f}%")
        print(f"  Numbers: {number_acc*100:.2f}% ({number_correct}/{number_total})")
        print(f"  Words: {word_acc*100:.2f}% ({word_correct}/{word_total})")

    # =========================================================================
    # Ensemble Evaluation
    # =========================================================================

    if len(models) > 1:
        print("\n" + "=" * 70)
        print("ENSEMBLE (All v5 models)")
        print("=" * 70)

        correct = 0
        top3_correct = 0
        total = 0
        number_correct = 0
        number_total = 0
        word_correct = 0
        word_total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        for filepath, label in val_samples:
            data = np.load(filepath).astype(np.float32)
            input_tensor = prepare_input(data)

            ensemble_logits = None
            for model in models:
                with torch.no_grad():
                    logits = model(input_tensor)
                if ensemble_logits is None:
                    ensemble_logits = logits
                else:
                    ensemble_logits = ensemble_logits + logits

            ensemble_logits = ensemble_logits / len(models)

            pred = ensemble_logits.argmax(dim=1).item()
            _, top3 = ensemble_logits.topk(3, dim=1)
            top3 = top3.squeeze().cpu().numpy()

            if pred == label:
                correct += 1
                per_class_correct[label] += 1
            if label in top3:
                top3_correct += 1
            per_class_total[label] += 1
            total += 1

            # Separate accuracy
            if label < num_number_classes:
                number_total += 1
                if pred == label:
                    number_correct += 1
            else:
                word_total += 1
                if pred == label:
                    word_correct += 1

        acc = correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        number_acc = number_correct / number_total if number_total > 0 else 0
        word_acc = word_correct / word_total if word_total > 0 else 0

        print(f"Top-1 Accuracy: {acc*100:.2f}%")
        print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
        print(f"Numbers: {number_acc*100:.2f}% ({number_correct}/{number_total})")
        print(f"Words: {word_acc*100:.2f}% ({word_correct}/{word_total})")

        print("\nPer-class accuracy:")
        print("\n  NUMBERS:")
        for i, class_name in enumerate(sorted(NUMBER_CLASSES)):
            idx = ALL_CLASSES.index(class_name)
            if per_class_total[idx] > 0:
                class_acc = per_class_correct[idx] / per_class_total[idx] * 100
                print(f"    {class_name:12s}: {class_acc:5.1f}% ({per_class_correct[idx]}/{per_class_total[idx]})")

        print("\n  WORDS:")
        for class_name in sorted(WORD_CLASSES):
            idx = ALL_CLASSES.index(class_name)
            if per_class_total[idx] > 0:
                class_acc = per_class_correct[idx] / per_class_total[idx] * 100
                print(f"    {class_name:12s}: {class_acc:5.1f}% ({per_class_correct[idx]}/{per_class_total[idx]})")

        results["Ensemble"] = {
            "accuracy": acc,
            "top3": top3_acc,
            "number_acc": number_acc,
            "word_acc": word_acc,
        }

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("KSL v5 Evaluation")
    print("=" * 70)

    results = evaluate_models.remote()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        print(f"{name}: Top-1={res['accuracy']*100:.2f}%, Top-3={res['top3']*100:.2f}%, Numbers={res['number_acc']*100:.2f}%, Words={res['word_acc']*100:.2f}%")
