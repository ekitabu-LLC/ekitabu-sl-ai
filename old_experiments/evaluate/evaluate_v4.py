"""
KSL v4 Model Evaluation
Evaluates v4 models with face/lips features.

Usage:
    modal run evaluate_v4.py
"""

import modal
import os

app = modal.App("ksl-evaluate-v4")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

ALL_CLASSES = [
    "100", "125", "17", "22", "268", "35", "388", "444", "48", "54", "66", "73", "89", "9", "91",
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]

CONFIG = {
    "max_frames": 90,
    "feature_dim": 549,
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
    print("KSL v4 Evaluation - Face/Lips Features")
    print("=" * 70)

    # =========================================================================
    # Model Definitions (must match train_ksl_v4.py)
    # =========================================================================

    class BodyPartAttention(nn.Module):
        def __init__(self, hidden_dim: int):
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
            pose = x[:, :, :99]
            left_hand = x[:, :, 99:162]
            right_hand = x[:, :, 162:225]
            face = x[:, :, 225:429]
            lips = x[:, :, 429:549]

            pose_feat = self.pose_proj(pose)
            lh_feat = self.left_hand_proj(left_hand)
            rh_feat = self.right_hand_proj(right_hand)
            face_feat = self.face_proj(face)
            lips_feat = self.lips_proj(lips)

            combined = torch.cat([pose_feat, lh_feat, rh_feat, face_feat, lips_feat], dim=-1)
            output = self.out_proj(combined)
            return self.norm(output)

    class HybridModel(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttention(hidden_dim)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=CONFIG["dropout"], activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, num_classes),
            )
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.body_attention(x)
            x_conv = x.transpose(1, 2)
            x_conv = self.temporal_conv(x_conv)
            x = x_conv.transpose(1, 2)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            avg_pool = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]
            attn_weights = F.softmax(self.attn_pool(x), dim=1)
            attn_pool = (x * attn_weights).sum(dim=1)
            combined = torch.cat([avg_pool, max_pool, attn_pool], dim=1)
            return self.classifier(combined)

    class EnhancedTransformer(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttention(hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=CONFIG["dropout"], activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.body_attention(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.input_dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class EnhancedLSTM(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 549, hidden_dim: int = 256):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.body_attention = BodyPartAttention(hidden_dim)
            self.input_dropout = nn.Dropout(0.1)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=3, batch_first=True, bidirectional=True,
                dropout=CONFIG["dropout"],
            )
            self.temporal_attention = nn.Sequential(
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
            x = self.body_attention(x)
            x = self.input_dropout(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = F.softmax(self.temporal_attention(lstm_out), dim=1)
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

    # Load v4 models
    lstm = load_model("/data/checkpoints_v4/ksl_lstm_best.pth", EnhancedLSTM)
    if lstm:
        models.append(lstm)
        model_names.append("LSTM-v4")
        print("Loaded: LSTM-v4")

    hybrid = load_model("/data/checkpoints_v4/ksl_hybrid_best.pth", HybridModel)
    if hybrid:
        models.append(hybrid)
        model_names.append("Hybrid-v4")
        print("Loaded: Hybrid-v4")

    transformer = load_model("/data/checkpoints_v4/ksl_transformer_best.pth", EnhancedTransformer)
    if transformer:
        models.append(transformer)
        model_names.append("Transformer-v4")
        print("Loaded: Transformer-v4")

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
        data = (data - mean) / std
        if data.shape[0] > CONFIG["max_frames"]:
            indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
            data = data[indices]
        elif data.shape[0] < CONFIG["max_frames"]:
            padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        return torch.from_numpy(data).unsqueeze(0).cuda()

    # =========================================================================
    # Evaluate Each Model
    # =========================================================================

    results = {}

    for model, name in zip(models, model_names):
        correct = 0
        top3_correct = 0
        total = 0
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

        acc = correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        results[name] = {"accuracy": acc, "top3": top3_acc}

        print(f"\n{name}:")
        print(f"  Top-1: {acc*100:.2f}%")
        print(f"  Top-3: {top3_acc*100:.2f}%")

    # =========================================================================
    # Ensemble Evaluation
    # =========================================================================

    if len(models) > 1:
        print("\n" + "=" * 70)
        print("ENSEMBLE (All v4 models)")
        print("=" * 70)

        correct = 0
        top3_correct = 0
        total = 0
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

        acc = correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0

        print(f"Top-1 Accuracy: {acc*100:.2f}%")
        print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")

        print("\nPer-class accuracy:")
        for i, class_name in enumerate(ALL_CLASSES):
            if per_class_total[i] > 0:
                class_acc = per_class_correct[i] / per_class_total[i] * 100
                print(f"  {class_name:12s}: {class_acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")

        results["Ensemble"] = {"accuracy": acc, "top3": top3_acc}

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("KSL v4 Evaluation")
    print("=" * 70)

    results = evaluate_models.remote()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        print(f"{name}: Top-1={res['accuracy']*100:.2f}%, Top-3={res['top3']*100:.2f}%")
