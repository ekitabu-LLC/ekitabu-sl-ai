"""
KSL Ensemble Inference Script
Combines predictions from multiple trained models for improved accuracy.

Usage:
    modal run ensemble_inference.py --video path/to/video.mp4
    modal run ensemble_inference.py --evaluate  # Evaluate on validation set
"""

import modal
import os

app = modal.App("ksl-ensemble-inference")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("torch", "numpy", "mediapipe==0.10.14", "opencv-python-headless", "scikit-learn")
)

ALL_CLASSES = [
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
def ensemble_evaluate() -> dict:
    """Evaluate ensemble on validation set."""
    import torch
    import torch.nn as nn
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL Ensemble Evaluation")
    print("=" * 70)

    # =========================================================================
    # Model Definitions (must match training scripts)
    # =========================================================================

    class TransformerModelV1(nn.Module):
        """Original Transformer from v1."""
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256, max_frames: int = 120):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_frames, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=512,
                dropout=0.3, activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModelV1(nn.Module):
        """Original LSTM from v1."""
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=3, batch_first=True, bidirectional=True, dropout=0.3,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2), nn.Dropout(0.3), nn.Linear(hidden_dim * 2, num_classes),
            )

        def forward(self, x):
            x = self.feature_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    class TransformerModelV2(nn.Module):
        """Smaller Transformer from v2."""
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 128, max_frames: int = 90):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_frames, hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.3)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=256,
                dropout=0.5, activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Dropout(0.5), nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.input_dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModelV2(nn.Module):
        """Smaller LSTM from v2."""
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 128):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.input_dropout = nn.Dropout(0.3)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.5,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
                nn.Dropout(0.3), nn.Linear(hidden_dim, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2), nn.Dropout(0.5),
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.feature_proj(x)
            x = self.input_dropout(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    # V3 models with velocity features (801 input features: 225 + 225 + 225 + 126)
    class TransformerModelV3(nn.Module):
        """V3 Transformer with velocity/acceleration/relative features."""
        def __init__(self, num_classes: int, feature_dim: int = 801, hidden_dim: int = 192, max_frames: int = 90):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pos_encoding = nn.Parameter(torch.randn(1, max_frames, hidden_dim) * 0.02)
            self.input_dropout = nn.Dropout(0.2)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=6, dim_feedforward=384,
                dropout=0.4, activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.input_dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModelV3(nn.Module):
        """V3 LSTM with velocity/acceleration/relative features."""
        def __init__(self, num_classes: int, feature_dim: int = 801, hidden_dim: int = 192):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_dropout = nn.Dropout(0.2)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.4,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.feature_proj(x)
            x = self.input_dropout(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def load_model(checkpoint_path, model_class, **kwargs):
        """Load a model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            return None
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model = model_class(num_classes=len(ALL_CLASSES), **kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda().eval()
        return model

    def prepare_input_v1_v2(data, max_frames):
        """Prepare input for v1/v2 models (225 features)."""
        if data.shape[0] > max_frames:
            indices = np.linspace(0, data.shape[0] - 1, max_frames, dtype=int)
            data = data[indices]
        elif data.shape[0] < max_frames:
            padding = np.zeros((max_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        return torch.from_numpy(data).unsqueeze(0).cuda()

    def compute_relative_positions(data):
        """Compute hand positions relative to shoulders."""
        relative = []
        for frame in data:
            frame_rel = []
            # Left shoulder position (landmark 11)
            left_shoulder = frame[33:36] if len(frame) > 35 else np.zeros(3)
            # Right shoulder position (landmark 12)
            right_shoulder = frame[36:39] if len(frame) > 38 else np.zeros(3)
            # Left hand relative to left shoulder (21 landmarks * 3 = 63)
            left_hand = frame[99:162].reshape(-1, 3) if len(frame) > 161 else np.zeros((21, 3))
            left_hand_rel = (left_hand - left_shoulder).flatten()
            frame_rel.extend(left_hand_rel)
            # Right hand relative to right shoulder (21 landmarks * 3 = 63)
            right_hand = frame[162:225].reshape(-1, 3) if len(frame) > 224 else np.zeros((21, 3))
            right_hand_rel = (right_hand - right_shoulder).flatten()
            frame_rel.extend(right_hand_rel)
            relative.append(frame_rel)
        return np.array(relative, dtype=np.float32)

    def prepare_input_v3(data, max_frames=90):
        """Prepare input for v3 models (801 features: landmarks + velocity + acceleration + relative)."""
        # Compute velocity
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        # Compute acceleration
        acceleration = np.zeros_like(data)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        # Compute relative positions
        relative = compute_relative_positions(data)
        # Concatenate all features: 225 + 225 + 225 + 126 = 801
        data_with_features = np.concatenate([data, velocity, acceleration, relative], axis=1)

        if data_with_features.shape[0] > max_frames:
            indices = np.linspace(0, data_with_features.shape[0] - 1, max_frames, dtype=int)
            data_with_features = data_with_features[indices]
        elif data_with_features.shape[0] < max_frames:
            padding = np.zeros((max_frames - data_with_features.shape[0], data_with_features.shape[1]), dtype=np.float32)
            data_with_features = np.vstack([data_with_features, padding])
        return torch.from_numpy(data_with_features).unsqueeze(0).cuda()

    def tta_augment(data, augment_type):
        """Apply test-time augmentation."""
        if augment_type == "original":
            return data
        elif augment_type == "noise":
            return data + np.random.normal(0, 0.01, data.shape).astype(np.float32)
        elif augment_type == "scale_up":
            return data * 1.05
        elif augment_type == "scale_down":
            return data * 0.95
        elif augment_type == "mirror":
            # Swap left and right hands
            mirrored = data.copy()
            left_hand = mirrored[:, 99:162].copy()
            right_hand = mirrored[:, 162:225].copy()
            mirrored[:, 99:162] = right_hand
            mirrored[:, 162:225] = left_hand
            return mirrored
        return data

    # =========================================================================
    # Load Models
    # =========================================================================

    models = []
    model_configs = [
        # (checkpoint_path, model_class, kwargs, prepare_func, weight)
        ("/data/checkpoints/ksl_transformer_best.pth", TransformerModelV1, {"max_frames": 120}, lambda d: prepare_input_v1_v2(d, 120), 1.0),
        ("/data/checkpoints/ksl_lstm_best.pth", LSTMModelV1, {}, lambda d: prepare_input_v1_v2(d, 120), 1.0),
        ("/data/checkpoints_v2/ksl_transformer_best.pth", TransformerModelV2, {"max_frames": 90}, lambda d: prepare_input_v1_v2(d, 90), 1.0),
        ("/data/checkpoints_v2/ksl_lstm_best.pth", LSTMModelV2, {}, lambda d: prepare_input_v1_v2(d, 90), 1.0),
        ("/data/checkpoints_v3/ksl_transformer_best.pth", TransformerModelV3, {"feature_dim": 801, "max_frames": 90}, prepare_input_v3, 1.5),  # Higher weight for v3
        ("/data/checkpoints_v3/ksl_lstm_best.pth", LSTMModelV3, {"feature_dim": 801}, prepare_input_v3, 1.5),
    ]

    print("\nLoading models...")
    total_weight = 0.0
    for checkpoint_path, model_class, kwargs, prepare_func, weight in model_configs:
        model = load_model(checkpoint_path, model_class, **kwargs)
        if model is not None:
            models.append((model, prepare_func, weight))
            total_weight += weight
            print(f"  Loaded: {checkpoint_path} (weight={weight})")
        else:
            print(f"  Not found: {checkpoint_path}")

    if not models:
        raise ValueError("No models found!")

    print(f"\nTotal models loaded: {len(models)}, Total weight: {total_weight}")

    # =========================================================================
    # Load Validation Data
    # =========================================================================

    print("\nLoading validation data...")
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val/{class_name}"
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy"):
                filepath = os.path.join(class_dir, filename)
                val_samples.append((filepath, ALL_CLASSES.index(class_name)))

    print(f"  Validation samples: {len(val_samples)}")

    # =========================================================================
    # Evaluate Ensemble
    # =========================================================================

    print("\nEvaluating ensemble...")
    tta_modes = ["original", "noise", "scale_up", "scale_down", "mirror"]

    correct = 0
    top3_correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for filepath, label in val_samples:
        data = np.load(filepath).astype(np.float32)

        # Ensemble prediction with TTA
        ensemble_logits = None

        for model, prepare_func, weight in models:
            model_logits = None
            for tta_mode in tta_modes:
                augmented_data = tta_augment(data.copy(), tta_mode)
                input_tensor = prepare_func(augmented_data)

                with torch.no_grad():
                    logits = model(input_tensor)

                if model_logits is None:
                    model_logits = logits
                else:
                    model_logits = model_logits + logits

            # Average over TTA modes
            model_logits = model_logits / len(tta_modes)

            # Add weighted contribution to ensemble
            if ensemble_logits is None:
                ensemble_logits = model_logits * weight
            else:
                ensemble_logits = ensemble_logits + model_logits * weight

        # Normalize by total weight
        ensemble_logits = ensemble_logits / total_weight

        # Get prediction
        pred = ensemble_logits.argmax(dim=1).item()
        _, top3_preds = ensemble_logits.topk(3, dim=1)
        top3_preds = top3_preds.squeeze().cpu().numpy()

        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        if label in top3_preds:
            top3_correct += 1
        per_class_total[label] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(ALL_CLASSES):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i] * 100
            print(f"  {class_name:12s}: {acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "total_samples": total,
        "num_models": len(models),
    }


@app.local_entrypoint()
def main(evaluate: bool = True):
    """Main entry point."""
    print("=" * 70)
    print("KSL Ensemble Inference")
    print("=" * 70)

    if evaluate:
        result = ensemble_evaluate.remote()
        print(f"\nFinal Results:")
        print(f"  Top-1 Accuracy: {result['accuracy'] * 100:.2f}%")
        print(f"  Top-3 Accuracy: {result['top3_accuracy'] * 100:.2f}%")
        print(f"  Models in ensemble: {result['num_models']}")
