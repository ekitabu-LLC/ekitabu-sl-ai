"""
KSL Model Evaluation Script
Evaluates trained models and generates confusion matrix and detailed metrics.

Usage:
    modal run evaluate.py --model transformer
    modal run evaluate.py --model lstm
"""

import modal
import os

# Modal app setup
app = modal.App("ksl-evaluator")
volume = modal.Volume.from_name("ksl-dataset-vol")

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    )
)

# All 30 classes
ALL_CLASSES = [
    "100", "125", "17", "22", "268", "35", "388", "444", "48", "54", "66", "73", "89", "9", "91",
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"
]

CONFIG = {
    "max_frames": 120,
    "feature_dim": 225,
    "batch_size": 32,
    "checkpoint_dir": "/data/checkpoints",
    "results_dir": "/data/results",
}


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def evaluate_model(model_type: str) -> dict:
    """
    Evaluate a trained KSL model on validation data.
    Generates confusion matrix and detailed metrics.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json

    print("=" * 70)
    print(f"KSL Model Evaluation - {model_type.upper()}")
    print("=" * 70)

    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # =========================================================================
    # Model Definitions (same as training)
    # =========================================================================

    class TransformerModel(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, CONFIG["max_frames"], hidden_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.3,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            x = self.feature_proj(x)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class LSTMModel(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int = 225, hidden_dim: int = 256):
            super().__init__()
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, num_classes),
            )

        def forward(self, x):
            x = self.feature_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    # =========================================================================
    # Dataset
    # =========================================================================

    class KSLDataset(Dataset):
        def __init__(self, data_dir: str, classes: list):
            self.samples = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue

                for filename in os.listdir(class_dir):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, self.class_to_idx[class_name]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            data = np.load(filepath).astype(np.float32)

            if data.shape[0] > CONFIG["max_frames"]:
                indices = np.linspace(0, data.shape[0] - 1, CONFIG["max_frames"], dtype=int)
                data = data[indices]
            elif data.shape[0] < CONFIG["max_frames"]:
                padding = np.zeros((CONFIG["max_frames"] - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])

            return torch.from_numpy(data), label

    # =========================================================================
    # Load Model
    # =========================================================================

    checkpoint_path = f"{CONFIG['checkpoint_dir']}/ksl_{model_type}_best.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    if model_type == "transformer":
        model = TransformerModel(num_classes=len(ALL_CLASSES))
    else:
        model = LSTMModel(num_classes=len(ALL_CLASSES))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()

    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Reported validation accuracy: {checkpoint['val_acc'] * 100:.2f}%")

    # =========================================================================
    # Load Validation Data
    # =========================================================================

    val_dataset = KSLDataset("/data/val", ALL_CLASSES)
    print(f"\nValidation samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # =========================================================================
    # Evaluation
    # =========================================================================

    all_preds = []
    all_labels = []
    all_probs = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.cuda()
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # =========================================================================
    # Compute Metrics
    # =========================================================================

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Top-3 accuracy
    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if label in top3_indices:
            top3_correct += 1
    top3_accuracy = top3_correct / len(all_labels)

    # Top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(all_labels):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if label in top5_indices:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:   {top3_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy:   {top5_accuracy * 100:.2f}%")

    # Classification report
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    report = classification_report(all_labels, all_preds, target_names=ALL_CLASSES, digits=3)
    print(report)

    # Save classification report
    report_dict = classification_report(all_labels, all_preds, target_names=ALL_CLASSES, output_dict=True)

    # =========================================================================
    # Confusion Matrix
    # =========================================================================

    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ALL_CLASSES,
        yticklabels=ALL_CLASSES,
    )
    plt.title(f"KSL {model_type.upper()} Model - Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    cm_path = f"{CONFIG['results_dir']}/confusion_matrix_{model_type}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")

    # =========================================================================
    # Per-Class Analysis
    # =========================================================================

    print("\n" + "-" * 70)
    print("Per-Class Accuracy:")
    print("-" * 70)

    # Separate numbers and words
    number_classes = ALL_CLASSES[:15]
    word_classes = ALL_CLASSES[15:]

    print("\nNUMBERS:")
    num_correct = 0
    num_total = 0
    for i, class_name in enumerate(number_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            print(f"  {class_name:6s}: {class_acc * 100:5.1f}% ({(all_preds[mask] == i).sum()}/{mask.sum()})")
            num_correct += (all_preds[mask] == i).sum()
            num_total += mask.sum()

    print(f"\n  Numbers Overall: {num_correct / num_total * 100:.1f}%" if num_total > 0 else "")

    print("\nWORDS:")
    word_correct = 0
    word_total = 0
    for i, class_name in enumerate(word_classes):
        idx = i + 15
        mask = all_labels == idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == idx).mean()
            print(f"  {class_name:10s}: {class_acc * 100:5.1f}% ({(all_preds[mask] == idx).sum()}/{mask.sum()})")
            word_correct += (all_preds[mask] == idx).sum()
            word_total += mask.sum()

    print(f"\n  Words Overall: {word_correct / word_total * 100:.1f}%" if word_total > 0 else "")

    # =========================================================================
    # Save Results
    # =========================================================================

    results = {
        "model_type": model_type,
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": ALL_CLASSES,
    }

    results_path = f"{CONFIG['results_dir']}/evaluation_{model_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Commit volume
    volume.commit()

    return results


@app.local_entrypoint()
def main(model: str = "transformer"):
    """
    Main entry point for evaluation.

    Args:
        model: "transformer", "lstm", or "both"
    """
    print("=" * 70)
    print("KSL Model Evaluation")
    print("=" * 70)

    if model == "both":
        results = list(evaluate_model.map(["transformer", "lstm"]))

        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        for r in results:
            print(f"\n{r['model_type'].upper()}:")
            print(f"  Accuracy:     {r['accuracy'] * 100:.2f}%")
            print(f"  Top-3 Acc:    {r['top3_accuracy'] * 100:.2f}%")
            print(f"  Top-5 Acc:    {r['top5_accuracy'] * 100:.2f}%")
    else:
        result = evaluate_model.remote(model)
        print(f"\n{model.upper()} Accuracy: {result['accuracy'] * 100:.2f}%")

    print("\nTo download results:")
    print("  modal volume get ksl-dataset-vol /data/results/ ./results/")
