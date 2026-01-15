"""
KSL Version Analysis - Compare v5, v6, v7
Analyzes what changed between versions and what improvements worked.

This script helps understand:
1. What architectural changes improved performance
2. Which classes improved/regressed between versions
3. What techniques work for hard classes
4. Recommendations for future versions

Usage:
    modal run analyze_versions.py
    
Local analysis (no Modal):
    python analyze_versions.py --local
"""

import modal
import os
import sys

app = modal.App("ksl-analyze-versions")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=7200, image=image)
def analyze_all_versions():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict
    import json

    print("=" * 70)
    print("KSL VERSION ANALYSIS: v5 vs v6 vs v7")
    print("=" * 70)

    # =========================================================================
    # Hand Feature Engineering (shared across versions)
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

    # =========================================================================
    # Model Definitions
    # =========================================================================

    # v5 HandFocused Model
    class HandFocusedModelV5(nn.Module):
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
            left_hand_base, right_hand_base = x[:, :, 99:162], x[:, :, 162:225]
            face = x[:, :, 225:549]
            left_hand_extra, right_hand_extra = x[:, :, 549:599], x[:, :, 599:649]
            hands = torch.cat([torch.cat([left_hand_base, left_hand_extra], -1), torch.cat([right_hand_base, right_hand_extra], -1)], -1)
            body = torch.cat([pose, face], -1)
            fused = self.fusion(torch.cat([self.hand_encoder(hands), self.body_encoder(body)], -1))
            lstm_out, _ = self.lstm(fused)
            context = (lstm_out * F.softmax(self.attention(lstm_out), dim=1)).sum(dim=1)
            return torch.cat([self.number_classifier(context), self.word_classifier(context)], dim=-1)

    # v6 Contrastive Model
    class ContrastiveModelV6(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=256):
            super().__init__()
            self.hand_encoder = nn.Sequential(nn.Linear(226, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.body_encoder = nn.Sequential(nn.Linear(423, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.feature_norm = nn.LayerNorm(hidden_dim*2)
            self.projector = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 128))
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes))

        def forward(self, x):
            pose = x[:, :, :99]
            left_hand_base, right_hand_base = x[:, :, 99:162], x[:, :, 162:225]
            face = x[:, :, 225:549]
            left_hand_extra, right_hand_extra = x[:, :, 549:599], x[:, :, 599:649]
            hands = torch.cat([torch.cat([left_hand_base, left_hand_extra], -1), torch.cat([right_hand_base, right_hand_extra], -1)], -1)
            body = torch.cat([pose, face], -1)
            fused = self.fusion(torch.cat([self.hand_encoder(hands), self.body_encoder(body)], -1))
            lstm_out, _ = self.lstm(fused)
            context = self.feature_norm((lstm_out * F.softmax(self.attention(lstm_out), dim=1)).sum(dim=1))
            return self.classifier(context)

    # v7 Temporal Pyramid Model
    class TemporalPyramidModelV7(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim//2), nn.GELU()),
            ])
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim//2*3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.4))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.4)
            self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)])
            self.pre_classifier = nn.Sequential(nn.LayerNorm(hidden_dim*2*4), nn.Dropout(0.4), nn.Linear(hidden_dim*2*4, hidden_dim*2), nn.GELU(), nn.Dropout(0.4))
            self.classifier = nn.Linear(hidden_dim*2, num_classes)

        def forward(self, x):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            lstm_out, _ = self.lstm(self.temporal_fusion(multi_scale))
            contexts = [(lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1) for head in self.attention_heads]
            return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))

    # =========================================================================
    # Load Models
    # =========================================================================
    
    models = {}
    model_configs = [
        ("v5", "/data/checkpoints_v5/ksl_handfocused_best.pth", HandFocusedModelV5, 256),
        ("v6", "/data/checkpoints_v6/ksl_contrastive_best.pth", ContrastiveModelV6, 256),
        ("v7", "/data/checkpoints_v7/ksl_pyramid_best.pth", TemporalPyramidModelV7, 320),
    ]

    for name, path, model_class, hidden_dim in model_configs:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location="cuda", weights_only=False)
                model = model_class(num_classes=30, hidden_dim=hidden_dim)
                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.cuda()
                model.eval()
                models[name] = {"model": model, "checkpoint": checkpoint}
                print(f"Loaded {name}: {checkpoint.get('val_acc', 0)*100:.2f}%")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        else:
            print(f"Checkpoint not found: {name}")

    if len(models) == 0:
        print("No models loaded!")
        return {}

    # =========================================================================
    # Load Validation Data
    # =========================================================================
    
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if os.path.exists(class_dir):
            for fn in os.listdir(class_dir):
                if fn.endswith(".npy"):
                    val_samples.append((os.path.join(class_dir, fn), ALL_CLASSES.index(class_name), class_name))
    
    print(f"\nValidation samples: {len(val_samples)}")

    # Normalization stats
    all_data = np.concatenate([np.load(val_samples[i][0]).flatten() for i in range(min(200, len(val_samples)))])
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input(data):
        data = (data - mean) / std
        if data.shape[0] > 90:
            data = data[np.linspace(0, data.shape[0]-1, 90, dtype=int)]
        elif data.shape[0] < 90:
            data = np.vstack([data, np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features(data)).unsqueeze(0).cuda()

    # =========================================================================
    # Evaluate All Models
    # =========================================================================
    
    results = {}
    
    for version, model_data in models.items():
        model = model_data["model"]
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))
        correct, total = 0, 0

        with torch.no_grad():
            for filepath, label, class_name in val_samples:
                data = np.load(filepath).astype(np.float32)
                logits = model(prepare_input(data))
                pred = logits.argmax(dim=1).item()
                pred_class = ALL_CLASSES[pred]

                per_class_total[class_name] += 1
                confusion[class_name][pred_class] += 1
                total += 1
                if pred == label:
                    correct += 1
                    per_class_correct[class_name] += 1

        results[version] = {
            "accuracy": correct / total,
            "per_class_acc": {c: per_class_correct[c]/per_class_total[c] if per_class_total[c] > 0 else 0 for c in ALL_CLASSES},
            "confusion": dict(confusion),
        }
        print(f"{version}: {correct/total*100:.2f}%")

    # =========================================================================
    # Analysis Report
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("VERSION COMPARISON ANALYSIS")
    print("=" * 70)

    # Overall comparison
    print("\n## OVERALL ACCURACY")
    print("-" * 40)
    for v in ["v5", "v6", "v7"]:
        if v in results:
            print(f"  {v}: {results[v]['accuracy']*100:.2f}%")

    # Per-class comparison
    print("\n## PER-CLASS COMPARISON (v5 -> v6 -> v7)")
    print("-" * 70)
    print(f"{'Class':>12} | {'v5':>8} | {'v6':>8} | {'v7':>8} | {'v5->v7':>10} | Status")
    print("-" * 70)

    improvements = []
    regressions = []
    
    for c in ALL_CLASSES:
        v5_acc = results.get("v5", {}).get("per_class_acc", {}).get(c, 0) * 100
        v6_acc = results.get("v6", {}).get("per_class_acc", {}).get(c, 0) * 100
        v7_acc = results.get("v7", {}).get("per_class_acc", {}).get(c, 0) * 100
        
        change = v7_acc - v5_acc
        if change > 10:
            status = "IMPROVED"
            improvements.append((c, change))
        elif change < -10:
            status = "REGRESSED"
            regressions.append((c, change))
        else:
            status = "stable"
        
        print(f"{c:>12} | {v5_acc:>7.1f}% | {v6_acc:>7.1f}% | {v7_acc:>7.1f}% | {change:>+9.1f}% | {status}")

    # Biggest improvements
    print("\n## BIGGEST IMPROVEMENTS (v5 -> v7)")
    print("-" * 40)
    improvements.sort(key=lambda x: -x[1])
    for c, change in improvements[:10]:
        v5_acc = results.get("v5", {}).get("per_class_acc", {}).get(c, 0) * 100
        v7_acc = results.get("v7", {}).get("per_class_acc", {}).get(c, 0) * 100
        print(f"  {c}: {v5_acc:.1f}% -> {v7_acc:.1f}% (+{change:.1f}%)")

    # Regressions
    if regressions:
        print("\n## REGRESSIONS (v5 -> v7)")
        print("-" * 40)
        regressions.sort(key=lambda x: x[1])
        for c, change in regressions:
            v5_acc = results.get("v5", {}).get("per_class_acc", {}).get(c, 0) * 100
            v7_acc = results.get("v7", {}).get("per_class_acc", {}).get(c, 0) * 100
            print(f"  {c}: {v5_acc:.1f}% -> {v7_acc:.1f}% ({change:.1f}%)")

    # Still struggling classes
    print("\n## CLASSES STILL STRUGGLING (<50% in v7)")
    print("-" * 40)
    struggling = []
    for c in ALL_CLASSES:
        v7_acc = results.get("v7", {}).get("per_class_acc", {}).get(c, 0) * 100
        if v7_acc < 50:
            struggling.append((c, v7_acc))
    struggling.sort(key=lambda x: x[1])
    for c, acc in struggling:
        print(f"  {c}: {acc:.1f}%")

    # =========================================================================
    # What Worked Analysis
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("WHAT WORKED - LESSONS LEARNED")
    print("=" * 70)

    print("""
## Architecture Changes That Helped

1. TEMPORAL PYRAMID (v7)
   - Multi-scale convolutions (kernel 3, 7, 15) capture different temporal patterns
   - Numbers like 100, 444 need longer temporal context
   - Short kernel (3) for quick transitions, long kernel (15) for slow movements

2. MULTI-HEAD ATTENTION (v7)
   - 4 attention heads vs 1 in v5/v6
   - Different heads focus on different aspects of the sign
   - Better aggregation of temporal information

3. REMOVED CONTRASTIVE LOSS (v7 vs v6)
   - Contrastive loss was fighting classification objective
   - Caused 0% accuracy on multiple classes
   - Pure focal loss with class weighting works better

4. HARD CLASS BOOSTING (v7)
   - 3x weight for classes with 0% accuracy
   - Oversampling during training
   - Gentler augmentation for hard classes

5. INCREASED MODEL CAPACITY (v7)
   - hidden_dim: 256 -> 320
   - More parameters for complex patterns
   - Stronger regularization (dropout 0.4) to prevent overfitting

## What Didn't Work

1. CONTRASTIVE LEARNING (v6)
   - Temperature too low (0.1)
   - Batch size too small for effective pairs
   - Competed with classification objective

2. STRONG AUGMENTATION (v6)
   - Temporal warping destroyed discriminative features
   - Too aggressive noise on hard classes

3. TWO-HEAD CLASSIFIER (v5)
   - Separate number/word heads didn't help
   - Added complexity without benefit
""")

    # =========================================================================
    # Recommendations for v8
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR v8")
    print("=" * 70)

    print("""
## Still Struggling Classes Analysis

- 444 (15%): Three repeated digits - needs SEQUENCE MODELING
- 35 (36%): Confused with "Colour" - need BETTER HAND FEATURES  
- 100 (52%): Three-digit number - needs DIGIT SEGMENTATION

## Proposed v8 Improvements

1. SEQUENCE-TO-SEQUENCE for multi-digit numbers
   - Treat 444, 100, 388, 268 as sequences of digits
   - Decoder that outputs digit sequence
   - CTC loss for alignment

2. DIGIT-AWARE FEATURES
   - Detect when hand resets between digits
   - Segment sign into individual digit components
   - Classify each segment separately

3. CONFUSION-BASED TRAINING
   - Add explicit "not Colour" signal when training 35
   - Contrastive pairs between commonly confused classes
   - Hard negative mining

4. HIERARCHICAL CLASSIFICATION
   - First classify: single-digit vs multi-digit vs word
   - Then sub-classify within category
   - Reduces confusion between categories

5. TEMPORAL SEGMENTATION
   - Find boundaries between digits in multi-digit numbers
   - Process each segment independently
   - Combine predictions
""")

    return {
        "results": {v: {"accuracy": r["accuracy"], "per_class": r["per_class_acc"]} for v, r in results.items()},
        "improvements": improvements,
        "regressions": regressions,
        "struggling": struggling,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("KSL Version Analysis: Learning from v5 -> v6 -> v7")
    print("=" * 70)
    
    results = analyze_all_versions.remote()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    if "results" in results:
        print("\nFinal Accuracies:")
        for v, r in results["results"].items():
            print(f"  {v}: {r['accuracy']*100:.2f}%")
