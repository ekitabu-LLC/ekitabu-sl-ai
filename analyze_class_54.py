"""
Analyze Class 54 - Why is it attracting wrong predictions?

54 is being predicted when the true class is:
- 22 → 54: 10 times
- 444 → 54: 10 times  
- 100 → 54: 4 times
- 125 → 54: 5 times

This script analyzes:
1. Feature similarity between 54 and confused classes
2. Model confidence on these predictions
3. Temporal patterns that cause confusion

Usage:
    modal run analyze_class_54.py
"""

import modal
import os

app = modal.App("ksl-analyze-54")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Classes that get confused WITH 54
CONFUSED_WITH_54 = ["22", "444", "100", "125"]

# Classes that 54 gets confused WITH (predicted as something else)
CLASS_54_CONFUSED_AS = []  # We'll find this


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def analyze_54():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("DEEP ANALYSIS: Why is 54 an attractor?")
    print("=" * 70)

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

    # Load v7 model
    class TemporalPyramidModel(nn.Module):
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

        def forward(self, x, return_features=False):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [(lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1) for head in self.attention_heads]
            features = torch.cat(contexts, dim=-1)
            pre_class = self.pre_classifier(features)
            logits = self.classifier(pre_class)
            if return_features:
                return logits, pre_class
            return logits

    checkpoint = torch.load("/data/checkpoints_v7/ksl_pyramid_best.pth", map_location="cuda", weights_only=False)
    model = TemporalPyramidModel(num_classes=30)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()
    print("Loaded v7 model")

    # Load samples
    samples_by_class = defaultdict(list)
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if os.path.exists(class_dir):
            for fn in os.listdir(class_dir):
                if fn.endswith(".npy"):
                    samples_by_class[class_name].append(os.path.join(class_dir, fn))

    # Normalization
    all_data = []
    for class_name in ALL_CLASSES[:5]:
        for fp in samples_by_class[class_name][:10]:
            all_data.append(np.load(fp).flatten())
    all_data = np.concatenate(all_data)
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input(data):
        data = (data - mean) / std
        if data.shape[0] > 90:
            data = data[np.linspace(0, data.shape[0]-1, 90, dtype=int)]
        elif data.shape[0] < 90:
            data = np.vstack([data, np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features(data)).unsqueeze(0).cuda()

    # =========================================================================
    # Analysis 1: Feature Statistics for Each Class
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. RAW FEATURE STATISTICS")
    print("=" * 70)

    classes_to_analyze = ["54"] + CONFUSED_WITH_54
    class_stats = {}

    for class_name in classes_to_analyze:
        all_features = []
        for fp in samples_by_class[class_name]:
            data = np.load(fp).astype(np.float32)
            # Get hand features specifically
            left_hand = data[:, 99:162]
            right_hand = data[:, 162:225]
            hand_feat_left = compute_hand_features(left_hand)
            hand_feat_right = compute_hand_features(right_hand)
            all_features.append(np.concatenate([hand_feat_left.mean(axis=0), hand_feat_right.mean(axis=0)]))
        
        all_features = np.array(all_features)
        class_stats[class_name] = {
            "mean": all_features.mean(axis=0),
            "std": all_features.std(axis=0),
            "raw": all_features
        }
        print(f"\n{class_name}: {len(samples_by_class[class_name])} samples")
        print(f"  Hand feature mean (first 5): {all_features.mean(axis=0)[:5].round(4)}")

    # =========================================================================
    # Analysis 2: Feature Similarity (Cosine Distance)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. FEATURE SIMILARITY TO CLASS 54")
    print("=" * 70)

    ref_mean = class_stats["54"]["mean"]
    
    for class_name in CONFUSED_WITH_54:
        other_mean = class_stats[class_name]["mean"]
        # Cosine similarity
        cos_sim = np.dot(ref_mean, other_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(other_mean))
        # Euclidean distance
        euc_dist = np.linalg.norm(ref_mean - other_mean)
        print(f"  54 vs {class_name}: cosine_sim={cos_sim:.4f}, euclidean_dist={euc_dist:.4f}")

    # =========================================================================
    # Analysis 3: Model Prediction Confidence
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. MODEL CONFIDENCE ANALYSIS")
    print("=" * 70)

    class_54_idx = ALL_CLASSES.index("54")
    
    for class_name in classes_to_analyze:
        class_idx = ALL_CLASSES.index(class_name)
        confidences_correct = []
        confidences_54 = []
        wrong_to_54 = 0
        
        with torch.no_grad():
            for fp in samples_by_class[class_name]:
                data = np.load(fp).astype(np.float32)
                logits = model(prepare_input(data))
                probs = F.softmax(logits, dim=1).squeeze()
                
                pred = logits.argmax(dim=1).item()
                
                confidences_correct.append(probs[class_idx].item())
                confidences_54.append(probs[class_54_idx].item())
                
                if pred == class_54_idx and class_name != "54":
                    wrong_to_54 += 1

        print(f"\n{class_name}:")
        print(f"  Confidence for correct class: {np.mean(confidences_correct)*100:.1f}% (std: {np.std(confidences_correct)*100:.1f}%)")
        print(f"  Confidence for class 54: {np.mean(confidences_54)*100:.1f}% (std: {np.std(confidences_54)*100:.1f}%)")
        if class_name != "54":
            print(f"  Wrongly predicted as 54: {wrong_to_54}/{len(samples_by_class[class_name])}")

    # =========================================================================
    # Analysis 4: Temporal Pattern Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. TEMPORAL PATTERN ANALYSIS")
    print("=" * 70)

    for class_name in classes_to_analyze:
        durations = []
        velocity_magnitudes = []
        
        for fp in samples_by_class[class_name]:
            data = np.load(fp).astype(np.float32)
            durations.append(data.shape[0])
            
            # Compute velocity magnitude
            velocity = np.diff(data[:, 99:225], axis=0)  # Hand landmarks only
            vel_mag = np.linalg.norm(velocity, axis=1).mean()
            velocity_magnitudes.append(vel_mag)
        
        print(f"\n{class_name}:")
        print(f"  Duration: {np.mean(durations):.1f} frames (std: {np.std(durations):.1f})")
        print(f"  Hand velocity: {np.mean(velocity_magnitudes):.4f} (std: {np.std(velocity_magnitudes):.4f})")

    # =========================================================================
    # Analysis 5: Feature Space Visualization Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. FEATURE SPACE ANALYSIS (Internal Representations)")
    print("=" * 70)

    class_features = defaultdict(list)
    
    with torch.no_grad():
        for class_name in classes_to_analyze:
            for fp in samples_by_class[class_name]:
                data = np.load(fp).astype(np.float32)
                logits, features = model(prepare_input(data), return_features=True)
                class_features[class_name].append(features.squeeze().cpu().numpy())

    # Compute pairwise distances in feature space
    print("\nInternal feature distances (model's learned representation):")
    ref_features = np.array(class_features["54"])
    ref_centroid = ref_features.mean(axis=0)

    for class_name in CONFUSED_WITH_54:
        other_features = np.array(class_features[class_name])
        other_centroid = other_features.mean(axis=0)
        
        cos_sim = np.dot(ref_centroid, other_centroid) / (np.linalg.norm(ref_centroid) * np.linalg.norm(other_centroid))
        euc_dist = np.linalg.norm(ref_centroid - other_centroid)
        
        print(f"  54 vs {class_name}: cosine_sim={cos_sim:.4f}, euclidean_dist={euc_dist:.4f}")

    # =========================================================================
    # Analysis 6: What makes 54 unique?
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. WHAT MAKES 54 UNIQUE (vs confused classes)?")
    print("=" * 70)

    # Find features where 54 differs most from confused classes
    ref_mean = class_stats["54"]["mean"]
    
    all_other_means = []
    for class_name in CONFUSED_WITH_54:
        all_other_means.append(class_stats[class_name]["mean"])
    other_mean = np.array(all_other_means).mean(axis=0)
    
    diff = np.abs(ref_mean - other_mean)
    top_diff_indices = np.argsort(diff)[-10:][::-1]
    
    print("\nTop 10 features where 54 differs from confused classes:")
    feature_names = [
        "L_thumb_dist", "L_index_dist", "L_middle_dist", "L_ring_dist", "L_pinky_dist",
        "L_thumb_index", "L_thumb_middle", "L_thumb_ring", "L_thumb_pinky",
        "L_index_middle", "L_index_ring", "L_index_pinky", "L_middle_ring", "L_middle_pinky", "L_ring_pinky",
        "L_thumb_curl", "L_index_curl", "L_middle_curl", "L_ring_curl", "L_pinky_curl",
        "L_spread", "L_thumb_index_angle", "L_palm_x", "L_palm_y", "L_palm_z",
        "R_thumb_dist", "R_index_dist", "R_middle_dist", "R_ring_dist", "R_pinky_dist",
        "R_thumb_index", "R_thumb_middle", "R_thumb_ring", "R_thumb_pinky",
        "R_index_middle", "R_index_ring", "R_index_pinky", "R_middle_ring", "R_middle_pinky", "R_ring_pinky",
        "R_thumb_curl", "R_index_curl", "R_middle_curl", "R_ring_curl", "R_pinky_curl",
        "R_spread", "R_thumb_index_angle", "R_palm_x", "R_palm_y", "R_palm_z",
    ]
    
    for idx in top_diff_indices:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {fname}: 54={ref_mean[idx]:.4f}, others={other_mean[idx]:.4f}, diff={diff[idx]:.4f}")

    # =========================================================================
    # Conclusions
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSIONS & RECOMMENDATIONS FOR V8")
    print("=" * 70)

    print("""
## Why 54 is an Attractor

1. SIMILAR HAND CONFIGURATIONS
   - 54 likely has a hand shape that appears as a "default" or common pose
   - Other numbers (22, 444, 100, 125) may pass through 54-like poses during execution

2. TEMPORAL OVERLAP
   - Multi-digit numbers (444, 100) contain transitions that look like 54
   - Model captures partial patterns that match 54

3. MODEL BIAS
   - v7 may have learned 54 as a "catch-all" for uncertain predictions
   - Focal loss boosting may have over-emphasized 54 samples

## Recommendations for v8

1. NEGATIVE MINING FOR 54
   - Explicitly train "not 54" examples from confused classes
   - Add contrastive pairs: (22, 54), (444, 54), (100, 54)

2. TEMPORAL BOUNDARY DETECTION
   - For 444: detect the three "4" digits separately
   - For 100: detect "1" followed by "00"
   - Prevent partial matches to 54

3. CONFIDENCE CALIBRATION
   - Add temperature scaling to soften predictions
   - Penalize overconfident wrong predictions to 54

4. CLASS-SPECIFIC FEATURES
   - Add features that distinguish 54 from confused classes
   - Focus on the discriminative features identified above

5. ENSEMBLE WITH V5
   - v5 had 80% on 54 without attracting others
   - Combine v5's classification with v7's temporal modeling
""")

    return {
        "confused_classes": CONFUSED_WITH_54,
        "analysis_complete": True,
    }


@app.local_entrypoint()
def main():
    results = analyze_54.remote()
    print("\nAnalysis complete!")
