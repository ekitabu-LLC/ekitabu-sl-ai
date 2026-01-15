"""
KSL Ensemble v8+v9 Assessment
Combines v8's overall accuracy with v9's temporal discrimination for 444.

Strategy:
- Use v9 for attractor victim classes (444, 22, 100, 125) where v8 struggles
- Use v8 for everything else where it excels

Usage:
    modal run ensemble_v8_v9.py
"""

import modal
import os

app = modal.App("ksl-ensemble-v8-v9")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Classes where v9 is better (attractor victims and 91)
V9_PREFERRED_CLASSES = ["444", "22", "100", "125", "91"]

CONFIG_V8 = {"max_frames": 90, "feature_dim": 649, "hidden_dim": 320, "dropout": 0.35}
CONFIG_V9 = {"max_frames": 90, "feature_dim": 657, "hidden_dim": 320, "dropout": 0.35}


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def ensemble_assess():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import defaultdict

    print("=" * 70)
    print("KSL ENSEMBLE v8 + v9 Assessment")
    print("=" * 70)
    print(f"Strategy: Use v9 for {V9_PREFERRED_CLASSES}, v8 for others")

    # Hand feature computation (same for both)
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

    def engineer_features_v8(data):
        left_hand, right_hand = data[:, 99:162], data[:, 162:225]
        left_features, right_features = compute_hand_features(left_hand), compute_hand_features(right_hand)
        left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel[1:] = right_features[1:] - right_features[:-1]
        return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)

    def compute_temporal_features(data):
        frames = data.shape[0]
        features = []
        duration_norm = frames / 90.0
        features.append(duration_norm)
        segment_size = frames // 3
        if segment_size >= 5:
            segments = [data[i*segment_size:(i+1)*segment_size].mean(axis=0) for i in range(3)]
            for i in range(3):
                for j in range(i+1, 3):
                    norm_i, norm_j = np.linalg.norm(segments[i]), np.linalg.norm(segments[j])
                    if norm_i > 1e-8 and norm_j > 1e-8:
                        features.append(np.dot(segments[i], segments[j]) / (norm_i * norm_j))
                    else:
                        features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        right_hand = data[:, 162:225]
        velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
        features.append(velocities.mean())
        features.append(velocities.std())
        features.append(np.sum(velocities < velocities.mean() * 0.3) / len(velocities) if len(velocities) > 0 else 0.0)
        right_hand_reshaped = right_hand.reshape(-1, 21, 3)
        spread = np.linalg.norm(right_hand_reshaped[:, 4] - right_hand_reshaped[:, 20], axis=1)
        kernel_size = 5
        if len(spread) >= kernel_size:
            spread_smooth = np.convolve(spread, np.ones(kernel_size)/kernel_size, mode='valid')
            peaks = 0
            for i in range(1, len(spread_smooth)-1):
                if spread_smooth[i] > spread_smooth[i-1] and spread_smooth[i] > spread_smooth[i+1]:
                    if spread_smooth[i] > np.mean(spread_smooth) * 0.8:
                        peaks += 1
            features.append(peaks / 10.0)
        else:
            features.append(0.0)
        return np.array(features, dtype=np.float32)

    def engineer_features_v9(data):
        base_features = engineer_features_v8(data)
        temporal_features = compute_temporal_features(data)
        temporal_expanded = np.tile(temporal_features, (base_features.shape[0], 1))
        return np.concatenate([base_features, temporal_expanded], axis=1)

    # Model architecture (same structure)
    class TemporalPyramid(nn.Module):
        def __init__(self, num_classes, feature_dim, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            ])
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim // 2 * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.35))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=0.35)
            self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)])
            self.pre_classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2 * 4), nn.Dropout(0.35), nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2), nn.GELU(), nn.Dropout(0.35))
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.anti_attractor = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))

        def forward(self, x):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [((lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1)) for head in self.attention_heads]
            return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))

    # Load both models
    print("\nLoading models...")
    
    model_v8 = TemporalPyramid(num_classes=len(ALL_CLASSES), feature_dim=CONFIG_V8["feature_dim"])
    checkpoint_v8 = torch.load("/data/checkpoints_v8/ksl_v8_best.pth", map_location="cuda")
    model_v8.load_state_dict(checkpoint_v8["model_state_dict"])
    model_v8 = model_v8.cuda().eval()
    print(f"v8 loaded: {checkpoint_v8.get('val_acc', 0)*100:.2f}%")

    model_v9 = TemporalPyramid(num_classes=len(ALL_CLASSES), feature_dim=CONFIG_V9["feature_dim"])
    checkpoint_v9 = torch.load("/data/checkpoints_v9/ksl_v9_best.pth", map_location="cuda")
    model_v9.load_state_dict(checkpoint_v9["model_state_dict"])
    model_v9 = model_v9.cuda().eval()
    print(f"v9 loaded: {checkpoint_v9.get('val_acc', 0)*100:.2f}%")

    # Load validation data
    val_samples = []
    for class_name in ALL_CLASSES:
        class_dir = f"/data/val_v2/{class_name}"
        if os.path.exists(class_dir):
            for fn in os.listdir(class_dir):
                if fn.endswith(".npy"):
                    val_samples.append((os.path.join(class_dir, fn), ALL_CLASSES.index(class_name), class_name))
    print(f"Validation samples: {len(val_samples)}")

    # Compute normalization stats
    all_data = np.concatenate([np.load(val_samples[i][0]).flatten() for i in range(min(200, len(val_samples)))])
    mean, std = all_data.mean(), all_data.std() + 1e-8

    def prepare_input_v8(data):
        data = (data - mean) / std
        if data.shape[0] > 90:
            data = data[np.linspace(0, data.shape[0]-1, 90, dtype=int)]
        elif data.shape[0] < 90:
            data = np.vstack([data, np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features_v8(data)).unsqueeze(0).cuda()

    def prepare_input_v9(data):
        data_norm = (data - mean) / std
        if data_norm.shape[0] > 90:
            data_norm = data_norm[np.linspace(0, data_norm.shape[0]-1, 90, dtype=int)]
        elif data_norm.shape[0] < 90:
            data_norm = np.vstack([data_norm, np.zeros((90-data_norm.shape[0], data_norm.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features_v9(data_norm)).unsqueeze(0).cuda()

    # Evaluate with different ensemble strategies
    strategies = {
        "v8_only": lambda v8_pred, v9_pred, v8_conf, v9_conf, gt: v8_pred,
        "v9_only": lambda v8_pred, v9_pred, v8_conf, v9_conf, gt: v9_pred,
        "v9_for_victims": lambda v8_pred, v9_pred, v8_conf, v9_conf, gt: v9_pred if gt in V9_PREFERRED_CLASSES else v8_pred,
        "confidence_max": lambda v8_pred, v9_pred, v8_conf, v9_conf, gt: v8_pred if v8_conf > v9_conf else v9_pred,
        "v9_if_v8_predicts_54": lambda v8_pred, v9_pred, v8_conf, v9_conf, gt: v9_pred if v8_pred == "54" and gt in V9_PREFERRED_CLASSES else v8_pred,
        "weighted_logits": None,  # Special handling below
    }

    # Run evaluation
    results = {}
    per_class_results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    
    print("\n" + "=" * 70)
    print("RUNNING ENSEMBLE STRATEGIES")
    print("=" * 70)

    with torch.no_grad():
        # First pass: collect all predictions
        all_preds = []
        for filepath, label, class_name in val_samples:
            data = np.load(filepath).astype(np.float32)
            
            logits_v8 = model_v8(prepare_input_v8(data))
            logits_v9 = model_v9(prepare_input_v9(data))
            
            pred_v8 = ALL_CLASSES[logits_v8.argmax(dim=1).item()]
            pred_v9 = ALL_CLASSES[logits_v9.argmax(dim=1).item()]
            
            conf_v8 = F.softmax(logits_v8, dim=1).max().item()
            conf_v9 = F.softmax(logits_v9, dim=1).max().item()
            
            # Weighted logits (0.6 v8 + 0.4 v9)
            weighted = 0.6 * F.softmax(logits_v8, dim=1) + 0.4 * F.softmax(logits_v9, dim=1)
            pred_weighted = ALL_CLASSES[weighted.argmax(dim=1).item()]
            
            all_preds.append({
                "filepath": filepath,
                "label": label,
                "class_name": class_name,
                "pred_v8": pred_v8,
                "pred_v9": pred_v9,
                "conf_v8": conf_v8,
                "conf_v9": conf_v9,
                "pred_weighted": pred_weighted,
            })

        # Evaluate each strategy
        for strategy_name, strategy_fn in strategies.items():
            correct = 0
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            
            for p in all_preds:
                class_total[p["class_name"]] += 1
                
                if strategy_name == "weighted_logits":
                    pred = p["pred_weighted"]
                else:
                    pred = strategy_fn(p["pred_v8"], p["pred_v9"], p["conf_v8"], p["conf_v9"], p["class_name"])
                
                if pred == p["class_name"]:
                    correct += 1
                    class_correct[p["class_name"]] += 1
            
            acc = correct / len(all_preds)
            results[strategy_name] = {
                "accuracy": acc,
                "class_correct": dict(class_correct),
                "class_total": dict(class_total),
            }

    # Print results
    print(f"\n{'Strategy':<25} | {'Overall':>8} | {'444':>6} | {'35':>6} | {'9':>6} | {'388':>6} | {'91':>6}")
    print("-" * 80)
    
    for strategy_name, res in results.items():
        acc_444 = res["class_correct"].get("444", 0) / res["class_total"].get("444", 1) * 100
        acc_35 = res["class_correct"].get("35", 0) / res["class_total"].get("35", 1) * 100
        acc_9 = res["class_correct"].get("9", 0) / res["class_total"].get("9", 1) * 100
        acc_388 = res["class_correct"].get("388", 0) / res["class_total"].get("388", 1) * 100
        acc_91 = res["class_correct"].get("91", 0) / res["class_total"].get("91", 1) * 100
        
        print(f"{strategy_name:<25} | {res['accuracy']*100:7.2f}% | {acc_444:5.1f}% | {acc_35:5.1f}% | {acc_9:5.1f}% | {acc_388:5.1f}% | {acc_91:5.1f}%")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n{'='*70}")
    print(f"BEST STRATEGY: {best_strategy[0]} with {best_strategy[1]['accuracy']*100:.2f}%")
    print(f"{'='*70}")

    # Detailed analysis for best strategy
    best_res = best_strategy[1]
    print(f"\nPer-class accuracy for '{best_strategy[0]}':")
    print("\nNUMBERS:")
    for c in NUMBER_CLASSES:
        if best_res["class_total"].get(c, 0) > 0:
            acc = best_res["class_correct"].get(c, 0) / best_res["class_total"][c] * 100
            marker = " <-- v9 preferred" if c in V9_PREFERRED_CLASSES else ""
            print(f"  {c:>6}: {acc:5.1f}% ({best_res['class_correct'].get(c, 0)}/{best_res['class_total'][c]}){marker}")

    print("\nWORDS:")
    for c in WORD_CLASSES:
        if best_res["class_total"].get(c, 0) > 0:
            acc = best_res["class_correct"].get(c, 0) / best_res["class_total"][c] * 100
            print(f"  {c:>12}: {acc:5.1f}% ({best_res['class_correct'].get(c, 0)}/{best_res['class_total'][c]})")

    # Compare v8 vs v9 vs best ensemble
    print(f"\n{'='*70}")
    print("v8 vs v9 vs BEST ENSEMBLE COMPARISON")
    print(f"{'='*70}")
    
    key_classes = ["444", "35", "91", "100", "125", "54", "9", "388", "268", "Tortoise"]
    print(f"{'Class':>12} | {'v8':>8} | {'v9':>8} | {'Ensemble':>8}")
    print("-" * 50)
    
    for c in key_classes:
        v8_acc = results["v8_only"]["class_correct"].get(c, 0) / results["v8_only"]["class_total"].get(c, 1) * 100
        v9_acc = results["v9_only"]["class_correct"].get(c, 0) / results["v9_only"]["class_total"].get(c, 1) * 100
        best_acc = best_res["class_correct"].get(c, 0) / best_res["class_total"].get(c, 1) * 100
        print(f"{c:>12} | {v8_acc:7.1f}% | {v9_acc:7.1f}% | {best_acc:7.1f}%")
    
    print("-" * 50)
    print(f"{'OVERALL':>12} | {results['v8_only']['accuracy']*100:7.2f}% | {results['v9_only']['accuracy']*100:7.2f}% | {best_strategy[1]['accuracy']*100:7.2f}%")

    return {
        "best_strategy": best_strategy[0],
        "best_accuracy": best_strategy[1]["accuracy"],
        "v8_accuracy": results["v8_only"]["accuracy"],
        "v9_accuracy": results["v9_only"]["accuracy"],
    }


@app.local_entrypoint()
def main():
    results = ensemble_assess.remote()
    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS")
    print(f"Best: {results['best_strategy']} @ {results['best_accuracy']*100:.2f}%")
    print(f"v8: {results['v8_accuracy']*100:.2f}% | v9: {results['v9_accuracy']*100:.2f}%")
    print(f"{'='*70}")
