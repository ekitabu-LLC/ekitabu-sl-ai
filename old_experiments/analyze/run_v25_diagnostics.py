#!/usr/bin/env python3
"""
V25 Diagnostics Phase: 6 measurements BEFORE any model changes.

D1. Per-signer loss variance at v22 convergence
D2. Feature visualization (t-SNE) — signer vs class
D3. BatchNorm statistics comparison (train vs real-tester activations)
D4. Landmark quality comparison (jitter between train and test)
D5. Side-view impact quantification (Signer 3 numbers)
D6. Sink class analysis (confusion patterns on real testers)

Usage:
    python run_v25_diagnostics.py

Outputs:
    data/results/v25_diagnostics/
        d1_per_signer_loss.json
        d2_tsne_by_signer.png
        d2_tsne_by_class.png
        d3_bn_stats_shift.json
        d3_bn_stats_shift.png
        d4_landmark_jitter.json
        d4_landmark_jitter.png
        d5_side_view_impact.json
        d6_sink_class_analysis.json
        d6_confusion_matrices.png
        diagnostics_summary.json
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from pathlib import Path

# Add project root
PROJECT_ROOT = "/scratch/alpine/hama5612/ksl-dir-2"
sys.path.insert(0, PROJECT_ROOT)

from train_ksl_v22 import (
    KSLGraphNetV22, KSLGraphDataset, V22_CONFIG,
    build_adj, NUMBER_CLASSES, WORD_CLASSES, NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS,
    normalize_wrist_palm, compute_bones, compute_joint_angles,
    compute_fingertip_distances, extract_signer_id, POSE_INDICES,
)

NUMBERS = NUMBER_CLASSES
WORDS = WORD_CLASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train_v2")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val_v2")
CKPT_DIR = os.path.join(PROJECT_ROOT, "data", "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results", "v25_diagnostics")
REAL_TESTER_RESULTS = os.path.join(
    PROJECT_ROOT, "data", "results", "v22_real_testers_20260209_132917.json"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def ts():
    return time.strftime("%H:%M:%S")


def load_v22_model(split_name, classes, device):
    """Load trained v22 model checkpoint."""
    config = V22_CONFIG
    adj = build_adj(config["num_nodes"]).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS

    # Load checkpoint to get num_signers
    ckpt_path = os.path.join(CKPT_DIR, f"v22_{split_name}", "best_model.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_signers = state.get("num_signers", 5)

    # Load training dataset (aug=False for evaluation)
    train_ds = KSLGraphDataset(TRAIN_DIR, classes, config, aug=False)

    model = KSLGraphNetV22(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        hd=config["hidden_dim"],
        nl=config["num_layers"],
        tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)

    model.load_state_dict(state["model"])
    model.eval()

    print(f"  Loaded checkpoint: epoch={state.get('epoch')}, "
          f"val_acc={state.get('val_acc', 0):.1f}%, "
          f"num_signers={num_signers}")

    return model, train_ds


# =========================================================================
# D1: Per-Signer Loss Variance
# =========================================================================

def run_d1_per_signer_loss(device):
    """Compute CE loss per signer on training data using v22 checkpoint."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D1: Per-Signer Loss Variance")
    print(f"[{ts()}] {'='*60}")

    results = {}

    for split_name, classes in [("numbers", NUMBERS), ("words", WORDS)]:
        print(f"\n[{ts()}] --- {split_name} ---")
        model, train_ds = load_v22_model(split_name, classes, device)

        # Group samples by signer
        signer_indices = defaultdict(list)
        for idx in range(len(train_ds)):
            signer_label = train_ds.signer_labels[idx]
            signer_indices[signer_label].append(idx)

        signer_to_name = {v: k for k, v in train_ds.signer_to_idx.items()}
        per_signer_losses = {}
        per_signer_accs = {}

        with torch.no_grad():
            for signer_label, indices in sorted(signer_indices.items()):
                signer_name = signer_to_name.get(signer_label, f"signer_{signer_label}")
                losses = []
                correct = 0
                total = 0

                for idx in indices:
                    gcn_t, aux_t, label, _ = train_ds[idx]
                    gcn_t = gcn_t.unsqueeze(0).to(device)
                    aux_t = aux_t.unsqueeze(0).to(device)
                    target = torch.tensor([label], device=device)

                    logits, _, _ = model(gcn_t, aux_t, grl_lambda=0.0)
                    loss = F.cross_entropy(logits, target).item()
                    losses.append(loss)

                    pred = logits.argmax(1).item()
                    if pred == label:
                        correct += 1
                    total += 1

                mean_loss = np.mean(losses)
                per_signer_losses[signer_name] = mean_loss
                per_signer_accs[signer_name] = correct / total * 100
                print(f"  Signer {signer_name}: loss={mean_loss:.4f}, "
                      f"acc={correct}/{total} ({correct/total*100:.1f}%), "
                      f"n_samples={len(indices)}")

        loss_values = list(per_signer_losses.values())
        loss_variance = float(np.var(loss_values))
        loss_std = float(np.std(loss_values))
        loss_mean = float(np.mean(loss_values))
        loss_range = float(max(loss_values) - min(loss_values))

        results[split_name] = {
            "per_signer_losses": per_signer_losses,
            "per_signer_accuracies": per_signer_accs,
            "loss_variance": loss_variance,
            "loss_std": loss_std,
            "loss_mean": loss_mean,
            "loss_range": loss_range,
            "loss_cov": loss_std / loss_mean if loss_mean > 0 else 0,
            "num_signers": len(signer_indices),
        }

        print(f"\n  Loss stats: mean={loss_mean:.4f}, std={loss_std:.4f}, "
              f"var={loss_variance:.6f}, range={loss_range:.4f}, "
              f"CoV={loss_std/loss_mean:.3f}")
        print(f"  Decision: {'V-REx has potential (var > 0.05)' if loss_variance > 0.05 else 'V-REx may not help (var < 0.05)'}")

    with open(os.path.join(RESULTS_DIR, "d1_per_signer_loss.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# D2: Feature Visualization (t-SNE)
# =========================================================================

def run_d2_feature_tsne(device):
    """Extract 320-dim embeddings, plot t-SNE colored by signer and class."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D2: Feature Visualization (t-SNE)")
    print(f"[{ts()}] {'='*60}")

    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{ts()}] WARNING: sklearn or matplotlib not available, skipping D2")
        return None

    results = {}

    for split_name, classes in [("numbers", NUMBERS), ("words", WORDS)]:
        print(f"\n[{ts()}] --- {split_name} ---")
        model, train_ds = load_v22_model(split_name, classes, device)

        # Also load val data
        val_ds = KSLGraphDataset(VAL_DIR, classes, V22_CONFIG, aug=False)

        embeddings = []
        labels = []
        signer_ids = []
        sources = []  # 'train' or 'val'

        with torch.no_grad():
            for ds, source_name in [(train_ds, "train"), (val_ds, "val")]:
                for idx in range(len(ds)):
                    gcn_t, aux_t, label, signer_label = ds[idx]
                    gcn_t = gcn_t.unsqueeze(0).to(device)
                    aux_t = aux_t.unsqueeze(0).to(device)

                    _, _, emb = model(gcn_t, aux_t, grl_lambda=0.0)
                    embeddings.append(emb.cpu().numpy().squeeze())
                    labels.append(label)
                    signer_ids.append(signer_label)
                    sources.append(source_name)

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        signer_ids = np.array(signer_ids)
        sources = np.array(sources)

        print(f"  Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
        print(f"  Train: {(sources == 'train').sum()}, Val: {(sources == 'val').sum()}")

        # Run t-SNE
        print(f"  Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                     random_state=42, max_iter=1000)
        coords = tsne.fit_transform(embeddings)

        # Plot 1: Colored by signer
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        unique_signers = np.unique(signer_ids)
        cmap_signer = plt.cm.tab10
        for i, s in enumerate(unique_signers):
            mask = signer_ids == s
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                          c=[cmap_signer(i)], s=15, alpha=0.6,
                          label=f"Signer {s}")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"{split_name}: Embeddings by Signer")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")

        # Plot 2: Colored by class
        unique_labels = np.unique(labels)
        cmap_class = plt.cm.tab20
        for i, c in enumerate(unique_labels):
            mask = labels == c
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                          c=[cmap_class(i % 20)], s=15, alpha=0.6,
                          label=f"{classes[c]}" if i < 20 else "")
        if len(unique_labels) <= 20:
            axes[1].legend(fontsize=6, ncol=2)
        axes[1].set_title(f"{split_name}: Embeddings by Class")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"d2_tsne_{split_name}.png"), dpi=150)
        plt.close()

        # Compute signer separation metric (silhouette score)
        try:
            from sklearn.metrics import silhouette_score
            signer_silhouette = silhouette_score(embeddings, signer_ids)
            class_silhouette = silhouette_score(embeddings, labels)
            print(f"  Silhouette scores: signer={signer_silhouette:.3f}, "
                  f"class={class_silhouette:.3f}")
            print(f"  Decision: {'GRL NOT removing signer info (signer silhouette > 0.1)' if signer_silhouette > 0.1 else 'GRL working (signer silhouette <= 0.1)'}")
        except Exception as e:
            signer_silhouette = None
            class_silhouette = None
            print(f"  Silhouette computation failed: {e}")

        results[split_name] = {
            "n_embeddings": len(embeddings),
            "embed_dim": int(embeddings.shape[1]),
            "signer_silhouette": float(signer_silhouette) if signer_silhouette is not None else None,
            "class_silhouette": float(class_silhouette) if class_silhouette is not None else None,
            "n_train": int((sources == "train").sum()),
            "n_val": int((sources == "val").sum()),
        }

    with open(os.path.join(RESULTS_DIR, "d2_tsne_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# D3: BatchNorm Statistics Comparison
# =========================================================================

def run_d3_bn_stats(device):
    """Compare BN running stats vs actual activation stats on train/val data."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D3: BatchNorm Statistics Comparison")
    print(f"[{ts()}] {'='*60}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{ts()}] WARNING: matplotlib not available, skipping plots")
        plt = None

    results = {}

    for split_name, classes in [("numbers", NUMBERS), ("words", WORDS)]:
        print(f"\n[{ts()}] --- {split_name} ---")
        model, train_ds = load_v22_model(split_name, classes, device)
        val_ds = KSLGraphDataset(VAL_DIR, classes, V22_CONFIG, aug=False)

        # Collect BN layers and their running stats
        bn_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                bn_layers[name] = {
                    "type": type(module).__name__,
                    "num_features": module.num_features,
                    "running_mean": module.running_mean.cpu().numpy().copy(),
                    "running_var": module.running_var.cpu().numpy().copy(),
                }

        print(f"  Found {len(bn_layers)} BN layers")

        # Hook to capture activations
        activation_stats = {name: {"means": [], "vars": []} for name in bn_layers}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                x = input[0]
                if isinstance(module, torch.nn.BatchNorm2d):
                    # x: (B, C, H, W) → compute per-channel stats
                    mean = x.mean(dim=[0, 2, 3]).detach().cpu().numpy()
                    var = x.var(dim=[0, 2, 3]).detach().cpu().numpy()
                else:
                    # x: (B, C, ...) → compute per-channel stats
                    dims = list(range(x.dim()))
                    dims.remove(1)  # keep channel dim
                    mean = x.mean(dim=dims).detach().cpu().numpy()
                    var = x.var(dim=dims).detach().cpu().numpy()
                activation_stats[layer_name]["means"].append(mean)
                activation_stats[layer_name]["vars"].append(var)
            return hook_fn

        for name, module in model.named_modules():
            if name in bn_layers:
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Forward pass on val data (closest proxy for test distribution)
        model.eval()
        with torch.no_grad():
            for idx in range(len(val_ds)):
                gcn_t, aux_t, _, _ = val_ds[idx]
                gcn_t = gcn_t.unsqueeze(0).to(device)
                aux_t = aux_t.unsqueeze(0).to(device)
                model(gcn_t, aux_t, grl_lambda=0.0)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute shift magnitude
        layer_shifts = {}
        for name, info in bn_layers.items():
            act_means = np.mean(activation_stats[name]["means"], axis=0)
            act_vars = np.mean(activation_stats[name]["vars"], axis=0)
            running_mean = info["running_mean"]
            running_var = info["running_var"]

            # Shift in units of running std
            running_std = np.sqrt(running_var + 1e-5)
            mean_shift = np.abs(act_means - running_mean) / running_std
            var_ratio = act_vars / (running_var + 1e-5)

            layer_shifts[name] = {
                "type": info["type"],
                "num_features": info["num_features"],
                "mean_shift_avg": float(np.mean(mean_shift)),
                "mean_shift_max": float(np.max(mean_shift)),
                "var_ratio_avg": float(np.mean(var_ratio)),
                "var_ratio_min": float(np.min(var_ratio)),
                "var_ratio_max": float(np.max(var_ratio)),
            }

            shift_flag = "SHIFT" if np.mean(mean_shift) > 0.5 else "ok"
            print(f"  {name:40s} [{info['type']:15s}] "
                  f"mean_shift={np.mean(mean_shift):.3f} "
                  f"var_ratio={np.mean(var_ratio):.3f} "
                  f"[{shift_flag}]")

        # Overall decision
        shifts = [v["mean_shift_avg"] for v in layer_shifts.values()]
        avg_shift = np.mean(shifts)
        max_shift = np.max(shifts)
        n_shifted = sum(1 for s in shifts if s > 0.5)

        print(f"\n  Overall: avg_shift={avg_shift:.3f}, max_shift={max_shift:.3f}, "
              f"layers_shifted={n_shifted}/{len(shifts)}")
        print(f"  Decision: {'Tent TTA has potential' if n_shifted >= 3 else 'Tent TTA unlikely to help'}")

        results[split_name] = {
            "layer_shifts": layer_shifts,
            "avg_shift": float(avg_shift),
            "max_shift": float(max_shift),
            "n_shifted_layers": n_shifted,
            "total_bn_layers": len(shifts),
            "tent_potential": n_shifted >= 3,
        }

        # Plot
        if plt is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            names = list(layer_shifts.keys())
            mean_shifts = [layer_shifts[n]["mean_shift_avg"] for n in names]
            colors = ["red" if s > 0.5 else "steelblue" for s in mean_shifts]
            ax.barh(range(len(names)), mean_shifts, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n.split(".")[-1][:20] for n in names], fontsize=7)
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
            ax.set_xlabel("Mean Shift (in running std units)")
            ax.set_title(f"{split_name}: BN Statistics Shift (Val vs Running)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"d3_bn_shift_{split_name}.png"), dpi=150)
            plt.close()

    with open(os.path.join(RESULTS_DIR, "d3_bn_stats_shift.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# D4: Landmark Quality Comparison
# =========================================================================

def run_d4_landmark_jitter():
    """Compare frame-to-frame jitter between training data samples."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D4: Landmark Jitter Analysis")
    print(f"[{ts()}] {'='*60}")

    results = {}

    for split_name, classes in [("numbers", NUMBERS), ("words", WORDS)]:
        print(f"\n[{ts()}] --- {split_name} ---")

        signer_jitter = defaultdict(list)
        signer_frame_counts = defaultdict(list)

        for cn in classes:
            cd = os.path.join(TRAIN_DIR, cn)
            if not os.path.exists(cd):
                continue

            for fn in sorted(os.listdir(cd)):
                if not fn.endswith(".npy"):
                    continue

                signer = extract_signer_id(fn)
                d = np.load(os.path.join(cd, fn))
                f = d.shape[0]
                signer_frame_counts[signer].append(f)

                # Extract 48-node skeleton (same as v22)
                if d.shape[1] >= 225:
                    lh = d[:, 99:162].reshape(f, 21, 3)
                    rh = d[:, 162:225].reshape(f, 21, 3)
                    pose = np.zeros((f, 6, 3), dtype=np.float32)
                    for pi, idx_pose in enumerate(POSE_INDICES):
                        start = idx_pose * 3
                        pose[:, pi, :] = d[:, start:start + 3]
                    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)
                else:
                    continue

                # Normalize (same as v22)
                h = normalize_wrist_palm(h)

                # Compute jitter: frame-to-frame L2 displacement per joint
                if f > 1:
                    diff = h[1:] - h[:-1]  # (f-1, 48, 3)
                    jitter = np.sqrt((diff ** 2).sum(axis=-1))  # (f-1, 48)
                    mean_jitter = jitter.mean()
                    signer_jitter[signer].append(float(mean_jitter))

        # Report per-signer jitter
        print(f"  Per-signer jitter (after normalization):")
        signer_stats = {}
        for signer in sorted(signer_jitter.keys()):
            jitters = signer_jitter[signer]
            frames = signer_frame_counts[signer]
            mean_j = np.mean(jitters)
            std_j = np.std(jitters)
            mean_f = np.mean(frames)
            signer_stats[signer] = {
                "mean_jitter": float(mean_j),
                "std_jitter": float(std_j),
                "n_samples": len(jitters),
                "mean_frames": float(mean_f),
            }
            print(f"    Signer {signer}: jitter={mean_j:.4f}±{std_j:.4f}, "
                  f"n={len(jitters)}, avg_frames={mean_f:.1f}")

        # Compare max vs min signer jitter
        all_means = [s["mean_jitter"] for s in signer_stats.values()]
        jitter_ratio = max(all_means) / min(all_means) if min(all_means) > 0 else float("inf")
        print(f"\n  Jitter ratio (max/min signer): {jitter_ratio:.2f}")
        print(f"  Decision: {'Butterworth filter justified (ratio > 1.5)' if jitter_ratio > 1.5 else 'Jitter similar across signers (ratio < 1.5)'}")

        results[split_name] = {
            "per_signer": signer_stats,
            "jitter_ratio": float(jitter_ratio),
            "butterworth_justified": jitter_ratio > 1.5,
        }

    with open(os.path.join(RESULTS_DIR, "d4_landmark_jitter.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (split_name, split_data) in zip(axes, results.items()):
            signers = sorted(split_data["per_signer"].keys())
            means = [split_data["per_signer"][s]["mean_jitter"] for s in signers]
            stds = [split_data["per_signer"][s]["std_jitter"] for s in signers]
            ax.bar(signers, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
            ax.set_xlabel("Signer ID")
            ax.set_ylabel("Mean Jitter (L2/frame)")
            ax.set_title(f"{split_name}: Per-Signer Landmark Jitter")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "d4_landmark_jitter.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Plot failed: {e}")

    return results


# =========================================================================
# D5: Side-View Impact Quantification
# =========================================================================

def run_d5_side_view_impact():
    """Quantify Signer 3's side-view effect on number accuracy."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D5: Side-View Impact Quantification")
    print(f"[{ts()}] {'='*60}")

    with open(REAL_TESTER_RESULTS, "r") as f:
        real_results = json.load(f)

    results = {}

    for split_name in ["numbers", "words"]:
        split_data = real_results["results"][split_name]
        per_signer = split_data["per_signer"]

        total_correct = split_data["correct_noTTA"]
        total_samples = split_data["total"]

        # Get per-signer breakdown
        signer_stats = {}
        for signer_key, signer_data in per_signer.items():
            signer_stats[signer_key] = {
                "acc": signer_data["acc_noTTA"],
                "correct": signer_data["correct_noTTA"],
                "total": signer_data["total"],
            }
            print(f"  {split_name} - {signer_key}: "
                  f"{signer_data['correct_noTTA']}/{signer_data['total']} "
                  f"({signer_data['acc_noTTA']:.1f}%)")

        # Compute accuracy without Signer 3
        signer3_key = [k for k in signer_stats if "3" in k]
        if signer3_key:
            s3 = signer_stats[signer3_key[0]]
            without_s3_correct = total_correct - s3["correct"]
            without_s3_total = total_samples - s3["total"]
            without_s3_acc = without_s3_correct / without_s3_total * 100 if without_s3_total > 0 else 0

            overall_acc = total_correct / total_samples * 100

            print(f"\n  {split_name}: Overall={overall_acc:.1f}%, "
                  f"Without S3={without_s3_acc:.1f}%, "
                  f"S3 only={s3['acc']:.1f}%")
            print(f"  S3 drags accuracy by {without_s3_acc - overall_acc:.1f}pp")

            results[split_name] = {
                "overall_acc": float(overall_acc),
                "without_signer3_acc": float(without_s3_acc),
                "signer3_acc": float(s3["acc"]),
                "signer3_impact_pp": float(without_s3_acc - overall_acc),
                "per_signer": signer_stats,
            }
        else:
            results[split_name] = {"error": "Signer 3 not found in results"}

    # Combined impact
    total_correct_all = sum(
        real_results["results"][s]["correct_noTTA"] for s in ["numbers", "words"]
    )
    total_all = sum(
        real_results["results"][s]["total"] for s in ["numbers", "words"]
    )
    combined_acc = total_correct_all / total_all * 100

    # Without S3 combined
    s3_correct = sum(
        results[s]["per_signer"].get(
            [k for k in results[s]["per_signer"] if "3" in k][0], {}
        ).get("correct", 0)
        for s in ["numbers", "words"]
        if "per_signer" in results[s]
    )
    s3_total = sum(
        results[s]["per_signer"].get(
            [k for k in results[s]["per_signer"] if "3" in k][0], {}
        ).get("total", 0)
        for s in ["numbers", "words"]
        if "per_signer" in results[s]
    )
    without_s3_combined = (total_correct_all - s3_correct) / (total_all - s3_total) * 100

    results["combined"] = {
        "overall_acc": float(combined_acc),
        "without_signer3_acc": float(without_s3_combined),
        "signer3_impact_pp": float(without_s3_combined - combined_acc),
    }
    print(f"\n  Combined: Overall={combined_acc:.1f}%, Without S3={without_s3_combined:.1f}%")
    print(f"  Decision: {'View-angle augmentation needed (S3 impact > 5pp)' if abs(without_s3_combined - combined_acc) > 5 else 'Side-view not dominant issue'}")

    with open(os.path.join(RESULTS_DIR, "d5_side_view_impact.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# D6: Sink Class Analysis
# =========================================================================

def run_d6_sink_class_analysis():
    """Analyze confusion patterns and sink classes on real testers."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] D6: Sink Class Analysis")
    print(f"[{ts()}] {'='*60}")

    with open(REAL_TESTER_RESULTS, "r") as f:
        real_results = json.load(f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    results = {}

    for split_name, classes in [("numbers", NUMBERS), ("words", WORDS)]:
        split_data = real_results["results"][split_name]

        # Get confusion matrix
        cm = np.array(split_data.get("confusion_matrix", []))
        if cm.size == 0:
            print(f"  {split_name}: No confusion matrix available")
            results[split_name] = {"error": "No confusion matrix"}
            continue

        n_classes = len(classes)
        print(f"\n[{ts()}] --- {split_name} ({n_classes} classes) ---")

        # Predictions per class (columns of confusion matrix)
        preds_per_class = cm.sum(axis=0)
        true_per_class = cm.sum(axis=1)
        correct_per_class = np.diag(cm)

        # Find sink classes (predicted much more than their true frequency)
        pred_ratio = preds_per_class / np.maximum(true_per_class, 1)

        print(f"  Per-class prediction analysis:")
        sink_classes = []
        zero_acc_classes = []
        for i, cn in enumerate(classes):
            acc = correct_per_class[i] / max(true_per_class[i], 1) * 100
            if preds_per_class[i] > 2 * true_per_class[i] and preds_per_class[i] > 5:
                sink_classes.append({
                    "class": cn,
                    "predicted_as": int(preds_per_class[i]),
                    "true_count": int(true_per_class[i]),
                    "correct": int(correct_per_class[i]),
                    "ratio": float(pred_ratio[i]),
                })
                print(f"    SINK: {cn:12s} predicted {preds_per_class[i]:3d}x "
                      f"(true={true_per_class[i]:2d}, correct={correct_per_class[i]:2d}, "
                      f"ratio={pred_ratio[i]:.1f}x)")
            if acc == 0 and true_per_class[i] > 0:
                zero_acc_classes.append(cn)
                # What does this class get confused with?
                if true_per_class[i] > 0:
                    confused_with = np.argsort(-cm[i])[:3]
                    confusions = [(classes[j], int(cm[i, j])) for j in confused_with if cm[i, j] > 0]
                    print(f"    ZERO: {cn:12s} (n={true_per_class[i]:2d}) "
                          f"→ confused with: {confusions}")

        results[split_name] = {
            "sink_classes": sink_classes,
            "zero_accuracy_classes": zero_acc_classes,
            "per_class_accuracy": {
                classes[i]: {
                    "accuracy": float(correct_per_class[i] / max(true_per_class[i], 1) * 100),
                    "correct": int(correct_per_class[i]),
                    "total": int(true_per_class[i]),
                    "predicted_as_this": int(preds_per_class[i]),
                }
                for i in range(n_classes)
            },
        }

        # Plot confusion matrix
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{split_name}: V22 Confusion Matrix (Real Testers)")

            # Add numbers
            for i in range(n_classes):
                for j in range(n_classes):
                    if cm[i, j] > 0:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                               fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")

            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"d6_confusion_{split_name}.png"), dpi=150)
            plt.close()

    with open(os.path.join(RESULTS_DIR, "d6_sink_class_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# Summary
# =========================================================================

def generate_summary(d1, d2, d3, d4, d5, d6):
    """Generate a summary of all diagnostics with go/no-go decisions."""
    print(f"\n[{ts()}] {'='*60}")
    print(f"[{ts()}] DIAGNOSTICS SUMMARY")
    print(f"[{ts()}] {'='*60}")

    decisions = {}

    # D1: V-REx decision
    if d1:
        for split in ["numbers", "words"]:
            var = d1[split]["loss_variance"]
            decisions[f"d1_vrex_{split}"] = {
                "metric": f"loss_variance={var:.6f}",
                "threshold": "var > 0.05",
                "decision": "IMPLEMENT" if var > 0.05 else "SKIP",
                "rationale": f"Per-signer loss variance is {'high enough' if var > 0.05 else 'too low'} "
                            f"for V-REx to have meaningful gradient signal",
            }
            print(f"  D1 V-REx ({split}): {'GO' if var > 0.05 else 'NO-GO'} "
                  f"(variance={var:.6f})")

    # D2: GRL effectiveness
    if d2:
        for split in ["numbers", "words"]:
            sil = d2[split].get("signer_silhouette")
            if sil is not None:
                decisions[f"d2_grl_{split}"] = {
                    "metric": f"signer_silhouette={sil:.3f}",
                    "threshold": "silhouette < 0.1",
                    "decision": "GRL EFFECTIVE" if sil < 0.1 else "GRL NOT EFFECTIVE",
                    "rationale": f"Signer information {'removed' if sil < 0.1 else 'still present'} "
                                f"in embeddings",
                }
                print(f"  D2 GRL ({split}): {'EFFECTIVE' if sil < 0.1 else 'NOT EFFECTIVE'} "
                      f"(signer_silhouette={sil:.3f})")

    # D3: Tent TTA decision
    if d3:
        for split in ["numbers", "words"]:
            n_shifted = d3[split]["n_shifted_layers"]
            total = d3[split]["total_bn_layers"]
            decisions[f"d3_tent_{split}"] = {
                "metric": f"shifted_layers={n_shifted}/{total}",
                "threshold": ">=3 layers shifted",
                "decision": "IMPLEMENT" if n_shifted >= 3 else "SKIP",
                "rationale": f"{n_shifted} BN layers show distribution shift > 0.5 std",
            }
            print(f"  D3 Tent ({split}): {'GO' if n_shifted >= 3 else 'NO-GO'} "
                  f"({n_shifted}/{total} layers shifted)")

    # D4: Butterworth decision
    if d4:
        for split in ["numbers", "words"]:
            ratio = d4[split]["jitter_ratio"]
            decisions[f"d4_butterworth_{split}"] = {
                "metric": f"jitter_ratio={ratio:.2f}",
                "threshold": "ratio > 1.5",
                "decision": "IMPLEMENT" if ratio > 1.5 else "SKIP",
                "rationale": f"Inter-signer jitter ratio is {ratio:.2f}x",
            }
            print(f"  D4 Butterworth ({split}): {'GO' if ratio > 1.5 else 'NO-GO'} "
                  f"(ratio={ratio:.2f})")

    # D5: Side-view augmentation decision
    if d5:
        impact = abs(d5.get("combined", {}).get("signer3_impact_pp", 0))
        decisions["d5_view_augmentation"] = {
            "metric": f"signer3_impact={impact:.1f}pp",
            "threshold": "impact > 3pp",
            "decision": "IMPLEMENT" if impact > 3 else "SKIP",
            "rationale": f"Removing Signer 3 changes combined accuracy by {impact:.1f}pp",
        }
        print(f"  D5 View-Aug: {'GO' if impact > 3 else 'NO-GO'} "
              f"(S3 impact={impact:.1f}pp)")

    # D6: Sink class analysis (informational)
    if d6:
        for split in ["numbers", "words"]:
            sinks = d6[split].get("sink_classes", [])
            zeros = d6[split].get("zero_accuracy_classes", [])
            decisions[f"d6_sinks_{split}"] = {
                "sink_classes": [s["class"] for s in sinks],
                "zero_accuracy_classes": zeros,
                "decision": "INFORMATIONAL",
            }
            print(f"  D6 Sinks ({split}): {[s['class'] for s in sinks]}, "
                  f"zero-acc: {zeros}")

    # Overall recommendation
    print(f"\n[{ts()}] --- RECOMMENDED V25 ACTIONS ---")
    go_items = [k for k, v in decisions.items() if v.get("decision") == "IMPLEMENT"]
    skip_items = [k for k, v in decisions.items() if v.get("decision") == "SKIP"]
    print(f"  IMPLEMENT: {go_items}")
    print(f"  SKIP: {skip_items}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decisions": decisions,
        "go_items": go_items,
        "skip_items": skip_items,
    }

    with open(os.path.join(RESULTS_DIR, "diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================================================================
# Main
# =========================================================================

def main():
    print(f"[{ts()}] V25 Diagnostics Phase")
    print(f"[{ts()}] Output: {RESULTS_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")

    d1, d2, d3, d4, d5, d6 = None, None, None, None, None, None

    try:
        d1 = run_d1_per_signer_loss(device)
    except Exception as e:
        print(f"[{ts()}] D1 FAILED: {e}")

    try:
        d2 = run_d2_feature_tsne(device)
    except Exception as e:
        print(f"[{ts()}] D2 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        d3 = run_d3_bn_stats(device)
    except Exception as e:
        print(f"[{ts()}] D3 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        d4 = run_d4_landmark_jitter()
    except Exception as e:
        print(f"[{ts()}] D4 FAILED: {e}")

    try:
        d5 = run_d5_side_view_impact()
    except Exception as e:
        print(f"[{ts()}] D5 FAILED: {e}")

    try:
        d6 = run_d6_sink_class_analysis()
    except Exception as e:
        print(f"[{ts()}] D6 FAILED: {e}")

    summary = generate_summary(d1, d2, d3, d4, d5, d6)

    print(f"\n[{ts()}] All diagnostics complete.")
    print(f"[{ts()}] Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
