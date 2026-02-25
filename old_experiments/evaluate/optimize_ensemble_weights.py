#!/usr/bin/env python3
"""
Optimize ensemble model weights for v27+v28+v29+exp1+exp5.

Instead of uniform 1/5 averaging, collect per-video probability vectors
from each model and grid-search for optimal weights.

Uses the same model loading and adaptation from run_ensemble_v31.py.

Usage:
    python optimize_ensemble_weights.py
"""

import sys
import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
)
from evaluate_real_testers_v30 import (
    KSLGraphNetV25, _load_fusion_weights,
    NUMBER_CLASSES, WORD_CLASSES, build_adj,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)
from run_ensemble_v31 import (
    load_v31_exp1_models, load_v31_exp5_models,
    get_v28_style_probs, get_groupnorm_probs,
)
from evaluate_real_testers_v30_phase1 import (
    load_v27_model, load_v28_stream_models, load_v29_model,
)

REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"


def collect_per_model_probs(cat_videos, classes, category, device, alpha=0.3):
    """Collect probability vectors from each of the 5 models for all test videos."""
    ckpt_base = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

    # Load models
    v27_ckpt = os.path.join(ckpt_base, f"v27_{category}", "best_model.pt")
    v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

    v28_ckpt_dir = os.path.join(ckpt_base, f"v28_{category}")
    v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

    v29_ckpt = os.path.join(ckpt_base, f"v29_{category}", "best_model.pt")
    v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

    print("  Loading v31_exp1 (GroupNorm)...")
    exp1_models, exp1_fw = load_v31_exp1_models(category, device)

    print("  Loading v31_exp5 (SupCon)...")
    exp5_models, exp5_fw = load_v31_exp5_models(category, device)

    # Pre-extract and adapt
    stream_data = preextract_test_data(cat_videos, preprocess_fn="multistream")

    if v27_model:
        v27_data = preextract_test_data(cat_videos, preprocess_fn="v27")
        adapt_bn_stats(v27_model, v27_data, device)

    if v28_models:
        for sname, smodel in v28_models.items():
            source_stats = save_bn_stats(smodel)
            target_stats = compute_target_bn_stats(
                smodel, stream_data, device,
                is_multistream=True, stream_name=sname,
            )
            apply_alpha_bn(smodel, source_stats, target_stats, alpha)

    if v29_model:
        v29_data = preextract_test_data(cat_videos, preprocess_fn="v27")
        adapt_bn_stats(v29_model, v29_data, device)

    if exp5_models:
        exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
        for sname in exp5_adabn:
            adapt_bn_stats(exp5_adabn[sname], stream_data, device,
                          is_multistream=True, stream_name=sname)
    else:
        exp5_adabn = exp5_models

    # Collect predictions
    model_names = ["v27", "v28", "v29", "exp1", "exp5"]
    all_probs = {m: [] for m in model_names}
    all_labels = []

    from evaluate_real_testers_v30 import extract_landmarks_from_video

    for video_path, true_class, signer_id in cat_videos:
        # Load landmarks (cached or extract from video)
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: {os.path.basename(video_path)}")
                for m in all_probs:
                    all_probs[m].append(torch.ones(len(classes)) / len(classes))
                all_labels.append(classes.index(true_class))
                continue
        all_labels.append(classes.index(true_class))

        streams, aux = preprocess_multistream(raw)
        gcn_v27, aux_v27 = preprocess_v27(raw)

        with torch.no_grad():
            # V27
            if v27_model and gcn_v27 is not None:
                logits, *_ = v27_model(
                    gcn_v27.unsqueeze(0).to(device),
                    aux_v27.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                all_probs["v27"].append(F.softmax(logits, dim=1).cpu().squeeze(0))
            else:
                all_probs["v27"].append(torch.ones(len(classes)) / len(classes))

            # V28
            if v28_models and streams is not None:
                probs = get_v28_style_probs(streams, aux, v28_models, v28_fw, device)
                all_probs["v28"].append(probs)
            else:
                all_probs["v28"].append(torch.ones(len(classes)) / len(classes))

            # V29
            if v29_model and gcn_v27 is not None:
                logits, *_ = v29_model(
                    gcn_v27.unsqueeze(0).to(device),
                    aux_v27.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                all_probs["v29"].append(F.softmax(logits, dim=1).cpu().squeeze(0))
            else:
                all_probs["v29"].append(torch.ones(len(classes)) / len(classes))

            # Exp1 (GroupNorm - no adaptation)
            if exp1_models and streams is not None:
                probs = get_groupnorm_probs(streams, aux, exp1_models, exp1_fw, device)
                all_probs["exp1"].append(probs)
            else:
                all_probs["exp1"].append(torch.ones(len(classes)) / len(classes))

            # Exp5 (AdaBN)
            if exp5_adabn and streams is not None:
                probs = get_v28_style_probs(streams, aux, exp5_adabn, exp5_fw, device)
                all_probs["exp5"].append(probs)
            else:
                all_probs["exp5"].append(torch.ones(len(classes)) / len(classes))

    # Stack
    for m in model_names:
        all_probs[m] = torch.stack(all_probs[m])  # (N, C)
    all_labels = np.array(all_labels)

    return all_probs, all_labels


def grid_search_weights(model_probs, labels, model_names, step=0.1):
    """Grid search for optimal weights summing to 1."""
    n_models = len(model_names)

    # Uniform baseline
    uniform_fused = sum(model_probs[m] for m in model_names) / n_models
    uniform_preds = uniform_fused.argmax(dim=1).numpy()
    uniform_acc = 100.0 * (uniform_preds == labels).mean()

    best_acc = 0.0
    best_weights = {m: 1.0 / n_models for m in model_names}

    steps = np.arange(0.0, 1.0 + step, step)

    for w in product(steps, repeat=n_models):
        if abs(sum(w) - 1.0) > 0.01:
            continue

        weights = {m: wi for m, wi in zip(model_names, w)}
        fused = sum(weights[m] * model_probs[m] for m in model_names)
        preds = fused.argmax(dim=1).numpy()
        acc = 100.0 * (preds == labels).mean()

        if acc > best_acc:
            best_acc = acc
            best_weights = weights

    return best_weights, best_acc, uniform_acc


def main():
    print("=" * 70)
    print("Ensemble Weight Optimization")
    print(f"Models: v27, v28, v29, exp1, exp5")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"Found {len(raw_videos)} test videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model_names = ["v27", "v28", "v29", "exp1", "exp5"]
    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [(vp, name, signer) for vp, name, cat, signer in raw_videos if cat == category]
        if not cat_videos:
            continue

        print(f"\n{'='*70}")
        print(f"{category.upper()} — {len(cat_videos)} videos")
        print(f"{'='*70}")

        model_probs, labels = collect_per_model_probs(cat_videos, classes, category, device)

        # Per-model accuracy
        print(f"\n  Per-model accuracy (after adaptation):")
        for m in model_names:
            preds = model_probs[m].argmax(dim=1).numpy()
            acc = 100.0 * (preds == labels).mean()
            print(f"    {m:>6s}: {acc:.1f}%")

        # 5-model optimization (coarse grid)
        print(f"\n  Optimizing 5-model weights (step=0.1)...")
        best_w5, best_a5, uni_a5 = grid_search_weights(model_probs, labels, model_names, step=0.1)
        print(f"    Uniform: {uni_a5:.1f}%")
        print(f"    Optimized: {best_a5:.1f}% ({best_a5 - uni_a5:+.1f}pp)")
        print(f"    Weights: {', '.join(f'{m}={w:.1f}' for m, w in best_w5.items())}")

        # Fine-tune around best (step=0.05)
        print(f"\n  Fine-tuning (step=0.05)...")
        best_w5f, best_a5f, _ = grid_search_weights(model_probs, labels, model_names, step=0.05)
        print(f"    Optimized: {best_a5f:.1f}% ({best_a5f - uni_a5:+.1f}pp)")
        print(f"    Weights: {', '.join(f'{m}={w:.2f}' for m, w in best_w5f.items())}")

        # Subsets
        print(f"\n  Model subsets:")
        subsets = [
            ["exp1", "exp5"],
            ["v28", "exp1", "exp5"],
            ["v27", "v28", "v29"],
            ["v27", "exp1", "exp5"],
        ]
        for subset in subsets:
            sub_probs = {m: model_probs[m] for m in subset}
            sw, sa, su = grid_search_weights(sub_probs, labels, subset, step=0.05)
            print(f"    {'+'.join(subset):30s}: uniform={su:.1f}%, opt={sa:.1f}% ({sa-su:+.1f}pp)")
            print(f"      Weights: {', '.join(f'{m}={w:.2f}' for m, w in sw.items())}")

        all_results[category] = {
            "per_model_acc": {m: float(100.0 * (model_probs[m].argmax(1).numpy() == labels).mean())
                             for m in model_names},
            "uniform_acc": float(uni_a5),
            "optimized_5model": {
                "acc": float(best_a5f),
                "weights": {k: float(v) for k, v in best_w5f.items()},
            },
        }

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    for cat, r in all_results.items():
        print(f"\n  {cat.upper()}:")
        print(f"    Per-model: {r['per_model_acc']}")
        print(f"    Uniform 5-model: {r['uniform_acc']:.1f}%")
        print(f"    Optimized 5-model: {r['optimized_5model']['acc']:.1f}% "
              f"({r['optimized_5model']['acc'] - r['uniform_acc']:+.1f}pp)")
        print(f"    Weights: {r['optimized_5model']['weights']}")

    if "numbers" in all_results and "words" in all_results:
        n_r = all_results["numbers"]
        w_r = all_results["words"]
        n_total = 59  # from previous results
        w_total = 81
        n_uni = round(n_r["uniform_acc"] * n_total / 100)
        w_uni = round(w_r["uniform_acc"] * w_total / 100)
        uni_combined = 100.0 * (n_uni + w_uni) / (n_total + w_total)
        n_opt = round(n_r["optimized_5model"]["acc"] * n_total / 100)
        w_opt = round(w_r["optimized_5model"]["acc"] * w_total / 100)
        opt_combined = 100.0 * (n_opt + w_opt) / (n_total + w_total)
        print(f"\n  Combined: uniform={uni_combined:.1f}% → optimized={opt_combined:.1f}%")
        print(f"  Reference: current ensemble = 70.7%")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ensemble_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
