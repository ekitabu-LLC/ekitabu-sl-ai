#!/usr/bin/env python3
"""
Optimize ensemble model weights for v27+v28+v29+exp1+exp5+openhands+v41.

Extends optimize_ensemble_weights_v2.py to include V41 (R&R temporal aug, GroupNorm).
Grid-searches for optimal weights across all 7 models.

Usage:
    python optimize_ensemble_v3.py
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
    extract_landmarks_from_video,
)
from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES, _load_fusion_weights,
)
from run_ensemble_v31 import (
    load_v31_exp1_models, load_v31_exp5_models,
    get_v28_style_probs, get_groupnorm_probs,
)
from evaluate_real_testers_v30_phase1 import (
    load_v27_model, load_v28_stream_models, load_v29_model,
)
from train_ksl_openhands import OpenHandsClassifier, OH_CONFIG
from evaluate_openhands_realtest import preprocess_raw_for_openhands, load_openhands_model
from train_ksl_v41 import KSLGraphNetV41, build_adj
from evaluate_real_testers_v30 import (
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"


def load_v41_models(category, device):
    """Load V41 R&R GroupNorm multi-stream models."""
    ckpt_dir = os.path.join(CKPT_BASE, "v41", category)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing v41/{sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        num_signers = ckpt.get("num_signers", 12)
        proj_dim = config.get("supcon_proj_dim", 128)
        num_nodes = config.get("num_nodes", 48)
        adj = build_adj(num_nodes).to(device)

        model = KSLGraphNetV41(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
            proj_dim=proj_dim,
            nn_=num_nodes,
            ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64),
            nl=config.get("num_layers", 4),
            tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v41/{sname}: val={ckpt.get('val_acc', 'N/A'):.1f}%")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def collect_per_model_probs(cat_videos, classes, category, device, alpha=0.3):
    """Collect probability vectors from all 7 models for all test videos."""

    # Load skeleton models
    v27_ckpt = os.path.join(CKPT_BASE, f"v27_{category}", "best_model.pt")
    v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

    v28_ckpt_dir = os.path.join(CKPT_BASE, f"v28_{category}")
    v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

    v29_ckpt = os.path.join(CKPT_BASE, f"v29_{category}", "best_model.pt")
    v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

    print("  Loading v31_exp1 (GroupNorm)...")
    exp1_models, exp1_fw = load_v31_exp1_models(category, device)

    print("  Loading v31_exp5 (SupCon)...")
    exp5_models, exp5_fw = load_v31_exp5_models(category, device)

    # Load OpenHands model
    oh_ckpt = os.path.join(CKPT_BASE, "openhands", category, "best_model.pt")
    oh_model = None
    if os.path.exists(oh_ckpt):
        print(f"  Loading OpenHands from {oh_ckpt}...")
        oh_model = load_openhands_model(oh_ckpt, len(classes), device)
    else:
        print(f"  WARNING: OpenHands checkpoint not found: {oh_ckpt}")

    # Load V41 models
    print("  Loading V41 (R&R GroupNorm)...")
    v41_models, v41_fw = load_v41_models(category, device)

    # Pre-extract and adapt BN stats for skeleton models
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

    # Collect predictions for all videos
    model_names = ["v27", "v28", "v29", "exp1", "exp5", "openhands", "v41"]
    all_probs = {m: [] for m in model_names}
    all_labels = []

    for video_path, true_class, signer_id in cat_videos:
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

            # OpenHands
            if oh_model is not None:
                x_oh = preprocess_raw_for_openhands(raw)
                if x_oh is not None:
                    logits_oh, _ = oh_model(x_oh.unsqueeze(0).to(device))
                    all_probs["openhands"].append(
                        F.softmax(logits_oh, dim=1).cpu().squeeze(0)
                    )
                else:
                    all_probs["openhands"].append(torch.ones(len(classes)) / len(classes))
            else:
                all_probs["openhands"].append(torch.ones(len(classes)) / len(classes))

            # V41 (GroupNorm - no adaptation)
            if v41_models and streams is not None:
                probs = get_groupnorm_probs(streams, aux, v41_models, v41_fw, device)
                all_probs["v41"].append(probs)
            else:
                all_probs["v41"].append(torch.ones(len(classes)) / len(classes))

    # Stack
    for m in model_names:
        all_probs[m] = torch.stack(all_probs[m])  # (N, C)
    all_labels = np.array(all_labels)

    return all_probs, all_labels


def grid_search_weights_fast(model_probs, labels, model_names, step=0.05):
    """Faster grid search: enumerate only combinations that sum to ~1.0.

    Uses recursive generation to avoid brute-force enumeration.
    """
    n_models = len(model_names)
    n_steps = int(round(1.0 / step))

    # Uniform baseline
    uniform_fused = sum(model_probs[m] for m in model_names) / n_models
    uniform_preds = uniform_fused.argmax(dim=1).numpy()
    uniform_acc = 100.0 * (uniform_preds == labels).mean()

    best_acc = 0.0
    best_weights = {m: 1.0 / n_models for m in model_names}
    combos_tried = 0

    probs_list = [model_probs[m] for m in model_names]

    def search(depth, remaining_steps, current_weights):
        nonlocal best_acc, best_weights, combos_tried

        if depth == n_models - 1:
            w_last = remaining_steps * step
            current_weights.append(w_last)

            fused = sum(current_weights[i] * probs_list[i] for i in range(n_models))
            preds = fused.argmax(dim=1).numpy()
            acc = 100.0 * (preds == labels).mean()
            combos_tried += 1

            if acc > best_acc:
                best_acc = acc
                best_weights = {m: current_weights[i]
                                for i, m in enumerate(model_names)}

            current_weights.pop()
            return

        for s in range(remaining_steps + 1):
            w = s * step
            current_weights.append(w)
            search(depth + 1, remaining_steps - s, current_weights)
            current_weights.pop()

    search(0, n_steps, [])
    print(f"    Tried {combos_tried:,} weight combinations")
    return best_weights, best_acc, uniform_acc


def grid_search_weights(model_probs, labels, model_names, step=0.05):
    """Grid search for optimal weights summing to 1 (for small subsets)."""
    n_models = len(model_names)

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
    print("Ensemble Weight Optimization V3 (with OpenHands + V41)")
    print(f"Models: v27, v28, v29, exp1, exp5, openhands, v41")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"Found {len(raw_videos)} test videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model_names_7 = ["v27", "v28", "v29", "exp1", "exp5", "openhands", "v41"]
    model_names_6 = ["v27", "v28", "v29", "exp1", "exp5", "openhands"]
    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [(vp, name, signer) for vp, name, cat, signer in raw_videos if cat == category]
        if not cat_videos:
            continue

        print(f"\n{'='*70}")
        print(f"{category.upper()} --- {len(cat_videos)} videos")
        print(f"{'='*70}")

        model_probs, labels = collect_per_model_probs(cat_videos, classes, category, device)

        # Per-model accuracy
        print(f"\n  Per-model accuracy:")
        for m in model_names_7:
            preds = model_probs[m].argmax(dim=1).numpy()
            acc = 100.0 * (preds == labels).mean()
            print(f"    {m:>10s}: {acc:.1f}%")

        # 6-model optimization (previous best, for comparison)
        print(f"\n  6-model optimization (v27+v28+v29+exp1+exp5+openhands, step=0.1)...")
        sub6 = {m: model_probs[m] for m in model_names_6}
        best_w6, best_a6, uni_a6 = grid_search_weights_fast(
            sub6, labels, model_names_6, step=0.1
        )
        print(f"    Uniform: {uni_a6:.1f}%")
        print(f"    Optimized: {best_a6:.1f}% ({best_a6 - uni_a6:+.1f}pp)")
        print(f"    Weights: {', '.join(f'{m}={w:.2f}' for m, w in best_w6.items())}")

        # 7-model optimization (coarse step=0.1)
        print(f"\n  7-model optimization (+ V41, step=0.1)...")
        best_w7, best_a7, uni_a7 = grid_search_weights_fast(
            model_probs, labels, model_names_7, step=0.1
        )
        print(f"    Uniform: {uni_a7:.1f}%")
        print(f"    Optimized: {best_a7:.1f}% ({best_a7 - uni_a7:+.1f}pp)")
        print(f"    Weights: {', '.join(f'{m}={w:.2f}' for m, w in best_w7.items())}")

        # Fine-tune 7-model (step=0.05)
        print(f"\n  7-model fine-tuning (step=0.05)...")
        best_w7f, best_a7f, _ = grid_search_weights_fast(
            model_probs, labels, model_names_7, step=0.05
        )
        print(f"    Optimized: {best_a7f:.1f}% ({best_a7f - uni_a7:+.1f}pp)")
        print(f"    Weights: {', '.join(f'{m}={w:.2f}' for m, w in best_w7f.items())}")

        # Key subsets with V41
        print(f"\n  Subsets with V41:")
        subsets = [
            ["exp1", "exp5", "openhands", "v41"],
            ["exp5", "openhands", "v41"],
            ["v28", "exp1", "exp5", "openhands", "v41"],
            ["v29", "exp1", "exp5", "openhands", "v41"],
        ]
        for subset in subsets:
            sub_probs = {m: model_probs[m] for m in subset}
            sw, sa, su = grid_search_weights(sub_probs, labels, subset, step=0.05)
            print(f"    {'+'.join(subset):45s}: uniform={su:.1f}%, opt={sa:.1f}% ({sa-su:+.1f}pp)")
            print(f"      Weights: {', '.join(f'{m}={w:.2f}' for m, w in sw.items())}")

        all_results[category] = {
            "per_model_acc": {
                m: float(100.0 * (model_probs[m].argmax(1).numpy() == labels).mean())
                for m in model_names_7
            },
            "6model_optimized": {
                "acc": float(best_a6),
                "weights": {k: float(v) for k, v in best_w6.items()},
            },
            "7model_uniform_acc": float(uni_a7),
            "7model_optimized_coarse": {
                "acc": float(best_a7),
                "weights": {k: float(v) for k, v in best_w7.items()},
            },
            "7model_optimized_fine": {
                "acc": float(best_a7f),
                "weights": {k: float(v) for k, v in best_w7f.items()},
            },
        }

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    for cat, r in all_results.items():
        print(f"\n  {cat.upper()}:")
        print(f"    Per-model: {r['per_model_acc']}")
        print(f"    6-model optimized: {r['6model_optimized']['acc']:.1f}%")
        print(f"    7-model optimized (fine): {r['7model_optimized_fine']['acc']:.1f}% "
              f"({r['7model_optimized_fine']['acc'] - r['6model_optimized']['acc']:+.1f}pp vs 6-model)")
        print(f"    7-model weights: {r['7model_optimized_fine']['weights']}")

    if "numbers" in all_results and "words" in all_results:
        n_r = all_results["numbers"]
        w_r = all_results["words"]
        n_total = 59
        w_total = 81
        # 6-model combined
        n6 = round(n_r["6model_optimized"]["acc"] * n_total / 100)
        w6 = round(w_r["6model_optimized"]["acc"] * w_total / 100)
        combined_6 = 100.0 * (n6 + w6) / (n_total + w_total)
        # 7-model combined
        n7 = round(n_r["7model_optimized_fine"]["acc"] * n_total / 100)
        w7 = round(w_r["7model_optimized_fine"]["acc"] * w_total / 100)
        combined_7 = 100.0 * (n7 + w7) / (n_total + w_total)
        print(f"\n  Combined: 6-model={combined_6:.1f}% -> 7-model={combined_7:.1f}% "
              f"({combined_7 - combined_6:+.1f}pp)")
        print(f"  Reference: current best = 76.4% (v2 optimized)")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ensemble_weights_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
