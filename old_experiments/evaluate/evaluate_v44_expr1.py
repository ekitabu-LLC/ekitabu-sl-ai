#!/usr/bin/env python3
"""
V44 Experiment 1: Anchor-based skeleton normalization (inference-time only).

Tests whether applying global anchor normalization (center on shoulder midpoint,
scale by shoulder width) to raw landmarks BEFORE the standard per-segment
wrist/palm normalization pipeline improves cross-signer generalization.

This is a pure inference-time experiment: no retraining. We load existing
checkpoints (exp1, exp5, v41, v43) and apply anchor_normalize() to raw
landmarks before feeding them through the standard preprocessing pipeline.

Models evaluated:
  - exp1 (v31_exp1): GroupNorm, no AdaBN needed
  - exp5 (v31_exp5): BatchNorm, needs AdaBN
  - v41:             GroupNorm + projection head, no AdaBN needed
  - v43:             BatchNorm + R&R numbers only, needs AdaBN

For each model, we compare:
  1. Baseline (no anchor norm) — should match known published numbers
  2. With anchor norm — the experiment

Usage:
    python evaluate_v44_expr1.py
    python evaluate_v44_expr1.py --category numbers
    python evaluate_v44_expr1.py --category words
    python evaluate_v44_expr1.py --models exp1 exp5
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from skeleton_normalize import anchor_normalize

from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES,
    check_mediapipe_version,
    discover_test_videos,
    preprocess_multistream,
    preextract_test_data,
    evaluate_method,
    _load_fusion_weights,
    adapt_bn_stats,
    extract_landmarks_from_video,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
from train_ksl_v31_exp1 import build_adj as build_adj_exp1

from train_ksl_v41 import KSLGraphNetV41
from train_ksl_v41 import build_adj as build_adj_v41

from train_ksl_v43 import KSLGraphNetV43
from train_ksl_v43 import build_adj as build_adj_v43

# exp5 uses the same architecture as v28 (KSLGraphNetV25 from eval v30)
from evaluate_real_testers_v30 import KSLGraphNetV25


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Model loaders (reused from individual eval scripts)
# ---------------------------------------------------------------------------

def load_exp1_models(ckpt_dir, classes, device):
    """Load v31 exp1 GroupNorm models."""
    from evaluate_real_testers_v30 import build_adj
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV31Exp1(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded exp1/{sname}: val_acc={ckpt.get('val_acc', 0):.1f}%, "
              f"epoch={ckpt.get('epoch', '?')}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_exp5_models(ckpt_dir, classes, device):
    """Load v31 exp5 BatchNorm models (same arch as v28: KSLGraphNetV25)."""
    from evaluate_real_testers_v30 import build_adj
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV25(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded exp5/{sname}: val_acc={ckpt.get('val_acc', 0):.1f}%, "
              f"epoch={ckpt.get('epoch', '?')}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v41_models(ckpt_dir, classes, device):
    """Load V41 GroupNorm + projection head models."""
    from evaluate_real_testers_v30 import build_adj
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        num_signers = ckpt.get("num_signers", 12)
        proj_dim = config.get("supcon_proj_dim", 128)

        model = KSLGraphNetV41(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            proj_dim=proj_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4),
            tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v41/{sname}: val_acc={ckpt.get('val_acc', 0):.1f}%, "
              f"epoch={ckpt.get('epoch', '?')}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v43_models(ckpt_dir, classes, device):
    """Load V43 BatchNorm + R&R numbers-only models."""
    from evaluate_real_testers_v30 import build_adj
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV43(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4),
            tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v43/{sname}: val_acc={ckpt.get('val_acc', 0):.1f}%, "
              f"epoch={ckpt.get('epoch', '?')}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


# ---------------------------------------------------------------------------
# Prediction function factory
# ---------------------------------------------------------------------------

def make_predict_fn(models, fusion_weights, classes, device, use_anchor_norm=False):
    """Create a prediction function, optionally with anchor normalization."""
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        if use_anchor_norm:
            raw = anchor_normalize(raw)
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            return None
        per_stream_probs = {}
        with torch.no_grad():
            for sname, model in models.items():
                gcn = streams[sname].unsqueeze(0).to(device)
                logits, *_ = model(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
        fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
        pred_idx = fused.argmax().item()
        return i2c[pred_idx], fused[pred_idx].item()

    return predict


def preextract_with_anchor_norm(videos):
    """Pre-extract test data with anchor normalization applied."""
    data_list = []
    for video_path, true_class, signer in videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                continue
        raw = anchor_normalize(raw)
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            continue
        data_list.append((streams, aux))
    return data_list


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "exp1": {
        "ckpt_base": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v31_exp1",
        "loader": load_exp1_models,
        "needs_adabn": False,  # GroupNorm
        "description": "v31 Exp1 (GroupNorm)",
    },
    "exp5": {
        "ckpt_base": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v31_exp5",
        "loader": load_exp5_models,
        "needs_adabn": True,   # BatchNorm
        "description": "v31 Exp5 (SupCon, BatchNorm)",
    },
    "v41": {
        "ckpt_base": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v41",
        "loader": load_v41_models,
        "needs_adabn": False,  # GroupNorm
        "description": "V41 (R&R, GroupNorm)",
    },
    "v43": {
        "ckpt_base": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v43",
        "loader": load_v43_models,
        "needs_adabn": True,   # BatchNorm
        "description": "V43 (R&R numbers, BatchNorm)",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="V44 Expr1: Anchor-based skeleton normalization (inference-time)")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--models", nargs="+", default=["exp1", "exp5", "v41", "v43"],
                        choices=["exp1", "exp5", "v41", "v43"],
                        help="Which models to evaluate")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline (no anchor norm) for verification")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results/v44_expr1"

    print("=" * 70)
    print("V44 Experiment 1: Anchor-Based Skeleton Normalization")
    print("Inference-time evaluation on existing checkpoints")
    print(f"Models:    {args.models}")
    print(f"Category:  {args.category}")
    print(f"Started:   {ts()}")
    print("=" * 70)

    check_mediapipe_version()

    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos   = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"Found {len(videos)} test videos "
          f"(Numbers: {len(numbers_videos)}, Words: {len(words_videos)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    categories = []
    if args.category in ("numbers", "both"):
        categories.append(("numbers", numbers_videos, NUMBER_CLASSES))
    if args.category in ("words", "both"):
        categories.append(("words", words_videos, WORD_CLASSES))

    all_results = {}

    for cat_name, cat_videos, classes in categories:
        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos found")
            continue

        print(f"\n{'='*70}")
        print(f"CATEGORY: {cat_name.upper()} ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        # Pre-extract test data for AdaBN (baseline, no anchor norm)
        print(f"\n  Pre-extracting test data (baseline)...")
        adabn_data_baseline = preextract_test_data(cat_videos)
        print(f"  Pre-extracted {len(adabn_data_baseline)} samples (baseline)")

        # Pre-extract test data for AdaBN (with anchor norm)
        if not args.baseline_only:
            print(f"  Pre-extracting test data (anchor normalized)...")
            adabn_data_anchor = preextract_with_anchor_norm(cat_videos)
            print(f"  Pre-extracted {len(adabn_data_anchor)} samples (anchor)")

        cat_results = {}

        for model_name in args.models:
            cfg = MODEL_CONFIGS[model_name]
            ckpt_dir = os.path.join(cfg["ckpt_base"], cat_name)

            if not os.path.isdir(ckpt_dir):
                print(f"\n  Skipping {model_name}/{cat_name}: {ckpt_dir} not found")
                continue

            print(f"\n  {'~'*60}")
            print(f"  Model: {cfg['description']} ({model_name})")
            print(f"  AdaBN: {'Yes' if cfg['needs_adabn'] else 'No (GroupNorm)'}")
            print(f"  {'~'*60}")

            # --- BASELINE (no anchor norm) ---
            print(f"\n  Loading {model_name} models for BASELINE...")
            models, fusion_weights = cfg["loader"](ckpt_dir, classes, device)
            if len(models) < 3:
                print(f"  ERROR: Only {len(models)}/3 streams loaded, skipping")
                continue

            if cfg["needs_adabn"]:
                print(f"  Applying AdaBN (baseline)...")
                for sname, model in models.items():
                    adapt_bn_stats(model, adabn_data_baseline, device, stream_name=sname)

            predict_fn_baseline = make_predict_fn(
                models, fusion_weights, classes, device, use_anchor_norm=False)
            result_baseline = evaluate_method(
                f"{model_name}_baseline_{cat_name}", cat_videos, predict_fn_baseline, classes)
            cat_results[f"{model_name}_baseline"] = result_baseline

            # --- WITH ANCHOR NORM ---
            if not args.baseline_only:
                # Reload models fresh for anchor norm (AdaBN needs clean state)
                print(f"\n  Loading {model_name} models for ANCHOR NORM...")
                models_an, fusion_weights_an = cfg["loader"](ckpt_dir, classes, device)

                if cfg["needs_adabn"]:
                    print(f"  Applying AdaBN (anchor normalized data)...")
                    for sname, model in models_an.items():
                        adapt_bn_stats(model, adabn_data_anchor, device, stream_name=sname)

                predict_fn_anchor = make_predict_fn(
                    models_an, fusion_weights_an, classes, device, use_anchor_norm=True)
                result_anchor = evaluate_method(
                    f"{model_name}_anchor_{cat_name}", cat_videos, predict_fn_anchor, classes)
                cat_results[f"{model_name}_anchor"] = result_anchor

        all_results[cat_name] = cat_results

    # --- SUMMARY ---
    print(f"\n{'='*70}")
    print("SUMMARY: V44 Experiment 1 — Anchor-Based Skeleton Normalization")
    print(f"{'='*70}")

    print(f"\n  Reference baselines (known published):")
    print(f"    exp1:  Numbers=61.0%, Words=61.7%, Combined=61.4%")
    print(f"    exp5:  Numbers=61.0%, Words=65.4%, Combined=63.2%")
    print(f"    v41:   Numbers=67.8%, Words=55.6%, Combined=60.7%")
    print(f"    v43:   Numbers=66.1%, Words=65.4%, Combined=65.7%")

    # Comparison table
    print(f"\n  {'Model':>12s}  {'Category':>10s}  {'Baseline':>10s}  {'Anchor':>10s}  {'Delta':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for cat_name, cat_results in all_results.items():
        for model_name in args.models:
            bl_key = f"{model_name}_baseline"
            an_key = f"{model_name}_anchor"
            bl_acc = cat_results.get(bl_key, {}).get("overall", 0.0)
            an_acc = cat_results.get(an_key, {}).get("overall", 0.0)
            delta = an_acc - bl_acc if an_acc > 0 else 0.0
            delta_str = f"{delta:+.1f}pp" if not args.baseline_only else "N/A"
            an_str = f"{an_acc:.1f}%" if not args.baseline_only else "N/A"
            print(f"  {model_name:>12s}  {cat_name:>10s}  {bl_acc:>9.1f}%  {an_str:>10s}  {delta_str:>8s}")

    # Combined accuracy
    if "numbers" in all_results and "words" in all_results:
        print(f"\n  Combined accuracy:")
        for model_name in args.models:
            for suffix in ["baseline", "anchor"]:
                if args.baseline_only and suffix == "anchor":
                    continue
                n_key = f"{model_name}_{suffix}"
                n_result = all_results.get("numbers", {}).get(n_key, {})
                w_result = all_results.get("words", {}).get(n_key, {})
                n_correct = n_result.get("correct", 0)
                n_total = n_result.get("total", 0)
                w_correct = w_result.get("correct", 0)
                w_total = w_result.get("total", 0)
                total = n_total + w_total
                if total > 0:
                    combined = 100.0 * (n_correct + w_correct) / total
                    print(f"    {model_name}_{suffix}: {combined:.1f}% "
                          f"({n_correct + w_correct}/{total})")

    # Ensemble comparison
    print(f"\n  Uniform ensemble comparison (equal weight across models):")
    for cat_name, cat_results in all_results.items():
        for suffix in ["baseline", "anchor"]:
            if args.baseline_only and suffix == "anchor":
                continue
            accs = []
            for model_name in args.models:
                key = f"{model_name}_{suffix}"
                if key in cat_results:
                    accs.append(cat_results[key]["overall"])
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {cat_name}_{suffix}: avg={avg:.1f}% "
                      f"(models: {', '.join(f'{a:.1f}' for a in accs)})")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir,
        f"expr1_{args.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "v44_expr1_anchor_normalization",
            "description": "Global anchor normalization before per-segment norm",
            "category": args.category,
            "models": args.models,
            "results": all_results,
            "timestamp": ts(),
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Done: {ts()}")


if __name__ == "__main__":
    main()
