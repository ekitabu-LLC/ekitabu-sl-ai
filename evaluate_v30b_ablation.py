#!/usr/bin/env python3
"""
Lightweight evaluation for v30b ablation experiments.

Loads models from a custom --checkpoint-dir, runs AdaBN Global on real testers,
and reports per-class + overall accuracy for quick comparison against v28 baseline
(Numbers AdaBN Global = 55.9%).

Supports Phase 2 (KSLGraphNetV30) and Phase 3 (KSLBlockGCNet) model types.

Usage:
    python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp1
    python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp3 --model-type phase3
    python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp1 --category words
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# Import everything we need from the full eval script
from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES,
    check_mediapipe_version,
    build_adj,
    discover_test_videos,
    preprocess_multistream,
    preextract_test_data,
    extract_landmarks_from_video,
    adapt_bn_stats,
    evaluate_method,
    load_v30_phase2_stream_models,
    load_v30_phase3_stream_models,
    confidence_level,
)


def main():
    parser = argparse.ArgumentParser(
        description="V30b Ablation Evaluation - Quick AdaBN Global comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Checkpoint directory for the experiment (e.g., data/checkpoints/v30b_exp1)")
    parser.add_argument("--model-type", type=str, default="phase2",
                        choices=["phase2", "phase3"],
                        help="Model architecture (phase2=KSLGraphNetV30, phase3=KSLBlockGCNet)")
    parser.add_argument("--category", type=str, default="numbers",
                        choices=["numbers", "words", "both"],
                        help="Which category to evaluate")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

    print("=" * 70)
    print(f"V30b Ablation Evaluation")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Category:   {args.category}")
    print(f"Started:    {ts}")
    print("=" * 70)

    check_mediapipe_version()

    # Discover videos
    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"Found {len(videos)} test videos (Numbers: {len(numbers_videos)}, Words: {len(words_videos)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build categories to evaluate
    categories = []
    if args.category in ("numbers", "both"):
        categories.append(("numbers", numbers_videos, NUMBER_CLASSES))
    if args.category in ("words", "both"):
        categories.append(("words", words_videos, WORD_CLASSES))

    all_results = {}

    for cat_name, cat_videos, classes in categories:
        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos")
            continue

        # Determine checkpoint subdir
        if args.model_type == "phase2":
            cat_ckpt_dir = os.path.join(args.checkpoint_dir, f"v30_{cat_name}")
        else:
            cat_ckpt_dir = os.path.join(args.checkpoint_dir, f"v30_phase3_{cat_name}")

        if not os.path.isdir(cat_ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {cat_ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} EVALUATION ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        # Load models
        print(f"\n  Loading {args.model_type} models from {cat_ckpt_dir}...")
        if args.model_type == "phase2":
            models, fusion_weights = load_v30_phase2_stream_models(
                cat_ckpt_dir, classes, device)
        else:
            models, fusion_weights = load_v30_phase3_stream_models(
                cat_ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  WARNING: Only {len(models)}/3 stream models loaded, skipping")
            continue

        # Pre-extract test data for AdaBN
        stream_data = preextract_test_data(cat_videos)
        print(f"  Pre-extracted {len(stream_data)} samples for AdaBN")

        # --- BASELINE (no AdaBN) ---
        print(f"\n--- BASELINE (no AdaBN) ---")
        def baseline_predict(raw, true_class, _models=models, _fw=fusion_weights):
            streams, aux = preprocess_multistream(raw)
            if streams is None:
                return None
            i2c = {i: c for i, c in enumerate(classes)}
            per_stream_probs = {}
            with torch.no_grad():
                for sname, smodel in _models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                    per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
            fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
            pred_idx = fused.argmax().item()
            return i2c[pred_idx], fused[pred_idx].item()

        baseline_result = evaluate_method(
            f"v30b_baseline", cat_videos, baseline_predict, classes
        )

        # --- AdaBN GLOBAL ---
        print(f"\n--- AdaBN GLOBAL ---")
        adabn_models = {s: copy.deepcopy(m) for s, m in models.items()}
        for sname in adabn_models:
            print(f"  Adapting BN stats for {sname} stream...")
            adapt_bn_stats(adabn_models[sname], stream_data, device, stream_name=sname)
            print(f"    Adapted on {len(stream_data)} samples")

        def adabn_predict(raw, true_class, _models=adabn_models, _fw=fusion_weights):
            streams, aux = preprocess_multistream(raw)
            if streams is None:
                return None
            i2c = {i: c for i, c in enumerate(classes)}
            per_stream_probs = {}
            with torch.no_grad():
                for sname, smodel in _models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                    per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
            fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
            pred_idx = fused.argmax().item()
            return i2c[pred_idx], fused[pred_idx].item()

        adabn_result = evaluate_method(
            f"v30b_adabn_global", cat_videos, adabn_predict, classes
        )

        all_results[cat_name] = {
            "baseline": baseline_result,
            "adabn_global": adabn_result,
        }

    # --- FINAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - V30b Ablation")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model type: {args.model_type}")
    print(f"{'='*70}")

    print(f"\n  V28 baseline (AdaBN Global): Numbers=55.9%, Words=60.5%")
    print(f"  V30 Phase 1 best (Ens+ABN): Numbers=62.7%, Words=69.1%\n")

    for cat_name, result in all_results.items():
        bl = result["baseline"]["overall"]
        ab = result["adabn_global"]["overall"]
        print(f"  {cat_name.upper():>10s}: Baseline={bl:.1f}%  AdaBN Global={ab:.1f}%")

    # Save results
    exp_name = os.path.basename(args.checkpoint_dir)
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results/v30b"
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"{exp_name}_{ts_str}.json")

    save_data = {
        "experiment": exp_name,
        "checkpoint_dir": args.checkpoint_dir,
        "model_type": args.model_type,
        "timestamp": ts_str,
        "results": {},
    }
    for cat_name, result in all_results.items():
        save_data["results"][cat_name] = {
            "baseline_accuracy": result["baseline"]["overall"],
            "adabn_global_accuracy": result["adabn_global"]["overall"],
            "baseline_per_class": result["baseline"].get("per_class", {}),
            "adabn_per_class": result["adabn_global"].get("per_class", {}),
            "baseline_per_signer": result["baseline"].get("per_signer", {}),
            "adabn_per_signer": result["adabn_global"].get("per_signer", {}),
        }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
