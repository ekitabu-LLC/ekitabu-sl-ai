#!/usr/bin/env python3
"""Quick experiment: Ensemble (v27+v28+v29) with Alpha-BN on v28 streams.

v27 and v29 use standard AdaBN, v28 uses Alpha-BN (alpha=0.3).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from the Phase 1 eval script
from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    ensemble_predict, evaluate_method,
    preprocess_v27, preprocess_multistream,
    REAL_TESTER_DIR, NUMBER_CLASSES, WORD_CLASSES,
)
import torch
import torch.nn.functional as F
import copy
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    alpha = args.alpha

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Alpha-BN alpha={alpha} for v28 streams, standard AdaBN for v27/v29")
    print()

    # Discover test videos
    videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"Found {len(videos)} test videos")

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [v for v in videos if v["category"] == category]
        if not cat_videos:
            continue

        print(f"\n{'='*60}")
        print(f"{category.upper()} ({len(cat_videos)} videos)")
        print(f"{'='*60}")

        # ---- Load models ----
        # V27
        from evaluate_real_testers_v30_phase1 import load_v27_model
        v27_model = load_v27_model(category, device)

        # V28
        from evaluate_real_testers_v30_phase1 import load_v28_models
        v28_models, v28_fusion_weights = load_v28_models(category, device)

        # V29
        from evaluate_real_testers_v30_phase1 import load_v29_model
        v29_model = load_v29_model(category, device)

        # ---- AdaBN on v27 (standard) ----
        if v27_model:
            print("  AdaBN on v27 (standard)...")
            v27_data = preextract_test_data(cat_videos, preprocess_fn="v27")
            adapt_bn_stats(v27_model, v27_data, device)

        # ---- Alpha-BN on v28 streams ----
        if v28_models:
            stream_data = preextract_test_data(cat_videos, preprocess_fn="multistream")
            for sname, smodel in v28_models.items():
                print(f"  Alpha-BN on v28/{sname} (alpha={alpha})...")
                source_stats = save_bn_stats(smodel)
                target_stats = compute_target_bn_stats(
                    smodel, stream_data, device,
                    is_multistream=True, stream_name=sname,
                )
                apply_alpha_bn(smodel, source_stats, target_stats, alpha)

        # ---- AdaBN on v29 (standard) ----
        if v29_model:
            print("  AdaBN on v29 (standard)...")
            v29_data = preextract_test_data(cat_videos, preprocess_fn="v27")
            adapt_bn_stats(v29_model, v29_data, device)

        # ---- Ensemble predict ----
        def predict_fn(raw, true_class):
            pred, conf, _ = ensemble_predict(
                raw, v27_model, v28_models, v28_fusion_weights, v29_model,
                device, classes,
            )
            if pred is None:
                return None
            return pred, conf

        results = evaluate_method(
            f"ensemble+alpha_bn({alpha})", cat_videos, predict_fn, classes
        )
        print(f"\n  {category}: {results['correct']}/{results['total']} = {results['accuracy']:.1f}%")


if __name__ == "__main__":
    main()
