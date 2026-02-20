#!/usr/bin/env python3
"""
V37 Real-Tester Evaluation.

GroupNorm throughout — no AdaBN needed at inference.
Forward: (logits, signer_logits, embedding, proj_features)

Usage:
    python evaluate_v37.py --category both
    python evaluate_v37.py --category numbers
    python evaluate_v37.py --category words
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES,
    check_mediapipe_version,
    build_adj,
    discover_test_videos,
    preprocess_multistream,
    preextract_test_data,
    evaluate_method,
    _load_fusion_weights,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

from train_ksl_v37 import KSLGraphNetV37


def load_v37_models(ckpt_dir, classes, device):
    """Load v37 GroupNorm multi-stream models."""
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
        proj_dim = config.get("proj_dim", 128)

        model = KSLGraphNetV37(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
            proj_dim=proj_dim,
            nn_=config.get("num_nodes", 48),
            ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64),
            nl=config.get("num_layers", 4),
            tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1),
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {sname}: val_acc={ckpt.get('val_acc', 0):.1f}%, "
              f"epoch={ckpt.get('epoch', '?')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def make_predict_fn(models, fusion_weights, classes, device):
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
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


def main():
    parser = argparse.ArgumentParser(description="V37 Real-Tester Evaluation")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--checkpoint-base", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v37")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("V37 Real-Tester Evaluation")
    print("GroupNorm + SupCon + Speed Augmentation (no AdaBN needed)")
    print(f"Checkpoint base: {args.checkpoint_base}")
    print(f"Category:        {args.category}")
    print(f"Started:         {ts_start}")
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

        ckpt_dir = os.path.join(args.checkpoint_base, cat_name)
        if not os.path.isdir(ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} ({len(cat_videos)} test videos)")
        print(f"{'='*70}")

        print(f"\n  Loading models from {ckpt_dir}...")
        models, fusion_weights = load_v37_models(ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  ERROR: Only {len(models)}/3 streams loaded, skipping")
            continue

        print(f"\n  Fusion weights: {fusion_weights}")

        _ = preextract_test_data(cat_videos)  # warm up mediapipe

        predict_fn = make_predict_fn(models, fusion_weights, classes, device)
        result = evaluate_method(f"v37_{cat_name}", cat_videos, predict_fn, classes)
        all_results[cat_name] = result

    # --- Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V37")
    print(f"{'='*70}")
    print(f"\n  Reference baselines:")
    print(f"    exp1 GroupNorm:             Numbers=61.0%, Words=61.7%")
    print(f"    exp5 SupCon:                Numbers=61.0%, Words=65.4%")
    print(f"    Weighted Ensemble (best):   Numbers=76.3%, Words=72.8%, Combined=74.3%\n")

    combined_correct = combined_total = 0
    for cat_name, result in all_results.items():
        acc = result["overall"]
        n = result.get("total", "?")
        print(f"  {cat_name.upper():>10s}: {acc:.1f}%  ({n} videos)")
        if isinstance(n, int):
            combined_correct += round(acc / 100 * n)
            combined_total += n

    if combined_total > 0:
        combined_acc = 100.0 * combined_correct / combined_total
        print(f"  {'COMBINED':>10s}: {combined_acc:.1f}%  ({combined_total} videos)")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"v37_eval_{ts_str}.json")
    save_data = {
        "version": "v37",
        "checkpoint_base": args.checkpoint_base,
        "timestamp": ts_str,
        "results": {
            cat: {
                "accuracy": r["overall"],
                "total": r.get("total"),
                "per_class": r.get("per_class", {}),
                "per_signer": r.get("per_signer", {}),
            }
            for cat, r in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
