#!/usr/bin/env python3
"""
V44 Expr2 Real-Tester Evaluation — Signer-Adversarial GRL + GroupNorm.

GroupNorm models do NOT need AdaBN (no signer-specific batch statistics).
Evaluate directly on real testers.

Usage:
    python evaluate_v44_expr2.py --category both
    python evaluate_v44_expr2.py --category numbers
    python evaluate_v44_expr2.py --category words
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
    discover_test_videos,
    preprocess_multistream,
    evaluate_method,
    _load_fusion_weights,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

from train_ksl_v44_expr2 import KSLGraphNetV44Expr2, build_adj


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_v44_expr2_models(ckpt_dir, classes, device):
    """Load V44 Expr2 models (GroupNorm, no AdaBN needed)."""
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
        num_nodes = config.get("num_nodes", 48)
        adj = build_adj(num_nodes).to(device)

        model = KSLGraphNetV44Expr2(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
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
    parser = argparse.ArgumentParser(
        description="V44 Expr2 Real-Tester Evaluation (Signer-Adversarial GRL)")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--checkpoint-base", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v44_expr2")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("V44 Expr2 Real-Tester Evaluation")
    print("Signer-Adversarial GRL + GroupNorm (no SupCon)")
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
        print(f"  GroupNorm — no AdaBN needed")
        print(f"{'='*70}")

        print(f"\n  Loading V44 Expr2 models from {ckpt_dir}...")
        models, fusion_weights = load_v44_expr2_models(ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  ERROR: Only {len(models)}/3 streams loaded, skipping")
            continue

        print(f"\n  Fusion weights: {fusion_weights}")

        # GroupNorm: no AdaBN needed
        predict_fn = make_predict_fn(models, fusion_weights, classes, device)
        result = evaluate_method(f"v44_expr2_{cat_name}", cat_videos, predict_fn, classes)
        all_results[cat_name] = result

    # --- Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V44 Expr2 Signer-Adversarial GRL")
    print(f"{'='*70}")
    print(f"\n  Reference baselines:")
    print(f"    exp1 GroupNorm:              Numbers=61.0%, Words=61.7%, Combined=61.4%")
    print(f"    exp5 SupCon:                 Numbers=61.0%, Words=65.4%, Combined=63.2%")
    print(f"    6-model uniform ensemble:    Numbers=74.6%, Words=71.6%, Combined=72.9%\n")

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

    os.makedirs(os.path.join(results_dir, "v44_expr2"), exist_ok=True)
    out_path = os.path.join(results_dir, "v44_expr2", "real_tester_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
