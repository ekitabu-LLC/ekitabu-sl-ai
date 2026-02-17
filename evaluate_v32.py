#!/usr/bin/env python3
"""
V32 Evaluation - GroupNorm + SupCon (Signer-Agnostic)

No BN adaptation needed — GroupNorm computes per-sample statistics.
This is the TRUE signer-agnostic evaluation.

Usage:
    python evaluate_v32.py --category both
    python evaluate_v32.py --category numbers
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
from train_ksl_v32 import KSLGraphNetV32


def load_v32_models(ckpt_dir, classes, device):
    """Load v32 GroupNorm models — no BN adaptation needed."""
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

        model = KSLGraphNetV32(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded v32/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def main():
    parser = argparse.ArgumentParser(description="V32 Evaluation (Signer-Agnostic)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v32")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"V32 Evaluation: GroupNorm + SupCon (Signer-Agnostic)")
    print(f"NO BN adaptation — true generalization test")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Category:   {args.category}")
    print(f"Started:    {ts}")
    print("=" * 70)

    check_mediapipe_version()

    print(f"\nScanning {base_dir}...")
    videos = discover_test_videos(base_dir)
    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"Found {len(videos)} test videos (Numbers: {len(numbers_videos)}, Words: {len(words_videos)})")

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
            print(f"\nSkipping {cat_name}: no test videos")
            continue

        cat_ckpt_dir = os.path.join(args.checkpoint_dir, cat_name)
        if not os.path.isdir(cat_ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {cat_ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} — V32 GroupNorm + SupCon ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        print(f"\n  Loading models from {cat_ckpt_dir}...")
        models, fusion_weights = load_v32_models(cat_ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  WARNING: Only {len(models)}/3 stream models loaded, skipping")
            continue

        print(f"  Fusion weights: {fusion_weights}")
        print(f"  No BN adaptation needed (GroupNorm)")

        def predict(raw, true_class, _models=models, _fw=fusion_weights):
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

        result = evaluate_method(f"v32_{cat_name}", cat_videos, predict, classes)
        all_results[cat_name] = result

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — V32 GroupNorm + SupCon (NO BN adaptation)")
    print(f"{'='*70}")
    for cat_name in all_results:
        r = all_results[cat_name]
        print(f"  {cat_name}: {r['overall']:.1f}%")

    if "numbers" in all_results and "words" in all_results:
        n_r = all_results["numbers"]
        w_r = all_results["words"]
        n_cor = round(n_r["overall"] * n_r["total"] / 100)
        w_cor = round(w_r["overall"] * w_r["total"] / 100)
        combined = 100.0 * (n_cor + w_cor) / (n_r["total"] + w_r["total"])
        print(f"  Combined: {combined:.1f}%")
        print(f"\n  Reference: v31 exp1 (GroupNorm only) = 61.4%")
        print(f"  Reference: v31 exp5 (SupCon + BN + AdaBN) = 63.2%")
        print(f"  Reference: v31 ensemble (5 models + AdaBN) = 70.7%")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v32_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "version": "v32",
            "description": "GroupNorm + SupCon (signer-agnostic, no BN adaptation)",
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
