#!/usr/bin/env python3
"""
V34 GroupNorm Ensemble Evaluation - Multi-Seed Signer-Agnostic Ensemble

Ensembles multiple GroupNorm models (all signer-agnostic, no BN adaptation needed).
This is the true signer-agnostic ensemble — no adaptation at inference time.

Models:
  - v31_exp1 (seed=42): 61.4% combined
  - v34a (seed=123): TBD
  - v34b (seed=456): TBD

Usage:
    python evaluate_v34_gn_ensemble.py --category both
    python evaluate_v34_gn_ensemble.py --models exp1,v34a,v34b
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
    evaluate_method,
    _load_fusion_weights,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)
from train_ksl_v31_exp1 import KSLGraphNetV31Exp1


# Default model directories
MODEL_DIRS = {
    "exp1": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v31_exp1",
    "v34a": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v34a_seed123",
    "v34b": "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v34b_seed456",
}


def load_groupnorm_model(ckpt_dir, cat_name, classes, device):
    """Load a GroupNorm model (3 streams + fusion weights)."""
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    cat_ckpt_dir = os.path.join(ckpt_dir, cat_name)

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(cat_ckpt_dir, sname, "best_model.pt")
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

    fusion_weights = _load_fusion_weights(cat_ckpt_dir)
    return models, fusion_weights


def predict_single_model(raw, models, fusion_weights, classes, device):
    """Get fused probability vector from one GroupNorm model."""
    streams, aux = preprocess_multistream(raw)
    if streams is None:
        return None

    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)

    fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
    return fused


def main():
    parser = argparse.ArgumentParser(description="V34 GroupNorm Ensemble Evaluation")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--models", type=str, default="exp1,v34a,v34b",
                        help="Comma-separated model names to ensemble")
    args = parser.parse_args()

    model_names = args.models.split(",")
    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"V34 GroupNorm Ensemble - Signer-Agnostic")
    print(f"Models: {model_names}")
    print(f"NO BN adaptation — all models use GroupNorm")
    print(f"Category: {args.category}")
    print(f"Started: {ts}")
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

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} — GroupNorm Ensemble ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        # Load all models
        all_models = {}
        for mname in model_names:
            ckpt_dir = MODEL_DIRS.get(mname, mname)
            cat_ckpt_dir = os.path.join(ckpt_dir, cat_name)
            if not os.path.isdir(cat_ckpt_dir):
                print(f"  WARNING: {mname} checkpoint dir not found: {cat_ckpt_dir}, skipping")
                continue
            print(f"\n  Loading {mname} from {ckpt_dir}...")
            models, fw = load_groupnorm_model(ckpt_dir, cat_name, classes, device)
            if len(models) == 3:
                all_models[mname] = (models, fw)
                print(f"  {mname}: loaded 3 streams, fusion={fw}")
            else:
                print(f"  WARNING: {mname} only has {len(models)}/3 streams, skipping")

        if not all_models:
            print(f"  No models available for {cat_name}")
            continue

        # Evaluate individual models first
        i2c = {i: c for i, c in enumerate(classes)}
        for mname, (models, fw) in all_models.items():
            def predict_ind(raw, true_class, _m=models, _fw=fw, _c=classes, _d=device):
                probs = predict_single_model(raw, _m, _fw, _c, _d)
                if probs is None:
                    return None
                pred_idx = probs.argmax().item()
                return i2c[pred_idx], probs[pred_idx].item()

            result = evaluate_method(f"{mname}_{cat_name}", cat_videos, predict_ind, classes)
            all_results[f"{cat_name}_{mname}"] = result

        # Ensemble: average probabilities
        if len(all_models) > 1:
            model_list = list(all_models.items())

            def predict_ensemble(raw, true_class, _ml=model_list, _c=classes, _d=device):
                probs_list = []
                for mname, (models, fw) in _ml:
                    p = predict_single_model(raw, models, fw, _c, _d)
                    if p is not None:
                        probs_list.append(p)
                if not probs_list:
                    return None
                avg_probs = torch.stack(probs_list).mean(dim=0)
                pred_idx = avg_probs.argmax().item()
                return i2c[pred_idx], avg_probs[pred_idx].item()

            ens_name = "+".join(all_models.keys())
            result_ens = evaluate_method(f"GN_ensemble_{cat_name}", cat_videos, predict_ensemble, classes)
            all_results[f"{cat_name}_ensemble"] = result_ens

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — V34 GroupNorm Ensemble (NO BN adaptation)")
    print(f"{'='*70}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {key}: {r['overall']:.1f}% ({r['correct']}/{r['total']})")

    # Combined accuracy
    for suffix in list(all_models.keys()) + ["ensemble"]:
        n_key = f"numbers_{suffix}"
        w_key = f"words_{suffix}"
        if n_key in all_results and w_key in all_results:
            n_r = all_results[n_key]
            w_r = all_results[w_key]
            combined = 100.0 * (n_r["correct"] + w_r["correct"]) / (n_r["total"] + w_r["total"])
            print(f"  Combined {suffix}: {combined:.1f}%")

    print(f"\n  Reference: v31 exp1 (single GroupNorm) = 61.4%")
    print(f"  Reference: v31 5-model ensemble (BN+AdaBN) = 70.7%")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v34_gn_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "version": "v34_gn_ensemble",
            "description": "GroupNorm multi-seed ensemble (signer-agnostic)",
            "models": model_names,
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
