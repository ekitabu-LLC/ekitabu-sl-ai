#!/usr/bin/env python3
"""
V40 Real-Tester Evaluation — VarCL (Variational Contrastive Learning).

V40 uses BatchNorm (not GroupNorm), so AdaBN Global is applied at inference.
At inference, mu (mean of the variational head) is used directly — no sampling.

Usage:
    python evaluate_v40.py --category both
    python evaluate_v40.py --category numbers
    python evaluate_v40.py --category words
"""

import argparse
import copy
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
    discover_test_videos,
    preprocess_multistream,
    preextract_test_data,
    evaluate_method,
    _load_fusion_weights,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

from train_ksl_v40 import KSLGraphNetV40, build_adj


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_v40_models(ckpt_dir, classes, device):
    """Load V40 VarCL multi-stream models with BatchNorm."""
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
        num_nodes = config.get("num_nodes", 48)
        num_layers = config.get("num_layers", 4)
        hidden_dim = config.get("hidden_dim", 64)
        temporal_kernels = tuple(config.get("temporal_kernels", [3, 5, 7]))
        dropout = config.get("dropout", 0.3)
        spatial_dropout = config.get("spatial_dropout", 0.1)
        adj = build_adj(num_nodes).to(device)

        model = KSLGraphNetV40(
            nc=len(classes),
            num_signers=num_signers,
            aux_dim=aux_dim,
            proj_dim=proj_dim,
            nn_=num_nodes,
            ic=config.get("in_channels", 3),
            hd=hidden_dim,
            nl=num_layers,
            tk=temporal_kernels,
            dr=dropout,
            spatial_dropout=spatial_dropout,
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


def adapt_bn_stats(model, data_list, device):
    """AdaBN Global: reset BN running stats and re-estimate from test data."""
    if not data_list:
        return
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()


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
    parser = argparse.ArgumentParser(description="V40 Real-Tester Evaluation (VarCL)")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--checkpoint-base", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v40")
    parser.add_argument("--no-adabn", action="store_true",
                        help="Disable AdaBN (test without BN adaptation)")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("V40 Real-Tester Evaluation")
    print("VarCL (Variational Contrastive Learning) + BatchNorm + AdaBN Global")
    print(f"Checkpoint base: {args.checkpoint_base}")
    print(f"Category:        {args.category}")
    print(f"AdaBN:           {'disabled' if args.no_adabn else 'enabled (Global)'}")
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
        print(f"{'='*70}")

        print(f"\n  Loading V40 VarCL models from {ckpt_dir}...")
        models, fusion_weights = load_v40_models(ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  ERROR: Only {len(models)}/3 streams loaded, skipping")
            continue

        print(f"\n  Fusion weights: {fusion_weights}")

        # Pre-extract skeleton data for all test videos
        print(f"\n  Pre-extracting skeleton data from {len(cat_videos)} test videos...")
        all_data = preextract_test_data(cat_videos)
        print(f"  Extracted {len(all_data)} valid samples")

        # AdaBN Global: adapt BN stats using all test data
        if not args.no_adabn and all_data:
            print(f"\n  Applying AdaBN Global (adapting BN stats from {len(all_data)} test samples)...")
            adapted_models = {}
            for sname, model in models.items():
                m_copy = copy.deepcopy(model)
                stream_data = []
                for (streams, aux) in all_data:
                    if streams and sname in streams:
                        stream_data.append((streams[sname], aux))
                adapt_bn_stats(m_copy, stream_data, device)
                adapted_models[sname] = m_copy
            models = adapted_models
            print(f"  AdaBN Global applied.")

        predict_fn = make_predict_fn(models, fusion_weights, classes, device)
        result = evaluate_method(f"v40_{cat_name}", cat_videos, predict_fn, classes)
        all_results[cat_name] = result

    # --- Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - V40 VarCL")
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
    os.makedirs(os.path.join(results_dir, "v40"), exist_ok=True)
    out_path = os.path.join(results_dir, "v40", "real_tester_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
