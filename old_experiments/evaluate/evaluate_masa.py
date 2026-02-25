#!/usr/bin/env python3
"""
MASA Real-Tester Evaluation.

GroupNorm classification head — no AdaBN needed at inference.

Usage:
    python evaluate_masa.py --category both
    python evaluate_masa.py --category numbers
    python evaluate_masa.py --category words
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
    preextract_test_data,
    evaluate_method,
)

from train_ksl_masa import MASAClassifier, adapt_joints_to_masa, MASA_CONFIG


def load_masa_model(ckpt_dir, classes, device):
    """Load MASA model from checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Missing checkpoint: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", MASA_CONFIG)

    model = MASAClassifier(
        num_classes=len(classes),
        d_model=1536, nhead=8, d_ff=2048,
        num_blocks=3, dropout=config.get("dropout", 0.1),
        cls_dropout=config.get("cls_dropout", 0.3),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model: val_acc={ckpt.get('val_acc', 0):.1f}%, "
          f"epoch={ckpt.get('epoch', '?')}, params={param_count:,}")
    return model


def make_predict_fn(model, classes, device):
    """Create prediction function compatible with evaluate_method."""
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        # preprocess_multistream returns (streams_dict, aux_tensor) or (None, _)
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            return None

        # Get joint positions from the 'joint' stream: (C=3, T=90, N=48)
        joint_data = streams["joint"]  # (3, T, 48)
        T = joint_data.shape[1]

        # Convert to (T, 48, 3)
        h = joint_data.permute(1, 2, 0).numpy()  # (T, 48, 3)

        # Adapt to MASA format
        rh, lh, body = adapt_joints_to_masa(h)

        with torch.no_grad():
            rh_t = torch.FloatTensor(rh).unsqueeze(0).to(device)  # (1, T, 21, 2)
            lh_t = torch.FloatTensor(lh).unsqueeze(0).to(device)
            body_t = torch.FloatTensor(body).unsqueeze(0).to(device)
            logits, _ = model(rh_t, lh_t, body_t)
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)

        pred_idx = probs.argmax().item()
        return i2c[pred_idx], probs[pred_idx].item()

    return predict


def main():
    parser = argparse.ArgumentParser(description="MASA Real-Tester Evaluation")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--checkpoint-base", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/masa")
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    results_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("MASA Real-Tester Evaluation")
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

        print(f"\n  Loading model from {ckpt_dir}...")
        model = load_masa_model(ckpt_dir, classes, device)

        if model is None:
            print(f"  ERROR: Could not load model, skipping")
            continue

        _ = preextract_test_data(cat_videos)

        predict_fn = make_predict_fn(model, classes, device)
        result = evaluate_method(f"masa_{cat_name}", cat_videos, predict_fn, classes)
        all_results[cat_name] = result

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - MASA")
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

    # Save
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"masa_eval_{ts_str}.json")
    save_data = {
        "version": "masa",
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
