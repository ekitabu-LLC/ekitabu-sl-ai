#!/usr/bin/env python3
"""
OpenHands DecoupledGCN Real-Tester Evaluation.

Evaluates the fine-tuned OpenHands model on the same 140 real-tester videos
used by all other skeleton models. Also saves per-sample logits for ensemble
weight optimization.

Usage:
    python evaluate_openhands_realtest.py --category both
    python evaluate_openhands_realtest.py --category numbers
"""

import argparse
import copy
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_real_testers_v30_phase1 import (
    discover_test_videos,
    evaluate_method,
    check_mediapipe_version,
)
from evaluate_real_testers_v30 import NUMBER_CLASSES, WORD_CLASSES
from train_ksl_openhands import (
    OpenHandsClassifier,
    OH_CONFIG,
    POSE_INDICES,
    normalize_wrist_palm,
    adapt_48_to_27,
)

REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/openhands"
RESULTS_DIR = "/scratch/alpine/hama5612/ksl-dir-2/data/results"


def preprocess_raw_for_openhands(raw, max_frames=90):
    """Convert raw (T, 549) MediaPipe landmarks to OpenHands (2, T, 27) tensor.

    Mirrors KSLOpenHandsDataset.__getitem__ logic (without augmentation).
    """
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None

    # Parse into 48-joint format
    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        pose[:, pi, :] = raw[:, idx_pose * 3:idx_pose * 3 + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # Normalize
    h = normalize_wrist_palm(h)

    # Temporal resampling
    f = h.shape[0]
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h = h[indices]
    else:
        pad = max_frames - f
        h = np.concatenate([h, np.zeros((pad, 48, 3), dtype=np.float32)])

    # Clip and convert to 27-joint 2D
    h = np.clip(h, -10, 10)
    oh27 = adapt_48_to_27(h)  # (T, 27, 2)
    oh27 = np.clip(oh27, -10, 10)

    # (2, T, 27)
    x = torch.FloatTensor(oh27).permute(2, 0, 1)
    return x


def load_openhands_model(ckpt_path, num_classes, device):
    """Load a trained OpenHands checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", OH_CONFIG)

    model = OpenHandsClassifier(
        num_classes=num_classes,
        in_channels=config.get("in_channels", 2),
        num_nodes=config.get("num_nodes", 27),
        n_out_features=config.get("n_out_features", 256),
        cls_dropout=config.get("cls_dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: val_acc={ckpt.get('val_acc', 0):.1f}%, "
          f"epoch={ckpt.get('epoch', '?')}, params={param_count:,}")
    return model


def make_predict_fn(model, classes, device, logits_store=None):
    """Create prediction function compatible with evaluate_method.

    predict_fn(raw, true_class) -> (predicted_class, confidence)

    If logits_store is a list, appends raw logits for each sample.
    """
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        x = preprocess_raw_for_openhands(raw)
        if x is None:
            return None

        with torch.no_grad():
            logits, _ = model(x.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)

            if logits_store is not None:
                logits_store.append(probs.numpy().copy())

            pred_idx = probs.argmax().item()
            return i2c[pred_idx], probs[pred_idx].item()

    return predict


def apply_adabn_global(model, raw_list, device, max_frames=90):
    """Apply AdaBN: reset BN running stats, forward all test samples in train mode."""
    model_adabn = copy.deepcopy(model)
    # Reset all BN layers
    for m in model_adabn.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average
            m.training = True
    model_adabn.eval()
    # Keep BN in train mode for stat update
    for m in model_adabn.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.training = True
    print(f"  [AdaBN] Adapting BN stats with {len(raw_list)} test samples...")
    with torch.no_grad():
        for raw in raw_list:
            if raw is None:
                continue
            x = preprocess_raw_for_openhands(raw, max_frames)
            if x is None:
                continue
            model_adabn(x.unsqueeze(0).to(device))
    # Switch BN back to eval
    for m in model_adabn.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.training = False
    return model_adabn


def main():
    parser = argparse.ArgumentParser(description="OpenHands Real-Tester Evaluation")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--checkpoint-base", type=str, default=CKPT_BASE)
    parser.add_argument("--save-logits", action="store_true", default=True,
                        help="Save per-sample logits for ensemble optimization")
    parser.add_argument("--no-adabn", action="store_true",
                        help="Skip AdaBN adaptation (OpenHands uses BatchNorm)")
    args = parser.parse_args()

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("OpenHands DecoupledGCN Real-Tester Evaluation")
    print(f"Checkpoint base: {args.checkpoint_base}")
    print(f"Category:        {args.category}")
    print(f"Started:         {ts_start}")
    print("=" * 70)

    check_mediapipe_version()

    print(f"\nScanning {REAL_TESTER_DIR}...")
    videos = discover_test_videos(REAL_TESTER_DIR)
    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
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
    all_logits = {}

    for cat_name, cat_videos, classes in categories:
        if not cat_videos:
            print(f"\nSkipping {cat_name}: no test videos found")
            continue

        ckpt_path = os.path.join(args.checkpoint_base, cat_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {cat_name}: checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} ({len(cat_videos)} test videos)")
        print(f"{'='*70}")

        print(f"\n  Loading OpenHands model from {ckpt_path}...")
        model = load_openhands_model(ckpt_path, len(classes), device)

        if not args.no_adabn:
            # Pre-extract raw data for AdaBN adaptation
            from evaluate_real_testers_v30_phase1 import extract_landmarks_from_video
            print(f"  Pre-extracting test data for AdaBN ({len(cat_videos)} videos)...")
            raw_list = []
            for video_path, _, _ in cat_videos:
                npy_cache = video_path + ".landmarks.npy"
                if os.path.exists(npy_cache):
                    raw = np.load(npy_cache).astype(np.float32)
                else:
                    raw = extract_landmarks_from_video(video_path)
                raw_list.append(raw)
            model = apply_adabn_global(model, raw_list, device)

        logits_list = []
        predict_fn = make_predict_fn(model, classes, device, logits_store=logits_list)
        result = evaluate_method(f"openhands_{cat_name}", cat_videos, predict_fn, classes)
        all_results[cat_name] = result

        # Save logits: list of (video_basename, true_class, signer, probs)
        # logits_list was filled by predict_fn in the same order as cat_videos
        # But evaluate_method may skip some videos; logits_list only has successful ones.
        # We need to match them. evaluate_method's details list has the correspondence.
        if args.save_logits:
            logits_data = {
                "classes": classes,
                "probs": np.array(logits_list),  # (N_success, num_classes)
                "details": result.get("details", []),
            }
            all_logits[cat_name] = logits_data

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - OpenHands DecoupledGCN")
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
            combined_correct += result.get("correct", 0)
            combined_total += n

    if combined_total > 0:
        combined_acc = 100.0 * combined_correct / combined_total
        print(f"  {'COMBINED':>10s}: {combined_acc:.1f}%  ({combined_total} videos)")

    # Save results and logits
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save evaluation results
    out_path = os.path.join(RESULTS_DIR, f"openhands_realtest_{ts_str}.json")
    save_data = {
        "version": "openhands",
        "backbone": "DecoupledGCN",
        "checkpoint_base": args.checkpoint_base,
        "timestamp": ts_str,
        "results": {
            cat: {
                "accuracy": r["overall"],
                "total": r.get("total"),
                "correct": r.get("correct"),
                "per_class": r.get("per_class", {}),
                "per_signer": r.get("per_signer", {}),
            }
            for cat, r in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save logits for ensemble optimization
    if args.save_logits and all_logits:
        logits_dir = os.path.join(RESULTS_DIR, "openhands_logits")
        os.makedirs(logits_dir, exist_ok=True)
        for cat_name, logits_data in all_logits.items():
            np.save(
                os.path.join(logits_dir, f"{cat_name}_probs.npy"),
                logits_data["probs"],
            )
            # Save video order for matching with other models
            with open(os.path.join(logits_dir, f"{cat_name}_details.json"), "w") as f:
                json.dump(logits_data["details"], f, indent=2)
            print(f"Logits saved: {logits_dir}/{cat_name}_probs.npy "
                  f"({logits_data['probs'].shape})")


if __name__ == "__main__":
    main()
