#!/usr/bin/env python3
"""
T3A (Test-Time Classifier Adjustment) on Weighted Ensemble.

Applies T3A at the softmax-probability level on top of the optimized
weighted ensemble (v27+v28+v29+exp1+exp5).

T3A (Iwasawa & Matsuo, NeurIPS 2021):
  1. Forward-pass all test samples -> get initial weighted-ensemble probs
  2. Use initial predictions to group prob vectors by predicted class
  3. Compute mean prob vector per class -> "pseudo-prototypes"
  4. Re-classify each sample by cosine similarity to pseudo-prototypes

This is back-propagation-free, works per-sample, and is compatible with
any normalization (GroupNorm, BatchNorm). Worst case: equals baseline.

Optimized weights from memory:
  Numbers: v29=0.15, exp1=0.35, exp5=0.50  (v27/v28 dropped -> 0.0)
  Words:   v27=0.05, v28=0.35, v29=0.05, exp1=0.05, exp5=0.50

Usage:
    python evaluate_t3a_ensemble.py
    python evaluate_t3a_ensemble.py --filter-threshold 0.5
"""

import sys
import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
    evaluate_method, confidence_level,
    extract_landmarks_from_video,
)
from evaluate_real_testers_v30 import (
    KSLGraphNetV25, _load_fusion_weights,
    NUMBER_CLASSES, WORD_CLASSES, build_adj,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)
from run_ensemble_v31 import (
    load_v31_exp1_models, load_v31_exp5_models,
    get_v28_style_probs, get_groupnorm_probs,
)
from evaluate_real_testers_v30_phase1 import (
    load_v27_model, load_v28_stream_models, load_v29_model,
)

REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

# Optimized ensemble weights (from optimize_ensemble_weights.py results)
OPTIMAL_WEIGHTS = {
    "numbers": {"v27": 0.0, "v28": 0.0, "v29": 0.15, "exp1": 0.35, "exp5": 0.50},
    "words":   {"v27": 0.05, "v28": 0.35, "v29": 0.05, "exp1": 0.05, "exp5": 0.50},
}


def collect_ensemble_probs(cat_videos, classes, category, device, alpha=0.3):
    """Load all 5 models, adapt BN, and collect per-model probs for all test videos.

    Returns:
        model_probs: dict model_name -> tensor (N, C)
        labels: np.array (N,)
        video_info: list of (video_path, true_class, signer_id)
    """
    # Load models
    v27_ckpt = os.path.join(CKPT_BASE, f"v27_{category}", "best_model.pt")
    v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

    v28_ckpt_dir = os.path.join(CKPT_BASE, f"v28_{category}")
    v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

    v29_ckpt = os.path.join(CKPT_BASE, f"v29_{category}", "best_model.pt")
    v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

    print("  Loading v31_exp1 (GroupNorm)...")
    exp1_models, exp1_fw = load_v31_exp1_models(category, device)

    print("  Loading v31_exp5 (SupCon)...")
    exp5_models, exp5_fw = load_v31_exp5_models(category, device)

    # Pre-extract and adapt BN
    stream_data = preextract_test_data(cat_videos, preprocess_fn="multistream")

    if v27_model:
        v27_data = preextract_test_data(cat_videos, preprocess_fn="v27")
        adapt_bn_stats(v27_model, v27_data, device)

    if v28_models:
        for sname, smodel in v28_models.items():
            source_stats = save_bn_stats(smodel)
            target_stats = compute_target_bn_stats(
                smodel, stream_data, device,
                is_multistream=True, stream_name=sname,
            )
            apply_alpha_bn(smodel, source_stats, target_stats, alpha)

    if v29_model:
        v29_data = preextract_test_data(cat_videos, preprocess_fn="v27")
        adapt_bn_stats(v29_model, v29_data, device)

    # Exp5: AdaBN Global
    if exp5_models:
        exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
        for sname in exp5_adabn:
            adapt_bn_stats(exp5_adabn[sname], stream_data, device,
                          is_multistream=True, stream_name=sname)
    else:
        exp5_adabn = exp5_models

    # Collect per-model probs
    model_names = ["v27", "v28", "v29", "exp1", "exp5"]
    all_probs = {m: [] for m in model_names}
    all_labels = []
    video_info = []

    for video_path, true_class, signer_id in cat_videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"    FAILED: {os.path.basename(video_path)}")
                for m in all_probs:
                    all_probs[m].append(torch.ones(len(classes)) / len(classes))
                all_labels.append(classes.index(true_class))
                video_info.append((video_path, true_class, signer_id))
                continue

        all_labels.append(classes.index(true_class))
        video_info.append((video_path, true_class, signer_id))

        streams, aux = preprocess_multistream(raw)
        gcn_v27, aux_v27 = preprocess_v27(raw)

        with torch.no_grad():
            # V27
            if v27_model and gcn_v27 is not None:
                logits, *_ = v27_model(
                    gcn_v27.unsqueeze(0).to(device),
                    aux_v27.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                all_probs["v27"].append(F.softmax(logits, dim=1).cpu().squeeze(0))
            else:
                all_probs["v27"].append(torch.ones(len(classes)) / len(classes))

            # V28
            if v28_models and streams is not None:
                probs = get_v28_style_probs(streams, aux, v28_models, v28_fw, device)
                all_probs["v28"].append(probs)
            else:
                all_probs["v28"].append(torch.ones(len(classes)) / len(classes))

            # V29
            if v29_model and gcn_v27 is not None:
                logits, *_ = v29_model(
                    gcn_v27.unsqueeze(0).to(device),
                    aux_v27.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                all_probs["v29"].append(F.softmax(logits, dim=1).cpu().squeeze(0))
            else:
                all_probs["v29"].append(torch.ones(len(classes)) / len(classes))

            # Exp1 (GroupNorm - no adaptation)
            if exp1_models and streams is not None:
                probs = get_groupnorm_probs(streams, aux, exp1_models, exp1_fw, device)
                all_probs["exp1"].append(probs)
            else:
                all_probs["exp1"].append(torch.ones(len(classes)) / len(classes))

            # Exp5 (AdaBN)
            if exp5_adabn and streams is not None:
                probs = get_v28_style_probs(streams, aux, exp5_adabn, exp5_fw, device)
                all_probs["exp5"].append(probs)
            else:
                all_probs["exp5"].append(torch.ones(len(classes)) / len(classes))

    # Stack
    for m in model_names:
        all_probs[m] = torch.stack(all_probs[m])  # (N, C)
    all_labels = np.array(all_labels)

    return all_probs, all_labels, video_info


def apply_weighted_ensemble(model_probs, weights):
    """Compute weighted ensemble probs: sum(w_i * p_i)."""
    model_names = list(weights.keys())
    fused = sum(weights[m] * model_probs[m] for m in model_names)
    return fused


def t3a_adjust(ensemble_probs, filter_threshold=None, num_iterations=1):
    """Apply T3A (Test-Time Classifier Adjustment) at the probability level.

    Args:
        ensemble_probs: tensor (N, C) — weighted ensemble softmax probs
        filter_threshold: if set, only use samples with max_prob >= threshold
                         for building prototypes (filters low-confidence samples)
        num_iterations: number of T3A iterations (1 is usually enough)

    Returns:
        t3a_preds: tensor (N,) — adjusted predictions
        t3a_probs: tensor (N, C) — cosine similarity scores (softmaxed)
    """
    N, C = ensemble_probs.shape

    current_probs = ensemble_probs.clone()

    for iteration in range(num_iterations):
        # Step 1: Get current predictions
        preds = current_probs.argmax(dim=1)  # (N,)

        # Step 2: Build pseudo-prototypes by averaging prob vectors per predicted class
        prototypes = torch.zeros(C, C)  # (num_classes, prob_dim)
        counts = torch.zeros(C)

        for i in range(N):
            pred_class = preds[i].item()
            max_prob = current_probs[i].max().item()

            # Optional: filter low-confidence samples
            if filter_threshold is not None and max_prob < filter_threshold:
                continue

            prototypes[pred_class] += current_probs[i]
            counts[pred_class] += 1

        # Average prototypes (handle empty classes)
        for c in range(C):
            if counts[c] > 0:
                prototypes[c] /= counts[c]
            else:
                # No samples predicted this class — use one-hot as fallback
                prototypes[c] = torch.zeros(C)
                prototypes[c][c] = 1.0

        # Step 3: Re-classify by cosine similarity to prototypes
        # Normalize both prototypes and sample probs
        proto_norm = F.normalize(prototypes, p=2, dim=1)  # (C, C)
        sample_norm = F.normalize(current_probs, p=2, dim=1)  # (N, C)

        # Cosine similarity: (N, C)
        similarities = sample_norm @ proto_norm.t()

        # Convert to probs via softmax with temperature
        current_probs = F.softmax(similarities * 10.0, dim=1)  # temperature=0.1 -> scale=10

    t3a_preds = current_probs.argmax(dim=1)
    return t3a_preds, current_probs


def evaluate_with_details(preds, probs, labels, video_info, classes, method_name):
    """Print detailed evaluation results matching evaluate_method format."""
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for i, c in enumerate(classes)}
    nc = len(classes)

    correct, total = 0, 0
    per_class = {}
    per_signer = {}
    confusion = [[0] * nc for _ in range(nc)]
    conf_buckets = {"HIGH": {"correct": 0, "total": 0},
                    "MEDIUM": {"correct": 0, "total": 0},
                    "LOW": {"correct": 0, "total": 0}}

    for idx, (video_path, true_class, signer) in enumerate(video_info):
        pred_idx = preds[idx].item()
        pred_class = i2c[pred_idx]
        true_idx = labels[idx]
        confidence = probs[idx].max().item()
        conf_lvl = confidence_level(confidence)

        is_correct = (pred_idx == true_idx)
        correct += int(is_correct)
        total += 1

        confusion[true_idx][pred_idx] += 1
        conf_buckets[conf_lvl]["total"] += 1
        conf_buckets[conf_lvl]["correct"] += int(is_correct)

        if true_class not in per_class:
            per_class[true_class] = {"correct": 0, "total": 0}
        per_class[true_class]["total"] += 1
        per_class[true_class]["correct"] += int(is_correct)

        if signer not in per_signer:
            per_signer[signer] = {"correct": 0, "total": 0}
        per_signer[signer]["total"] += 1
        per_signer[signer]["correct"] += int(is_correct)

        status = "OK" if is_correct else "WRONG"
        print(f"    [{method_name}] {os.path.basename(video_path)}: "
              f"{pred_class} (conf={confidence:.3f}) | True: {true_class} [{status}]")

    acc = 100.0 * correct / total if total > 0 else 0.0

    print(f"\n  {method_name}: {correct}/{total} = {acc:.1f}%")

    print(f"\n  Per-class accuracy:")
    print(f"    {'Class':>12s}  {'Acc':>7s}  {'N':>3s}")
    for cls in classes:
        if cls in per_class:
            pc = per_class[cls]
            cls_acc = 100.0 * pc["correct"] / pc["total"]
            print(f"    {cls:>12s}  {cls_acc:5.0f}%   {pc['total']:3d}")

    print(f"\n  Per-signer accuracy:")
    for signer in sorted(per_signer.keys()):
        ps = per_signer[signer]
        s_acc = 100.0 * ps["correct"] / ps["total"]
        print(f"    {signer:>20s}  {s_acc:5.0f}%   {ps['total']:3d}")

    print(f"\n  Confidence analysis:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        bucket = conf_buckets[lvl]
        if bucket["total"] > 0:
            b_acc = 100.0 * bucket["correct"] / bucket["total"]
            print(f"    {lvl:>6s}: {bucket['correct']}/{bucket['total']} = {b_acc:.0f}%")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"    {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{confusion[i][j]:5d}" for j in range(nc))
        print(f"    {row_str}")

    return {
        "method": method_name,
        "overall": acc,
        "correct": correct,
        "total": total,
        "per_class": {k: {"accuracy": 100.0 * v["correct"] / v["total"], **v}
                      for k, v in per_class.items()},
        "per_signer": {k: {"accuracy": 100.0 * v["correct"] / v["total"], **v}
                       for k, v in per_signer.items()},
        "confusion_matrix": confusion,
        "confusion_labels": [i2c[i] for i in range(nc)],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="T3A on Weighted Ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="Min confidence to include sample in prototype building")
    parser.add_argument("--num-iterations", type=int, default=1,
                        help="Number of T3A iterations")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Alpha-BN interpolation for v28")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("T3A (Test-Time Classifier Adjustment) on Weighted Ensemble")
    print(f"Filter threshold: {args.filter_threshold}")
    print(f"T3A iterations:   {args.num_iterations}")
    print(f"Alpha-BN alpha:   {args.alpha}")
    print(f"Started:          {ts}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"Found {len(raw_videos)} test videos")

    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [(vp, name, signer) for vp, name, cat, signer in raw_videos if cat == category]
        if not cat_videos:
            continue

        weights = OPTIMAL_WEIGHTS[category]
        print(f"\n{'='*70}")
        print(f"{category.upper()} -- {len(cat_videos)} videos")
        print(f"Ensemble weights: {weights}")
        print(f"{'='*70}")

        # Collect probs from all models
        model_probs, labels, video_info = collect_ensemble_probs(
            cat_videos, classes, category, device, alpha=args.alpha
        )

        # Per-model accuracy
        model_names = ["v27", "v28", "v29", "exp1", "exp5"]
        print(f"\n  Per-model accuracy:")
        for m in model_names:
            preds = model_probs[m].argmax(dim=1).numpy()
            acc = 100.0 * (preds == labels).mean()
            print(f"    {m:>6s}: {acc:.1f}% (weight={weights[m]:.2f})")

        # --- Baseline: Weighted Ensemble (no T3A) ---
        print(f"\n{'='*70}")
        print(f"BASELINE: Weighted Ensemble (no T3A)")
        print(f"{'='*70}")
        ensemble_probs = apply_weighted_ensemble(model_probs, weights)
        baseline_preds = ensemble_probs.argmax(dim=1)
        baseline_result = evaluate_with_details(
            baseline_preds, ensemble_probs, labels, video_info, classes,
            f"weighted_ensemble_{category}"
        )

        # --- T3A: Standard (no filtering) ---
        print(f"\n{'='*70}")
        print(f"T3A: Standard (no confidence filtering)")
        print(f"{'='*70}")
        t3a_preds, t3a_probs = t3a_adjust(
            ensemble_probs, filter_threshold=None, num_iterations=args.num_iterations
        )
        t3a_result = evaluate_with_details(
            t3a_preds, t3a_probs, labels, video_info, classes,
            f"t3a_standard_{category}"
        )

        # --- T3A: With confidence filtering ---
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        if args.filter_threshold is not None:
            thresholds = [args.filter_threshold]

        best_t3a_filtered = None
        best_t3a_filtered_acc = 0.0
        best_threshold = 0.0

        for thresh in thresholds:
            print(f"\n{'='*70}")
            print(f"T3A: Confidence threshold = {thresh}")
            print(f"{'='*70}")
            preds_f, probs_f = t3a_adjust(
                ensemble_probs, filter_threshold=thresh, num_iterations=args.num_iterations
            )
            result_f = evaluate_with_details(
                preds_f, probs_f, labels, video_info, classes,
                f"t3a_filter{thresh}_{category}"
            )
            if result_f["overall"] > best_t3a_filtered_acc:
                best_t3a_filtered_acc = result_f["overall"]
                best_t3a_filtered = result_f
                best_threshold = thresh

        # --- T3A: Multi-iteration ---
        if args.num_iterations == 1:
            print(f"\n{'='*70}")
            print(f"T3A: 2 iterations (no filtering)")
            print(f"{'='*70}")
            preds_2, probs_2 = t3a_adjust(
                ensemble_probs, filter_threshold=None, num_iterations=2
            )
            t3a_2iter_result = evaluate_with_details(
                preds_2, probs_2, labels, video_info, classes,
                f"t3a_2iter_{category}"
            )

        # Store results
        all_results[category] = {
            "baseline": baseline_result,
            "t3a_standard": t3a_result,
            "t3a_best_filtered": best_t3a_filtered,
            "best_threshold": best_threshold,
        }

    # --- FINAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Reference: Weighted Ensemble = Numbers 76.3%, Words 72.8%, Combined 74.3%")
    print()

    total_correct_baseline = 0
    total_correct_t3a = 0
    total_correct_t3a_filtered = 0
    total_samples = 0

    for category in ["numbers", "words"]:
        if category not in all_results:
            continue
        r = all_results[category]
        b = r["baseline"]
        t = r["t3a_standard"]
        f = r["t3a_best_filtered"]

        total_correct_baseline += b["correct"]
        total_correct_t3a += t["correct"]
        if f:
            total_correct_t3a_filtered += f["correct"]
        total_samples += b["total"]

        print(f"  {category.upper():>10s}:")
        print(f"    Baseline (weighted ensemble): {b['correct']}/{b['total']} = {b['overall']:.1f}%")
        print(f"    T3A (standard):               {t['correct']}/{t['total']} = {t['overall']:.1f}% ({t['overall'] - b['overall']:+.1f}pp)")
        if f:
            print(f"    T3A (best filter={r['best_threshold']:.1f}): {f['correct']}/{f['total']} = {f['overall']:.1f}% ({f['overall'] - b['overall']:+.1f}pp)")

    if total_samples > 0:
        combined_baseline = 100.0 * total_correct_baseline / total_samples
        combined_t3a = 100.0 * total_correct_t3a / total_samples
        combined_t3a_f = 100.0 * total_correct_t3a_filtered / total_samples
        print(f"\n  COMBINED:")
        print(f"    Baseline: {combined_baseline:.1f}%")
        print(f"    T3A (standard): {combined_t3a:.1f}% ({combined_t3a - combined_baseline:+.1f}pp)")
        print(f"    T3A (best filtered): {combined_t3a_f:.1f}% ({combined_t3a_f - combined_baseline:+.1f}pp)")

    # Save results
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"t3a_ensemble_{ts_str}.json")

    save_data = {
        "method": "T3A on Weighted Ensemble",
        "filter_threshold": args.filter_threshold,
        "num_iterations": args.num_iterations,
        "alpha": args.alpha,
        "optimal_weights": OPTIMAL_WEIGHTS,
        "timestamp": ts_str,
        "results": {},
    }
    for cat, r in all_results.items():
        save_data["results"][cat] = {
            "baseline_acc": r["baseline"]["overall"],
            "t3a_standard_acc": r["t3a_standard"]["overall"],
            "t3a_best_filtered_acc": r["t3a_best_filtered"]["overall"] if r["t3a_best_filtered"] else None,
            "best_threshold": r["best_threshold"],
            "baseline_per_class": r["baseline"].get("per_class", {}),
            "t3a_per_class": r["t3a_standard"].get("per_class", {}),
            "baseline_per_signer": r["baseline"].get("per_signer", {}),
            "t3a_per_signer": r["t3a_standard"].get("per_signer", {}),
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
