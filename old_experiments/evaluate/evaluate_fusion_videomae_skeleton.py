#!/usr/bin/env python3
"""
VideoMAE + Skeleton Ensemble Fusion Evaluation.

Loads the 5-model skeleton ensemble (v27+v28+v29+exp1+exp5) with optimized weights,
plus VideoMAE model, and fuses at the softmax probability level.

Grid-searches alpha in [0.3, 0.8] for:
    final_probs = alpha * skeleton_probs + (1 - alpha) * videomae_probs

Also tests 6-model uniform ensemble (all skeleton models + VideoMAE).

Usage:
    python evaluate_fusion_videomae_skeleton.py
    python evaluate_fusion_videomae_skeleton.py --alpha-step 0.05
    python evaluate_fusion_videomae_skeleton.py --videomae-ckpt-dir data/checkpoints/videomae_v2
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Skeleton pipeline imports
from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
)
from evaluate_real_testers_v30 import (
    KSLGraphNetV25, _load_fusion_weights, extract_landmarks_from_video,
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

# VideoMAE imports
from transformers import VideoMAEForVideoClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
BASE_DIR = "/scratch/alpine/hama5612/ksl-dir-2"
CKPT_BASE = os.path.join(BASE_DIR, "data", "checkpoints")
VIDEOMAE_CKPT_DIR = os.path.join(CKPT_BASE, "videomae")
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

# Best skeleton ensemble weights (from optimize_ensemble_weights.py)
BEST_SKELETON_WEIGHTS = {
    "numbers": {"v27": 0.0, "v28": 0.0, "v29": 0.15, "exp1": 0.35, "exp5": 0.50},
    "words": {"v27": 0.05, "v28": 0.35, "v29": 0.05, "exp1": 0.05, "exp5": 0.50},
}

NUM_FRAMES = 16
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# VideoMAE video loading (same as evaluate_videomae.py)
# ---------------------------------------------------------------------------
def load_video_frames(video_path, num_frames=NUM_FRAMES, size=IMAGE_SIZE):
    """Load video, sample num_frames uniformly, center crop to size x size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return None

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')

    sampled = [frames[i] for i in indices]

    processed = []
    for frame in sampled:
        crop_size = int(size * 1.15)
        frame = cv2.resize(frame, (crop_size, crop_size))
        off = (crop_size - size) // 2
        frame = frame[off:off + size, off:off + size]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        frame = frame.transpose(2, 0, 1)
        processed.append(frame)

    return np.stack(processed)


# ---------------------------------------------------------------------------
# Load VideoMAE model
# ---------------------------------------------------------------------------
def load_videomae_model(category, device, ckpt_dir=None):
    """Load VideoMAE checkpoint for the given category."""
    base = ckpt_dir if ckpt_dir is not None else VIDEOMAE_CKPT_DIR
    ckpt_path = os.path.join(base, category, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{ts()}] ERROR: VideoMAE checkpoint not found: {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[{ts()}] Loaded VideoMAE {category}: epoch={ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.1f}%, classes={num_classes}")
    return model, classes


def get_videomae_probs(model, video_path, device):
    """Get softmax probabilities from VideoMAE for a single video."""
    frames = load_video_frames(video_path)
    if frames is None:
        return None

    frames_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=frames_t)
        probs = F.softmax(outputs.logits, dim=1).cpu().squeeze(0)
    return probs


# ---------------------------------------------------------------------------
# Collect all probabilities (skeleton + VideoMAE) for all test videos
# ---------------------------------------------------------------------------
def collect_all_probs(cat_videos, classes, category, device, alpha_bn=0.3, videomae_ckpt_dir=None):
    """Collect probability vectors from skeleton ensemble + VideoMAE."""
    # --- Skeleton models ---
    print(f"\n[{ts()}] Loading skeleton models for {category}...")

    v27_ckpt = os.path.join(CKPT_BASE, f"v27_{category}", "best_model.pt")
    v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

    v28_ckpt_dir = os.path.join(CKPT_BASE, f"v28_{category}")
    v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

    v29_ckpt = os.path.join(CKPT_BASE, f"v29_{category}", "best_model.pt")
    v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

    print(f"[{ts()}] Loading v31_exp1 (GroupNorm)...")
    exp1_models, exp1_fw = load_v31_exp1_models(category, device)

    print(f"[{ts()}] Loading v31_exp5 (SupCon)...")
    exp5_models, exp5_fw = load_v31_exp5_models(category, device)

    # --- VideoMAE ---
    print(f"[{ts()}] Loading VideoMAE {category}...")
    videomae_model, videomae_classes = load_videomae_model(category, device, videomae_ckpt_dir)

    # --- Adapt BN stats for BN-based skeleton models ---
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
            apply_alpha_bn(smodel, source_stats, target_stats, alpha_bn)

    if v29_model:
        v29_data = preextract_test_data(cat_videos, preprocess_fn="v27")
        adapt_bn_stats(v29_model, v29_data, device)

    if exp5_models:
        exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
        for sname in exp5_adabn:
            adapt_bn_stats(exp5_adabn[sname], stream_data, device,
                          is_multistream=True, stream_name=sname)
    else:
        exp5_adabn = exp5_models

    # --- Collect per-video probabilities ---
    skeleton_names = ["v27", "v28", "v29", "exp1", "exp5"]
    all_model_names = skeleton_names + ["videomae"]
    all_probs = {m: [] for m in all_model_names}
    all_labels = []

    uniform_class_prob = torch.ones(len(classes)) / len(classes)

    print(f"\n[{ts()}] Evaluating {len(cat_videos)} videos...")
    for i, (video_path, true_class, signer_id) in enumerate(cat_videos):
        # --- Skeleton ---
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                print(f"  FAILED landmarks: {os.path.basename(video_path)}")
                for m in all_model_names:
                    all_probs[m].append(uniform_class_prob)
                all_labels.append(classes.index(true_class))
                continue

        all_labels.append(classes.index(true_class))
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
                all_probs["v27"].append(uniform_class_prob)

            # V28
            if v28_models and streams is not None:
                probs = get_v28_style_probs(streams, aux, v28_models, v28_fw, device)
                all_probs["v28"].append(probs)
            else:
                all_probs["v28"].append(uniform_class_prob)

            # V29
            if v29_model and gcn_v27 is not None:
                logits, *_ = v29_model(
                    gcn_v27.unsqueeze(0).to(device),
                    aux_v27.unsqueeze(0).to(device),
                    grl_lambda=0.0,
                )
                all_probs["v29"].append(F.softmax(logits, dim=1).cpu().squeeze(0))
            else:
                all_probs["v29"].append(uniform_class_prob)

            # Exp1 (GroupNorm)
            if exp1_models and streams is not None:
                probs = get_groupnorm_probs(streams, aux, exp1_models, exp1_fw, device)
                all_probs["exp1"].append(probs)
            else:
                all_probs["exp1"].append(uniform_class_prob)

            # Exp5 (SupCon + AdaBN)
            if exp5_adabn and streams is not None:
                probs = get_v28_style_probs(streams, aux, exp5_adabn, exp5_fw, device)
                all_probs["exp5"].append(probs)
            else:
                all_probs["exp5"].append(uniform_class_prob)

        # --- VideoMAE ---
        if videomae_model is not None:
            vprobs = get_videomae_probs(videomae_model, video_path, device)
            if vprobs is not None:
                all_probs["videomae"].append(vprobs)
            else:
                all_probs["videomae"].append(uniform_class_prob)
        else:
            all_probs["videomae"].append(uniform_class_prob)

        if (i + 1) % 20 == 0:
            print(f"  [{ts()}] Processed {i+1}/{len(cat_videos)} videos")

    # Stack
    for m in all_model_names:
        all_probs[m] = torch.stack(all_probs[m])  # (N, C)
    all_labels = np.array(all_labels)

    return all_probs, all_labels


# ---------------------------------------------------------------------------
# Fusion and evaluation
# ---------------------------------------------------------------------------
def compute_accuracy(probs, labels):
    """Compute accuracy from probability tensor and label array."""
    preds = probs.argmax(dim=1).numpy()
    return 100.0 * (preds == labels).mean()


def evaluate_alpha_fusion(skeleton_probs, videomae_probs, labels, alphas):
    """Grid search alpha: final = alpha * skeleton + (1-alpha) * videomae."""
    results = {}
    for alpha in alphas:
        fused = alpha * skeleton_probs + (1 - alpha) * videomae_probs
        acc = compute_accuracy(fused, labels)
        results[alpha] = acc
    return results


def evaluate_6model_uniform(all_probs, labels, model_names):
    """Uniform average across all 6 models."""
    fused = sum(all_probs[m] for m in model_names) / len(model_names)
    return compute_accuracy(fused, labels)


def evaluate_6model_grid_search(all_probs, labels, model_names, step=0.1):
    """Grid search optimal weights for 6 models (coarse)."""
    steps = np.arange(0.0, 1.0 + step, step)
    n_models = len(model_names)

    best_acc = 0.0
    best_weights = {m: 1.0 / n_models for m in model_names}

    for w in product(steps, repeat=n_models):
        if abs(sum(w) - 1.0) > 0.01:
            continue
        weights = {m: wi for m, wi in zip(model_names, w)}
        fused = sum(weights[m] * all_probs[m] for m in model_names)
        acc = compute_accuracy(fused, labels)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights

    return best_weights, best_acc


def main():
    parser = argparse.ArgumentParser(description="VideoMAE + Skeleton Fusion Evaluation")
    parser.add_argument("--alpha-step", type=float, default=0.05,
                        help="Step size for alpha grid search")
    parser.add_argument("--alpha-bn", type=float, default=0.3,
                        help="Alpha-BN interpolation for BN skeleton models")
    parser.add_argument("--videomae-ckpt-dir", type=str, default=VIDEOMAE_CKPT_DIR,
                        help="VideoMAE checkpoint directory (default: data/checkpoints/videomae)")
    args = parser.parse_args()

    print("=" * 70)
    print("VideoMAE + Skeleton Ensemble Fusion Evaluation")
    print(f"Started: {ts()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Discover test videos
    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"Found {len(raw_videos)} test videos")

    # Alpha values to test
    alphas = np.arange(0.3, 0.85, args.alpha_step).tolist()
    skeleton_names = ["v27", "v28", "v29", "exp1", "exp5"]
    all_model_names = skeleton_names + ["videomae"]

    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [(vp, name, signer) for vp, name, cat, signer in raw_videos if cat == category]
        if not cat_videos:
            continue

        print(f"\n{'='*70}")
        print(f"{category.upper()} — {len(cat_videos)} videos")
        print(f"{'='*70}")

        # Collect all probs
        all_probs, labels = collect_all_probs(cat_videos, classes, category, device,
                                               alpha_bn=args.alpha_bn,
                                               videomae_ckpt_dir=args.videomae_ckpt_dir)

        # --- Per-model accuracy ---
        print(f"\n  Per-model accuracy:")
        for m in all_model_names:
            acc = compute_accuracy(all_probs[m], labels)
            print(f"    {m:>10s}: {acc:.1f}%")

        # --- Skeleton-only weighted ensemble (reference: 74.3%) ---
        skel_weights = BEST_SKELETON_WEIGHTS[category]
        skeleton_fused = sum(skel_weights[m] * all_probs[m] for m in skeleton_names)
        skel_acc = compute_accuracy(skeleton_fused, labels)
        print(f"\n  Skeleton weighted ensemble: {skel_acc:.1f}%")
        print(f"    Weights: {skel_weights}")

        # --- Alpha fusion: skeleton_weighted + VideoMAE ---
        print(f"\n  Alpha fusion (skeleton_weighted + VideoMAE):")
        alpha_results = evaluate_alpha_fusion(skeleton_fused, all_probs["videomae"],
                                               labels, alphas)
        best_alpha = max(alpha_results, key=alpha_results.get)
        for alpha in sorted(alpha_results.keys()):
            acc = alpha_results[alpha]
            marker = " <-- BEST" if alpha == best_alpha else ""
            print(f"    alpha={alpha:.2f}: {acc:.1f}%{marker}")

        print(f"  Best alpha={best_alpha:.2f}: {alpha_results[best_alpha]:.1f}% "
              f"(vs skeleton-only: {skel_acc:.1f}%, "
              f"delta={alpha_results[best_alpha] - skel_acc:+.1f}pp)")

        # --- 6-model uniform ensemble ---
        uni6_acc = evaluate_6model_uniform(all_probs, labels, all_model_names)
        print(f"\n  6-model uniform ensemble: {uni6_acc:.1f}%")

        # --- 6-model grid search (coarse step=0.1) ---
        # Too expensive for 6 models — use step=0.2
        print(f"\n  6-model grid search (step=0.2)...")
        best_w6, best_a6 = evaluate_6model_grid_search(all_probs, labels,
                                                         all_model_names, step=0.2)
        print(f"  6-model optimized: {best_a6:.1f}%")
        print(f"    Weights: {', '.join(f'{m}={w:.1f}' for m, w in best_w6.items())}")

        all_results[category] = {
            "n_videos": len(cat_videos),
            "per_model_acc": {m: float(compute_accuracy(all_probs[m], labels))
                             for m in all_model_names},
            "skeleton_weighted_acc": float(skel_acc),
            "skeleton_weights": skel_weights,
            "alpha_fusion": {str(a): float(v) for a, v in alpha_results.items()},
            "best_alpha": float(best_alpha),
            "best_alpha_acc": float(alpha_results[best_alpha]),
            "uniform_6model_acc": float(uni6_acc),
            "optimized_6model_acc": float(best_a6),
            "optimized_6model_weights": {k: float(v) for k, v in best_w6.items()},
        }

    # --- Overall Summary ---
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")

    for cat, r in all_results.items():
        print(f"\n  {cat.upper()} ({r['n_videos']} videos):")
        print(f"    VideoMAE standalone:     {r['per_model_acc']['videomae']:.1f}%")
        print(f"    Skeleton weighted:       {r['skeleton_weighted_acc']:.1f}%")
        print(f"    Alpha fusion (best):     {r['best_alpha_acc']:.1f}% (alpha={r['best_alpha']:.2f})")
        print(f"    6-model uniform:         {r['uniform_6model_acc']:.1f}%")
        print(f"    6-model optimized:       {r['optimized_6model_acc']:.1f}%")

    # Combined accuracy
    if "numbers" in all_results and "words" in all_results:
        n_r = all_results["numbers"]
        w_r = all_results["words"]
        n_total = n_r["n_videos"]
        w_total = w_r["n_videos"]

        def combined(n_acc, w_acc):
            n_correct = round(n_acc * n_total / 100)
            w_correct = round(w_acc * w_total / 100)
            return 100.0 * (n_correct + w_correct) / (n_total + w_total)

        print(f"\n  COMBINED:")
        print(f"    VideoMAE standalone:     {combined(n_r['per_model_acc']['videomae'], w_r['per_model_acc']['videomae']):.1f}%")
        print(f"    Skeleton weighted:       {combined(n_r['skeleton_weighted_acc'], w_r['skeleton_weighted_acc']):.1f}%")
        print(f"    Alpha fusion (best):     {combined(n_r['best_alpha_acc'], w_r['best_alpha_acc']):.1f}%")
        print(f"    6-model uniform:         {combined(n_r['uniform_6model_acc'], w_r['uniform_6model_acc']):.1f}%")
        print(f"    6-model optimized:       {combined(n_r['optimized_6model_acc'], w_r['optimized_6model_acc']):.1f}%")
        print(f"\n    Reference: current best = 74.3% (skeleton-only weighted ensemble)")

    # Save results
    out_dir = os.path.join(BASE_DIR, "data", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fusion_videomae_skeleton_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{ts()}] Results saved to {out_path}")
    print(f"[{ts()}] Done!")


if __name__ == "__main__":
    main()
