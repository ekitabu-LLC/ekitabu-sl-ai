#!/usr/bin/env python3
"""
Clean Ensemble Weight Optimization.

Two-phase evaluation to avoid test-set overfitting:
  Phase 1 (weight tuning):  run all models on test_alpha (ksl-alpha signers 13-15, npy)
                             grid-search for optimal weights on test_alpha accuracy
  Phase 2 (blind eval):     apply Phase-1 weights to real_testers (actual human videos)
                             → reported as the unbiased final number

This separation ensures the reported accuracy is never used during weight selection.
AdaBN BN adaptation uses only input statistics (no labels), which is acceptable.

Models: v27, v28, v29, exp1 (GroupNorm), exp5 (SupCon), openhands

Usage:
    python optimize_ensemble_clean.py
    python optimize_ensemble_clean.py --step 0.05              # finer grid (slower)
    python optimize_ensemble_clean.py --min_weight 0.05        # prevent model exclusion
    python optimize_ensemble_clean.py --entropy_lambda 0.2     # entropy regularization

Audit finding (2026-02-19):
  Pure accuracy grid search on test_alpha gave weights that catastrophically failed
  on real_testers (66.4% clean < 72.9% uniform) because the grid zeroed out exp5 for
  words (the best words model!) and v27/v28/v29 for numbers. Root cause: test_alpha
  (trained signers 13-15) and real_testers (novice external signers) have different
  model ranking orders. Two fixes:
    --min_weight 0.05   prevents any model from being excluded entirely
    --entropy_lambda    adds entropy regularization to avoid extreme weights
"""

import sys
import os
import copy
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
    extract_landmarks_from_video,
    load_v27_model, load_v28_stream_models, load_v29_model,
)
from evaluate_real_testers_v30 import NUMBER_CLASSES, WORD_CLASSES
from run_ensemble_v31 import (
    load_v31_exp1_models, load_v31_exp5_models,
    get_v28_style_probs, get_groupnorm_probs,
)
from evaluate_openhands_realtest import preprocess_raw_for_openhands, load_openhands_model

TEST_ALPHA_DIR = "/scratch/alpine/hama5612/ksl-dir-2/data/test_alpha"
REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

MODEL_NAMES = ["v27", "v28", "v29", "exp1", "exp5", "openhands"]


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_test_alpha(category, classes):
    """Return list of (npy_path, true_class) from test_alpha for given category."""
    items = []
    for cls in classes:
        cls_dir = os.path.join(TEST_ALPHA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.endswith(".npy"):
                items.append((os.path.join(cls_dir, fn), cls))
    return items


# ---------------------------------------------------------------------------
# Model loading (shared for both phases)
# ---------------------------------------------------------------------------

def load_all_models(category, classes, device):
    """Load all 6 models for a given category."""
    models = {}

    # V27
    v27_ckpt = os.path.join(CKPT_BASE, f"v27_{category}", "best_model.pt")
    models["v27"] = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

    # V28
    v28_ckpt_dir = os.path.join(CKPT_BASE, f"v28_{category}")
    if os.path.isdir(v28_ckpt_dir):
        v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device)
        models["v28"] = (v28_models, v28_fw)
    else:
        models["v28"] = None

    # V29
    v29_ckpt = os.path.join(CKPT_BASE, f"v29_{category}", "best_model.pt")
    models["v29"] = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

    # Exp1 (GroupNorm — no BN adaptation needed)
    print("  Loading exp1 (GroupNorm)...")
    exp1_models, exp1_fw = load_v31_exp1_models(category, device)
    models["exp1"] = (exp1_models, exp1_fw)

    # Exp5 (SupCon — BN, needs AdaBN)
    print("  Loading exp5 (SupCon)...")
    exp5_models, exp5_fw = load_v31_exp5_models(category, device)
    models["exp5"] = (exp5_models, exp5_fw)

    # OpenHands
    oh_ckpt = os.path.join(CKPT_BASE, "openhands", category, "best_model.pt")
    if os.path.exists(oh_ckpt):
        print(f"  Loading OpenHands from {oh_ckpt}...")
        models["openhands"] = load_openhands_model(oh_ckpt, len(classes), device)
    else:
        print(f"  WARNING: OpenHands checkpoint not found: {oh_ckpt}")
        models["openhands"] = None

    return models


def adapt_models_to_data(models, raw_list, device, alpha=0.3):
    """
    Apply BN adaptation (AdaBN) using a list of raw (T,549) arrays.
    Returns adapted copies of BN models; GroupNorm models unchanged.
    raw_list: list of (streams, aux, gcn_v27, aux_v27, raw_oh) tuples
    """
    adapted = {}

    # V27 (BN) — adapt_bn_stats expects list of (gcn, aux) tuples
    if models["v27"] is not None:
        v27_data = [(item[2], item[3]) for item in raw_list if item[2] is not None]
        v27_adapted = copy.deepcopy(models["v27"])
        adapt_bn_stats(v27_adapted, v27_data, device)
        adapted["v27"] = v27_adapted
    else:
        adapted["v27"] = None

    # V28 (BN) — alpha-BN on streams; compute_target_bn_stats expects (streams, aux) tuples
    if models["v28"] is not None:
        v28_models, v28_fw = models["v28"]
        v28_adapted = {}
        stream_data = [(item[0], item[1]) for item in raw_list if item[0] is not None]
        for sname, smodel in v28_models.items():
            smodel_a = copy.deepcopy(smodel)
            source_stats = save_bn_stats(smodel_a)
            target_stats = compute_target_bn_stats(
                smodel_a, stream_data, device,
                is_multistream=True, stream_name=sname,
            )
            apply_alpha_bn(smodel_a, source_stats, target_stats, alpha)
            v28_adapted[sname] = smodel_a
        adapted["v28"] = (v28_adapted, v28_fw)
    else:
        adapted["v28"] = None

    # V29 (BN) — same (gcn, aux) format as v27
    if models["v29"] is not None:
        v27_data = [(item[2], item[3]) for item in raw_list if item[2] is not None]
        v29_adapted = copy.deepcopy(models["v29"])
        adapt_bn_stats(v29_adapted, v27_data, device)
        adapted["v29"] = v29_adapted
    else:
        adapted["v29"] = None

    # Exp1 (GroupNorm) — no adaptation, just copy reference
    adapted["exp1"] = models["exp1"]

    # Exp5 (BN) — AdaBN on streams; adapt_bn_stats expects (streams, aux) tuples
    if models["exp5"] is not None:
        exp5_models, exp5_fw = models["exp5"]
        exp5_adapted = {}
        stream_data = [(item[0], item[1]) for item in raw_list if item[0] is not None]
        for sname, smodel in exp5_models.items():
            smodel_a = copy.deepcopy(smodel)
            adapt_bn_stats(smodel_a, stream_data, device,
                          is_multistream=True, stream_name=sname)
            exp5_adapted[sname] = smodel_a
        adapted["exp5"] = (exp5_adapted, exp5_fw)
    else:
        adapted["exp5"] = None

    # OpenHands (BN) — AdaBN with 27-joint tensors
    if models["openhands"] is not None:
        oh_adapted = copy.deepcopy(models["openhands"])
        oh_tensors = [item[4] for item in raw_list if item[4] is not None]
        for m in oh_adapted.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.reset_running_stats()
                m.momentum = None
                m.training = True
        oh_adapted.eval()
        for m in oh_adapted.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.training = True
        with torch.no_grad():
            for x in oh_tensors:
                oh_adapted(x.unsqueeze(0).to(device))
        for m in oh_adapted.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.training = False
        adapted["openhands"] = oh_adapted
    else:
        adapted["openhands"] = None

    return adapted


def preprocess_raw(raw):
    """Preprocess a raw (T,549) array into all needed tensors."""
    streams, aux = preprocess_multistream(raw)
    gcn_v27, aux_v27 = preprocess_v27(raw)
    oh27 = preprocess_raw_for_openhands(raw)
    return streams, aux, gcn_v27, aux_v27, oh27


def get_probs_for_sample(streams, aux, gcn_v27, aux_v27, oh27, adapted_models, classes, device):
    """Run all adapted models on one sample → dict of prob vectors."""
    nc = len(classes)
    uniform = torch.ones(nc) / nc
    probs = {}

    with torch.no_grad():
        # V27
        m = adapted_models["v27"]
        if m is not None and gcn_v27 is not None:
            logits, *_ = m(gcn_v27.unsqueeze(0).to(device), aux_v27.unsqueeze(0).to(device), grl_lambda=0.0)
            probs["v27"] = F.softmax(logits, dim=1).cpu().squeeze(0)
        else:
            probs["v27"] = uniform

        # V28
        v28 = adapted_models["v28"]
        if v28 is not None and streams is not None:
            v28m, v28fw = v28
            probs["v28"] = get_v28_style_probs(streams, aux, v28m, v28fw, device)
        else:
            probs["v28"] = uniform

        # V29
        m = adapted_models["v29"]
        if m is not None and gcn_v27 is not None:
            logits, *_ = m(gcn_v27.unsqueeze(0).to(device), aux_v27.unsqueeze(0).to(device), grl_lambda=0.0)
            probs["v29"] = F.softmax(logits, dim=1).cpu().squeeze(0)
        else:
            probs["v29"] = uniform

        # Exp1 (GroupNorm)
        exp1 = adapted_models["exp1"]
        if exp1 is not None and streams is not None:
            exp1m, exp1fw = exp1
            probs["exp1"] = get_groupnorm_probs(streams, aux, exp1m, exp1fw, device)
        else:
            probs["exp1"] = uniform

        # Exp5 (AdaBN)
        exp5 = adapted_models["exp5"]
        if exp5 is not None and streams is not None:
            exp5m, exp5fw = exp5
            probs["exp5"] = get_v28_style_probs(streams, aux, exp5m, exp5fw, device)
        else:
            probs["exp5"] = uniform

        # OpenHands
        m = adapted_models["openhands"]
        if m is not None and oh27 is not None:
            logits_oh, _ = m(oh27.unsqueeze(0).to(device))
            probs["openhands"] = F.softmax(logits_oh, dim=1).cpu().squeeze(0)
        else:
            probs["openhands"] = uniform

    return probs


# ---------------------------------------------------------------------------
# Phase 1: collect probs on test_alpha (npy files, no video extraction)
# ---------------------------------------------------------------------------

def collect_probs_test_alpha(npy_items, classes, models, device, alpha=0.3):
    """
    Load test_alpha npy files, preprocess, adapt BN, collect per-model probs.

    npy_items: list of (npy_path, true_class)
    Returns: (all_probs dict, labels array)
    """
    print(f"  Preprocessing {len(npy_items)} test_alpha samples...")
    preprocessed = []
    labels = []

    for npy_path, true_class in npy_items:
        raw = np.load(npy_path).astype(np.float32)
        item = preprocess_raw(raw)
        preprocessed.append(item)
        labels.append(classes.index(true_class))

    # Adapt BN using test_alpha data
    print(f"  Adapting BN stats from {len(preprocessed)} test_alpha samples...")
    adapted = adapt_models_to_data(models, preprocessed, device, alpha)

    # Collect probabilities
    all_probs = {m: [] for m in MODEL_NAMES}
    for i, (npy_path, true_class) in enumerate(npy_items):
        streams, aux, gcn_v27, aux_v27, oh27 = preprocessed[i]
        p = get_probs_for_sample(streams, aux, gcn_v27, aux_v27, oh27, adapted, classes, device)
        for m in MODEL_NAMES:
            all_probs[m].append(p[m])

    for m in MODEL_NAMES:
        all_probs[m] = torch.stack(all_probs[m])
    labels = np.array(labels)

    return all_probs, labels


# ---------------------------------------------------------------------------
# Phase 2: collect probs on real_testers (video files)
# ---------------------------------------------------------------------------

def collect_probs_real_testers(cat_videos, classes, models, device, alpha=0.3):
    """
    Extract landmarks from real_testers videos, adapt BN, collect per-model probs.

    cat_videos: list of (video_path, true_class, signer_id)
    Returns: (all_probs dict, labels array)
    """
    print(f"  Extracting/loading landmarks from {len(cat_videos)} real_testers videos...")
    preprocessed = []
    labels = []
    failed_indices = []

    for i, (video_path, true_class, _) in enumerate(cat_videos):
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache).astype(np.float32)
        else:
            raw = extract_landmarks_from_video(video_path)
        if raw is None:
            print(f"    FAILED: {os.path.basename(video_path)}")
            failed_indices.append(i)
            preprocessed.append(None)
            labels.append(classes.index(true_class))
            continue
        item = preprocess_raw(raw)
        preprocessed.append(item)
        labels.append(classes.index(true_class))

    # Filter out failed for BN adaptation
    valid_prep = [p for p in preprocessed if p is not None]
    print(f"  Adapting BN stats from {len(valid_prep)} real_testers samples...")
    adapted = adapt_models_to_data(models, valid_prep, device, alpha)

    # Collect probabilities
    nc = len(classes)
    all_probs = {m: [] for m in MODEL_NAMES}
    for i, item in enumerate(preprocessed):
        if item is None:
            for m in MODEL_NAMES:
                all_probs[m].append(torch.ones(nc) / nc)
        else:
            streams, aux, gcn_v27, aux_v27, oh27 = item
            p = get_probs_for_sample(streams, aux, gcn_v27, aux_v27, oh27, adapted, classes, device)
            for m in MODEL_NAMES:
                all_probs[m].append(p[m])

    for m in MODEL_NAMES:
        all_probs[m] = torch.stack(all_probs[m])
    labels = np.array(labels)

    return all_probs, labels


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def grid_search_weights_fast(model_probs, labels, model_names, step=0.1,
                              min_weight=0.0, entropy_lambda=0.0):
    """
    Efficient recursive grid search: enumerate only valid (sum=1) combos.

    Args:
        min_weight:     Minimum weight per model (0.0 = unconstrained). Setting
                        0.05 prevents catastrophic model exclusion (audit fix).
        entropy_lambda: Entropy regularization strength. Score = acc + λ * H(w)
                        where H is weight entropy. Higher λ → push toward uniform.
                        0.0 = pure accuracy (original behavior).
    """
    n_models = len(model_names)
    n_steps = int(round(1.0 / step))
    min_steps = math.ceil(min_weight / step)  # minimum grid steps per model
    probs_list = [model_probs[m] for m in model_names]

    uniform_fused = sum(probs_list) / n_models
    uniform_acc = 100.0 * (uniform_fused.argmax(1).numpy() == labels).mean()

    best_score = -1e9
    best_acc = 0.0
    best_weights = {m: 1.0 / n_models for m in model_names}
    combos = [0]

    def entropy(weights_list):
        """Shannon entropy of a weight distribution."""
        total = 0.0
        for w in weights_list:
            if w > 1e-9:
                total -= w * np.log(w)
        return total

    def search(depth, remaining_steps, current_weights):
        nonlocal best_score, best_acc, best_weights
        combos[0] += 1
        # Remaining models (including current) must each get at least min_steps
        max_for_this = remaining_steps - min_steps * (n_models - depth - 1)
        if depth == n_models - 1:
            w_last = remaining_steps * step
            if w_last < min_weight - 1e-9:
                return
            current_weights.append(w_last)
            fused = sum(current_weights[i] * probs_list[i] for i in range(n_models))
            acc = 100.0 * (fused.argmax(1).numpy() == labels).mean()
            score = acc + entropy_lambda * entropy(current_weights)
            if score > best_score:
                best_score = score
                best_acc = acc
                best_weights = {m: current_weights[i] for i, m in enumerate(model_names)}
            current_weights.pop()
            return
        for s in range(min_steps, max_for_this + 1):
            current_weights.append(s * step)
            search(depth + 1, remaining_steps - s, current_weights)
            current_weights.pop()

    search(0, n_steps, [])
    print(f"    Tried {combos[0]:,} weight combinations")
    return best_weights, best_acc, uniform_acc


def apply_weights(model_probs, weights, labels):
    """Apply fixed weights to model_probs and compute accuracy."""
    fused = sum(weights[m] * model_probs[m] for m in weights)
    preds = fused.argmax(dim=1).numpy()
    acc = 100.0 * (preds == labels).mean()
    correct = (preds == labels).sum()
    return acc, int(correct), len(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean Ensemble Weight Optimization")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Weight grid step (default 0.1; use 0.05 for finer search)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Alpha-BN mixing coefficient (default 0.3)")
    parser.add_argument("--min_weight", type=float, default=0.0,
                        help="Minimum weight per model (0=unconstrained; 0.05 prevents "
                             "catastrophic model exclusion; audit-recommended fix)")
    parser.add_argument("--entropy_lambda", type=float, default=0.0,
                        help="Entropy regularization strength (0=pure accuracy; "
                             "higher values push weights toward uniform; try 0.1-0.3)")
    args = parser.parse_args()

    print("=" * 70)
    print("Clean Ensemble Weight Optimization")
    print("  Phase 1 (tuning):     test_alpha  (ksl-alpha signers 13-15)")
    print("  Phase 2 (blind eval): real_testers (actual human test subjects)")
    print(f"Models: {', '.join(MODEL_NAMES)}")
    print(f"Grid step: {args.step}  Alpha-BN: {args.alpha}  "
          f"min_weight: {args.min_weight}  entropy_lambda: {args.entropy_lambda}")
    print(f"Started: {ts()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Discover real_testers videos
    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    print(f"\nreal_testers: {len(raw_videos)} videos total")

    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        print(f"\n{'='*70}")
        print(f"{category.upper()}")
        print(f"{'='*70}")

        # test_alpha samples (Phase 1)
        alpha_items = discover_test_alpha(category, classes)
        print(f"\ntest_alpha: {len(alpha_items)} npy files")

        # real_testers videos (Phase 2)
        cat_videos = [(vp, nm, sg) for vp, nm, cat, sg in raw_videos if cat == category]
        print(f"real_testers: {len(cat_videos)} videos")

        # Load models once (shared between phases)
        print(f"\nLoading models for {category}...")
        models = load_all_models(category, classes, device)

        # --- Phase 1: optimize weights on test_alpha ---
        print(f"\n--- Phase 1: weight optimization on test_alpha ---")
        alpha_probs, alpha_labels = collect_probs_test_alpha(alpha_items, classes, models, device, args.alpha)

        print(f"\n  Per-model accuracy on test_alpha:")
        for m in MODEL_NAMES:
            preds = alpha_probs[m].argmax(1).numpy()
            acc = 100.0 * (preds == alpha_labels).mean()
            print(f"    {m:>10s}: {acc:.1f}%")

        print(f"\n  Grid-searching optimal weights on test_alpha (step={args.step})...")
        best_weights, best_alpha_acc, uniform_alpha_acc = grid_search_weights_fast(
            alpha_probs, alpha_labels, MODEL_NAMES, step=args.step,
            min_weight=args.min_weight, entropy_lambda=args.entropy_lambda,
        )
        print(f"  test_alpha uniform:    {uniform_alpha_acc:.1f}%")
        print(f"  test_alpha optimized:  {best_alpha_acc:.1f}% ({best_alpha_acc - uniform_alpha_acc:+.1f}pp)")
        print(f"  Optimal weights: {', '.join(f'{m}={w:.2f}' for m, w in best_weights.items())}")

        # --- Phase 2: evaluate on real_testers with Phase-1 weights ---
        print(f"\n--- Phase 2: blind evaluation on real_testers ---")
        rt_probs, rt_labels = collect_probs_real_testers(cat_videos, classes, models, device, args.alpha)

        print(f"\n  Per-model accuracy on real_testers:")
        for m in MODEL_NAMES:
            preds = rt_probs[m].argmax(1).numpy()
            acc = 100.0 * (preds == rt_labels).mean()
            print(f"    {m:>10s}: {acc:.1f}%")

        # Uniform ensemble on real_testers
        rt_unif_acc, _, _ = apply_weights(rt_probs, {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}, rt_labels)
        print(f"\n  real_testers uniform ensemble: {rt_unif_acc:.1f}%")

        # Apply Phase-1 weights to real_testers
        rt_acc, rt_correct, rt_total = apply_weights(rt_probs, best_weights, rt_labels)
        print(f"  real_testers CLEAN result:     {rt_acc:.1f}%  ({rt_correct}/{rt_total})")
        print(f"  (weights were tuned on test_alpha, never on real_testers)")

        # Also report oracle (weights tuned on real_testers, for comparison)
        print(f"\n  For comparison — oracle weights (tuned ON real_testers, BIASED):")
        oracle_weights, oracle_acc, _ = grid_search_weights_fast(
            rt_probs, rt_labels, MODEL_NAMES, step=args.step,
            min_weight=0.0, entropy_lambda=0.0,  # oracle always unconstrained
        )
        print(f"  real_testers oracle: {oracle_acc:.1f}%")
        print(f"  Oracle weights: {', '.join(f'{m}={w:.2f}' for m, w in oracle_weights.items())}")
        print(f"  Overfit gap: {oracle_acc - rt_acc:.1f}pp (oracle - clean)")

        all_results[category] = {
            "test_alpha_samples": len(alpha_items),
            "real_tester_videos": rt_total,
            "per_model_alpha_acc": {
                m: float(100.0 * (alpha_probs[m].argmax(1).numpy() == alpha_labels).mean())
                for m in MODEL_NAMES
            },
            "per_model_rt_acc": {
                m: float(100.0 * (rt_probs[m].argmax(1).numpy() == rt_labels).mean())
                for m in MODEL_NAMES
            },
            "phase1_weights": {k: float(v) for k, v in best_weights.items()},
            "test_alpha_optimized_acc": float(best_alpha_acc),
            "real_tester_uniform_acc": float(rt_unif_acc),
            "real_tester_clean_acc": float(rt_acc),
            "real_tester_oracle_acc": float(oracle_acc),
            "overfit_gap_pp": float(oracle_acc - rt_acc),
        }

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Phase-1 weights tuned on test_alpha (ksl-alpha signers 13-15)")
    print(f"  Phase-2 evaluation on real_testers (actual human signers)\n")

    for cat, r in all_results.items():
        print(f"  {cat.upper()}:")
        print(f"    Per-model real_testers: {r['per_model_rt_acc']}")
        print(f"    Uniform ensemble:       {r['real_tester_uniform_acc']:.1f}%")
        print(f"    CLEAN result:           {r['real_tester_clean_acc']:.1f}%   ← paper-worthy")
        print(f"    Oracle (biased):        {r['real_tester_oracle_acc']:.1f}%")
        print(f"    Overfit gap:            {r['overfit_gap_pp']:.1f}pp")
        print()

    if "numbers" in all_results and "words" in all_results:
        n_r, w_r = all_results["numbers"], all_results["words"]
        n_tot, w_tot = n_r["real_tester_videos"], w_r["real_tester_videos"]

        def combined(key):
            nc = round(n_r[key] * n_tot / 100)
            wc = round(w_r[key] * w_tot / 100)
            return 100.0 * (nc + wc) / (n_tot + w_tot)

        print(f"  Combined uniform:     {combined('real_tester_uniform_acc'):.1f}%")
        print(f"  Combined CLEAN:       {combined('real_tester_clean_acc'):.1f}%   ← paper-worthy")
        print(f"  Combined oracle:      {combined('real_tester_oracle_acc'):.1f}%  (biased)")
        print(f"\n  Reference: prev oracle = 76.4% (74.3% before OpenHands)")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ensemble_clean_{ts_str}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
