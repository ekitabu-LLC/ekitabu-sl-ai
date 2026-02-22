#!/usr/bin/env python3
"""
V44 Experiment 4: EDTN — Exponential Decay Test-time Normalization.

Instead of full AdaBN (replace all BN stats with test-time statistics),
EDTN blends source and test-time stats per layer depth:
    stats_at_depth_d = alpha^d * test_stats + (1 - alpha^d) * source_stats
Shallow layers get more test-time adaptation, deep layers retain more source knowledge.

Only applies to BatchNorm models: exp5, v43, v27, v28, v29, openhands.
GroupNorm models (exp1, v41) have no BN layers — skipped.

Phase 1: Tune alpha on test_alpha (signers 13-15)
Phase 2: Apply best alpha to real_testers (blind eval)

Usage:
    python evaluate_v44_expr4.py --phase tune     # Phase 1: tune alpha on test_alpha
    python evaluate_v44_expr4.py --phase eval     # Phase 2: eval best alpha on real_testers
    python evaluate_v44_expr4.py --phase both     # Both phases
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
from collections import defaultdict
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
    adapt_bn_stats,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"
TEST_ALPHA_DIR = "/scratch/alpine/hama5612/ksl-dir-2/data/test_alpha"
REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
RESULTS_DIR = "/scratch/alpine/hama5612/ksl-dir-2/data/results/v44_expr4"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# EDTN: Exponential Decay Test-time Normalization
# ---------------------------------------------------------------------------

def apply_edtn(model, data_list, device, stream_name=None, alpha=0.7):
    """
    EDTN: Exponential Decay Test-time Normalization.

    For each BN layer at depth d (0-indexed from shallow to deep):
      new_mean = alpha^d * test_mean + (1 - alpha^d) * source_mean
      new_var  = alpha^d * test_var  + (1 - alpha^d) * source_var

    alpha in (0, 1):
      alpha=1.0 -> full AdaBN (all test stats)
      alpha=0.0 -> no adaptation (keep source stats)
      alpha=0.7 -> recommended starting point

    Args:
        model: nn.Module with BatchNorm layers (will be modified in-place)
        data_list: list of (streams_dict, aux_tensor) tuples
        device: torch device
        stream_name: if not None, index streams_dict[stream_name] for gcn input
        alpha: decay rate (0, 1]
    """
    if len(data_list) == 0:
        return

    # Step 1: Save original source statistics
    source_stats = {}
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                'mean': module.running_mean.clone(),
                'var': module.running_var.clone(),
            }
            bn_layers.append((name, module))

    if not bn_layers:
        return  # No BN layers (GroupNorm model)

    # Step 2: Collect test-time statistics (same as AdaBN)
    for name, module in bn_layers:
        module.reset_running_stats()
        module.momentum = None  # cumulative moving average

    model.train()
    with torch.no_grad():
        for item in data_list:
            streams, aux = item
            if stream_name is not None:
                gcn = streams[stream_name].unsqueeze(0).to(device)
            else:
                gcn = streams.unsqueeze(0).to(device)
            model(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)

    # Step 3: Apply exponential decay blending
    model.eval()
    for depth, (name, module) in enumerate(bn_layers):
        w = alpha ** depth  # weight for test stats (decays with depth)
        src = source_stats[name]
        test_mean = module.running_mean.clone()
        test_var = module.running_var.clone()
        module.running_mean.copy_(w * test_mean + (1.0 - w) * src['mean'])
        module.running_var.copy_(w * test_var + (1.0 - w) * src['var'])


def apply_edtn_openhands(model, raw_list, device, alpha=0.7, max_frames=90):
    """EDTN for OpenHands models (different preprocessing pipeline)."""
    from evaluate_openhands_realtest import preprocess_raw_for_openhands

    # Step 1: Save source statistics
    source_stats = {}
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                'mean': module.running_mean.clone(),
                'var': module.running_var.clone(),
            }
            bn_layers.append((name, module))

    if not bn_layers:
        return

    # Step 2: Collect test-time statistics
    for name, module in bn_layers:
        module.reset_running_stats()
        module.momentum = None
        module.training = True

    model.eval()
    # Keep BN in train mode for stat update
    for name, module in bn_layers:
        module.training = True

    with torch.no_grad():
        for raw in raw_list:
            if raw is None:
                continue
            x = preprocess_raw_for_openhands(raw, max_frames)
            if x is None:
                continue
            model(x.unsqueeze(0).to(device))

    # Step 3: Apply exponential decay blending
    for name, module in bn_layers:
        module.training = False

    for depth, (name, module) in enumerate(bn_layers):
        w = alpha ** depth
        src = source_stats[name]
        test_mean = module.running_mean.clone()
        test_var = module.running_var.clone()
        module.running_mean.copy_(w * test_mean + (1.0 - w) * src['mean'])
        module.running_var.copy_(w * test_var + (1.0 - w) * src['var'])


# ---------------------------------------------------------------------------
# Model Loading (all BatchNorm models)
# ---------------------------------------------------------------------------

def load_multistream_models(ckpt_dir, classes, device, model_class, build_adj_fn):
    """Load multi-stream (joint/bone/velocity) models."""
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"    WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        num_signers = ckpt.get("num_signers", 12)
        num_nodes = config.get("num_nodes", 48)
        adj = build_adj_fn(num_nodes).to(device)

        model = model_class(
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
    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v27_model(ckpt_path, classes, device):
    """Load V27 single-model (9ch joint+velocity+bone concatenated)."""
    from train_ksl_v27 import KSLGraphNetV25, build_adj

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_signers = ckpt.get("num_signers", 12)
    num_nodes = config.get("num_nodes", 48)
    adj = build_adj(num_nodes).to(device)

    model = KSLGraphNetV25(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=num_nodes,
        ic=config.get("in_channels", 9),
        hd=config.get("hidden_dim", 64),
        nl=config.get("num_layers", 4),
        tk=tuple(config.get("temporal_kernels", [3, 5, 7])),
        dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_v29_model(ckpt_path, classes, device):
    """Load V29 single-model (9ch, 8 layers, dilated TCN)."""
    from train_ksl_v29 import KSLGraphNetV29, build_adj

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_signers = ckpt.get("num_signers", 12)
    num_nodes = config.get("num_nodes", 48)
    adj = build_adj(num_nodes).to(device)

    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=num_nodes,
        ic=config.get("in_channels", 9),
        hd=config.get("hidden_dim", 64),
        nl=config.get("num_layers", 8),
        td=tuple(config.get("temporal_dilations", [1, 2, 4])),
        dr=config.get("dropout", 0.2),
        spatial_dropout=config.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_openhands_model(ckpt_path, num_classes, device):
    """Load OpenHands model."""
    from train_ksl_openhands import OpenHandsClassifier, OH_CONFIG
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
    return model


# ---------------------------------------------------------------------------
# Test-alpha evaluation (for tuning alpha)
# ---------------------------------------------------------------------------

def load_test_alpha_data(classes):
    """Load pre-extracted .npy files from test_alpha as labeled data."""
    c2i = {c: i for i, c in enumerate(classes)}
    samples = []
    for cls_name in sorted(os.listdir(TEST_ALPHA_DIR)):
        cls_path = os.path.join(TEST_ALPHA_DIR, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in c2i:
            continue
        for fn in sorted(os.listdir(cls_path)):
            if not fn.endswith(".npy"):
                continue
            raw = np.load(os.path.join(cls_path, fn)).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] < 225:
                continue
            samples.append((raw, cls_name))
    return samples


def eval_multistream_on_samples(models, fusion_weights, samples, classes, device):
    """Evaluate multi-stream model on list of (raw, true_class) samples."""
    i2c = {i: c for i, c in enumerate(classes)}
    correct = 0
    total = 0
    for raw, true_class in samples:
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            continue
        per_stream_probs = {}
        with torch.no_grad():
            for sname, model in models.items():
                gcn = streams[sname].unsqueeze(0).to(device)
                logits, *_ = model(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
        fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
        pred_idx = fused.argmax().item()
        pred_class = i2c[pred_idx]
        correct += int(pred_class == true_class)
        total += 1
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


def eval_v27_on_samples(model, samples, classes, device):
    """Evaluate V27 single-model on test_alpha samples."""
    from evaluate_real_testers_v27 import preprocess_landmarks
    i2c = {i: c for i, c in enumerate(classes)}
    correct = 0
    total = 0
    for raw, true_class in samples:
        gcn_tensor, aux_tensor = preprocess_landmarks(raw)
        if gcn_tensor is None:
            continue
        with torch.no_grad():
            logits, *_ = model(
                gcn_tensor.unsqueeze(0).to(device),
                aux_tensor.unsqueeze(0).to(device),
                grl_lambda=0.0,
            )
            pred_idx = logits.argmax(dim=1).item()
        pred_class = i2c[pred_idx]
        correct += int(pred_class == true_class)
        total += 1
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


def eval_v29_on_samples(model, samples, classes, device):
    """Evaluate V29 single-model on test_alpha samples."""
    from evaluate_real_testers_v29 import preprocess_landmarks as preprocess_v29
    i2c = {i: c for i, c in enumerate(classes)}
    correct = 0
    total = 0
    for raw, true_class in samples:
        gcn_tensor, aux_tensor = preprocess_v29(raw)
        if gcn_tensor is None:
            continue
        with torch.no_grad():
            logits, *_ = model(
                gcn_tensor.unsqueeze(0).to(device),
                aux_tensor.unsqueeze(0).to(device),
                grl_lambda=0.0,
            )
            pred_idx = logits.argmax(dim=1).item()
        pred_class = i2c[pred_idx]
        correct += int(pred_class == true_class)
        total += 1
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


def eval_openhands_on_samples(model, samples, classes, device):
    """Evaluate OpenHands on test_alpha samples."""
    from evaluate_openhands_realtest import preprocess_raw_for_openhands
    i2c = {i: c for i, c in enumerate(classes)}
    correct = 0
    total = 0
    for raw, true_class in samples:
        x = preprocess_raw_for_openhands(raw)
        if x is None:
            continue
        with torch.no_grad():
            logits, _ = model(x.unsqueeze(0).to(device))
            pred_idx = logits.argmax(dim=1).item()
        pred_class = i2c[pred_idx]
        correct += int(pred_class == true_class)
        total += 1
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


# ---------------------------------------------------------------------------
# EDTN for multi-stream models on test_alpha
# ---------------------------------------------------------------------------

def prepare_multistream_adabn_data(samples, stream_name=None):
    """Convert raw samples to format expected by adapt_bn_stats / apply_edtn."""
    data_list = []
    for raw, _ in samples:
        streams, aux = preprocess_multistream(raw)
        if streams is None:
            continue
        data_list.append((streams, aux))
    return data_list


def prepare_v27_adabn_data(samples):
    """Convert raw samples to (gcn, aux) for V27 single model."""
    from evaluate_real_testers_v27 import preprocess_landmarks
    data_list = []
    for raw, _ in samples:
        gcn_tensor, aux_tensor = preprocess_landmarks(raw)
        if gcn_tensor is None:
            continue
        data_list.append((gcn_tensor, aux_tensor))
    return data_list


def prepare_v29_adabn_data(samples):
    """Convert raw samples to (gcn, aux) for V29 single model."""
    from evaluate_real_testers_v29 import preprocess_landmarks as preprocess_v29
    data_list = []
    for raw, _ in samples:
        gcn_tensor, aux_tensor = preprocess_v29(raw)
        if gcn_tensor is None:
            continue
        data_list.append((gcn_tensor, aux_tensor))
    return data_list


def apply_edtn_v27(model, data_list, device, alpha=0.7):
    """EDTN for V27 single-model (no stream_name needed)."""
    if len(data_list) == 0:
        return

    source_stats = {}
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                'mean': module.running_mean.clone(),
                'var': module.running_var.clone(),
            }
            bn_layers.append((name, module))

    if not bn_layers:
        return

    for name, module in bn_layers:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)

    model.eval()
    for depth, (name, module) in enumerate(bn_layers):
        w = alpha ** depth
        src = source_stats[name]
        test_mean = module.running_mean.clone()
        test_var = module.running_var.clone()
        module.running_mean.copy_(w * test_mean + (1.0 - w) * src['mean'])
        module.running_var.copy_(w * test_var + (1.0 - w) * src['var'])


def adapt_bn_v27(model, data_list, device):
    """Standard AdaBN for V27 single-model."""
    if len(data_list) == 0:
        return
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None

    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
    model.eval()


# ---------------------------------------------------------------------------
# Phase 1: Tune alpha on test_alpha
# ---------------------------------------------------------------------------

def tune_alpha_multistream(model_name, ckpt_dir, classes, device, alphas):
    """Tune EDTN alpha for a multi-stream model on test_alpha."""
    if model_name in ("v43",):
        from train_ksl_v43 import KSLGraphNetV43, build_adj
        model_class, build_adj_fn = KSLGraphNetV43, build_adj
    else:
        from train_ksl_v28 import KSLGraphNetV25, build_adj
        model_class, build_adj_fn = KSLGraphNetV25, build_adj

    print(f"\n  Loading {model_name} models from {ckpt_dir}...")
    models_orig, fusion_weights = load_multistream_models(
        ckpt_dir, classes, device, model_class, build_adj_fn)

    if len(models_orig) < 3:
        print(f"    ERROR: Only {len(models_orig)}/3 streams loaded, skipping")
        return None

    # Load test_alpha samples
    samples = load_test_alpha_data(classes)
    print(f"  Loaded {len(samples)} test_alpha samples")

    # Prepare AdaBN data
    adabn_data = prepare_multistream_adabn_data(samples)
    print(f"  Prepared {len(adabn_data)} preprocessed samples for BN adaptation")

    results = {}
    for alpha in alphas:
        models = {sname: copy.deepcopy(m) for sname, m in models_orig.items()}

        if alpha == 0.0:
            label = "no_adapt"
        elif alpha == 1.0:
            label = "full_adabn"
            for sname, model in models.items():
                adapt_bn_stats(model, adabn_data, device, stream_name=sname)
        else:
            label = f"edtn_{alpha}"
            for sname, model in models.items():
                apply_edtn(model, adabn_data, device, stream_name=sname, alpha=alpha)

        acc, correct, total = eval_multistream_on_samples(
            models, fusion_weights, samples, classes, device)
        results[label] = {"accuracy": acc, "correct": correct, "total": total, "alpha": alpha}
        print(f"    alpha={alpha:.1f} ({label:>12s}): {acc:.1f}% ({correct}/{total})")

        # Free GPU memory
        del models

    return results


def tune_alpha_v27(classes, device, alphas, cat_name):
    """Tune EDTN alpha for V27 single-model on test_alpha."""
    ckpt_path = os.path.join(CKPT_BASE, f"v27_{cat_name}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"    Skipping v27 {cat_name}: no checkpoint at {ckpt_path}")
        return None

    print(f"\n  Loading V27 {cat_name} from {ckpt_path}...")
    model_orig = load_v27_model(ckpt_path, classes, device)

    samples = load_test_alpha_data(classes)
    print(f"  Loaded {len(samples)} test_alpha samples")

    adabn_data = prepare_v27_adabn_data(samples)
    print(f"  Prepared {len(adabn_data)} preprocessed samples for BN adaptation")

    results = {}
    for alpha in alphas:
        model = copy.deepcopy(model_orig)

        if alpha == 0.0:
            label = "no_adapt"
        elif alpha == 1.0:
            label = "full_adabn"
            adapt_bn_v27(model, adabn_data, device)
        else:
            label = f"edtn_{alpha}"
            apply_edtn_v27(model, adabn_data, device, alpha=alpha)

        acc, correct, total = eval_v27_on_samples(model, samples, classes, device)
        results[label] = {"accuracy": acc, "correct": correct, "total": total, "alpha": alpha}
        print(f"    alpha={alpha:.1f} ({label:>12s}): {acc:.1f}% ({correct}/{total})")

        del model

    return results


def tune_alpha_v29(classes, device, alphas, cat_name):
    """Tune EDTN alpha for V29 single-model on test_alpha."""
    ckpt_path = os.path.join(CKPT_BASE, f"v29_{cat_name}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"    Skipping v29 {cat_name}: no checkpoint at {ckpt_path}")
        return None

    print(f"\n  Loading V29 {cat_name} from {ckpt_path}...")
    model_orig = load_v29_model(ckpt_path, classes, device)

    samples = load_test_alpha_data(classes)
    print(f"  Loaded {len(samples)} test_alpha samples")

    adabn_data = prepare_v29_adabn_data(samples)
    print(f"  Prepared {len(adabn_data)} preprocessed samples for BN adaptation")

    results = {}
    for alpha in alphas:
        model = copy.deepcopy(model_orig)

        if alpha == 0.0:
            label = "no_adapt"
        elif alpha == 1.0:
            label = "full_adabn"
            # Reuse adapt_bn_v27 — same interface (gcn, aux) tuples
            adapt_bn_v27(model, adabn_data, device)
        else:
            label = f"edtn_{alpha}"
            # Reuse apply_edtn_v27 — same interface
            apply_edtn_v27(model, adabn_data, device, alpha=alpha)

        acc, correct, total = eval_v29_on_samples(model, samples, classes, device)
        results[label] = {"accuracy": acc, "correct": correct, "total": total, "alpha": alpha}
        print(f"    alpha={alpha:.1f} ({label:>12s}): {acc:.1f}% ({correct}/{total})")

        del model

    return results


def tune_alpha_openhands(classes, device, alphas, cat_name):
    """Tune EDTN alpha for OpenHands on test_alpha."""
    ckpt_path = os.path.join(CKPT_BASE, "openhands", cat_name, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"    Skipping openhands {cat_name}: no checkpoint at {ckpt_path}")
        return None

    print(f"\n  Loading OpenHands {cat_name} from {ckpt_path}...")
    model_orig = load_openhands_model(ckpt_path, len(classes), device)

    samples = load_test_alpha_data(classes)
    print(f"  Loaded {len(samples)} test_alpha samples")

    # Prepare raw data for OpenHands AdaBN
    raw_list = [raw for raw, _ in samples]
    print(f"  Prepared {len(raw_list)} raw samples for BN adaptation")

    results = {}
    for alpha in alphas:
        model = copy.deepcopy(model_orig)

        if alpha == 0.0:
            label = "no_adapt"
        elif alpha == 1.0:
            label = "full_adabn"
            from evaluate_openhands_realtest import apply_adabn_global
            model = apply_adabn_global(model, raw_list, device)
        else:
            label = f"edtn_{alpha}"
            apply_edtn_openhands(model, raw_list, device, alpha=alpha)

        acc, correct, total = eval_openhands_on_samples(model, samples, classes, device)
        results[label] = {"accuracy": acc, "correct": correct, "total": total, "alpha": alpha}
        print(f"    alpha={alpha:.1f} ({label:>12s}): {acc:.1f}% ({correct}/{total})")

        del model

    return results


def run_phase1_tune(device):
    """Phase 1: Tune alpha on test_alpha for all BatchNorm models."""
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    print("\n" + "=" * 70)
    print("PHASE 1: Tune EDTN alpha on test_alpha (signers 13-15)")
    print(f"Alpha values: {alphas}")
    print("=" * 70)

    all_results = {}

    # --- Multi-stream models (exp5, v43, v28 have 3 separate stream models) ---
    multistream_models = {
        "exp5": {"numbers": os.path.join(CKPT_BASE, "v31_exp5", "numbers"),
                 "words": os.path.join(CKPT_BASE, "v31_exp5", "words")},
        "v43": {"numbers": os.path.join(CKPT_BASE, "v43", "numbers"),
                "words": os.path.join(CKPT_BASE, "v43", "words")},
        "v28": {"numbers": os.path.join(CKPT_BASE, "v28_numbers"),
                "words": os.path.join(CKPT_BASE, "v28_words")},
    }

    for model_name, cat_dirs in multistream_models.items():
        for cat_name, ckpt_dir in cat_dirs.items():
            if not os.path.isdir(ckpt_dir):
                print(f"\n  Skipping {model_name}/{cat_name}: {ckpt_dir} not found")
                continue

            classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
            print(f"\n{'='*70}")
            print(f"{model_name.upper()} / {cat_name.upper()} — EDTN alpha tuning")
            print(f"{'='*70}")

            result = tune_alpha_multistream(
                model_name, ckpt_dir, classes, device, alphas)
            if result is not None:
                all_results[f"{model_name}_{cat_name}"] = result

    # --- V27 (single 9ch model) ---
    for cat_name in ["numbers", "words"]:
        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        print(f"\n{'='*70}")
        print(f"V27 / {cat_name.upper()} — EDTN alpha tuning")
        print(f"{'='*70}")

        result = tune_alpha_v27(classes, device, alphas, cat_name)
        if result is not None:
            all_results[f"v27_{cat_name}"] = result

    # --- V29 (single 9ch model, different architecture) ---
    for cat_name in ["numbers", "words"]:
        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        print(f"\n{'='*70}")
        print(f"V29 / {cat_name.upper()} — EDTN alpha tuning")
        print(f"{'='*70}")

        result = tune_alpha_v29(classes, device, alphas, cat_name)
        if result is not None:
            all_results[f"v29_{cat_name}"] = result

    # --- OpenHands ---
    for cat_name in ["numbers", "words"]:
        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        print(f"\n{'='*70}")
        print(f"OPENHANDS / {cat_name.upper()} — EDTN alpha tuning")
        print(f"{'='*70}")

        result = tune_alpha_openhands(classes, device, alphas, cat_name)
        if result is not None:
            all_results[f"openhands_{cat_name}"] = result

    # --- Summary ---
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY: Best alpha per model/category on test_alpha")
    print(f"{'='*70}")

    best_alphas = {}
    for key, results in all_results.items():
        best_label = max(results, key=lambda k: results[k]["accuracy"])
        best = results[best_label]
        best_alphas[key] = best["alpha"]
        print(f"  {key:>25s}: best={best_label:>12s} ({best['accuracy']:.1f}%)")

        # Print full table
        for label, r in sorted(results.items(), key=lambda x: x[1]["alpha"]):
            marker = " <-- best" if label == best_label else ""
            print(f"    alpha={r['alpha']:.1f}: {r['accuracy']:.1f}%{marker}")

    return all_results, best_alphas


# ---------------------------------------------------------------------------
# Phase 2: Eval on real_testers with best alpha
# ---------------------------------------------------------------------------

def make_predict_fn_multistream(models, fusion_weights, classes, device):
    """Create prediction function for multi-stream model."""
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


def make_predict_fn_v27(model, classes, device):
    """Create prediction function for V27 single-model."""
    from evaluate_real_testers_v27 import preprocess_landmarks
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        gcn_tensor, aux_tensor = preprocess_landmarks(raw)
        if gcn_tensor is None:
            return None
        with torch.no_grad():
            logits, *_ = model(
                gcn_tensor.unsqueeze(0).to(device),
                aux_tensor.unsqueeze(0).to(device),
                grl_lambda=0.0,
            )
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx = probs.argmax().item()
        return i2c[pred_idx], probs[pred_idx].item()

    return predict


def make_predict_fn_v29(model, classes, device):
    """Create prediction function for V29 single-model."""
    from evaluate_real_testers_v29 import preprocess_landmarks as preprocess_v29
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        gcn_tensor, aux_tensor = preprocess_v29(raw)
        if gcn_tensor is None:
            return None
        with torch.no_grad():
            logits, *_ = model(
                gcn_tensor.unsqueeze(0).to(device),
                aux_tensor.unsqueeze(0).to(device),
                grl_lambda=0.0,
            )
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx = probs.argmax().item()
        return i2c[pred_idx], probs[pred_idx].item()

    return predict


def make_predict_fn_openhands(model, classes, device):
    """Create prediction function for OpenHands."""
    from evaluate_openhands_realtest import preprocess_raw_for_openhands
    i2c = {i: c for i, c in enumerate(classes)}

    def predict(raw, true_class):
        x = preprocess_raw_for_openhands(raw)
        if x is None:
            return None
        with torch.no_grad():
            logits, _ = model(x.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred_idx = probs.argmax().item()
        return i2c[pred_idx], probs[pred_idx].item()

    return predict


def run_phase2_eval(device, best_alphas=None):
    """Phase 2: Evaluate best alpha on real_testers."""
    if best_alphas is None:
        # Default alphas if Phase 1 was not run
        best_alphas = {}

    # For any model not in best_alphas, test all alphas
    test_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    print("\n" + "=" * 70)
    print("PHASE 2: Evaluate EDTN on real_testers")
    if best_alphas:
        print(f"Best alphas from Phase 1: {best_alphas}")
    else:
        print(f"Testing all alphas: {test_alphas}")
    print("=" * 70)

    check_mediapipe_version()

    print(f"\nScanning {REAL_TESTER_DIR}...")
    videos = discover_test_videos(REAL_TESTER_DIR)
    numbers_videos = [(v, n, s) for v, n, c, s in videos if c == "numbers"]
    words_videos = [(v, n, s) for v, n, c, s in videos if c == "words"]
    print(f"Found {len(videos)} test videos "
          f"(Numbers: {len(numbers_videos)}, Words: {len(words_videos)})")

    all_results = {}

    # --- Multi-stream models (exp5, v43, v28) ---
    multistream_configs = {
        "exp5": {"numbers": os.path.join(CKPT_BASE, "v31_exp5", "numbers"),
                 "words": os.path.join(CKPT_BASE, "v31_exp5", "words")},
        "v43": {"numbers": os.path.join(CKPT_BASE, "v43", "numbers"),
                "words": os.path.join(CKPT_BASE, "v43", "words")},
        "v28": {"numbers": os.path.join(CKPT_BASE, "v28_numbers"),
                "words": os.path.join(CKPT_BASE, "v28_words")},
    }

    for model_name, cat_dirs in multistream_configs.items():
        if model_name in ("v43",):
            from train_ksl_v43 import KSLGraphNetV43, build_adj
            model_class, build_adj_fn = KSLGraphNetV43, build_adj
        else:
            from train_ksl_v28 import KSLGraphNetV25, build_adj
            model_class, build_adj_fn = KSLGraphNetV25, build_adj

        for cat_name, ckpt_dir in cat_dirs.items():
            if not os.path.isdir(ckpt_dir):
                continue

            key = f"{model_name}_{cat_name}"
            classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
            cat_videos = numbers_videos if cat_name == "numbers" else words_videos

            if not cat_videos:
                continue

            # Determine which alphas to test
            if key in best_alphas:
                alphas_to_test = [0.0, best_alphas[key], 1.0]
                # Deduplicate
                alphas_to_test = sorted(set(alphas_to_test))
            else:
                alphas_to_test = test_alphas

            print(f"\n{'='*70}")
            print(f"{model_name.upper()} / {cat_name.upper()} — Real Tester Eval")
            print(f"  Alphas: {alphas_to_test}")
            print(f"{'='*70}")

            models_orig, fusion_weights = load_multistream_models(
                ckpt_dir, classes, device, model_class, build_adj_fn)
            if len(models_orig) < 3:
                continue

            # Pre-extract AdaBN data from real testers
            adabn_data = preextract_test_data(cat_videos)

            results_for_key = {}
            for alpha in alphas_to_test:
                models = {s: copy.deepcopy(m) for s, m in models_orig.items()}

                if alpha == 0.0:
                    label = "no_adapt"
                elif alpha == 1.0:
                    label = "full_adabn"
                    for sname, model in models.items():
                        adapt_bn_stats(model, adabn_data, device, stream_name=sname)
                else:
                    label = f"edtn_{alpha}"
                    for sname, model in models.items():
                        apply_edtn(model, adabn_data, device, stream_name=sname, alpha=alpha)

                predict_fn = make_predict_fn_multistream(
                    models, fusion_weights, classes, device)
                result = evaluate_method(
                    f"{model_name}_{cat_name}_{label}", cat_videos, predict_fn, classes)
                results_for_key[label] = {
                    "accuracy": result["overall"],
                    "correct": result.get("correct", 0),
                    "total": result.get("total", 0),
                    "alpha": alpha,
                    "per_class": result.get("per_class", {}),
                    "per_signer": result.get("per_signer", {}),
                }
                del models

            all_results[key] = results_for_key

    # --- V27 ---
    for cat_name in ["numbers", "words"]:
        key = f"v27_{cat_name}"
        ckpt_path = os.path.join(CKPT_BASE, f"v27_{cat_name}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue

        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        cat_videos = numbers_videos if cat_name == "numbers" else words_videos
        if not cat_videos:
            continue

        if key in best_alphas:
            alphas_to_test = sorted(set([0.0, best_alphas[key], 1.0]))
        else:
            alphas_to_test = test_alphas

        print(f"\n{'='*70}")
        print(f"V27 / {cat_name.upper()} — Real Tester Eval")
        print(f"  Alphas: {alphas_to_test}")
        print(f"{'='*70}")

        model_orig = load_v27_model(ckpt_path, classes, device)

        # Pre-extract data for V27 AdaBN using raw video data
        from evaluate_real_testers_v27 import preprocess_landmarks
        v27_adabn_data = []
        for video_path, true_class, signer in cat_videos:
            npy_cache = video_path + ".landmarks.npy"
            if os.path.exists(npy_cache):
                raw = np.load(npy_cache)
            else:
                from evaluate_real_testers_v30 import extract_landmarks_from_video
                raw = extract_landmarks_from_video(video_path)
                if raw is None:
                    continue
            gcn_tensor, aux_tensor = preprocess_landmarks(raw)
            if gcn_tensor is not None:
                v27_adabn_data.append((gcn_tensor, aux_tensor))

        results_for_key = {}
        for alpha in alphas_to_test:
            model = copy.deepcopy(model_orig)

            if alpha == 0.0:
                label = "no_adapt"
            elif alpha == 1.0:
                label = "full_adabn"
                adapt_bn_v27(model, v27_adabn_data, device)
            else:
                label = f"edtn_{alpha}"
                apply_edtn_v27(model, v27_adabn_data, device, alpha=alpha)

            predict_fn = make_predict_fn_v27(model, classes, device)
            result = evaluate_method(
                f"v27_{cat_name}_{label}", cat_videos, predict_fn, classes)
            results_for_key[label] = {
                "accuracy": result["overall"],
                "correct": result.get("correct", 0),
                "total": result.get("total", 0),
                "alpha": alpha,
            }
            del model

        all_results[key] = results_for_key

    # --- V29 (single 9ch model, different architecture) ---
    for cat_name in ["numbers", "words"]:
        key = f"v29_{cat_name}"
        ckpt_path = os.path.join(CKPT_BASE, f"v29_{cat_name}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue

        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        cat_videos = numbers_videos if cat_name == "numbers" else words_videos
        if not cat_videos:
            continue

        if key in best_alphas:
            alphas_to_test = sorted(set([0.0, best_alphas[key], 1.0]))
        else:
            alphas_to_test = test_alphas

        print(f"\n{'='*70}")
        print(f"V29 / {cat_name.upper()} — Real Tester Eval")
        print(f"  Alphas: {alphas_to_test}")
        print(f"{'='*70}")

        model_orig = load_v29_model(ckpt_path, classes, device)

        # Pre-extract data for V29 AdaBN using raw video data
        from evaluate_real_testers_v29 import preprocess_landmarks as preprocess_v29
        v29_adabn_data = []
        for video_path, true_class, signer in cat_videos:
            npy_cache = video_path + ".landmarks.npy"
            if os.path.exists(npy_cache):
                raw = np.load(npy_cache)
            else:
                from evaluate_real_testers_v30 import extract_landmarks_from_video
                raw = extract_landmarks_from_video(video_path)
                if raw is None:
                    continue
            gcn_tensor, aux_tensor = preprocess_v29(raw)
            if gcn_tensor is not None:
                v29_adabn_data.append((gcn_tensor, aux_tensor))

        results_for_key = {}
        for alpha in alphas_to_test:
            model = copy.deepcopy(model_orig)

            if alpha == 0.0:
                label = "no_adapt"
            elif alpha == 1.0:
                label = "full_adabn"
                adapt_bn_v27(model, v29_adabn_data, device)
            else:
                label = f"edtn_{alpha}"
                apply_edtn_v27(model, v29_adabn_data, device, alpha=alpha)

            predict_fn = make_predict_fn_v29(model, classes, device)
            result = evaluate_method(
                f"v29_{cat_name}_{label}", cat_videos, predict_fn, classes)
            results_for_key[label] = {
                "accuracy": result["overall"],
                "correct": result.get("correct", 0),
                "total": result.get("total", 0),
                "alpha": alpha,
            }
            del model

        all_results[key] = results_for_key

    # --- OpenHands ---
    for cat_name in ["numbers", "words"]:
        key = f"openhands_{cat_name}"
        ckpt_path = os.path.join(CKPT_BASE, "openhands", cat_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue

        classes = NUMBER_CLASSES if cat_name == "numbers" else WORD_CLASSES
        cat_videos = numbers_videos if cat_name == "numbers" else words_videos
        if not cat_videos:
            continue

        if key in best_alphas:
            alphas_to_test = sorted(set([0.0, best_alphas[key], 1.0]))
        else:
            alphas_to_test = test_alphas

        print(f"\n{'='*70}")
        print(f"OPENHANDS / {cat_name.upper()} — Real Tester Eval")
        print(f"  Alphas: {alphas_to_test}")
        print(f"{'='*70}")

        model_orig = load_openhands_model(ckpt_path, len(classes), device)

        # Pre-extract raw data for OpenHands
        from evaluate_real_testers_v30 import extract_landmarks_from_video
        oh_raw_list = []
        for video_path, _, _ in cat_videos:
            npy_cache = video_path + ".landmarks.npy"
            if os.path.exists(npy_cache):
                raw = np.load(npy_cache).astype(np.float32)
            else:
                raw = extract_landmarks_from_video(video_path)
            oh_raw_list.append(raw)

        results_for_key = {}
        for alpha in alphas_to_test:
            model = copy.deepcopy(model_orig)

            if alpha == 0.0:
                label = "no_adapt"
            elif alpha == 1.0:
                label = "full_adabn"
                from evaluate_openhands_realtest import apply_adabn_global
                model = apply_adabn_global(model, oh_raw_list, device)
            else:
                label = f"edtn_{alpha}"
                apply_edtn_openhands(model, oh_raw_list, device, alpha=alpha)

            predict_fn = make_predict_fn_openhands(model, classes, device)
            result = evaluate_method(
                f"openhands_{cat_name}_{label}", cat_videos, predict_fn, classes)
            results_for_key[label] = {
                "accuracy": result["overall"],
                "correct": result.get("correct", 0),
                "total": result.get("total", 0),
                "alpha": alpha,
            }
            del model

        all_results[key] = results_for_key

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("PHASE 2 SUMMARY: EDTN on Real Testers")
    print(f"{'='*70}")

    for key, results in sorted(all_results.items()):
        best_label = max(results, key=lambda k: results[k]["accuracy"])
        best = results[best_label]
        print(f"\n  {key}:")
        for label, r in sorted(results.items(), key=lambda x: x[1]["alpha"]):
            marker = " <-- best" if label == best_label else ""
            print(f"    alpha={r['alpha']:.1f} ({label:>12s}): {r['accuracy']:.1f}%"
                  f" ({r['correct']}/{r['total']}){marker}")

    # Compute combined numbers/words for each model at each alpha
    print(f"\n  Combined accuracy per model:")
    model_names = set(k.rsplit("_", 1)[0] for k in all_results.keys())
    for mn in sorted(model_names):
        num_key = f"{mn}_numbers"
        word_key = f"{mn}_words"
        if num_key not in all_results or word_key not in all_results:
            continue

        # Find common labels
        num_results = all_results[num_key]
        word_results = all_results[word_key]
        common_labels = set(num_results.keys()) & set(word_results.keys())

        print(f"\n    {mn}:")
        for label in sorted(common_labels, key=lambda l: num_results[l]["alpha"]):
            nr = num_results[label]
            wr = word_results[label]
            combined_correct = nr["correct"] + wr["correct"]
            combined_total = nr["total"] + wr["total"]
            combined_acc = 100.0 * combined_correct / combined_total if combined_total > 0 else 0.0
            print(f"      alpha={nr['alpha']:.1f}: N={nr['accuracy']:.1f}% W={wr['accuracy']:.1f}% "
                  f"Combined={combined_acc:.1f}%")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V44 Expr 4: EDTN — Exponential Decay Test-time Normalization")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["tune", "eval", "both"],
                        help="Phase 1 (tune on test_alpha), Phase 2 (eval on real_testers), or both")
    args = parser.parse_args()

    print("=" * 70)
    print("V44 Experiment 4: EDTN — Exponential Decay Test-time Normalization")
    print(f"Phase: {args.phase}")
    print(f"Started: {ts()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_alphas = None
    phase1_results = None
    phase2_results = None

    if args.phase in ("tune", "both"):
        phase1_results, best_alphas = run_phase1_tune(device)

        # Save Phase 1 results
        p1_path = os.path.join(RESULTS_DIR, "phase1_tune_results.json")
        with open(p1_path, "w") as f:
            json.dump({"results": phase1_results, "best_alphas": best_alphas}, f, indent=2)
        print(f"\nPhase 1 results saved to {p1_path}")

    if args.phase in ("eval", "both"):
        # Load best_alphas from Phase 1 if not already available
        if best_alphas is None:
            p1_path = os.path.join(RESULTS_DIR, "phase1_tune_results.json")
            if os.path.exists(p1_path):
                with open(p1_path) as f:
                    p1_data = json.load(f)
                best_alphas = {k: v for k, v in p1_data.get("best_alphas", {}).items()}
                print(f"\nLoaded best alphas from Phase 1: {best_alphas}")
            else:
                print(f"\nNo Phase 1 results found, testing all alphas on real_testers")
                best_alphas = {}

        phase2_results = run_phase2_eval(device, best_alphas)

        # Save Phase 2 results
        p2_path = os.path.join(RESULTS_DIR, "phase2_eval_results.json")
        serializable = {}
        for key, results in phase2_results.items():
            serializable[key] = {}
            for label, r in results.items():
                sr = {k: v for k, v in r.items() if k not in ("per_class", "per_signer")}
                serializable[key][label] = sr
        with open(p2_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nPhase 2 results saved to {p2_path}")

    print(f"\nFinished: {ts()}")


if __name__ == "__main__":
    main()
