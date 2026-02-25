#!/usr/bin/env python3
"""Run the best v31 ensemble (v27+v28+v29+exp1+exp5) on the alpha test set.

Uses pre-extracted .npy landmark files for speed.
Alpha test set: signers 13-15, 450 videos (15 per class × 30 classes).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
)
from evaluate_real_testers_v30 import (
    KSLGraphNetV25, _load_fusion_weights,
    NUMBER_CLASSES, WORD_CLASSES, build_adj,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

import torch
import torch.nn.functional as F
import numpy as np
import copy
import json
import argparse
from datetime import datetime
from collections import defaultdict

ALPHA_TEST_DIR = "/scratch/alpine/hama5612/ksl-dir-2/data/test_alpha"


def discover_alpha_test_videos():
    """Discover .npy files in the alpha test directory.

    Returns list of (npy_path, class_name, category, signer_id) tuples.
    """
    videos = []
    for cls_dir in sorted(os.listdir(ALPHA_TEST_DIR)):
        cls_path = os.path.join(ALPHA_TEST_DIR, cls_dir)
        if not os.path.isdir(cls_path):
            continue

        # Determine category
        if cls_dir in [c for c in NUMBER_CLASSES]:
            category = "numbers"
        elif cls_dir in [c for c in WORD_CLASSES]:
            category = "words"
        else:
            print(f"  SKIP: Unknown class '{cls_dir}'")
            continue

        for fname in sorted(os.listdir(cls_path)):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(cls_path, fname)
            # Extract signer from filename like "Agreement-13-A342.npy"
            parts = fname.replace(".npy", "").split("-")
            signer_id = parts[1] if len(parts) >= 2 else "?"
            videos.append((npy_path, cls_dir, category, signer_id))

    return videos


def load_v31_exp1_models(category, device):
    """Load v31 exp1 (GroupNorm) models."""
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    ckpt_dir = f"data/checkpoints/v31_exp1/{category}"
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing v31_exp1/{sname}: {ckpt_path}")
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
        print(f"  Loaded v31_exp1/{sname}: val={ckpt.get('val_acc', 'N/A'):.1f}%")

    fw = _load_fusion_weights(ckpt_dir) if len(models) == 3 else None
    if fw is None and len(models) == 3:
        fw = {"joint": 0.33, "bone": 0.34, "velocity": 0.33}
    return models, fw


def load_v31_exp5_models(category, device):
    """Load v31 exp5 (SupCon) models."""
    ckpt_dir = f"data/checkpoints/v31_exp5/{category}"
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing v31_exp5/{sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        model = KSLGraphNetV25(
            nc=len(classes), num_signers=ckpt.get("num_signers", 12),
            aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v31_exp5/{sname}: val={ckpt.get('val_acc', 'N/A'):.1f}%")

    fw = _load_fusion_weights(ckpt_dir) if len(models) == 3 else None
    if fw is None and len(models) == 3:
        fw = {"joint": 0.33, "bone": 0.34, "velocity": 0.33}
    return models, fw


def get_v28_style_probs(streams, aux, models, fusion_weights, device, temperature=1.0):
    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, *_ = smodel(gcn, aux_t, grl_lambda=0.0)
            probs = F.softmax(logits / temperature, dim=1).cpu().squeeze(0)
            per_stream_probs[sname] = probs
    fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
    return fused


def get_groupnorm_probs(streams, aux, models, fusion_weights, device, temperature=1.0):
    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, *_ = smodel(gcn, aux_t)
            probs = F.softmax(logits / temperature, dim=1).cpu().squeeze(0)
            per_stream_probs[sname] = probs
    fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    alpha = args.alpha

    print("V31 Ensemble - Alpha Test Set Evaluation")
    print(f"Alpha-BN alpha={alpha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Discover test npys
    raw_videos = discover_alpha_test_videos()
    print(f"Found {len(raw_videos)} alpha test samples")

    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_items = [(p, cls, signer) for p, cls, cat, signer in raw_videos if cat == category]
        if not cat_items:
            continue

        signers = sorted(set(s for _, _, s in cat_items))
        print(f"\n{'='*70}")
        print(f"{category.upper()} ({len(cat_items)} samples, signers: {signers})")
        print(f"{'='*70}")

        # ---- Load all models ----
        ckpt_base = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"
        from evaluate_real_testers_v30_phase1 import (
            load_v27_model, load_v28_stream_models, load_v29_model,
        )

        v27_ckpt = os.path.join(ckpt_base, f"v27_{category}", "best_model.pt")
        v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

        v28_ckpt_dir = os.path.join(ckpt_base, f"v28_{category}")
        v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

        v29_ckpt = os.path.join(ckpt_base, f"v29_{category}", "best_model.pt")
        v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

        print("\n  Loading v31_exp1 (GroupNorm)...")
        exp1_models, exp1_fw = load_v31_exp1_models(category, device)

        print("\n  Loading v31_exp5 (SupCon)...")
        exp5_models, exp5_fw = load_v31_exp5_models(category, device)

        # ---- Adapt BN stats using alpha test data ----
        # Build video-like tuples for preextract: (npy_path, class, signer)
        # preextract_test_data expects (video_path, true_class, signer) and checks for .landmarks.npy
        # We need to feed raw numpy arrays directly

        # Collect all raw arrays for BN adaptation
        all_raw = []
        for npy_path, cls, signer in cat_items:
            raw = np.load(npy_path)
            all_raw.append(raw)

        # Preprocess for multistream adaptation
        stream_data_list = []
        for raw in all_raw:
            result = preprocess_multistream(raw)
            if result[0] is not None:
                stream_data_list.append(result)

        v27_data_list = []
        for raw in all_raw:
            result = preprocess_v27(raw)
            if result[0] is not None:
                v27_data_list.append(result)

        # V27: AdaBN
        if v27_model and v27_data_list:
            adapt_bn_stats(v27_model, v27_data_list, device)
            print("  v27: AdaBN Global applied")

        # V28: Alpha-BN
        if v28_models:
            for sname, smodel in v28_models.items():
                source_stats = save_bn_stats(smodel)
                target_stats = compute_target_bn_stats(
                    smodel, stream_data_list, device,
                    is_multistream=True, stream_name=sname,
                )
                apply_alpha_bn(smodel, source_stats, target_stats, alpha)
            print("  v28: Alpha-BN applied")

        # V29: AdaBN
        if v29_model and v27_data_list:
            adapt_bn_stats(v29_model, v27_data_list, device)
            print("  v29: AdaBN Global applied")

        # V31 Exp 1: No adaptation needed
        print("  v31_exp1: No adaptation needed (GroupNorm)")

        # V31 Exp 5: AdaBN Global
        if exp5_models:
            exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
            for sname in exp5_adabn:
                adapt_bn_stats(exp5_adabn[sname], stream_data_list, device,
                              is_multistream=True, stream_name=sname)
            print("  v31_exp5: AdaBN Global applied")

        # ---- Evaluate ----
        correct = 0
        total = 0
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        per_signer = defaultdict(lambda: {"correct": 0, "total": 0})
        details = []

        for npy_path, true_class, signer in cat_items:
            raw = np.load(npy_path)
            streams, aux = preprocess_multistream(raw)
            gcn_v27, aux_v27 = preprocess_v27(raw)

            all_probs = []

            # V27
            if v27_model and gcn_v27 is not None:
                with torch.no_grad():
                    logits, *_ = v27_model(
                        gcn_v27.unsqueeze(0).to(device),
                        aux_v27.unsqueeze(0).to(device),
                        grl_lambda=0.0,
                    )
                    all_probs.append(F.softmax(logits, dim=1).cpu().squeeze(0))

            # V28
            if v28_models and streams is not None:
                probs = get_v28_style_probs(streams, aux, v28_models, v28_fw, device)
                all_probs.append(probs)

            # V29
            if v29_model and gcn_v27 is not None:
                with torch.no_grad():
                    logits, *_ = v29_model(
                        gcn_v27.unsqueeze(0).to(device),
                        aux_v27.unsqueeze(0).to(device),
                        grl_lambda=0.0,
                    )
                    all_probs.append(F.softmax(logits, dim=1).cpu().squeeze(0))

            # V31 Exp1 (GroupNorm)
            if exp1_models and streams is not None:
                probs = get_groupnorm_probs(streams, aux, exp1_models, exp1_fw, device)
                all_probs.append(probs)

            # V31 Exp5 (SupCon)
            if exp5_models and streams is not None:
                probs = get_v28_style_probs(streams, aux, exp5_adabn, exp5_fw, device)
                all_probs.append(probs)

            if not all_probs:
                continue

            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            pred_idx = ensemble_probs.argmax().item()
            confidence = ensemble_probs[pred_idx].item()
            i2c = {i: c for i, c in enumerate(classes)}
            pred_class = i2c[pred_idx]
            is_correct = (pred_class == true_class)

            correct += int(is_correct)
            total += 1
            per_class[true_class]["correct"] += int(is_correct)
            per_class[true_class]["total"] += 1
            per_signer[signer]["correct"] += int(is_correct)
            per_signer[signer]["total"] += 1

            status = "OK" if is_correct else "WRONG"
            fname = os.path.basename(npy_path)
            if not is_correct:
                details.append(f"    {fname}: {pred_class} (conf={confidence:.3f}) | True: {true_class} [{status}]")

        acc = 100.0 * correct / total if total > 0 else 0
        print(f"\n  Ensemble accuracy: {correct}/{total} = {acc:.1f}%")

        # Per-class
        print(f"\n  Per-class accuracy:")
        print(f"  {'Class':>15s}  {'Acc':>6s}  {'N':>4s}")
        for cls in sorted(per_class.keys()):
            d = per_class[cls]
            cls_acc = 100.0 * d["correct"] / d["total"] if d["total"] > 0 else 0
            print(f"  {cls:>15s}  {cls_acc:5.0f}%  {d['total']:4d}")

        # Per-signer
        print(f"\n  Per-signer accuracy:")
        for sid in sorted(per_signer.keys()):
            d = per_signer[sid]
            s_acc = 100.0 * d["correct"] / d["total"] if d["total"] > 0 else 0
            print(f"    Signer {sid}: {s_acc:.0f}% ({d['correct']}/{d['total']})")

        # Wrong predictions
        if details:
            print(f"\n  Wrong predictions ({len(details)}):")
            for d in details:
                print(d)

        all_results[category] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "per_class": dict(per_class),
            "per_signer": dict(per_signer),
        }

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    n = all_results.get("numbers", {})
    w = all_results.get("words", {})
    n_acc = n.get("accuracy", 0)
    w_acc = w.get("accuracy", 0)
    n_c, n_t = n.get("correct", 0), n.get("total", 0)
    w_c, w_t = w.get("correct", 0), w.get("total", 0)
    combined = (n_c + w_c) / (n_t + w_t) * 100 if (n_t + w_t) > 0 else 0
    print(f"  Numbers: {n_c}/{n_t} = {n_acc:.1f}%")
    print(f"  Words:   {w_c}/{w_t} = {w_acc:.1f}%")
    print(f"  Combined: {n_c + w_c}/{n_t + w_t} = {combined:.1f}%")
    print(f"\n  Reference: Real-world testers = N=71.2%, W=70.4%, Combined=70.7%")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/results/v31_ensemble_alpha_test_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
