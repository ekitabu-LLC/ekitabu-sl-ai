#!/usr/bin/env python3
"""Run the best v31 ensemble (v27+v28+v29+exp1+exp5) on ALL ksl-alpha videos.

Uses pre-extracted .npy landmark files.
- train_alpha: signers 1-10 (1487 videos) — SEEN during training
- val_alpha: signers 11-12 (300 videos) — SEEN during training (v27+)
- test_alpha: signers 13-15 (450 videos) — UNSEEN
- real_testers: 3 signers (140 videos) — UNSEEN, different recording conditions
Total: ~2377 videos
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    adapt_bn_stats, save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
    load_v27_model, load_v28_stream_models, load_v29_model,
    discover_test_videos, extract_landmarks_from_video,
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

DATA_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data"
REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

SPLIT_DIRS = {
    "train": os.path.join(DATA_BASE, "train_alpha"),
    "val": os.path.join(DATA_BASE, "val_alpha"),
    "test": os.path.join(DATA_BASE, "test_alpha"),
}

SPLIT_SIGNERS = {
    "train": [str(i) for i in [1,2,3,4,5,6,7,8,9,10]],
    "val": ["11", "12"],
    "test": ["13", "14", "15"],
    "real_testers": ["1. Signer", "2. Signer", "3. Signer"],
}


def discover_npy_videos(data_dir):
    """Discover .npy files. Returns (npy_path, class_name, category, signer_id)."""
    videos = []
    for cls_dir in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        if cls_dir in NUMBER_CLASSES:
            category = "numbers"
        elif cls_dir in WORD_CLASSES:
            category = "words"
        else:
            continue
        for fname in sorted(os.listdir(cls_path)):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(cls_path, fname)
            parts = fname.replace(".npy", "").split("-")
            signer_id = parts[1] if len(parts) >= 2 else "?"
            videos.append((npy_path, cls_dir, category, signer_id))
    return videos


def load_v31_exp1_models(category, device):
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    ckpt_dir = f"data/checkpoints/v31_exp1/{category}"
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES
    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
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
    fw = _load_fusion_weights(ckpt_dir) if len(models) == 3 else None
    if fw is None and len(models) == 3:
        fw = {"joint": 0.33, "bone": 0.34, "velocity": 0.33}
    print(f"  v31_exp1: {len(models)} streams loaded")
    return models, fw


def load_v31_exp5_models(category, device):
    ckpt_dir = f"data/checkpoints/v31_exp5/{category}"
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES
    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
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
    fw = _load_fusion_weights(ckpt_dir) if len(models) == 3 else None
    if fw is None and len(models) == 3:
        fw = {"joint": 0.33, "bone": 0.34, "velocity": 0.33}
    print(f"  v31_exp5: {len(models)} streams loaded")
    return models, fw


def get_fused_probs(streams, aux, models, fw, device, use_grl=True):
    per_stream = {}
    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            if use_grl:
                logits, *_ = smodel(gcn, aux_t, grl_lambda=0.0)
            else:
                logits, *_ = smodel(gcn, aux_t)
            per_stream[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
    return sum(fw[s] * per_stream[s] for s in models)


def predict_one(raw, classes, device, v27_model, v28_models, v28_fw,
                v29_model, exp1_models, exp1_fw, exp5_models, exp5_fw):
    """Run ensemble prediction on a single raw (T, 549) numpy array."""
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
        all_probs.append(get_fused_probs(streams, aux, v28_models, v28_fw, device))

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
        all_probs.append(get_fused_probs(streams, aux, exp1_models, exp1_fw, device, use_grl=False))

    # V31 Exp5 (SupCon)
    if exp5_models and streams is not None:
        all_probs.append(get_fused_probs(streams, aux, exp5_models, exp5_fw, device))

    if not all_probs:
        return None, 0.0

    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    pred_idx = ensemble_probs.argmax().item()
    i2c = {i: c for i, c in enumerate(classes)}
    return i2c[pred_idx], ensemble_probs[pred_idx].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    print("V31 Ensemble - ALL VIDEOS Evaluation")
    print(f"Alpha-BN alpha={args.alpha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Discover all videos ----
    all_videos = {}  # split -> list of (npy_path, class_name, category, signer_id)
    for split_name, split_dir in SPLIT_DIRS.items():
        vids = discover_npy_videos(split_dir)
        all_videos[split_name] = vids
        print(f"  {split_name}: {len(vids)} npy files")

    # Real testers (video-based, need landmark extraction)
    rt_raw = discover_test_videos(REAL_TESTER_DIR)
    rt_videos = [(vp, name, cat, signer) for vp, name, cat, signer in rt_raw
                 if cat in ("numbers", "words")]
    print(f"  real_testers: {len(rt_videos)} videos")
    total = sum(len(v) for v in all_videos.values()) + len(rt_videos)
    print(f"  TOTAL: {total} videos\n")

    # ---- Per-category evaluation ----
    results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        print(f"\n{'='*70}")
        print(f"LOADING MODELS: {category.upper()}")
        print(f"{'='*70}")

        ckpt_base = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

        v27_ckpt = os.path.join(ckpt_base, f"v27_{category}", "best_model.pt")
        v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

        v28_ckpt_dir = os.path.join(ckpt_base, f"v28_{category}")
        v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

        v29_ckpt = os.path.join(ckpt_base, f"v29_{category}", "best_model.pt")
        v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

        exp1_models, exp1_fw = load_v31_exp1_models(category, device)
        exp5_models, exp5_fw = load_v31_exp5_models(category, device)

        # ---- Adapt BN using test_alpha data (unseen signers) ----
        # We adapt on test_alpha only (not train/val) to keep it fair
        test_items = [(p, c, s) for p, c, cat, s in all_videos["test"] if cat == category]
        test_raws = [np.load(p) for p, _, _ in test_items]

        stream_data = [preprocess_multistream(r) for r in test_raws]
        stream_data = [x for x in stream_data if x[0] is not None]
        v27_data = [preprocess_v27(r) for r in test_raws]
        v27_data = [x for x in v27_data if x[0] is not None]

        if v27_model and v27_data:
            adapt_bn_stats(v27_model, v27_data, device)
        if v28_models:
            for sname, smodel in v28_models.items():
                src = save_bn_stats(smodel)
                tgt = compute_target_bn_stats(smodel, stream_data, device,
                                              is_multistream=True, stream_name=sname)
                apply_alpha_bn(smodel, src, tgt, args.alpha)
        if v29_model and v27_data:
            adapt_bn_stats(v29_model, v27_data, device)
        exp5_adabn = {}
        if exp5_models:
            exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
            for sname in exp5_adabn:
                adapt_bn_stats(exp5_adabn[sname], stream_data, device,
                              is_multistream=True, stream_name=sname)
        print("  BN adaptation done (using test_alpha signers 13-15)")

        # ---- Evaluate all splits ----
        cat_results = {}

        for split_name in ["train", "val", "test"]:
            items = [(p, c, s) for p, c, cat, s in all_videos[split_name] if cat == category]
            if not items:
                continue

            correct = 0
            total_count = 0
            per_signer = defaultdict(lambda: {"correct": 0, "total": 0})
            per_class = defaultdict(lambda: {"correct": 0, "total": 0})
            wrong = []

            for npy_path, true_class, signer in items:
                raw = np.load(npy_path)
                pred, conf = predict_one(
                    raw, classes, device,
                    v27_model, v28_models, v28_fw,
                    v29_model, exp1_models, exp1_fw,
                    exp5_adabn if exp5_models else None, exp5_fw,
                )
                if pred is None:
                    continue

                is_correct = (pred == true_class)
                correct += int(is_correct)
                total_count += 1
                per_signer[signer]["correct"] += int(is_correct)
                per_signer[signer]["total"] += 1
                per_class[true_class]["correct"] += int(is_correct)
                per_class[true_class]["total"] += 1
                if not is_correct:
                    wrong.append((os.path.basename(npy_path), pred, conf, true_class))

            acc = 100.0 * correct / total_count if total_count > 0 else 0
            cat_results[split_name] = {
                "accuracy": acc, "correct": correct, "total": total_count,
                "per_signer": {k: dict(v) for k, v in per_signer.items()},
                "per_class": {k: dict(v) for k, v in per_class.items()},
                "wrong_count": len(wrong),
            }

            print(f"\n  {split_name.upper()} ({category}): {correct}/{total_count} = {acc:.1f}%")
            for sid in sorted(per_signer.keys(), key=lambda x: int(x) if x.isdigit() else 99):
                d = per_signer[sid]
                s_acc = 100.0 * d["correct"] / d["total"] if d["total"] > 0 else 0
                print(f"    Signer {sid}: {s_acc:.0f}% ({d['correct']}/{d['total']})")

            # Show classes with <100% accuracy
            weak = [(c, d) for c, d in per_class.items()
                    if d["total"] > 0 and d["correct"] < d["total"]]
            if weak:
                weak.sort(key=lambda x: x[1]["correct"]/x[1]["total"])
                print(f"    Weak classes:")
                for c, d in weak:
                    ca = 100.0 * d["correct"] / d["total"]
                    print(f"      {c}: {ca:.0f}% ({d['correct']}/{d['total']})")

        # Real testers
        rt_items = [(vp, name, signer) for vp, name, cat, signer in rt_videos if cat == category]
        if rt_items:
            correct = 0
            total_count = 0
            per_signer = defaultdict(lambda: {"correct": 0, "total": 0})
            wrong = []

            for vp, true_class, signer in rt_items:
                npy_cache = vp + ".landmarks.npy"
                if os.path.exists(npy_cache):
                    raw = np.load(npy_cache)
                else:
                    raw = extract_landmarks_from_video(vp)
                    if raw is None:
                        continue

                pred, conf = predict_one(
                    raw, classes, device,
                    v27_model, v28_models, v28_fw,
                    v29_model, exp1_models, exp1_fw,
                    exp5_adabn if exp5_models else None, exp5_fw,
                )
                if pred is None:
                    continue

                is_correct = (pred == true_class)
                correct += int(is_correct)
                total_count += 1
                per_signer[signer]["correct"] += int(is_correct)
                per_signer[signer]["total"] += 1
                if not is_correct:
                    wrong.append((os.path.basename(vp), pred, conf, true_class))

            acc = 100.0 * correct / total_count if total_count > 0 else 0
            cat_results["real_testers"] = {
                "accuracy": acc, "correct": correct, "total": total_count,
                "per_signer": {k: dict(v) for k, v in per_signer.items()},
                "wrong_count": len(wrong),
            }
            print(f"\n  REAL TESTERS ({category}): {correct}/{total_count} = {acc:.1f}%")
            for sid in sorted(per_signer.keys()):
                d = per_signer[sid]
                s_acc = 100.0 * d["correct"] / d["total"] if d["total"] > 0 else 0
                print(f"    {sid}: {s_acc:.0f}% ({d['correct']}/{d['total']})")

        results[category] = cat_results

    # ---- Grand Summary ----
    print(f"\n{'='*70}")
    print(f"GRAND SUMMARY")
    print(f"{'='*70}")
    print(f"{'Split':<15s} {'Numbers':>10s} {'Words':>10s} {'Combined':>10s}")
    print("-" * 50)
    for split in ["train", "val", "test", "real_testers"]:
        n = results.get("numbers", {}).get(split, {})
        w = results.get("words", {}).get(split, {})
        nc, nt = n.get("correct", 0), n.get("total", 0)
        wc, wt = w.get("correct", 0), w.get("total", 0)
        na = n.get("accuracy", 0)
        wa = w.get("accuracy", 0)
        ca = (nc + wc) / (nt + wt) * 100 if (nt + wt) > 0 else 0
        label = split.replace("_", " ").title()
        print(f"{label:<15s} {na:>9.1f}% {wa:>9.1f}% {ca:>9.1f}%")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/results/v31_ensemble_all_videos_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
