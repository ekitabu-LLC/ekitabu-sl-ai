#!/usr/bin/env python3
"""
V31 Ablation Evaluation - Evaluate any v31 experiment on real testers.

Handles different model types:
  - exp1 (GroupNorm): KSLGraphNetV31Exp1, no AdaBN needed
  - exp2 (Prototypical): ProtoNetEncoder, prototype-based classification
  - exp3/4/5 (v28 variants): KSLGraphNetV25, standard AdaBN Global

Usage:
    python evaluate_v31_ablation.py --exp 1 --checkpoint-dir data/checkpoints/v31_exp1
    python evaluate_v31_ablation.py --exp 2 --checkpoint-dir data/checkpoints/v31_exp2
    python evaluate_v31_ablation.py --exp 3  # uses default checkpoint dir
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

# Import shared infrastructure from v30 eval
from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES,
    check_mediapipe_version,
    build_adj,
    discover_test_videos,
    preprocess_multistream,
    preextract_test_data,
    adapt_bn_stats,
    evaluate_method,
    confidence_level,
    _load_fusion_weights,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)


# ---------------------------------------------------------------------------
# Model loaders for each experiment type
# ---------------------------------------------------------------------------

def load_v28_style_models(ckpt_dir, classes, device):
    """Load v28-style models (exp3, exp4, exp5) — same as v28."""
    from evaluate_real_testers_v30 import KSLGraphNetV25, load_v28_stream_models
    # v28 load function expects v28_{cat} subdir format, but v31 uses {cat} directly
    # So we use the same loading logic but with our ckpt_dir
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = KSLGraphNetV25(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_groupnorm_models(ckpt_dir, classes, device):
    """Load exp1 GroupNorm models."""
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
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

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded GroupNorm/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_proto_models(ckpt_dir, classes, device):
    """Load exp2 prototypical models."""
    from train_ksl_v31_exp2 import ProtoNetEncoder
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Missing {sname}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)

        model = ProtoNetEncoder(
            aux_dim=aux_dim,
            nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
            hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
            dr=config.get("dropout", 0.3),
            spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded Proto/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


# ---------------------------------------------------------------------------
# Prototypical evaluation helpers
# ---------------------------------------------------------------------------

def compute_prototypes(models, train_dir, classes, device):
    """Compute class prototypes from training data for prototypical evaluation."""
    c2i = {c: i for i, c in enumerate(classes)}
    # Collect embeddings per class per stream
    stream_embeds = {sname: {i: [] for i in range(len(classes))} for sname in models}

    for cls_name in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in c2i:
            continue
        cls_idx = c2i[cls_name]
        for fn in sorted(os.listdir(cls_path)):
            if not fn.endswith(".npy"):
                continue
            raw = np.load(os.path.join(cls_path, fn)).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] < 225:
                continue
            streams, aux = preprocess_multistream(raw)
            if streams is None:
                continue
            with torch.no_grad():
                for sname, smodel in models.items():
                    gcn = streams[sname].unsqueeze(0).to(device)
                    embed = smodel(gcn, aux.unsqueeze(0).to(device))
                    # ProtoNetEncoder returns embedding directly
                    if isinstance(embed, tuple):
                        embed = embed[-1]  # last element is embedding
                    stream_embeds[sname][cls_idx].append(embed.cpu().squeeze(0))

    # Average to get prototypes
    prototypes = {}
    for sname in models:
        proto = torch.zeros(len(classes), stream_embeds[sname][0][0].shape[0])
        for i in range(len(classes)):
            if stream_embeds[sname][i]:
                proto[i] = torch.stack(stream_embeds[sname][i]).mean(0)
        prototypes[sname] = proto.to(device)
    return prototypes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V31 Ablation Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Experiment number")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory")
    parser.add_argument("--category", type=str, default="numbers",
                        choices=["numbers", "words", "both"])
    args = parser.parse_args()

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(base_data, "checkpoints", f"v31_exp{args.exp}")

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
    train_dirs = [os.path.join(base_data, "train_alpha"), os.path.join(base_data, "val_alpha")]

    exp_names = {
        1: "GroupNorm (no BN domain gap)",
        2: "Prototypical Networks",
        3: "Joint Mixing Augmentation",
        4: "Hard-Example Mining + Focal Loss",
        5: "Cross-Signer Contrastive Loss",
    }

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"V31 Exp {args.exp}: {exp_names[args.exp]}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Category:   {args.category}")
    print(f"Started:    {ts}")
    print("=" * 70)

    check_mediapipe_version()

    # Discover videos
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

        cat_ckpt_dir = os.path.join(args.checkpoint_dir, cat_name)
        if not os.path.isdir(cat_ckpt_dir):
            print(f"\nSkipping {cat_name}: checkpoint dir not found: {cat_ckpt_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} EVALUATION ({len(cat_videos)} videos)")
        print(f"Exp {args.exp}: {exp_names[args.exp]}")
        print(f"{'='*70}")

        # Load models based on experiment type
        print(f"\n  Loading models from {cat_ckpt_dir}...")
        if args.exp == 1:
            models, fusion_weights = load_groupnorm_models(cat_ckpt_dir, classes, device)
        elif args.exp == 2:
            models, fusion_weights = load_proto_models(cat_ckpt_dir, classes, device)
        else:
            models, fusion_weights = load_v28_style_models(cat_ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  WARNING: Only {len(models)}/3 stream models loaded, skipping")
            continue

        # Pre-extract test data
        stream_data = preextract_test_data(cat_videos)
        print(f"  Pre-extracted {len(stream_data)} samples")

        cat_results = {}

        # --- BASELINE (no adaptation) ---
        if args.exp == 2:
            # Prototypical: compute prototypes from training data
            print(f"\n--- PROTOTYPICAL CLASSIFICATION ---")
            print(f"  Computing prototypes from training data...")
            prototypes = {}
            for td in train_dirs:
                p = compute_prototypes(models, td, classes, device)
                for sname in p:
                    if sname not in prototypes:
                        prototypes[sname] = []
                    prototypes[sname].append(p[sname])
            # Average prototypes across train dirs
            for sname in prototypes:
                prototypes[sname] = torch.stack(prototypes[sname]).mean(0)
            print(f"  Prototypes computed for {len(prototypes)} streams")

            def proto_predict(raw, true_class, _models=models, _fw=fusion_weights,
                            _protos=prototypes, _classes=classes):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(_classes)}
                per_stream_probs = {}
                with torch.no_grad():
                    for sname, smodel in _models.items():
                        gcn = streams[sname].unsqueeze(0).to(device)
                        embed = smodel(gcn, aux.unsqueeze(0).to(device))
                        if isinstance(embed, tuple):
                            embed = embed[-1]
                        # Negative Euclidean distance to prototypes
                        dists = -torch.cdist(embed, _protos[sname].unsqueeze(0)).squeeze(0)
                        per_stream_probs[sname] = F.softmax(dists, dim=1).cpu().squeeze(0)
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results["prototypical"] = evaluate_method(
                f"v31_exp2_proto", cat_videos, proto_predict, classes
            )
        else:
            # Standard baseline
            print(f"\n--- BASELINE (no AdaBN) ---")
            def baseline_predict(raw, true_class, _models=models, _fw=fusion_weights):
                streams, aux = preprocess_multistream(raw)
                if streams is None:
                    return None
                i2c = {i: c for i, c in enumerate(classes)}
                per_stream_probs = {}
                with torch.no_grad():
                    for sname, smodel in _models.items():
                        gcn = streams[sname].unsqueeze(0).to(device)
                        logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                        per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
                fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                pred_idx = fused.argmax().item()
                return i2c[pred_idx], fused[pred_idx].item()

            cat_results["baseline"] = evaluate_method(
                f"v31_exp{args.exp}_baseline", cat_videos, baseline_predict, classes
            )

        # --- AdaBN GLOBAL (skip for exp1 GroupNorm, exp2 Proto — incompatible forward sig) ---
        if args.exp not in (1, 2):
            has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                        for model in models.values() for m in model.modules())
            if has_bn:
                print(f"\n--- AdaBN GLOBAL ---")
                adabn_models = {s: copy.deepcopy(m) for s, m in models.items()}
                for sname in adabn_models:
                    print(f"  Adapting BN stats for {sname} stream...")
                    adapt_bn_stats(adabn_models[sname], stream_data, device, stream_name=sname)
                    print(f"    Adapted on {len(stream_data)} samples")

                if args.exp == 2:
                    # Prototypical with AdaBN
                    def adabn_proto_predict(raw, true_class, _models=adabn_models,
                                          _fw=fusion_weights, _protos=prototypes, _classes=classes):
                        streams, aux = preprocess_multistream(raw)
                        if streams is None:
                            return None
                        i2c = {i: c for i, c in enumerate(_classes)}
                        per_stream_probs = {}
                        with torch.no_grad():
                            for sname, smodel in _models.items():
                                gcn = streams[sname].unsqueeze(0).to(device)
                                embed = smodel(gcn, aux.unsqueeze(0).to(device))
                                if isinstance(embed, tuple):
                                    embed = embed[-1]
                                dists = -torch.cdist(embed, _protos[sname].unsqueeze(0)).squeeze(0)
                                per_stream_probs[sname] = F.softmax(dists, dim=1).cpu().squeeze(0)
                        fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                        pred_idx = fused.argmax().item()
                        return i2c[pred_idx], fused[pred_idx].item()

                    cat_results["adabn_proto"] = evaluate_method(
                        f"v31_exp2_adabn_proto", cat_videos, adabn_proto_predict, classes
                    )
                else:
                    def adabn_predict(raw, true_class, _models=adabn_models, _fw=fusion_weights):
                        streams, aux = preprocess_multistream(raw)
                        if streams is None:
                            return None
                        i2c = {i: c for i, c in enumerate(classes)}
                        per_stream_probs = {}
                        with torch.no_grad():
                            for sname, smodel in _models.items():
                                gcn = streams[sname].unsqueeze(0).to(device)
                                logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                                per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
                        fused = sum(_fw[s] * per_stream_probs[s] for s in _models)
                        pred_idx = fused.argmax().item()
                        return i2c[pred_idx], fused[pred_idx].item()

                    cat_results["adabn_global"] = evaluate_method(
                        f"v31_exp{args.exp}_adabn_global", cat_videos, adabn_predict, classes
                    )
            else:
                print(f"\n  No BatchNorm layers found, skipping AdaBN")

        all_results[cat_name] = cat_results

    # --- FINAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - V31 Exp {args.exp}: {exp_names[args.exp]}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"{'='*70}")

    print(f"\n  Reference baselines:")
    print(f"    V28 AdaBN Global:          Numbers=55.9%, Words=60.5%")
    print(f"    V30 Ensemble+Alpha-BN:     Numbers=62.7%, Words=69.1%\n")

    for cat_name, result in all_results.items():
        parts = []
        for method, data in result.items():
            parts.append(f"{method}={data['overall']:.1f}%")
        print(f"  {cat_name.upper():>10s}: {', '.join(parts)}")

    # Save results
    results_dir = os.path.join(base_data, "results", "v31")
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"exp{args.exp}_{ts_str}.json")

    save_data = {
        "experiment": args.exp,
        "experiment_name": exp_names[args.exp],
        "checkpoint_dir": args.checkpoint_dir,
        "timestamp": ts_str,
        "results": {},
    }
    for cat_name, result in all_results.items():
        save_data["results"][cat_name] = {
            method: {
                "accuracy": data["overall"],
                "per_class": data.get("per_class", {}),
                "per_signer": data.get("per_signer", {}),
            }
            for method, data in result.items()
        }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
