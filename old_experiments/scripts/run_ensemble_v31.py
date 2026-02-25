#!/usr/bin/env python3
"""Ensemble evaluation adding v31 models to the v27+v28+v29 ensemble.

Tests multiple ensemble combinations:
  1. v27+v28+v29 (baseline ensemble, should match ~65.9% with Alpha-BN)
  2. v27+v28+v29+v31_exp5 (add SupCon)
  3. v27+v28+v29+v31_exp1 (add GroupNorm)
  4. v27+v28+v29+v31_exp1+v31_exp5 (all 5 models)
  5. v31_exp1+v31_exp5 only (just the v31 winners)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    preprocess_v27, preprocess_multistream,
    evaluate_method,
)
from evaluate_real_testers_v30 import (
    KSLGraphNetV25, _load_fusion_weights,
    NUMBER_CLASSES, WORD_CLASSES, build_adj,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
)

REAL_TESTER_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import argparse
from datetime import datetime


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

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def load_v31_exp5_models(category, device):
    """Load v31 exp5 (SupCon) models — same architecture as v28."""
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
        print(f"  Loaded v31_exp5/{sname}: val={ckpt.get('val_acc', 'N/A'):.1f}%")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


def get_v28_style_probs(streams, aux, models, fusion_weights, device, temperature=1.0):
    """Get fused softmax probs from a multi-stream v28-style model."""
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
    """Get fused softmax probs from GroupNorm model (no grl_lambda)."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Alpha-BN alpha={alpha}")
    print()

    raw_videos = discover_test_videos(REAL_TESTER_DIR)
    # discover_test_videos returns (video_path, name, category, signer_dir)
    # evaluate_method expects (video_path, true_class, signer)
    print(f"Found {len(raw_videos)} test videos")

    all_results = {}

    for category, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        cat_videos = [(vp, name, signer) for vp, name, cat, signer in raw_videos if cat == category]
        if not cat_videos:
            continue

        print(f"\n{'='*70}")
        print(f"{category.upper()} ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        # ---- Load all models ----
        ckpt_base = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"

        from evaluate_real_testers_v30_phase1 import (
            load_v27_model, load_v28_stream_models, load_v29_model,
        )

        # V27
        v27_ckpt = os.path.join(ckpt_base, f"v27_{category}", "best_model.pt")
        v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None

        # V28
        v28_ckpt_dir = os.path.join(ckpt_base, f"v28_{category}")
        v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir, classes, device) if os.path.isdir(v28_ckpt_dir) else (None, None)

        # V29
        v29_ckpt = os.path.join(ckpt_base, f"v29_{category}", "best_model.pt")
        v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None

        # V31 Exp 1 (GroupNorm)
        print("\n  Loading v31_exp1 (GroupNorm)...")
        exp1_models, exp1_fw = load_v31_exp1_models(category, device)

        # V31 Exp 5 (SupCon)
        print("\n  Loading v31_exp5 (SupCon)...")
        exp5_models, exp5_fw = load_v31_exp5_models(category, device)

        # ---- Pre-extract test data ----
        stream_data = preextract_test_data(cat_videos, preprocess_fn="multistream")

        # ---- Adapt BN stats ----
        # V27: standard AdaBN
        if v27_model:
            v27_data = preextract_test_data(cat_videos, preprocess_fn="v27")
            adapt_bn_stats(v27_model, v27_data, device)

        # V28: Alpha-BN
        if v28_models:
            for sname, smodel in v28_models.items():
                source_stats = save_bn_stats(smodel)
                target_stats = compute_target_bn_stats(
                    smodel, stream_data, device,
                    is_multistream=True, stream_name=sname,
                )
                apply_alpha_bn(smodel, source_stats, target_stats, alpha)

        # V29: standard AdaBN
        if v29_model:
            v29_data = preextract_test_data(cat_videos, preprocess_fn="v27")
            adapt_bn_stats(v29_model, v29_data, device)

        # V31 Exp 1: NO adaptation needed (GroupNorm)
        print("  v31_exp1: No adaptation needed (GroupNorm)")

        # V31 Exp 5: AdaBN Global
        if exp5_models:
            exp5_adabn = {s: copy.deepcopy(m) for s, m in exp5_models.items()}
            for sname in exp5_adabn:
                adapt_bn_stats(exp5_adabn[sname], stream_data, device,
                              is_multistream=True, stream_name=sname)
            print("  v31_exp5: AdaBN Global applied")

        # ---- Define ensemble combinations ----
        cat_results = {}

        def make_predict_fn(model_list):
            """Create a predict function for given list of (name, get_probs_fn) tuples."""
            def predict_fn(raw, true_class):
                streams, aux = preprocess_multistream(raw)
                gcn_v27, aux_v27 = preprocess_v27(raw)
                all_probs = []

                for name, get_probs in model_list:
                    try:
                        if name in ("v27", "v29"):
                            if gcn_v27 is None:
                                continue
                            model = get_probs  # it's just the model
                            with torch.no_grad():
                                logits, *_ = model(
                                    gcn_v27.unsqueeze(0).to(device),
                                    aux_v27.unsqueeze(0).to(device),
                                    grl_lambda=0.0,
                                )
                                probs = F.softmax(logits, dim=1).cpu().squeeze(0)
                                all_probs.append(probs)
                        elif name == "v31_exp1":
                            if streams is None:
                                continue
                            models, fw = get_probs
                            probs = get_groupnorm_probs(streams, aux, models, fw, device)
                            all_probs.append(probs)
                        else:  # v28, v31_exp5
                            if streams is None:
                                continue
                            models, fw = get_probs
                            probs = get_v28_style_probs(streams, aux, models, fw, device)
                            all_probs.append(probs)
                    except Exception as e:
                        print(f"    WARNING: {name} failed: {e}")
                        continue

                if not all_probs:
                    return None

                ensemble_probs = torch.stack(all_probs).mean(dim=0)
                pred_idx = ensemble_probs.argmax().item()
                i2c = {i: c for i, c in enumerate(classes)}
                return i2c[pred_idx], ensemble_probs[pred_idx].item()

            return predict_fn

        # Ensemble configs to test
        ensembles = {
            "v27+v28+v29": [
                ("v27", v27_model),
                ("v28", (v28_models, v28_fw)),
                ("v29", v29_model),
            ],
            "v27+v28+v29+exp5": [
                ("v27", v27_model),
                ("v28", (v28_models, v28_fw)),
                ("v29", v29_model),
                ("v31_exp5", (exp5_adabn, exp5_fw)),
            ],
            "v27+v28+v29+exp1": [
                ("v27", v27_model),
                ("v28", (v28_models, v28_fw)),
                ("v29", v29_model),
                ("v31_exp1", (exp1_models, exp1_fw)),
            ],
            "v27+v28+v29+exp1+exp5": [
                ("v27", v27_model),
                ("v28", (v28_models, v28_fw)),
                ("v29", v29_model),
                ("v31_exp1", (exp1_models, exp1_fw)),
                ("v31_exp5", (exp5_adabn, exp5_fw)),
            ],
            "exp1+exp5": [
                ("v31_exp1", (exp1_models, exp1_fw)),
                ("v31_exp5", (exp5_adabn, exp5_fw)),
            ],
        }

        for ens_name, model_list in ensembles.items():
            print(f"\n--- Ensemble: {ens_name} ---")
            predict_fn = make_predict_fn(model_list)
            result = evaluate_method(ens_name, cat_videos, predict_fn, classes)
            cat_results[ens_name] = result

        all_results[category] = cat_results

        # Print category summary
        print(f"\n{'='*70}")
        print(f"{category.upper()} SUMMARY")
        print(f"{'='*70}")
        for ens_name, result in cat_results.items():
            acc = result["overall"]
            c = result["correct"]
            t = result["total"]
            print(f"  {ens_name:35s}: {c}/{t} = {acc:.1f}%")

    # Print overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  Reference: V30 Ensemble+Alpha-BN = Numbers 62.7%, Words 69.1%, Combined 65.9%")
    print()
    for ens_name in ensembles:
        nums = all_results.get("numbers", {}).get(ens_name, {})
        wrds = all_results.get("words", {}).get(ens_name, {})
        n_acc = nums.get("overall", 0)
        w_acc = wrds.get("overall", 0)
        n_c, n_t = nums.get("correct", 0), nums.get("total", 0)
        w_c, w_t = wrds.get("correct", 0), wrds.get("total", 0)
        combined = (n_c + w_c) / (n_t + w_t) * 100 if (n_t + w_t) > 0 else 0
        print(f"  {ens_name:35s}: N={n_acc:.1f}%, W={w_acc:.1f}%, Combined={combined:.1f}%")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/results/v31_ensemble_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
