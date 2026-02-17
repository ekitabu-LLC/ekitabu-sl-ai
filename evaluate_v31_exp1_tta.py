#!/usr/bin/env python3
"""
V31 Exp1 TTA Evaluation - Test-Time Augmentation for GroupNorm Model

GroupNorm means no BN stats to worry about - TTA is straightforward.
For each test video, we run multiple augmentation variants and average
the softmax predictions.

Augmentations:
  Temporal: original, reversed, speed 0.8x, speed 1.2x
  Spatial: original, mirrored (swap L/R hands), small rotation (+10, -10 deg)

Total: 4 temporal x 3 spatial = 12 variants per video.

Reports results with and without TTA for comparison.

Usage:
    python evaluate_v31_exp1_tta.py --category both
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_real_testers_v30 import (
    NUMBER_CLASSES, WORD_CLASSES,
    check_mediapipe_version,
    build_adj,
    discover_test_videos,
    evaluate_method,
    _load_fusion_weights,
    normalize_wrist_palm,
    compute_bones, compute_joint_angles,
    compute_fingertip_distances, compute_hand_body_features,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
    POSE_INDICES, MAX_FRAMES,
    extract_landmarks_from_video,
    confidence_level,
)
from train_ksl_v31_exp1 import KSLGraphNetV31Exp1


# ---------------------------------------------------------------------------
# Raw landmark extraction (same as preprocess_multistream but returns raw h)
# ---------------------------------------------------------------------------

def extract_48node(raw):
    """Extract 48-node skeleton from raw MediaPipe landmarks."""
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None
    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)
    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)
    return h


# ---------------------------------------------------------------------------
# TTA Augmentation Functions (operate on raw 48-node skeleton before norm)
# ---------------------------------------------------------------------------

def temporal_reverse(h):
    """Reverse the temporal order of frames."""
    # Find the last non-zero frame
    frame_norms = np.abs(h).sum(axis=(1, 2))
    valid = np.where(frame_norms > 0.01)[0]
    if len(valid) == 0:
        return h.copy()
    last_valid = valid[-1] + 1
    h_out = h.copy()
    h_out[:last_valid] = h_out[:last_valid][::-1]
    return h_out


def temporal_resample(h, speed_factor):
    """Resample temporal sequence by speed factor.
    speed_factor > 1 = faster (fewer frames), < 1 = slower (more frames)."""
    frame_norms = np.abs(h).sum(axis=(1, 2))
    valid = np.where(frame_norms > 0.01)[0]
    if len(valid) == 0:
        return h.copy()
    last_valid = valid[-1] + 1
    valid_frames = h[:last_valid]
    n_orig = valid_frames.shape[0]
    n_new = max(1, int(n_orig / speed_factor))
    indices = np.linspace(0, n_orig - 1, n_new, dtype=int)
    resampled = valid_frames[indices]
    # Pad back to original length if needed, or truncate
    out = np.zeros_like(h)
    n_copy = min(resampled.shape[0], h.shape[0])
    out[:n_copy] = resampled[:n_copy]
    return out


def spatial_mirror(h):
    """Mirror by swapping left/right hands and flipping x-coordinates."""
    h_out = h.copy()
    # Swap left hand (0:21) and right hand (21:42)
    lh = h_out[:, :21, :].copy()
    rh = h_out[:, 21:42, :].copy()
    h_out[:, :21, :] = rh
    h_out[:, 21:42, :] = lh
    # Flip x-coordinate for all nodes
    h_out[:, :42, 0] = -h_out[:, :42, 0]
    h_out[:, 42:48, 0] = -h_out[:, 42:48, 0]
    return h_out


def spatial_rotate(h, angle_deg):
    """Rotate hands around their wrist centers by angle_deg in xy plane."""
    h_out = h.copy()
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    # Rotate left hand around wrist
    lh_valid = np.abs(h_out[:, :21, :]).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = h_out[lh_valid, 0:1, :]
        centered = h_out[lh_valid, :21, :] - wrist
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h_out[lh_valid, :21, :] = rotated + wrist

    # Rotate right hand around wrist
    rh_valid = np.abs(h_out[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = h_out[rh_valid, 21:22, :]
        centered = h_out[rh_valid, 21:42, :] - wrist
        rotated = np.einsum('ij,ntj->nti', rot, centered)
        h_out[rh_valid, 21:42, :] = rotated + wrist

    return h_out


# ---------------------------------------------------------------------------
# Preprocessing pipeline (raw 48-node -> streams + aux tensors)
# ---------------------------------------------------------------------------

def preprocess_to_tensors(h, max_frames=MAX_FRAMES):
    """Convert 48-node skeleton to stream tensors and aux tensor."""
    hand_body_feats = compute_hand_body_features(h)
    h = normalize_wrist_palm(h)
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    f = h.shape[0]
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h, velocity, bones = h[indices], velocity[indices], bones[indices]
        joint_angles, fingertip_dists = joint_angles[indices], fingertip_dists[indices]
        hand_body_feats = hand_body_feats[indices]
    else:
        pad_len = max_frames - f
        h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
        joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
        fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
        hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

    streams = {
        "joint": torch.FloatTensor(np.clip(h, -10, 10)).permute(2, 0, 1),
        "bone": torch.FloatTensor(np.clip(bones, -10, 10)).permute(2, 0, 1),
        "velocity": torch.FloatTensor(np.clip(velocity, -10, 10)).permute(2, 0, 1),
    }
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_tensor = torch.FloatTensor(np.clip(aux_features, -10, 10))
    return streams, aux_tensor


# ---------------------------------------------------------------------------
# TTA augmentation list
# ---------------------------------------------------------------------------

def get_tta_variants(h_raw):
    """Generate all TTA variants of a raw 48-node skeleton.
    Returns list of (name, h_augmented) tuples."""
    temporal_augs = [
        ("orig", lambda x: x.copy()),
        ("rev", temporal_reverse),
        ("slow0.8", lambda x: temporal_resample(x, 0.8)),
        ("fast1.2", lambda x: temporal_resample(x, 1.2)),
    ]
    spatial_augs = [
        ("none", lambda x: x.copy()),
        ("mirror", spatial_mirror),
        ("rot+10", lambda x: spatial_rotate(x, 10.0)),
        ("rot-10", lambda x: spatial_rotate(x, -10.0)),
    ]

    variants = []
    for t_name, t_fn in temporal_augs:
        h_t = t_fn(h_raw)
        for s_name, s_fn in spatial_augs:
            h_ts = s_fn(h_t)
            name = f"{t_name}_{s_name}"
            variants.append((name, h_ts))
    return variants


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_exp1_models(ckpt_dir, classes, device):
    """Load v31 exp1 GroupNorm models."""
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
        print(f"  Loaded v31_exp1/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%, "
              f"epoch={ckpt.get('epoch', 'N/A')}, params={param_count:,}")

    fusion_weights = _load_fusion_weights(ckpt_dir)
    return models, fusion_weights


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_no_tta(raw, true_class, models, fusion_weights, classes, device):
    """Standard prediction without TTA (baseline)."""
    h_raw = extract_48node(raw)
    if h_raw is None:
        return None
    streams, aux = preprocess_to_tensors(h_raw)
    i2c = {i: c for i, c in enumerate(classes)}

    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)

    fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
    pred_idx = fused.argmax().item()
    return i2c[pred_idx], fused[pred_idx].item()


def predict_with_tta(raw, true_class, models, fusion_weights, classes, device):
    """TTA prediction: average softmax across all augmented variants."""
    h_raw = extract_48node(raw)
    if h_raw is None:
        return None
    i2c = {i: c for i, c in enumerate(classes)}

    variants = get_tta_variants(h_raw)
    all_fused_probs = []

    with torch.no_grad():
        for var_name, h_aug in variants:
            streams, aux = preprocess_to_tensors(h_aug)
            per_stream_probs = {}
            for sname, smodel in models.items():
                gcn = streams[sname].unsqueeze(0).to(device)
                logits, *_ = smodel(gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
                per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
            fused = sum(fusion_weights[s] * per_stream_probs[s] for s in models)
            all_fused_probs.append(fused)

    # Average across all TTA variants
    avg_probs = torch.stack(all_fused_probs).mean(dim=0)
    pred_idx = avg_probs.argmax().item()
    return i2c[pred_idx], avg_probs[pred_idx].item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V31 Exp1 TTA Evaluation")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/v31_exp1")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    args = parser.parse_args()

    base_dir = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"V31 Exp1 TTA Evaluation: GroupNorm (Signer-Agnostic)")
    print(f"Test-Time Augmentation: 4 temporal x 4 spatial = 16 variants")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Category:   {args.category}")
    print(f"Started:    {ts}")
    print("=" * 70)

    check_mediapipe_version()

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
        print(f"{cat_name.upper()} — V31 Exp1 GroupNorm ({len(cat_videos)} videos)")
        print(f"{'='*70}")

        print(f"\n  Loading models from {cat_ckpt_dir}...")
        models, fusion_weights = load_exp1_models(cat_ckpt_dir, classes, device)

        if len(models) < 3:
            print(f"  WARNING: Only {len(models)}/3 stream models loaded, skipping")
            continue

        print(f"  Fusion weights: {fusion_weights}")

        # --- Baseline (no TTA) ---
        print(f"\n  --- Baseline (no TTA) ---")

        def predict_base(raw, true_class, _m=models, _fw=fusion_weights, _c=classes, _d=device):
            return predict_no_tta(raw, true_class, _m, _fw, _c, _d)

        result_base = evaluate_method(f"v31exp1_noTTA_{cat_name}", cat_videos, predict_base, classes)
        all_results[f"{cat_name}_noTTA"] = result_base

        # --- TTA ---
        print(f"\n  --- With TTA (16 variants) ---")

        def predict_tta(raw, true_class, _m=models, _fw=fusion_weights, _c=classes, _d=device):
            return predict_with_tta(raw, true_class, _m, _fw, _c, _d)

        result_tta = evaluate_method(f"v31exp1_TTA_{cat_name}", cat_videos, predict_tta, classes)
        all_results[f"{cat_name}_TTA"] = result_tta

        # --- Comparison ---
        delta = result_tta["overall"] - result_base["overall"]
        print(f"\n  TTA Impact on {cat_name}: {result_base['overall']:.1f}% -> {result_tta['overall']:.1f}% ({delta:+.1f}pp)")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — V31 Exp1 TTA Evaluation")
    print(f"{'='*70}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {key}: {r['overall']:.1f}% ({r['correct']}/{r['total']})")

    # Combined accuracy
    for suffix in ["noTTA", "TTA"]:
        n_key = f"numbers_{suffix}"
        w_key = f"words_{suffix}"
        if n_key in all_results and w_key in all_results:
            n_r = all_results[n_key]
            w_r = all_results[w_key]
            combined = 100.0 * (n_r["correct"] + w_r["correct"]) / (n_r["total"] + w_r["total"])
            print(f"  Combined {suffix}: {combined:.1f}%")

    print(f"\n  Reference: v31 exp1 baseline = 61.4%")
    print(f"  Reference: v31 5-model ensemble = 70.7%")

    # Save
    out_dir = "/scratch/alpine/hama5612/ksl-dir-2/data/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v31_exp1_tta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "version": "v31_exp1_tta",
            "description": "V31 Exp1 GroupNorm with Test-Time Augmentation",
            "tta_variants": "4 temporal (orig, rev, slow0.8, fast1.2) x 4 spatial (none, mirror, rot+10, rot-10) = 16",
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
