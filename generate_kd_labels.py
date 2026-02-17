#!/usr/bin/env python3
"""
V33 Phase 1: Generate Knowledge Distillation Soft Labels

Loads all 5 teacher models (v27, v28, v29, v31_exp1, v31_exp5) and generates
soft ensemble logits for every training sample. These are saved as .pt files
for use by the v33 KD student training.

Teacher models use their original BN stats (trained on this data) — no adaptation needed.
GroupNorm models (exp1) need no adaptation by design.

Output:
    data/kd_labels/numbers_ensemble_logits.pt
    data/kd_labels/words_ensemble_logits.pt

Usage:
    python generate_kd_labels.py --category both
    python generate_kd_labels.py --category numbers
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_real_testers_v30_phase1 import (
    KSLGraphNetV25, KSLGraphNetV29,
    build_adj, normalize_wrist_palm, compute_bones, compute_joint_angles,
    compute_fingertip_distances, compute_hand_body_features,
    NUM_ANGLE_FEATURES, NUM_FINGERTIP_PAIRS, NUM_HAND_BODY_FEATURES,
    POSE_INDICES, PARENT_MAP,
)
from train_ksl_v31_exp1 import KSLGraphNetV31Exp1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

MAX_FRAMES = 90
CKPT_BASE = "/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Preprocessing: raw .npy -> model inputs
# ---------------------------------------------------------------------------

def preprocess_npy_to_v27(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (T, 549) to v27/v29 format: (9, T, 48) + (T, D_aux)."""
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None, None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)
    hand_body_feats = compute_hand_body_features(h)
    h = normalize_wrist_palm(h)
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

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

    gcn_features = np.concatenate([h, velocity, bones], axis=2)  # (T, 48, 9)
    gcn_tensor = torch.FloatTensor(np.clip(gcn_features, -10, 10)).permute(2, 0, 1)  # (9, T, 48)
    aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
    aux_tensor = torch.FloatTensor(np.clip(aux_features, -10, 10))
    return gcn_tensor, aux_tensor


def preprocess_npy_to_multistream(raw, max_frames=MAX_FRAMES):
    """Preprocess raw (T, 549) to multistream format: {stream: (3,T,48)} + (T, D_aux)."""
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None, None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)

    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)
    hand_body_feats = compute_hand_body_features(h)
    h = normalize_wrist_palm(h)
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = compute_bones(h)
    joint_angles = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

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
# Teacher model loading
# ---------------------------------------------------------------------------

def load_v27_teacher(category, classes, device):
    """Load v27 model (ic=9, single model)."""
    ckpt_path = os.path.join(CKPT_BASE, f"v27_{category}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: v27 checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(config["num_nodes"]).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES)

    model = KSLGraphNetV25(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config["num_nodes"], ic=config["in_channels"], hd=config["hidden_dim"],
        nl=config["num_layers"], tk=tk, dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded v27: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}")
    return model


def load_v28_teacher(category, classes, device):
    """Load v28 multi-stream models + fusion weights."""
    ckpt_dir = os.path.join(CKPT_BASE, f"v28_{category}")
    if not os.path.isdir(ckpt_dir):
        print(f"  WARNING: v28 checkpoint dir not found: {ckpt_dir}")
        return None, None
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
        num_signers = ckpt.get("num_signers", 12)
        model = KSLGraphNetV25(
            nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
            nn_=config["num_nodes"], ic=config["in_channels"],
            hd=config["hidden_dim"], nl=config["num_layers"], tk=tk,
            dr=config["dropout"], spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[sname] = model
        print(f"  Loaded v28/{sname}: val_acc={ckpt['val_acc']:.1f}%")

    # Load fusion weights
    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            fusion_weights = json.load(f)["weights"]
    else:
        fusion_weights = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
    return models, fusion_weights


def load_v29_teacher(category, classes, device):
    """Load v29 model (ic=9, deeper)."""
    ckpt_path = os.path.join(CKPT_BASE, f"v29_{category}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: v29 checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(48).to(device)
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES)

    model = KSLGraphNetV29(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=48, ic=config.get("in_channels", 9),
        hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 8),
        td=tuple(config.get("temporal_dilations", [1, 2, 4])),
        dr=config.get("dropout", 0.2),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded v29: val_acc={ckpt['val_acc']:.1f}%, epoch={ckpt['epoch']}")
    return model


def load_v31_exp1_teacher(category, classes, device):
    """Load v31 exp1 (GroupNorm) multi-stream models."""
    ckpt_dir = os.path.join(CKPT_BASE, f"v31_exp1/{category}")
    if not os.path.isdir(ckpt_dir):
        print(f"  WARNING: v31_exp1 checkpoint dir not found: {ckpt_dir}")
        return None, None
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

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
        print(f"  Loaded v31_exp1/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%")

    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            fusion_weights = json.load(f)["weights"]
    else:
        fusion_weights = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
    return models, fusion_weights


def load_v31_exp5_teacher(category, classes, device):
    """Load v31 exp5 (SupCon) multi-stream models — same arch as v28."""
    ckpt_dir = os.path.join(CKPT_BASE, f"v31_exp5/{category}")
    if not os.path.isdir(ckpt_dir):
        print(f"  WARNING: v31_exp5 checkpoint dir not found: {ckpt_dir}")
        return None, None
    adj = build_adj(48).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    models = {}
    for sname in ["joint", "bone", "velocity"]:
        ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
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
        print(f"  Loaded v31_exp5/{sname}: val_acc={ckpt.get('val_acc', 'N/A'):.1f}%")

    weights_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            fusion_weights = json.load(f)["weights"]
    else:
        fusion_weights = {"joint": 0.4, "bone": 0.35, "velocity": 0.25}
    return models, fusion_weights


# ---------------------------------------------------------------------------
# Get teacher probs for a single sample
# ---------------------------------------------------------------------------

def get_v27_probs(v27_model, gcn_v27, aux_v27, device):
    """Get softmax probs from v27 model."""
    with torch.no_grad():
        logits, _, _ = v27_model(
            gcn_v27.unsqueeze(0).to(device),
            aux_v27.unsqueeze(0).to(device),
            grl_lambda=0.0,
        )
        return F.softmax(logits, dim=1).cpu().squeeze(0)


def get_v28_probs(v28_models, v28_fw, streams, aux, device):
    """Get fused softmax probs from v28 multi-stream model."""
    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in v28_models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, _, _ = smodel(gcn, aux_t, grl_lambda=0.0)
            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
    fused = sum(v28_fw[s] * per_stream_probs[s] for s in v28_models)
    return fused


def get_v29_probs(v29_model, gcn_v27, aux_v27, device):
    """Get softmax probs from v29 model (same input format as v27)."""
    with torch.no_grad():
        logits, _, _ = v29_model(
            gcn_v27.unsqueeze(0).to(device),
            aux_v27.unsqueeze(0).to(device),
            grl_lambda=0.0,
        )
        return F.softmax(logits, dim=1).cpu().squeeze(0)


def get_exp1_probs(exp1_models, exp1_fw, streams, aux, device):
    """Get fused softmax probs from v31_exp1 GroupNorm model."""
    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in exp1_models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, _, _ = smodel(gcn, aux_t)  # GroupNorm, no grl_lambda needed
            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
    fused = sum(exp1_fw[s] * per_stream_probs[s] for s in exp1_models)
    return fused


def get_exp5_probs(exp5_models, exp5_fw, streams, aux, device):
    """Get fused softmax probs from v31_exp5 SupCon model."""
    per_stream_probs = {}
    with torch.no_grad():
        for sname, smodel in exp5_models.items():
            gcn = streams[sname].unsqueeze(0).to(device)
            aux_t = aux.unsqueeze(0).to(device)
            logits, _, _ = smodel(gcn, aux_t, grl_lambda=0.0)
            per_stream_probs[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)
    fused = sum(exp5_fw[s] * per_stream_probs[s] for s in exp5_models)
    return fused


# ---------------------------------------------------------------------------
# Main: Generate soft labels
# ---------------------------------------------------------------------------

def generate_labels_for_category(category, classes, device):
    """Generate ensemble soft labels for all training samples in a category."""

    print(f"\n{'='*70}")
    print(f"Generating KD labels for {category.upper()} ({len(classes)} classes)")
    print(f"{'='*70}")

    # Load all 5 teachers
    print(f"\n[{ts()}] Loading teacher models...")

    v27_model = load_v27_teacher(category, classes, device)
    v28_models, v28_fw = load_v28_teacher(category, classes, device)
    v29_model = load_v29_teacher(category, classes, device)
    exp1_models, exp1_fw = load_v31_exp1_teacher(category, classes, device)
    exp5_models, exp5_fw = load_v31_exp5_teacher(category, classes, device)

    teacher_count = sum([
        v27_model is not None,
        v28_models is not None and len(v28_models) == 3,
        v29_model is not None,
        exp1_models is not None and len(exp1_models) == 3,
        exp5_models is not None and len(exp5_models) == 3,
    ])
    print(f"\n[{ts()}] Loaded {teacher_count}/5 teacher models")

    if teacher_count == 0:
        print(f"[{ts()}] ERROR: No teacher models found!")
        return

    # Discover training samples
    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    data_dirs = [
        os.path.join(base_data, "train_alpha"),
        os.path.join(base_data, "val_alpha"),
    ]

    sample_paths = []
    sample_labels = []
    c2i = {c: i for i, c in enumerate(classes)}

    for data_dir in data_dirs:
        for cn in classes:
            cd = os.path.join(data_dir, cn)
            if os.path.exists(cd):
                for fn in sorted(os.listdir(cd)):
                    if fn.endswith(".npy"):
                        sample_paths.append(os.path.join(cd, fn))
                        sample_labels.append(c2i[cn])

    print(f"[{ts()}] Found {len(sample_paths)} training samples")

    # Generate soft labels
    logits_dict = {}
    correct = 0
    total = 0
    t_start = time.time()

    for idx, (path, label) in enumerate(zip(sample_paths, sample_labels)):
        raw = np.load(path)
        filename = os.path.basename(path)

        # Preprocess to both formats
        gcn_v27, aux_v27 = preprocess_npy_to_v27(raw)
        streams, aux_ms = preprocess_npy_to_multistream(raw)

        if streams is None and gcn_v27 is None:
            print(f"  WARNING: Could not preprocess {filename}, skipping")
            continue

        # Collect probs from all teachers
        all_probs = []

        if v27_model is not None and gcn_v27 is not None:
            all_probs.append(get_v27_probs(v27_model, gcn_v27, aux_v27, device))

        if v28_models is not None and streams is not None:
            all_probs.append(get_v28_probs(v28_models, v28_fw, streams, aux_ms, device))

        if v29_model is not None and gcn_v27 is not None:
            all_probs.append(get_v29_probs(v29_model, gcn_v27, aux_v27, device))

        if exp1_models is not None and streams is not None:
            all_probs.append(get_exp1_probs(exp1_models, exp1_fw, streams, aux_ms, device))

        if exp5_models is not None and streams is not None:
            all_probs.append(get_exp5_probs(exp5_models, exp5_fw, streams, aux_ms, device))

        if not all_probs:
            continue

        # Average probs, convert to log-probs (pseudo-logits)
        ensemble_probs = torch.stack(all_probs).mean(dim=0)
        pseudo_logits = torch.log(ensemble_probs + 1e-8)

        logits_dict[filename] = pseudo_logits

        # Track accuracy
        pred = ensemble_probs.argmax().item()
        if pred == label:
            correct += 1
        total += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            print(f"[{ts()}] Processed {idx + 1}/{len(sample_paths)} "
                  f"({rate:.1f} samples/s, acc={100.0 * correct / total:.1f}%)")

    elapsed = time.time() - t_start
    teacher_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\n[{ts()}] {category.upper()} complete: {total} samples, "
          f"teacher accuracy={teacher_acc:.1f}%, time={elapsed:.1f}s")

    # Save
    out_dir = os.path.join(base_data, "kd_labels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{category}_ensemble_logits.pt")
    torch.save(logits_dict, out_path)
    print(f"[{ts()}] Saved {len(logits_dict)} soft labels to {out_path}")

    return logits_dict, teacher_acc


def main():
    parser = argparse.ArgumentParser(description="Generate KD soft labels from 5-model ensemble")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    args = parser.parse_args()

    print("=" * 70)
    print("V33 Phase 1: Knowledge Distillation Label Generation")
    print(f"Started: {ts()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if args.category in ("numbers", "both"):
        logits, acc = generate_labels_for_category("numbers", NUMBER_CLASSES, device)
        results["numbers"] = {"count": len(logits), "teacher_acc": acc}

    if args.category in ("words", "both"):
        logits, acc = generate_labels_for_category("words", WORD_CLASSES, device)
        results["words"] = {"count": len(logits), "teacher_acc": acc}

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for cat, r in results.items():
        print(f"  {cat}: {r['count']} samples, teacher accuracy={r['teacher_acc']:.1f}%")
    print(f"\nDone: {ts()}")


if __name__ == "__main__":
    main()
