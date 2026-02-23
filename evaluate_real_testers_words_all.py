#!/usr/bin/env python3
"""
evaluate_real_testers_words_all.py

Evaluate ALL KSL models (v22-v43) on the real_testers words videos.

 - 81 word videos from 3 signers (from real_testers_metadata.csv)
 - CPU only (no GPU)
 - Uses existing .npy landmark caches (no video re-extraction needed)
 - AdaBN enabled by default (81 samples is sufficient)
 - Per-signer breakdown in summary

Usage:
    python evaluate_real_testers_words_all.py
    python evaluate_real_testers_words_all.py --no-adabn
    python evaluate_real_testers_words_all.py --models v27 v41 v43
    python evaluate_real_testers_words_all.py --rt-base /scratch/alpine/hama5612/ksl-alpha
"""

import os
import sys
import csv
import json
import copy
import argparse
import importlib
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants (matching train_ksl_v28.py / evaluate_real_testers_v30.py)
# ---------------------------------------------------------------------------

POSE_INDICES = [11, 12, 13, 14, 15, 16]

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

LH_PARENT  = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT  = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP  = LH_PARENT + RH_PARENT + POSE_PARENT

MAX_FRAMES = 90

LH_TIPS = [4, 8, 12, 16, 20]
RH_TIPS  = [25, 29, 33, 37, 41]
NUM_FINGERTIP_PAIRS = 10  # C(5,2) per hand
NUM_HAND_BODY_FEATURES = 8
NUM_NODES = 48


def _build_angle_joints():
    children = defaultdict(list)
    for child_idx, parent_idx in enumerate(PARENT_MAP):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    angle_joints = []
    for node in range(NUM_NODES):
        parent = PARENT_MAP[node]
        if parent < 0:
            continue
        for child in children[node]:
            angle_joints.append((node, parent, child))
    return angle_joints


ANGLE_JOINTS = _build_angle_joints()
NUM_ANGLE_FEATURES = len(ANGLE_JOINTS)          # 33
AUX_DIM_NO_HANDBODY = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS           # 53
AUX_DIM_WITH_HANDBODY = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES  # 61

# ---------------------------------------------------------------------------
# Model Registry (same as evaluate_ksl_dictionary.py)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    # ─── Era 1: Single-stream 9ch, no hand_body ───────────────────────────
    dict(name="v22", ckpt_words="v22_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="no_hand_body",
         train_module="train_ksl_v22", model_class="KSLGraphNetV22",
         build_adj_fn="build_adj"),

    # ─── Era 2: Multi-stream 4-stream, no hand_body ───────────────────────
    dict(name="v23", ckpt_words="v23_words",
         input_type="4stream", norm="BN", aux_type="no_hand_body",
         train_module="train_ksl_v23", model_class="KSLGraphNetV23",
         build_adj_fn="build_adj"),

    # ─── Era 3: Single-stream 9ch, with hand_body ─────────────────────────
    dict(name="v24", ckpt_words="v24_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v24", model_class="KSLGraphNetV24",
         build_adj_fn="build_adj"),

    dict(name="v25", ckpt_words="v25_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v25", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v26", ckpt_words="v26_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v25", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v27", ckpt_words="v27_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v27", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    # ─── Era 4: Multi-stream 3-stream v28 / v29 single ────────────────────
    dict(name="v28", ckpt_words="v28_words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v28", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v29", ckpt_words="v29_words/best_model.pt",
         input_type="single_9ch", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v29", model_class="KSLGraphNetV29",
         build_adj_fn="build_adj"),

    dict(name="v30", ckpt_words="v30_words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v30", model_class="KSLGraphNetV30",
         build_adj_fn="build_adj"),

    # ─── Era 5: v31 experiments ───────────────────────────────────────────
    dict(name="v31_exp1", ckpt_words="v31_exp1/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v31_exp3", ckpt_words="v31_exp3/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp3", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v31_exp4", ckpt_words="v31_exp4/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp4", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v31_exp5", ckpt_words="v31_exp5/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp5", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    # ─── Era 6: v32-v43 ───────────────────────────────────────────────────
    dict(name="v32", ckpt_words="v32/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v32", model_class="KSLGraphNetV32",
         build_adj_fn="build_adj"),

    dict(name="v32b", ckpt_words="v32b/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v32", model_class="KSLGraphNetV32",
         build_adj_fn="build_adj"),

    dict(name="v33", ckpt_words="v33/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v33", model_class="KSLGraphNetV33",
         build_adj_fn="build_adj"),

    dict(name="v33b", ckpt_words="v33b/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v33", model_class="KSLGraphNetV33",
         build_adj_fn="build_adj"),

    dict(name="v34a_seed123", ckpt_words="v34a_seed123/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v34b_seed456", ckpt_words="v34b_seed456/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v31_exp1", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v35", ckpt_words="v35/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v35", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v36", ckpt_words="v36/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v36", model_class="KSLGraphNetV31Exp1",
         build_adj_fn="build_adj"),

    dict(name="v37", ckpt_words="v37/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v37", model_class="KSLGraphNetV37",
         build_adj_fn="build_adj"),

    dict(name="v38", ckpt_words="v38/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v38", model_class="KSLGraphNetV38",
         build_adj_fn="build_graph_distance"),

    dict(name="v39", ckpt_words="v39/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v39", model_class="KSLGraphNetV25",
         build_adj_fn="build_adj"),

    dict(name="v40", ckpt_words="v40/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v40", model_class="KSLGraphNetV40",
         build_adj_fn="build_adj"),

    dict(name="v41", ckpt_words="v41/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v41", model_class="KSLGraphNetV41",
         build_adj_fn="build_adj"),

    dict(name="v42", ckpt_words="v42/words",
         input_type="3stream", norm="GN", aux_type="with_hand_body",
         train_module="train_ksl_v42", model_class="KSLGraphNetV42",
         build_adj_fn="build_adj"),

    dict(name="v43", ckpt_words="v43/words",
         input_type="3stream", norm="BN", aux_type="with_hand_body",
         train_module="train_ksl_v43", model_class="KSLGraphNetV43",
         build_adj_fn="build_adj"),
]


# ---------------------------------------------------------------------------
# Preprocessing helpers (identical to evaluate_ksl_dictionary.py)
# ---------------------------------------------------------------------------

def compute_bones(h):
    bones = np.zeros_like(h)
    for child in range(NUM_NODES):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def compute_joint_angles(h):
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)
    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        bone_in  = h[:, node, :] - h[:, parent, :]
        bone_out = h[:, child, :] - h[:, node, :]
        n_in  = np.maximum(np.linalg.norm(bone_in,  axis=-1, keepdims=True), 1e-8)
        n_out = np.maximum(np.linalg.norm(bone_out, axis=-1, keepdims=True), 1e-8)
        dot   = np.sum((bone_in / n_in) * (bone_out / n_out), axis=-1)
        angles[:, idx] = np.arccos(np.clip(dot, -1.0, 1.0))
    return angles


def compute_fingertip_distances(h):
    T   = h.shape[0]
    col = 0
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
    for tips in [LH_TIPS, RH_TIPS]:
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                diff = h[:, tips[i], :] - h[:, tips[j], :]
                distances[:, col] = np.linalg.norm(diff, axis=-1)
                col += 1
    return distances


def compute_hand_body_features(h_raw):
    T = h_raw.shape[0]
    feats = np.zeros((T, NUM_HAND_BODY_FEATURES), dtype=np.float32)
    mid_shoulder   = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2
    shoulder_width = np.maximum(
        np.linalg.norm(h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True), 1e-6
    )
    lh_centroid = h_raw[:, :21, :].mean(axis=1)
    rh_centroid = h_raw[:, 21:42, :].mean(axis=1)
    face_approx = mid_shoulder.copy()
    face_approx[:, 1] -= shoulder_width[:, 0] * 0.7
    sw = shoulder_width[:, 0]
    feats[:, 0] = (lh_centroid[:, 1] - mid_shoulder[:, 1]) / sw
    feats[:, 1] = (rh_centroid[:, 1] - mid_shoulder[:, 1]) / sw
    feats[:, 2] = (lh_centroid[:, 0] - mid_shoulder[:, 0]) / sw
    feats[:, 3] = (rh_centroid[:, 0] - mid_shoulder[:, 0]) / sw
    feats[:, 4] = np.linalg.norm(lh_centroid - rh_centroid, axis=-1) / sw
    feats[:, 5] = np.linalg.norm(lh_centroid - face_approx, axis=-1) / sw
    feats[:, 6] = np.linalg.norm(rh_centroid - face_approx, axis=-1) / sw
    feats[:, 7] = np.abs(lh_centroid[:, 1] - rh_centroid[:, 1]) / sw
    return feats


def normalize_wrist_palm(h):
    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        wrist = lh[:, 0:1, :]
        lh[lh_valid] -= wrist[lh_valid]
        psz = np.maximum(np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        lh[lh_valid] /= psz[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        wrist = rh[:, 0:1, :]
        rh[rh_valid] -= wrist[rh_valid]
        psz = np.maximum(np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        rh[rh_valid] /= psz[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] -= mid[pose_valid]
        sw = np.maximum(
            np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True), 1e-6
        )
        pose[pose_valid] /= sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose
    return h


def preprocess_raw(raw, max_frames=MAX_FRAMES, include_hand_body=True):
    """Preprocess raw (N_frames, 225) landmarks → streams dict + aux tensor."""
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None, None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)
    h  = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    hb_feats = compute_hand_body_features(h) if include_hand_body else None
    h = normalize_wrist_palm(h)

    velocity      = np.zeros_like(h)
    velocity[1:]  = h[1:] - h[:-1]
    bones         = compute_bones(h)
    bone_velocity = np.zeros_like(bones)
    bone_velocity[1:] = bones[1:] - bones[:-1]

    joint_angles    = compute_joint_angles(h)
    fingertip_dists = compute_fingertip_distances(h)

    def _resamp(arr, nf):
        if f >= max_frames:
            idx = np.linspace(0, f - 1, max_frames, dtype=int)
            return arr[idx]
        pad = max_frames - f
        pad_shape = (pad,) + arr.shape[1:]
        return np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)])

    h             = _resamp(h,             f)
    velocity      = _resamp(velocity,      f)
    bones         = _resamp(bones,         f)
    bone_velocity = _resamp(bone_velocity, f)
    joint_angles  = _resamp(joint_angles,  f)
    fingertip_dists = _resamp(fingertip_dists, f)
    if hb_feats is not None:
        hb_feats = _resamp(hb_feats, f)

    def _t(x):
        return torch.FloatTensor(np.clip(x, -10, 10)).permute(2, 0, 1)

    streams = {
        "joint":         _t(h),
        "bone":          _t(bones),
        "velocity":      _t(velocity),
        "bone_velocity": _t(bone_velocity),
    }

    if include_hand_body:
        aux_np = np.concatenate([joint_angles, fingertip_dists, hb_feats], axis=1)
    else:
        aux_np = np.concatenate([joint_angles, fingertip_dists], axis=1)
    aux = torch.FloatTensor(np.clip(aux_np, -10, 10))

    return streams, aux


def stack_9ch(streams):
    """Stack joint + velocity + bone into (9, T, N) for single-model versions."""
    return torch.cat([streams["joint"], streams["velocity"], streams["bone"]], dim=0)


# ---------------------------------------------------------------------------
# AdaBN
# ---------------------------------------------------------------------------

def adapt_bn_stats(model, data_list, device):
    """Reset BN running stats and forward-pass all data to collect new stats."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None  # cumulative moving average — equal weight to all samples

    model.train()
    with torch.no_grad():
        for gcn, aux in data_list:
            try:
                model(gcn.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
            except Exception:
                pass
    model.eval()


# ---------------------------------------------------------------------------
# Fusion weights
# ---------------------------------------------------------------------------

def _load_fusion_weights(ckpt_dir, streams=("joint", "bone", "velocity")):
    w_path = os.path.join(ckpt_dir, "fusion_weights.json")
    if os.path.exists(w_path):
        with open(w_path) as fh:
            data = json.load(fh)
        w = data.get("weights", data)
        if isinstance(w, dict) and "weights" in w:
            w = w["weights"]
        return {s: w.get(s, 1.0 / len(streams)) for s in streams}
    return {s: 1.0 / len(streams) for s in streams}


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def _lazy_import(module_name, class_name, build_adj_fn="build_adj"):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    build_adj = getattr(mod, build_adj_fn)
    return cls, build_adj


def _build_model_from_ckpt(ckpt, cls, build_adj, nc, device):
    config      = ckpt.get("config", {})
    num_signers = ckpt.get("num_signers", 12)
    num_nodes   = config.get("num_nodes", NUM_NODES)
    adj         = build_adj(num_nodes).to(device)
    aux_dim     = ckpt.get("aux_dim", AUX_DIM_WITH_HANDBODY)
    hd          = config.get("hidden_dim", 64)
    nl          = config.get("num_layers", 4)
    ic          = config.get("in_channels", 3)
    tk          = tuple(config.get("temporal_kernels", [3, 5, 7]))
    dr          = config.get("dropout", 0.3)
    sd          = config.get("spatial_dropout", 0.1)

    kwargs_full = dict(
        nc=nc, num_signers=num_signers, aux_dim=aux_dim,
        nn_=num_nodes, ic=ic, hd=hd, nl=nl, tk=tk, dr=dr,
        spatial_dropout=sd, adj=adj,
    )
    proj_dim = config.get("supcon_proj_dim", 128)
    kwargs_with_proj = dict(**kwargs_full, proj_dim=proj_dim)

    for kwargs in [kwargs_with_proj, kwargs_full]:
        try:
            return cls(**kwargs)
        except TypeError:
            pass

    essential = dict(nc=nc, aux_dim=aux_dim, nn_=num_nodes, ic=ic, adj=adj)
    for k in ["hd", "nl", "tk", "dr", "spatial_dropout", "num_signers"]:
        try:
            return cls(**essential)
        except TypeError:
            essential[k] = kwargs_full[k]

    raise RuntimeError(f"Cannot instantiate {cls.__name__} with checkpoint config")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_single(model, gcn_9ch, aux, device):
    with torch.no_grad():
        out    = model(gcn_9ch.unsqueeze(0).to(device), aux.unsqueeze(0).to(device), grl_lambda=0.0)
        logits = out[0]
        probs  = F.softmax(logits, dim=1).cpu().squeeze(0)
    return probs.argmax().item(), probs


def run_inference_multistream(models_dict, fusion_weights, streams, aux, device,
                              stream_names=("joint", "bone", "velocity")):
    per_stream = {}
    with torch.no_grad():
        for sname in stream_names:
            if sname not in models_dict:
                continue
            gcn = streams[sname].unsqueeze(0).to(device)
            out = models_dict[sname](gcn, aux.unsqueeze(0).to(device), grl_lambda=0.0)
            logits = out[0]
            per_stream[sname] = F.softmax(logits, dim=1).cpu().squeeze(0)

    fused    = sum(fusion_weights.get(s, 1.0 / len(per_stream)) * per_stream[s]
                   for s in per_stream)
    pred_idx = fused.argmax().item()
    return pred_idx, fused


# ---------------------------------------------------------------------------
# Evaluate one model entry
# ---------------------------------------------------------------------------

def evaluate_model_entry(entry, video_data, classes, device, use_adabn, ckpt_base):
    """Evaluate one model on all word videos.

    video_data: list of dicts {uid, true_class, signer, raw (np.ndarray)}
    Returns: dict {uid: {pred, conf, correct, top3}} or None
    """
    name        = entry["name"]
    input_type  = entry["input_type"]
    norm        = entry["norm"]
    aux_type    = entry["aux_type"]
    include_hb  = (aux_type == "with_hand_body")

    ckpt_dir = os.path.join(ckpt_base, entry["ckpt_words"])

    try:
        cls, build_adj = _lazy_import(entry["train_module"], entry["model_class"], entry["build_adj_fn"])
    except Exception as e:
        print(f"  [{name}] Import error: {e}")
        return None

    # Preprocess all videos
    preprocessed = {}
    for vd in video_data:
        streams, aux = preprocess_raw(vd["raw"], include_hand_body=include_hb)
        preprocessed[vd["uid"]] = (streams, aux) if streams is not None else None

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = classes

    if input_type == "single_9ch":
        ckpt_path = ckpt_dir
        if not os.path.exists(ckpt_path):
            print(f"  [{name}] Checkpoint not found: {ckpt_path}")
            return None
        try:
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = _build_model_from_ckpt(ckpt, cls, build_adj, len(classes), device)
            model.load_state_dict(ckpt["model"])
            model.eval()
        except Exception as e:
            print(f"  [{name}] Load error: {e}")
            return None

        val_acc = ckpt.get("val_acc", 0.0)
        print(f"  [{name}] Loaded single model (val_acc={val_acc:.1f}%, norm={norm})")

        if use_adabn and norm == "BN":
            adabn_data = []
            for vd in video_data:
                if preprocessed[vd["uid"]] is not None:
                    s, a = preprocessed[vd["uid"]]
                    adabn_data.append((stack_9ch(s), a))
            if adabn_data:
                adapt_bn_stats(model, adabn_data, device)
                print(f"  [{name}] AdaBN applied ({len(adabn_data)} samples)")

        results = {}
        for vd in video_data:
            uid = vd["uid"]
            if preprocessed[uid] is None:
                results[uid] = None
                continue
            s, a = preprocessed[uid]
            gcn_9ch = stack_9ch(s)
            pred_idx, probs = run_inference_single(model, gcn_9ch, a, device)
            pred_class  = i2c[pred_idx]
            confidence  = probs[pred_idx].item()
            is_correct  = (pred_class == vd["true_class"])
            results[uid] = {
                "pred": pred_class, "conf": confidence, "correct": is_correct,
                "top3": [(i2c[i], probs[i].item()) for i in probs.argsort(descending=True)[:3]],
            }

    elif input_type in ("3stream", "4stream"):
        stream_names = (["joint", "bone", "velocity", "bone_velocity"]
                        if input_type == "4stream" else
                        ["joint", "bone", "velocity"])

        models_dict = {}
        for sname in stream_names:
            ckpt_path = os.path.join(ckpt_dir, sname, "best_model.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [{name}] Missing stream {sname}: {ckpt_path}")
                continue
            try:
                ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = _build_model_from_ckpt(ckpt, cls, build_adj, len(classes), device)
                model.load_state_dict(ckpt["model"])
                model.eval()
                models_dict[sname] = model
            except Exception as e:
                print(f"  [{name}] Load error ({sname}): {e}")

        if not models_dict:
            print(f"  [{name}] No streams loaded, skipping")
            return None

        fusion_weights = _load_fusion_weights(ckpt_dir, stream_names)
        val_acc = ckpt.get("val_acc", 0.0)
        print(f"  [{name}] Loaded {len(models_dict)}/{len(stream_names)} streams "
              f"(val_acc={val_acc:.1f}%, norm={norm}, fw={fusion_weights})")

        if use_adabn and norm == "BN":
            for sname, model in models_dict.items():
                adabn_data = []
                for vd in video_data:
                    if preprocessed[vd["uid"]] is not None:
                        s, a = preprocessed[vd["uid"]]
                        adabn_data.append((s[sname], a))
                if adabn_data:
                    adapt_bn_stats(model, adabn_data, device)
            print(f"  [{name}] AdaBN applied to all {len(models_dict)} stream models")

        results = {}
        for vd in video_data:
            uid = vd["uid"]
            if preprocessed[uid] is None:
                results[uid] = None
                continue
            s, a = preprocessed[uid]
            pred_idx, probs = run_inference_multistream(
                models_dict, fusion_weights, s, a, device, stream_names
            )
            pred_class  = i2c[pred_idx]
            confidence  = probs[pred_idx].item()
            is_correct  = (pred_class == vd["true_class"])
            results[uid] = {
                "pred": pred_class, "conf": confidence, "correct": is_correct,
                "top3": [(i2c[i], probs[i].item()) for i in probs.argsort(descending=True)[:3]],
            }

    else:
        print(f"  [{name}] Unknown input_type: {input_type}")
        return None

    return results


# ---------------------------------------------------------------------------
# Summary table (adapted for many videos — show per-model + per-signer)
# ---------------------------------------------------------------------------

def print_summary(all_results, video_data):
    signers = sorted(set(vd["signer"] for vd in video_data))
    uids    = [vd["uid"] for vd in video_data]
    uid_to_signer = {vd["uid"]: vd["signer"] for vd in video_data}
    uid_to_class  = {vd["uid"]: vd["true_class"] for vd in video_data}

    print("\n" + "=" * 80)
    print("RESULTS — Real Testers Words (all models)")
    print("=" * 80)

    # Header
    hdr = f"{'Model':<18s}  {'Overall':>8s}"
    for s in signers:
        short = s.split(".")[0].strip()  # "1", "2", "3"
        hdr += f"  {'S'+short:>8s}"
    print(hdr)
    print("-" * (18 + 10 + 10 * len(signers)))

    rows = []
    for model_name, results in all_results.items():
        if results is None:
            print(f"{model_name:<18s}  [FAILED]")
            continue

        # Overall
        correct_all = sum(1 for uid in uids if results.get(uid) and results[uid]["correct"])
        total_all   = sum(1 for uid in uids if results.get(uid) is not None)
        acc_all = 100.0 * correct_all / total_all if total_all else 0.0

        # Per-signer
        signer_accs = []
        for s in signers:
            uids_s = [uid for uid in uids if uid_to_signer[uid] == s]
            c = sum(1 for uid in uids_s if results.get(uid) and results[uid]["correct"])
            t = sum(1 for uid in uids_s if results.get(uid) is not None)
            signer_accs.append((c, t))

        row = f"{model_name:<18s}  {correct_all:3d}/{total_all:2d} ({acc_all:5.1f}%)"
        for c, t in signer_accs:
            a = 100.0 * c / t if t else 0.0
            row += f"  {c:2d}/{t:2d} ({a:4.1f}%)"
        print(row)
        rows.append((model_name, acc_all, correct_all, total_all))

    print("=" * 80)

    # Ranking
    rows_valid = [(n, a, c, t) for n, a, c, t in rows]
    rows_valid.sort(key=lambda x: -x[1])
    print("\nRanking by overall accuracy:")
    for rank, (n, a, c, t) in enumerate(rows_valid, 1):
        print(f"  {rank:2d}. {n:<18s} {c}/{t} ({a:.1f}%)")

    # Per-class accuracy across all models
    print("\nPer-class accuracy (fraction of {model×video} correct):")
    for cls in WORD_CLASSES:
        cls_uids = [uid for uid in uids if uid_to_class[uid] == cls]
        n_total = 0
        n_correct = 0
        for r in all_results.values():
            if r is None:
                continue
            for uid in cls_uids:
                if r.get(uid) is not None:
                    n_total += 1
                    if r[uid]["correct"]:
                        n_correct += 1
        pct = 100.0 * n_correct / n_total if n_total else 0.0
        bar = "#" * int(pct / 5)
        print(f"  {cls:12s}: {n_correct:3d}/{n_total:3d} = {pct:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all KSL models on real_testers words")
    parser.add_argument("--rt-base",
                        default="/scratch/alpine/hama5612/ksl-alpha",
                        help="Base directory containing data/real_testers/")
    parser.add_argument("--metadata-csv", default=None,
                        help="Path to real_testers_metadata.csv (default: <rt-base>/data/real_testers/real_testers_metadata.csv)")
    parser.add_argument("--ckpt-base",
                        default=os.path.join(PROJECT_ROOT, "data/checkpoints"),
                        help="Base checkpoint directory")
    parser.add_argument("--results-dir",
                        default=os.path.join(PROJECT_ROOT, "data/results/real_testers_words_all"),
                        help="Output directory for JSON results")
    parser.add_argument("--no-adabn", action="store_true",
                        help="Disable AdaBN (not recommended for 81 samples)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to evaluate (default: all)")
    args = parser.parse_args()

    use_adabn = not args.no_adabn
    device    = torch.device("cpu")

    rt_base = args.rt_base
    metadata_csv = args.metadata_csv or os.path.join(rt_base, "data/real_testers/real_testers_metadata.csv")

    print("=" * 70)
    print("Real Testers Words — All Models Evaluation")
    print(f"RT base:     {rt_base}")
    print(f"Metadata:    {metadata_csv}")
    print(f"Ckpt base:   {args.ckpt_base}")
    print(f"Device:      {device}")
    print(f"AdaBN:       {'enabled (BN models only)' if use_adabn else 'DISABLED'}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Read metadata CSV ──────────────────────────────────────────────────
    if not os.path.exists(metadata_csv):
        print(f"[ERROR] Metadata CSV not found: {metadata_csv}")
        sys.exit(1)

    video_data = []
    with open(metadata_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls_name = row["class_name"].strip()
            # Skip numbers
            if cls_name.startswith("No_") or cls_name not in WORD_CLASSES:
                continue
            rel_path   = row["video_path"].strip()
            full_path  = os.path.join(rt_base, rel_path)
            # Parse signer from path: data/real_testers/{signer_dir}/...
            parts      = rel_path.replace("\\", "/").split("/")
            signer_dir = parts[2] if len(parts) > 2 else "unknown"
            subdir     = parts[3] if len(parts) > 3 else ""
            # Unique ID: use relative path without extension
            uid = rel_path  # full relative path is globally unique

            npy_path = full_path + ".landmarks.npy"
            if not os.path.exists(npy_path):
                print(f"  [SKIP] Missing .npy cache: {npy_path}")
                continue

            raw = np.load(npy_path)
            video_data.append({
                "uid":        uid,
                "true_class": cls_name,
                "signer":     signer_dir,
                "subdir":     subdir,
                "path":       full_path,
                "raw":        raw,
            })

    print(f"\nLoaded {len(video_data)} word videos:")
    signer_counts = defaultdict(int)
    for vd in video_data:
        signer_counts[vd["signer"]] += 1
    for s, cnt in sorted(signer_counts.items()):
        print(f"  {s}: {cnt} videos")

    if not video_data:
        print("[ERROR] No word videos loaded")
        sys.exit(1)

    # ── Filter registry ────────────────────────────────────────────────────
    if args.models:
        registry = [e for e in MODEL_REGISTRY if e["name"] in args.models]
        print(f"\nEvaluating {len(registry)} model(s): {[e['name'] for e in registry]}")
    else:
        registry = MODEL_REGISTRY
        print(f"\nEvaluating all {len(registry)} model versions")

    classes = WORD_CLASSES

    # ── Evaluate each model ────────────────────────────────────────────────
    all_results = {}
    for entry in registry:
        print(f"\n{'─'*50}")
        print(f"[{entry['name']}]  input={entry['input_type']}  norm={entry['norm']}  "
              f"aux={entry['aux_type']}")
        try:
            results = evaluate_model_entry(
                entry, video_data, classes, device, use_adabn, args.ckpt_base
            )
        except Exception as e:
            print(f"  [{entry['name']}] EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results = None
        all_results[entry["name"]] = results

        if results is not None:
            correct = sum(1 for r in results.values() if r and r["correct"])
            total   = sum(1 for r in results.values() if r is not None)
            print(f"  → {correct}/{total} correct ({100.0*correct/total:.1f}%)" if total else "  → No results")

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(all_results, video_data)

    # ── Save JSON ──────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"rt_words_eval_{ts}.json")

    def _ser(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        return obj

    out = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_videos": len(video_data),
        "use_adabn":  use_adabn,
        "videos":     [{"uid": vd["uid"], "true_class": vd["true_class"],
                        "signer": vd["signer"]} for vd in video_data],
        "results":    _ser(all_results),
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
