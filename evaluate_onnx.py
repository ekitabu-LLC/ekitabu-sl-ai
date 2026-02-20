#!/usr/bin/env python3
"""
ONNX Real-Tester Evaluation — validate all ensemble models via ONNX Runtime.

Strategy:
  - GroupNorm models (exp1, v41): run static ONNX directly (bit-exact with PyTorch).
  - BatchNorm models (v27, v28, v29, exp5, v43):
      load PyTorch → apply AdaBN on real-tester features → re-export adapted ONNX
  - OpenHands: same but uses its own AdaBN pipeline (raw arrays, no aux input)

Models evaluated:
  Individual: v27, v28, v29, v31_exp1, v31_exp5, openhands, v41, v43
  Ensemble:   6-model uniform (v27+v28+v29+exp1+exp5+openhands)

Reference numbers (PyTorch + AdaBN):
  v27:       Numbers 53.7%
  v28:       Numbers 58.2%,  Words 58.2%
  v29:       Numbers 58.4%,  Words 58.4%
  exp1:      Numbers 61.0%,  Words 61.7%
  exp5:      Numbers 61.0%,  Words 65.4%
  openhands: Numbers 37.3%,  Words 37.0%
  v41:       Numbers 67.8%,  Words 55.6%
  v43:       Numbers 66.1%,  Words 65.4%
  ensemble_6 (uniform): Numbers 74.6%, Words 71.6%, Combined 72.9%

Usage:
    python evaluate_onnx.py --category both
    python evaluate_onnx.py --category numbers --model v28
    python evaluate_onnx.py --skip-adapt   # static ONNX only (lower perf for BN models)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CKPT_BASE   = os.path.join(BASE_DIR, "data", "checkpoints")
ONNX_BASE   = os.path.join(BASE_DIR, "onnx_models")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results", "onnx")
RT_BASE     = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"

REFERENCE = {
    "v27":       {"numbers": 53.7, "words": None},
    "v28":       {"numbers": 58.2, "words": 58.2},
    "v29":       {"numbers": 58.4, "words": 58.4},
    "v31_exp1":  {"numbers": 61.0, "words": 61.7},
    "v31_exp5":  {"numbers": 61.0, "words": 65.4},
    "openhands": {"numbers": 37.3, "words": 37.0},
    "v41":       {"numbers": 67.8, "words": 55.6},
    "v43":       {"numbers": 66.1, "words": 65.4},
    "ensemble_6_uniform": {"numbers": 74.6, "words": 71.6, "combined": 72.9},
}

GROUPNORM_MODELS = {"v31_exp1", "v41"}  # no AdaBN needed
STREAMS = ["joint", "bone", "velocity"]
ENSEMBLE_6 = ["v27", "v28", "v29", "v31_exp1", "v31_exp5", "openhands"]


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Adjacency matrix (replicated to avoid cv2 imports)
# ---------------------------------------------------------------------------
def build_adj(n=48):
    HAND_EDGES = [
        (0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
    ]
    POSE_EDGES = [(42,43),(43,44),(44,45),(45,46),(45,47)]
    adj = np.zeros((n, n))
    for i, j in HAND_EDGES:
        adj[i,j] = adj[j,i] = 1
    for i, j in HAND_EDGES:
        adj[i+21,j+21] = adj[j+21,i+21] = 1
    for i, j in POSE_EDGES:
        adj[i,j] = adj[j,i] = 1
    adj[0,21] = adj[21,0] = 0.3
    adj += np.eye(n)
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0
    return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))


# ---------------------------------------------------------------------------
# ONNX wrappers (same as export_onnx.py)
# ---------------------------------------------------------------------------
class ONNXWrapperV25(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, gcn, aux):
        logits, signer_logits, embedding = self.model(gcn, aux, grl_lambda=0.0)
        return logits, embedding, signer_logits

class ONNXWrapperV41(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, gcn, aux):
        logits, signer_logits, embedding, _ = self.model(gcn, aux, grl_lambda=0.0)
        return logits, embedding, signer_logits

class ONNXWrapperOpenHands(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, x):
        logits, embedding = self.model(x)
        return logits, embedding


# ---------------------------------------------------------------------------
# PyTorch model loaders
# ---------------------------------------------------------------------------
def _load_v27(ckpt_path, classes, device):
    from train_ksl_v27 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    adj = build_adj(c["num_nodes"]).to(device)
    m = KSLGraphNetV25(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12),
        aux_dim=ckpt.get("aux_dim", 61), nn_=c["num_nodes"],
        ic=c["in_channels"], hd=c["hidden_dim"], nl=c["num_layers"],
        tk=tuple(c.get("temporal_kernels", [3,5,7])), dr=c["dropout"],
        spatial_dropout=c.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, c["in_channels"]

def _load_v28(ckpt_path, classes, device):
    from train_ksl_v28 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    m = KSLGraphNetV25(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12), aux_dim=61,
        nn_=c.get("num_nodes", 48), ic=c.get("in_channels", 3),
        hd=c.get("hidden_dim", 64), nl=c.get("num_layers", 4),
        tk=tuple(c.get("temporal_kernels", [3,5,7])), dr=c.get("dropout", 0.3),
        spatial_dropout=c.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, 3

def _load_v29(ckpt_path, classes, device):
    from train_ksl_v29 import KSLGraphNetV29
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    m = KSLGraphNetV29(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12),
        aux_dim=ckpt.get("aux_dim", 61), nn_=48, ic=c.get("in_channels", 9),
        hd=c.get("hidden_dim", 64), nl=c.get("num_layers", 8),
        td=tuple(c.get("temporal_dilations", [1,2,4])), dr=c.get("dropout", 0.2),
        spatial_dropout=c.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, c.get("in_channels", 9)

def _load_exp1(ckpt_path, classes, device):
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    m = KSLGraphNetV31Exp1(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12), aux_dim=61,
        nn_=c.get("num_nodes", 48), ic=c.get("in_channels", 3),
        hd=c.get("hidden_dim", 64), nl=c.get("num_layers", 4),
        tk=tuple(c.get("temporal_kernels", [3,5,7])), dr=c.get("dropout", 0.3),
        spatial_dropout=c.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, 3

def _load_exp5(ckpt_path, classes, device):
    from train_ksl_v31_exp5 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    m = KSLGraphNetV25(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12), aux_dim=61,
        nn_=c.get("num_nodes", 48), ic=c.get("in_channels", 3),
        hd=c.get("hidden_dim", 64), nl=c.get("num_layers", 4),
        tk=tuple(c.get("temporal_kernels", [3,5,7])), dr=c.get("dropout", 0.3),
        spatial_dropout=c.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, 3

def _load_openhands(ckpt_path, num_classes, device):
    from train_ksl_openhands import OpenHandsClassifier
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    m = OpenHandsClassifier(
        num_classes=num_classes, in_channels=c.get("in_channels", 2),
        num_nodes=c.get("num_nodes", 27), n_out_features=c.get("n_out_features", 256),
        cls_dropout=c.get("cls_dropout", 0.3),
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m

def _load_v41(ckpt_path, classes, device):
    from train_ksl_v41 import KSLGraphNetV41
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    nn_ = c.get("num_nodes", 48)
    adj = build_adj(nn_).to(device)
    m = KSLGraphNetV41(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12), aux_dim=61,
        proj_dim=c.get("supcon_proj_dim", 128), nn_=nn_,
        ic=c.get("in_channels", 3), hd=c.get("hidden_dim", 64),
        nl=c.get("num_layers", 4), tk=tuple(c.get("temporal_kernels", [3,5,7])),
        dr=c.get("dropout", 0.3), spatial_dropout=c.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, 3

def _load_v43(ckpt_path, classes, device):
    from train_ksl_v43 import KSLGraphNetV43
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    nn_ = c.get("num_nodes", 48)
    adj = build_adj(nn_).to(device)
    m = KSLGraphNetV43(
        nc=len(classes), num_signers=ckpt.get("num_signers", 12), aux_dim=61,
        nn_=nn_, ic=c.get("in_channels", 3), hd=c.get("hidden_dim", 64),
        nl=c.get("num_layers", 4), tk=tuple(c.get("temporal_kernels", [3,5,7])),
        dr=c.get("dropout", 0.3), spatial_dropout=c.get("spatial_dropout", 0.1),
        adj=adj,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, 3


# ---------------------------------------------------------------------------
# AdaBN for each model type
# ---------------------------------------------------------------------------
def _adabn_multistream(pt_model, adabn_data_ms, device, stream_name):
    """Standard AdaBN for multi-stream models (v28, exp1, exp5, v41, v43)."""
    from evaluate_real_testers_v30 import adapt_bn_stats
    adapt_bn_stats(pt_model, adabn_data_ms, device, stream_name=stream_name)


def _adabn_v27(pt_model, adabn_data_v27, device):
    """Standard AdaBN for v27/v29 (ic=9 single tensor, stream_name=None)."""
    from evaluate_real_testers_v30 import adapt_bn_stats
    adapt_bn_stats(pt_model, adabn_data_v27, device, stream_name=None)


def _adabn_openhands(pt_model, raw_list, device):
    """OpenHands AdaBN: forward raw arrays through preprocess + model in train mode."""
    from evaluate_openhands_realtest import preprocess_raw_for_openhands
    for m in pt_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None
    pt_model.train()
    with torch.no_grad():
        for raw in raw_list:
            if raw is None:
                continue
            x = preprocess_raw_for_openhands(raw)
            if x is None:
                continue
            pt_model(x.unsqueeze(0).to(device))
    pt_model.eval()


# ---------------------------------------------------------------------------
# ONNX re-export after adaptation
# ---------------------------------------------------------------------------
def _export_gcn_onnx(pt_model, wrapper_cls, ic, nn_, out_path, device):
    wrapper = wrapper_cls(pt_model)
    wrapper.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gcn_dummy = torch.randn(1, ic, 90, nn_, device=device)
    aux_dummy = torch.randn(1, 90, 61, device=device)
    torch.onnx.export(
        wrapper, (gcn_dummy, aux_dummy), out_path,
        input_names=["gcn", "aux"],
        output_names=["logits", "embeddings", "signer_logits"],
        dynamic_axes={"gcn": {0: "batch"}, "aux": {0: "batch"},
                      "logits": {0: "batch"}, "embeddings": {0: "batch"},
                      "signer_logits": {0: "batch"}},
        opset_version=17,
    )


def _export_openhands_onnx(pt_model, out_path, device):
    wrapper = ONNXWrapperOpenHands(pt_model)
    wrapper.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x_dummy = torch.randn(1, 2, 90, 27, device=device)
    torch.onnx.export(
        wrapper, (x_dummy,), out_path,
        input_names=["x"], output_names=["logits", "embeddings"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"},
                      "embeddings": {0: "batch"}},
        opset_version=17,
    )


# ---------------------------------------------------------------------------
# Build adapted ONNX path helpers
# ---------------------------------------------------------------------------
def _onnx_static_path(model_name, category, stream=None):
    base = os.path.join(ONNX_BASE, category, model_name)
    return os.path.join(base, stream, "model.onnx") if stream else os.path.join(base, "model.onnx")

def _onnx_adapted_path(model_name, category, stream=None):
    base = os.path.join(ONNX_BASE, category, model_name)
    return os.path.join(base, stream, "model_adapted.onnx") if stream else os.path.join(base, "model_adapted.onnx")

def _ckpt_path(pattern, category, stream=None):
    return os.path.join(CKPT_BASE, pattern.format(category=category, stream=stream))

def _fw_path(model_name, category):
    return os.path.join(ONNX_BASE, category, model_name, "fusion_weights.json")

def _load_fw(model_name, category, streams):
    p = _fw_path(model_name, category)
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        # fusion_weights.json may have {"weights": {...}} or be flat
        if "weights" in data and isinstance(data["weights"], dict):
            return data["weights"]
        # Flat format (direct stream→weight mapping)
        if all(k in data for k in streams):
            return {s: data[s] for s in streams}
    # Fallback: equal weights
    return {s: 1.0 / len(streams) for s in streams}

def _ort(onnx_path, use_gpu):
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_gpu else ["CPUExecutionProvider"])
    return ort.InferenceSession(onnx_path, providers=providers)


# ---------------------------------------------------------------------------
# Pre-extract data for v27/v29 AdaBN (ic=9 tensors, not dict)
# ---------------------------------------------------------------------------
def preextract_v27_data(cat_videos):
    """Extract (gcn_9ch, aux) pairs using preprocess_v27 for v27/v29 AdaBN."""
    from evaluate_real_testers_v30 import preprocess_v27
    from evaluate_real_testers_v30 import extract_landmarks_from_video
    data = []
    for video_path, true_class, signer in cat_videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
            if raw is None:
                continue
        gcn, aux = preprocess_v27(raw)
        if gcn is None:
            continue
        data.append((gcn, aux))
    return data


# ---------------------------------------------------------------------------
# Pre-extract raw arrays for OpenHands AdaBN
# ---------------------------------------------------------------------------
def preextract_raw_arrays(cat_videos):
    """Load raw landmark arrays for openhands AdaBN."""
    from evaluate_real_testers_v30 import extract_landmarks_from_video
    raws = []
    for video_path, true_class, signer in cat_videos:
        npy_cache = video_path + ".landmarks.npy"
        if os.path.exists(npy_cache):
            raw = np.load(npy_cache)
        else:
            raw = extract_landmarks_from_video(video_path)
        raws.append(raw)
    return raws


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "v27": {
        "loader": _load_v27,
        "wrapper": ONNXWrapperV25,
        "has_streams": False,
        "needs_adabn": True,
        "adabn_type": "v27",
        "ckpt": "v27_{category}/best_model.pt",
        "nn_": 48,
    },
    "v28": {
        "loader": _load_v28,
        "wrapper": ONNXWrapperV25,
        "has_streams": True,
        "needs_adabn": True,
        "adabn_type": "multistream",
        "ckpt": "v28_{category}/{stream}/best_model.pt",
        "nn_": 48,
    },
    "v29": {
        "loader": _load_v29,
        "wrapper": ONNXWrapperV25,
        "has_streams": False,
        "needs_adabn": True,
        "adabn_type": "v27",          # same preprocess as v27
        "ckpt": "v29_{category}/best_model.pt",
        "nn_": 48,
    },
    "v31_exp1": {
        "loader": _load_exp1,
        "wrapper": ONNXWrapperV25,
        "has_streams": True,
        "needs_adabn": False,
        "adabn_type": None,
        "ckpt": "v31_exp1/{category}/{stream}/best_model.pt",
        "nn_": 48,
    },
    "v31_exp5": {
        "loader": _load_exp5,
        "wrapper": ONNXWrapperV25,
        "has_streams": True,
        "needs_adabn": True,
        "adabn_type": "multistream",
        "ckpt": "v31_exp5/{category}/{stream}/best_model.pt",
        "nn_": 48,
    },
    "openhands": {
        "loader": _load_openhands,
        "wrapper": ONNXWrapperOpenHands,
        "has_streams": False,
        "needs_adabn": True,
        "adabn_type": "openhands",
        "ckpt": "openhands/{category}/best_model.pt",
        "nn_": 27,
    },
    "v41": {
        "loader": _load_v41,
        "wrapper": ONNXWrapperV41,
        "has_streams": True,
        "needs_adabn": False,
        "adabn_type": None,
        "ckpt": "v41/{category}/{stream}/best_model.pt",
        "nn_": 48,
    },
    "v43": {
        "loader": _load_v43,
        "wrapper": ONNXWrapperV25,
        "has_streams": True,
        "needs_adabn": True,
        "adabn_type": "multistream",
        "ckpt": "v43/{category}/{stream}/best_model.pt",
        "nn_": 48,
    },
}


# ---------------------------------------------------------------------------
# Build ONNX sessions for one model + category
# ---------------------------------------------------------------------------
def build_sessions(model_name, category, classes,
                   adabn_data_ms, adabn_data_v27, adabn_raw,
                   device, skip_adapt, use_gpu):
    cfg = MODEL_CONFIGS[model_name]
    streams = STREAMS if cfg["has_streams"] else [None]
    sessions = {}

    for stream in streams:
        static_path = _onnx_static_path(model_name, category, stream)
        if not os.path.exists(static_path):
            print(f"    SKIP: static ONNX not found: {static_path}")
            return None, None

        if cfg["needs_adabn"] and not skip_adapt:
            adapted_path = _onnx_adapted_path(model_name, category, stream)
            if not os.path.exists(adapted_path):
                ckpt = _ckpt_path(cfg["ckpt"], category, stream)
                if not os.path.exists(ckpt):
                    print(f"    SKIP: checkpoint not found: {ckpt}")
                    return None, None

                # Load PyTorch model
                if model_name == "openhands":
                    pt_model = cfg["loader"](ckpt, len(classes), device)
                else:
                    pt_model, ic = cfg["loader"](ckpt, classes, device)

                # Apply AdaBN
                label = f"{model_name}/{category}" + (f"/{stream}" if stream else "")
                print(f"    [{ts()}] Applying AdaBN → {label}...")

                if cfg["adabn_type"] == "multistream":
                    _adabn_multistream(pt_model, adabn_data_ms, device, stream)
                elif cfg["adabn_type"] == "v27":
                    _adabn_v27(pt_model, adabn_data_v27, device)
                elif cfg["adabn_type"] == "openhands":
                    _adabn_openhands(pt_model, adabn_raw, device)

                # Re-export ONNX with adapted BN stats
                if model_name == "openhands":
                    _export_openhands_onnx(pt_model, adapted_path, device)
                else:
                    _export_gcn_onnx(pt_model, cfg["wrapper"], ic,
                                     cfg["nn_"], adapted_path, device)
                print(f"    [{ts()}] Saved adapted ONNX: {adapted_path}")
            else:
                print(f"    [{ts()}] Using cached adapted ONNX")
            use_path = adapted_path
        else:
            if cfg["needs_adabn"] and skip_adapt:
                print(f"    NOTE: using static ONNX for {model_name} "
                      f"(BatchNorm, no AdaBN — performance will be lower)")
            use_path = static_path

        skey = stream if stream else "model"
        sessions[skey] = _ort(use_path, use_gpu)

    fw = _load_fw(model_name, category, list(sessions.keys()))
    return sessions, fw


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _infer_multistream(sessions, fw, streams, aux):
    per = {}
    aux_np = aux.unsqueeze(0).numpy()
    for s, sess in sessions.items():
        gcn_np = streams[s].unsqueeze(0).numpy()
        logits = sess.run(["logits"], {"gcn": gcn_np, "aux": aux_np})[0]
        per[s] = F.softmax(torch.tensor(logits), dim=1).squeeze(0)
    return sum(fw[s] * per[s] for s in sessions)

def _infer_combined9(sessions, streams, aux):
    # v27/v29 trained with order: joint | velocity | bone (matches preprocess_v27)
    gcn_9 = torch.cat([streams["joint"], streams["velocity"], streams["bone"]], dim=0)
    gcn_np = gcn_9.unsqueeze(0).numpy()
    aux_np = aux.unsqueeze(0).numpy()
    logits = sessions["model"].run(["logits"], {"gcn": gcn_np, "aux": aux_np})[0]
    return F.softmax(torch.tensor(logits), dim=1).squeeze(0)

def _infer_openhands(sessions, raw):
    from evaluate_openhands_realtest import preprocess_raw_for_openhands
    x = preprocess_raw_for_openhands(raw)
    if x is None:
        return None
    x_np = x.unsqueeze(0).numpy()
    logits = sessions["model"].run(["logits"], {"x": x_np})[0]
    return F.softmax(torch.tensor(logits), dim=1).squeeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ONNX Real-Tester Evaluation")
    parser.add_argument("--category", default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--model", default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--skip-adapt", action="store_true",
                        help="Use static ONNX (no AdaBN). Lower perf for BN models.")
    parser.add_argument("--no-ensemble", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    print("=" * 70)
    print("ONNX Real-Tester Evaluation")
    print(f"  Device:   {device}")
    print(f"  Category: {args.category}")
    print(f"  AdaBN:    {'DISABLED (static ONNX)' if args.skip_adapt else 'ENABLED'}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Import heavy deps lazily
    from evaluate_real_testers_v30 import (
        NUMBER_CLASSES, WORD_CLASSES,
        check_mediapipe_version,
        discover_test_videos,
        preprocess_multistream,
        preextract_test_data,
        evaluate_method,
    )
    check_mediapipe_version()

    print(f"\nScanning {RT_BASE}...")
    all_videos = discover_test_videos(RT_BASE)
    numbers_videos = [(v, n, s) for v, n, c, s in all_videos if c == "numbers"]
    words_videos   = [(v, n, s) for v, n, c, s in all_videos if c == "words"]
    print(f"Found {len(all_videos)} test videos "
          f"(Numbers: {len(numbers_videos)}, Words: {len(words_videos)})")

    cats = []
    if args.category in ("numbers", "both"):
        cats.append(("numbers", numbers_videos, NUMBER_CLASSES))
    if args.category in ("words", "both"):
        cats.append(("words", words_videos, WORD_CLASSES))

    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS.keys())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for cat_name, cat_videos, classes in cats:
        i2c = {i: c for i, c in enumerate(classes)}
        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} ({len(cat_videos)} test videos)")
        print(f"{'='*70}")

        # Pre-extract feature data for AdaBN
        if not args.skip_adapt:
            print(f"\n  Pre-extracting multistream data for AdaBN...")
            adabn_data_ms  = preextract_test_data(cat_videos)
            print(f"  Pre-extracting v27/v29 (ic=9) data for AdaBN...")
            adabn_data_v27 = preextract_v27_data(cat_videos)
            print(f"  Pre-extracting raw arrays for OpenHands AdaBN...")
            adabn_raw      = preextract_raw_arrays(cat_videos)
            print(f"  Done ({len(adabn_data_ms)} ms, {len(adabn_data_v27)} v27, "
                  f"{len(adabn_raw)} raw)")
        else:
            adabn_data_ms = adabn_data_v27 = adabn_raw = []

        cat_results = {}
        all_sessions_cat   = {}
        all_fw_cat         = {}

        # Per-model evaluation
        print(f"\n  {'MODEL':<22} {'ONNX %':>8} {'PyTorch Ref':>13} {'Delta':>7}")
        print(f"  {'-'*22} {'-'*8} {'-'*13} {'-'*7}")

        for mname in models_to_run:
            cfg = MODEL_CONFIGS[mname]

            print(f"\n  [{ts()}] Loading {mname}/{cat_name}...")
            sessions, fw = build_sessions(
                mname, cat_name, classes,
                adabn_data_ms, adabn_data_v27, adabn_raw,
                device, args.skip_adapt, use_gpu,
            )
            if sessions is None:
                print(f"  {mname:<22} {'SKIP':>8}")
                continue

            all_sessions_cat[mname] = sessions
            all_fw_cat[mname]       = fw

            # Build predict function
            def make_predict(mname_=mname, sessions_=sessions, fw_=fw):
                def predict(raw, true_class):
                    cfg_ = MODEL_CONFIGS[mname_]
                    if cfg_["has_streams"]:
                        streams, aux = preprocess_multistream(raw)
                        if streams is None:
                            return None
                        probs = _infer_multistream(sessions_, fw_, streams, aux)
                    elif cfg_["adabn_type"] == "openhands":
                        probs = _infer_openhands(sessions_, raw)
                        if probs is None:
                            return None
                    else:  # combined9 (v27/v29)
                        streams, aux = preprocess_multistream(raw)
                        if streams is None:
                            return None
                        probs = _infer_combined9(sessions_, streams, aux)
                    pred_idx = probs.argmax().item()
                    return i2c[pred_idx], probs[pred_idx].item()
                return predict

            result = evaluate_method(f"onnx_{mname}_{cat_name}",
                                     cat_videos, make_predict(), classes)
            cat_results[mname] = result

            acc = result["overall"]
            ref = REFERENCE.get(mname, {}).get(cat_name)
            ref_str = f"{ref:.1f}%" if ref is not None else "  N/A "
            delta = f"{acc-ref:+.1f}pp" if ref is not None else "  N/A "
            print(f"  {mname:<22} {acc:>7.1f}% {ref_str:>13} {delta:>7}")

        # Ensemble evaluation
        if not args.no_ensemble:
            avail = [m for m in ENSEMBLE_6 if m in all_sessions_cat]
            if len(avail) == len(ENSEMBLE_6):
                print(f"\n  [{ts()}] Running 6-model uniform ensemble...")

                def make_ens_predict(avail_=avail, sessions_=all_sessions_cat,
                                     fw_=all_fw_cat, i2c_=i2c):
                    def predict(raw, true_class):
                        streams_cache = None
                        ensemble_probs = None
                        n = len(avail_)
                        for mname in avail_:
                            cfg_ = MODEL_CONFIGS[mname]
                            sess = sessions_[mname]
                            fw = fw_[mname]
                            if cfg_["has_streams"]:
                                if streams_cache is None:
                                    streams_cache, aux_cache = preprocess_multistream(raw)
                                if streams_cache is None:
                                    return None
                                probs = _infer_multistream(sess, fw, streams_cache, aux_cache)
                            elif cfg_["adabn_type"] == "openhands":
                                probs = _infer_openhands(sess, raw)
                                if probs is None:
                                    return None
                            else:  # combined9
                                if streams_cache is None:
                                    streams_cache, aux_cache = preprocess_multistream(raw)
                                if streams_cache is None:
                                    return None
                                probs = _infer_combined9(sess, streams_cache, aux_cache)
                            ensemble_probs = probs if ensemble_probs is None else (ensemble_probs + probs)
                        if ensemble_probs is None:
                            return None
                        ensemble_probs = ensemble_probs / n
                        pred_idx = ensemble_probs.argmax().item()
                        return i2c_[pred_idx], ensemble_probs[pred_idx].item()
                    return predict

                result = evaluate_method(f"onnx_ensemble6_{cat_name}",
                                         cat_videos, make_ens_predict(), classes)
                cat_results["ensemble_6_uniform"] = result

                acc = result["overall"]
                ref = REFERENCE["ensemble_6_uniform"].get(cat_name)
                ref_str = f"{ref:.1f}%" if ref is not None else "  N/A "
                delta = f"{acc-ref:+.1f}pp" if ref is not None else "  N/A "
                print(f"\n  {'ensemble_6_uniform':<22} {acc:>7.1f}% {ref_str:>13} {delta:>7}")
            else:
                missing = set(ENSEMBLE_6) - set(avail)
                print(f"\n  Ensemble_6 SKIP: missing models {missing}")

        all_results[cat_name] = cat_results

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY (Numbers + Words)")
    print(f"{'='*70}")
    all_keys = list(MODEL_CONFIGS.keys()) + ["ensemble_6_uniform"]
    if "numbers" in all_results and "words" in all_results:
        n_res_all = all_results["numbers"]
        w_res_all = all_results["words"]
        for k in all_keys:
            if k not in n_res_all or k not in w_res_all:
                continue
            n_acc = n_res_all[k]["overall"]
            w_acc = w_res_all[k]["overall"]
            n_tot = n_res_all[k].get("total", 0)
            w_tot = w_res_all[k].get("total", 0)
            total = n_tot + w_tot
            combined = 100.0 * (round(n_acc/100*n_tot) + round(w_acc/100*w_tot)) / total
            ref_c = REFERENCE.get(k, {}).get("combined")
            ref_str = f"(ref {ref_c:.1f}%)" if ref_c else ""
            print(f"  {k:<22} N={n_acc:.1f}%  W={w_acc:.1f}%  "
                  f"Combined={combined:.1f}% {ref_str}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "onnx_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"  Adapted ONNX models saved in {ONNX_BASE}/{{category}}/{{model}}/model_adapted.onnx")


if __name__ == "__main__":
    main()
