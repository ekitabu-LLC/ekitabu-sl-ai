#!/usr/bin/env python3
"""
Export all KSL ensemble models to ONNX format.

Exports v27, v28, v29, exp1, exp5, openhands, v41, v43 models for both
numbers and words categories.

Models use different architectures:
  - v27: KSLGraphNetV25 (ic=9, single model per category)
  - v28: KSLGraphNetV25 (ic=3, multi-stream: joint/bone/velocity)
  - v29: KSLGraphNetV29 (ic=9, single model, 8-layer dilated TCN)
  - exp1: KSLGraphNetV31Exp1 (ic=3, multi-stream, GroupNorm)
  - exp5: KSLGraphNetV25 (ic=3, multi-stream, SupCon)
  - openhands: OpenHandsClassifier (ic=2, 27 nodes, no aux input)
  - v41: KSLGraphNetV41 (ic=3, multi-stream, GroupNorm + proj_head)
  - v43: KSLGraphNetV43 (ic=3, multi-stream, BatchNorm)

Usage:
    python export_onnx.py                   # Export all models
    python export_onnx.py --model v28       # Export single model
    python export_onnx.py --category numbers  # Export numbers only
    python export_onnx.py --dry-run         # Show what would be exported
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_BASE = os.path.join(BASE_DIR, "data", "checkpoints")
ONNX_BASE = os.path.join(BASE_DIR, "onnx_models")

MAX_FRAMES = 90
NUM_NODES = 48

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])
WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market",
    "Monday", "Picture", "Proud", "Sweater", "Teach", "Tomatoes",
    "Tortoise", "Twin", "Ugali",
])

# Auxiliary feature dimensions
LH_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP = LH_PARENT + RH_PARENT + POSE_PARENT

def _build_angle_joints():
    children = defaultdict(list)
    for child_idx, parent_idx in enumerate(PARENT_MAP):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    angle_joints = []
    for node in range(48):
        parent = PARENT_MAP[node]
        if parent < 0:
            continue
        for child in children[node]:
            angle_joints.append((node, parent, child))
    return angle_joints

NUM_ANGLE_FEATURES = len(_build_angle_joints())  # 33
NUM_FINGERTIP_PAIRS = 10
NUM_HAND_BODY_FEATURES = 8
AUX_DIM = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES  # 61


# ---------------------------------------------------------------------------
# Graph topology (shared across v27/v28/v29/exp1/exp5/v41/v43)
# ---------------------------------------------------------------------------

HAND_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
POSE_EDGES = [(42, 43), (43, 44), (44, 45), (45, 46), (45, 47)]


def build_adj(n=48):
    adj = np.zeros((n, n))
    for i, j in HAND_EDGES:
        adj[i, j] = adj[j, i] = 1
    for i, j in HAND_EDGES:
        adj[i + 21, j + 21] = adj[j + 21, i + 21] = 1
    for i, j in POSE_EDGES:
        adj[i, j] = adj[j, i] = 1
    adj[0, 21] = adj[21, 0] = 0.3
    adj += np.eye(n)
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0
    return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))


# ---------------------------------------------------------------------------
# ONNX Wrapper modules — fix grl_lambda=0.0 and return only logits
# ---------------------------------------------------------------------------

class ONNXWrapperV25(nn.Module):
    """Wrapper for KSLGraphNetV25/V29/V31Exp1/V43 (returns logits, signer_logits, embedding)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, gcn, aux):
        logits, signer_logits, embedding = self.model(gcn, aux, grl_lambda=0.0)
        return logits, embedding, signer_logits


class ONNXWrapperV41(nn.Module):
    """Wrapper for KSLGraphNetV41 (returns logits, signer_logits, embedding, proj_features)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, gcn, aux):
        logits, signer_logits, embedding, proj_features = self.model(gcn, aux, grl_lambda=0.0)
        return logits, embedding, signer_logits


class ONNXWrapperOpenHands(nn.Module):
    """Wrapper for OpenHandsClassifier (returns logits, embedding)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, embedding = self.model(x)
        return logits, embedding


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_v27_model(ckpt_path, classes, device):
    """Load v27 single-model checkpoint (ic=9)."""
    from train_ksl_v27 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(config["num_nodes"]).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", AUX_DIM)

    model = KSLGraphNetV25(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config["num_nodes"], ic=config["in_channels"],
        hd=config["hidden_dim"], nl=config["num_layers"], tk=tk,
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ic = config["in_channels"]
    return model, ic, aux_dim


def load_v28_stream(ckpt_path, classes, device):
    """Load a v28 single-stream checkpoint (ic=3)."""
    from train_ksl_v28 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(48).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = AUX_DIM

    model = KSLGraphNetV25(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
        hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
        dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, 3, aux_dim


def load_v29_model(ckpt_path, classes, device):
    """Load v29 single-model checkpoint (ic=9, 8-layer dilated TCN)."""
    from train_ksl_v29 import KSLGraphNetV29
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    adj = build_adj(48).to(device)
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = ckpt.get("aux_dim", AUX_DIM)

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
    ic = config.get("in_channels", 9)
    return model, ic, aux_dim


def load_exp1_stream(ckpt_path, classes, device):
    """Load v31 exp1 (GroupNorm) single-stream checkpoint."""
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = AUX_DIM

    model = KSLGraphNetV31Exp1(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
        hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
        dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, 3, aux_dim


def load_exp5_stream(ckpt_path, classes, device):
    """Load v31 exp5 (SupCon) single-stream checkpoint — same arch as v28."""
    from train_ksl_v31_exp5 import KSLGraphNetV25
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    adj = build_adj(48).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = AUX_DIM

    model = KSLGraphNetV25(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=config.get("num_nodes", 48), ic=config.get("in_channels", 3),
        hd=config.get("hidden_dim", 64), nl=config.get("num_layers", 4), tk=tk,
        dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, 3, aux_dim


def load_openhands_model(ckpt_path, num_classes, device):
    """Load OpenHands DecoupledGCN model."""
    from train_ksl_openhands import OpenHandsClassifier
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

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


def load_v41_stream(ckpt_path, classes, device):
    """Load V41 (GroupNorm + R&R + proj_head) single-stream checkpoint."""
    from train_ksl_v41 import KSLGraphNetV41
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_nodes = config.get("num_nodes", 48)
    adj = build_adj(num_nodes).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    proj_dim = config.get("supcon_proj_dim", 128)
    aux_dim = AUX_DIM

    model = KSLGraphNetV41(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        proj_dim=proj_dim, nn_=num_nodes,
        ic=config.get("in_channels", 3),
        hd=config.get("hidden_dim", 64),
        nl=config.get("num_layers", 4),
        tk=tk, dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, 3, aux_dim


def load_v43_stream(ckpt_path, classes, device):
    """Load V43 (BatchNorm, category-conditional R&R) single-stream checkpoint."""
    from train_ksl_v43 import KSLGraphNetV43
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_nodes = config.get("num_nodes", 48)
    adj = build_adj(num_nodes).to(device)
    tk = tuple(config.get("temporal_kernels", [3, 5, 7]))
    num_signers = ckpt.get("num_signers", 12)
    aux_dim = AUX_DIM

    model = KSLGraphNetV43(
        nc=len(classes), num_signers=num_signers, aux_dim=aux_dim,
        nn_=num_nodes, ic=config.get("in_channels", 3),
        hd=config.get("hidden_dim", 64),
        nl=config.get("num_layers", 4),
        tk=tk, dr=config.get("dropout", 0.3),
        spatial_dropout=config.get("spatial_dropout", 0.1), adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, 3, aux_dim


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_gcn_model(model, wrapper_cls, ic, aux_dim, output_path, device):
    """Export a GCN-based model (v27/v28/v29/exp1/exp5/v41/v43) to ONNX."""
    wrapper = wrapper_cls(model)
    wrapper.eval()

    gcn_dummy = torch.randn(1, ic, MAX_FRAMES, NUM_NODES, device=device)
    aux_dummy = torch.randn(1, MAX_FRAMES, aux_dim, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (gcn_dummy, aux_dummy),
        output_path,
        input_names=["gcn", "aux"],
        output_names=["logits", "embeddings", "signer_logits"],
        dynamic_axes={
            "gcn": {0: "batch"},
            "aux": {0: "batch"},
            "logits": {0: "batch"},
            "embeddings": {0: "batch"},
            "signer_logits": {0: "batch"},
        },
        opset_version=17,
    )
    return True


def export_openhands_model(model, output_path, device):
    """Export OpenHands DecoupledGCN model to ONNX."""
    wrapper = ONNXWrapperOpenHands(model)
    wrapper.eval()

    # OpenHands input: (B, 2, T, 27)
    x_dummy = torch.randn(1, 2, MAX_FRAMES, 27, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (x_dummy,),
        output_path,
        input_names=["x"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "x": {0: "batch"},
            "logits": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=17,
    )
    return True


def verify_onnx(onnx_path):
    """Verify ONNX model with onnx.checker."""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return True


# ---------------------------------------------------------------------------
# Model export configurations
# ---------------------------------------------------------------------------

# Each entry: (model_name, has_streams, loader_fn, wrapper_cls, ckpt_pattern)
# ckpt_pattern uses {category} and optionally {stream}
MODEL_CONFIGS = {
    "v27": {
        "has_streams": False,
        "loader": load_v27_model,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v27_{category}/best_model.pt",
    },
    "v28": {
        "has_streams": True,
        "loader": load_v28_stream,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v28_{category}/{stream}/best_model.pt",
        "fusion_weights_dir": "v28_{category}",
    },
    "v29": {
        "has_streams": False,
        "loader": load_v29_model,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v29_{category}/best_model.pt",
    },
    "v31_exp1": {
        "has_streams": True,
        "loader": load_exp1_stream,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v31_exp1/{category}/{stream}/best_model.pt",
        "fusion_weights_dir": "v31_exp1/{category}",
    },
    "v31_exp5": {
        "has_streams": True,
        "loader": load_exp5_stream,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v31_exp5/{category}/{stream}/best_model.pt",
        "fusion_weights_dir": "v31_exp5/{category}",
    },
    "openhands": {
        "has_streams": False,
        "loader": load_openhands_model,
        "wrapper_cls": ONNXWrapperOpenHands,
        "ckpt_pattern": "openhands/{category}/best_model.pt",
        "is_openhands": True,
    },
    "v41": {
        "has_streams": True,
        "loader": load_v41_stream,
        "wrapper_cls": ONNXWrapperV41,
        "ckpt_pattern": "v41/{category}/{stream}/best_model.pt",
        "fusion_weights_dir": "v41/{category}",
    },
    "v43": {
        "has_streams": True,
        "loader": load_v43_stream,
        "wrapper_cls": ONNXWrapperV25,
        "ckpt_pattern": "v43/{category}/{stream}/best_model.pt",
        "fusion_weights_dir": "v43/{category}",
    },
}

STREAMS = ["joint", "bone", "velocity"]


def ts():
    return datetime.now().strftime("%H:%M:%S")


def export_model(model_name, category, device, dry_run=False):
    """Export a single model for a single category."""
    config = MODEL_CONFIGS[model_name]
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES
    is_openhands = config.get("is_openhands", False)
    results = []

    if config["has_streams"]:
        streams_to_export = STREAMS
    else:
        streams_to_export = [None]

    for stream in streams_to_export:
        # Build checkpoint path
        ckpt_rel = config["ckpt_pattern"].format(category=category, stream=stream)
        ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)

        # Build output path
        if stream is not None:
            onnx_path = os.path.join(ONNX_BASE, category, model_name, stream, "model.onnx")
        else:
            onnx_path = os.path.join(ONNX_BASE, category, model_name, "model.onnx")

        if not os.path.exists(ckpt_path):
            print(f"  [{ts()}] SKIP {model_name}/{category}"
                  f"{'/' + stream if stream else ''}: checkpoint not found: {ckpt_path}")
            results.append((ckpt_rel, "SKIP", "checkpoint not found"))
            continue

        label = f"{model_name}/{category}" + (f"/{stream}" if stream else "")

        if dry_run:
            print(f"  [{ts()}] DRY-RUN {label}: {ckpt_path} -> {onnx_path}")
            results.append((label, "DRY-RUN", onnx_path))
            continue

        print(f"  [{ts()}] Exporting {label}...")
        try:
            if is_openhands:
                model = config["loader"](ckpt_path, len(classes), device)
                export_openhands_model(model, onnx_path, device)
            else:
                model, ic, aux_dim = config["loader"](ckpt_path, classes, device)
                export_gcn_model(model, config["wrapper_cls"], ic, aux_dim, onnx_path, device)

            # Verify
            verify_onnx(onnx_path)
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  [{ts()}] OK {label}: {size_mb:.1f} MB, verified")
            results.append((label, "OK", f"{size_mb:.1f} MB"))

        except Exception as e:
            print(f"  [{ts()}] FAIL {label}: {e}")
            results.append((label, "FAIL", str(e)))

    # Copy fusion weights if applicable
    fw_dir_pattern = config.get("fusion_weights_dir")
    if fw_dir_pattern and not dry_run:
        fw_src_dir = os.path.join(CKPT_BASE, fw_dir_pattern.format(category=category))
        fw_src = os.path.join(fw_src_dir, "fusion_weights.json")
        fw_dst = os.path.join(ONNX_BASE, category, model_name, "fusion_weights.json")
        if os.path.exists(fw_src):
            shutil.copy2(fw_src, fw_dst)
            print(f"  [{ts()}] Copied fusion_weights.json -> {fw_dst}")
        else:
            print(f"  [{ts()}] NOTE: No fusion_weights.json at {fw_src}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export KSL models to ONNX")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Export a single model (default: all)")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"],
                        help="Category to export (default: both)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model loading (default: cpu)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be exported without exporting")
    args = parser.parse_args()

    device = torch.device(args.device)
    categories = ["numbers", "words"] if args.category == "both" else [args.category]
    models_to_export = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    print(f"[{ts()}] ONNX Export")
    print(f"  Models: {models_to_export}")
    print(f"  Categories: {categories}")
    print(f"  Device: {device}")
    print(f"  Output: {ONNX_BASE}")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print()

    all_results = []
    for category in categories:
        print(f"[{ts()}] === {category.upper()} ===")
        for model_name in models_to_export:
            results = export_model(model_name, category, device, dry_run=args.dry_run)
            all_results.extend(results)
        print()

    # Summary
    ok_count = sum(1 for _, status, _ in all_results if status == "OK")
    fail_count = sum(1 for _, status, _ in all_results if status == "FAIL")
    skip_count = sum(1 for _, status, _ in all_results if status == "SKIP")
    dry_count = sum(1 for _, status, _ in all_results if status == "DRY-RUN")

    print(f"[{ts()}] Summary: {ok_count} exported, {fail_count} failed, "
          f"{skip_count} skipped, {dry_count} dry-run")

    if fail_count > 0:
        print("\nFailed exports:")
        for label, status, msg in all_results:
            if status == "FAIL":
                print(f"  {label}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
