#!/usr/bin/env python3
"""
KSL v17 Evaluation & Comparison Script

Loads v16 and v17 results JSONs and prints a side-by-side comparison with
ANSI color highlighting (green for improvements, red for regressions).

Usage:
    # Compare v17 results against v16 baseline
    python evaluate_v17.py

    # Specify custom result paths
    python evaluate_v17.py --v16 data/results/v16_both_20260208_125613.json \
                           --v17 data/results/v17_both_TIMESTAMP.json

    # Run confusion matrix analysis from checkpoint
    python evaluate_v17.py --confusion

    # Auto-find most recent v17 results
    python evaluate_v17.py --results-dir data/results/
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Class Definitions
# ---------------------------------------------------------------------------

NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

V8_BASELINE = 83.65  # target to beat

# Known hard classes for numbers
HARD_NUMBER_CLASSES = {"444", "35", "73", "89", "91"}

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    WHITE = "\033[97m"


def colored_delta(delta, threshold=0.5):
    """Return colored string for a delta value."""
    if abs(delta) < threshold:
        return f"{C.DIM}{delta:+6.1f}{C.RESET}"
    elif delta > 0:
        return f"{C.GREEN}{delta:+6.1f}{C.RESET}"
    else:
        return f"{C.RED}{delta:+6.1f}{C.RESET}"


def colored_acc(acc):
    """Color-code an accuracy value."""
    if acc >= 90:
        return f"{C.GREEN}{acc:6.1f}%{C.RESET}"
    elif acc >= 70:
        return f"{C.YELLOW}{acc:6.1f}%{C.RESET}"
    elif acc >= 40:
        return f"{C.RED}{acc:6.1f}%{C.RESET}"
    else:
        return f"{C.RED}{C.BOLD}{acc:6.1f}%{C.RESET}"


# ---------------------------------------------------------------------------
# Results Loading
# ---------------------------------------------------------------------------

def find_latest_results(results_dir, prefix):
    """Find the most recent results JSON matching a prefix."""
    pattern = os.path.join(results_dir, f"{prefix}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_results(path):
    """Load a results JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_per_class(results, model_type):
    """Extract per-class accuracy dict from results JSON."""
    if "results" in results and model_type in results["results"]:
        return results["results"][model_type].get("per_class", {})
    return {}


def extract_overall(results, model_type):
    """Extract overall accuracy for a model type."""
    if "results" in results and model_type in results["results"]:
        return results["results"][model_type].get("overall", 0.0)
    return 0.0


def extract_meta(results, model_type):
    """Extract metadata (epoch, params) for a model type."""
    meta = {}
    if "results" in results and model_type in results["results"]:
        r = results["results"][model_type]
        meta["best_epoch"] = r.get("best_epoch", "?")
        meta["params"] = r.get("params", "?")
    return meta


def compute_combined(results):
    """Compute combined accuracy across numbers and words."""
    num_overall = extract_overall(results, "numbers")
    word_overall = extract_overall(results, "words")

    num_per_class = extract_per_class(results, "numbers")
    word_per_class = extract_per_class(results, "words")

    # Weighted by number of classes (15 each, so simple average)
    if num_per_class and word_per_class:
        all_accs = list(num_per_class.values()) + list(word_per_class.values())
        return sum(all_accs) / len(all_accs)
    elif num_per_class:
        return num_overall
    elif word_per_class:
        return word_overall
    return 0.0


# ---------------------------------------------------------------------------
# Comparison Display
# ---------------------------------------------------------------------------

def print_header(title):
    """Print a section header."""
    width = 78
    print()
    print(f"{C.BOLD}{'=' * width}{C.RESET}")
    print(f"{C.BOLD}  {title}{C.RESET}")
    print(f"{C.BOLD}{'=' * width}{C.RESET}")


def print_model_info(v16, v17):
    """Print model metadata comparison."""
    print_header("Model Information")

    for model_type, label in [("numbers", "Numbers"), ("words", "Words")]:
        m16 = extract_meta(v16, model_type)
        m17 = extract_meta(v17, model_type)

        print(f"\n  {C.CYAN}{label}{C.RESET}:")
        print(f"    {'':20s}  {'v16':>12s}  {'v17':>12s}")
        print(f"    {'Parameters':20s}  {str(m16.get('params', '?')):>12s}  {str(m17.get('params', '?')):>12s}")
        print(f"    {'Best Epoch':20s}  {str(m16.get('best_epoch', '?')):>12s}  {str(m17.get('best_epoch', '?')):>12s}")

    # Configs comparison (key hyperparams)
    for model_type, label in [("numbers", "Numbers"), ("words", "Words")]:
        config_key = f"{model_type}_config"
        c16 = v16.get(config_key, {})
        c17 = v17.get(config_key, {})
        if c16 and c17:
            changes = []
            for key in sorted(set(list(c16.keys()) + list(c17.keys()))):
                v16_val = c16.get(key)
                v17_val = c17.get(key)
                if v16_val != v17_val:
                    changes.append((key, v16_val, v17_val))
            if changes:
                print(f"\n  {C.CYAN}{label} Config Changes:{C.RESET}")
                for key, old, new in changes:
                    print(f"    {key:30s}  {str(old):>12s} -> {str(new):<12s}")


def print_class_comparison(classes, v16_per_class, v17_per_class, section_name, hard_classes=None):
    """Print per-class comparison table."""
    if hard_classes is None:
        hard_classes = set()

    print(f"\n  {C.CYAN}{section_name}{C.RESET}")
    print(f"  {'Class':>12s}  {'v16':>8s}  {'v17':>8s}  {'Delta':>8s}  {'Status'}")
    print(f"  {'-' * 60}")

    improvements = 0
    regressions = 0
    unchanged = 0
    total_v16 = 0.0
    total_v17 = 0.0

    for cls in classes:
        v16_acc = v16_per_class.get(cls, 0.0)
        v17_acc = v17_per_class.get(cls, 0.0)
        delta = v17_acc - v16_acc
        total_v16 += v16_acc
        total_v17 += v17_acc

        # Status indicator
        if delta > 0.5:
            improvements += 1
            status = f"{C.GREEN}IMPROVED{C.RESET}"
        elif delta < -0.5:
            regressions += 1
            status = f"{C.RED}REGRESSED{C.RESET}"
        else:
            unchanged += 1
            status = f"{C.DIM}---{C.RESET}"

        # Mark hard classes
        marker = ""
        if cls in hard_classes:
            marker = f" {C.YELLOW}*{C.RESET}"

        # Mark zero-accuracy classes
        if v17_acc < 0.5:
            status = f"{C.RED}{C.BOLD}ZERO{C.RESET}"

        delta_str = colored_delta(delta)
        v16_str = colored_acc(v16_acc)
        v17_str = colored_acc(v17_acc)

        print(f"  {cls:>12s}  {v16_str}  {v17_str}  {delta_str}  {status}{marker}")

    # Summary line
    avg_v16 = total_v16 / len(classes) if classes else 0
    avg_v17 = total_v17 / len(classes) if classes else 0
    avg_delta = avg_v17 - avg_v16

    print(f"  {'-' * 60}")
    print(f"  {'Mean':>12s}  {colored_acc(avg_v16)}  {colored_acc(avg_v17)}  {colored_delta(avg_delta)}")
    print(f"  {C.DIM}  Improved: {improvements}  |  Regressed: {regressions}  |  Unchanged: {unchanged}{C.RESET}")

    return improvements, regressions, unchanged


def print_overall_comparison(v16, v17):
    """Print overall accuracy comparison."""
    print_header("Overall Accuracy")

    num_v16 = extract_overall(v16, "numbers")
    num_v17 = extract_overall(v17, "numbers")
    word_v16 = extract_overall(v16, "words")
    word_v17 = extract_overall(v17, "words")
    comb_v16 = compute_combined(v16)
    comb_v17 = compute_combined(v17)

    print(f"\n  {'Category':>12s}  {'v16':>8s}  {'v17':>8s}  {'Delta':>8s}")
    print(f"  {'-' * 45}")
    print(f"  {'Numbers':>12s}  {colored_acc(num_v16)}  {colored_acc(num_v17)}  {colored_delta(num_v17 - num_v16)}")
    print(f"  {'Words':>12s}  {colored_acc(word_v16)}  {colored_acc(word_v17)}  {colored_delta(word_v17 - word_v16)}")
    print(f"  {'-' * 45}")
    print(f"  {'Combined':>12s}  {colored_acc(comb_v16)}  {colored_acc(comb_v17)}  {colored_delta(comb_v17 - comb_v16)}")
    print(f"  {'v8 Target':>12s}  {C.BOLD}{V8_BASELINE:6.1f}%{C.RESET}")

    if comb_v17 >= V8_BASELINE:
        print(f"\n  {C.BG_GREEN}{C.WHITE}{C.BOLD}  TARGET MET: v17 ({comb_v17:.2f}%) >= v8 baseline ({V8_BASELINE}%)  {C.RESET}")
    else:
        gap = V8_BASELINE - comb_v17
        print(f"\n  {C.BG_RED}{C.WHITE}  Gap to target: {gap:.2f}% below v8 baseline  {C.RESET}")


def print_highlights(v16, v17):
    """Print biggest gains and losses."""
    print_header("Highlights")

    deltas = []
    for model_type, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        v16_pc = extract_per_class(v16, model_type)
        v17_pc = extract_per_class(v17, model_type)
        for cls in classes:
            v16_acc = v16_pc.get(cls, 0.0)
            v17_acc = v17_pc.get(cls, 0.0)
            delta = v17_acc - v16_acc
            deltas.append((cls, model_type, v16_acc, v17_acc, delta))

    deltas.sort(key=lambda x: x[4], reverse=True)

    # Top gains
    gains = [d for d in deltas if d[4] > 0.5]
    if gains:
        print(f"\n  {C.GREEN}Biggest Gains:{C.RESET}")
        for cls, mt, v16a, v17a, delta in gains[:5]:
            print(f"    {cls:>12s} ({mt:7s}): {v16a:5.1f}% -> {v17a:5.1f}% ({C.GREEN}{delta:+.1f}{C.RESET})")

    # Top losses
    losses = [d for d in deltas if d[4] < -0.5]
    losses.sort(key=lambda x: x[4])
    if losses:
        print(f"\n  {C.RED}Biggest Losses:{C.RESET}")
        for cls, mt, v16a, v17a, delta in losses[:5]:
            print(f"    {cls:>12s} ({mt:7s}): {v16a:5.1f}% -> {v17a:5.1f}% ({C.RED}{delta:+.1f}{C.RESET})")

    # Zero-accuracy classes in v17
    zeros_v17 = [d for d in deltas if d[3] < 0.5]
    if zeros_v17:
        print(f"\n  {C.RED}{C.BOLD}Zero-Accuracy Classes in v17:{C.RESET}")
        for cls, mt, v16a, v17a, delta in zeros_v17:
            was = f"was {v16a:.1f}%" if v16a > 0 else "also 0% in v16"
            print(f"    {cls:>12s} ({mt:7s}): {C.RED}0%{C.RESET} ({was})")

    # Classes that went from 0 to nonzero
    recovered = [d for d in deltas if d[2] < 0.5 and d[3] > 0.5]
    if recovered:
        print(f"\n  {C.GREEN}{C.BOLD}Recovered Classes (0% -> nonzero):{C.RESET}")
        for cls, mt, v16a, v17a, delta in recovered:
            print(f"    {cls:>12s} ({mt:7s}): 0% -> {C.GREEN}{v17a:.1f}%{C.RESET}")


def print_config_summary(v17):
    """Print v17 config summary for quick reference."""
    print_header("v17 Configuration Summary")

    for model_type, label in [("numbers", "Numbers"), ("words", "Words")]:
        config = v17.get(f"{model_type}_config", {})
        if not config:
            continue

        print(f"\n  {C.CYAN}{label}:{C.RESET}")
        key_params = [
            ("hidden_dim", "Hidden Dim"),
            ("num_layers", "Layers"),
            ("dropout", "Dropout"),
            ("learning_rate", "LR"),
            ("weight_decay", "Weight Decay"),
            ("label_smoothing", "Label Smoothing"),
            ("batch_size", "Batch Size"),
            ("epochs", "Max Epochs"),
            ("patience", "Patience"),
        ]
        for key, label_name in key_params:
            if key in config:
                print(f"    {label_name:20s}: {config[key]}")


# ---------------------------------------------------------------------------
# Confusion Matrix Analysis (requires checkpoint + val data)
# ---------------------------------------------------------------------------

def run_confusion_analysis(v17_results_path, val_dir, device_str="cuda"):
    """Run confusion matrix analysis by loading v17 checkpoints and val data."""
    import torch
    import torch.nn as nn

    # Determine checkpoint paths from results
    results = load_results(v17_results_path)
    version = results.get("version", "v17")

    # Graph topology
    HAND_EDGES = [
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
    ]
    POSE_EDGES = [
        (42, 43), (42, 44), (43, 45), (44, 46), (45, 47), (46, 0), (47, 21),
    ]
    POSE_INDICES = [11, 12, 13, 14, 15, 16]

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

    class GConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.fc = nn.Linear(i, o)
        def forward(self, x, adj):
            return self.fc(torch.matmul(adj, x))

    class STGCNBlock(nn.Module):
        def __init__(self, ic, oc, adj, ks=9, st=1, dr=0.3):
            super().__init__()
            self.register_buffer("adj", adj)
            self.gcn = GConv(ic, oc)
            self.bn1 = nn.BatchNorm2d(oc)
            self.tcn = nn.Sequential(
                nn.Conv2d(oc, oc, (ks, 1), padding=(ks // 2, 0), stride=(st, 1)),
                nn.BatchNorm2d(oc),
            )
            self.residual = (
                nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc))
                if ic != oc or st != 1
                else nn.Identity()
            )
            self.dropout = nn.Dropout(dr)

        def forward(self, x):
            r = self.residual(x)
            b, c, t, n = x.shape
            x = self.gcn(x.permute(0, 2, 3, 1).reshape(b * t, n, c), self.adj)
            x = self.dropout(
                self.tcn(torch.relu(self.bn1(x.reshape(b, t, n, -1).permute(0, 3, 1, 2))))
            )
            return torch.relu(x + r)

    class KSLGraphNet(nn.Module):
        def __init__(self, nc, nn_=48, ic=3, hd=128, nl=6, tk=9, dr=0.3, adj=None,
                     use_anti_attractor=False):
            super().__init__()
            self.register_buffer("adj", adj)
            self.data_bn = nn.BatchNorm1d(nn_ * ic)
            ch = [ic] + [hd] * 2 + [hd * 2] * 2 + [hd * 4] * 2
            ch = ch[: nl + 1]
            self.layers = nn.ModuleList(
                [STGCNBlock(ch[i], ch[i + 1], adj, tk, 2 if i in [2, 4] else 1, dr) for i in range(nl)]
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            final_ch = ch[-1]
            self.classifier = nn.Sequential(
                nn.Linear(final_ch, hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc)
            )
            self.use_anti_attractor = use_anti_attractor
            if use_anti_attractor:
                self.anti_attractor = nn.Sequential(
                    nn.Linear(final_ch, hd // 2), nn.GELU(), nn.Linear(hd // 2, 1),
                )

        def forward(self, x):
            b, c, t, n = x.shape
            x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * n, t)).reshape(b, c, n, t).permute(0, 1, 3, 2)
            for layer in self.layers:
                x = layer(x)
            return self.classifier(self.pool(x).view(b, -1))

    def preprocess_sample(data, max_frames=90):
        """Preprocess a single .npy sample into model input."""
        f = data.shape[0]
        if data.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                start = idx_pose * 3
                pose[:, pi, :] = data[:, start:start + 3]
            lh = data[:, 99:162].reshape(f, 21, 3)
            rh = data[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
        rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01
        if np.any(lh_valid):
            h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
        if np.any(rh_valid):
            h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

        mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
        h[:, 42:48, :] -= mid_shoulder

        max_val = np.abs(h).max()
        if max_val > 0.01:
            h = np.clip(h / max_val, -1, 1).astype(np.float32)

        if f >= max_frames:
            indices = np.linspace(0, f - 1, max_frames, dtype=int)
            h = h[indices]
        else:
            h = np.concatenate([h, np.zeros((max_frames - f, 48, 3), dtype=np.float32)])

        return torch.FloatTensor(h).permute(2, 0, 1).unsqueeze(0)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    adj = build_adj(48).to(device)

    base_dir = os.path.dirname(os.path.dirname(v17_results_path))
    ckpt_dirs = {
        "numbers": os.path.join(base_dir, "checkpoints", "v17_numbers", "best_model.pt"),
        "words": os.path.join(base_dir, "checkpoints", "v17_words", "best_model.pt"),
    }

    for model_type, classes in [("numbers", NUMBER_CLASSES), ("words", WORD_CLASSES)]:
        ckpt_path = ckpt_dirs[model_type]
        if not os.path.exists(ckpt_path):
            print(f"\n  {C.YELLOW}Checkpoint not found: {ckpt_path}{C.RESET}")
            print(f"  Skipping {model_type} confusion matrix analysis.")
            continue

        print_header(f"Confusion Matrix: {model_type.title()}")

        # Load config from results to match model architecture
        config = results.get(f"{model_type}_config", {})
        hd = config.get("hidden_dim", 128)
        nl = config.get("num_layers", 6)
        tk = config.get("temporal_kernel", 9)
        dr = config.get("dropout", 0.3)

        model = KSLGraphNet(
            nc=len(classes), nn_=48, ic=3, hd=hd, nl=nl, tk=tk, dr=dr,
            adj=adj, use_anti_attractor=False,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle both key conventions
        state_key = "model" if "model" in ckpt else "model_state_dict"
        model.load_state_dict(ckpt[state_key], strict=False)
        model.eval()

        c2i = {c: i for i, c in enumerate(classes)}
        confusion = np.zeros((len(classes), len(classes)), dtype=int)
        class_total = np.zeros(len(classes), dtype=int)
        class_correct = np.zeros(len(classes), dtype=int)

        print(f"  Running inference on validation data...")
        with torch.no_grad():
            for cls in classes:
                cls_dir = os.path.join(val_dir, cls)
                if not os.path.exists(cls_dir):
                    continue
                label = c2i[cls]
                for fn in sorted(os.listdir(cls_dir)):
                    if not fn.endswith(".npy"):
                        continue
                    data = np.load(os.path.join(cls_dir, fn)).astype(np.float32)
                    if data.shape[0] < 5:
                        continue
                    inp = preprocess_sample(data).to(device)
                    pred = model(inp).argmax(dim=1).item()
                    confusion[label, pred] += 1
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

        # Print confusion matrix
        print(f"\n  {'':>12s}", end="")
        for j, cls in enumerate(classes):
            short = cls[:4]
            print(f" {short:>5s}", end="")
        print(f"  {'Acc':>6s}  {'N':>4s}")
        print(f"  {'-' * (12 + 5 * len(classes) + 14)}")

        for i, cls in enumerate(classes):
            print(f"  {cls:>12s}", end="")
            for j in range(len(classes)):
                val = confusion[i, j]
                if i == j:
                    # Diagonal - correct predictions
                    if val > 0:
                        print(f" {C.GREEN}{val:5d}{C.RESET}", end="")
                    else:
                        print(f" {C.RED}{val:5d}{C.RESET}", end="")
                elif val > 0:
                    print(f" {C.RED}{val:5d}{C.RESET}", end="")
                else:
                    print(f" {C.DIM}{val:5d}{C.RESET}", end="")

            acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
            print(f"  {colored_acc(acc)}  {class_total[i]:4d}")

        # Top confusion pairs
        pairs = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and confusion[i, j] > 0:
                    pairs.append((classes[i], classes[j], confusion[i, j]))
        pairs.sort(key=lambda x: -x[2])

        if pairs:
            print(f"\n  {C.CYAN}Top-5 Confusion Pairs:{C.RESET}")
            for true_c, pred_c, count in pairs[:5]:
                pct = count / class_total[c2i[true_c]] * 100 if class_total[c2i[true_c]] > 0 else 0
                print(f"    {true_c:>12s} -> {pred_c:<12s}  {count:3d} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare KSL v17 results against v16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_v17.py
  python evaluate_v17.py --v17 data/results/v17_both_20260208_200000.json
  python evaluate_v17.py --confusion --val-dir data/val_v2
""",
    )
    parser.add_argument(
        "--v16", type=str,
        default="/scratch/alpine/hama5612/ksl-dir-2/data/results/v16_both_20260208_125613.json",
        help="Path to v16 results JSON",
    )
    parser.add_argument(
        "--v17", type=str, default=None,
        help="Path to v17 results JSON (auto-detected if not specified)",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/scratch/alpine/hama5612/ksl-dir-2/data/results",
        help="Directory to search for results files",
    )
    parser.add_argument(
        "--confusion", action="store_true",
        help="Run confusion matrix analysis (loads checkpoint + val data)",
    )
    parser.add_argument(
        "--val-dir", type=str,
        default="/scratch/alpine/hama5612/ksl-dir-2/data/val_v2",
        help="Validation data directory",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors",
    )
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(C):
            if not attr.startswith("_"):
                setattr(C, attr, "")

    # Load v16 results
    if not os.path.exists(args.v16):
        print(f"{C.RED}ERROR: v16 results not found at {args.v16}{C.RESET}")
        sys.exit(1)
    v16 = load_results(args.v16)

    # Find v17 results
    v17_path = args.v17
    if v17_path is None:
        v17_path = find_latest_results(args.results_dir, "v17_")
    if v17_path is None or not os.path.exists(v17_path):
        print(f"{C.YELLOW}v17 results not found yet.{C.RESET}")
        print(f"Searched in: {args.results_dir}")
        print(f"Run training first, then re-run this script.")
        print(f"\nShowing v16 baseline for reference:\n")

        # Show v16 summary even without v17
        print_header("v16 Baseline (Numbers)")
        v16_num = extract_per_class(v16, "numbers")
        for cls in NUMBER_CLASSES:
            acc = v16_num.get(cls, 0.0)
            print(f"  {cls:>12s}  {colored_acc(acc)}")
        print(f"\n  Overall: {colored_acc(extract_overall(v16, 'numbers'))}")

        print_header("v16 Baseline (Words)")
        v16_word = extract_per_class(v16, "words")
        for cls in WORD_CLASSES:
            acc = v16_word.get(cls, 0.0)
            print(f"  {cls:>12s}  {colored_acc(acc)}")
        print(f"\n  Overall: {colored_acc(extract_overall(v16, 'words'))}")
        print(f"\n  Combined: {colored_acc(compute_combined(v16))}")
        sys.exit(0)

    v17 = load_results(v17_path)

    # Print banner
    print(f"\n{C.BOLD}{C.CYAN}  KSL Model Comparison: v16 vs v17{C.RESET}")
    print(f"  v16: {args.v16}")
    print(f"  v17: {v17_path}")
    v16_ts = v16.get("timestamp", "?")
    v17_ts = v17.get("timestamp", "?")
    print(f"  v16 timestamp: {v16_ts}")
    print(f"  v17 timestamp: {v17_ts}")

    # Model info
    print_model_info(v16, v17)

    # Overall comparison
    print_overall_comparison(v16, v17)

    # Per-class Numbers
    print_header("Per-Class Comparison: Numbers")
    v16_num = extract_per_class(v16, "numbers")
    v17_num = extract_per_class(v17, "numbers")
    print_class_comparison(NUMBER_CLASSES, v16_num, v17_num, "Numbers", HARD_NUMBER_CLASSES)

    # Per-class Words
    print_header("Per-Class Comparison: Words")
    v16_word = extract_per_class(v16, "words")
    v17_word = extract_per_class(v17, "words")
    print_class_comparison(WORD_CLASSES, v16_word, v17_word, "Words")

    # Highlights
    print_highlights(v16, v17)

    # Config summary
    print_config_summary(v17)

    # Confusion matrix
    if args.confusion:
        run_confusion_analysis(v17_path, args.val_dir)

    print()


if __name__ == "__main__":
    main()
