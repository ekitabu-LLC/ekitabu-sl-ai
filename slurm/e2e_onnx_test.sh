#!/bin/bash
#SBATCH --job-name=e2e_onnx
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/e2e_onnx_%j.out
#SBATCH --error=slurm/e2e_onnx_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python - <<'PYEOF'
import os, sys, tempfile, shutil
import numpy as np
import onnxruntime as ort
import torch, torch.nn.functional as F

sys.path.insert(0, "/scratch/alpine/hama5612/ksl-dir-2")
from preprocess_for_onnx import (
    extract_landmarks_from_video,
    preprocess_multistream,
    preprocess_v27,
    preprocess_openhands,
)

# ---------------------------------------------------------------------------
# Test videos: (path, true_class, category)
# ---------------------------------------------------------------------------
RT = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
TESTS = [
    (f"{RT}/1. Signer/Numbers/No. 35.mov",     "35",        "numbers"),
    (f"{RT}/1. Signer/Numbers/No. 100.mov",    "100",       "numbers"),
    (f"{RT}/2. Signer/Numbers/22.mov",         "22",        "numbers"),
    (f"{RT}/1. Signer/Words and numbers- Front/Apple.mov",  "Apple",  "words"),
    (f"{RT}/2. Signer/Words and numbers- Front/Monday.mov", "Monday", "words"),
]

NUMBER_CLASSES = sorted([
    "9","17","22","35","48","54","66","73","89","91",
    "100","125","268","388","444",
])
WORD_CLASSES = sorted([
    "Agreement","Apple","Colour","Friend","Gift","Market",
    "Monday","Picture","Proud","Sweater","Teach","Tomatoes",
    "Tortoise","Twin","Ugali",
])

ONNX_BASE = "/scratch/alpine/hama5612/ksl-dir-2/onnx_models"

# ---------------------------------------------------------------------------
# ONNX session helpers
# ---------------------------------------------------------------------------
def make_sess(path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)

def infer_multistream(model_name, category, streams, aux):
    """Fuse 3-stream logits with saved fusion weights."""
    import json
    fw_path = f"{ONNX_BASE}/{category}/{model_name}/fusion_weights.json"
    with open(fw_path) as f:
        data = json.load(f)
    fw = data["weights"] if "weights" in data else data

    probs = {}
    for s in ["joint", "bone", "velocity"]:
        onnx_path = f"{ONNX_BASE}/{category}/{model_name}/{s}/model_adapted.onnx"
        if not os.path.exists(onnx_path):
            onnx_path = f"{ONNX_BASE}/{category}/{model_name}/{s}/model.onnx"
        sess = make_sess(onnx_path)
        gcn_np = streams[s]           # (1,3,90,48)
        logits = sess.run(["logits"], {"gcn": gcn_np, "aux": aux})[0]
        probs[s] = F.softmax(torch.tensor(logits), dim=1).squeeze(0)

    return sum(fw[s] * probs[s] for s in probs)

def infer_v27(model_name, category, gcn9, aux):
    onnx_path = f"{ONNX_BASE}/{category}/{model_name}/model_adapted.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = f"{ONNX_BASE}/{category}/{model_name}/model.onnx"
    sess = make_sess(onnx_path)
    logits = sess.run(["logits"], {"gcn": gcn9, "aux": aux})[0]
    return F.softmax(torch.tensor(logits), dim=1).squeeze(0)

def infer_openhands(model_name, category, x_oh):
    onnx_path = f"{ONNX_BASE}/{category}/{model_name}/model_adapted.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = f"{ONNX_BASE}/{category}/{model_name}/model.onnx"
    sess = make_sess(onnx_path)
    logits = sess.run(["logits"], {"x": x_oh})[0]
    return F.softmax(torch.tensor(logits), dim=1).squeeze(0)

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
all_correct = 0
all_total   = 0

for video_path, true_class, category in TESTS:
    classes = NUMBER_CLASSES if category == "numbers" else WORD_CLASSES
    true_idx = classes.index(true_class)

    print(f"\n{'='*65}")
    print(f"Video   : {os.path.basename(video_path)}")
    print(f"True    : {true_class}  (idx={true_idx}, category={category})")
    print(f"{'='*65}")

    if not os.path.exists(video_path):
        print(f"  SKIP: file not found")
        continue

    # --- Step 1: MediaPipe ---
    print("[1/3] MediaPipe extraction...")
    raw = extract_landmarks_from_video(video_path, verbose=True)
    if raw is None:
        print("  FAILED")
        continue

    # --- Step 2: Preprocessing ---
    print("[2/3] Preprocessing...")
    joint, bone, velocity, aux = preprocess_multistream(raw)
    gcn9, _                    = preprocess_v27(raw)
    x_oh                       = preprocess_openhands(raw)
    streams = {"joint": joint, "bone": bone, "velocity": velocity}
    print(f"  joint {joint.shape}  gcn9 {gcn9.shape}  openhands {x_oh.shape}")

    # --- Step 3: ONNX inference ---
    print("[3/3] ONNX inference...")

    results = {}

    # GroupNorm models (static ONNX)
    for m in ["v31_exp1", "v41"]:
        onnx_dir = f"{ONNX_BASE}/{category}/{m}"
        if os.path.isdir(onnx_dir):
            p = infer_multistream(m, category, streams, aux)
            results[m] = (classes[p.argmax()], float(p.max()))

    # BatchNorm multistream models (adapted ONNX)
    for m in ["v28", "v31_exp5", "v43"]:
        onnx_dir = f"{ONNX_BASE}/{category}/{m}"
        if os.path.isdir(onnx_dir):
            p = infer_multistream(m, category, streams, aux)
            results[m] = (classes[p.argmax()], float(p.max()))

    # v27 / v29 single-model
    for m in ["v27", "v29"]:
        onnx_dir = f"{ONNX_BASE}/{category}/{m}"
        if os.path.isdir(onnx_dir):
            p = infer_v27(m, category, gcn9, aux)
            results[m] = (classes[p.argmax()], float(p.max()))

    # OpenHands
    m = "openhands"
    onnx_dir = f"{ONNX_BASE}/{category}/{m}"
    if os.path.isdir(onnx_dir):
        p = infer_openhands(m, category, x_oh)
        results[m] = (classes[p.argmax()], float(p.max()))

    # --- Ensemble (uniform average of all available probs) ---
    all_probs = []
    for m, (pred, conf) in results.items():
        # Re-run to get full prob vector for ensemble
        pass
    # Build ensemble from scratch
    prob_list = []
    for m in ["v27","v28","v29","v31_exp1","v31_exp5","openhands"]:
        onnx_dir = f"{ONNX_BASE}/{category}/{m}"
        if not os.path.isdir(onnx_dir):
            continue
        if m in ["v27","v29"]:
            p = infer_v27(m, category, gcn9, aux)
        elif m == "openhands":
            p = infer_openhands(m, category, x_oh)
        else:
            p = infer_multistream(m, category, streams, aux)
        prob_list.append(p)
    if prob_list:
        ens_prob = sum(prob_list) / len(prob_list)
        results["ensemble_6_uniform"] = (classes[ens_prob.argmax()], float(ens_prob.max()))

    # --- Print results ---
    print(f"\n  {'Model':<22} {'Pred':<12} {'Conf':>6}  {'':>4}")
    print(f"  {'-'*22} {'-'*12} {'-'*6}  {'-'*4}")
    for m, (pred, conf) in results.items():
        ok = "OK" if pred == true_class else "WRONG"
        bold = "**" if m == "ensemble_6_uniform" else "  "
        print(f"  {bold}{m:<20}{bold} {pred:<12} {conf:>6.1%}  {ok}")

    ens_pred = results.get("ensemble_6_uniform", (None,))[0]
    correct = ens_pred == true_class
    all_correct += int(correct)
    all_total   += 1

print(f"\n{'='*65}")
print(f"OVERALL: {all_correct}/{all_total} correct from ensemble_6_uniform")
print(f"{'='*65}")
PYEOF

echo "Job finished: $(date)"
