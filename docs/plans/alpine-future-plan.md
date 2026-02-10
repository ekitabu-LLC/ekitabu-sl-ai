# KSL Project Future Plan: Alpine HPC Migration & Model Development

> **Date:** 2026-02-07
> **Author:** Auto-generated comprehensive plan
> **Project:** Kenya Sign Language (KSL) Recognition System
> **Target:** >85% accuracy, no class below 50%, full Alpine HPC workflow

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Phase 1: Alpine HPC Migration (HIGHEST PRIORITY)](#phase-1-alpine-hpc-migration)
4. [Phase 2: Evaluate Existing Models on Alpine](#phase-2-evaluate-existing-models-on-alpine)
5. [Phase 3: Model Improvements](#phase-3-model-improvements)
6. [Phase 4: Code Quality & Infrastructure](#phase-4-code-quality--infrastructure)
7. [Phase 5: Deployment & Academic](#phase-5-deployment--academic)
8. [Appendix: Alpine HPC Reference](#appendix-alpine-hpc-reference)

---

## Executive Summary

The KSL sign language recognition project has reached a critical juncture. After 14 model
versions trained on Modal cloud GPUs, the best documented accuracy is 83.65% (v8), with
models v10-v14 built but lacking performance documentation. The project must now:

1. **Fully migrate** from Modal to CU Boulder Alpine HPC (SLURM-based)
2. **Evaluate** all existing models (v10-v14) to establish baselines
3. **Improve** model accuracy beyond 85% with no class at 0%
4. **Refactor** duplicated code and add proper engineering practices
5. **Deploy** for academic use and paper preparation

**Key constraint:** All future compute (training, evaluation, inference) runs on Alpine HPC
via SLURM jobs. Modal is permanently retired.

**Key accelerator:** An older project at `/projects/hama5612/ksl-alpha/alpine/` already has
working SLURM scripts, a Modal-free training script, an install script, and a setup guide
that can be directly adapted for this project's ST-GCN models.

---

## Current State Assessment

### Model Performance

| Version | Architecture | Overall Accuracy | 444 Accuracy | Status |
|---------|-------------|-----------------|--------------|--------|
| v7 | Temporal Pyramid | ~80% | 0% | Documented |
| v8 | Temporal Pyramid + Confusion Penalties | 83.65% | 0% | **Best documented** |
| v9 | Temporal Segmentation | TBD | TBD | Checkpoint on Modal volume |
| v10 | ST-GCN (42 nodes) | TBD | TBD | Checkpoint on Modal volume |
| v11 | ST-GCN variant | TBD | TBD | Checkpoint on Modal volume |
| v12 | ST-GCN variant | TBD | TBD | Checkpoint on Modal volume |
| v13 | ST-GCN + Sparse Dropout (48 nodes) | TBD | TBD | Checkpoint on Modal volume |
| v14 | ST-GCN + Extreme Dropout (48 nodes) | TBD | TBD | Checkpoint on Modal volume |

### Critical Bugs

- **Class 444:** 0% accuracy (0.999 cosine similarity with class 54)
- **Confusion pairs:** 89->9, 66->17, 91->73, 73->17
- **Text bug:** Frontend says "Kuwaiti Sign Language" -- should be "Kenya Sign Language"

### Dataset

| Split | Samples | Signers | Location |
|-------|---------|---------|----------|
| Train (raw video) | 750 | 5 (signers 1-5) | `/scratch/alpine/hama5612/ksl-dir-2/dataset/` (2.8 GB) |
| Val (raw video) | 146 videos | 5 (signers 6-10) | `/scratch/alpine/hama5612/ksl-dir-2/validation-data/` (3.1 GB) |
| Train (preprocessed .npy) | 750 | 5 | `/scratch/alpine/hama5612/ksl-dir-2/modal_backup/train_v2/` |
| Val (preprocessed .npy) | ~728 | 5 | `/scratch/alpine/hama5612/ksl-dir-2/modal_backup/val_v2/` |

**Total preprocessed data:** ~403 MB in `modal_backup/` -- already on Alpine scratch, no transfer needed.

### Infrastructure

- **50 Modal-dependent files** (124 `modal` import/usage patterns across the codebase):
  - 14 training scripts (`train_ksl_v2.py` - `train_ksl_v14.py`)
  - 7 evaluation scripts, 7 test scripts
  - 6 analysis scripts, 5 feature extraction scripts
  - 3 ensemble scripts, 3 utility scripts
- **Frontend:** React/Vite/TypeScript -- COMPLETE in `frontend/`
- **Backend:** FastAPI -- functional but needs real model checkpoints
- **No unit tests, no CI/CD, no experiment tracking**
- **All code is inline/monolithic** per Modal script -- needs refactoring into shared modules

### Reusable ksl-alpha Infrastructure

The older project at `/projects/hama5612/ksl-alpha/alpine/` provides directly reusable patterns:

| File | What It Does | Reuse Strategy |
|------|-------------|----------------|
| `alpine/SETUP_GUIDE.md` | 5-phase Alpine setup guide | Adapt for ksl-dir-2 |
| `alpine/install.sh` | Automated env installer | Adapt for ST-GCN deps |
| `alpine/train_alpine.py` | Modal-free training (argparse, no decorators) | Use as template for ST-GCN |
| `alpine/slurm/test_training.sh` | Working SLURM script | Copy and modify paths |
| `alpine/slurm/test_training_updated.sh` | Updated SLURM with `python/3.10.2` + `cuda/11.8.0` | **Use this version** |
| `alpine/requirements.txt` | Alpine Python deps | Extend for ST-GCN |

### Alpine HPC Resources (Verified Live)

| Resource | Details |
|----------|---------|
| **GPU Partitions** | `aa100` (A100 40GB, 11 nodes), `al40` (L40 48GB, 3 nodes), `ami100` (MI100 32GB, 8 nodes -- **AMD ROCm, avoid for PyTorch**), `gh200` (GH200, 2 nodes, 7-day limit) |
| **Testing Partitions** | `atesting_a100` (1 node, 6x A100 MIG 20GB, 1-hour limit), `atesting_mi100` (8 nodes, 1-hour limit) |
| **CPU Partitions** | `amilan` (387 nodes, 24h), `amem` (24 nodes, 7-day limit) |
| **Home storage** | `/home/hama5612/` -- **1.3G/2.0G -- NEARLY FULL, do NOT install here** |
| **Projects storage** | `/projects/hama5612/` -- 18G/250G (code, conda envs, persistent backups) |
| **Scratch storage** | `/scratch/alpine/hama5612/` -- 194G/9.5TB (data, checkpoints, logs) |
| **SLURM accounts** | `ucb-general`, `ucb765_asc1` -- both access all partitions |
| **Conda envs** | `ksl` and `ksl_train` at `/projects/hama5612/software/anaconda/envs/` |
| **Module system** | **Hierarchical** -- load `python/3.10.2` first, then `cuda/11.8.0` becomes visible |

**IMPORTANT storage warnings:**
- `/home/` is at 65% capacity (1.3G/2.0G). Never install packages or store data here.
- `/scratch/` has a 90-day purge policy. Back up critical results to `/projects/`.
- `ami100` partition uses AMD MI100 GPUs requiring ROCm, NOT CUDA. Avoid for PyTorch training.

---

## Phase 1: Alpine HPC Migration (HIGHEST PRIORITY)

**Goal:** Convert all training and evaluation from Modal cloud GPU to Alpine HPC SLURM jobs.
**Estimated time:** 1-2 weeks
**Estimated GPU hours:** ~5 hours (testing and debugging)
**SU cost:** ~30 SUs (A100 = 6 SUs/hour)

### 1.1 Data Organization on Alpine

The preprocessed `.npy` data from Modal is already on scratch. Set up the canonical
data layout:

```
/scratch/alpine/hama5612/ksl-dir-2/data/
  train_v2/          <-- symlink to modal_backup/train_v2/
    100/
    125/
    ...
    444/
      444-1-1.npy
      444-1-2.npy
      ...
  val_v2/            <-- symlink to modal_backup/val_v2/
    100/
    ...
  checkpoints/       <-- model checkpoints (output)
    v10_numbers/
    v10_words/
    v13_numbers/
    ...
  results/           <-- experiment logs, metrics, TensorBoard runs
```

**Action items:**

```bash
# Create canonical data directory with symlinks (no data copy needed)
cd /scratch/alpine/hama5612/ksl-dir-2
mkdir -p data/checkpoints data/results
ln -sf /scratch/alpine/hama5612/ksl-dir-2/modal_backup/train_v2 data/train_v2
ln -sf /scratch/alpine/hama5612/ksl-dir-2/modal_backup/val_v2 data/val_v2

# Create persistent backup location
mkdir -p /projects/hama5612/ksl-models
```

**Storage strategy:**
- `/scratch/alpine/` -- Training data, checkpoints, experiment logs (fast I/O, large, 90-day purge)
- `/projects/hama5612/` -- Final models, code, conda envs, important results (persistent, 250 GB)
- Back up best checkpoints: `cp data/checkpoints/best_*.pt /projects/hama5612/ksl-models/`

### 1.2 Environment Setup

Two conda environments already exist. Verify the training environment has GPU support.

**Module loading (hierarchical -- order matters):**

```bash
# CORRECT order -- Python first, then CUDA becomes visible
module purge
module load python/3.10.2
module load cuda/11.8.0
```

**Verify existing conda environment:**

```bash
module load python/3.10.2
module load cuda/11.8.0
conda activate ksl_train

# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# If CUDA is False, reinstall PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ensure all deps present
pip install numpy>=1.24.0 scikit-learn matplotlib seaborn tqdm pyyaml tensorboard
```

**Required packages for training:**
- `torch >= 2.0.0` (with CUDA 11.8 support)
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `tqdm >= 4.65.0`
- `pyyaml >= 6.0`
- `tensorboard >= 2.14.0` (experiment tracking)

### 1.3 Template SLURM Scripts

These are adapted from the proven ksl-alpha SLURM scripts at
`/projects/hama5612/ksl-alpha/alpine/slurm/test_training_updated.sh`.

#### Template: GPU Training Job (A100)

Create file: `slurm/train_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=ksl-train
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.err
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# KSL ST-GCN Training on Alpine HPC
# Adapted from: /projects/hama5612/ksl-alpha/alpine/slurm/test_training_updated.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL Training - Alpine HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# ---- Module Loading (hierarchical: python first, then cuda) ----
module purge
module load python/3.10.2
module load cuda/11.8.0

# ---- Activate Conda Environment ----
conda activate ksl_train

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ---- Paths ----
PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
DATA_DIR="${PROJECT_DIR}/data"
TRAIN_DIR="${DATA_DIR}/train_v2"
VAL_DIR="${DATA_DIR}/val_v2"
CKPT_DIR="${DATA_DIR}/checkpoints"
RESULTS_DIR="${DATA_DIR}/results"

mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}"

# ---- GPU Info ----
echo "GPU Information:"
nvidia-smi
echo ""

# ---- Set TMPDIR to scratch (not /tmp which is small) ----
export TMPDIR=/scratch/alpine/$USER/tmp
mkdir -p $TMPDIR

# ---- Run Training ----
echo "Starting training version: ${VERSION:-v15}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

python scripts/train.py \
    --version ${VERSION:-v15} \
    --train-dir ${TRAIN_DIR} \
    --val-dir ${VAL_DIR} \
    --checkpoint-dir ${CKPT_DIR} \
    --results-dir ${RESULTS_DIR} \
    --epochs ${EPOCHS:-200} \
    --batch-size ${BATCH_SIZE:-32} \
    --learning-rate ${LR:-1e-3} \
    --patience ${PATIENCE:-40}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training successful! Checkpoint at: $CKPT_DIR"
else
    echo "Training failed! Check logs."
fi

exit $EXIT_CODE
```

**Submit with:**
```bash
sbatch slurm/train_gpu.sh                           # defaults: v15, lr=1e-3
sbatch --export=VERSION=v15,LR=5e-4 slurm/train_gpu.sh  # custom
```

#### Template: Evaluation Job (testing partition, 1-hour limit)

Create file: `slurm/evaluate_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=ksl-eval
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/eval_%x_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/eval_%x_%j.err
#SBATCH --account=ucb-general

set -e

module purge
module load python/3.10.2
module load cuda/11.8.0
conda activate ksl_train

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR=/scratch/alpine/$USER/tmp
mkdir -p $TMPDIR

echo "Evaluating version: ${VERSION:-v14}"
nvidia-smi

cd ${PROJECT_DIR}
python scripts/evaluate.py \
    --version ${VERSION:-v14} \
    --checkpoint-dir ${PROJECT_DIR}/data/checkpoints \
    --val-dir ${PROJECT_DIR}/data/val_v2 \
    --output-dir ${PROJECT_DIR}/data/results

echo "Evaluation completed at $(date)"
```

**Submit with:**
```bash
sbatch --export=VERSION=v14 slurm/evaluate_gpu.sh
```

#### Template: Feature Extraction (CPU-only, amilan)

Create file: `slurm/extract_features.sh`

```bash
#!/bin/bash
#SBATCH --job-name=ksl-extract
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/extract_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/extract_%j.err
#SBATCH --account=ucb-general

set -e

module purge
module load python/3.10.2
conda activate ksl_train

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR=/scratch/alpine/$USER/tmp
mkdir -p $TMPDIR

cd ${PROJECT_DIR}
python scripts/extract_features.py \
    --input-dir ${PROJECT_DIR}/dataset \
    --output-dir ${PROJECT_DIR}/data/train_v3 \
    --workers 16

echo "Feature extraction completed at $(date)"
```

#### Template: Hyperparameter Sweep (SLURM Job Array)

Create file: `slurm/sweep_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=ksl-sweep
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/sweep_%A_%a.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/sweep_%A_%a.err
#SBATCH --account=ucb-general
#SBATCH --array=0-5

set -e

module purge
module load python/3.10.2
module load cuda/11.8.0
conda activate ksl_train

# Define hyperparameter grid
LRS=(1e-3 5e-4 2e-4 1e-3 5e-4 2e-4)
DROPOUTS=(0.3 0.3 0.3 0.5 0.5 0.5)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
DROPOUT=${DROPOUTS[$SLURM_ARRAY_TASK_ID]}

echo "Sweep task $SLURM_ARRAY_TASK_ID: lr=$LR, dropout=$DROPOUT"

PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR=/scratch/alpine/$USER/tmp
mkdir -p $TMPDIR

cd ${PROJECT_DIR}
python scripts/train.py \
    --version "v15-sweep${SLURM_ARRAY_TASK_ID}" \
    --train-dir ${PROJECT_DIR}/data/train_v2 \
    --val-dir ${PROJECT_DIR}/data/val_v2 \
    --checkpoint-dir ${PROJECT_DIR}/data/checkpoints \
    --results-dir ${PROJECT_DIR}/data/results \
    --learning-rate $LR \
    --dropout $DROPOUT \
    --epochs 200 \
    --patience 40
```

**Submit with:**
```bash
sbatch slurm/sweep_gpu.sh   # Submits 6 jobs in parallel
```

### 1.4 Convert Training Scripts from Modal to Alpine

**Current Modal pattern (all 50 files follow this):**
```python
import modal
app = modal.App("ksl-trainer-v14")
volume = modal.Volume.from_name("ksl-dataset-vol")
image = modal.Image.debian_slim(python_version="3.11").pip_install(...)

@app.function(gpu="T4", volumes={"/data": volume}, timeout=21600, image=image)
def train_extreme_dropout():
    # ALL code is inside this function (imports, classes, training loop)
    import torch
    import numpy as np
    ...  # 300+ lines of inline code
    volume.commit()

@app.local_entrypoint()
def main():
    r = train_extreme_dropout.remote()
```

**New Alpine pattern (adapted from ksl-alpha's `train_alpine.py`):**
```python
#!/usr/bin/env python3
"""
KSL Training v15 - Alpine HPC Version (ST-GCN)

Usage:
    # Interactive (compile node):
    sinteractive --partition=atesting_a100 --gres=gpu:a100_3g.20gb:1 --time=00:30:00
    python scripts/train.py --version v15 --test-mode

    # SLURM job:
    sbatch --export=VERSION=v15 slurm/train_gpu.sh
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Import shared modules (Phase 4 refactoring target)
# from src.ksl.models.stgcn import STGCNModel
# from src.ksl.data.dataset import KSLGraphDataset
# from src.ksl.data.graph import build_adjacency

def parse_args():
    parser = argparse.ArgumentParser(description="KSL ST-GCN Training on Alpine HPC")
    parser.add_argument("--version", type=str, default="v15")
    parser.add_argument("--train-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/train_v2")
    parser.add_argument("--val-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/val_v2")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints")
    parser.add_argument("--results-dir", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick test: 1 epoch, 10% data")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"KSL Training {args.version} - Alpine HPC")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ... ST-GCN model definition (moved out of Modal decorator) ...
    # ... Dataset class (moved out of Modal decorator) ...
    # ... Training loop (same logic as v14, no Modal calls) ...
    # ... Save checkpoints with torch.save() (no volume.commit()) ...

    # Save results as JSON
    results_path = Path(args.results_dir) / f"results_{args.version}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
```

**Key conversion steps for each of the 50 Modal scripts:**
1. Remove `import modal`, `app = modal.App(...)`, `volume = ...`, `image = ...`
2. Remove `@app.function(...)` decorator and `@app.local_entrypoint()`
3. Move all code out of the decorated function to top-level or `main()`
4. Replace hardcoded `/data/` paths with command-line arguments
5. Replace `volume.commit()` with standard file I/O
6. Add `argparse` for configurable paths (follow ksl-alpha's `train_alpine.py` pattern)
7. Add `if __name__ == "__main__": main()`
8. Move delayed imports (`import torch` inside function) to top of file

### 1.5 Download Checkpoints from Modal (One-Time, URGENT)

Before decommissioning Modal, download all existing checkpoints. This is the most
time-sensitive step -- Modal data could become inaccessible.

```bash
# Run this ONCE from a machine with Modal CLI access
# (personal laptop or a machine where `modal` is installed)

# Download all checkpoint directories
for v in v8 v9 v10 v11 v12 v13 v14; do
    echo "Downloading $v checkpoints..."
    modal volume get ksl-dataset-vol /checkpoints_${v}_numbers/ ./modal_checkpoints/${v}_numbers/ 2>/dev/null || true
    modal volume get ksl-dataset-vol /checkpoints_${v}_words/ ./modal_checkpoints/${v}_words/ 2>/dev/null || true
    modal volume get ksl-dataset-vol /checkpoints_${v}/ ./modal_checkpoints/${v}/ 2>/dev/null || true
done

# Download any result JSON files
modal volume get ksl-dataset-vol /checkpoints_v13_results.json ./modal_checkpoints/ 2>/dev/null || true
modal volume get ksl-dataset-vol /checkpoints_v14_results.json ./modal_checkpoints/ 2>/dev/null || true

# Transfer to Alpine scratch
rsync -avz --progress ./modal_checkpoints/ \
    hama5612@login.rc.colorado.edu:/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/
```

### 1.6 Remove Modal Dependencies

After all checkpoints are safely on Alpine:

1. **Archive old Modal scripts** (preserve for reference):
   ```bash
   cd /scratch/alpine/hama5612/ksl-dir-2
   mkdir -p archive/modal_scripts
   mv train_ksl_v*.py archive/modal_scripts/
   mv analyze_*.py archive/modal_scripts/
   mv assess_*.py archive/modal_scripts/
   mv evaluate_*.py archive/modal_scripts/
   mv test_v*.py archive/modal_scripts/
   mv ensemble_*.py archive/modal_scripts/
   mv extract*.py archive/modal_scripts/
   mv download_modal_data.py archive/modal_scripts/
   ```
2. Remove `modal` from `requirements.txt` (root level)
3. `frontend/backend/requirements.txt` -- already clean (no Modal)
4. Update `.planning/PROJECT.md` -- change "Training: Modal (cloud GPU - A10G)" to "Training: Alpine HPC (NVIDIA A100)"

### 1.7 Phase 1 Checklist

- [ ] Create `data/` directory structure with symlinks to preprocessed data
- [ ] Verify conda `ksl_train` env has PyTorch + CUDA (module load python/3.10.2 + cuda/11.8.0)
- [ ] Create `slurm/` directory with 4 template job scripts (train, eval, extract, sweep)
- [ ] **URGENT:** Download all v8-v14 checkpoints from Modal volume to Alpine scratch
- [ ] Convert v14 training script to Alpine format as reference implementation
- [ ] Run a test training job on `atesting_a100` partition (1-hour limit, --test-mode)
- [ ] Verify training runs, produces checkpoints, and logs GPU info on Alpine
- [ ] Archive all 50 Modal scripts to `archive/modal_scripts/`
- [ ] Remove `modal` from requirements, update PROJECT.md
- [ ] Copy ksl-alpha's `install.sh` pattern for a one-command setup script

---

## Phase 2: Evaluate Existing Models on Alpine

**Goal:** Run all v10-v14 models to get actual accuracy metrics and establish the true baseline.
**Estimated time:** 1 week
**Estimated GPU hours:** ~2-3 hours (evaluation is fast)
**SU cost:** ~18 SUs

### 2.1 Create Unified Evaluation Script

Create `scripts/evaluate.py` that can load any model version and produce standardized metrics.
Follow the pattern from ksl-alpha's `train_alpine.py` (argparse, JSON output, clean logging).

**Key requirements:**
- Auto-detect model architecture from checkpoint metadata (v8=Temporal Pyramid, v10-v14=ST-GCN)
- Handle both 42-node (v10) and 48-node (v13/v14) graph structures
- Handle separate numbers/words models (v10+) vs unified models (v8/v9)

```python
# scripts/evaluate.py
# Usage: python scripts/evaluate.py --version v14 --checkpoint-dir data/checkpoints --val-dir data/val_v2

# Produces:
# - Overall accuracy
# - Per-class accuracy (all 30 classes)
# - Confusion matrix (saved as .png and .json)
# - Comparison against v8 baseline (83.65%)
# - Specific tracking for class 444
# - Classes below 50% flagged
```

**Output format** (JSON for programmatic access + comparison):
```json
{
    "version": "v14",
    "architecture": "ST-GCN",
    "num_nodes": 48,
    "overall_accuracy": 85.2,
    "numbers_accuracy": 82.1,
    "words_accuracy": 88.3,
    "per_class": {
        "444": 45.0,
        "54": 90.0,
        "9": 85.0,
        ...
    },
    "confusion_pairs": [
        {"true": "444", "pred": "54", "count": 5},
        ...
    ],
    "vs_v8_delta": "+1.55%",
    "classes_below_50": ["444"],
    "timestamp": "2026-02-10T14:30:00",
    "gpu": "NVIDIA A100",
    "eval_time_seconds": 45
}
```

### 2.2 Evaluation Jobs

Submit evaluation for all versions in parallel using the testing partition (fast, 1-hour limit):

```bash
# Evaluate all versions (each takes ~5-10 min)
for v in v10 v11 v12 v13 v14; do
    sbatch --export=VERSION=$v --job-name=ksl-eval-$v slurm/evaluate_gpu.sh
done

# Monitor all at once
watch -n 5 'squeue -u $USER'
```

### 2.3 Results Comparison Table

After running all evaluations, produce `data/results/model_comparison.md`:

| Version | Architecture | Nodes | Overall | 444 | Worst Class | vs v8 | Notes |
|---------|-------------|-------|---------|-----|-------------|-------|-------|
| v8 | Temporal Pyramid | N/A | 83.65% | 0% | 444 (0%) | baseline | Best documented |
| v10 | ST-GCN | 42 | ? | ? | ? | ? | First GNN attempt |
| v11 | ST-GCN | ? | ? | ? | ? | ? | |
| v12 | ST-GCN | ? | ? | ? | ? | ? | |
| v13 | ST-GCN + dropout | 48 | ? | ? | ? | ? | 30-70% hand dropout |
| v14 | ST-GCN + extreme dropout | 48 | ? | ? | ? | ? | 50-90% hand dropout |

### 2.4 Decision Gate

Based on evaluation results:
- **If any v10-v14 model > 85%:** Focus on that architecture for improvements
- **If v10-v14 models are 80-85%:** Hybrid/ensemble approach with v8
- **If v10-v14 models < 80%:** Re-examine ST-GCN implementation, consider Graph Transformer

### 2.5 Phase 2 Checklist

- [ ] All v8-v14 checkpoints present in `data/checkpoints/`
- [ ] Create unified `scripts/evaluate.py` with JSON output
- [ ] Run evaluation for v10 through v14 on `atesting_a100`
- [ ] Document results in `data/results/model_comparison.json`
- [ ] Identify best performing model
- [ ] Write comparison report with decision on Phase 3 direction
- [ ] Generate confusion matrix heatmaps for top 3 models

---

## Phase 3: Model Improvements

**Goal:** Achieve >85% overall accuracy with no class below 50%.
**Estimated time:** 3-5 weeks
**Estimated GPU hours:** ~50-100 hours
**SU cost:** ~300-600 SUs

### 3.1 Fix 444 vs 54 Confusion (CRITICAL)

**Root cause:** 0.999 cosine similarity between 444 and 54 in raw features.
**Key insight from analysis:** 444 has THREE "4" movements; 54 has ONE "5-4" sequence.

**Approach A: Temporal Repetition Detection (try first)**
- Segment sequences into thirds
- Compute inter-segment cosine similarity
- 444: HIGH similarity across thirds (same gesture repeated 3x)
- 54: LOW similarity (different gestures "5" then "4")
- Add segment similarity as additional feature channels to ST-GCN input

**Approach B: Gesture Peak Counting**
- Count distinct gesture peaks (hand spread local maxima)
- 444 should have ~3 peaks; 54 should have ~2
- Add peak count as auxiliary input

**Approach C: Velocity Dip Analysis**
- 444 has "reset" movements between digit repetitions (hand returns to neutral)
- Detect velocity dips as transition markers
- Add velocity profile features

**Approach D: Two-Stage Classification**
```python
class TwoStageSTGCN(nn.Module):
    def __init__(self, num_classes, adj):
        super().__init__()
        self.backbone = STGCNBackbone(adj)  # shared feature extractor

        # Stage 1: Is this a multi-digit number? (444, 100)
        self.multi_digit_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Stage 2: Class-specific heads
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        is_multi_digit = self.multi_digit_head(features)
        logits = self.classifier(features)
        return logits, is_multi_digit
```

**Implementation order:** A -> evaluate -> if insufficient, add B+C -> if still insufficient, try D

### 3.2 Address Other Confusion Pairs

| Confusion | Count (v8) | Root Cause | Proposed Fix |
|-----------|------------|------------|-------------|
| 89 -> 9, Teach, 73 | 11 errors | Similar hand shapes | Temporal order features: "8" then "9" vs just "9" |
| 66 -> 17 | 6 errors | Both use similar finger counts | Hand position trajectory: "6-6" vs "1-7" |
| 73 -> 17, 89 | 6 errors | Overlapping finger configs | Wrist movement features |
| 91 -> 73 | 5 errors | Similar two-digit pattern | Duration + sequence features |

### 3.3 Training Strategy on Alpine

**Primary partition:** `aa100` (NVIDIA A100 40GB) -- use `--gres=gpu:1`

```bash
# Single A100 is vastly more powerful than Modal's T4/A10G
# Model has ~500K params, 750 training samples -- A100 is overkill but fast
# Expected training time: ~30-60 min per run (vs 2-3 hours on Modal T4)
```

**Hyperparameter sweep using SLURM job arrays:**

```bash
# Submit 6 experiments in parallel
sbatch slurm/sweep_gpu.sh

# Monitor
squeue -u hama5612
# JOBID  PARTITION  NAME        USER     ST  TIME  NODES
# 12345  aa100      ksl-sweep   hama5612 R   0:10  1
# 12346  aa100      ksl-sweep   hama5612 R   0:10  1
# ...
```

**SU budget planning:**
- Each training run: ~1 hour on A100 = 6 SUs
- Hyperparameter sweep (6 runs): 36 SUs
- Iterative improvements (10 runs): 60 SUs
- Evaluation (20 runs): 10 SUs (testing partition, cheaper)
- Total Phase 3: ~100-200 SUs

### 3.4 Ensemble Approaches

If no single model exceeds 85%, consider ensembling:

1. **Temporal Pyramid (v8) + ST-GCN (best of v10-v14)**
   - Complementary architectures capture different features
   - v8 excels at words (88.6% avg); ST-GCN may be better at numbers
   - Weighted voting: `combined = alpha * logits_v8 + (1-alpha) * logits_stgcn`

2. **Numbers specialist + Words specialist**
   - Already training separate models (v13/v14 split numbers vs words)
   - Use best-of-each for final prediction

3. **Ensemble SLURM job:**
   ```bash
   # Run ensemble evaluation
   sbatch --export=MODELS="v8,v14" slurm/evaluate_gpu.sh
   ```

### 3.5 Phase 3 Checklist

- [ ] Implement temporal repetition features (Approach A) for 444
- [ ] Train v15 with enhanced temporal features on Alpine `aa100`
- [ ] Evaluate v15 -- target: 444 > 40%, overall > 85%
- [ ] If insufficient, implement gesture peak + velocity features (B+C)
- [ ] If still insufficient, implement two-stage classifier (D)
- [ ] Address 66->17, 89->73, 91->73 confusions
- [ ] Run hyperparameter sweep (6 configs) via SLURM job array
- [ ] Try ensemble of best temporal pyramid + best ST-GCN
- [ ] Achieve target: >85% overall, no class below 50%
- [ ] Back up best model to `/projects/hama5612/ksl-models/`

---

## Phase 4: Code Quality & Infrastructure

**Goal:** Professional-quality codebase with proper engineering practices.
**Estimated time:** 2-3 weeks (can overlap with Phases 2-3)
**GPU hours:** ~0 (mostly refactoring)

### 4.1 Refactor Duplicated Model Code

Currently: 50 Modal scripts with ~300 lines of duplicated model/dataset code each. The same
ST-GCN model definition is copy-pasted across v10, v11, v12, v13, and v14.

**Target structure (following ksl-alpha's clean separation):**
```
ksl-dir-2/
  src/
    ksl/
      __init__.py
      models/
        __init__.py
        stgcn.py          # ST-GCN model (from v13/v14, 48 nodes)
        temporal_pyramid.py  # Temporal Pyramid model (from v8)
      data/
        __init__.py
        dataset.py         # KSLGraphDataset, KSLTemporalDataset
        augmentation.py    # All augmentation transforms (dropout, noise, etc.)
        graph.py           # build_adjacency(), HAND_EDGES, POSE_EDGES
      training/
        __init__.py
        trainer.py         # Generic training loop with early stopping
        losses.py          # Focal loss, confusion penalties, class weights
        schedulers.py      # Cosine with warmup (from ksl-alpha pattern)
      evaluation/
        __init__.py
        evaluator.py       # Unified evaluation with JSON output
        metrics.py         # Per-class accuracy, confusion matrix, visualization
      config.py            # Centralized config (YAML-based, like ksl-alpha)
      constants.py         # NUMBER_CLASSES, WORD_CLASSES, HAND_EDGES, etc.
  scripts/
    train.py               # Main training entry point (argparse)
    evaluate.py            # Main evaluation entry point
    extract_features.py    # Feature extraction from raw video
    compare_models.py      # Compare multiple model results
  configs/
    training_config.yaml   # Training hyperparameters
    augmentation_config.yaml  # Augmentation settings
    evaluation_config.yaml    # Evaluation settings
  slurm/
    train_gpu.sh           # A100 training
    evaluate_gpu.sh        # Testing partition eval
    extract_features.sh    # CPU feature extraction
    sweep_gpu.sh           # Hyperparameter sweep array
  tests/
    test_dataset.py
    test_model.py
    test_augmentation.py
    test_graph.py
  archive/
    modal_scripts/         # All 50 old Modal scripts (preserved for reference)
  data -> /scratch/.../data/  # symlink to scratch data
```

### 4.2 Experiment Tracking

**Start with TensorBoard** (zero external setup, works on Alpine):

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"data/results/runs/{version}_{timestamp}")
writer.add_scalar("train/loss", loss, epoch)
writer.add_scalar("val/accuracy", val_acc, epoch)
writer.add_scalar("val/444_accuracy", class_acc.get("444", 0), epoch)
writer.add_scalar("val/worst_class", min(class_acc.values()), epoch)
```

View via SSH tunnel:
```bash
# On Alpine compile node:
sinteractive --partition=acompile --time=02:00:00
tensorboard --logdir /scratch/alpine/$USER/ksl-dir-2/data/results/runs/ --port 6006

# On local machine:
ssh -L 6006:localhost:6006 hama5612@login.rc.colorado.edu
# Then open http://localhost:6006
```

**Later upgrade to W&B** if needed (ksl-alpha already supports it optionally):
```python
# W&B is already coded in ksl-alpha's train_alpine.py as optional
if os.environ.get("WANDB_API_KEY"):
    import wandb
    wandb.init(project="ksl-recognition", name=f"{version}")
```

### 4.3 YAML Configuration (from ksl-alpha pattern)

Replace hardcoded CONFIG dictionaries with YAML files:

```yaml
# configs/training_config.yaml
model:
  architecture: stgcn
  num_nodes: 48
  in_channels: 3
  hidden_dim: 128
  num_layers: 6
  temporal_kernel: 9
  dropout: 0.5

training:
  batch_size: 32
  max_epochs: 200
  learning_rate: 1e-3
  min_lr: 1e-6
  weight_decay: 1e-4
  label_smoothing: 0.2
  warmup_epochs: 10
  gradient_accumulation: 1

early_stopping:
  patience: 40
  min_delta: 0.001

data:
  max_frames: 90
  num_workers: 4
  pin_memory: true

augmentation:
  hand_dropout_prob: 0.5
  hand_dropout_rate: 0.5
  noise_std: 0.03
  scale_range: [0.8, 1.2]
  temporal_shift_max: 8

logging:
  tensorboard: true
  wandb_project: ksl-recognition
```

### 4.4 Unit Tests

```python
# tests/test_dataset.py
def test_graph_dataset_shape():
    """Verify dataset returns correct tensor shapes."""
    ds = KSLGraphDataset("data/train_v2", max_frames=90)
    x, y = ds[0]
    assert x.shape == (3, 90, 48)  # (channels, frames, nodes)
    assert isinstance(y, int)

def test_adjacency_matrix():
    """Verify adjacency matrix is symmetric and properly normalized."""
    adj = build_adjacency(48)
    assert adj.shape == (48, 48)
    assert torch.allclose(adj, adj.T, atol=1e-6)
    # Self-loops present
    assert all(adj.diagonal() > 0)

def test_augmentation_preserves_shape():
    """Verify augmentations don't change tensor shape."""
    x = torch.randn(90, 48, 3)
    augmented = apply_augmentation(x, training=True)
    assert augmented.shape == x.shape

# tests/test_model.py
def test_stgcn_forward_pass():
    """Verify ST-GCN accepts input and produces correct output shape."""
    adj = build_adjacency(48)
    model = STGCNModel(num_classes=15, num_nodes=48, adj=adj)
    x = torch.randn(4, 3, 90, 48)  # batch=4
    out = model(x)
    assert out.shape == (4, 15)

def test_stgcn_30_classes():
    """Verify model works for all 30 classes."""
    adj = build_adjacency(48)
    model = STGCNModel(num_classes=30, num_nodes=48, adj=adj)
    x = torch.randn(2, 3, 90, 48)
    out = model(x)
    assert out.shape == (2, 30)
```

Run on compile node:
```bash
sinteractive --partition=acompile --time=01:00:00
module load python/3.10.2
conda activate ksl_train
cd /scratch/alpine/hama5612/ksl-dir-2
python -m pytest tests/ -v
```

### 4.5 Fix Text Bug

In the frontend, change "Kuwaiti Sign Language" to "Kenya Sign Language":

```bash
# Find and replace all occurrences
cd /scratch/alpine/hama5612/ksl-dir-2
grep -r "Kuwaiti" frontend/
# Fix each occurrence
```

### 4.6 Phase 4 Checklist

- [ ] Create `src/ksl/` package with shared model code
- [ ] Extract ST-GCN model (v13/v14) into `src/ksl/models/stgcn.py`
- [ ] Extract dataset class into `src/ksl/data/dataset.py`
- [ ] Extract graph construction into `src/ksl/data/graph.py`
- [ ] Extract augmentation into `src/ksl/data/augmentation.py`
- [ ] Extract constants into `src/ksl/constants.py`
- [ ] Create YAML config files (from ksl-alpha pattern)
- [ ] Create unified `scripts/train.py` entry point with argparse
- [ ] Add TensorBoard logging to training loop
- [ ] Write unit tests for dataset, model, augmentation, graph
- [ ] Fix "Kuwaiti" -> "Kenya" text bug in frontend
- [ ] Archive all 50 Modal scripts to `archive/modal_scripts/`
- [ ] Update `.planning/PROJECT.md` and `.planning/ROADMAP.md`

---

## Phase 5: Deployment & Academic

**Goal:** Deploy inference capability and prepare for academic publication.
**Estimated time:** 2-3 weeks

### 5.1 Deploy Inference on Alpine

**Option A: FastAPI on Alpine via SLURM (for lab/team use)**

```bash
#!/bin/bash
#SBATCH --job-name=ksl-server
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --account=ucb-general

module purge
module load python/3.10.2
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2/frontend/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Access via SSH tunnel:
```bash
ssh -L 8000:NODENAME:8000 hama5612@login.rc.colorado.edu
```

**Option B: Export to ONNX for lightweight/local deployment**
```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 90, 48)
torch.onnx.export(model, dummy_input, "ksl_stgcn.onnx",
                  input_names=["landmarks"],
                  output_names=["prediction"],
                  dynamic_axes={"landmarks": {0: "batch_size"}})
```

### 5.2 Connect Frontend to Real Backend

1. Copy best checkpoint to `frontend/backend/models/`
2. Update `frontend/backend/app.py` to load the real ST-GCN model
3. Update model version dropdown to highlight the best version
4. Test end-to-end: webcam -> MediaPipe landmarks -> ST-GCN model -> prediction display
5. Fix "Kuwaiti" -> "Kenya" throughout frontend (if not done in Phase 4)

### 5.3 Academic Paper Preparation

**Recommended structure:**
1. **Introduction:** KSL recognition challenge, dataset description (30 classes, 5+5 signers)
2. **Related Work:** Sign language recognition, ST-GCN for action recognition, domain shift
3. **Method:** Architecture evolution (v7-v15), data pipeline, augmentation strategy
4. **Experiments:** Progressive model comparison, ablation studies, 444 challenge analysis
5. **Results:** Full comparison table, confusion matrix, per-class metrics
6. **Discussion:** Domain shift (signer generalization), 444 challenge, Alpine HPC benefits
7. **Conclusion:** Best accuracy achieved, remaining challenges, future work

**Key figures to generate:**
- ST-GCN architecture diagram (48-node graph with hand+pose connections)
- Model evolution chart (v7 -> v15 accuracy progression)
- Confusion matrix heatmap (best model)
- Per-class accuracy bar chart (all 30 classes)
- Training curves (loss + accuracy over epochs)
- 444 vs 54 temporal analysis (segment similarity visualization)
- Hand skeleton graph visualization

**Alpine HPC as selling point:** Free compute, A100 GPUs vs cloud T4, reproducible SLURM jobs

### 5.4 Phase 5 Checklist

- [ ] Deploy backend with real model on Alpine (SLURM or Open OnDemand)
- [ ] Test end-to-end prediction pipeline (webcam -> prediction)
- [ ] Export best model to ONNX
- [ ] Prepare full comparison table (v7-v15)
- [ ] Generate confusion matrix and per-class accuracy visualizations
- [ ] Draft paper outline and key figures
- [ ] Back up all final models to `/projects/hama5612/ksl-models/`

---

## Appendix: Alpine HPC Reference

### GPU Partition Quick Reference

| Partition | GPU | Memory | GPUs/Node | Nodes | Max Time | SU/hr | Best For |
|-----------|-----|--------|-----------|-------|----------|-------|----------|
| `aa100` | NVIDIA A100 | 40GB | 3 | 11 | 24 hours | 6 | **Primary training** |
| `al40` | NVIDIA L40 | 48GB | 3 | 3 | 24 hours | 4 | Alternative training |
| `gh200` | NVIDIA GH200 | 96GB | 1 | 2 | 7 days | ? | Long runs |
| `atesting_a100` | A100 MIG | 20GB | 6 | 1 | 1 hour | low | **Quick eval/debug** |
| `ami100` | AMD MI100 | 32GB | 3 | 8 | 24 hours | 3 | **AVOID** (needs ROCm, not CUDA) |

**Recommended:** `aa100` for training, `atesting_a100` for evaluation/debugging

### Module Loading (CRITICAL: Order Matters)

Alpine uses a **hierarchical module system**. CUDA modules are only visible after loading Python:

```bash
# CORRECT:
module purge
module load python/3.10.2    # Load Python FIRST
module load cuda/11.8.0       # Now CUDA is visible

# WRONG (will fail):
module load cuda/11.8.0       # Error: module not found
module load python/3.10.2
```

### Useful SLURM Commands

```bash
# Submit a job
sbatch slurm/train_gpu.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# Check job output while running
tail -f data/results/ksl-train_<JOBID>.out

# Check account usage
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,MaxRSS,State -S 2026-02-01

# Interactive GPU session (for debugging)
sinteractive --partition=atesting_a100 --gres=gpu:a100_3g.20gb:1 --time=00:30:00 --account=ucb-general --qos=testing

# Interactive compile node (no GPU, for installing packages)
sinteractive --partition=acompile --time=02:00:00

# Check GPU partition availability
sinfo -p aa100 --format="%P %a %G %D %T"

# Check storage quota
curc-quota
```

### Key File Paths

| Resource | Path | Persistent? |
|----------|------|-------------|
| Project code | `/scratch/alpine/hama5612/ksl-dir-2/` | 90-day purge |
| Training data (.npy) | `/scratch/alpine/hama5612/ksl-dir-2/data/train_v2/` | 90-day purge |
| Validation data (.npy) | `/scratch/alpine/hama5612/ksl-dir-2/data/val_v2/` | 90-day purge |
| Raw training videos | `/scratch/alpine/hama5612/ksl-dir-2/dataset/` | 90-day purge |
| Raw validation videos | `/scratch/alpine/hama5612/ksl-dir-2/validation-data/` | 90-day purge |
| Model checkpoints | `/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/` | 90-day purge |
| Experiment logs | `/scratch/alpine/hama5612/ksl-dir-2/data/results/` | 90-day purge |
| **Persistent model backups** | `/projects/hama5612/ksl-models/` | **Permanent** |
| **Conda environments** | `/projects/hama5612/software/anaconda/envs/` | **Permanent** |
| ksl-alpha reference | `/projects/hama5612/ksl-alpha/alpine/` | **Permanent** |
| Home directory | `/home/hama5612/` | Permanent (**FULL, avoid**) |

### Storage Warnings

1. **`/home/` is nearly full** (1.3G/2.0G). Never install packages or store data here.
2. **`/scratch/` purges after 90 days.** Regularly back up important results to `/projects/`.
3. **`/projects/` is persistent** but limited (250 GB). Use for code, envs, final models only.
4. **No internet on compute nodes.** All `pip install` / `conda install` must happen on login or compile nodes before submitting SLURM jobs.

### SU Budget Estimates

| Activity | Hours | SU Rate | Total SUs |
|----------|-------|---------|-----------|
| Phase 1: Migration testing | 5 | 6/hr (A100) | 30 |
| Phase 2: Evaluation (5 models) | 3 | 6/hr | 18 |
| Phase 3: Training runs (20 experiments) | 20 | 6/hr | 120 |
| Phase 3: Sweeps (6 configs x 3 rounds) | 18 | 6/hr | 108 |
| Phase 3: Ensemble experiments | 5 | 6/hr | 30 |
| Phase 5: Inference testing | 2 | 6/hr | 12 |
| **Total estimated** | **53** | | **~318 SUs** |

With the `ucb-general` allocation this is well within budget.

### Reference: ksl-alpha Files (Reusable Templates)

| ksl-alpha File | Adapt For |
|----------------|-----------|
| `/projects/hama5612/ksl-alpha/alpine/train_alpine.py` | ST-GCN training script template |
| `/projects/hama5612/ksl-alpha/alpine/slurm/test_training_updated.sh` | SLURM job template |
| `/projects/hama5612/ksl-alpha/alpine/install.sh` | One-command setup script |
| `/projects/hama5612/ksl-alpha/alpine/requirements.txt` | Python dependency template |
| `/projects/hama5612/ksl-alpha/alpine/SETUP_GUIDE.md` | Documentation template |

---

## Timeline Summary

| Phase | Duration | GPU Hours | SUs | Priority | Dependencies |
|-------|----------|-----------|-----|----------|-------------|
| Phase 1: Alpine Migration | 1-2 weeks | ~5 | ~30 | HIGHEST | None |
| Phase 2: Model Evaluation | 1 week | ~3 | ~18 | HIGH | Phase 1 |
| Phase 3: Model Improvements | 3-5 weeks | ~50 | ~300 | HIGH | Phase 2 |
| Phase 4: Code Quality | 2-3 weeks | ~0 | ~0 | MEDIUM | Can parallel Phases 2-3 |
| Phase 5: Deployment & Academic | 2-3 weeks | ~2 | ~12 | MEDIUM | Phases 3-4 |
| **Total** | **~8-12 weeks** | **~60** | **~360** | | |

**Critical path:** Phase 1 -> Phase 2 -> Phase 3 (model accuracy is the primary blocker)
**Parallel work:** Phase 4 can proceed alongside Phases 2-3

---

## Success Criteria (v1.0 Release)

- [ ] Overall accuracy > 85%
- [ ] Class 444 accuracy > 40% (stretch: > 50%)
- [ ] No class below 50% accuracy
- [ ] All training runs on Alpine HPC (zero Modal dependency)
- [ ] Shared model code (no duplication across training scripts)
- [ ] Unit test coverage for core modules (dataset, model, graph, augmentation)
- [ ] Experiment tracking with TensorBoard
- [ ] Frontend connected to real model with correct "Kenya Sign Language" text
- [ ] All checkpoints and results backed up to `/projects/hama5612/ksl-models/`
- [ ] Documentation updated to reflect Alpine workflow
