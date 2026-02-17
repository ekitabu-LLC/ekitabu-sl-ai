#!/bin/bash
#SBATCH --job-name=v30_train
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30_train_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30_train_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30 Phase 2 Training Script on Alpine
# ==============================================================================
# Phase 2: Full training with all improvements:
#   MixStyle, DropGraph, confused-pair loss, label smoothing, R-Drop,
#   SWAD, aggressive augmentation, skeleton noise
# Trains numbers then words sequentially (~40 min each)
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 2 hours
# Usage: sbatch slurm/v30_train.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL V30 Phase 2 Training - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $SLURM_GPUS"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Memory:    $SLURM_MEM_PER_NODE MB"
echo "Start:     $(date)"
echo ""

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ------------------------------------------------------------------------------
# Directory Setup
# ------------------------------------------------------------------------------

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export CKPT_DIR="$DATA_DIR/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$CKPT_DIR" "$TMPDIR" "$DATA_DIR/results/v30"

echo "Project:     $PROJECT_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

# ------------------------------------------------------------------------------
# GPU Info
# ------------------------------------------------------------------------------

echo "GPU Information:"
nvidia-smi
echo ""

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# Train Numbers Model
# ------------------------------------------------------------------------------

COMMON_FLAGS="--mixstyle --dropgraph --confused-pairs --label-smooth 0.1 --r-drop 1.0 --swad --aggressive-aug --skeleton-noise"

echo "=========================================="
echo "Training NUMBERS model (Phase 2, all improvements)"
echo "Flags: $COMMON_FLAGS"
echo "=========================================="
echo ""

python train_ksl_v30.py --model-type numbers $COMMON_FLAGS --seed 42 && NUM_CODE=0 || NUM_CODE=$?

echo ""
echo "Numbers training exit code: $NUM_CODE"
echo ""

# ------------------------------------------------------------------------------
# Train Words Model
# ------------------------------------------------------------------------------

echo "=========================================="
echo "Training WORDS model (Phase 2, all improvements)"
echo "Flags: $COMMON_FLAGS"
echo "=========================================="
echo ""

python train_ksl_v30.py --model-type words $COMMON_FLAGS --seed 42 && WORD_CODE=0 || WORD_CODE=$?

echo ""
echo "Words training exit code: $WORD_CODE"
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "V30 Phase 2 Training Complete"
echo "End time: $(date)"
echo "=========================================="
echo "Numbers exit code: $NUM_CODE"
echo "Words exit code:   $WORD_CODE"
echo "Checkpoints: $CKPT_DIR"

if [ $NUM_CODE -eq 0 ] && [ $WORD_CODE -eq 0 ]; then
    echo "Both models trained successfully!"
    exit 0
else
    echo "Training failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/slurm/v30_train_${SLURM_JOB_ID}.err"
    exit 1
fi
