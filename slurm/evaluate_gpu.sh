#!/bin/bash
#SBATCH --job-name=ksl-eval
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.err
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# GPU Evaluation Script for KSL Recognition on Alpine
# ==============================================================================
# Partition: al40 (NVIDIA L40)
# Duration: Up to 30 minutes
# Usage: VERSION=15 sbatch slurm/evaluate_gpu.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL Evaluation - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
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
echo ""

# ------------------------------------------------------------------------------
# Directory Setup
# ------------------------------------------------------------------------------

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export TRAIN_DIR="$DATA_DIR/train_v2"
export VAL_DIR="$DATA_DIR/val_v2"
export CKPT_DIR="$DATA_DIR/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$TMPDIR" "$DATA_DIR/results"

echo "Project:     $PROJECT_DIR"
echo "Val data:    $VAL_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

# ------------------------------------------------------------------------------
# GPU Info
# ------------------------------------------------------------------------------

echo "GPU Information:"
nvidia-smi
echo ""

# ------------------------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------------------------

VERSION="${VERSION:-15}"
echo "Evaluating model version: v${VERSION}"
echo ""

cd "$PROJECT_DIR"

python evaluate_alpine.py --version "$VERSION"

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation successful!"
else
    echo "Evaluation failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/data/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
