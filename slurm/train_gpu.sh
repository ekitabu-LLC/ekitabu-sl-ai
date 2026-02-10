#!/bin/bash
#SBATCH --job-name=ksl-train
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# GPU Training Script for KSL Recognition on Alpine
# ==============================================================================
# Partition: aa100 (NVIDIA A100 40GB)
# Duration: Up to 6 hours
# Usage: MODEL_TYPE=both sbatch slurm/train_gpu.sh
#   MODEL_TYPE options: numbers, words, both (default: both)
# ==============================================================================

set -e

echo "=========================================="
echo "KSL Training - Alpine HPC"
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
export TRAIN_DIR="$DATA_DIR/train_v2"
export VAL_DIR="$DATA_DIR/val_v2"
export CKPT_DIR="$DATA_DIR/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$CKPT_DIR" "$TMPDIR" "$DATA_DIR/results"

echo "Project:     $PROJECT_DIR"
echo "Train data:  $TRAIN_DIR"
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
# Run Training
# ------------------------------------------------------------------------------

MODEL_TYPE="${MODEL_TYPE:-both}"
echo "Starting training: v16 (model-type=${MODEL_TYPE})"
echo ""

cd "$PROJECT_DIR"

python train_ksl_v16.py --model-type "$MODEL_TYPE" && EXIT_CODE=0 || EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training successful!"
    echo "Checkpoints saved to: $CKPT_DIR"
else
    echo "Training failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/data/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
