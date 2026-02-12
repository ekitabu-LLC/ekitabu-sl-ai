#!/bin/bash
#SBATCH --job-name=ksl-v25
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v25_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v25_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# GPU Training Script for KSL v25 on Alpine
# ==============================================================================
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 1 hour
# Usage: sbatch slurm/v25.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL v25 Training - Alpine HPC"
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

mkdir -p "$CKPT_DIR" "$TMPDIR" "$DATA_DIR/results"

echo "Project:     $PROJECT_DIR"
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

echo "Starting training: v25 (model-type=both, seed=42)"
echo ""

cd "$PROJECT_DIR"

python train_ksl_v25.py --model-type both --seed 42 && EXIT_CODE=0 || EXIT_CODE=$?

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
    echo "  /scratch/alpine/$USER/ksl-dir-2/slurm/v25_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
