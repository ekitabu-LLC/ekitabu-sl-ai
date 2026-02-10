#!/bin/bash
#SBATCH --job-name=ksl-sweep
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%A_%a.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%A_%a.err
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu
#SBATCH --array=0-5

# ==============================================================================
# Hyperparameter Sweep Script for KSL Recognition on Alpine
# ==============================================================================
# Partition: al40 (NVIDIA L40)
# Job Array: 6 jobs (indices 0-5), one per hyperparameter combo
# Duration: Up to 6 hours per job
# Usage: sbatch slurm/sweep_gpu.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Hyperparameter Grid
# ------------------------------------------------------------------------------

LEARNING_RATES=(0.0001 0.0003 0.001 0.0001 0.0003 0.001)
DROPOUT_VALUES=(0.3    0.3    0.3   0.5    0.5    0.5)

LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
DROPOUT=${DROPOUT_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "KSL Hyperparameter Sweep - Alpine HPC"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "Start:        $(date)"
echo ""
echo "Hyperparameters:"
echo "  Learning Rate: $LR"
echo "  Dropout:       $DROPOUT"
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
export CKPT_DIR="$DATA_DIR/checkpoints/sweep_${SLURM_ARRAY_JOB_ID}"
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
# Run Training with Sweep Parameters
# ------------------------------------------------------------------------------

echo "Starting sweep task $SLURM_ARRAY_TASK_ID (lr=$LR, dropout=$DROPOUT)..."
echo ""

cd "$PROJECT_DIR"

python train_ksl_alpine.py \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --sweep-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Sweep task $SLURM_ARRAY_TASK_ID completed with exit code: $EXIT_CODE"
echo "  Learning Rate: $LR"
echo "  Dropout:       $DROPOUT"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Sweep task successful!"
    echo "Checkpoints saved to: $CKPT_DIR"
else
    echo "Sweep task failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/data/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
fi

exit $EXIT_CODE
