#!/bin/bash
#SBATCH --job-name=ksl-features
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/data/results/%x_%j.err
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# CPU Feature Extraction Script for KSL Recognition on Alpine
# ==============================================================================
# Partition: amilan (CPU-only, 16 cores, 64GB RAM)
# Duration: Up to 4 hours
# Usage: sbatch slurm/extract_features.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL Feature Extraction - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
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
echo ""

# ------------------------------------------------------------------------------
# Directory Setup
# ------------------------------------------------------------------------------

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export TRAIN_DIR="$DATA_DIR/modal_backup/train_v2"
export VAL_DIR="$DATA_DIR/modal_backup/val_v2"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$TMPDIR" "$DATA_DIR/results"

echo "Project:    $PROJECT_DIR"
echo "Train data: $TRAIN_DIR"
echo "Val data:   $VAL_DIR"
echo ""

# ------------------------------------------------------------------------------
# Run Feature Extraction
# ------------------------------------------------------------------------------

echo "Starting feature extraction..."
echo ""

cd "$PROJECT_DIR"

python extract_features_alpine.py

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Feature extraction completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Feature extraction successful!"
else
    echo "Feature extraction failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/data/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
