#!/bin/bash
#SBATCH --job-name=ksl-extract-v27
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/extract_alpha_v27_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/extract_alpha_v27_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V27: Re-extract ksl-alpha landmarks with MediaPipe 0.10.5 on Alpine
# ==============================================================================
# Deletes all existing .npy files (extracted with 0.10.14) and re-extracts
# with the Alpine MediaPipe version (0.10.5) to fix the version mismatch.
#
# Usage: sbatch slurm/extract_alpha_v27.sh
# ==============================================================================

set -e

echo "=========================================="
echo "V27: KSL-Alpha Landmark Re-extraction"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Memory:    $SLURM_MEM_PER_NODE MB"
echo "Start:     $(date)"
echo ""

# Environment Setup
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:    $(which python)"
echo "MediaPipe: $(python -c 'import mediapipe; print(mediapipe.__version__)')"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export ALPHA_DIR="/scratch/alpine/$USER/ksl-alpha/cleaned_data"
export DATA_DIR="$PROJECT_DIR/data"

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# Step 1: Delete all existing .npy files (extracted with MediaPipe 0.10.14)
# ------------------------------------------------------------------------------

echo "Step 1: Deleting existing .npy files..."

for SPLIT_DIR in train_alpha val_alpha test_alpha; do
    TARGET="$DATA_DIR/$SPLIT_DIR"
    if [ -d "$TARGET" ]; then
        COUNT=$(find "$TARGET" -name "*.npy" | wc -l)
        echo "  $SPLIT_DIR: deleting $COUNT .npy files"
        find "$TARGET" -name "*.npy" -delete
    else
        echo "  $SPLIT_DIR: directory not found, skipping"
    fi
done

echo "Deletion complete."
echo ""

# ------------------------------------------------------------------------------
# Step 2: Re-extract with --force flag and MediaPipe 0.10.5
# ------------------------------------------------------------------------------

echo "Step 2: Re-extracting landmarks with MediaPipe 0.10.5..."
echo "Source: $ALPHA_DIR"
echo "Output: $DATA_DIR/{train_alpha,val_alpha,test_alpha}/"
echo "Workers: 32"
echo ""

python extract_landmarks_alpha.py \
    --alpha-dir "$ALPHA_DIR" \
    --output-dir "$DATA_DIR" \
    --split all \
    --workers 32 \
    --force && EXIT_CODE=0 || EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Step 3: Verify output
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Extraction completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Verifying output..."
    TRAIN_COUNT=$(find "$DATA_DIR/train_alpha" -name "*.npy" | wc -l)
    VAL_COUNT=$(find "$DATA_DIR/val_alpha" -name "*.npy" | wc -l)
    TEST_COUNT=$(find "$DATA_DIR/test_alpha" -name "*.npy" | wc -l)
    TOTAL=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
    echo "  Train files: $TRAIN_COUNT"
    echo "  Val files:   $VAL_COUNT"
    echo "  Test files:  $TEST_COUNT"
    echo "  Total:       $TOTAL (expected ~2237)"
    echo ""
    echo "MediaPipe version used: $(python -c 'import mediapipe; print(mediapipe.__version__)')"
    if [ $TOTAL -lt 2000 ]; then
        echo "WARNING: Expected ~2237 files, got $TOTAL. Check extraction logs."
    else
        echo "SUCCESS: All files re-extracted with MediaPipe 0.10.5"
    fi
else
    echo "Extraction failed! Check error log."
fi

exit $EXIT_CODE
