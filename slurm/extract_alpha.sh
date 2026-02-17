#!/bin/bash
#SBATCH --job-name=ksl-extract-alpha
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/extract_alpha_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/extract_alpha_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# Landmark Extraction from ksl-alpha videos (128-core parallel CPU job)
# ==============================================================================
# No GPU needed - MediaPipe runs on CPU, parallelized across 128 workers
# Usage: sbatch slurm/extract_alpha.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL-Alpha Landmark Extraction (64 cores)"
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

cd "$PROJECT_DIR"

echo "Extracting landmarks from ksl-alpha videos..."
echo "Source: $ALPHA_DIR"
echo "Output: $PROJECT_DIR/data/{train_alpha,val_alpha,test_alpha}/"
echo "Workers: 128"
echo ""

python extract_landmarks_alpha.py \
    --alpha-dir "$ALPHA_DIR" \
    --output-dir "$PROJECT_DIR/data" \
    --split all \
    --workers 32 && EXIT_CODE=0 || EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Extraction completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Verifying output..."
    echo "Train files:"
    find "$PROJECT_DIR/data/train_alpha" -name "*.npy" | wc -l
    echo "Val files:"
    find "$PROJECT_DIR/data/val_alpha" -name "*.npy" | wc -l
    echo "Test files:"
    find "$PROJECT_DIR/data/test_alpha" -name "*.npy" | wc -l
else
    echo "Extraction failed! Check error log."
fi

exit $EXIT_CODE
