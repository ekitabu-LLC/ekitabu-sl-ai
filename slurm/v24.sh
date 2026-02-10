#!/bin/bash
#SBATCH --job-name=ksl-v24
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v24_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v24_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# GPU Training Script for KSL v24 on Alpine
# ==============================================================================
# V24: Prototypical classification + MixStyle + wider augmentation
# Duration: ~8 min without LOSO, ~40 min with LOSO
# Usage: sbatch slurm/v24.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL v24 Training - Alpine HPC"
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

# Environment Setup
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Directory Setup
export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$DATA_DIR/checkpoints" "$TMPDIR" "$DATA_DIR/results"

echo "Project: $PROJECT_DIR"
echo ""

# GPU Info
echo "GPU Information:"
nvidia-smi
echo ""

# Run Training (standard mode, no LOSO)
echo "Starting training: v24 (model-type=both, seed=42)"
echo ""

cd "$PROJECT_DIR"

python train_ksl_v24.py --model-type both --seed 42 && EXIT_CODE=0 || EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training successful!"
    echo "Checkpoints: $DATA_DIR/checkpoints/v24_numbers/ and v24_words/"
else
    echo "Training failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/slurm/v24_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
