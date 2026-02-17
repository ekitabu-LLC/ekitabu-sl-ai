#!/bin/bash
#SBATCH --job-name=ksl-v27w
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v27_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v27_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V27 Words-Only Retrain (patience 80 instead of 40)
# ==============================================================================

set -e

echo "=========================================="
echo "KSL v27 Words Retrain (patience=80) - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export CKPT_DIR="$PROJECT_DIR/data/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"
mkdir -p "$CKPT_DIR" "$TMPDIR"

echo "GPU Information:"
nvidia-smi
echo ""

cd "$PROJECT_DIR"

echo "Starting training: v27 words only (patience=80, seed=42)"
echo "  12 signers train, 3 signers val"
echo "  Focal loss (gamma=2, class-balanced alpha)"
echo ""

python train_ksl_v27.py --model-type words --seed 42 && EXIT_CODE=0 || EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
