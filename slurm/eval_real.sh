#!/bin/bash
#SBATCH --job-name=ksl-eval-real
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_real_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_real_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# Evaluate v19 on Real-World Test Signers
# ==============================================================================
# Extracts MediaPipe landmarks from .mov videos and runs v19 inference.
# Requires: GPU (model inference), mediapipe, opencv
# ==============================================================================

set -e

echo "=========================================="
echo "KSL v19 Real-World Evaluation"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:     $(date)"
echo ""

# Environment
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Set headless mode for OpenCV/MediaPipe (no display on compute nodes)
export DISPLAY=""
export MPLBACKEND=Agg

cd /scratch/alpine/$USER/ksl-dir-2

python evaluate_real_testers.py

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
