#!/bin/bash
#SBATCH --job-name=ksl-eval-v26
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v26_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v26_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# Real-Tester Evaluation Script for KSL v26 on Alpine
# ==============================================================================
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 1 hour
# Usage: sbatch slurm/eval_v26.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL v26 Real-Tester Evaluation - Alpine HPC"
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
echo ""

# Set headless mode for OpenCV/MediaPipe (no display on compute nodes)
export DISPLAY=""
export MPLBACKEND=Agg

# ------------------------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------------------------

cd /scratch/alpine/$USER/ksl-dir-2

echo "Starting real-tester evaluation for v26..."
echo ""

python evaluate_real_testers_v26.py && EXIT_CODE=0 || EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation successful! Results saved to data/results/"
else
    echo "Evaluation failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/slurm/eval_v26_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
