#!/bin/bash
#SBATCH --job-name=v30_eval
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30_eval_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30_eval_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30 Evaluation Script for Phase 2 and Phase 3 Models
# ==============================================================================
# Evaluates trained models on real testers with TENT + Alpha-BN
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 1 hour
# Usage: sbatch slurm/v30_eval.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL V30 Evaluation - Alpine HPC"
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
# Directory Setup
# ------------------------------------------------------------------------------

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$TMPDIR" "$DATA_DIR/results/v30"

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# GPU Info
# ------------------------------------------------------------------------------

echo "GPU Information:"
nvidia-smi
echo ""

# ------------------------------------------------------------------------------
# Evaluate Phase 2 Model
# ------------------------------------------------------------------------------

echo "=========================================="
echo "Evaluating Phase 2 Model (AdaBN + Alpha-BN + Ensemble)"
echo "=========================================="
echo ""

python evaluate_real_testers_v30.py --model-type phase2 --alpha-bn 0.3 --ensemble

echo ""
echo "=========================================="
echo "V30 Evaluation Complete"
echo "End time: $(date)"
echo "=========================================="
echo "Results saved to: $DATA_DIR/results/v30/"
