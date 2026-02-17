#!/bin/bash
#SBATCH --job-name=v30_phase1_eval
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30_phase1_eval_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30_phase1_eval_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30 Phase 1 Evaluation Script (No Retraining)
# ==============================================================================
# Inference-time improvements on existing v27/v28/v29 models:
#   - Temperature scaling, Alpha-BN, TENT+AdaBN, ensemble,
#     confidence-weighted fusion, T3A prototype classifier
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 1 hour
# Usage: sbatch slurm/v30_phase1_eval.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL V30 Phase 1 Evaluation - Alpine HPC"
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
# Phase 1: Full Ablation (all techniques)
# ------------------------------------------------------------------------------

echo "=========================================="
echo "Phase 1 Full Ablation"
echo "=========================================="

python evaluate_real_testers_v30_phase1.py --ablation && ABLATION_CODE=0 || ABLATION_CODE=$?

echo ""
echo "Ablation exit code: $ABLATION_CODE"
echo ""

# ------------------------------------------------------------------------------
# Phase 1: Individual Technique Runs
# ------------------------------------------------------------------------------

echo "=========================================="
echo "Individual Technique Evaluation"
echo "=========================================="

echo ""
echo "--- TENT (3 steps) ---"
python evaluate_real_testers_v30_phase1.py --tent 3 && echo "TENT: OK" || echo "TENT: FAILED"

echo ""
echo "--- Version Ensemble (v27+v28+v29) ---"
python evaluate_real_testers_v30_phase1.py --ensemble && echo "Ensemble: OK" || echo "Ensemble: FAILED"

echo ""
echo "--- Alpha-BN (alpha=0.3) ---"
python evaluate_real_testers_v30_phase1.py --alpha-bn 0.3 && echo "Alpha-BN: OK" || echo "Alpha-BN: FAILED"

echo ""
echo "--- Confidence-Weighted Fusion ---"
python evaluate_real_testers_v30_phase1.py --conf-fusion && echo "Conf-Fusion: OK" || echo "Conf-Fusion: FAILED"

echo ""
echo "--- T3A Prototype Classifier ---"
python evaluate_real_testers_v30_phase1.py --t3a && echo "T3A: OK" || echo "T3A: FAILED"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Phase 1 evaluation completed"
echo "End time: $(date)"
echo "=========================================="
echo "Results saved to: $DATA_DIR/results/v30/"

exit ${ABLATION_CODE:-0}
