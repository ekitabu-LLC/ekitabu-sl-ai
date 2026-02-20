#!/bin/bash
#SBATCH --job-name=eval_t3a
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_t3a_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_t3a_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# T3A (Test-Time Classifier Adjustment) on Weighted Ensemble
# Applies T3A at softmax-prob level on top of optimized 5-model ensemble
# ==============================================================================

set -e

echo "==========================================="
echo "T3A on Weighted Ensemble - Real Testers"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

echo "=== T3A Evaluation ==="
python evaluate_t3a_ensemble.py

echo ""
echo "==========================================="
echo "T3A Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
