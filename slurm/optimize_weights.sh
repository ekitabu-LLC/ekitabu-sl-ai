#!/bin/bash
#SBATCH --job-name=opt_wts
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/opt_weights_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/opt_weights_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# Optimize ensemble weights for v27+v28+v29+exp1+exp5
# Zero-cost improvement: no training, just find best weights
# ==============================================================================

set -e

echo "==========================================="
echo "Ensemble Weight Optimization"
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

python optimize_ensemble_weights.py

echo ""
echo "==========================================="
echo "Optimization Complete"
echo "End time: $(date)"
echo "==========================================="
