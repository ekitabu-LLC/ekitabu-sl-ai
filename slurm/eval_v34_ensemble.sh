#!/bin/bash
#SBATCH --job-name=eval_v34_ens
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v34_ensemble_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v34_ensemble_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V34 GroupNorm Multi-Seed Ensemble Evaluation
# Combines exp1 (seed=42) + v34a (seed=123) + v34b (seed=456)
# ALL GroupNorm — true signer-agnostic ensemble, no BN adaptation
# ==============================================================================

set -e

echo "==========================================="
echo "V34 GroupNorm Ensemble Evaluation"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python: $(which python)"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

python evaluate_v34_gn_ensemble.py --category both --models exp1,v34a,v34b

echo ""
echo "==========================================="
echo "V34 Ensemble Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
