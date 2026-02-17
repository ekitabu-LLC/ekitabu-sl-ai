#!/bin/bash
#SBATCH --job-name=eval_v36
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v36_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v36_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V36 Evaluation on Real Testers
# GroupNorm — no BN adaptation needed
# ==============================================================================

set -e

echo "==========================================="
echo "V36 Synthetic Signer Evaluation - Real Testers"
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

echo "=== V36 Evaluation ==="
python evaluate_v32.py --category both \
  --checkpoint-dir data/checkpoints/v36

echo ""
echo "==========================================="
echo "V36 Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
