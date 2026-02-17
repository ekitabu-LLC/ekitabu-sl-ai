#!/bin/bash
#SBATCH --job-name=eval_v33
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v33_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v33_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V33 Evaluation on Real Testers (Signer-Agnostic)
# Uses evaluate_v32.py with v33 checkpoint dir (same GroupNorm arch)
# NO BN adaptation needed — true signer-agnostic evaluation
# ==============================================================================

set -e

echo "=========================================="
echo "V33 KD Evaluation - Real Testers"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

# Evaluate v33 using the same evaluation script as v32 (both are GroupNorm)
# Just point to v33 checkpoint directory
python evaluate_v33.py --category both \
  --checkpoint-dir data/checkpoints/v33b

echo ""
echo "=========================================="
echo "V33 Evaluation Complete"
echo "End time: $(date)"
echo "=========================================="
