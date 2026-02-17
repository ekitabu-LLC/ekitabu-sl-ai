#!/bin/bash
#SBATCH --job-name=v33_labels
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v33_labels_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v33_labels_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V33 Phase 1: Generate KD soft labels from 5-model ensemble
# Loads v27, v28, v29, v31_exp1, v31_exp5 → generates soft labels for all
# training samples. Output: data/kd_labels/{numbers,words}_ensemble_logits.pt
# ==============================================================================

set -e

echo "=========================================="
echo "V33 Phase 1: Generate KD Labels"
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
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR="/scratch/alpine/$USER/tmp"
mkdir -p "$PROJECT_DIR/data/kd_labels" "$TMPDIR"

cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

python generate_kd_labels.py --category both

echo ""
echo "=========================================="
echo "V33 Labels Complete"
echo "End time: $(date)"
echo "=========================================="
