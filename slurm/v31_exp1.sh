#!/bin/bash
#SBATCH --job-name=v31_exp1
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp1_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp1_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V31 Exp 1: GroupNorm replacing BatchNorm in v28 architecture
# Baseline: v28 AdaBN Global Numbers = 55.9%, Words = 60.5%
# ==============================================================================

set -e

echo "=========================================="
echo "V31 Exp 1: GroupNorm - Numbers"
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
mkdir -p "$PROJECT_DIR/data/checkpoints/v31_exp1" "$TMPDIR"

cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Training: GroupNorm replacing BatchNorm"
echo "=========================================="
echo ""

python train_ksl_v31_exp1.py --model-type numbers --seed 42 \
  --checkpoint-dir data/checkpoints/v31_exp1

echo ""
echo "=========================================="
echo "V31 Exp 1 (GroupNorm) Complete"
echo "End time: $(date)"
echo "=========================================="
