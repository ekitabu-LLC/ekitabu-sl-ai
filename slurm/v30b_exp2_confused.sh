#!/bin/bash
#SBATCH --job-name=v30b_cp
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_exp2_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_exp2_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30b Ablation Exp 2: Confused-pair contrastive alone (weight 0.01)
# Baseline: v28 AdaBN Global Numbers = 55.9%
# ==============================================================================

set -e

echo "=========================================="
echo "V30b Exp 2: CONFUSED-PAIRS ALONE - Numbers"
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
mkdir -p "$PROJECT_DIR/data/checkpoints/v30b_exp2" "$TMPDIR"

cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Training: Confused-pairs only (weight=0.01), all other regularizers OFF"
echo "=========================================="
echo ""

python train_ksl_v30.py --model-type numbers --seed 42 \
  --no-mixstyle --no-dropgraph --confused-pairs --cp-weight 0.01 \
  --label-smooth 0 --r-drop 0 --no-swad \
  --no-aggressive-aug --no-skeleton-noise \
  --checkpoint-dir data/checkpoints/v30b_exp2

echo ""
echo "=========================================="
echo "V30b Exp 2 (Confused-Pairs) Complete"
echo "End time: $(date)"
echo "=========================================="
