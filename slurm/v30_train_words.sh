#!/bin/bash
#SBATCH --job-name=v30_words
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30_train_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30_train_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30 Phase 2 Training - WORDS ONLY
# ==============================================================================
# Trains words model (3 streams: joint, bone, velocity)
# Partition: aa100 (NVIDIA A100), 1hr limit
# Usage: sbatch slurm/v30_train_words.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL V30 Phase 2 Training - WORDS ONLY"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
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
export DATA_DIR="$PROJECT_DIR/data"
export CKPT_DIR="$DATA_DIR/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$CKPT_DIR" "$TMPDIR" "$DATA_DIR/results/v30"

echo "Project:     $PROJECT_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

cd "$PROJECT_DIR"

COMMON_FLAGS="--mixstyle --dropgraph --confused-pairs --label-smooth 0.1 --r-drop 1.0 --no-swad --aggressive-aug --skeleton-noise"

echo "=========================================="
echo "Training WORDS model (Phase 2, all improvements)"
echo "Flags: $COMMON_FLAGS"
echo "=========================================="
echo ""

python train_ksl_v30.py --model-type words $COMMON_FLAGS --seed 42

echo ""
echo "=========================================="
echo "V30 Words Training Complete"
echo "End time: $(date)"
echo "=========================================="
