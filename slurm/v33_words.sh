#!/bin/bash
#SBATCH --job-name=v33_wrd
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v33_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v33_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V33 KD Training - Words
# Teacher: v27+v28+v29+exp1+exp5 ensemble (70.7%)
# Student: GroupNorm ST-GCN (signer-agnostic)
# ==============================================================================

set -e

echo "=========================================="
echo "V33 KD Training - Words"
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
mkdir -p "$PROJECT_DIR/data/checkpoints/v33" "$TMPDIR"

cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

python train_ksl_v33.py --model-type words --seed 42 \
  --checkpoint-dir data/checkpoints/v33b

echo ""
echo "=========================================="
echo "V33 Words Complete"
echo "End time: $(date)"
echo "=========================================="
