#!/bin/bash
#SBATCH --job-name=v34a_num
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v34a_seed123_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v34a_seed123_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V34a: GroupNorm with seed=123 (same arch as v31_exp1, different seed)
# For multi-seed GroupNorm ensemble (signer-agnostic)
# ==============================================================================

set -e

echo "==========================================="
echo "V34a GroupNorm Seed=123 - Numbers + Words"
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
export TMPDIR="/scratch/alpine/$USER/tmp"
mkdir -p "$PROJECT_DIR/data/checkpoints/v34a_seed123" "$TMPDIR"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

# Train numbers
echo "=== Training Numbers ==="
python train_ksl_v31_exp1.py --model-type numbers --seed 123 \
  --checkpoint-dir data/checkpoints/v34a_seed123

echo ""
echo "=== Training Words ==="
python train_ksl_v31_exp1.py --model-type words --seed 123 \
  --checkpoint-dir data/checkpoints/v34a_seed123

echo ""
echo "==========================================="
echo "V34a Seed=123 Complete"
echo "End time: $(date)"
echo "==========================================="
