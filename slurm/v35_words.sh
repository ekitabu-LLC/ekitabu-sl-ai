#!/bin/bash
#SBATCH --job-name=v35_wrd
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v35_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v35_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V35: GroupNorm + EMA + RSC - Words
# ==============================================================================

set -e

echo "==========================================="
echo "V35 GroupNorm + EMA + RSC - Words"
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
mkdir -p "$PROJECT_DIR/data/checkpoints/v35" "$TMPDIR"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

python train_ksl_v35.py --model-type words --seed 42

echo ""
echo "==========================================="
echo "V35 Words Complete"
echo "End time: $(date)"
echo "==========================================="
