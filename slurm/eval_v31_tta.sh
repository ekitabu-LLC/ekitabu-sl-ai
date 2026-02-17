#!/bin/bash
#SBATCH --job-name=eval_v31_tta
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v31_tta_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v31_tta_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e

echo "==========================================="
echo "V31 Exp1 TTA Evaluation - GroupNorm"
echo "==========================================="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python: $(which python)"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

python evaluate_v31_exp1_tta.py --category both

echo ""
echo "==========================================="
echo "TTA Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
