#!/bin/bash
#SBATCH --job-name=v30b_eval
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_eval_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_eval_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30b Ablation Evaluation - All completed experiments
# ==============================================================================

set -e

echo "=========================================="
echo "V30b Ablation Evaluation"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo ""

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

echo "GPU Information:"
nvidia-smi
echo ""

# Evaluate Exp 1: SWAD alone
echo ""
echo "############################################################"
echo "# EXP 1: SWAD ALONE"
echo "############################################################"
echo ""
python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp1

# Evaluate Exp 2: Confused-pairs alone
echo ""
echo "############################################################"
echo "# EXP 2: CONFUSED-PAIRS ALONE (weight=0.01)"
echo "############################################################"
echo ""
python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp2

echo ""
echo "=========================================="
echo "V30b Evaluation Complete"
echo "End time: $(date)"
echo "=========================================="
