#!/bin/bash
#SBATCH --job-name=eval_v33b
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v33b_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v33b_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e

echo "==========================================="
echo "V33b KD (T=20, alpha=0.3) Evaluation"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

nvidia-smi
echo ""

python evaluate_v32.py --category both \
  --checkpoint-dir data/checkpoints/v33b

echo ""
echo "==========================================="
echo "V33b Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
