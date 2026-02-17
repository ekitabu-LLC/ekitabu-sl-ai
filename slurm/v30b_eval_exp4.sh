#!/bin/bash
#SBATCH --job-name=v30b_ev4
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_eval_exp4_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30b_eval_exp4_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/$USER/ksl-dir-2

echo "=========================================="
echo "V30b Exp 4 Evaluation: SWAD + Confused-Pairs"
echo "=========================================="
nvidia-smi
echo ""

python evaluate_v30b_ablation.py --checkpoint-dir data/checkpoints/v30b_exp4

echo ""
echo "Done: $(date)"
