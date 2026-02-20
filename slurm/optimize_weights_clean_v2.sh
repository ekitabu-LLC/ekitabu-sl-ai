#!/bin/bash
#SBATCH --job-name=opt_clean_v2
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/opt_clean_v2_%j.out
#SBATCH --error=slurm/opt_clean_v2_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

# Audit fix: --min_weight 0.05 prevents any model from being zeroed out.
# Root cause: pure accuracy on test_alpha zeroed exp5 for words (the best words
# model!) and dropped v27/v28/v29 for numbers, causing 66.4% clean < 72.9% uniform.
python optimize_ensemble_clean.py --step 0.1 --min_weight 0.05

echo "Job finished: $(date)"
