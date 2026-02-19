#!/bin/bash
#SBATCH --job-name=opt_clean_v3
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/opt_clean_v3_%j.out
#SBATCH --error=slurm/opt_clean_v3_%j.err

echo "Job started: $(date)"
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/hama5612/ksl-dir-2

# min_weight=0.1 with step=0.1 → each model gets at least 10% weight
# (enforces all 6 models contribute, preventing catastrophic exclusion)
python optimize_ensemble_clean.py --step 0.1 --min_weight 0.1

echo "Job finished: $(date)"
