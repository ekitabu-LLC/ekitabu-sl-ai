#!/bin/bash
#SBATCH --job-name=opt_w_v2
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/opt_w_v2_%j.out
#SBATCH --error=slurm/opt_w_v2_%j.err

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python optimize_ensemble_weights_v2.py
