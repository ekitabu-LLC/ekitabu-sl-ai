#!/bin/bash
#SBATCH --job-name=ksl_v44_expr2_words
#SBATCH --account=ucb765_asc1
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/v44_expr2_words_%j.out
#SBATCH --error=slurm/v44_expr2_words_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python train_ksl_v44_expr2.py \
    --model-type words \
    --lambda-max 0.1 \
    --seed 42

echo "Job finished: $(date)"
