#!/bin/bash
#SBATCH --job-name=ksl_videomae_v2_words
#SBATCH --account=ucb765_asc1
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/videomae_v2_words_%j.out
#SBATCH --error=slurm/videomae_v2_words_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python train_ksl_videomae_v2.py \
    --model-type words \
    --seed 42 \
    --batch-size 6 \
    --epochs 30 \
    --lr 1e-5 \
    --val-signer 5

echo "Job finished: $(date)"
