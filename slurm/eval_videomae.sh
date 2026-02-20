#!/bin/bash
#SBATCH --job-name=ksl_eval_videomae
#SBATCH --account=ucb765_asc1
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/eval_videomae_%j.out
#SBATCH --error=slurm/eval_videomae_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

# Evaluate v1 (5-signer) or v2 (10-signer) by setting VIDEOMAE_CKPT_DIR:
#   v1: data/checkpoints/videomae   (default)
#   v2: data/checkpoints/videomae_v2
VIDEOMAE_CKPT_DIR="${VIDEOMAE_CKPT_DIR:-data/checkpoints/videomae}"

python evaluate_videomae.py --category both --videomae-ckpt-dir "$VIDEOMAE_CKPT_DIR"

echo "Job finished: $(date)"
