#!/bin/bash
#SBATCH --job-name=v44_expr1
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/eval_v44_expr1_%j.out
#SBATCH --error=slurm/eval_v44_expr1_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python evaluate_v44_expr1.py \
    --category both \
    --models exp1 exp5 v41 v43

echo "Job finished: $(date)"
