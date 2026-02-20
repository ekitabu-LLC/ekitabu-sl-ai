#!/bin/bash
#SBATCH --job-name=eval_masa
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/eval_masa_%j.out
#SBATCH --error=slurm/eval_masa_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python evaluate_masa.py --category both

echo "Job finished: $(date)"
