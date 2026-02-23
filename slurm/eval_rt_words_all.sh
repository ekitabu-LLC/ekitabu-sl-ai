#!/bin/bash
#SBATCH --job-name=eval_rt_words
#SBATCH --output=/scratch/alpine/hama5612/ksl-dir-2/slurm/eval_rt_words_%j.out
#SBATCH --error=/scratch/alpine/hama5612/ksl-dir-2/slurm/eval_rt_words_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# CPU-only evaluation — no GPU needed
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "Real Testers Words — All Models (v22-v43)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "=========================================="

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python evaluate_real_testers_words_all.py \
    --rt-base   /scratch/alpine/hama5612/ksl-alpha \
    --ckpt-base data/checkpoints \
    --results-dir data/results/real_testers_words_all

echo ""
echo "Finished: $(date)"
