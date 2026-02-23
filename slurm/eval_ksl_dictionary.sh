#!/bin/bash
#SBATCH --job-name=eval_ksl_dict
#SBATCH --output=/scratch/alpine/hama5612/ksl-dir-2/slurm/eval_ksl_dict_%j.out
#SBATCH --error=/scratch/alpine/hama5612/ksl-dir-2/slurm/eval_ksl_dict_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# CPU-only evaluation — no GPU needed
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "KSL Dictionary Evaluation — v22 to v43"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "=========================================="

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python evaluate_ksl_dictionary.py \
    --videos-dir data/ksl_dictionary_videos \
    --ckpt-base  data/checkpoints \
    --results-dir data/results/ksl_dictionary

echo ""
echo "Finished: $(date)"
