#!/bin/bash
#SBATCH --job-name=eval_v32b
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/eval_v32b_%j.out
#SBATCH --error=slurm/eval_v32b_%j.err
#SBATCH --account=ucb765_asc1

echo "V32b Evaluation - GroupNorm + SupCon bs=32 (Signer-Agnostic)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/hama5612/ksl-dir-2

python evaluate_v32.py --category both --checkpoint-dir data/checkpoints/v32b

echo "Done: $(date)"
