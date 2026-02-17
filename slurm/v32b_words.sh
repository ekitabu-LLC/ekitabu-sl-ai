#!/bin/bash
#SBATCH --job-name=v32b_wrd
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/v32b_words_%j.out
#SBATCH --error=slurm/v32b_words_%j.err
#SBATCH --account=ucb765_asc1

echo "V32b Words - GroupNorm + SupCon (bs=32, 300 epochs)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/hama5612/ksl-dir-2

python train_ksl_v32.py --model-type words --seed 42 --epochs 300

echo "Done: $(date)"
