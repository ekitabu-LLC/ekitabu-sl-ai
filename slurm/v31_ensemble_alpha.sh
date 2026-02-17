#!/bin/bash
#SBATCH --job-name=v31_ens_a
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/v31_ens_alpha_%j.out
#SBATCH --error=slurm/v31_ens_alpha_%j.err
#SBATCH --account=ucb765_asc1

echo "V31 Ensemble - Alpha Test Set"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/hama5612/ksl-dir-2

python run_ensemble_alpha_test.py --alpha 0.3

echo "Done: $(date)"
