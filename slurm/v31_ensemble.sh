#!/bin/bash
#SBATCH --job-name=v31_ens
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v31_ensemble_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v31_ensemble_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/$USER/ksl-dir-2

echo "V31 Ensemble Evaluation"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

python run_ensemble_v31.py --alpha 0.3

echo "Done: $(date)"
