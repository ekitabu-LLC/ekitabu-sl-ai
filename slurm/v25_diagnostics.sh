#!/bin/bash
#SBATCH --job-name=v25_diagnostics
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=testing
#SBATCH --output=slurm/v25_diag_%j.out
#SBATCH --error=slurm/v25_diag_%j.err

echo "=== V25 Diagnostics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

python run_v25_diagnostics.py

echo "End: $(date)"
echo "=== Done ==="
