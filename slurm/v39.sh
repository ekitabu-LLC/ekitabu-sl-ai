#!/bin/bash
#SBATCH --job-name=v39
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v39_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v39_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/$USER/ksl-dir-2

echo "=========================================="
echo "V39: SupCon on Cleaned Data (Numbers)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo ""

python train_ksl_v39.py --model-type numbers --seed 42

echo ""
echo "Done: $(date)"
