#!/bin/bash
#SBATCH --job-name=v31_exp3
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp3_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp3_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/$USER/ksl-dir-2

echo "=========================================="
echo "V31 Exp 3: Joint Mixing (Numbers)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo ""

python train_ksl_v31_exp3.py --model-type numbers --seed 42 \
  --checkpoint-dir data/checkpoints/v31_exp3

echo ""
echo "Done: $(date)"
