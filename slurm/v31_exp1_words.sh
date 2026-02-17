#!/bin/bash
#SBATCH --job-name=v31_e1w
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp1_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v31_exp1_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR="/scratch/alpine/$USER/tmp"
mkdir -p "$PROJECT_DIR/data/checkpoints/v31_exp1" "$TMPDIR"
cd "$PROJECT_DIR"

echo "V31 Exp 1 Words Training"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

python train_ksl_v31_exp1.py --model-type words --seed 42 \
  --checkpoint-dir data/checkpoints/v31_exp1

echo "Done: $(date)"
