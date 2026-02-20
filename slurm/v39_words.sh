#!/bin/bash
#SBATCH --job-name=v39w
#SBATCH --partition=al40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v39_words_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v39_words_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export TMPDIR="/scratch/alpine/$USER/tmp"
mkdir -p "$PROJECT_DIR/data/checkpoints/v39" "$TMPDIR"
cd "$PROJECT_DIR"

echo "V39: SupCon on Cleaned Data (Words)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi

python train_ksl_v39.py --model-type words --seed 42

echo "Done: $(date)"
