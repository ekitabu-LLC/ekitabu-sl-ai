#!/bin/bash
#SBATCH --job-name=clean_data
#SBATCH --partition=atesting
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/clean_data_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/clean_data_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# Data Cleaning Pipeline — No GPU needed
# Interpolate missing hand frames, DBSCAN outlier cleaning
# ==============================================================================

set -e

echo "==========================================="
echo "KSL Data Cleaning Pipeline"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
cd "$PROJECT_DIR"

echo "=== Running Data Cleaning ==="
python clean_data.py \
    --bad-frame-threshold 0.70 \
    --dbscan-eps 0.15 \
    --dbscan-min-samples 3

echo ""
echo "==========================================="
echo "Data Cleaning Complete"
echo "End time: $(date)"
echo "==========================================="
