#!/bin/bash
#SBATCH --job-name=ksl-extract-rt
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/extract_rt_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/extract_rt_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e

echo "=========================================="
echo "Real Tester Landmark Pre-Extraction (32 cores)"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/$USER/ksl-dir-2

python extract_landmarks_real_testers.py --workers 32

echo ""
echo "Done: $(date)"
