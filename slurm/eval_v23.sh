#!/bin/bash
#SBATCH --job-name=ksl-eval-v23
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v23_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v23_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e

echo "=========================================="
echo "KSL v23 Real-Tester Evaluation - Multi-Stream - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo ""

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Set headless mode for OpenCV/MediaPipe (no display on compute nodes)
export DISPLAY=""
export MPLBACKEND=Agg

cd /scratch/alpine/$USER/ksl-dir-2

python evaluate_real_testers_v23.py && EXIT_CODE=0 || EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
