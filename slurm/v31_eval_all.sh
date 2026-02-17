#!/bin/bash
#SBATCH --job-name=v31_eval
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v31_eval_all_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v31_eval_all_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

set -e
module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/$USER/ksl-dir-2

echo "=========================================="
echo "V31 All Experiments Evaluation"
echo "=========================================="
nvidia-smi
echo ""

for EXP in 1 2 3 4 5; do
    CKPT_DIR="data/checkpoints/v31_exp${EXP}"
    if [ -d "$CKPT_DIR" ]; then
        echo ""
        echo "############################################################"
        echo "# EXP ${EXP}"
        echo "############################################################"
        echo ""
        python evaluate_v31_ablation.py --exp ${EXP} --checkpoint-dir ${CKPT_DIR} --category both || echo "Exp ${EXP} FAILED"
    else
        echo "Skipping exp ${EXP}: no checkpoint dir"
    fi
done

echo ""
echo "=========================================="
echo "V31 Evaluation Complete: $(date)"
echo "=========================================="
