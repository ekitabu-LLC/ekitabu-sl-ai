#!/bin/bash
#SBATCH --job-name=v30_ablation
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/v30_ablation_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/v30_ablation_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V30 Ablation Study Script on Alpine
# ==============================================================================
# Tests each Phase 2 improvement independently (numbers only for speed)
# 8 runs: baseline + 6 individual features + all combined
# Partition: atesting_a100 (NVIDIA A100 MIG)
# Duration: Up to 4 hours (~30 min x 8 runs)
# Usage: sbatch slurm/v30_ablation.sh
# ==============================================================================

set -e

echo "=========================================="
echo "KSL V30 Ablation Study - Alpine HPC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $SLURM_GPUS"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Memory:    $SLURM_MEM_PER_NODE MB"
echo "Start:     $(date)"
echo ""

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ------------------------------------------------------------------------------
# Directory Setup
# ------------------------------------------------------------------------------

export PROJECT_DIR="/scratch/alpine/$USER/ksl-dir-2"
export DATA_DIR="$PROJECT_DIR/data"
export CKPT_DIR="$DATA_DIR/checkpoints"
export TMPDIR="/scratch/alpine/$USER/tmp"

mkdir -p "$CKPT_DIR" "$TMPDIR" "$DATA_DIR/results/v30/ablation"

echo "Project:     $PROJECT_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

# ------------------------------------------------------------------------------
# GPU Info
# ------------------------------------------------------------------------------

echo "GPU Information:"
nvidia-smi
echo ""

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# Ablation: All runs use --model-type numbers only (faster)
# Base flags to disable all new features
# ------------------------------------------------------------------------------

BASE_OFF="--no-mixstyle --no-dropgraph --no-confused-pairs --no-r-drop --no-swad"
TASK="--model-type numbers --seed 42"
RUN=0
FAIL=0

run_ablation() {
    local name="$1"
    local flags="$2"
    RUN=$((RUN + 1))

    echo ""
    echo "=========================================="
    echo "Ablation $RUN/8: $name"
    echo "Flags: $flags"
    echo "Start: $(date)"
    echo "=========================================="
    echo ""

    if python train_ksl_v30.py $TASK $flags --ablation-name "$name"; then
        echo ">>> $name: SUCCESS"
    else
        echo ">>> $name: FAILED"
        FAIL=$((FAIL + 1))
    fi
}

# ------------------------------------------------------------------------------
# Run 1: Baseline (all new features OFF)
# ------------------------------------------------------------------------------
run_ablation "baseline" "$BASE_OFF"

# ------------------------------------------------------------------------------
# Run 2: +MixStyle only
# ------------------------------------------------------------------------------
run_ablation "mixstyle_only" "--mixstyle --no-dropgraph --no-confused-pairs --no-r-drop --no-swad"

# ------------------------------------------------------------------------------
# Run 3: +DropGraph only
# ------------------------------------------------------------------------------
run_ablation "dropgraph_only" "--no-mixstyle --dropgraph --no-confused-pairs --no-r-drop --no-swad"

# ------------------------------------------------------------------------------
# Run 4: +Confused-pairs only
# ------------------------------------------------------------------------------
run_ablation "confused_pairs_only" "--no-mixstyle --no-dropgraph --confused-pairs --no-r-drop --no-swad"

# ------------------------------------------------------------------------------
# Run 5: +Label smoothing only
# ------------------------------------------------------------------------------
run_ablation "label_smooth_only" "$BASE_OFF --label-smooth 0.1"

# ------------------------------------------------------------------------------
# Run 6: +R-Drop only
# ------------------------------------------------------------------------------
run_ablation "r_drop_only" "--no-mixstyle --no-dropgraph --no-confused-pairs --r-drop 1.0 --no-swad"

# ------------------------------------------------------------------------------
# Run 7: +SWAD only
# ------------------------------------------------------------------------------
run_ablation "swad_only" "--no-mixstyle --no-dropgraph --no-confused-pairs --no-r-drop --swad"

# ------------------------------------------------------------------------------
# Run 8: All combined (default)
# ------------------------------------------------------------------------------
run_ablation "all_combined" "--mixstyle --dropgraph --confused-pairs --label-smooth 0.1 --r-drop 1.0 --swad --aggressive-aug --skeleton-noise"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "V30 Ablation Study Complete"
echo "End time: $(date)"
echo "=========================================="
echo "Total runs:  $RUN"
echo "Successful:  $((RUN - FAIL))"
echo "Failed:      $FAIL"
echo "Results:     $DATA_DIR/results/v30/ablation/"

if [ $FAIL -eq 0 ]; then
    echo "All ablation runs completed successfully!"
    exit 0
else
    echo "Some runs failed! Check error log:"
    echo "  /scratch/alpine/$USER/ksl-dir-2/slurm/v30_ablation_${SLURM_JOB_ID}.err"
    exit 1
fi
