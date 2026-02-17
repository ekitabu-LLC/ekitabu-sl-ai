#!/bin/bash
#SBATCH --job-name=eval_v35
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v35_%j.out
#SBATCH --error=/scratch/alpine/%u/ksl-dir-2/slurm/eval_v35_%j.err
#SBATCH --account=ucb765_asc1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hama5612@colorado.edu

# ==============================================================================
# V35 Evaluation on Real Testers
# Tests both regular model and EMA model
# GroupNorm — no BN adaptation needed
# ==============================================================================

set -e

echo "==========================================="
echo "V35 EMA+RSC Evaluation - Real Testers"
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

nvidia-smi
echo ""

# Evaluate regular best model
echo "=== Regular Model ==="
python evaluate_v32.py --category both \
  --checkpoint-dir data/checkpoints/v35

# Evaluate EMA model (copy ema checkpoints to temp dir)
echo ""
echo "=== EMA Model ==="
# Create temp dir with EMA checkpoints as best_model.pt
EMA_DIR="data/checkpoints/v35_ema_eval"
mkdir -p "$EMA_DIR/numbers/joint" "$EMA_DIR/numbers/bone" "$EMA_DIR/numbers/velocity"
mkdir -p "$EMA_DIR/words/joint" "$EMA_DIR/words/bone" "$EMA_DIR/words/velocity"

for cat in numbers words; do
  for stream in joint bone velocity; do
    ema_src="data/checkpoints/v35/$cat/$stream/best_ema_model.pt"
    if [ -f "$ema_src" ]; then
      cp "$ema_src" "$EMA_DIR/$cat/$stream/best_model.pt"
    fi
  done
  # Copy fusion weights
  fw_src="data/checkpoints/v35/$cat/fusion_weights.json"
  if [ -f "$fw_src" ]; then
    cp "$fw_src" "$EMA_DIR/$cat/fusion_weights.json"
  fi
done

python evaluate_v32.py --category both \
  --checkpoint-dir "$EMA_DIR"

# Cleanup
rm -rf "$EMA_DIR"

echo ""
echo "==========================================="
echo "V35 Evaluation Complete"
echo "End time: $(date)"
echo "==========================================="
