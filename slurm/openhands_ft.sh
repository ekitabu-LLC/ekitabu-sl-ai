#!/bin/bash
#SBATCH --job-name=oh_ft
#SBATCH --account=ucb765_asc1
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/openhands_ft_%j.out
#SBATCH --error=slurm/openhands_ft_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl

cd /scratch/alpine/hama5612/ksl-dir-2

# Download pretrained checkpoint if not present
CKPT_DIR="data/checkpoints/openhands_pretrained"
if [ ! -f "$CKPT_DIR/wlasl_slgcn.zip" ]; then
    echo "Downloading WLASL SL-GCN pretrained checkpoint..."
    mkdir -p "$CKPT_DIR"
    wget -O "$CKPT_DIR/wlasl_slgcn.zip" \
        "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/wlasl_slgcn.zip" \
        2>&1
    echo "Download complete: $(ls -la $CKPT_DIR/wlasl_slgcn.zip)"
fi

python train_ksl_openhands.py --model-type both

echo "Job finished: $(date)"
