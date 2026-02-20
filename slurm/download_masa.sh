#!/bin/bash
#SBATCH --job-name=dl_masa
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/download_masa_%j.out
#SBATCH --error=slurm/download_masa_%j.err

echo "Job started: $(date)"

CKPT_DIR="/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints/masa_pretrained"
mkdir -p "$CKPT_DIR"

# MASA pretrained checkpoint (161K SLR samples)
# From: https://rec.ustc.edu.cn/share/d8766290-0475-11ef-a181-0b1056e2faed
echo "Downloading MASA pretrained checkpoint..."
wget -O "$CKPT_DIR/checkpoint.pth" \
    "https://rec.ustc.edu.cn/share/d8766290-0475-11ef-a181-0b1056e2faed" \
    --no-check-certificate 2>&1

if [ -f "$CKPT_DIR/checkpoint.pth" ]; then
    SIZE=$(stat --format=%s "$CKPT_DIR/checkpoint.pth" 2>/dev/null || stat -f%z "$CKPT_DIR/checkpoint.pth" 2>/dev/null)
    echo "Downloaded: $CKPT_DIR/checkpoint.pth ($SIZE bytes)"

    # Verify it's a valid PyTorch checkpoint
    source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
    conda activate ksl
    python -c "
import torch
ckpt = torch.load('$CKPT_DIR/checkpoint.pth', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    print('Keys:', list(ckpt.keys())[:10])
    if 'state_dict' in ckpt:
        print('state_dict keys:', len(ckpt['state_dict']))
        for k in sorted(ckpt['state_dict'].keys())[:20]:
            print(f'  {k}: {ckpt[\"state_dict\"][k].shape}')
    elif 'model' in ckpt:
        print('model keys:', len(ckpt['model']))
        for k in sorted(ckpt['model'].keys())[:20]:
            print(f'  {k}: {ckpt[\"model\"][k].shape}')
    else:
        print('First 20 keys:')
        for k in sorted(ckpt.keys())[:20]:
            if hasattr(ckpt[k], 'shape'):
                print(f'  {k}: {ckpt[k].shape}')
            else:
                print(f'  {k}: {type(ckpt[k]).__name__}')
else:
    print('Checkpoint type:', type(ckpt).__name__)
print('Checkpoint verified OK')
" 2>&1
else
    echo "ERROR: Download failed"
fi

echo "Job finished: $(date)"
