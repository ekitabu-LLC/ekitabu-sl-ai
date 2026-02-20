#!/bin/bash
#SBATCH --account=ucb765_asc1
#SBATCH --partition=atesting_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=v30_ens_abn
#SBATCH --output=slurm/v30_ens_abn_%j.out
#SBATCH --error=slurm/v30_ens_abn_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hama5612@colorado.edu
#SBATCH --qos=testing

module purge
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
cd /scratch/alpine/hama5612/ksl-dir-2

echo "=== Ensemble + Alpha-BN Combined ==="
echo "Testing alpha=0.3, 0.5, 0.7 with ensemble"

python -c "
import sys, os, json, copy
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

# Import from phase1 eval
from evaluate_real_testers_v30_phase1 import (
    discover_test_videos, preextract_test_data, adapt_bn_stats,
    save_bn_stats, compute_target_bn_stats, apply_alpha_bn,
    ensemble_predict, evaluate_method,
    load_v27_model, load_v28_stream_models, load_v29_model,
    NUMBER_CLASSES, WORD_CLASSES,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = '/scratch/alpine/hama5612/ksl-alpha/data/real_testers'
ckpt_dir = '/scratch/alpine/hama5612/ksl-dir-2/data/checkpoints'

videos = discover_test_videos(base_dir)
print(f'Found {len(videos)} test videos')

for alpha in [0.3, 0.5, 0.7]:
    print(f'\n{\"=\"*60}')
    print(f'ENSEMBLE + ALPHA-BN (alpha={alpha})')
    print(f'{\"=\"*60}')
    
    for cat_name, classes in [('numbers', NUMBER_CLASSES), ('words', WORD_CLASSES)]:
        # Strip category from 4-tuples to 3-tuples (path, name, signer)
        cat_videos = [(v[0], v[1], v[3]) for v in videos if v[2] == cat_name]
        if not cat_videos:
            continue
        
        print(f'\n--- {cat_name.upper()} ({len(cat_videos)} videos) ---')
        
        # Load models
        v27_ckpt = os.path.join(ckpt_dir, f'v27_{cat_name}', 'best_model.pt')
        v28_ckpt_dir_cat = os.path.join(ckpt_dir, f'v28_{cat_name}')
        v29_ckpt = os.path.join(ckpt_dir, f'v29_{cat_name}', 'best_model.pt')
        
        v27_model = load_v27_model(v27_ckpt, classes, device) if os.path.exists(v27_ckpt) else None
        v28_models, v28_fw = load_v28_stream_models(v28_ckpt_dir_cat, classes, device) if os.path.isdir(v28_ckpt_dir_cat) else (None, None)
        v29_model = load_v29_model(v29_ckpt, classes, device) if os.path.exists(v29_ckpt) else None
        
        # Standard AdaBN on v27, v29
        if v27_model:
            v27_data = preextract_test_data(cat_videos, preprocess_fn='v27')
            adapt_bn_stats(v27_model, v27_data, device)
        if v29_model:
            v29_data = preextract_test_data(cat_videos, preprocess_fn='v27')
            adapt_bn_stats(v29_model, v29_data, device)
        
        # Alpha-BN on v28 streams
        if v28_models:
            stream_data = preextract_test_data(cat_videos, preprocess_fn='multistream')
            for sname, smodel in v28_models.items():
                source_stats = save_bn_stats(smodel)
                target_stats = compute_target_bn_stats(smodel, stream_data, device,
                    is_multistream=True, stream_name=sname)
                apply_alpha_bn(smodel, source_stats, target_stats, alpha)
        
        # Ensemble predict
        def predict_fn(raw, true_class):
            pred, conf, _ = ensemble_predict(raw, v27_model, v28_models, v28_fw, v29_model, device, classes)
            if pred is None: return None
            return pred, conf
        
        results = evaluate_method(f'ensemble+alpha_bn({alpha})', cat_videos, predict_fn, classes)
        
        # Cleanup
        del v27_model, v28_models, v29_model
        torch.cuda.empty_cache()
"

echo "=== Done ==="
