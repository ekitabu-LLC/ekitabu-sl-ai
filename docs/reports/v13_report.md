# V13 Report - Robust to Sparse Hand Detection

## Version Summary

Introduced aggressive landmark dropout augmentation (30-70% per hand) and frame-level dropout to make the model robust to poor/sparse hand detection in real-world inference.

## Architecture

- **Model class:** `Model` (custom ST-GCN) -- identical architecture to v12
- **Type:** Spatial-Temporal Graph Convolutional Network
- **Blocks:** `Block` = GCN + BatchNorm + TCN + residual connection
- **Layers:** 6 blocks with channel progression `[3, 128, 128, 256, 256, 512, 512]`
- **Temporal downsampling:** stride 2 at layers 2 and 4
- **Temporal kernel size:** 9
- **Input features:** 3 channels (x, y, z) per node
- **Graph nodes:** 48 (21 left hand + 21 right hand + 6 pose joints)
- **Graph:** Same as v12 (hand edges + pose edges + 0.3 cross-hand link)
- **Classifier head:** Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
- **Pooling:** AdaptiveAvgPool2d(1)

## Parameters

Identical to v12 (~700K params per model). Same architecture, only augmentation and regularization changed.

## Key Changes from Previous Version (v12)

1. **Hand landmark dropout augmentation** (new):
   - 50% probability of applying (`hand_dropout_prob=0.5`)
   - Drops 30-70% of frames per hand (random rate per sample)
   - Applied independently to left and right hands (50% prob each)
2. **Frame-level dropout** (new):
   - 40% probability of applying (`frame_drop_prob=0.4`)
   - Drops 10-30% of frames (sets hand landmarks to zero)
   - Safety floor: always keeps at least 30% of frames
3. **Hand swap augmentation** (new):
   - 30% probability: swaps left and right hands + mirrors x-coordinates
   - Data augmentation that exploits hand symmetry
4. **Robust normalization** -- only normalizes hands with valid (non-zero) data, checks per-frame validity
5. **Clipping after normalization** -- `np.clip(h / max_val, -1, 1)` with threshold check
6. **Speed perturbation augmentation** (new):
   - 30% probability, speed factor 0.8-1.2x
   - Resamples temporal axis to simulate signing speed variation
7. **Stronger noise** -- std increased to 0.03 (from 0.02)
8. **Wider scale jitter** -- range [0.8, 1.2] (from [0.9, 1.1])
9. **Wider temporal shift** -- [-8, +8] frames (from [-5, +5])
10. **Higher noise probability** -- 40% (from 30%)
11. **Increased epochs** -- 200 (from 150)
12. **Increased patience** -- 40 (from 30)
13. **Increased dropout** -- 0.5 (from 0.4)
14. **Increased label smoothing** -- 0.2 (from 0.15)

## Training Configuration

| Parameter | Value | Change from v12 |
|-----------|-------|-----------------|
| Optimizer | AdamW | same |
| Learning rate | 1e-3 | same |
| Min LR | 1e-6 | same |
| Weight decay | 1e-4 | same |
| Batch size | 32 | same |
| Epochs | **200** | +50 |
| Patience (early stop) | **40** | +10 |
| Warmup epochs | 10 | same |
| Scheduler | Cosine annealing with warmup | same |
| Loss | CrossEntropyLoss + weights + label smoothing | same |
| Label smoothing | **0.2** | +0.05 |
| Dropout | **0.5** | +0.1 |
| Gradient clipping | max_norm=1.0 | same |

## Data Pipeline

- **Same landmark extraction as v12** (hands + pose = 48 nodes)
- **Augmentation pipeline (train only), applied in order:**
  1. Hand landmark dropout (50% prob, 30-70% frames zeroed per hand)
  2. Frame dropout (40% prob, 10-30% frames zeroed, min 30% kept)
  3. Hand swap (30% prob, swap L/R hands + mirror x)
  4. Robust per-hand normalization (only on valid frames)
  5. Pose normalization (mid-shoulder)
  6. Global max-abs normalization with clipping to [-1, 1]
  7. Temporal resampling to 90 frames
  8. Scale jitter (50% prob, [0.8, 1.2])
  9. Gaussian noise (40% prob, std=0.03)
  10. Temporal shift (30% prob, [-8, +8] frames)
  11. Speed perturbation (30% prob, 0.8-1.2x)
- **Input tensor shape:** `(C=3, T=90, N=48)`

## Results

No result JSON files found for v13. However, the v15 analysis report (which analyzes v15 results) notes that the aggressive dropout approach from v13/v14 was carried into v15 and was identified as a primary cause of poor number accuracy (48.9% in v15), because hand dropout destroys the fine finger position signals that distinguish number signs.

## Platform

- **Modal** (cloud GPU)
- **GPU:** A10G
- **Python:** 3.11
- **Timeout:** 21600s (6 hours)
- **Storage:** Modal Volume (`ksl-dataset-vol`)

## Lessons / Notes

- **Key hypothesis:** Real-world hand detection is unreliable/sparse, so training with aggressive dropout should improve generalization. This was later shown to be counterproductive for numbers.
- The v15 analysis conclusively demonstrated that hand dropout destroys discriminative signal for number classes, where recognition depends on fine static hand shapes (finger positions, spreads, curls)
- The hand swap augmentation is an interesting idea -- KSL signs should be recognizable regardless of hand dominance
- The robust normalization (checking for valid frames before centering) is a good defensive practice that persists into later versions
- Speed perturbation is a form of temporal data augmentation that simulates natural variation in signing speed
- The combination of high dropout (0.5), high label smoothing (0.2), and aggressive augmentation creates very heavy regularization -- likely too aggressive given the small dataset (~25 samples per class)
- Checkpoint now includes version tag (`"version": "v13"`) for tracking
