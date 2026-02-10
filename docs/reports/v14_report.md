# V14 Report - Extreme Dropout for Domain Shift

## Version Summary

Pushed hand landmark dropout to extreme levels (50-90% per frame) with complete hand drop capability, hypothesizing that this would bridge the domain shift between training and real-world test data where hand detection is sparse.

## Architecture

- **Model class:** `Model` (custom ST-GCN) -- identical architecture to v12/v13
- **Type:** Spatial-Temporal Graph Convolutional Network
- **Blocks:** `Block` = GCN + BatchNorm + TCN + residual connection
- **Layers:** 6 blocks with channel progression `[3, 128, 128, 256, 256, 512, 512]`
- **Temporal downsampling:** stride 2 at layers 2 and 4
- **Temporal kernel size:** 9
- **Input features:** 3 channels (x, y, z) per node
- **Graph nodes:** 48 (21 left hand + 21 right hand + 6 pose joints)
- **Graph:** Same as v12/v13 (hand edges + pose edges + 0.3 cross-hand link)
- **Classifier head:** Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
- **Pooling:** AdaptiveAvgPool2d(1)

## Parameters

Identical to v12/v13 (~700K params per model).

## Key Changes from Previous Version (v13)

1. **Much more aggressive hand dropout:**
   - Probability increased: 70% (from 50%)
   - Dropout rate range: **50-90%** (from 30-70%)
   - Per-hand application probability: 60% (from 50%)
2. **Complete hand drop** (new):
   - 20% probability of dropping an entire hand for all frames
   - Randomly selects left or right hand (50/50)
   - Simulates scenarios where one hand is completely undetected
3. **Removed frame-level dropout** -- no longer drops entire frames of data (was in v13)
4. **Removed speed perturbation augmentation** -- no temporal speed changes (was in v13)
5. **Downgraded GPU** -- uses T4 instead of A10G
6. **Removed defaultdict import** -- simplified validation loop (no per-class tracking during training)

## Training Configuration

| Parameter | Value | Change from v13 |
|-----------|-------|-----------------|
| Optimizer | AdamW | same |
| Learning rate | 1e-3 | same |
| Min LR | 1e-6 | same |
| Weight decay | 1e-4 | same |
| Batch size | 32 | same |
| Epochs | 200 | same |
| Patience (early stop) | 40 | same |
| Warmup epochs | 10 | same |
| Scheduler | Cosine annealing with warmup | same |
| Loss | CrossEntropyLoss + weights + label smoothing | same |
| Label smoothing | 0.2 | same |
| Dropout | 0.5 | same |
| Gradient clipping | max_norm=1.0 | same |

## Data Pipeline

- **Same landmark extraction as v12/v13** (hands + pose = 48 nodes)
- **Augmentation pipeline (train only), applied in order:**
  1. Hand landmark dropout (70% prob, **50-90%** frames zeroed per hand, 60% per-hand prob)
  2. Complete hand drop (20% prob, zeros out entire L or R hand for all frames)
  3. Hand swap (30% prob, swap L/R hands + mirror x)
  4. Robust per-hand normalization (only on valid frames)
  5. Pose normalization (mid-shoulder)
  6. Global max-abs normalization with clipping to [-1, 1]
  7. Temporal resampling to 90 frames
  8. Scale jitter (50% prob, [0.8, 1.2])
  9. Gaussian noise (40% prob, std=0.03)
  10. Temporal shift (30% prob, [-8, +8] frames)
- **Removed from v13:** Frame dropout, speed perturbation
- **Input tensor shape:** `(C=3, T=90, N=48)`

## Results

No result JSON files found for v14. The v15 analysis report identifies the extreme dropout levels from v14 (carried into v15) as a primary cause of the number model's poor performance:
- At 50-90% dropout, the majority of hand landmarks are zeroed out per frame
- Number signs are distinguished by fine finger positions that are destroyed by this level of dropout
- The complete hand drop (20%) further removes discriminative signal
- Words are less affected because they rely on larger arm movements captured by the pose joints

## Platform

- **Modal** (cloud GPU)
- **GPU:** T4 (downgraded from A10G in v11-v13)
- **Python:** 3.11
- **Timeout:** 21600s (6 hours)
- **Storage:** Modal Volume (`ksl-dataset-vol`)

## Lessons / Notes

- **Extreme dropout was the wrong direction** -- later analysis (v15 report) proved that this approach destroyed the signal needed for number classification
- The shift from A10G to T4 GPU suggests cost optimization, as the model architecture is small enough for a T4
- The complete hand drop augmentation is conceptually interesting (one hand may be occluded in real-world video) but at 20% probability combined with 70% hand dropout, the model sees very corrupted inputs during training
- Frame dropout and speed perturbation were removed, likely to isolate the effect of hand dropout
- The validation loop was simplified to remove per-class tracking during training, only computing overall accuracy per epoch
- This version represents the peak of the "dropout as regularization" hypothesis before it was identified as harmful in the v15 analysis
- The v15 analysis report recommended drastically reducing hand dropout for numbers (to 15% prob with 5-20% rate) as the single biggest fix
- **Net assessment:** v14 doubled down on an approach (aggressive dropout) that was ultimately counterproductive. The lesson was that domain-shift robustness cannot be achieved by destroying the discriminative signal -- it requires other approaches (signer invariance, better features, more diverse training data)
