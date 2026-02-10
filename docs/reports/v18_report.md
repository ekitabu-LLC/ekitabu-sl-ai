# V18 Model Report

**Date:** 2026-02-09
**Script:** `train_ksl_v18.py`
**Result:** `data/results/v18_both_20260209_035539.json`

---

## Version Summary

Bug-fix release addressing multiple issues found in a v17 pipeline review. Added velocity features (in_channels 3->6), AttentionPool replacing AdaptiveAvgPool, separate hand/pose normalization, fixed SWA activation, re-added anti-attractor for numbers, moderated numbers augmentation, and added reproducibility seed.

---

## Architecture

| Property | Numbers | Words |
|----------|---------|-------|
| Model class | `KSLGraphNet` | `KSLGraphNet` |
| Type | ST-GCN + AttentionPool + anti-attractor | ST-GCN + AttentionPool |
| Layers | 6 | 6 |
| Channel progression | 6->128->128->256->256->512->512 | 6->128->128->256->256->512->512 |
| Hidden dim | 128 | 128 |
| Input features | **6 channels** (xyz + velocity xyz) | **6 channels** (xyz + velocity xyz) |
| Nodes | 48 | 48 |
| Pooling | **AttentionPool** (2 heads) | **AttentionPool** (2 heads) |
| Anti-attractor | Yes (class 54, weight=0.15) | No |
| Classifier input | 1024 (2 x 512 from attention heads) | 1024 |
| Parameters | **7,342,418** | **7,276,753** |

### AttentionPool

New pooling mechanism replacing AdaptiveAvgPool2d:
- First averages over node dimension: (B, C, T, N) -> (B, C, T)
- Then applies 2 attention heads, each: Linear(C, C/2) -> Tanh -> Linear(C/2, 1) -> softmax over T
- Output: concatenation of 2 weighted sums = (B, 2*C)
- Allows the model to learn which temporal frames are most important

### Velocity Features

Frame-difference velocity computed after all spatial preprocessing:
- `velocity[t] = position[t] - position[t-1]` (zero for first frame)
- Concatenated as 3 additional channels per node: (T, 48, 3) -> (T, 48, 6)
- Captures motion dynamics in addition to static pose

---

## Key Changes from V17

### Bug Fixes (11 fixes)

1. **Separate normalization for hands vs pose** -- previously, hands and pose were jointly normalized by global max. Since hands have much larger range than pose landmarks, this squashed pose features to ~1/5 of their effective range.
2. **SWA swa_start_frac 0.8->0.4** -- SWA never activated for numbers in v17 because early stopping (epoch 143) triggered before SWA start (epoch 240). Now starts at epoch 120.
3. **SWA validates with BASE model** for early stopping decisions. SWA model has stale BatchNorm statistics during training, causing artificially low val accuracy which triggered premature early stopping.
4. **Anti-attractor re-added** for numbers (class 54, weight=0.15). Removed in v17 simplification but v16 showed it helped class 444 specifically (0%->45%).
5. **Removed class weighting / WeightedRandomSampler** -- data is balanced at 25 samples/class, so inverse-frequency weighting was unnecessary and may have distorted gradients.
6. **Numbers augmentation moderated**: hand dropout 50%->30% (range 10-40% vs 30-70%), complete hand drop 10%->5%, noise std 0.015->0.01, noise prob 40%->30%
7. **weight_decay equalized**: numbers 0.003->0.001 (matching words)
8. **label_smoothing reduced** for numbers: 0.15->0.1
9. **Patience increased**: numbers 60->80, words 40->50
10. **Hand swap also flips pose X coordinates** -- previously only hand X was flipped during swap, creating inconsistent spatial representation
11. **Temporal shift uses pad+slice** instead of np.roll -- np.roll wraps frames from end to beginning, creating discontinuities

### New Features (4 additions)

1. **Velocity features**: in_channels 3->6 (xyz + vxyz), frame-difference velocity
2. **AttentionPool**: 2-head attention pooling over temporal dimension
3. **Confusion matrix logging**: full confusion matrix printed for numbers evaluation
4. **Reproducibility seed**: `--seed 42` (sets random, numpy, torch, cudnn)

---

## Training Configuration

### Numbers

| Parameter | Value | Changed from v17 |
|-----------|-------|-------------------|
| Optimizer | AdamW | - |
| Learning rate | 5e-4 | - |
| Weight decay | 0.001 | was 0.003 |
| Scheduler | Cosine + warmup, then SWA | - |
| Batch size | 32 | - |
| Max epochs | 300 | - |
| Patience | 80 | was 60 |
| Loss | CE (smoothing=0.1) + anti-attractor BCE (0.15) | was CE only (smoothing=0.15) |
| Dropout | 0.4 | - |
| Sampler | **shuffle=True** (no WeightedRandomSampler) | was WeightedRandom |
| SWA start | Epoch 120 (40% of 300) | was 240 (80%) |
| Seed | 42 | New |

### Words

| Parameter | Value | Changed from v17 |
|-----------|-------|-------------------|
| Weight decay | 0.001 | - |
| Patience | 50 | was 40 |
| Label smoothing | 0.15 | - |
| Sampler | **shuffle=True** | was WeightedRandom |
| SWA start | Epoch 80 (40% of 200) | was 160 (80%) |

### Augmentation

| Augmentation | Numbers | Words | Notes |
|-------------|---------|-------|-------|
| Hand dropout | 30% (range 10-40%) | 50% (range 30-70%) | Numbers moderated |
| Complete hand drop | 5% | 10% | Numbers moderated |
| Hand swap | 30% + **pose X flip** | 30% + **pose X flip** | Bug fix |
| Scale jitter | 50% (0.8-1.2x) | 50% (0.8-1.2x) | - |
| Gaussian noise | 30% (std=0.01) | 40% (std=0.015) | Numbers moderated |
| Temporal shift | 30% (**pad+slice**) | 30% (**pad+slice**) | Bug fix |

---

## Data Pipeline

- **Landmarks:** 48 nodes, now 6 channels (xyz + velocity)
- **Normalization (FIXED):** Hands and pose normalized separately -- hand data divided by hand max, pose data divided by pose max, both clipped to [-1, 1]
- **Velocity:** Computed after spatial preprocessing as frame differences
- **Temporal:** Linear interpolation / zero-pad to 90 frames
- **Sampler:** Simple shuffle (no class weighting, data is balanced at ~25/class)

---

## Results

| Split | Val Accuracy | Best Epoch | Parameters | SWA Used |
|-------|-------------|------------|------------|----------|
| Numbers | **53.6%** | 124/300 | 7,342,418 | No |
| Words | **87.5%** | 41/200 | 7,276,753 | No |
| **Combined** | **70.6%** | - | - | - |
| **Total time** | **11.2 min** | | | |

### Per-Class Numbers

| Class | V17 | V18 | Delta |
|-------|-----|-----|-------|
| 17 | 96% | **100%** | +4 |
| 268 | 65% | **95%** | +30 |
| 91 | 64% | **92%** | +28 |
| 48 | 60% | **84%** | +24 |
| 54 | 84% | **84%** | 0 |
| 9 | 76% | **72%** | -4 |
| 89 | 16% | **52%** | +36 |
| 66 | 76% | **48%** | -28 |
| 100 | 48% | **44%** | -4 |
| 125 | 40% | **36%** | -4 |
| 73 | 68% | **36%** | -32 |
| 388 | 70% | **25%** | -45 |
| 444 | 10% | **20%** | +10 |
| 22 | 52% | **12%** | -40 |
| 35 | 0% | **0%** | 0 |

### Per-Class Words

| Class | V17 | V18 | Delta |
|-------|-----|-----|-------|
| Colour | 100% | **100%** | 0 |
| Sweater | 56% | **100%** | +44 |
| Teach | 100% | **100%** | 0 |
| Ugali | 84% | **100%** | +16 |
| Apple | 84% | **96%** | +12 |
| Friend | 96% | **96%** | 0 |
| Gift | 92% | **96%** | +4 |
| Monday | 96% | **96%** | 0 |
| Tomatoes | 84% | **92%** | +8 |
| Picture | 96% | **84%** | -12 |
| Tortoise | 96% | **80%** | -16 |
| Twin | 92% | **80%** | -12 |
| Agreement | 100% | **76%** | -24 |
| Proud | 96% | **68%** | -28 |
| Market | 79.2% | **50.0%** | -29.2 |

---

## Platform

| Property | Value |
|----------|-------|
| Cluster | Alpine HPC (CU Boulder) |
| GPU | NVIDIA A100-PCIE-40GB MIG 3g.20gb |
| Device | CUDA |
| Total time | 669.3 seconds (11.2 minutes) |
| Timestamp | 2026-02-09 03:55:39 |
| Seed | 42 |

---

## Lessons / Notes

1. **Numbers slightly regressed** (55.3% -> 53.6%, -1.7pp) despite 11 bug fixes. The redistribution pattern continued: some classes improved dramatically (268: +30, 91: +28, 89: +36) while others collapsed (388: -45, 22: -40, 73: -32). This suggests the underlying problem is data-limited, not architecture-limited.

2. **Words regressed more significantly** (89.9% -> 87.5%, -2.4pp). The separate hand/pose normalization and velocity features may have disrupted the learned representations. Market dropped to 50%, Agreement to 76%, Proud to 68%.

3. **SWA did not help.** Despite fixing swa_start_frac to 0.4 (activating at epoch 120), best models were saved before SWA contributions. The `swa_used=false` field confirms SWA did not produce the best checkpoint for either split.

4. **Words early-stopped very early** (epoch 41/200), suggesting the velocity features + attention pooling made training unstable or the separate normalization changed the loss landscape significantly.

5. **Velocity features increased parameters** slightly (6.95M -> 7.28-7.34M) due to the larger input channel dimension (6 vs 3) and the AttentionPool having 2*C input to the classifier.

6. **Class 35 remains at 0%** across all four versions (v15-v18). This class appears fundamentally inseparable from class 54 with the current feature set and data volume.

7. **Anti-attractor partially helped 444** (10% -> 20%) but not enough. The weight=0.15 (up from 0.1 in v16) still couldn't overcome the embedding similarity.

8. **Many bug fixes, net neutral result.** This is a cautionary tale about making too many simultaneous changes. Each fix was individually motivated, but the combined effect was unpredictable. The separate normalization fix in particular may have hurt words while theoretically helping numbers.

9. **Combined accuracy dropped** from v17's 72.6% to 70.6%, making this a regression despite being the most "correct" pipeline. This motivated the v19+ direction of focusing on signer generalization rather than architecture fixes.
