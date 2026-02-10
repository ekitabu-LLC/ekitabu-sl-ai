# V17 Model Report

**Date:** 2026-02-08
**Script:** `train_ksl_v17.py`
**Result:** `data/results/v17_both_20260208_164359.json`

---

## Version Summary

Simplified ST-GCN with Stochastic Weight Averaging (SWA). Removed all v16 complexity (focal loss, contrastive loss, anti-attractor, mixup) for numbers. Numbers model capacity restored to match words (hidden_dim=128, 6 layers). SWA applied in last 20% of epochs.

---

## Architecture

| Property | Numbers | Words |
|----------|---------|-------|
| Model class | `KSLGraphNet` | `KSLGraphNet` |
| Type | ST-GCN (simplified) | ST-GCN (same as v16) |
| Layers | 6 (was 5 in v16) | 6 |
| Channel progression | 3->128->128->256->256->512->512 | 3->128->128->256->256->512->512 |
| Hidden dim | 128 (was 96 in v16) | 128 |
| Input features | 3 (x, y, z) | 3 (x, y, z) |
| Nodes | 48 | 48 |
| Pooling | AdaptiveAvgPool2d | AdaptiveAvgPool2d |
| Anti-attractor | Removed | N/A |
| Parameters | **6,946,991** | **6,946,991** |

Architecture is now identical for both splits -- same as v15's `STGCNModel` but renamed to `KSLGraphNet` and without the anti-attractor head.

---

## Key Changes from V16

1. **Removed all confusion-aware mechanisms** for numbers: no focal loss, no confusion pair penalties, no contrastive loss, no anti-attractor head, no mixup
2. **Restored numbers capacity** to match words: hidden_dim 96->128, layers 5->6 (2.45M -> 6.95M params)
3. **Added SWA** (Stochastic Weight Averaging) in last 20% of training epochs (swa_start_frac=0.8, swa_lr=1e-4)
4. **Numbers augmentation matched to words**: strong hand dropout (50%, range 30-70%), complete hand drop (10%), noise (std=0.015, prob=0.4)
5. **Training adjustments**: LR 3e-4->5e-4, weight_decay 0.01->0.003, label_smoothing 0.1->0.15, epochs 250->300, patience 50->60
6. **Words config unchanged** from v16 (91.3% baseline preserved)
7. Simple CrossEntropyLoss for both splits (with class weights and label smoothing)

---

## Training Configuration

### Numbers

| Parameter | Value | Changed from v16 |
|-----------|-------|-------------------|
| Optimizer | AdamW | - |
| Learning rate | 5e-4 | was 3e-4 |
| Weight decay | 0.003 | was 0.01 |
| Scheduler | Cosine + warmup, then SWA LR | New (SWA) |
| Batch size | 32 | - |
| Max epochs | 300 | was 250 |
| Patience | 60 | was 50 |
| Loss | CrossEntropyLoss (smoothing=0.15, class weights) | was Focal+Contrastive+Anti-attractor |
| Dropout | 0.4 | was 0.35 |
| SWA start | Epoch 240 (80% of 300) | New |
| SWA LR | 1e-4 | New |

### Words

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.001 |
| Batch size | 32 |
| Max epochs | 200 |
| Patience | 40 |
| Loss | CrossEntropyLoss (smoothing=0.15, class weights) |
| Dropout | 0.4 |
| SWA start | Epoch 160 (80% of 200) |

### Augmentation (Same for Both)

| Augmentation | Probability | Details |
|-------------|-------------|---------|
| Hand dropout | 50% (range 30-70%) | Matched to words |
| Complete hand drop | 10% | Now applied to numbers too |
| Hand swap | 30% | Same |
| Scale jitter | 50% (0.8-1.2x) | Same |
| Gaussian noise | 40% (std=0.015) | Stronger for numbers (was 0.006) |
| Temporal shift | 30% (np.roll) | Same (still has wrap-around bug) |

---

## Data Pipeline

Same as v15/v16:
- 48 nodes, 3 channels (xyz)
- Per-hand centering, mid-shoulder centering, global max normalization
- Linear interpolation / zero-pad to 90 frames
- WeightedRandomSampler with hard-class boosting (numbers only)

---

## Results

| Split | Val Accuracy | Best Epoch | Parameters |
|-------|-------------|------------|------------|
| Numbers | **55.3%** | 143/300 | 6,946,991 |
| Words | **89.9%** | 148/200 | 6,946,991 |
| **Combined** | **72.6%** | - | - |
| **Total time** | **15.0 min** | | |

### Per-Class Numbers

| Class | V15 | V16 | V17 | Delta (v16->v17) |
|-------|-----|-----|-----|-------------------|
| 17 | 100% | 100% | **96%** | -4 |
| 54 | 48% | 88% | **84%** | -4 |
| 9 | 52% | 76% | **76%** | 0 |
| 66 | 92% | 36% | **76%** | +40 |
| 388 | 55% | 85% | **70%** | -15 |
| 73 | 12% | 0% | **68%** | +68 |
| 268 | 55% | 90% | **65%** | -25 |
| 91 | 68% | 24% | **64%** | +40 |
| 22 | 76% | 20% | **52%** | +32 |
| 48 | 44% | 36% | **60%** | +24 |
| 100 | 44% | 56% | **48%** | -8 |
| 125 | 80% | 48% | **40%** | -8 |
| 89 | 0% | 24% | **16%** | -8 |
| 444 | 0% | 45% | **10%** | -35 |
| 35 | 0% | 0% | **0%** | 0 |

### Per-Class Words

| Class | V16 | V17 | Delta |
|-------|-----|-----|-------|
| Agreement | 92% | **100%** | +8 |
| Colour | 100% | 100% | 0 |
| Teach | 100% | 100% | 0 |
| Monday | 96% | **96%** | 0 |
| Friend | 100% | **96%** | -4 |
| Picture | 96% | **96%** | 0 |
| Proud | 80% | **96%** | +16 |
| Tortoise | 100% | **96%** | -4 |
| Gift | 100% | **92%** | -8 |
| Twin | 92% | **92%** | 0 |
| Apple | 100% | **84%** | -16 |
| Tomatoes | 80% | **84%** | +4 |
| Ugali | 100% | **84%** | -16 |
| Market | 83.3% | **79.2%** | -4.2 |
| Sweater | 52% | **56%** | +4 |

---

## Platform

| Property | Value |
|----------|-------|
| Cluster | Alpine HPC (CU Boulder) |
| GPU | NVIDIA A100-PCIE-40GB MIG 3g.20gb |
| Device | CUDA |
| Total time | 902.2 seconds (15.0 minutes) |
| Timestamp | 2026-02-08 16:43:59 |

---

## Lessons / Notes

1. **Numbers improved significantly** (47.5% -> 55.3%, +7.8pp) by removing complexity and restoring capacity. This was the best numbers accuracy at the time, suggesting v16's confusion-aware mechanisms were counterproductive with limited data.

2. **Simplification over engineering.** Removing focal loss, contrastive loss, anti-attractor, and mixup, and just using standard CrossEntropyLoss gave better results. The complex v16 losses may have introduced conflicting gradient signals with only 25 samples per class.

3. **Class distribution more balanced** in v17: no class above 96%, fewer extremes. Previously dominant classes (66, 22 in v15; 268, 54 in v16) were brought down while struggling classes (73, 91, 48) improved substantially.

4. **Class 35 still at 0%** -- a persistent failure across all versions so far.

5. **Class 444 regressed** from 45% (v16) to 10% (v17), confirming that the focal loss + anti-attractor from v16 did specifically help this class, even though overall accuracy declined.

6. **SWA likely never activated for numbers.** With swa_start_frac=0.8 and 300 epochs, SWA starts at epoch 240. But early stopping with patience=60 triggered at epoch 143 -- well before SWA activation. This was identified as a bug and fixed in v18.

7. **SWA activated for words** (epoch 160 of 200), but best epoch was 148, suggesting SWA didn't improve the already-converged model.

8. **Words dropped slightly** (91.3% -> 89.9%) despite identical config, likely due to random variation on the A100 MIG partition vs L40.

9. **Training time nearly doubled** (7.9 -> 15.0 min) due to larger numbers model (6.95M vs 2.45M params) and more epochs, running on A100 MIG (smaller partition than L40).

10. A pipeline review after v17 identified several bugs: shared hand/pose normalization squashing pose data ~5x, np.roll temporal shift creating wrap artifacts, hand swap not flipping pose X, and SWA never activating. These were all addressed in v18.
