# V16 Model Report

**Date:** 2026-02-08
**Script:** `train_ksl_v16.py`
**Result:** `data/results/v16_both_20260208_125613.json`

---

## Version Summary

ST-GCN with confusion-aware training: separate configs for numbers vs words, focal loss with confusion pair penalties, anti-attractor head, mixup, supervised contrastive loss, and hard-class boosting (all numbers-only). Reduced numbers model capacity.

---

## Architecture

| Property | Numbers | Words |
|----------|---------|-------|
| Model class | `KSLGraphNet` | `KSLGraphNet` |
| Type | ST-GCN + anti-attractor head | ST-GCN (standard) |
| Layers | 5 | 6 |
| Channel progression | 3->96->96->192->192->384 | 3->128->128->256->256->512->512 |
| Hidden dim | 96 | 128 |
| Input features | 3 (x, y, z) | 3 (x, y, z) |
| Nodes | 48 | 48 |
| Pooling | AdaptiveAvgPool2d | AdaptiveAvgPool2d |
| Anti-attractor head | Yes (class 54) | No |
| Parameters | **2,452,528** | **6,946,991** |

The anti-attractor head is a small auxiliary binary classifier (Linear(384, 48) -> GELU -> Linear(48, 1)) that predicts whether a sample is NOT the attractor class (54). This provides an additional gradient signal to separate confusable classes.

---

## Key Changes from V15

1. **Separate configs** for numbers vs words (different hidden dims, layers, augmentation, loss)
2. **ConfusionAwareFocalLoss** for numbers: focal loss (gamma=2.0) with explicit confusion pair penalties (e.g., 444->54 penalized 3.5x)
3. **Anti-attractor head** targeting class 54 (weight=0.1) for numbers
4. **Mixup augmentation** for numbers (alpha=0.2, prob=0.3)
5. **Supervised contrastive loss** for numbers (weight=0.3) pushing apart confusable class embeddings
6. **Hard-class boosting** in sampler: classes like 444, 35 get 2.0x sampling weight
7. **Reduced numbers capacity:** hidden_dim 128->96, layers 6->5 (2.45M vs 6.95M params)
8. **Gentler numbers augmentation:** hand dropout 70%->15%, dropout range 50-90%->5-20%, no complete hand drop
9. **Training adjustments:** LR 1e-3->3e-4, weight_decay 1e-4->0.01, label_smoothing 0.2->0.1, patience 40->50, warmup 10->15

---

## Training Configuration

### Numbers

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Scheduler | Cosine annealing with 15-epoch linear warmup |
| Batch size | 32 |
| Max epochs | 250 |
| Patience | 50 |
| Loss | ConfusionAwareFocalLoss (gamma=2.0, smoothing=0.1) + SupervisedContrastiveLoss (weight=0.3) + Anti-attractor BCE (weight=0.1) |
| Dropout | 0.35 |

### Words

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.001 |
| Scheduler | Cosine annealing with 10-epoch linear warmup |
| Batch size | 32 |
| Max epochs | 200 |
| Patience | 40 |
| Loss | CrossEntropyLoss (smoothing=0.15, class weights) |
| Dropout | 0.4 |

### Augmentation

| Augmentation | Numbers | Words |
|-------------|---------|-------|
| Hand dropout prob | 15% (range 5-20%) | 50% (range 30-70%) |
| Complete hand drop | 0% | 10% |
| Hand swap | 30% | 30% |
| Scale jitter | 50% (0.8-1.2x) | 50% (0.8-1.2x) |
| Gaussian noise | 30% (std=0.006) | 40% (std=0.015) |
| Temporal shift | 30% (np.roll) | 30% (np.roll) |
| Mixup | 30% (alpha=0.2) | None |

### Confusion Pairs (Numbers)

| True Class | Predicted As | Penalty |
|-----------|-------------|---------|
| 444 | 54 | 3.5x |
| 35 | 54 | 3.0x |
| 89 | 9 | 3.0x |
| 73 | 91 | 2.5x |
| 35 | 100 | 2.0x |
| 100 | 54 | 2.0x |
| 125 | 54 | 2.0x |

---

## Data Pipeline

Same as v15:
- 48 nodes (21 LH + 21 RH + 6 pose), 3 channels (xyz)
- Per-hand centering, mid-shoulder centering, global max normalization to [-1, 1]
- Linear interpolation / zero-pad to 90 frames
- Config-driven noise parameters (numbers: std=0.006, words: std=0.015)

---

## Results

| Split | Val Accuracy | Best Epoch | Parameters |
|-------|-------------|------------|------------|
| Numbers | **47.5%** | 173/250 | 2,452,528 |
| Words | **91.3%** | 121/200 | 6,946,991 |
| **Combined** | **69.4%** | - | - |
| **Total time** | **7.9 min** | | |

### Per-Class Numbers

| Class | V15 | V16 | Delta |
|-------|-----|-----|-------|
| 17 | 100% | 100% | 0 |
| 268 | 55% | **90%** | +35 |
| 54 | 48% | **88%** | +40 |
| 388 | 55% | **85%** | +30 |
| 9 | 52% | **76%** | +24 |
| 100 | 44% | **56%** | +12 |
| 125 | 80% | 48% | -32 |
| 444 | 0% | **45%** | +45 |
| 48 | 44% | 36% | -8 |
| 66 | 92% | 36% | -56 |
| 91 | 68% | 24% | -44 |
| 89 | 0% | **24%** | +24 |
| 22 | 76% | 20% | -56 |
| 73 | 12% | 0% | -12 |
| 35 | 0% | 0% | 0 |

### Per-Class Words

| Class | V15 | V16 | Delta |
|-------|-----|-----|-------|
| Apple | 92% | **100%** | +8 |
| Colour | 100% | 100% | 0 |
| Friend | 96% | **100%** | +4 |
| Gift | 100% | 100% | 0 |
| Tortoise | 84% | **100%** | +16 |
| Teach | 100% | 100% | 0 |
| Ugali | 100% | 100% | 0 |
| Monday | 92% | **96%** | +4 |
| Picture | 100% | 96% | -4 |
| Agreement | 100% | 92% | -8 |
| Market | 87.5% | 83.3% | -4.2 |
| Twin | 84% | **92%** | +8 |
| Tomatoes | 52% | **80%** | +28 |
| Proud | 84% | 80% | -4 |
| Sweater | 76% | 52% | -24 |

---

## Platform

| Property | Value |
|----------|-------|
| Cluster | Alpine HPC (CU Boulder) |
| GPU | NVIDIA L40 |
| Device | CUDA |
| Total time | 472.6 seconds (7.9 minutes) |
| Timestamp | 2026-02-08 12:56:13 |

---

## Lessons / Notes

1. **Numbers overall barely changed** (48.9% -> 47.5%) despite massive architectural overhaul. The confusion-aware mechanisms helped specific classes (444: 0%->45%, 89: 0%->24%, 268: 55%->90%) but others regressed badly (66: 92%->36%, 22: 76%->20%, 91: 68%->24%).

2. **Reduced capacity helped targeted classes but hurt others.** The smaller numbers model (2.45M params) better learned the focal/contrastive signals for hard classes, but lost capacity for previously well-classified classes.

3. **Words improved** from 89.7% to 91.3% with the separate config -- stronger augmentation works well for words.

4. **Confusion-aware training redistributed errors** rather than reducing them. Total error stayed similar, but shifted from attractor classes (54, 9) toward previously stable classes.

5. **Class 35 still at 0%** -- the confusion pair penalty was insufficient to separate it from class 54.

6. **Anti-attractor head effect unclear** -- class 54 went from 48% to 88% (the attractor got stronger), suggesting the mechanism may have backfired by making the model more confident about class 54 predictions.

7. The v16 analysis confirmed that the numbers problem requires fundamentally stronger regularization rather than loss engineering alone.
