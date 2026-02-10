# V15 Model Report

**Date:** 2026-02-08
**Script:** `train_ksl_alpine.py` (ported from v14 Modal script)
**Result:** `data/results/15_both_20260208_075703.json`

---

## Version Summary

ST-GCN baseline on Alpine HPC with extreme hand dropout augmentation, ported directly from v14 Modal code. First Alpine-native training run.

---

## Architecture

| Property | Value |
|----------|-------|
| Model class | `STGCNModel` |
| Type | Spatio-Temporal Graph Convolutional Network |
| Layers | 6 ST-GCN blocks |
| Channel progression | 3 -> 128 -> 128 -> 256 -> 256 -> 512 -> 512 |
| Hidden dim | 128 |
| Input features | 3 channels (x, y, z) |
| Nodes | 48 (21 left hand + 21 right hand + 6 pose) |
| Temporal kernel | 9 |
| Pooling | AdaptiveAvgPool2d |
| Classifier | Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes) |
| Parameters | 6,946,991 (shared architecture for both numbers and words) |
| Temporal stride | 2x at layers 2 and 4 |

Graph topology: Hand skeleton edges (23 per hand), pose edges connecting shoulders/elbows/wrists, inter-hand edge (wrist-to-wrist, weight 0.3). Normalized symmetric adjacency with self-loops.

---

## Key Changes from Previous Version

V15 is a direct port of v14 from Modal cloud to Alpine HPC. No architectural changes. Key characteristics inherited from v14:

- Extreme hand dropout (50-90% landmark dropout at 70% probability)
- Complete hand drop (20% probability of zeroing entire hand)
- Hand swap augmentation (30% probability)
- Single shared config for both numbers and words
- WeightedRandomSampler with inverse-frequency class weights (capped at 2.0x)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Min LR | 1e-6 |
| Weight decay | 1e-4 |
| Scheduler | Cosine annealing with 10-epoch linear warmup |
| Batch size | 32 |
| Max epochs | 200 |
| Early stopping | 40 epochs patience |
| Loss | CrossEntropyLoss with label_smoothing=0.2, class weights |
| Grad clipping | Max norm 1.0 |
| Sampler | WeightedRandomSampler (inverse-frequency, capped 2.0x) |

### Augmentation

| Augmentation | Probability | Details |
|-------------|-------------|---------|
| Hand dropout | 70% | Drop 50-90% of hand landmarks per frame |
| Complete hand drop | 20% | Zero out entire left or right hand |
| Hand swap | 30% | Swap left/right hands + flip X |
| Scale jitter | 50% | Multiply by uniform(0.8, 1.2) |
| Gaussian noise | 40% | std=0.03 |
| Temporal shift | 30% | np.roll by -8 to +8 frames (wrap-around) |

---

## Data Pipeline

- **Landmarks:** 48 nodes extracted from MediaPipe: left hand (21), right hand (21), upper body pose (6 from indices 11-16)
- **Normalization:** Per-hand centering (subtract wrist), pose centering (subtract mid-shoulder), global max normalization to [-1, 1]
- **Temporal:** Linear interpolation to 90 frames if longer; zero-pad if shorter
- **Data dirs:** `train_v2` / `val_v2` (signer-independent split)
- **No deduplication**

---

## Results

| Split | Val Accuracy | Best Epoch | Training Time |
|-------|-------------|------------|---------------|
| Numbers | **48.9%** | 160/200 | - |
| Words | **89.7%** | 95/200 | - |
| **Combined** | **69.3%** | - | **8.7 min** |

### Per-Class Numbers

| Class | Accuracy | Notes |
|-------|----------|-------|
| 17 | 100.0% | |
| 66 | 92.0% | |
| 125 | 80.0% | |
| 22 | 76.0% | |
| 91 | 68.0% | |
| 268 | 55.0% | |
| 388 | 55.0% | |
| 9 | 52.0% | Absorbs class 89 |
| 54 | 48.0% | Absorbs 444, 35 |
| 100 | 44.0% | |
| 48 | 44.0% | |
| 73 | 12.0% | Confused with 91 |
| 35 | **0.0%** | Collapsed into 54 |
| 444 | **0.0%** | Collapsed into 54 |
| 89 | **0.0%** | Collapsed into 9 |

### Per-Class Words

| Class | Accuracy |
|-------|----------|
| Agreement, Colour, Gift, Picture, Teach, Ugali | 100.0% |
| Friend | 96.0% |
| Apple, Monday | 92.0% |
| Market | 87.5% |
| Proud, Tortoise, Twin | 84.0% |
| Sweater | 76.0% |
| Tomatoes | 52.0% |

---

## Platform

| Property | Value |
|----------|-------|
| Cluster | Alpine HPC (CU Boulder) |
| GPU | NVIDIA L40 |
| Device | CUDA |
| Total time | 522.3 seconds (8.7 minutes) |
| Timestamp | 2026-02-08 07:57:03 |

---

## Lessons / Notes

1. **Extreme hand dropout destroys number recognition.** Numbers are distinguished by static hand shapes and fine finger positions. Dropping 50-90% of landmarks removes the discriminative signal. Words use larger arm/body motions that survive dropout, explaining the 40pp gap.

2. **Three classes at 0% accuracy** (35, 444, 89) due to near-identical feature spaces with attractor classes (54, 9). Cosine similarity between centroids exceeds 0.99 for these pairs.

3. **Massive overparameterization:** 6.95M parameters for 375 training samples (18,526 params/sample). Train accuracy saturates at 95%+ while val oscillates wildly.

4. **Single config for both splits is suboptimal.** Numbers and words have fundamentally different discriminative features and require different augmentation strategies.

5. **No confusion-aware training mechanisms.** Unlike v8 which had focal loss, confusion pair penalties, and anti-attractor heads, v15 uses only standard cross-entropy.

6. **Label smoothing at 0.2 is too aggressive** for already-confusable classes, further weakening the learning signal.

7. A detailed analysis was performed and saved to `data/results/v15_analysis_report.md`, which directly informed the v16 design.
