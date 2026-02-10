# Model Version Report: V20

**Date:** 2026-02-09
**Training Script:** `train_ksl_v20.py`
**Result File:** `data/results/v20_both_20260209_071135.json`
**Real-Tester File:** `data/results/v20_real_testers_20260209_074352.json`

---

## 1. Version Summary

V20 was a comprehensive overhaul focused on **real-tester generalization**. Key changes: aggressive file-level deduplication, wrist-centric + palm-size normalization, model shrink to ~413K parameters (from 17.3M), signer-mimicking augmentation suite, dual classification heads, and supervised contrastive loss. The architecture was unified for both numbers and words.

**Headline result:** Val 86.9% combined (numbers 82.2%, words 91.6%). Real testers: numbers 20.3% TTA / 18.6% noTTA, words 40.7% TTA / 42.0% noTTA. Despite more principled design, real accuracy did not improve for numbers and decreased for words compared to v19. The aggressive deduplication removed too many samples (~240 remaining), hurting training diversity.

---

## 2. Architecture

**Model:** KSLGraphNet (Lightweight ST-GCN + Dual Head)

- **Graph Convolution:** GConv with learnable adjacency matrix
- **ST-GCN Blocks:** GConv -> BatchNorm -> ReLU -> Multi-Scale TCN (kernels [3, 5, 7]) -> Residual
- **SpatialNodeDropout:** Randomly drops entire nodes (new in v20)
- **Attention Pooling:** Multi-head temporal attention aggregation
- **Dual Heads:**
  - Main head: Full 9-channel feature classification
  - Bone auxiliary head: Bone-only (3ch) feature classification (weight=0.3)
- **Supervised Contrastive Loss (SupCon):** Cross-signer invariant embedding (weight=0.1, temp=0.07)

**Unified config (both numbers and words):**
- hidden_dim=48, num_layers=4
- Parameters: 413,312

This is a ~42x reduction from v19's total parameter count (17.3M -> 413K).

---

## 3. Key Changes from Previous Version (V19)

| Change | Description | Rationale |
|--------|-------------|-----------|
| **File-level deduplication** | Removed byte-identical files across Signers 2 & 3 | Prevent leaking signer-specific patterns |
| **Wrist-centric normalization** | Hands centered on wrist, scaled by palm size; pose centered on mid-shoulder, scaled by shoulder width | Signer-invariant spatial representation |
| **Model shrink** | hidden_dim 96/128 -> 48, layers 5/6 -> 4, unified architecture | Reduce overfitting (17.3M -> 413K params) |
| **Signer-mimicking augmentation** | Bone perturbation (0.8-1.2x), hand size scaling (0.85-1.15x), temporal warp, rotation (+/-15deg) | Simulate signer variation |
| **Dual classification heads** | Main (9ch) + bone-only auxiliary (3ch) | Force learning bone-invariant features |
| **SupCon loss** | Supervised contrastive loss (weight=0.1) | Pull same-class, different-signer samples together |
| **Spatial node dropout** | Drop entire nodes randomly (p=0.1) | Structural regularization |
| **Unified architecture** | Same model config for numbers and words | Simplify and prevent overfitting |

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 48 |
| num_layers | 4 |
| in_channels | 9 |
| temporal_kernels | [3, 5, 7] |
| batch_size | 32 |
| epochs (max) | 300 |
| learning_rate | 0.001 |
| min_lr | 1e-6 |
| weight_decay | 0.001 |
| dropout | 0.3 |
| spatial_dropout | 0.1 |
| label_smoothing | 0.1 |
| patience | 50 |
| warmup_epochs | 10 |
| mixup_alpha | 0.2 |
| bone_head_weight | 0.3 |
| supcon_weight | 0.1 |
| supcon_temperature | 0.07 |

**Augmentation:**
- Bone perturbation: prob=0.5, range=[0.8, 1.2]
- Hand size scaling: prob=0.4, range=[0.85, 1.15]
- Temporal warp: prob=0.4, sigma=0.2
- Rotation: prob=0.3, max_deg=15
- Hand dropout: prob=0.3, min=0.1, max=0.4
- Complete hand drop: prob=0.05
- Gaussian noise: std=0.02, prob=0.5

**Hardware:** NVIDIA A100-PCIE-40GB MIG 3g.20gb
**Total training time:** 347.1s (~5.8 minutes) -- 3.2x faster than v19

---

## 5. Data Pipeline

- **Skeleton extraction:** MediaPipe Holistic (48 landmarks)
- **Input features:** 9 channels = XYZ (3) + velocity (3) + bone (3)
- **Normalization:** Wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose) -- NEW
- **Velocity:** Computed before temporal resampling (same as v19)
- **Temporal resampling:** 90 frames
- **Deduplication:** File-level byte-identical removal across Signers 2 & 3 (~240 samples remaining, down from ~600)
- **MediaPipe version check:** Added warning for version mismatches
- **Classes:** 15 numbers + 15 words (same as v19)

**Note:** The aggressive deduplication reduced training data from ~600 to ~240 samples. This was later identified as too aggressive.

---

## 6. Validation Results

### Numbers (Val: 82.2%)
Best epoch: 198 / 300

| Class | Val Acc (%) |
|-------|-------------|
| 17 | 100.0 |
| 66 | 100.0 |
| 268 | 100.0 |
| 9 | 96.0 |
| 444 | 95.0 |
| 54 | 88.0 |
| 125 | 84.0 |
| 48 | 84.0 |
| 91 | 84.0 |
| 22 | 80.0 (was 40.0 in v19) -- IMPROVED |
| 73 | 80.0 |
| 100 | 76.0 |
| 388 | 75.0 |
| 89 | 68.0 (was 44.0 in v19) -- IMPROVED |
| 35 | 68.0 (was 52.0 in v19) -- IMPROVED |

Val numbers improved from 78.6% to 82.2% (+3.6pp), with the weakest classes all improving.

### Words (Val: 91.6%)
Best epoch: 46 / 300

| Class | Val Acc (%) |
|-------|-------------|
| Agreement | 100.0 |
| Apple | 100.0 |
| Friend | 100.0 |
| Gift | 100.0 |
| Monday | 100.0 |
| Teach | 100.0 |
| Tomatoes | 100.0 |
| Tortoise | 100.0 |
| Ugali | 100.0 |
| Colour | 96.0 |
| Sweater | 96.0 |
| Picture | 88.0 |
| Twin | 80.0 |
| Market | 75.0 |
| Proud | 40.0 (was 76.0 in v19) -- REGRESSED |

Val words decreased slightly from 95.9% to 91.6% (-4.3pp). Proud collapsed from 76% to 40%.

---

## 7. Real-Tester Results

### Numbers

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | 20.3% (12/59) | 18.6% (11/59) |

TTA effect: +1.7pp (minimal help).

**Per-class (TTA / no-TTA):**

| Class | TTA | No-TTA |
|-------|-----|--------|
| 66 | 75.0% | 50.0% |
| 100 | 50.0% | 50.0% |
| 444 | 50.0% | 50.0% |
| 9 | 50.0% | 50.0% |
| 35 | 33.3% | 33.3% |
| 17 | 25.0% | 25.0% |
| 73 | 25.0% | 25.0% |
| 125 | 0.0% | 0.0% |
| 22 | 0.0% | 0.0% |
| 268 | 0.0% | 0.0% |
| 388 | 0.0% | 0.0% |
| 48 | 0.0% | 0.0% |
| 54 | 0.0% | 0.0% |
| 89 | 0.0% | 0.0% |
| 91 | 0.0% | 0.0% |

**8 classes at 0% real accuracy** (down from 9 in v19). 66 improved dramatically from 0% to 75%.

### Per-Signer Numbers (TTA)
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 24.1% | 7/29 |
| Signer 2 | 20.0% | 3/15 |
| Signer 3 | 13.3% | 2/15 |

Signer 2 improved from 6.7% to 20.0%, but Signer 1 decreased from 31.0% to 24.1%.

### Words

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | 40.7% (33/81) | 42.0% (34/81) |

TTA effect: -1.3pp (TTA slightly hurts).

**Per-class (TTA / no-TTA):**

| Class | TTA | No-TTA |
|-------|-----|--------|
| Friend | 100.0% | 100.0% |
| Teach | 83.3% | 83.3% |
| Picture | 75.0% | 75.0% |
| Tortoise | 66.7% | 33.3% |
| Twin | 66.7% | 66.7% |
| Monday | 50.0% | 50.0% |
| Gift | 50.0% | 50.0% |
| Apple | 33.3% | 33.3% |
| Sweater | 33.3% | 50.0% |
| Colour | 16.7% | 33.3% |
| Ugali | 16.7% | 16.7% |
| Tomatoes | 0.0% | 20.0% |
| Agreement | 0.0% | 0.0% |
| Market | 0.0% | 0.0% |
| Proud | 0.0% | 0.0% |

### Per-Signer Words (TTA / no-TTA)
| Signer | TTA | No-TTA |
|--------|-----|--------|
| Signer 1 | 53.6% | 53.6% |
| Signer 2 | 40.0% | 44.0% |
| Signer 3 | 28.6% | 28.6% |

### Val-to-Real Gap
- Numbers: 82.2% val -> 20.3% TTA / 18.6% noTTA = **~62pp gap**
- Words: 91.6% val -> 40.7% TTA / 42.0% noTTA = **~50pp gap**

---

## 8. Confidence Analysis

### Numbers Confidence Buckets (TTA)
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 0.0% | 1 |
| MEDIUM (0.4-0.7) | 28.6% | 7 |
| LOW (<0.4) | 19.6% | 51 |

86.4% of number predictions fell in the LOW confidence bucket, and the single HIGH-confidence prediction was wrong. The model is fundamentally uncertain about numbers.

### Words Confidence Buckets (TTA)
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 86.7% | 15 |
| MEDIUM (0.4-0.7) | 33.3% | 24 |
| LOW (<0.4) | 28.6% | 42 |

Words HIGH confidence was well-calibrated at 86.7% accuracy. The model knows when it is confident about words, but 51.9% of predictions fell in LOW confidence.

---

## 9. Error Analysis

### Numbers Sink Classes
- **66** became the dominant sink class, attracting predictions from multiple classes (similar to 444 in v19)
- Classes 125, 22, 268, 388, 48, 54, 89, 91 all had 0% real accuracy
- The confusion matrix shows 66 column had many off-diagonal entries

### Words Error Patterns
- **Agreement** went from 66.7% (v19) to 0% -- regression
- **Market** remained at 0% across both versions
- **Proud** dropped from 33.3% to 0%
- The model learned some classes well (Friend, Teach at 100%) but lost others

### Impact of Aggressive Deduplication
The file-level deduplication was **too aggressive**, removing ~60% of training data:
- Some "duplicates" were actually valid variations by the same signer at different times
- With only ~240 training samples for 30 classes (8 per class average), the model lacked diversity
- This is the primary reason real performance didn't improve despite better architecture

### Wrist-Centric Normalization
The new normalization was theoretically sound but may not have been sufficient:
- Hands normalized by wrist + palm size
- Pose normalized by mid-shoulder + shoulder width
- However, signing style variation goes beyond spatial scaling

---

## 10. Lessons Learned

1. **Deduplication must be careful:** File-level byte-identical removal was too aggressive. Many "duplicates" were valid data variants. Reducing from ~600 to ~240 samples significantly hurt model diversity.

2. **Wrist-centric normalization is a step forward** but insufficient alone. The val accuracy improved for numbers, but the real-tester gap widened, suggesting the normalization helps within-signer but not cross-signer generalization.

3. **Model compression works for validation:** 413K params achieved comparable or better val accuracy than 17.3M params, confirming v19 was severely overparameterized.

4. **TTA is neutral or harmful:** For numbers, TTA gave +1.7pp. For words, TTA gave -1.3pp. Not worth the 5x inference cost.

5. **Words confidence is well-calibrated:** HIGH bucket at 86.7% accuracy means confidence thresholds could be used for deployment decisions.

6. **SupCon and dual heads** are promising directions but need more training data to show their value. With only ~240 samples, these regularization techniques may not have enough diversity to learn from.

7. **Key takeaway for v21:** Reduce deduplication aggressiveness, increase model capacity back to ~700K, add stronger signer-adversarial training.
