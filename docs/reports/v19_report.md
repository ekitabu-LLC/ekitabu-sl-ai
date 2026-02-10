# Model Version Report: V19

**Date:** 2026-02-09
**Training Script:** `train_ksl_v19.py`
**Result File:** `data/results/v19_both_20260209_042922.json`
**Real-Tester File:** `data/results/v19_real_testers_20260209_062337.json`

---

## 1. Version Summary

V19 introduced **bone features** (child-parent joint vectors) to complement the existing XYZ coordinates and velocity features, bringing the input to 9 channels. It also replaced the single-kernel temporal convolution with **multi-scale TCN** (kernels 3, 5, 9) and added **confusion-pair margin loss** targeting 9 specific pairs identified from v18's confusion matrix. SWA (Stochastic Weight Averaging) was removed after consistently degrading accuracy. Separate model configs were used for numbers (smaller) and words (larger).

**Headline result:** Val 87.3% combined (numbers 78.6%, words 95.9%). Real-tester accuracy dropped to 34.2% combined (numbers 20.3%, words 48.1%), highlighting a 53pp val-to-real gap.

---

## 2. Architecture

**Model:** KSLGraphNet (ST-GCN + Multi-Scale TCN + AttentionPool)

- **Graph Convolution:** GConv layers implementing spectral graph convolution with learnable adjacency matrix
- **ST-GCN Blocks:** Each block = GConv -> BatchNorm -> ReLU -> Multi-Scale TCN -> Residual connection
- **Multi-Scale TCN:** Parallel temporal convolutions with kernels [3, 5, 9], outputs concatenated and projected
- **Attention Pooling:** Multi-head attention for temporal aggregation (replaces simple mean pooling)
- **Classification Head:** Linear layer on pooled features
- **Input:** 48 nodes (21 left hand + 21 right hand + 6 upper body), 9 channels (xyz + velocity + bone), 90 frames

**Numbers model:**
- hidden_dim=96, num_layers=5
- Parameters: 4,543,505

**Words model:**
- hidden_dim=128, num_layers=6
- Parameters: 12,793,585

Total parameters: ~17.3M (separate architectures for numbers vs words).

---

## 3. Key Changes from Previous Version (V18)

| Change | Description |
|--------|-------------|
| **Bone features** | Added child-parent joint vectors as 3 additional input channels (6ch -> 9ch) |
| **Velocity fix** | Velocity computed BEFORE temporal resampling (was after in v18) |
| **Numbers model shrink** | hidden_dim 128->96, layers 6->5 to reduce overfitting on smaller numbers dataset |
| **Mixup augmentation** | Added for numbers only (alpha=0.2) |
| **Multi-scale TCN** | Replaced single-kernel temporal conv with parallel kernels [3, 5, 9] |
| **Confusion-pair margin loss** | Extra margin penalty on 9 confused class pairs from v18 confusion matrix (weight=0.1, starts epoch 15) |
| **SWA removed** | Stochastic Weight Averaging consistently degraded accuracy |
| **Updated confusion pairs** | Pairs from v18's actual confusion matrix rather than theoretical ones |

---

## 4. Training Configuration

### Numbers
| Parameter | Value |
|-----------|-------|
| hidden_dim | 96 |
| num_layers | 5 |
| temporal_kernels | [3, 5, 9] |
| batch_size | 32 |
| epochs (max) | 250 |
| learning_rate | 0.0005 |
| min_lr | 1e-6 |
| weight_decay | 0.002 |
| dropout | 0.35 |
| label_smoothing | 0.1 |
| patience | 60 |
| warmup_epochs | 10 |
| mixup_alpha | 0.2 |
| confusion_penalty_weight | 0.1 |

### Words
| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_layers | 6 |
| temporal_kernels | [3, 5, 9] |
| batch_size | 32 |
| epochs (max) | 200 |
| learning_rate | 0.0005 |
| min_lr | 1e-6 |
| weight_decay | 0.001 |
| dropout | 0.4 |
| label_smoothing | 0.15 |
| patience | 50 |
| warmup_epochs | 10 |
| mixup_alpha | 0.0 (disabled) |
| confusion_penalty_weight | 0.0 (disabled) |

**Augmentation:**
- Hand dropout: prob=0.3/0.5 (numbers/words), min=0.1/0.3, max=0.4/0.7
- Complete hand drop: prob=0.05/0.1
- Gaussian noise: std=0.01/0.015, prob=0.3/0.4

**Hardware:** NVIDIA A100-PCIE-40GB MIG 3g.20gb
**Total training time:** 1105.6s (~18.4 minutes)

---

## 5. Data Pipeline

- **Skeleton extraction:** MediaPipe Holistic (21 left hand + 21 right hand + 6 upper body = 48 landmarks)
- **Input features:** 9 channels per node = XYZ coordinates (3) + velocity (3) + bone vectors (3)
- **Velocity:** Computed as frame-to-frame difference BEFORE temporal resampling (fixed from v18)
- **Bone features:** Child-parent joint vectors (new in v19)
- **Temporal resampling:** Uniform resampling to 90 frames
- **Normalization:** Per-sample max(abs) normalization (not yet wrist-centric)
- **Data split:** Leave-one-signer-out cross-validation (4 signers train, validate on held-out)
- **No deduplication:** All training samples used (including duplicates across signers)
- **Classes:** 15 numbers (9, 17, 22, 35, 48, 54, 66, 73, 89, 91, 100, 125, 268, 388, 444) + 15 words

---

## 6. Validation Results

### Numbers (Val: 78.6%)
Best epoch: 180 / 250

| Class | Val Acc (%) |
|-------|-------------|
| 17 | 100.0 |
| 54 | 100.0 |
| 66 | 100.0 |
| 73 | 100.0 |
| 444 | 95.0 |
| 268 | 85.0 |
| 9 | 84.0 |
| 22 | 80.0 |
| 388 | 75.0 |
| 100 | 72.0 |
| 48 | 72.0 |
| 125 | 68.0 |
| 91 | 56.0 |
| 35 | 52.0 |
| 89 | 44.0 |

Weakest classes: 89 (44%), 35 (52%), 91 (56%).

### Words (Val: 95.9%)
Best epoch: 116 / 200

| Class | Val Acc (%) |
|-------|-------------|
| Colour | 100.0 |
| Friend | 100.0 |
| Gift | 100.0 |
| Monday | 100.0 |
| Picture | 100.0 |
| Sweater | 100.0 |
| Teach | 100.0 |
| Tortoise | 100.0 |
| Twin | 100.0 |
| Ugali | 100.0 |
| Agreement | 96.0 |
| Apple | 96.0 |
| Market | 87.5 |
| Tomatoes | 84.0 |
| Proud | 76.0 |

Weakest classes: Proud (76%), Tomatoes (84%), Market (87.5%).

---

## 7. Real-Tester Results

### Numbers: 20.3% (12/59)

| Class | Real Acc (%) | Correct/Total |
|-------|-------------|---------------|
| 100 | 75.0 | 3/4 |
| 73 | 50.0 | 2/4 |
| 17 | 50.0 | 2/4 |
| 444 | 75.0 | 3/4 |
| 35 | 33.3 | 1/3 |
| 125 | 25.0 | 1/4 |
| 22 | 0.0 | 0/4 |
| 268 | 0.0 | 0/4 |
| 388 | 0.0 | 0/4 |
| 48 | 0.0 | 0/4 |
| 54 | 0.0 | 0/4 |
| 66 | 0.0 | 0/4 |
| 89 | 0.0 | 0/4 |
| 9 | 0.0 | 0/4 |
| 91 | 0.0 | 0/4 |

**9 out of 15 classes at 0% real accuracy.** Massive gap from val (78.6% vs 20.3%).

### Per-Signer Numbers
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 31.0% | 9/29 |
| Signer 2 | 6.7% | 1/15 |
| Signer 3 | 13.3% | 2/15 |

Extreme signer variance: Signer 1 at 31% vs Signer 2 at only 6.7%.

### Words: 48.1% (39/81)

| Class | Real Acc (%) | Correct/Total |
|-------|-------------|---------------|
| Teach | 100.0 | 6/6 |
| Friend | 83.3 | 5/6 |
| Sweater | 83.3 | 5/6 |
| Picture | 75.0 | 3/4 |
| Agreement | 66.7 | 4/6 |
| Twin | 66.7 | 4/6 |
| Gift | 50.0 | 2/4 |
| Colour | 33.3 | 2/6 |
| Monday | 33.3 | 2/6 |
| Tortoise | 33.3 | 2/6 |
| Proud | 33.3 | 1/3 |
| Tomatoes | 20.0 | 1/5 |
| Apple | 16.7 | 1/6 |
| Ugali | 16.7 | 1/6 |
| Market | 0.0 | 0/5 |

### Per-Signer Words
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 57.1% | 16/28 |
| Signer 2 | 56.0% | 14/25 |
| Signer 3 | 32.1% | 9/28 |

Words performance more consistent across signers, but Signer 3 still notably weaker.

### Val-to-Real Gap
- Numbers: 78.6% val -> 20.3% real = **58.3pp gap**
- Words: 95.9% val -> 48.1% real = **47.8pp gap**
- Combined: 87.3% val -> 34.2% real = **53.1pp gap**

---

## 8. Confidence Analysis

V19 evaluation did not include TTA or confidence bucket analysis (older evaluation format). Raw confidence scores were available per-prediction.

Looking at the prediction details:
- Correct predictions averaged ~0.35-0.50 confidence
- Incorrect predictions averaged ~0.20-0.35 confidence
- Some incorrect predictions had high confidence (e.g., 54->444 at 0.59 for Signer 1, 54->444 at 0.62 for Signer 3)
- No systematic confidence calibration was performed

---

## 9. Error Analysis

### Numbers Sink Classes
- **444** was the dominant sink class, attracting many misclassifications (48->444, 388->444, 54->444, 89->444, etc.)
- **17** and **66** also attracted some false positives
- Pattern: The model learned signer-specific features from Signer 1 (who had the most data at 29 samples) but failed to generalize

### Words Error Patterns
- **Sweater** attracted false positives from Monday, Market, and Tortoise
- **Friend** attracted false positives from other signers' Picture, Colour, Market samples
- **Market** had 0% accuracy with predictions scattered across Sweater, Friend, and Picture
- Signer 3 had notably lower accuracy (32.1%) suggesting the model learned signer-specific patterns from Signers 1-2

### Root Causes
1. **No deduplication:** Duplicate samples across signers inflated training accuracy without improving generalization
2. **Per-sample normalization:** Max(abs) normalization didn't account for hand/body size variation across signers
3. **Large model size:** 17.3M parameters for ~600 training samples -- severe overfitting
4. **Separate architectures:** Different configs for numbers/words prevented knowledge sharing

---

## 10. Lessons Learned

1. **Bone features help validation** but don't bridge the val-to-real gap. Val improved significantly with 9ch input, but real-tester performance remained limited.

2. **Confusion-pair margin loss** showed minimal impact -- the real confusion patterns differ between val and real testers because the underlying issue is signer variation, not class similarity.

3. **Multi-scale TCN** was an architectural improvement but didn't address the fundamental generalization problem.

4. **Model size matters:** 17.3M parameters for ~600 samples is heavily overparameterized. The numbers model (4.5M) underperformed relative to the words model (12.8M) despite being smaller, suggesting the issue is data diversity, not capacity.

5. **Per-sample normalization is insufficient:** Different signers have different hand sizes and signing speeds. Max(abs) normalization doesn't account for this, leading to signer-dependent feature distributions.

6. **Val accuracy is misleading:** 95.9% val for words vs 48.1% real shows that LOSO validation with only 4 signers drastically overestimates generalization ability.

7. **Key takeaway for v20:** Need deduplication, better normalization, smaller model, and signer-aware augmentation.
