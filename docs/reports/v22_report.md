# Model Version Report: V22

**Date:** 2026-02-09
**Training Script:** `train_ksl_v22.py`
**Result File:** `data/results/v22_both_20260209_124246.json`
**Real-Tester File:** `data/results/v22_real_testers_20260209_132917.json`
**Analysis:** `data/results/v22_analysis/` (7 plots, 2 reports, analysis_data.json)

---

## 1. Version Summary

V22 was a disciplined simplification: **remove ArcFace**, **single-stage training** (CE + SupCon from epoch 0), **GRL from epoch 0 with lambda_max=0.3**, **reduced augmentation**, and **bigger auxiliary branch** (64-dim with temporal Conv1D). It also introduced **signer-balanced batch sampling** to ensure equal representation.

**Headline result:** Val 93.3% combined (numbers 91.9%, words 94.6%) -- best validation ever. Real testers: numbers 33.9% noTTA / 28.8% TTA, words 45.7% noTTA / 46.9% TTA. Combined noTTA: **39.8%** -- best real-world performance ever. V22 confirmed that **removing complexity helps more than adding it**.

---

## 2. Architecture

**Model:** KSLGraphNetV22 (ST-GCN + Bigger Aux MLP + GRL, NO ArcFace)

- **Backbone:** ST-GCN with GConv, Multi-Scale TCN (kernels [3, 5, 7]), SpatialNodeDropout
- **Attention Pooling:** Multi-head temporal attention
- **Auxiliary Branch (bigger):**
  - Input: 53 hand-crafted features (joint angles + fingertip distances)
  - Architecture: 53 -> 128 -> 64, with **temporal Conv1D** (new)
  - Output: 64-dim embedding (~20% of total embedding)
  - Concatenated with GCN backbone embedding
- **Classification Head:** Single CE head (no ArcFace)
- **SupCon Loss:** weight=0.1, temperature=0.07 (single-stage, from epoch 0)
- **Signer-Adversarial (GRL):**
  - Starts epoch 0 (was epoch 50 in v21)
  - lambda_max=0.3 (was 0.1 in v21)
  - Ramps over 100 epochs
- **Signer-Balanced Sampling:** Equal representation per signer per batch (NEW)

**Config:**
- hidden_dim=64, num_layers=4
- Parameters: 776,544 (slightly larger than v21 due to bigger aux branch)

---

## 3. Key Changes from Previous Version (V21)

| Change | Description | Rationale | Result |
|--------|-------------|-----------|--------|
| **Remove ArcFace** | Back to standard CE classification | ArcFace was catastrophic in v21 | SUCCESS (+30pp val numbers) |
| **Single-stage training** | CE + SupCon + GRL from epoch 0 | Two-stage transition was fragile | SUCCESS (stable training) |
| **GRL from epoch 0** | lambda ramp starts immediately | v21 GRL started too late (epoch 50) | PARTIAL (signer ~32%, better than chance) |
| **GRL lambda 0.3** | 3x stronger than v21's 0.1 | v21's lambda was too weak | PARTIAL |
| **Reduced augmentation** | Rotation 15deg (was 30), shear 0.2 (was 0.5), no axis mask, noise 0.015 (was 0.03) | v21 augmentation was too aggressive | SUCCESS (less overfitting) |
| **Bigger aux branch** | 53->128->64 with temporal Conv1D (was 53->64->32) | Capture more temporal structure in aux features | LIKELY HELPFUL |
| **Signer-balanced sampling** | Equal signer representation per batch | Prevent model from memorizing dominant signer | LIKELY HELPFUL |
| **SupCon weight 0.1** | Reduced from v21's 0.5/0.3 | Match simpler single-stage setup | OK |

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 64 |
| num_layers | 4 |
| in_channels | 9 |
| temporal_kernels | [3, 5, 7] |
| batch_size | 32 |
| epochs (max) | 200 |
| learning_rate | 0.001 |
| min_lr | 1e-6 |
| weight_decay | 0.001 |
| dropout | 0.3 |
| spatial_dropout | 0.1 |
| label_smoothing | 0.1 |
| patience | 40 |
| warmup_epochs | 10 |
| supcon_weight | 0.1 |
| supcon_temperature | 0.07 |
| grl_lambda_max | 0.3 |
| grl_start_epoch | 0 |
| grl_ramp_epochs | 100 |
| mixup_alpha | 0.2 |

**Augmentation (reduced from v21):**
- Rotation: prob=0.5, max_deg=**15** (was 30 in v21)
- Shear: prob=0.3, max=**0.2** (was 0.5 in v21)
- Joint dropout: prob=0.2, rate=**0.08** (was 0.3/0.15 in v21)
- **No axis mask** (removed from v21)
- Gaussian noise: std=**0.015** (was 0.03 in v21), prob=0.4
- Bone perturbation: prob=0.5, range=[**0.8, 1.2**] (was 0.7-1.3 in v21)
- Hand size: prob=0.5, range=[**0.8, 1.2**] (was 0.7-1.3 in v21)
- Temporal warp: prob=0.4, sigma=**0.2** (was 0.3 in v21)
- Hand dropout: prob=0.2, min=0.1, max=0.3
- Complete hand drop: prob=0.03

**Hardware:** NVIDIA A100-PCIE-40GB MIG 3g.20gb
**Total training time:** 435.5s (~7.3 minutes)

---

## 5. Data Pipeline

- **Skeleton extraction:** MediaPipe Holistic (48 landmarks)
- **Input features:** GCN: 9ch (xyz + velocity + bone) | AUX: 53 features (angles + fingertip distances) with temporal Conv1D
- **Normalization:** Wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)
- **Temporal resampling:** 90 frames
- **Deduplication:** Signer-group level (same as v21, ~600-650 samples)
- **Signer-balanced sampling:** New batch sampler ensures equal signer representation per batch
- **Classes:** 15 numbers + 15 words

---

## 6. Validation Results

### Numbers (Val: 91.9%) -- BEST NUMBERS VAL EVER
Best epoch: 122 / 200

| Class | Val Acc (%) | vs V21 |
|-------|-------------|--------|
| 17 | 100.0 | +32pp |
| 268 | 100.0 | same |
| 66 | 100.0 | +32pp |
| 444 | 100.0 | +20pp |
| 9 | 96.0 | +8pp |
| 125 | 92.0 | +12pp |
| 91 | 92.0 | +8pp |
| 388 | 90.0 | **+90pp** |
| 54 | 88.0 | **+88pp** |
| 89 | 88.0 | **+88pp** |
| 22 | 84.0 | same |
| 48 | 80.0 | +20pp |
| 73 | 80.0 | +44pp |
| 100 | 76.0 | -4pp |
| 35 | 72.0 | -24pp |

Massive recovery from v21's ArcFace catastrophe. The three classes that had 0% in v21 (388, 54, 89) all recovered to 88-90%.

### Words (Val: 94.6%)
Best epoch: 77 / 200

| Class | Val Acc (%) | vs V21 |
|-------|-------------|--------|
| Apple | 100.0 | same |
| Colour | 100.0 | same |
| Friend | 100.0 | same |
| Monday | 100.0 | +8pp |
| Teach | 100.0 | same |
| Twin | 100.0 | +20pp |
| Ugali | 100.0 | +4pp |
| Sweater | 96.0 | +4pp |
| Gift | 96.0 | same |
| Tortoise | 96.0 | same |
| Picture | 92.0 | -8pp |
| Tomatoes | 88.0 | -12pp |
| Agreement | 84.0 | -16pp |
| Proud | 72.0 | +20pp |
| Market | 96.0 | +29.3pp |

Words val improved from 91.3% to 94.6% (+3.3pp). Proud and Market improved significantly.

---

## 7. Real-Tester Results

### Numbers

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | 28.8% (17/59) | **33.9% (20/59)** |

**TTA effect: -5.1pp (TTA HURTS numbers).** This is the key finding -- no-TTA is significantly better.

**Per-class (no-TTA, the better metric):**

| Class | No-TTA | vs V21 no-TTA |
|-------|--------|---------------|
| 22 | 75.0% | same |
| 9 | 75.0% | +50pp |
| 100 | 50.0% | same |
| 17 | 50.0% | +50pp |
| 444 | 50.0% | same |
| 388 | 50.0% | +50pp |
| 66 | 50.0% | +50pp |
| 268 | 25.0% | same |
| 54 | 25.0% | +25pp |
| 89 | 25.0% | +25pp |
| 91 | 25.0% | -25pp |
| 125 | 0.0% | same |
| 35 | 0.0% | -33.3pp |
| 48 | 0.0% | -50pp |
| 73 | 0.0% | -25pp |

Real numbers improved from 25.4% to 33.9% (+8.5pp). Five zero-accuracy classes remain, but multiple classes showed dramatic improvement.

### Per-Signer Numbers (no-TTA)
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 31.0% | 9/29 |
| Signer 2 | 33.3% | 5/15 |
| Signer 3 | 40.0% | 6/15 |

**Most balanced signer performance ever.** Signer 3 actually best for the first time. GRL and signer-balanced sampling are working.

### Words

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | **46.9% (38/81)** | 45.7% (37/81) |

TTA effect: +1.2pp (slight help for words).

**Per-class (no-TTA):**

| Class | No-TTA | vs V21 no-TTA |
|-------|--------|---------------|
| Friend | 100.0% | +66.7pp |
| Teach | 100.0% | +33.3pp |
| Gift | 100.0% | same |
| Tortoise | 66.7% | +50pp |
| Twin | 66.7% | same |
| Picture | 50.0% | -25pp |
| Colour | 33.3% | same |
| Monday | 33.3% | +16.6pp |
| Apple | 33.3% | same |
| Ugali | 33.3% | +33.3pp |
| Proud | 33.3% | -33.4pp |
| Tomatoes | 20.0% | +20pp |
| Agreement | 16.7% | -50pp |
| Sweater | 0.0% | same |
| Market | 0.0% | -20pp |

Words improved from 37.0% to 46.9% TTA / 45.7% noTTA (+8-9pp).

### Per-Signer Words (no-TTA)
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 46.4% | 13/28 |
| Signer 2 | 56.0% | 14/25 |
| Signer 3 | 35.7% | 10/28 |

### Val-to-Real Gap
- Numbers: 91.9% val -> 33.9% noTTA = **58.0pp gap**
- Words: 94.6% val -> 45.7% noTTA = **48.9pp gap**
- Combined: 93.3% val -> 39.8% noTTA = **53.5pp gap**

---

## 8. Confidence Analysis

### Numbers Confidence Buckets (no-TTA)
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | **18.2%** | 11 |
| MEDIUM (0.4-0.7) | 54.5% | 11 |
| LOW (<0.4) | 24.3% | 37 |

**Numbers HIGH confidence is severely miscalibrated.** At 18.2% accuracy, the model is confidently wrong most of the time for numbers. MEDIUM bucket is actually better at 54.5%. This is a critical deployment concern.

### Words Confidence Buckets (no-TTA)
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | **88.9%** | 18 |
| MEDIUM (0.4-0.7) | 45.5% | 22 |
| LOW (<0.4) | 29.3% | 41 |

**Words HIGH confidence is well-calibrated at 88.9%.** This means confidence thresholds are reliable for words. The model knows what it knows for words but not for numbers.

### Deployment Implications
- At confidence >= 0.7: Words can be trusted (88.9%). Numbers cannot (18.2%).
- At confidence >= 0.5: Combined accuracy ~63.2%, with 64% rejection rate.
- Different confidence thresholds should be used for numbers vs words.

---

## 9. Error Analysis

### Numbers Sink Class: 66
- Class 66 was predicted 19 out of 59 times (32.2% of all number predictions)
- Only 2 of those 19 predictions were correct (10.5% precision)
- Classes 100, 125, 17, 268, 91 were all frequently misclassified as 66
- The confusion matrix shows 66 column has many off-diagonal entries

### Words Sink Class: Tortoise
- Tortoise was predicted 16 out of 81 times (19.8% of all word predictions)
- 4 of 16 were correct (25% precision)
- Classes attracted to Tortoise: Monday, Colour, Sweater, Ugali

### Zero-Accuracy Classes
**Numbers:** 125, 268 (from TTA perspective), 35, 48, 73
**Words:** Market, Sweater (noTTA), Tomatoes (near-zero)

### Signer Invariance (from analysis report)
- Cramer's V signer effect: Numbers 0.059, Words 0.135
- These are low values, meaning the model is more signer-invariant than previous versions
- GRL + signer-balanced sampling appears to reduce signer dependence

### Cross-Version Class Stability
Class accuracy is highly unstable across versions:
- 66: 0% (v19) -> 75% (v20) -> 0% (v21) -> 50% (v22)
- 48: 0% (v19) -> 0% (v20) -> 50% (v21) -> 0% (v22)
- Market: 0% across all versions
- Friend: 83% -> 100% -> 33% -> 100%

This instability suggests the model learns fragile, version-specific decision boundaries.

---

## 10. Lessons Learned

1. **Removing ArcFace was the single most impactful change.** Val numbers recovered by 30pp, and real numbers improved to best ever. Simplification beat complexity.

2. **GRL from epoch 0 with stronger lambda works.** Signer accuracy dropped to ~32% (from 100% in v21), and per-signer real accuracy became more balanced. The signer-adversarial training is having the intended effect.

3. **Signer-balanced sampling matters.** Combined with GRL, it produced the most balanced per-signer results ever. Signer 3 at 40% (numbers) is the best any signer has performed.

4. **Reduced augmentation prevents overfitting.** Cutting augmentation intensity in half compared to v21 improved both val and real accuracy.

5. **TTA hurts for numbers (-5.1pp).** This is consistent and significant. TTA should be disabled in production, especially for numbers.

6. **Confidence calibration is split.** Words HIGH = 88.9% is excellent. Numbers HIGH = 18.2% is dangerous. Any deployment must handle these differently.

7. **4-signer ceiling reached at ~40%.** V22 pushed real combined accuracy to 39.8% noTTA, likely near the ceiling possible with 4 training signers. Further improvement requires more signers.

8. **Sink classes are a persistent phenomenon** but change identity across versions (444->66->444->66). They represent the model's confusion when it lacks discriminative features for OOD signers.

9. **Key recommendations for v23:** (a) Collect more signers (P0), (b) Disable TTA (P0), (c) Multi-stream ensemble to reduce dependence on any single feature type, (d) Focal loss for sink classes, (e) Confidence-aware deployment (reject low confidence, show top-3).
