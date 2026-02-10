# Model Version Report: V21

**Date:** 2026-02-09
**Training Script:** `train_ksl_v21.py`
**Result File:** `data/results/v21_both_20260209_091813.json`
**Real-Tester File:** `data/results/v21_real_testers_20260209_103810.json`
**Analysis:** `data/results/v21_analysis/ml_recommendations.md`

---

## 1. Version Summary

V21 was the most experimental version, introducing multiple new techniques simultaneously: **ArcFace angular margin classification**, **Gradient Reversal Layer (GRL)** for signer-adversarial training, **auxiliary MLP** for joint angles and fingertip distances, **two-stage training** (CE -> ArcFace), and significantly wider augmentation. It also fixed the overly aggressive deduplication from v20 by switching to signer-group dedup (~600-650 samples retained vs v20's ~240).

**Headline result:** Val 76.5% combined (numbers 61.7%, words 91.3%). Real testers: numbers 25.4%, words 37.0%, combined 31.2%. Numbers validation was catastrophically low due to ArcFace instability, but real-tester numbers actually improved over v20. Words regressed on real testers. Overall, v21 was a step backward driven primarily by the ArcFace transition failure.

---

## 2. Architecture

**Model:** KSLGraphNetV21 (ST-GCN + Aux MLP + ArcFace + GRL)

- **Backbone:** ST-GCN with GConv layers, Multi-Scale TCN (kernels [3, 5, 7]), SpatialNodeDropout
- **Attention Pooling:** Multi-head temporal attention (same as v20)
- **Auxiliary MLP Branch:**
  - Input: 53 hand-crafted features (joint angles + fingertip distances)
  - Architecture: 53 -> 64 -> 32 (with BatchNorm, ReLU, Dropout)
  - Output: 32-dim embedding concatenated with GCN embedding
- **Classification Heads:**
  - Stage 1 (epochs 1-50): Standard CE + SupCon (weight=0.5)
  - Stage 2 (epochs 51-200): ArcFace (s=30, m=0.3) + SupCon (weight=0.3) + GRL
- **Signer-Adversarial (GRL):**
  - Gradient reversal layer with lambda ramp from 0 to 0.1
  - Starts at epoch 50, ramps over 100 epochs
  - Signer classification head for adversarial training

**Config:**
- hidden_dim=64, num_layers=4
- Parameters: 735,520

---

## 3. Key Changes from Previous Version (V20)

| Change | Description | Rationale |
|--------|-------------|-----------|
| **Signer-group dedup** | Keep ~600-650 samples (vs v20's ~240) | Fix overly aggressive deduplication |
| **Joint angles + fingertip distances** | 53 auxiliary features via MLP branch | Capture hand configuration explicitly |
| **ArcFace head** | Angular margin classifier (s=30, m=0.3) in stage 2 | Better class separation in embedding space |
| **SupCon increased** | Stage1=0.5, Stage2=0.3 (was 0.1 in v20) | Stronger cross-signer invariance |
| **GRL signer-adversarial** | Gradient reversal + signer classifier, lambda 0->0.1 | Learn signer-invariant features |
| **Wider augmentation** | Rotation +/-30deg (was 15), shear 0.5 (new), noise 0.03 (was 0.02), bone perturb 0.7-1.3 (was 0.8-1.2), hand size 0.7-1.3, temporal warp sigma 0.3, axis mask (new) | More aggressive signer simulation |
| **Two-stage training** | Stage 1 (CE+SupCon), Stage 2 (ArcFace+SupCon+GRL) | Warm start before margin-based training |
| **Model size increase** | hidden_dim 48->64 (~735K params, up from 413K) | More capacity for auxiliary branch |

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
| arcface_s | 30.0 |
| arcface_m | 0.3 |
| supcon_weight_stage1 | 0.5 |
| supcon_weight_stage2 | 0.3 |
| supcon_temperature | 0.07 |
| grl_lambda_max | 0.1 |
| grl_start_epoch | 50 |
| grl_ramp_epochs | 100 |
| stage2_start_epoch | 50 |
| mixup_alpha | 0.2 |

**Augmentation:**
- Rotation: prob=0.5, max_deg=**30** (2x from v20)
- Shear: prob=0.4, max=**0.5** (NEW)
- Joint dropout: prob=0.3, rate=0.15
- Axis mask: prob=0.1 (NEW)
- Gaussian noise: std=**0.03** (1.5x from v20), prob=0.5
- Bone perturbation: prob=0.5, range=[**0.7, 1.3**] (wider than v20's 0.8-1.2)
- Hand size: prob=0.5, range=[**0.7, 1.3**] (wider than v20's 0.85-1.15)
- Temporal warp: prob=0.5, sigma=**0.3** (1.5x from v20)
- Hand dropout: prob=0.3, min=0.1, max=0.4
- Complete hand drop: prob=0.05

**Hardware:** NVIDIA A100-PCIE-40GB MIG 3g.20gb
**Total training time:** 278.2s (~4.6 minutes) -- fastest version, but numbers early-stopped at epoch 54

---

## 5. Data Pipeline

- **Skeleton extraction:** MediaPipe Holistic (48 landmarks)
- **Input features:** 9 channels (xyz + velocity + bone) for GCN, 53 auxiliary features for MLP
- **Auxiliary features:**
  - Joint angles: angles between connected bone pairs
  - Fingertip distances: pairwise distances between fingertips
  - Fed through separate MLP branch, concatenated with GCN embedding
- **Normalization:** Wrist-centric + palm-size (same as v20)
- **Temporal resampling:** 90 frames
- **Deduplication:** Signer-group level only (keep ~600-650 samples, up from v20's ~240)
- **Classes:** 15 numbers + 15 words

---

## 6. Validation Results

### Numbers (Val: 61.7%) -- CATASTROPHIC REGRESSION
Best epoch: 54 / 200 (early stopped!)

| Class | Val Acc (%) | vs V20 |
|-------|-------------|--------|
| 268 | 100.0 | same |
| 35 | 96.0 | +28pp |
| 9 | 88.0 | -8pp |
| 22 | 84.0 | +44pp |
| 91 | 84.0 | same |
| 100 | 80.0 | +4pp |
| 125 | 80.0 | -4pp |
| 444 | 80.0 | -15pp |
| 17 | 68.0 | -32pp |
| 66 | 68.0 | -32pp |
| 48 | 60.0 | -24pp |
| 73 | 36.0 | -44pp |
| **388** | **0.0** | **-75pp** |
| **54** | **0.0** | **-88pp** |
| **89** | **0.0** | **-68pp** |

Three classes collapsed to 0% val accuracy. The ArcFace transition at epoch 50 destabilized training, and early stopping at epoch 54 means the model barely trained in stage 2. Numbers val dropped from 82.2% to 61.7% (**-20.5pp**).

### Words (Val: 91.3%)
Best epoch: 47 / 200

| Class | Val Acc (%) | vs V20 |
|-------|-------------|--------|
| Agreement | 100.0 | same |
| Apple | 100.0 | same |
| Colour | 100.0 | +4pp |
| Friend | 100.0 | same |
| Teach | 100.0 | same |
| Tomatoes | 100.0 | same |
| Gift | 96.0 | -4pp |
| Tortoise | 96.0 | -4pp |
| Ugali | 96.0 | -4pp |
| Monday | 92.0 | -8pp |
| Sweater | 92.0 | -4pp |
| Twin | 80.0 | same |
| Market | 66.7 | -8.3pp |
| Proud | 52.0 | +12pp |

Words val was roughly stable at 91.3% (-0.3pp from v20). Best epoch 47 means words was also primarily trained in stage 1, before ArcFace kicked in.

---

## 7. Real-Tester Results

### Numbers

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | 25.4% (15/59) | 25.4% (15/59) |

TTA effect: 0.0pp (no change). This is the first version where TTA had zero effect on numbers.

**Per-class (TTA / no-TTA):**

| Class | TTA | No-TTA | vs V20 TTA |
|-------|-----|--------|------------|
| 22 | 50.0% | 75.0% | +50pp |
| 100 | 50.0% | 50.0% | same |
| 444 | 50.0% | 50.0% | same |
| 48 | 50.0% | 50.0% | +50pp |
| 91 | 50.0% | 50.0% | +50pp |
| 35 | 33.3% | 33.3% | same |
| 268 | 25.0% | 25.0% | +25pp |
| 73 | 25.0% | 25.0% | same |
| 89 | 25.0% | 0.0% | +25pp |
| 9 | 25.0% | 25.0% | -25pp |
| 17 | 0.0% | 0.0% | -25pp |
| 125 | 0.0% | 0.0% | same |
| 388 | 0.0% | 0.0% | same |
| 54 | 0.0% | 0.0% | same |
| 66 | 0.0% | 0.0% | -75pp |

Despite catastrophic val regression, real numbers actually improved by +5.1pp. Classes 22, 48, 91 jumped from 0% to 50%. But 66 crashed from 75% to 0%.

### Per-Signer Numbers
| Signer | TTA | No-TTA |
|--------|-----|--------|
| Signer 1 | 27.6% | 31.0% |
| Signer 2 | 20.0% | 20.0% |
| Signer 3 | 26.7% | 20.0% |

More balanced across signers compared to v20.

### Words

| Metric | TTA | No-TTA |
|--------|-----|--------|
| Overall | 37.0% (30/81) | 37.0% (30/81) |

TTA effect: 0.0pp (zero effect on words too).

**Per-class (TTA / no-TTA):**

| Class | TTA | No-TTA | vs V20 TTA |
|-------|-----|--------|------------|
| Gift | 100.0% | 100.0% | +50pp |
| Picture | 75.0% | 75.0% | same |
| Agreement | 66.7% | 66.7% | +66.7pp |
| Teach | 66.7% | 66.7% | -16.6pp |
| Twin | 66.7% | 66.7% | same |
| Colour | 33.3% | 33.3% | +16.6pp |
| Apple | 33.3% | 33.3% | same |
| Friend | 33.3% | 33.3% | -66.7pp |
| Proud | 33.3% | 66.7% | +33.3pp |
| Tortoise | 33.3% | 16.7% | -33.4pp |
| Market | 20.0% | 20.0% | +20pp |
| Monday | 16.7% | 16.7% | -33.3pp |
| Sweater | 0.0% | 0.0% | -33.3pp |
| Tomatoes | 0.0% | 0.0% | same |
| Ugali | 0.0% | 0.0% | -16.7pp |

Words regressed from 40.7% to 37.0% (-3.7pp). Friend crashed from 100% to 33.3%. Gift improved from 50% to 100%.

### Per-Signer Words
| Signer | TTA | No-TTA |
|--------|-----|--------|
| Signer 1 | 28.6% | 28.6% |
| Signer 2 | 52.0% | 52.0% |
| Signer 3 | 32.1% | 32.1% |

### Val-to-Real Gap
- Numbers: 61.7% val -> 25.4% real = **36.3pp gap** (smallest yet, but only because val was terrible)
- Words: 91.3% val -> 37.0% real = **54.3pp gap**

---

## 8. Confidence Analysis

### Numbers Confidence Buckets
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 0.0% | 0 |
| MEDIUM (0.4-0.7) | 25.0% | 4 |
| LOW (<0.4) | 25.5% | 55 |

No HIGH-confidence predictions at all for numbers. 93.2% of predictions fell in LOW bucket. The model is maximally uncertain about numbers.

### Words Confidence Buckets
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 75.0% | 12 |
| MEDIUM (0.4-0.7) | 45.5% | 22 |
| LOW (<0.4) | 23.4% | 47 |

Words HIGH confidence decreased from 86.7% (v20) to 75.0%, with fewer HIGH predictions (12 vs 15). Calibration degraded.

---

## 9. Error Analysis

### ArcFace Catastrophe (Numbers)
The ArcFace transition at epoch 50 was the primary failure mode:
- Numbers early-stopped at epoch 54 (only 4 epochs of stage 2 training)
- The angular margin penalty (m=0.3) was too aggressive for the small dataset
- Three classes (388, 54, 89) collapsed to 0% val accuracy
- The stage 2 learning rate was not reduced, causing instability

### GRL Too Late and Too Weak
- GRL started at epoch 50 (same as ArcFace transition) and ramped slowly over 100 epochs
- lambda_max=0.1 was too weak to force signer-invariant features
- By the time GRL would have been effective, training had already early-stopped

### Augmentation Too Aggressive
- Rotation +/-30deg, shear up to 0.5, noise 0.03 -- significantly wider than v20
- The training data was already small (~600 samples); aggressive augmentation may have made learning harder
- Combined with ArcFace instability, this created a hostile training environment

### Dedup Fix Helped
- Going from ~240 to ~600-650 samples improved numbers real accuracy (+5.1pp)
- This confirms v20's dedup was too aggressive

### Sink Class Shift
- 66 went from a dominant sink in v20 to 0% accuracy in v21
- 444 became the new sink for numbers, with 22 also attracting errors
- This instability across versions suggests sink classes are an artifact of overfitting, not meaningful class confusion

---

## 10. Lessons Learned

1. **ArcFace is catastrophic for small datasets.** The angular margin penalty requires well-separated clusters to begin with. With ~600 samples and high signer variation, ArcFace destabilized training completely. Must be removed for v22.

2. **Two-stage training is fragile.** The transition between stages caused training instability, especially for numbers where the model early-stopped just 4 epochs into stage 2.

3. **GRL needs early start and higher lambda.** Starting at epoch 50 with lambda 0.1 was insufficient. GRL should start from epoch 0/1 with lambda 0.3+ to be effective.

4. **Augmentation can be too aggressive.** Rotation +/-30deg and shear 0.5 with noise 0.03 was excessive. The model struggled to learn basic patterns through so much augmentation.

5. **Val-to-real gap can be misleading in both directions.** V21 numbers had the smallest val-to-real gap (36pp), but only because val accuracy collapsed. Good val-to-real correlation requires both metrics to be meaningful.

6. **Deduplication fix confirmed.** Restoring from 240 to 650 samples was clearly beneficial for real accuracy.

7. **TTA has zero effect** when the model is poorly trained -- it cannot improve fundamentally confused predictions.

8. **Key recommendations for v22:** Remove ArcFace entirely. Single-stage training. GRL from epoch 0 with lambda_max >= 0.3. Reduce augmentation back to v20 levels. Bigger aux branch. Signer-balanced sampling.
