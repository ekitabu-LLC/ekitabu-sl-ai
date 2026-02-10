# Model Version Report: V23

**Date:** 2026-02-10
**Training Script:** `train_ksl_v23.py`
**Result File:** `data/results/v23_both_20260210_024408.json`
**Real-Tester File:** `data/results/v23_real_testers_20260210_035550.json`

---

## 1. Version Summary

V23 was the most architecturally ambitious version: a **multi-stream late fusion ensemble** with 4 independent ST-GCN streams (joint, bone, velocity, bone_velocity), **focal loss** with per-class alpha weighting, **grid-searched fusion weights**, and **post-hoc per-class calibration** (temperature + bias via L-BFGS-B). TTA was disabled based on v22 findings. All other training settings (GRL, SupCon, signer-balanced sampling, reduced augmentation) were carried over from v22.

**Headline result:** Val 95.6% combined (numbers 93.6%, words 97.6%) -- best validation accuracy ever. Real testers: numbers 33.9%, words 42.0%, combined **37.9%** -- WORSE than v22's 39.8%. Despite 4x the parameters (3.1M vs 776K) and significantly higher validation accuracy, the multi-stream ensemble failed to improve real-world performance. V23 confirmed the **4-signer ceiling at ~40%** and demonstrated that fusion of correlated streams provides no benefit on OOD data.

---

## 2. Architecture

**Model:** KSLGraphNetV23 x4 (Multi-Stream ST-GCN Ensemble + Learned Fusion + Calibration)

### Per-Stream Architecture (identical for each stream)
- **Backbone:** ST-GCN with GConv, Multi-Scale TCN (kernels [3, 5, 7]), SpatialNodeDropout
- **Attention Pooling:** Multi-head temporal attention
- **Auxiliary Branch:** 53 -> 128 -> 64 with temporal Conv1D (same as v22)
- **Classification Head:** Standard CE (no ArcFace)
- **SupCon Loss:** weight=0.1, temp=0.07
- **GRL:** lambda_max=0.3, from epoch 0

### 4 Streams (3 channels each)
| Stream | Input Channels | Description |
|--------|---------------|-------------|
| joint | XYZ (3ch) | Raw joint coordinates |
| bone | bone (3ch) | Child-parent joint vectors |
| velocity | velocity (3ch) | Frame-to-frame differences |
| bone_velocity | bone_velocity (3ch) | Frame-to-frame bone changes |

### Fusion
- **Method:** Weighted softmax average of per-stream predictions
- **Weight Search:** Grid search over weights in 0.1 increments on validation set
- **Numbers weights:** joint=0.1, bone=0.4, velocity=0.3, bone_velocity=0.2
- **Words weights:** joint=0.1, bone=0.2, velocity=0.2, bone_velocity=0.5

### Calibration
- **Method:** Per-class temperature + bias, optimized via L-BFGS-B on validation NLL
- **Applied after fusion:** Calibrated logits = logit / temperature + bias (per class)

**Total parameters:** 3,100,800 (775,200 per stream x 4) -- 4x v22

---

## 3. Key Changes from Previous Version (V22)

| Change | Description | Rationale | Result |
|--------|-------------|-----------|--------|
| **Multi-stream** | 4 independent 3ch streams (was single 9ch model) | Diverse feature perspectives | FAILED (no real improvement) |
| **Focal loss** | gamma=2.0, per-class alpha from inverse frequency | Address class imbalance and sink classes | No measurable benefit |
| **Late fusion** | Grid-searched per-stream weights on validation | Combine complementary predictions | FAILED (streams too correlated on OOD) |
| **Post-hoc calibration** | Per-class temperature + bias via L-BFGS-B | Fix confidence miscalibration | FAILED (val calibration doesn't transfer) |
| **TTA disabled** | Based on v22 finding that TTA hurts | Free +1.9% from v22 analysis | DONE (correct decision) |
| **3ch per stream** | Each stream sees only its own feature type | Force learning from different perspectives | Partial (streams learn different features but aren't complementary on OOD) |

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 64 |
| num_layers | 4 |
| streams | joint(3ch), bone(3ch), velocity(3ch), bone_velocity(3ch) |
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
| focal_gamma | 2.0 |
| supcon_weight | 0.1 |
| supcon_temperature | 0.07 |
| grl_lambda_max | 0.3 |
| grl_start_epoch | 0 |
| grl_ramp_epochs | 100 |
| mixup_alpha | 0.2 |

**Augmentation (same as v22):**
- Rotation: prob=0.5, max_deg=15
- Shear: prob=0.3, max=0.2
- Joint dropout: prob=0.2, rate=0.08
- Noise: std=0.015, prob=0.4
- Bone perturbation: prob=0.5, range=[0.8, 1.2]
- Hand size: prob=0.5, range=[0.8, 1.2]
- Temporal warp: prob=0.4, sigma=0.2
- Hand dropout: prob=0.2, min=0.1, max=0.3
- Complete hand drop: prob=0.03

**Hardware:** NVIDIA A100-PCIE-40GB MIG 3g.20gb
**Total training time:** 2119.6s (~35.3 minutes) -- 4.9x slower than v22

---

## 5. Data Pipeline

- **Skeleton extraction:** MediaPipe Holistic (48 landmarks)
- **Input features per stream:** 3 channels each (joint=xyz, bone=bone_vectors, velocity=frame_diff, bone_velocity=bone_frame_diff)
- **Auxiliary features:** Same 53 features (angles + fingertip distances) per stream
- **Normalization:** Wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)
- **Temporal resampling:** 90 frames
- **Deduplication:** Signer-group level (~600-650 samples)
- **Signer-balanced sampling:** Same as v22
- **Classes:** 15 numbers + 15 words
- **Training:** Each of 4 streams trained independently, then fusion weights learned on validation set

---

## 6. Validation Results

### Per-Stream Validation Accuracy (Numbers)

| Stream | Val Acc (%) | Parameters |
|--------|-------------|------------|
| bone | 91.1% | 775,200 |
| velocity | 84.7% | 775,200 |
| joint | 83.6% | 775,200 |
| bone_velocity | 78.6% | 775,200 |

### Per-Stream Validation Accuracy (Words)

| Stream | Val Acc (%) | Parameters |
|--------|-------------|------------|
| bone | 92.1% | 775,200 |
| bone_velocity | 92.1% | 775,200 |
| joint | 88.9% | 775,200 |
| velocity | 88.3% | 775,200 |

### Fused + Calibrated Results

| Split | Uncalibrated | Calibrated | Improvement |
|-------|-------------|------------|-------------|
| Numbers | 91.1% | **93.6%** | +2.5pp |
| Words | 93.75% | **97.6%** | +3.8pp |
| Combined | ~92.4% | **95.6%** | +3.2pp |

**Best validation accuracy ever**, exceeding v22's 93.3% by 2.3pp. Calibration provided ~3pp boost on validation.

### Fusion Weights (learned via grid search)

**Numbers:** bone=0.4, velocity=0.3, bone_velocity=0.2, joint=0.1
**Words:** bone_velocity=0.5, bone=0.2, velocity=0.2, joint=0.1

Bone-based streams dominated both splits. Joint (raw xyz) received the lowest weight in both cases.

---

## 7. Real-Tester Results

### Numbers: 33.9% (20/59) -- TIED with V22

**Per-stream real accuracy (numbers):**

| Stream | Real Acc (%) |
|--------|-------------|
| joint | 37.3% |
| bone_velocity | 35.6% |
| bone | 33.9% |
| velocity | 30.5% |

The joint stream alone (37.3%) **outperformed** the fused result (33.9%). Fusion hurt by 3.4pp on real data despite helping on validation.

**Per-class:**

| Class | V23 | V22 noTTA | Change |
|-------|-----|-----------|--------|
| 22 | 100.0% | 75.0% | +25pp |
| 100 | 50.0% | 50.0% | same |
| 268 | 50.0% | 25.0% | +25pp |
| 388 | 50.0% | 50.0% | same |
| 444 | 50.0% | 50.0% | same |
| 73 | 50.0% | 0.0% | +50pp |
| 17 | 25.0% | 50.0% | -25pp |
| 48 | 25.0% | 0.0% | +25pp |
| 54 | 25.0% | 25.0% | same |
| 66 | 25.0% | 50.0% | -25pp |
| 89 | 25.0% | 25.0% | same |
| 91 | 25.0% | 25.0% | same |
| 125 | 0.0% | 0.0% | same |
| 9 | 0.0% | 75.0% | **-75pp** |
| 35 | 0.0% | 0.0% | same |

Some classes improved (22, 268, 73, 48) while others regressed badly (9: -75pp, 17: -25pp, 66: -25pp). The overall accuracy is identical to v22.

### Per-Signer Numbers
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 31.0% | 9/29 |
| Signer 2 | 46.7% | 7/15 |
| Signer 3 | 26.7% | 4/15 |

Signer 2 improved dramatically to 46.7% (from 33.3% in v22), but Signer 3 dropped from 40.0% to 26.7%.

### Words: 42.0% (34/81) -- WORSE than V22's 45.7%

**Per-stream real accuracy (words):**

| Stream | Real Acc (%) |
|--------|-------------|
| joint | 42.0% |
| bone | 42.0% |
| bone_velocity | 39.5% |
| velocity | 37.0% |

The joint and bone streams (42.0%) **matched** the fused result. Fusion provided zero benefit.

**Per-class:**

| Class | V23 | V22 noTTA | Change |
|-------|-----|-----------|--------|
| Friend | 100.0% | 100.0% | same |
| Gift | 100.0% | 100.0% | same |
| Teach | 83.3% | 100.0% | -16.7pp |
| Tortoise | 66.7% | 66.7% | same |
| Colour | 50.0% | 33.3% | +16.7pp |
| Twin | 50.0% | 66.7% | -16.7pp |
| Sweater | 33.3% | 0.0% | +33.3pp |
| Ugali | 33.3% | 33.3% | same |
| Proud | 33.3% | 33.3% | same |
| Picture | 25.0% | 50.0% | -25pp |
| Market | 20.0% | 0.0% | +20pp |
| Monday | 16.7% | 33.3% | -16.6pp |
| Apple | 16.7% | 33.3% | -16.6pp |
| Agreement | 0.0% | 16.7% | -16.7pp |
| Tomatoes | 0.0% | 20.0% | -20pp |

### Per-Signer Words
| Signer | Accuracy | Correct/Total |
|--------|----------|---------------|
| Signer 1 | 50.0% | 14/28 |
| Signer 2 | 44.0% | 11/25 |
| Signer 3 | 32.1% | 9/28 |

### Val-to-Real Gap
- Numbers: 93.6% val -> 33.9% real = **59.7pp gap** (worst gap ever for numbers)
- Words: 97.6% val -> 42.0% real = **55.6pp gap** (worst gap ever for words)
- Combined: 95.6% val -> 37.9% real = **57.7pp gap**

The gap widened despite (or because of) the elaborate calibration and fusion. Higher val accuracy did not translate to better real-world performance.

---

## 8. Confidence Analysis

### Numbers Confidence Buckets
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 33.3% | 48 |
| MEDIUM (0.4-0.7) | 40.0% | 10 |
| LOW (<0.4) | 0.0% | 1 |

**81.4% of number predictions are HIGH confidence** (up from 18.6% in v22). But accuracy at HIGH is only 33.3% -- worse than v22's MEDIUM bucket (54.5%). The post-hoc calibration inflated confidence without improving accuracy. The model is now more confidently wrong.

### Words Confidence Buckets
| Bucket | Accuracy | Count |
|--------|----------|-------|
| HIGH (>=0.7) | 48.4% | 64 |
| MEDIUM (0.4-0.7) | 21.4% | 14 |
| LOW (<0.4) | 0.0% | 3 |

**79.0% of word predictions are HIGH confidence**, but accuracy dropped from v22's 88.9% to 48.4%. Calibration massively degraded words confidence reliability. The calibration was trained on validation data and does not transfer to real testers.

### Calibration Failure Analysis
The per-class temperature + bias calibration was fit on validation data (from training signers). Real testers have different distributions:
- Example: Class 125 had 3/4 streams predicting correctly, but calibration changed the fused prediction to 100 (wrong)
- Class 66 had calibration temperature 0.21 and bias +0.33, inflating its probability for real testers
- The calibration essentially overfits to validation data patterns that don't generalize

---

## 9. Error Analysis

### Stream Agreement
**Numbers:** Only 11/59 samples (18.6%) had all 4 streams agree. Majority-correct rate was 37.3% (matching joint stream alone).

**Words:** 24/81 samples (29.6%) had all 4 streams agree. Majority-correct rate was 44.4%.

Low agreement indicates the streams are making **different errors** on OOD data, but their errors are not complementary enough for fusion to help.

### Fusion Didn't Help
On real data:
- Joint stream alone: Numbers 37.3%, Words 42.0%
- Fused result: Numbers 33.9%, Words 42.0%

Fusion either matched or degraded performance. The streams are not diverse enough on OOD data -- they all struggle with the same signer variation problem.

### Calibration Distorted Predictions
Multiple examples where uncalibrated predictions were correct but calibration changed them:
- Class 125 samples: 3/4 streams predicted "125" correctly, but post-calibration class 100 bias (+5.0) and class 100 low temperature (0.104) inflated 100's probability, overriding the correct prediction
- The per-class biases (ranging from -5.0 to +5.0) acted as learned priors that were specific to validation data distribution

### New Sink Classes
- **444** became the new numbers sink (replacing 66 in v22)
- **Friend** became a new words attractor
- Sink class identity shifted again, confirming this is an overfitting artifact

### Zero-Accuracy Classes
- Numbers: 125, 9, 35
- Words: Agreement, Tomatoes

### Comparison with V22
| Metric | V22 (noTTA) | V23 | Difference |
|--------|-------------|-----|------------|
| Numbers | 33.9% | 33.9% | 0pp |
| Words | 45.7% | 42.0% | **-3.7pp** |
| Combined | 39.8% | 37.9% | **-1.9pp** |
| Parameters | 776K | 3.1M | +4x |
| Training time | 7.3 min | 35.3 min | +4.8x |
| Num HIGH confidence | 29 | 112 | +3.9x |
| HIGH accuracy | 53.5% avg | 40.9% avg | -12.6pp |

V23 used 4x the parameters, took 5x the training time, and performed worse than v22 on real data.

---

## 10. Lessons Learned

1. **4-signer ceiling confirmed at ~40%.** Both v22 and v23 hit the same wall. Architectural improvements (multi-stream, calibration, focal loss) cannot overcome the fundamental lack of signer diversity.

2. **Multi-stream fusion is not helpful when streams are correlated on OOD data.** All 4 streams struggle with the same signer variation problem. They don't provide complementary information for unseen signers.

3. **Post-hoc calibration on validation does NOT transfer to real testers.** Calibration fit to 4 training signers' validation patterns produces biases that actively hurt when applied to new signers.

4. **Higher validation accuracy does NOT mean better real performance.** V23's 95.6% val vs v22's 93.3% val, but v23's real 37.9% vs v22's 39.8%. The 57.7pp gap is the worst ever, suggesting the additional model complexity enabled more sophisticated overfitting.

5. **Focal loss showed no measurable benefit.** Per-class alpha weighting and gamma=2.0 did not address the core signer generalization problem.

6. **Sink classes just shift identity across versions** (66 -> 444 -> 66 -> 444). This is not a fixable architectural bug; it's a symptom of insufficient signer diversity.

7. **Disabling TTA was correct.** V23 ran without TTA following v22 analysis, avoiding the -1.9pp penalty.

8. **Simplicity wins.** V22 (776K params, 7.3 min, single model) outperformed v23 (3.1M params, 35.3 min, 4-stream ensemble). The marginal complexity yielded negative returns.

9. **The ONLY path to >50% real accuracy is collecting more signers (7-10 needed).** No architectural or training technique explored across v19-v23 has broken the ~40% barrier. The model's failure mode is consistent: it cannot generalize to signing styles it has never seen, regardless of how the features are computed or combined.

10. **Project next steps:** Focus on data collection (more signers), not model architecture. Deploy v22 with confidence-aware UI (reject LOW, show top-3). Accept ~40% as the current ceiling until more diverse training data is available.
