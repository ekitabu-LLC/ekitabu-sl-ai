# V25 Research Plan: Breaking the 4-Signer Ceiling

**Date**: 2026-02-11
**Based on**: Deep literature review across 5 parallel research threads
**Goal**: Push real-tester accuracy from ~40% (v22) toward 55-60% without collecting new signers

---

## Executive Summary

V22-V24 confirmed a **37-40% ceiling** on real-world testers with 4 training signers. Architecture changes (multi-stream ensemble, prototypical networks, MixStyle) gave zero net improvement. Literature review across signer-independent SLR, domain generalization, skeleton augmentation, and test-time adaptation reveals **three untried high-impact strategies**:

1. **Angle-based signer-invariant features** — replace raw coordinates with joint angles that are inherently body-proportion-independent
2. **Test-Time Adaptation (TTA v2)** — modern entropy-minimization TTA (Tent), not the naive augmentation-TTA that hurt v22
3. **V-REx variance penalty** — explicitly penalize risk variance across signers during training

Combined with improved preprocessing (Butterworth smoothing, visibility filtering) and sequence-level augmentation (Mixup/CutMix), we estimate a realistic target of **50-55% combined real-tester accuracy** — a 10-15pp improvement over v22.

---

## What We Know (Lessons from v19-v24)

| Version | Architecture Change | Real Combined | Verdict |
|---------|-------------------|---------------|---------|
| v19 | Baseline | 34.2% | — |
| v20 | Aggressive dedup | 30.5% | Hurt |
| v21 | ArcFace | 31.2% | Catastrophic |
| v22 | No ArcFace, strong GRL, reduced aug | **39.8%** | **BEST** |
| v23 | Multi-stream ensemble (3.1M params) | 37.9% | No help |
| v24 | Prototypical + MixStyle | 36.8% | Worse |

**Key insight**: Removing complexity (v22) worked better than adding it (v23, v24). V25 must focus on **data representation and training objectives**, not architecture scale.

---

## V25 Strategy: Three Pillars

### Pillar 1: Signer-Invariant Feature Representation
### Pillar 2: Better Training Objectives (V-REx + Improved SupCon)
### Pillar 3: Test-Time Adaptation v2 (Tent)

Plus supporting improvements in preprocessing and augmentation.

---

## P0 — Implement First (High Impact, Low Effort)

### 1. Angle-Based Skeleton Features

**Source**: Du et al. (2025, Computer Animation and Virtual Worlds); Aiman & Ahmad (2023)

**Key finding**: "Selecting angles between skeleton vectors effectively mitigates inter-individual differences due to limb proportion variations." First application to skeleton-based recognition in 2025.

**What to do**:
- Compute **joint angles** at all major articulation points (elbows, shoulders, wrists, finger joints)
- Compute **inter-landmark distances** normalized by a reference length (e.g., palm width for hands, shoulder width for body)
- Replace or augment the current 9-channel input (xyz + velocity + bone) with angle-based features
- Target: expand from 3 base features to ~15-25 features per joint

**Why it should work**: Raw xyz coordinates encode body proportions (arm length, hand size) which differ per signer. Joint angles encode **pose geometry** which is signer-invariant for the same sign. This directly attacks the root cause of the domain gap.

**Implementation**:
```python
# Current: 9ch = xyz(3) + velocity(3) + bone(3)
# V25: 15-25ch = angles(6-10) + normalized_distances(4-6) + velocity_angles(3-5) + bone_angles(3-5)
# Keep velocity and bone but compute in angle space too
```

**Expected impact**: +3-8pp (addresses signer body-proportion variance directly)
**Effort**: 2-3 days
**Risk**: Low — pure feature engineering, no architecture change needed

---

### 2. MediaPipe Preprocessing Improvements

**Source**: Conelea et al. (2023, Movement Disorders); Yoon et al. (2023, JCAL)

**What to do**:
- **Butterworth low-pass filter** (3rd order, critical freq 0.25) on landmark trajectories to remove jitter
- **Visibility-based filtering**: weight or mask landmarks with visibility < 0.5
- **Diagnostic**: Compare landmark quality (visibility scores, jitter magnitude) between training signers and real testers to quantify preprocessing-related domain gap

**Why it should work**: MediaPipe produces noisy landmarks with unnatural jitters. Different recording conditions (lighting, camera, distance) affect landmark quality differently per signer. Smoothing reduces this source of variance.

**Current state**: V22 has no temporal smoothing of raw landmarks.

**Expected impact**: +1-3pp
**Effort**: 1 day
**Risk**: Very low — preprocessing only

---

### 3. V-REx (Variance Risk Extrapolation)

**Source**: Krueger et al. (2021); validated in few-domain settings by Lteif et al. (2024)

**Key idea**: Add a penalty term that minimizes the **variance of per-signer losses** during training. This forces the model to find solutions that work equally well across all signers, not just on average.

**What to do**:
```python
# Current loss: CE + 0.1*SupCon + GRL
# V25 addition:
per_signer_losses = [CE(signer_k_samples) for k in signers]
vrex_penalty = torch.var(torch.stack(per_signer_losses))
total_loss = CE + 0.1*SupCon + GRL + beta * vrex_penalty
# beta: ramp from 0 to 1.0 over first 50 epochs
```

**Why it should work**: Current training minimizes average loss, allowing the model to "sacrifice" hard signers. V-REx explicitly prevents this. With only 4 signers, per-signer loss variance is a meaningful and cheap signal.

**Interaction with GRL**: V-REx is complementary to GRL. GRL makes features signer-invariant; V-REx makes the classifier perform uniformly across signers. They attack different aspects of the problem.

**Expected impact**: +2-5pp
**Effort**: 1 day (single penalty term)
**Risk**: Low — well-established technique, simple to implement and tune

---

### 4. Sequence Mixup and CutMix

**Source**: Demir et al. (2023, Expert Systems); Jeong et al. (2025, ETRI Journal)

**What to do**:
- **Mixup**: Linearly interpolate between two skeleton sequences from different signers: `x_new = λ*x_i + (1-λ)*x_j`, with cross-entropy on mixed labels
- **CutMix**: Replace a temporal segment of one sequence with a segment from another signer performing the same or different sign
- Apply with p=0.3-0.5 during training

**Why it should work**: Creates "virtual signers" by blending characteristics of existing signers. Unlike spatial augmentation (already in v22), this operates in the **identity mixing** dimension. Mixup is proven not to hurt calibration (Demir et al. 2023). CutMix for temporal data "significantly improves classification accuracy" (Jeong et al. 2025).

**Implementation note**: For Mixup, interpolate in feature space (after preprocessing, before GCN) for better results. For CutMix, cut/paste in raw skeleton space.

**Expected impact**: +2-4pp
**Effort**: 1-2 days
**Risk**: Low — standard technique, easy to ablate

---

## P1 — Implement Next (High Impact, Medium Effort)

### 5. Test-Time Adaptation v2 (Tent)

**Source**: Wang et al. (2021 - Tent); Niu et al. (2024 - SAR); Qian et al. (2024 - single-sample TTA)

**IMPORTANT**: This is NOT the same as v22's TTA (which was naive augmentation-based and hurt by -1.9pp). Tent-style TTA **adapts the model's batch normalization statistics** to each test sample by minimizing prediction entropy.

**What to do**:
```python
# At test time, for each new signer's batch of samples:
model.train()  # Enable BN stats tracking
for param in model.parameters():
    param.requires_grad = False
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.weight.requires_grad = True
        module.bias.requires_grad = True

# Adapt on test samples (no labels needed)
for x in test_signer_samples:
    logits = model(x)
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    entropy.backward()
    optimizer.step()
```

**Why it should work**: BN statistics encode domain-specific information. Updating them at test time to match the new signer's distribution is the fastest way to reduce domain gap. Medical imaging TTA shows **+10-15pp absolute improvement** with single-sample adaptation.

**Key difference from v22 TTA**: V22 TTA averaged predictions over augmented versions of the same input (noisy). Tent TTA adapts the model itself to the test distribution (fundamentally different mechanism).

**Deployment consideration**: In production, can buffer first 5-10 predictions from a new user, run Tent adaptation, then re-predict with adapted model. Or adapt continuously.

**Expected impact**: +5-10pp
**Effort**: 3-5 days (careful implementation + validation)
**Risk**: Medium — need to prevent catastrophic forgetting; use small learning rate + SAR filtering

---

### 6. Per-Signer Expert Ensemble (Multisource Transfer)

**Source**: Gao et al. (2022); validated in domain generalization literature

**Key idea**: Instead of one model trained on all 4 signers, train **4 specialist models** (one per signer) + 1 generalist. At test time, weight predictions by similarity of test sample to each signer's distribution.

**What to do**:
```
Model_1: trained on Signer1 data only
Model_2: trained on Signer2 data only
Model_3: trained on Signer3 data only
Model_4: trained on Signer4 data only
Model_G: trained on all signers (current v22)

Prediction = Σ w_k * Model_k(x) + w_G * Model_G(x)
where w_k = softmax(similarity(x, Signer_k_prototype))
```

**Why it should work**: V23's multi-stream fusion failed because the streams were architecturally different views of the **same data**. Per-signer experts are trained on **genuinely different distributions**. A new test signer might match one training signer's style more than others.

**Expected impact**: +2-5pp
**Effort**: 1 week (train 4+1 models, implement fusion)
**Risk**: Medium — may not help if test signers are equally dissimilar to all training signers

---

### 7. Enhanced Signer-Invariant Normalization

**Source**: Ansari et al. (2023); Goel et al. (2023); Li & Rajesh (2023)

**What to do** (building on v22's wrist-centric normalization):
- **Torso-length normalization**: Divide all body landmarks by shoulder-to-hip distance
- **Palm-width normalization**: Divide all hand landmarks by palm width (wrist-to-middle-finger-base distance)
- **Canonical alignment**: Rotate skeleton so shoulders are always horizontal, spine is always vertical
- **Remove global translation**: All frames relative to hip midpoint

**Current v22 state**: Already uses wrist-centric + palm-size normalization for hands, mid-shoulder + shoulder-width for pose. But does NOT do canonical rotation alignment or torso-length normalization of body joints.

**Expected impact**: +1-3pp
**Effort**: 2-3 days
**Risk**: Low

---

## P2 — Explore Later (Medium Impact, Higher Effort)

### 8. Contrastive Cross-Signer Pre-training

**Source**: Li et al. (2025 - S2C-HAR); Dixon et al. (2024 - MACL)

**Key idea**: Use the different pose modalities (joint, bone, velocity) as natural "views" for contrastive learning. Add a cross-signer contrastive objective: pull together embeddings of the same sign across different signers, push apart different signs regardless of signer.

**What to do**:
- Phase 1: Self-supervised pretraining with multi-view contrastive loss (no labels)
- Phase 2: Fine-tune with labeled data + V-REx

**Expected impact**: +3-6pp
**Effort**: 1-2 weeks
**Risk**: Medium-high — self-supervised methods can be finicky with small datasets

---

### 9. MLDG (Meta-Learning Domain Generalization)

**Source**: Li et al. (2018); tested with 4 domains by Lteif et al. (2024)

**Key idea**: Split 4 signers into meta-train (3 signers) and meta-test (1 signer) during each training iteration. Optimize for generalization to the held-out signer.

**What to do**:
```python
for each iteration:
    meta_train_signers = random.sample(signers, 3)
    meta_test_signer = remaining_signer

    # Inner loop: train on 3 signers
    theta_prime = theta - lr * grad(loss_train(theta))

    # Outer loop: optimize for generalization to held-out signer
    theta = theta - meta_lr * grad(loss_test(theta_prime))
```

**Expected impact**: +2-4pp
**Effort**: 1 week
**Risk**: Medium — computational overhead, requires careful tuning

---

### 10. Generative Skeleton Augmentation (Diffusion-Based)

**Source**: Muhammad Kamal et al. (2025); Tevet et al. (2023 - MDM)

**Key idea**: Use a motion diffusion model to generate synthetic skeleton sequences that look like new signers performing existing signs.

**Approach**:
- Pre-train on a public motion dataset (HumanML3D, NTU RGB+D)
- Fine-tune on KSL data conditioned on sign class labels
- Generate 100-500 synthetic sequences per class

**Expected impact**: +5-10pp (highly uncertain)
**Effort**: 2-4 weeks
**Risk**: High — small source dataset may not train good generative model; generated samples may amplify existing biases

---

## Implementation Order & Timeline

```
Week 1: P0 items (Features + Preprocessing + V-REx + Mixup)
  Day 1-2: Angle-based features (Item 1)
  Day 2:   Butterworth smoothing + visibility filtering (Item 2)
  Day 3:   V-REx loss penalty (Item 3)
  Day 3-4: Sequence Mixup/CutMix (Item 4)
  Day 4-5: Train & evaluate on real testers

Week 2: P1 items (TTA v2 + Normalization)
  Day 1-3: Tent TTA implementation (Item 5)
  Day 3-4: Enhanced normalization (Item 7)
  Day 4-5: Per-signer expert ensemble (Item 6)
  Day 5:   Final evaluation with all P0+P1

Week 3+: P2 items if needed
  Contrastive pre-training, MLDG, generative augmentation
```

---

## Expected Accuracy Progression

| Stage | Est. Real Combined | Delta |
|-------|-------------------|-------|
| V22 baseline | 39.8% | — |
| + Angle features | 43-47% | +3-8pp |
| + Preprocessing | 44-49% | +1-3pp |
| + V-REx | 46-52% | +2-5pp |
| + Mixup/CutMix | 48-54% | +2-4pp |
| + Tent TTA v2 | 53-60% | +5-10pp |
| + Per-signer experts | 55-62% | +2-5pp |

**Realistic target: 50-55% combined** (conservative, assuming gains don't fully stack)
**Optimistic target: 55-60%** (if Tent TTA works as well as in medical imaging literature)

**To reach 85% target**: Still requires 7-10 training signers (literature consensus).

---

## Ablation Strategy

Each component should be tested independently against the v22 baseline:

1. v22 + angle features only → measure delta
2. v22 + preprocessing only → measure delta
3. v22 + V-REx only → measure delta
4. v22 + Mixup only → measure delta
5. v22 + all P0 → measure combined effect
6. v22 + all P0 + Tent TTA → measure TTA delta on top of P0

This tells us which components actually help and which can be dropped.

---

## Key References

**Signer-Invariant Features:**
- Du et al. (2025). Skeleton angle features. Computer Animation & Virtual Worlds. DOI: 10.1002/cav.70073
- Aiman & Ahmad (2023). Angle-based hand gesture GCN. DOI: 10.1002/cav.2207

**Domain Generalization:**
- Krueger et al. (2021). V-REx: Variance Risk Extrapolation.
- Li et al. (2018). MLDG: Meta-Learning Domain Generalization.
- Lteif et al. (2024). Disease-driven DG with 4 cohorts. Human Brain Mapping.

**Test-Time Adaptation:**
- Wang et al. (2021). Tent: Fully test-time adaptation.
- Niu et al. (2024). SAR: Sharpness-aware reliable entropy.
- Qian et al. (2024). Adaptive single-sample TTA. Medical Physics.

**Augmentation:**
- Santiago & Cifuentes (2024). Spatial augmentation for skeleton. Expert Systems. DOI: 10.1111/exsy.13706
- Li et al. (2025). S2C-HAR: Semi-supervised with contrastive. DOI: 10.1002/cpe.70027
- Demir et al. (2023). Mixup/CutMix calibration. Expert Systems.

**Preprocessing:**
- Conelea et al. (2023). Butterworth smoothing for MediaPipe. Movement Disorders. DOI: 10.1002/mds.29593
- Yoon et al. (2023). Visibility-based filtering. JCAL. DOI: 10.1111/jcal.12933

**Cross-Signer SLR:**
- Wei et al. (2024). Fuzzy encoding for cross-user generalization. InfoMat.
- Gao et al. (2022). Multisource transfer learning.
- Liu et al. (2022). Few-shot with MediaPipe + autoencoder.
- Muhammad Kamal et al. (2025). Diffusion for CSLR. DOI: 10.1002/cpe.8385
