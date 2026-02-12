# V25 Research Plan — Validated & Revised

**Date**: 2026-02-11
**Status**: Post-validation revision (original plan had critical errors)

---

## Validation Summary: 18 Issues Found

Three parallel reviews (ML engineer, peer reviewer, scientific critical thinker) identified **7 critical** and **11 moderate** issues with the original v25 plan. The most important:

### Critical Errors in Original Plan

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Angle features already exist in v22** — 33 joint angles + 20 fingertip distances fed through aux branch. Plan called this "untried" and estimated +3-8pp. | Plan's #1 P0 item is invalid |
| 2 | **Mixup already exists in v22** — `mixup_alpha=0.2`, applied every batch. Plan proposed it as new. | Plan's #4 P0 item is redundant |
| 3 | **Visibility scores not stored in .npy data** — extraction script saves only xyz, discards MediaPipe visibility. Can't filter without re-extraction. | Half of Plan's #2 is infeasible |
| 4 | **Angle features can't replace GCN input** — 33 angle features ≠ 48 skeleton nodes. Dimensionality mismatch with adjacency matrix. | Plan's architecture claim wrong |
| 5 | **Tent TTA pseudocode targets wrong BN type** — v22 has 21 BatchNorm2d + 2 BatchNorm1d. Plan only unfreezes BN1d, missing 91% of BN layers. | Plan's #5 code is broken |
| 6 | **5 training signers, not 4** — Plan repeatedly says "4 signers" but data has 5 unique. Signer2≈Signer3 byte-duplicate was deduped in v20+, but data shows 5 IDs. | Factual error throughout |
| 7 | **Test set too small for claimed precision** — 118 samples, 95% CI ±8.8pp. Cannot reliably detect <8pp improvements. | Ablation strategy unreliable |

### Overestimated Impact (Original vs. Revised)

| Technique | Original Est. | Revised Est. | Reason |
|-----------|--------------|-------------|--------|
| Angle features | +3-8pp | +0-1pp | Already in v22 aux branch |
| Preprocessing | +1-3pp | +0-2pp | No visibility data; filter risk |
| V-REx | +2-5pp | +0-2pp | Noisy with ~6 samples/signer/batch |
| Mixup/CutMix | +2-4pp | +0-1pp | Mixup already in v22 |
| Tent TTA | +5-10pp | +0-3pp | Medical imaging analogy doesn't transfer; small test batches |
| Per-signer experts | +2-5pp | -2-0pp | 150 samples per model = severe overfit |
| **Combined** | **50-60%** | **40-45%** | Gains don't stack; v23/v24 proved this |

### Missing from Original Plan

1. **No diagnostics before treatment** — plan doesn't measure whether its assumptions hold
2. **No LOSO cross-validation** — primary metric should be LOSO (4x more test data)
3. **No side-view analysis** — Signer 3 numbers filmed from side, never addressed
4. **No MediaPipe version consistency** — training vs eval version mismatch
5. **No sink-class analysis** — every version creates different degenerate attractors
6. **No evaluation pipeline updates** — 900+ line eval script must match training changes
7. **No numbers/words divergence analysis** — v24 showed they can move in opposite directions

---

## Revised V25 Plan: Diagnostics-First

### Philosophy Change

The original plan was a **literature review organized as an implementation order**. The revised plan is a **diagnostic-driven engineering plan**. We measure first, then intervene.

---

## Phase 0: Diagnostics (2-3 days, BEFORE any code changes)

These take hours, cost nothing, and determine whether each intervention has potential.

### D1. Per-Signer Loss Variance at V22 Convergence

**Why**: V-REx only helps if per-signer loss variance is high. V22 already uses GRL + signer-balanced sampling. If these already equalize per-signer losses, V-REx has nothing to penalize.

**How**: Load v22 checkpoint, compute CE loss separately for each of the 5 training signers on training data. Report mean and variance.

**Decision**: If variance < 0.01 → skip V-REx. If variance > 0.05 → implement V-REx.

### D2. Feature Visualization (t-SNE/UMAP)

**Why**: Determines whether the domain gap is in feature space or classifier space.

**How**: Extract 320-dim embeddings from v22 for all training samples + all real-tester samples. Plot t-SNE colored by (a) signer, (b) class. Check if training signers cluster separately from test signers. Check if GRL actually removed signer information.

**Decision**: If training/test signers overlap in feature space → problem is classifier, not features. If they're clearly separated → feature-level interventions needed.

### D3. BatchNorm Statistics Comparison

**Why**: Determines whether Tent TTA can help. If BN running stats already match test-time activations, Tent has nothing to correct.

**How**: Forward-pass real-tester data through v22. Compare per-layer activation distributions (mean, var) against training-time BN running stats. Quantify distributional shift per layer.

**Decision**: If shift > 0.5 std in multiple layers → Tent has potential. If shift < 0.2 std → Tent won't help.

### D4. Landmark Quality Comparison

**Why**: Determines whether preprocessing matters.

**How**: Compare MediaPipe landmark jitter (frame-to-frame L2 displacement) between training data and real testers. Compare per-joint noise levels.

**Decision**: If test-time jitter is 2x+ training jitter → Butterworth filter justified. If similar → skip preprocessing.

### D5. Side-View Impact Quantification

**Why**: Signer 3 numbers are filmed from the side. This is a camera-angle problem, not a signer problem.

**How**: Report v22 accuracy separately for Signer 3 numbers vs. other signers. Compute how much combined accuracy changes if Signer 3 numbers are excluded.

**Decision**: If Signer 3 numbers drag accuracy by >5pp → add view-angle augmentation to training pipeline.

### D6. Sink Class Analysis

**Why**: Every version creates different "attractor" classes (v22: 66/Tortoise, v23: 444/Friend, v24: 444/Friend). Understanding why reveals whether this is calibration, feature collapse, or class proximity.

**How**: Analyze v22 confusion matrix on real testers. For each sink class, check: (a) what classes get misclassified as the sink, (b) feature space proximity of those classes to the sink, (c) calibration of predictions for the sink class.

---

## Phase 1: Targeted Interventions (based on diagnostic results)

Only implement techniques whose diagnostics show potential.

### I1. V-REx Loss Penalty (IF D1 shows high variance)

**What changed from original plan**: Use 5 signers (not 4). Increase batch size or use gradient accumulation to get 10+ samples per signer per batch (reduces CoV from 0.57 to ~0.35). Start with small beta (0.1) and search {0.01, 0.1, 0.5, 1.0}.

```python
# Corrected implementation
per_signer_losses = [CE(signer_k_samples) for k in range(5)]  # 5 signers
vrex_penalty = torch.var(torch.stack(per_signer_losses))
total_loss = CE + 0.1*SupCon + GRL + beta * vrex_penalty
```

**Expected impact**: +0-2pp
**Effort**: 1 day

### I2. Butterworth Smoothing (IF D4 shows high jitter)

**What changed from original plan**: Dropped visibility filtering (not feasible without re-extraction). Filter must be applied BEFORE velocity computation and BEFORE temporal resampling. Frame-rate varies per recording — use adaptive cutoff frequency.

**Risk acknowledged**: Low-pass filtering may destroy rapid hand shape transitions that distinguish confusable signs (e.g., numbers 35 vs 48). Must validate on training data first.

```python
from scipy.signal import butter, filtfilt

# Per-landmark trajectory smoothing
# Apply to raw (T, 48, 3) BEFORE velocity/bone computation
fps = raw_frames / duration  # varies per recording
nyquist = fps / 2
cutoff = min(8.0, nyquist * 0.4)  # adaptive cutoff
b, a = butter(3, cutoff / nyquist, btype='low')
for node in range(48):
    for dim in range(3):
        h[:, node, dim] = filtfilt(b, a, h[:, node, dim])
```

**Expected impact**: +0-2pp
**Effort**: 1 day (including eval pipeline update)

### I3. Temporal CutMix (genuinely new, unlike Mixup)

**What changed from original plan**: Removed Mixup (already in v22 with alpha=0.2). Only CutMix is truly new. Must handle velocity discontinuities at splice boundaries by recomputing velocity AFTER splicing.

```python
# CutMix for skeleton sequences
# Splice temporal segment from signer B into signer A's sequence
if np.random.random() < 0.3:  # p=0.3
    cut_start = np.random.randint(0, T - cut_len)
    h[cut_start:cut_start+cut_len] = h_other[cut_start:cut_start+cut_len]
    # CRITICAL: recompute velocity after splice to avoid artifacts
    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
```

**Expected impact**: +0-1pp
**Effort**: 0.5 days

### I4. Tent TTA v2 (IF D3 shows BN distributional shift)

**What changed from original plan**: Fixed to target BatchNorm2d (21 layers), not just BatchNorm1d (2 layers). Require minimum batch size of 15+ samples. Use SAR (sharpness-aware) variant for stability. Lowered expected impact from +5-10pp to +0-3pp.

```python
# Corrected Tent implementation
model.train()
for param in model.parameters():
    param.requires_grad = False
for module in model.modules():
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):  # BOTH types
        module.weight.requires_grad = True
        module.bias.requires_grad = True

# Process ALL samples from one signer as a single batch
# Minimum 15 samples for stable BN estimation
signer_batch = collate(all_signer_samples)  # 15-29 samples
for step in range(3):  # few adaptation steps
    logits = model(signer_batch)
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    entropy.backward()
    optimizer.step()  # lr=1e-4 (10x smaller than training)
```

**Expected impact**: +0-3pp (not +5-10pp as originally claimed)
**Effort**: 3 days (including eval pipeline restructuring)

### I5. View-Angle Augmentation (IF D5 shows side-view impact)

**Not in original plan**. Discovered by peer review.

**What to do**: Rotate the 3D skeleton around the Y-axis (vertical) by ±30-45° during training. This simulates non-frontal camera angles, directly addressing Signer 3's side-view recordings.

```python
# Rotation around Y-axis (simulates camera viewpoint change)
if np.random.random() < 0.3:
    theta = np.radians(np.random.uniform(-45, 45))
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    h = h @ R.T  # (T, 48, 3) @ (3, 3)
```

**Expected impact**: +1-3pp (directly addresses known test-set issue)
**Effort**: 0.5 days

### I6. Canonical Skeleton Alignment

**What to do**: Before all other processing, rotate each frame so shoulders are horizontal and spine is vertical. This removes camera-angle variance at the source.

**Expected impact**: +0-2pp
**Effort**: 1 day

---

## Phase 2: Evaluation Infrastructure

### E1. LOSO Cross-Validation as Primary Metric

**Why**: Real-tester accuracy has ±8.8pp CI with 118 samples. LOSO on training data gives ~150 test samples per fold (5 folds × 150 samples), tighter CIs, and faster iteration.

**What**: Implement leave-one-signer-out CV during training. Report mean±std accuracy across 5 folds. Use this as the primary development metric, with real-tester accuracy as the final validation.

### E2. Bootstrap Confidence Intervals

**What**: For all real-tester evaluations, report bootstrap 95% CI alongside point estimates. Any improvement smaller than the CI should be reported as "within noise."

### E3. Evaluation Pipeline Parity

**What**: Any preprocessing change in training must be exactly replicated in `evaluate_real_testers_v25.py`. Create shared preprocessing functions imported by both scripts.

---

## Revised Timeline

```
Phase 0: Diagnostics (Days 1-3)
  Day 1: D1 (per-signer loss) + D2 (feature t-SNE) + D3 (BN stats)
  Day 2: D4 (landmark quality) + D5 (side-view) + D6 (sink class)
  Day 3: Analyze results, decide which interventions to pursue

Phase 1: Interventions (Days 4-8, only those justified by diagnostics)
  Day 4: I5 (view-angle aug) + I6 (canonical alignment)
  Day 5: I1 (V-REx, if justified) + I3 (CutMix)
  Day 6: I2 (Butterworth, if justified)
  Day 7-8: I4 (Tent TTA, if justified)

Phase 2: Evaluation (Days 9-10)
  Day 9: LOSO CV implementation + full ablation
  Day 10: Real-tester evaluation with bootstrap CIs
```

---

## Revised Expected Outcomes

| Scenario | Est. Real Combined | Basis |
|----------|-------------------|-------|
| V22 baseline | 39.8% | Measured |
| Conservative (1-2 techniques work) | 41-43% | +1-3pp, within CI |
| Moderate (3-4 techniques work) | 43-46% | +3-6pp, barely detectable |
| Optimistic (everything works + stacks) | 46-49% | +6-9pp, would be unprecedented |
| **To reach 85%** | **Need 7-10 signers** | Literature consensus |

**Honest assessment**: Based on v22-v24 history where every "should work" technique produced zero or negative real-world gain, the most likely outcome of v25 is **+0-5pp** over v22. The diagnostics phase will tell us whether even this is achievable before we invest in implementation.

---

## Items Dropped from Original Plan

| Item | Reason for dropping |
|------|-------------------|
| Angle features as GCN input | Already in v22 aux branch; dimensionality mismatch with 48-node GCN |
| Visibility filtering | Visibility scores not stored in .npy data |
| Mixup | Already in v22 (alpha=0.2) |
| Per-signer expert ensemble | 150 samples/model = severe overfit |
| Contrastive pre-training | Too speculative for 750 samples |
| MLDG meta-learning | Diminishing returns with <6 domains |
| Generative augmentation | Insufficient source data to train generative model |

---

## Items Added (Not in Original Plan)

| Item | Source |
|------|--------|
| Diagnostic pipeline (D1-D6) | Peer review: "diagnostic before treatment" |
| LOSO as primary metric | Peer review: CI analysis |
| View-angle augmentation | Peer review: side-view gap |
| Canonical skeleton alignment | Peer review: camera-angle normalization |
| Bootstrap CIs | Statistical review: 118 samples too small |
| Evaluation pipeline parity | ML engineer review: eval script divergence risk |

---

## Key References

*Same as original plan, plus:*

- Wang et al. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR.
- Krueger et al. (2021). Out-of-Distribution Generalization via Risk Extrapolation (V-REx). ICML.
- Wilson (1927). Probable Inference, the Law of Succession, and Statistical Inference. JASA. (for CI calculation)
