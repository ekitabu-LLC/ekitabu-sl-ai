# V25 Final Plan — Post-Critique Synthesis

**Date**: 2026-02-11
**Status**: Final (synthesized from 4 independent critic reviews)
**Baseline**: V22 real-tester accuracy = 39.8% combined (33.9% numbers, 45.7% words)

---

## Executive Summary

Four independent critics (ML methodology, statistics, domain expertise, implementation) reviewed the validated v25 plan. Their consensus is brutal but clarifying:

1. **The 4-signer ceiling is real.** No architecture or training trick has moved real-tester accuracy beyond ~40% across 6 versions (v19-v24). The critics agree: only more signer diversity can break this.
2. **Most proposed interventions are dead.** V-REx, stronger regularization, rotation augmentation, and LOSO-as-primary-metric were all killed with evidence.
3. **Three genuine opportunities survived**: hand-to-body spatial features (new signal), MediaPipe version alignment (free fix), and CutMix (genuinely untried augmentation).
4. **Statistical reality**: Our real test set (n=140) cannot detect improvements smaller than 16pp. All v19-v24 results are statistically indistinguishable.

**Honest expected outcome**: +0-5pp over v22 (39.8% -> 40-45%). Most likely: +0-2pp.

---

## What Was Killed (and Why)

### K1. V-REx Loss Penalty -- KILLED
- **ML Critic**: V22 already has GRL + signer-balanced sampling. Signer silhouette = 0.006 (features already signer-invariant). V-REx penalizes variance that GRL already removed.
- **Impl Critic**: Only ~8 samples/signer/batch with batch_size=32. Noisy gradient signal.
- **Statistician**: Can't detect +0-2pp gain with n=140. Effort wasted even if it works.
- **Verdict**: Skip entirely. GRL already does what V-REx would do.

### K2. Stronger Regularization -- KILLED
- **ML Critic**: Model has 7+ regularizers already. Val > train accuracy in some cases means the model is UNDER-fitting, not overfitting. The 53.5pp gap is domain shift, not overfitting (train-val gap is only 6.7pp).
- **Impl Critic**: Confirmed — adding regularization would be counterproductive.
- **Verdict**: Wrong diagnosis. The problem is distribution shift, not overfitting.

### K3. Y-Axis Rotation Augmentation -- KILLED
- **ML Critic**: V22 already has Z-rotation +/-15 degrees. Y-axis rotation would mix unreliable MediaPipe Z-depth (relative, not absolute) into the reliable X coordinate.
- **Domain Expert**: Palm orientation is a phonological parameter in KSL. Y-axis rotation >15 degrees on hands would change sign meaning (e.g., "Agreement" vs "Gift" differ by palm facing).
- **Impl Critic**: MediaPipe outputs 2.5D (x,y = image plane, z = relative depth). Y-axis rotation is physically invalid.
- **Verdict**: Sounds good on paper, breaks in practice. MediaPipe Z is not real 3D.

### K4. LOSO as Primary Metric -- DEMOTED
- **ML Critic**: n=1 per fold (one signer held out). 95% CI of +/-5pp per fold. After averaging 4 folds, resolution is ~10pp. Cannot use for hyperparameter selection.
- **Statistician**: Folds are NOT independent (share 4/5 training data). True resolution probably ~10pp. Only usable for coarse GO/NO-GO decisions.
- **Impl Critic**: Signers 2 and 3 are byte-identical duplicates (MD5 verified). After dedup, only 4 unique signers for numbers. LOSO is 4-fold, not 5-fold.
- **Verdict**: Use as sanity check only. NOT for comparing configurations. If LOSO says a change is catastrophic (<25%), abort. Otherwise, results are noise.

### K5. Tent TTA -- KILLED (conditional revival possible)
- **ML Critic**: Requires large enough batches for stable BN stats. Real testers have 15-29 samples per signer, which is marginal.
- **Statistician**: Cannot detect the +0-3pp it might provide.
- **Impl Critic**: Complex implementation (3 days), eval pipeline restructuring required. High effort, uncertain payoff.
- **Verdict**: Only revisit if D3 diagnostic shows massive BN distributional shift (>0.5 std in multiple layers). Otherwise skip.

### K6. Canonical Skeleton Alignment -- KILLED
- **Domain Expert**: After wrist-centric normalization, the skeleton is already locally aligned. Global alignment (shoulders horizontal) would only help if camera angle varies significantly across samples, which overlaps with the (killed) rotation augmentation.
- **ML Critic**: Redundant with existing normalization. Pose nodes are already centered at mid-shoulder and scaled by shoulder width.
- **Verdict**: Marginal benefit, adds preprocessing complexity.

### K7. Butterworth Smoothing -- KILLED
- **Domain Expert**: MediaPipe jitter is NOT the bottleneck (confirmed by D4 diagnostic design). The gap is genuine signer variation, not measurement noise.
- **ML Critic**: Low-pass filtering risks destroying rapid hand shape transitions that distinguish confusable signs.
- **Verdict**: Fixing noise that isn't the problem. Skip.

### K8. Per-Class "Fixes" for Zero-Accuracy Classes -- KILLED
- **Statistician**: With 3-4 samples per class, 0% -> 25% is ONE sample. "Fix zero-accuracy classes" is statistically meaningless.
- **ML Critic**: Sink classes change every version (v22: 66/Tortoise, v23: 444/Friend, v24: 444/Friend). They are a symptom of limited signer diversity, not a fixable bug.
- **Verdict**: Chasing per-class accuracy with n=3 is noise-fitting.

---

## What Survived (3 Items)

### S1. Hand-to-Body Spatial Features (NEW SIGNAL) -- HIGHEST PRIORITY

**Source**: Domain Expert Critic (only critic to identify genuinely new information).

**The Problem**: After wrist-centric normalization, the spatial relationship between hands and body is DISCARDED. In sign language, WHERE a sign is performed (face level, chest level, waist level) is as important as hand shape. This is a known phonological parameter called "location" in SL linguistics.

**What V22 Loses**: When each hand is normalized to its own wrist origin, we lose:
- Hand centroid height relative to face/shoulders
- Inter-hand distance
- Hand position relative to signing space (face, chest, waist)
- These are precisely the features that distinguish signs like Market vs Tortoise, or numbers that share hand shapes but differ in location.

**Implementation**:
```python
def compute_hand_body_features(h_raw):
    """
    Compute hand-to-body spatial features BEFORE wrist-centric normalization.
    h_raw: (T, 48, 3) -- raw landmarks, not yet normalized

    Returns: (T, 8) feature vector per frame
    """
    T = h_raw.shape[0]
    features = np.zeros((T, 8), dtype=np.float32)

    # Body reference points (pose nodes 42-47 = shoulders, elbows, wrists)
    mid_shoulder = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2  # (T, 3)
    shoulder_width = np.linalg.norm(
        h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True
    )  # (T, 1)
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    # Hand centroids (mean of all hand landmarks)
    lh_centroid = h_raw[:, :21, :].mean(axis=1)   # (T, 3)
    rh_centroid = h_raw[:, 21:42, :].mean(axis=1)  # (T, 3)

    # Feature 1-2: Left/Right hand height relative to mid-shoulder (normalized by shoulder width)
    features[:, 0] = (lh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]
    features[:, 1] = (rh_centroid[:, 1] - mid_shoulder[:, 1]) / shoulder_width[:, 0]

    # Feature 3-4: Left/Right hand lateral position relative to mid-shoulder
    features[:, 2] = (lh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
    features[:, 3] = (rh_centroid[:, 0] - mid_shoulder[:, 0]) / shoulder_width[:, 0]

    # Feature 5: Inter-hand distance (normalized by shoulder width)
    features[:, 4] = np.linalg.norm(
        lh_centroid - rh_centroid, axis=-1
    ) / shoulder_width[:, 0]

    # Feature 6-7: Hand height relative to face (approximate face as above shoulders)
    # Y-axis is typically inverted in image coords, so sign depends on convention
    face_approx = mid_shoulder.copy()
    face_approx[:, 1] -= shoulder_width[:, 0] * 0.7  # approximate face position
    features[:, 5] = np.linalg.norm(
        lh_centroid - face_approx, axis=-1
    ) / shoulder_width[:, 0]
    features[:, 6] = np.linalg.norm(
        rh_centroid - face_approx, axis=-1
    ) / shoulder_width[:, 0]

    # Feature 7: Symmetry score (are hands at same height?)
    features[:, 7] = np.abs(lh_centroid[:, 1] - rh_centroid[:, 1]) / shoulder_width[:, 0]

    return features
```

**Integration**: Append these 8 features to the existing aux_input (angles + distances). Current aux_dim = NUM_ANGLE_FEATURES + 20 fingertip distances = ~53 features. Adding 8 hand-body features makes it ~61. The aux MLP (Linear(aux_dim, 128)) absorbs this trivially.

**Why This Might Actually Help**:
- It adds genuinely NEW information that v22 currently discards
- It is signer-normalized (divided by shoulder width), so it should generalize
- Location is a fundamental phonological parameter in all sign languages
- Unlike other interventions, this addresses the feature gap, not the learning gap

**Risk**: MediaPipe body pose (6 nodes) is lower quality than hand landmarks. If shoulder detection is noisy across signers, this could add noise rather than signal. Must validate on LOSO sanity check.

**Expected Impact**: +0-3pp. Could be higher if location confusion is a major error mode.
**Effort**: 1 day (feature computation + aux_dim adjustment in both train and eval scripts)

---

### S2. MediaPipe Version Alignment -- FREE WIN

**Source**: Implementation Critic.

**The Problem**: Training data was extracted with MediaPipe 0.10.14. The evaluation script uses whatever is installed (currently 0.10.5 based on conda env). Different MediaPipe versions produce different landmark coordinates for the same video frames.

**Evidence**: The v22 training script already has a `check_mediapipe_version()` function that warns about this. The evaluation script does NOT check.

**Fix**:
```bash
# In conda env ksl:
pip install mediapipe==0.10.14
```

**Why This Is Important**: Every other intervention tries to bridge the distribution gap between training and test. This one REMOVES an artificial source of distribution gap that we are creating ourselves. It is the only intervention that is unambiguously correct with zero risk.

**Expected Impact**: +0-2pp (unknown — depends on how much landmarks differ between 0.10.5 and 0.10.14). Could be 0pp if the versions are similar, or could be meaningful if they changed hand landmark detection.
**Effort**: 5 minutes.

---

### S3. Temporal CutMix -- GENUINELY UNTRIED

**Source**: Original plan, survived all 4 critics.

**Why It Survived**:
- Mixup is in v22 (alpha=0.2), but CutMix is NOT. They are different augmentations.
- CutMix splices temporal segments between signers, creating chimeric sequences. This is structurally different from linear interpolation (Mixup).
- Domain Expert: temporal splicing is linguistically valid for single signs (signs are short gestures, not sentences).
- Impl Critic: Must implement at batch level, recompute velocity after splice.

**Implementation**:
```python
# In the training loop, after collating batch (NOT in __getitem__)
def temporal_cutmix(gcn_input, aux_input, labels, signer_labels, alpha=1.0, p=0.3):
    """
    CutMix at the temporal dimension for skeleton sequences.
    gcn_input: (B, C, T, N)
    aux_input: (B, T, D_aux)
    """
    if np.random.random() > p:
        return gcn_input, aux_input, labels, labels, 1.0

    B, C, T, N = gcn_input.shape
    lam = np.random.beta(alpha, alpha)
    cut_len = int(T * (1 - lam))
    cut_start = np.random.randint(0, T - cut_len + 1)

    # Shuffle indices (cross-signer preferred)
    indices = torch.randperm(B)

    gcn_mixed = gcn_input.clone()
    aux_mixed = aux_input.clone()

    gcn_mixed[:, :, cut_start:cut_start+cut_len, :] = gcn_input[indices, :, cut_start:cut_start+cut_len, :]
    aux_mixed[:, cut_start:cut_start+cut_len, :] = aux_input[indices, cut_start:cut_start+cut_len, :]

    # Recompute velocity channels (channels 3:6 in gcn_input) at splice boundaries
    # to avoid discontinuity artifacts
    if cut_start > 0:
        gcn_mixed[:, 3:6, cut_start, :] = gcn_mixed[:, 0:3, cut_start, :] - gcn_mixed[:, 0:3, cut_start-1, :]
    if cut_start + cut_len < T:
        gcn_mixed[:, 3:6, cut_start+cut_len, :] = gcn_mixed[:, 0:3, cut_start+cut_len, :] - gcn_mixed[:, 0:3, cut_start+cut_len-1, :]

    labels_b = labels[indices]

    return gcn_mixed, aux_mixed, labels, labels_b, lam

# Loss computation:
# loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)
```

**Expected Impact**: +0-1pp
**Effort**: 0.5 days

---

## Signer 2/3 Duplication -- CRITICAL DATA FIX

**Source**: Implementation Critic + Statistician (both independently confirmed).

**The Problem**: Signers 2 and 3 are byte-identical duplicates (MD5 verified). V22 already deduplicates them, but this has implications:
- After dedup, Numbers has 250 training samples from 4 unique signers (not 5).
- LOSO for numbers is 4-fold, not 5-fold.
- Any fold holding out signer 2 or 3 has the other's identical data still in training = DATA LEAKAGE.

**Fix**: Verify that v22's `deduplicate_signer_groups()` correctly removes signer 3 (or 2). If LOSO is implemented, ensure the deduplication runs BEFORE signer assignment.

**Impact on V25**: This is not a model change, but it affects LOSO design. LOSO must use post-dedup data only, and must be understood as 4-fold for numbers.

---

## Implementation Plan

### Phase 0: Quick Wins (Day 1, <2 hours)

**Step 0a: Fix MediaPipe version**
```bash
pip install mediapipe==0.10.14
```
Then re-run v22 evaluation with version-matched MediaPipe. Compare against existing v22 results.

**Decision Gate 0**: If MediaPipe fix alone gives +2pp or more, it confirms that version mismatch was a meaningful source of error. Proceed with remaining interventions. If +0pp, mismatch was not the issue.

### Phase 1: Hand-to-Body Features (Day 1-2)

**Step 1a**: Implement `compute_hand_body_features()` in training script.

**Step 1b**: Compute features BEFORE wrist-centric normalization (this is critical -- the raw landmark positions are needed).

**Step 1c**: Append to existing aux_input. Adjust `aux_dim` parameter.

**Step 1d**: Mirror EXACTLY in evaluation script (shared preprocessing function imported by both scripts).

**Step 1e**: Train v25-alpha (v22 + hand-body features only). Compare val accuracy.

**Step 1f**: Run LOSO sanity check. If any fold drops below 25% combined, the features are hurting. Abort.

**Step 1g**: Run real-tester evaluation.

**Decision Gate 1**:
- If real-tester accuracy improves by any amount -> keep features, proceed to Phase 2.
- If real-tester accuracy drops -> remove features, skip to Phase 2 with v22 base.
- Note: we cannot statistically distinguish <16pp differences, so "improvement" means the point estimate is higher.

### Phase 2: CutMix (Day 2-3)

**Step 2a**: Implement temporal CutMix at the batch level in training loop.

**Step 2b**: Ensure velocity channels are recomputed at splice boundaries.

**Step 2c**: Train v25-beta (best of Phase 1 + CutMix). Compare val accuracy.

**Step 2d**: Run real-tester evaluation.

**Decision Gate 2**:
- If combined accuracy > 40% -> this is our v25 release.
- If combined accuracy < v22 (39.8%) -> revert to v22 as best model.

### Phase 3: Diagnostic-Only (Day 3, optional)

Run D2 (feature t-SNE) and D3 (BN stats comparison) from the original plan. These are purely diagnostic -- they generate understanding, not model improvements. They are valuable for the thesis/paper to explain WHY the ceiling exists.

**Step 3a**: Extract 320-dim embeddings from v25 for all training + real-tester samples. Plot t-SNE colored by signer and by class.

**Step 3b**: Compare BN running stats against real-tester activation stats. Quantify per-layer distributional shift.

**These diagnostics inform the "what next" decision (more data vs more architecture).**

---

## What We Expect and What It Means

### Realistic Outcome Table

| Scenario | Real Combined Acc | Probability | What It Means |
|----------|------------------|-------------|---------------|
| Regression | <38% | 20% | Hand-body features or CutMix introduced noise |
| No change | 38-42% | 50% | Ceiling is real, +/-2pp is noise |
| Small gain | 42-45% | 25% | Hand-body features captured useful new signal |
| Large gain | >45% | 5% | Would be unprecedented; verify carefully |

### Statistical Reality Check (from Statistician Critic)
- n=140 real test samples. MDE = 16.4pp at power=0.80.
- ALL v19-v24 results (34-40%) are within noise of each other.
- Even a "gain" to 45% would NOT be statistically significant vs v22's 39.8%.
- We should report results as "comparable" unless the difference exceeds ~16pp.
- The only way to get statistical significance is to collect more test data.

### Per-Intervention Expected Gains (Honest)

| Intervention | Expected Gain | Confidence | Basis |
|-------------|--------------|------------|-------|
| MediaPipe version fix | +0-2pp | Medium | Unknown magnitude of landmark differences |
| Hand-to-body features | +0-3pp | Low-Medium | New signal, but noisy body landmarks |
| Temporal CutMix | +0-1pp | Low | Marginal augmentation addition to 13 existing |
| **Combined (optimistic)** | **+0-5pp** | Low | Gains unlikely to stack fully |
| **Combined (realistic)** | **+0-2pp** | Medium | Most likely outcome |

---

## Decision Tree

```
START
  |
  v
Fix MediaPipe version (5 min)
  |
  v
Re-evaluate v22 with correct MediaPipe
  |
  +-- If +2pp or more --> Great, this was low-hanging fruit
  +-- If <2pp ---------> Expected, proceed anyway
  |
  v
Implement hand-to-body features
Train v25-alpha
  |
  v
LOSO sanity check
  |
  +-- Any fold <25% --> Features hurt. Remove. Use v22 base for CutMix.
  +-- All folds >25% --> Features OK. Keep.
  |
  v
Real-tester evaluation
  |
  +-- Accuracy > v22 --> Keep features
  +-- Accuracy < v22 --> Remove features, note in report
  |
  v
Add CutMix
Train v25-beta
  |
  v
Real-tester evaluation
  |
  +-- Accuracy > v22 -------> V25 is our new best model
  +-- Accuracy <= v22 ------> V22 remains best. Write conclusion:
  |                            "4-signer ceiling confirmed. More data needed."
  |
  v
Run diagnostics (t-SNE, BN stats) regardless of outcome
  |
  v
Write thesis section on generalization ceiling
  |
  v
DONE (or: collect more signers)
```

---

## Items NOT In This Plan (Deliberately)

| Item | Why Excluded |
|------|-------------|
| V-REx | GRL already achieves signer invariance (silhouette=0.006) |
| Stronger regularization | Wrong diagnosis (domain shift, not overfitting) |
| Y-axis rotation | Invalid with 2.5D MediaPipe coordinates + breaks sign meaning |
| Butterworth smoothing | Jitter is not the bottleneck |
| Canonical skeleton alignment | Redundant with existing normalization |
| Tent TTA | High effort (3 days), marginal benefit, fragile |
| Per-signer experts | 150 samples/model = severe overfit |
| Per-class accuracy fixes | 3-4 samples per class = noise |
| LOSO for hyperparameter selection | Resolution too coarse (~10pp) |
| Prototypical networks | v24 proved: same features, same limits |
| Multi-stream ensemble | v23 proved: fusion of correlated streams doesn't help |
| MixStyle | v24 proved: no benefit for domain generalization here |
| Post-hoc calibration | v23 proved: doesn't transfer to real testers |

---

## The Real Path to 85%

All 4 critics converge on the same conclusion:

| Signers | Expected Real Accuracy | Source |
|---------|----------------------|--------|
| 4-5 (current) | 35-45% | Measured (v19-v24) |
| 7-8 | 45-55% | Literature extrapolation |
| 10-12 | 55-65% | Literature extrapolation |
| 15-20 | 70-80% | Literature (MSASL, WLASL) |
| 20+ | 80-85%+ | Literature (mature systems) |

**V25's purpose is not to reach 85%.** It is to:
1. Squeeze out any remaining gains from the current 4-signer data
2. Build the diagnostic infrastructure to understand WHY the ceiling exists
3. Make the empirical case for data collection as the necessary next step
4. Provide a clean, well-documented codebase for future work with more data

---

## Implementation Checklist

- [ ] Fix MediaPipe to 0.10.14 in conda env
- [ ] Re-evaluate v22 with version-matched MediaPipe
- [ ] Implement `compute_hand_body_features()` (compute before normalization)
- [ ] Update aux_dim in model instantiation
- [ ] Create shared preprocessing module for train/eval parity
- [ ] Train v25-alpha (v22 + hand-body features)
- [ ] LOSO sanity check (4-fold, post-dedup)
- [ ] Real-tester evaluation of v25-alpha
- [ ] Implement temporal CutMix at batch level
- [ ] Train v25-beta (best of alpha + CutMix)
- [ ] Real-tester evaluation of v25-beta
- [ ] Run t-SNE diagnostic on best model
- [ ] Run BN stats comparison diagnostic
- [ ] Write results report with bootstrap CIs
- [ ] Update thesis with generalization analysis

---

## Acknowledgments

This plan was shaped by rigorous critique from four independent reviewers:
- **ML Methodology Critic**: Killed V-REx, rotation augmentation, and stronger regularization with evidence from v22's existing signer invariance
- **Statistician Critic**: Established that n=140 cannot detect <16pp improvements, and that all v19-v24 results are statistically indistinguishable
- **Domain Expert Critic**: Identified the missing hand-to-body features (genuinely new signal) and confirmed that 40% accuracy is expected for 5 signers in the SL literature
- **Implementation Critic**: Confirmed signer 2/3 duplication, identified MediaPipe version mismatch, and verified v22 codebase correctness

The consensus across all four: **architecture improvements have hit diminishing returns. The only path to >50% is more signer data.**
