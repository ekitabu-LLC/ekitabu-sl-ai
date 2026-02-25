# KSL Next Experiments: Critically Evaluated & Ranked

**Date:** 2026-02-15
**Current best:** Weighted Ensemble 74.3% (5 models), Single best: exp5 SupCon 63.2%
**Target:** 85%
**Gap:** 10.7pp (ensemble), 21.8pp (single model)

---

## Critical Context: Why Most Ideas Will Fail

Our core bottleneck is **signer diversity** (12 training signers), not model capacity.
We have exhaustively tried model-side improvements: KD, TTA, multi-seed ensembles,
EMA+RSC, synthetic augmentation, SWAD, MixStyle, hard mining, prototypical networks,
BlockGCN, and 8 different regularizers. All failed or regressed.

The only techniques that worked:
1. **More real signers** (+1.6pp/signer historically)
2. **GroupNorm** (+3.2pp by eliminating BN domain gap)
3. **SupCon loss** (+5.0pp by learning signer-invariant embeddings)
4. **Weighted ensemble** (+3.6pp over uniform by optimizing per-category weights)

Any new experiment must be **fundamentally different** from what failed.

---

## Ranked Experiments (by expected value = probability of success x gain)

### Experiment 1: Transfer Learning from WLASL/AUTSL Pretrained Backbone
**Priority: HIGH | Expected gain: +3-6pp | Risk: MEDIUM**

**What:** Pretrain our ST-GCN backbone (or a CTR-GCN replacement) on WLASL-100 or AUTSL
(both skeleton-based, ~2000-5000 samples), then fine-tune on KSL with a low learning
rate. This gives the model exposure to ~100+ signers before ever seeing KSL data.

**Why this is different from what failed:**
- Directly addresses the signer diversity bottleneck (100+ signers in pretraining)
- Not a regularization trick or augmentation hack — genuinely new data/knowledge
- Published evidence: +1-7pp gains from cross-dataset transfer in SLR (Chen et al. CVPR 2022;
  cross-lingual few-shot SLR, Pattern Recognition 2024)
- Works on small target datasets (WLASL itself is small, ~5000 samples)

**Why it might fail:**
- Different sign languages have different phonology — KSL signs may not share structure
  with ASL/Turkish SL
- Skeleton topology must match (MediaPipe 48 nodes vs. OpenPose 25+hands)
  — may need topology adapter or re-extraction
- Our custom ST-GCN architecture may not align with standard pretrained checkpoints

**Implementation plan:**
1. Download WLASL-100 skeleton data (or re-extract with MediaPipe to match our 48-node format)
2. Pretrain our GroupNorm ST-GCN architecture on WLASL-100 for 200 epochs
3. Fine-tune on KSL with lr=1e-4 (10x lower than from-scratch), freeze early layers for 50 epochs
4. Evaluate with and without SupCon loss during fine-tuning

**Compute:** ~4hr pretrain + 3hr fine-tune = fits in 8hr job
**Verdict:** Our single best bet. Even partial transfer should help because pretraining
exposes the model to signer variation patterns absent from our 12 signers.

---

### Experiment 2: Label Smoothing + Self-Distillation (Born-Again Networks)
**Priority: HIGH | Expected gain: +1-3pp per model | Risk: LOW**

**What:** Train the same GroupNorm architecture with label smoothing (alpha=0.1),
then use that model's soft predictions as targets for a second generation
("born-again" self-distillation). Repeat for 2-3 generations.

**Why this is different from what failed:**
- Our KD failed because the *teacher* was overconfident (100% train acc, one-hot outputs)
- Self-distillation avoids this: generation-1 trains with label smoothing so its outputs
  are inherently NOT one-hot — genuine soft knowledge exists
- Born-Again Networks consistently beat teachers by +0.5-1.5pp per generation
  (Furlanello et al. ICML 2018; NeurIPS 2020 analysis)
- Label smoothing alone is a simple regularizer we have NOT tried
- Zero architecture change, zero extra data, minimal compute overhead

**Why it might fail:**
- Gain per generation is typically small (+0.5-1.5pp)
- On 895 samples, even label smoothing might not be enough to prevent memorization
- If generation-1 still reaches near-100% train accuracy, soft labels collapse again

**Implementation plan:**
1. Train exp1 (GroupNorm) with label smoothing alpha=0.1 — call this Gen-0
2. Generate soft labels from Gen-0 (should NOT be one-hot due to smoothing)
3. Train Gen-1 with KD loss from Gen-0 soft labels (T=4, alpha=0.5) + label smoothing
4. Optionally train Gen-2 from Gen-1
5. Add all generations to ensemble weight optimization

**Compute:** ~2hr per generation x 3 = 6hr total
**Verdict:** Low risk, low cost. Even if individual gain is small, adding 2-3 diverse
generations to the ensemble could yield +1-2pp on the weighted ensemble.

---

### Experiment 3: Signer-Adversarial Consistency + Gradient Reversal Enhancement
**Priority: MEDIUM | Expected gain: +1-3pp | Risk: MEDIUM**

**What:** Strengthen the existing signer-adversarial training with:
(a) Consistency regularization: enforce that augmented versions of the same sample
    produce similar embeddings (not just same classification)
(b) Gradient penalty on the signer discriminator for smoother adversarial training
(c) Separate the numbers/words response by using category-specific adversarial weights

**Why this is different from what failed:**
- We already HAVE a signer-adversarial head (GRL) but it's lightly weighted
- Recent NAACL 2024 work (Signer Diversity-driven Data Augmentation) showed that
  consistency constraints + adversarial training gives SOTA signer-independent results
- This directly targets signer invariance (our actual bottleneck)
- Category-specific weighting addresses the numbers/words divergence problem

**Why it might fail:**
- GRL training is already present — increasing its weight may hurt convergence
- Adversarial training on 12 signers may not generalize to unseen signers
- Consistency regularization is another regularizer, and we've seen over-regularization fail

**Implementation plan:**
1. Increase GRL lambda from current value, with warmup schedule (0 for 50 epochs, ramp to 0.1)
2. Add embedding consistency loss: MSE(embed(x), embed(aug(x))) with weight 0.01
3. Train numbers model with GRL_lambda=0.1, words model with GRL_lambda=0.05
4. Compare against exp1 baseline

**Compute:** ~3hr per model type = 6hr total
**Verdict:** Moderate confidence. This is a refinement of existing machinery, not
a new paradigm. But the NAACL 2024 evidence for consistency + adversarial training
is strong and directly relevant.

---

### Experiment 4: TENT/Entropy Minimization at Test Time (for Ensemble)
**Priority: MEDIUM-LOW | Expected gain: +1-2pp on ensemble | Risk: LOW**

**What:** At inference, before classifying a test signer's samples, adapt the ensemble's
predictions using entropy minimization (TENT-style). For each test sample batch:
update only the affine parameters of GroupNorm layers to minimize prediction entropy.

**Why this is different from what failed:**
- TTA (test-time augmentation) failed because it added noise to well-calibrated predictions
- TENT is fundamentally different: it adapts the MODEL to the test distribution,
  not the INPUT
- GroupNorm has learnable affine params (gamma, beta) that can be adapted
- Our AdaBN gave +6-8pp on BN models — TENT is the GroupNorm analog
- CVPR 2024 showed improved self-training for TTA with pseudo-label correction

**Why it might fail:**
- GroupNorm stats are not batch-dependent like BN — less room for adaptation
- With only ~60 test samples per signer, adaptation may overfit
- TENT requires multiple forward passes per test batch — slower inference
- GroupNorm was chosen precisely to avoid domain-gap issues

**Implementation plan:**
1. Take the weighted ensemble's 5 models
2. For each test signer: clone models, set GroupNorm affine params as trainable
3. Run 5-10 steps of entropy minimization on test signer's data (lr=1e-4)
4. Evaluate adapted models on that test signer
5. If helpful, apply to ensemble predictions

**Compute:** ~30min evaluation-only
**Verdict:** Quick to try, minimal downside. But GroupNorm's design philosophy
(domain-invariant) may mean there's nothing left to adapt. Worth 30 minutes.

---

### Experiment 5: Cross-Dataset Pretraining with Contrastive Learning
**Priority: MEDIUM | Expected gain: +2-5pp | Risk: HIGH**

**What:** Pretrain the backbone using self-supervised contrastive learning (SimCLR/MoCo
style) on a large unlabeled skeleton dataset (NTU-RGB+D has 56K skeleton sequences
with 40 subjects). Then fine-tune on KSL.

**Why this is different from what failed:**
- Our SupCon (exp5) already showed contrastive learning works (+5pp)
- Self-supervised pretraining on 56K sequences gives the model a strong prior
  for skeleton motion understanding before seeing any sign language
- 40 subjects in NTU provides massive signer/subject diversity
- Graph Contrastive Learning for Skeleton-based Action Recognition (SkeletonGCL, 2023)
  showed consistent gains when combined with CTR-GCN/InfoGCN

**Why it might fail:**
- NTU-RGB+D is body actions, not hand-focused sign language — domain mismatch
- Skeleton topology mismatch (NTU uses 25 body joints, we use 48 hand+pose joints)
- Self-supervised pretraining requires significant engineering effort
- May need topology-agnostic approach (expensive to implement)

**Implementation plan:**
1. Use NTU-RGB+D hand subset or re-extract hand keypoints if available
2. Implement SimCLR with spatial/temporal augmentations on skeleton sequences
3. Pretrain backbone for 100 epochs
4. Fine-tune on KSL with SupCon + CE loss

**Compute:** ~6-8hr (tight fit)
**Verdict:** High potential but high implementation risk due to topology mismatch.
Deprioritize unless Experiment 1 succeeds and we want to push further.

---

## Rejected Proposals (with reasons)

| Proposal | Reason for Rejection |
|---|---|
| **Diffusion-based skeleton generation (SALAD, MDM)** | These models are trained on body motion (walking, dancing), not hand signs. Requires massive training data to learn the generation distribution. Our 895 samples are far too few to train a diffusion model, and pretrained ones won't generate sign language. |
| **Few-shot prototypical networks** | Already tried in v31 exp2 — words collapsed to 50.6%. Episodic training is unstable with 15 classes and 12 signers. |
| **Mixture of Experts (MoE)** | Adds model complexity, not signer diversity. With 895 samples, more parameters = more overfitting. We need more data, not more capacity. |
| **Neural Architecture Search (NAS)** | Computationally prohibitive on HPC with job time limits. Architecture is not the bottleneck — our ST-GCN is sufficient (100% train accuracy proves this). |
| **More synthetic signer augmentation** | v36 already failed (-5pp). Fixed transforms on normalized coords are redundant with existing augmentation. |
| **Additional regularizers** | v30 Phase 2 tested 8 regularizers — all failed or regressed. Over-regularization on 895 samples is a pattern. |
| **Higher-temperature KD** | Tried T=4 and T=20 in v33/v33b — both failed. Teacher's 100% train accuracy means no dark knowledge at any temperature. |
| **AnyTop skeleton diffusion** | While AnyTop works with 3 examples per topology, it generates body animations, not sign language hand movements. Domain mismatch is severe. |

---

## Realistic Path to 85%

Being brutally honest: **model-side improvements alone cannot close the 10.7pp gap.**

**Best case with all 5 experiments succeeding:**
- Exp 1 (Transfer): +4pp → 78.3% ensemble
- Exp 2 (Self-distill): +1.5pp → 79.8% ensemble
- Exp 3 (Adv consistency): +1.5pp → 81.3% ensemble
- Exp 4 (TENT): +1pp → 82.3% ensemble
- **Total optimistic: ~82%** (still 3pp short)

**To reach 85%, we need BOTH:**
1. The experiments above (~+8pp optimistic)
2. **At least 3-5 more real signers** (~+5-8pp at 1.6pp/signer)

**Recommended execution order:**
1. Exp 2 (self-distill) — fastest, lowest risk, run first as sanity check
2. Exp 4 (TENT) — 30 min evaluation, test while Exp 1 runs
3. Exp 1 (transfer learning) — highest expected value, but needs WLASL data prep
4. Exp 3 (adversarial consistency) — run after Exp 1 results inform approach
5. Exp 5 (contrastive pretrain) — only if Exp 1 shows transfer learning works

**Parallel track:** Recruit more signers. This has been our most reliable improvement
throughout the entire project history.
