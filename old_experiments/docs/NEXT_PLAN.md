# KSL Next Experiments Plan
*Research completed: 2026-02-20 | Current best: 72.9% uniform ensemble (65.7% best single model)*

---

## Summary of Research Findings

Five parallel web-search agents covered: self-supervised learning, cross-lingual transfer,
advanced architectures, synthetic augmentation, and domain generalization.

Key diagnosis across all reports:
- **Root cause of plateau**: 12 training signers → insufficient signer diversity
- **Not fixable by regularization** (IRM/SWAD/dropout all failed — confirmed)
- **Three complementary attack vectors**:
  1. Remove signer identity from the **input** (normalization)
  2. Remove signer identity from the **features** (adversarial training)
  3. Expose the encoder to **more signers** via pretraining on external datasets

---

## Naming Convention
All experiments in this round: `v44_expr_N` (e.g. v44_expr_1, v44_expr_2, ...)

---

## TIER 1 — Quick Wins (1-2 days each, low risk, try immediately)

### v44_expr_1: Anchor-Based Skeleton Normalization
**Source:** DG report — Roh et al., LREC-COLING 2024; SAM-SLR CVPR 2021
**Expected gain:** +2-5pp | **Risk:** Near-zero (preprocessing only)

Normalize each frame to remove signer body-proportion variation:
- **Hands**: center on wrist, scale by palm width (wrist → middle-finger-base distance)
- **Body**: center on torso midpoint, scale by shoulder width

This is a preprocessing change only — no model retraining needed. Apply to all 6 ensemble
models by reprocessing the test data. The +6pp reported on WLASL is the benchmark; our
gain may be smaller since we already center, but signer hand-size normalization is missing.

**Action**: Modify `preprocess_for_onnx.py` and training pipeline to add anchor normalization.
Re-evaluate all existing checkpoints on real_testers.

---

### v44_expr_2: Signer-Adversarial GRL
**Source:** DG report — Zuo & Mak, ACM TOMM 2024 (arxiv 2212.13023)
**Expected gain:** +2-4pp | **Risk:** Low (new model, ~20 lines added)

Add a gradient reversal layer (GRL) + signer classification head after the GCN backbone:
- Signer head: 2-layer MLP → 12-class signer ID (training signers)
- GRL reverses gradient → backbone learns SIGN-discriminative, SIGNER-invariant features
- Lambda schedule: ramp from 0.01 → 0.1 over training

Start with exp1 (GroupNorm) architecture to avoid conflict with SupCon.
If successful, also try with exp5 (SupCon) architecture.

GRL is ~10 lines of PyTorch autograd.Function. Full implementation: 1-2 days.

---

### v44_expr_3: Skeleton-CutMix Augmentation
**Source:** Synth report — Liu et al., IEEE TIP 2023 (https://github.com/HanchaoLiu/Skeleton-CutMix)
**Expected gain:** +2-4pp | **Risk:** Low (augmentation only, no model changes)

Exchange hand sub-skeletons between training signers probabilistically:
- Group 1: Left hand (joints 0–20)
- Group 2: Right hand (joints 21–41)
- Group 3: Pose/body (joints 42–47)

Take signer A's left hand + signer B's right hand → novel composite identity.
Unlike V36 (geometric warping = redundant with existing augmentation), bone exchange
creates samples that are genuinely unattributable to any single training signer.

Code available. Adapt joint groupings for our 48-joint MediaPipe skeleton. ~200 lines.

---

### v44_expr_4: EDTN — Better Test-Time Adaptation Beyond AdaBN
**Source:** DG report — Wei et al., IMWUT/UbiComp 2024 (https://github.com/Claydon-Wang/OFTTA)
**Expected gain:** +1-3pp on BatchNorm models | **Risk:** Low (no retraining)

Instead of replacing ALL BN stats with test-time stats (our current AdaBN), blend per layer depth:

```
stats_layer_d = alpha^d * test_stats + (1 - alpha^d) * source_stats
```

Shallow layers get more test-time stats (capturing low-level domain shift);
deep layers keep more source stats (preserving semantic features).
Reported +2-3pp over standard AdaBN on cross-person HAR benchmarks.

Tune alpha on test_alpha, apply to real_testers.
Only affects BatchNorm models: exp5, v43, openhands, v27/v28/v29.
GroupNorm models (exp1, v41) unaffected.

---

## TIER 2 — Medium Effort (3-7 days, high expected gain)

### v44_expr_5: HA-GCN Hand-Body Sub-Graphs
**Source:** Arch report — HA-GCN 2024 (https://github.com/snorlaxse/HA-SLR-GCN)
**Expected gain:** +2-5pp | **Risk:** Medium (architecture change)

Our current ST-GCN treats all 48 joints equally. But 42 of 48 joints are hand joints — they
need finer-grained processing than the 6 body joints. HA-GCN defines two sub-graphs:
- **Local hand sub-graph**: finger-level topology within each hand (21 joints × 2)
- **Global body sub-graph**: coarse pose topology (6 joints)

With separate GCN layers for each sub-graph + a fusion module.
This is a surgical modification to our existing architecture, not a full rewrite.
Start with exp1 (GroupNorm) architecture as the base.

---

### v44_expr_6: WLASL Pretrain → KSL Finetune
**Source:** Transfer report — Logos EMNLP 2025; WLASL dataset
**Expected gain:** +3-8pp | **Risk:** Medium (new training pipeline)

Pretrain our existing GCN backbone (from exp5/v28) on WLASL sign language data:
1. Download WLASL (21K videos, 2000 ASL classes) — skeleton data available
2. Re-extract skeletons with MediaPipe Holistic to match our 48-joint topology exactly
3. Pretrain for ~200 epochs on WLASL classification
4. Replace WLASL head with KSL 15-class head, finetune on our 1,787 samples

**Why this helps**: The encoder sees 100s of ASL signers' hand/body motion patterns
before ever touching KSL. This is the closest thing to "recruiting more signers" without
actually doing so.

Even if pretrained single model only matches current best (~65%), it will have different
error patterns → valuable ensemble member.

**Estimated time**: ~4-8h pretraining on A100 + ~1h finetuning. 1-2 days implementation.

---

### v44_expr_7: JMDA SpatialMix + TemporalMix
**Source:** Synth report — Xiang et al., ACM TOMM 2024 (https://github.com/aidarikako/JMDA)
**Expected gain:** +1-3pp | **Risk:** Low (augmentation only)

Complementary to Skeleton-CutMix (V46):
- **SpatialMix**: mix spatial joint positions between two samples (cross-signer)
- **TemporalMix**: mix temporal segments between samples (cross-temporal)

TemporalMix is particularly useful for words where trajectory timing matters.
Official code available. Adapt joint groupings. Run in parallel with V46 on HPC.

---

### v44_expr_8: SkeletonMAE Pretraining on WLASL + AUTSL
**Source:** SSL report — Yan et al., ICCV 2023 (https://github.com/HongYan1123/SkeletonMAE)
**Expected gain:** +3-8pp | **Risk:** Medium

Mask ~75% of skeleton joints+frames, train encoder to reconstruct them using graph topology
priors. No labels needed — pretrain on any skeleton data:
- WLASL: 21K sign language sequences
- AUTSL: 38K Turkish SL sequences (43 diverse signers)
- Combined: ~59K sequences (33× our training data)

After pretraining on these 59K unlabeled sequences, fine-tune on our 1,787 KSL samples.
Our GCN encoder adapts naturally as the masked encoder. Need to adapt adjacency graph
to our 48-joint topology (vs their 25-joint body skeleton).

Follow up with STARS Stage 2 (ECCV 2024): nearest-neighbor contrastive tuning for a few
epochs to improve cluster separation for unseen signers.

---

## TIER 3 — High Effort, Highest Ceiling (1-2 weeks)

### v44_expr_9: Multi-Dataset Co-Training (Logos approach)
**Source:** Transfer report — Ovodov et al., EMNLP 2025 (SOTA WLASL 66.82%)
**Expected gain:** +5-10pp | **Risk:** Medium-High

Shared GCN backbone + separate classification heads for WLASL, AUTSL, and KSL.
Train jointly by sampling batches from all 3 datasets. At inference, use only KSL head.

Requirements:
1. Re-extract WLASL + AUTSL skeletons with our MediaPipe pipeline
2. Implement multi-head training loop with dataset-specific heads
3. Handle class imbalance across datasets (2000 vs 226 vs 30 classes)

**Estimated time**: 3-5 days implementation + 1-2 days training. Highest ceiling.

---

### v44_expr_10: SkeleMoCLR — Semi-Supervised with Momentum Contrastive Teacher
**Source:** SSL report — IEEE TIP 2024
**Expected gain:** +3-7pp | **Risk:** Medium

Extends our existing SupCon (exp5) with:
- Momentum encoder (teacher) generating pseudo-labels for WLASL/AUTSL unlabeled data
- Contrastive memory queue for hard negative mining
- Joint optimization: labeled KSL CE + pseudo-labeled CE + contrastive loss

Builds directly on our SupCon experience. Unlabeled sign language data (WLASL/AUTSL)
acts as free extra training signal.

---

### v44_expr_11: Siformer Hand Pose Rectification
**Source:** Arch report — ACM MM 2024 (https://github.com/mpuu00001/Siformer)
**Expected gain:** +3-8pp | **Risk:** Medium-High

Applies kinematic constraints to fix noisy MediaPipe hand keypoints before GCN input.
Adds a feature-isolated attention mechanism for spatial-temporal context.

Key insight: noisy hand keypoints from MediaPipe are a consistent source of error.
Fixing them via kinematic priors (finger angles, palm constraints) benefits all downstream models.

Can be implemented as a preprocessing module (kinematic rectification only, skip Transformer)
or as a full architecture replacement.

---

## Priority Order for HPC Experiments

```
Week 1 (run in parallel):
  - v44_expr_1: Skeleton normalization (1 day, eval existing checkpoints)
  - v44_expr_4: EDTN test-time adaptation (1 day, eval existing checkpoints)
  - v44_expr_3: Skeleton-CutMix training (new model train)
  - v44_expr_2: Signer-adversarial GRL training (new model train)

Week 2 (run in parallel):
  - v44_expr_6: WLASL pretrain → KSL finetune
  - v44_expr_5: HA-GCN hand-body sub-graphs
  - v44_expr_7: JMDA augmentation

Week 3+:
  - v44_expr_8: SkeletonMAE pretraining on WLASL+AUTSL (largest potential)
  - v44_expr_9: Multi-dataset co-training (highest ceiling)
  - v44_expr_10: SkeleMoCLR semi-supervised
```

---

## Expected Cumulative Gains

| Experiment | Gain (individual) | Cumulative from 72.9% |
|---|---|---|
| v44_expr_1 skeleton normalization | +2-5pp | 74-78% |
| v44_expr_4 EDTN | +1-2pp | 75-80% |
| v44_expr_2+3 new models in ensemble | +2-4pp ensemble diversity | 77-82% |
| v44_expr_6 WLASL pretrain | +3-5pp new model | 78-84% |
| v44_expr_8+9 pretraining | +5-8pp | 83-90% |

*Note: gains are cumulative estimates, not guaranteed. Each improves ensemble diversity
even if individual accuracy doesn't increase dramatically.*

---

## Key Papers & Code

| Method | Paper | Code |
|---|---|---|
| SkeletonMAE | ICCV 2023 | github.com/HongYan1123/SkeletonMAE |
| STARS | ECCV 2024 | — |
| HA-GCN | ScienceDirect 2024 | github.com/snorlaxse/HA-SLR-GCN |
| Siformer | ACM MM 2024 | github.com/mpuu00001/Siformer |
| HD-GCN | ICCV 2023 | github.com/Jho-Yonsei/HD-GCN |
| Skeleton-CutMix | IEEE TIP 2023 | github.com/HanchaoLiu/Skeleton-CutMix |
| JMDA | ACM TOMM 2024 | github.com/aidarikako/JMDA |
| OFTTA/EDTN | UbiComp 2024 | github.com/Claydon-Wang/OFTTA |
| Signer-GRL | ACM TOMM 2024 | github.com/2000ZRL/LCSA_C2SLR_SRM |
| Logos co-training | EMNLP 2025 | arxiv.org/abs/2505.10481 |
| WLASL | — | github.com/dxli94/WLASL |
| AUTSL | — | cvml.ankara.edu.tr/datasets/ |
