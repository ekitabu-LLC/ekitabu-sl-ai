# V45 Experiment Plan — KSL Sign Language Recognition
**Date:** 2026-02-22 | **Research basis:** 5-agent web search across 30+ papers (2024-2025)

## Current State
- Best honest result: 6-model uniform ensemble **72.9%** (numbers 74.6%, words 71.6%)
- Best single model: v43 **65.7%** combined (numbers 66.1%, words 65.4%)
- Root cause of plateau: 12 training signers insufficient; single-model ceiling ~66%
- V44 conclusion: all 6 experiments failed; GRL proved signer identity ≈ motion features for words

---

## TIER 1 — Quick Wins (1–2 days each, high confidence)

### V45_expr1: Sequence-Level Signer Adversarial GRL
**Motivation:** V44_expr2 GRL failed because it applied gradient reversal at frame-level spatial features, destroying motion encoding. ACM TOMM 2024 paper shows sentence/sequence-level GRL (after temporal pooling) achieves >50% relative WER reduction without breaking motion features.

**Method:**
- Apply GRL *after* temporal pooling (mean+std over time dimension → statistics vector)
- Signer classifier: Linear(640→64)→ReLU→Linear(64→12) on pooled features
- Loss: `CE_sign + λ * CE_signer_reversed` — same as V44_expr2 but applied at sequence level
- Keep backbone unchanged; only reverse gradients on the temporal summary vector
- Stream strategy: apply to joint+bone streams only, leave velocity stream completely untouched

**Expected gain:** Numbers ~67% (keeps GRL numbers benefit), Words ~63–65% (recovers from 53.1% collapse)
**Base architecture:** exp5 (BatchNorm, SupCon w=0.05) or exp1 (GroupNorm)
**Paper:** Zuo & Mak, ACM TOMM 2024 — statistics pooling signer removal

---

### V45_expr2: Class-Aware Hard Negative SupCon
**Motivation:** Our SupCon (exp5) uses equal-weight random negatives. ACCV 2024 shows prioritizing hard negatives from confusable classes improves fine-grained discrimination significantly on NTU120 and FineGym (exactly our setting: many similar classes).

**Method:**
- Build confusion matrix on val set after each epoch
- Weight negatives in SupCon by inter-class confusion score: `w_ij = confusion(i,j) / sum(confusion)`
- More weight to negatives from classes that look similar (e.g., numbers that share hand shapes)
- Memory bank (size=1787) stores embeddings for cross-batch hard negative mining
- Loss: `SupCon_hardneg = SupCon * negative_weights` — drop-in replacement for existing SupCon head

**Expected gain:** +1–3pp combined (mainly helps confusable sign pairs)
**Base architecture:** exp5 (already has SupCon, just replace loss computation)
**Paper:** Bian et al., ACCV 2024 — class-aware contrastive with memory bank

---

### V45_expr3: Multi-Scale Temporal Convolution (MS-TCN)
**Motivation:** Every top-performing 2024–2025 skeleton paper (DSTSA-GCN, LMSTGCN, SML) uses dilated multi-scale temporal convolutions. Our current ST-GCN uses a single kernel=9. Adding parallel dilated branches captures both fast finger transitions and slower arm trajectories.

**Method:**
- Replace each temporal conv block with parallel branches:
  - Branch 1: kernel=3, dilation=1 (fast local motion)
  - Branch 2: kernel=3, dilation=2 (medium motion)
  - Branch 3: kernel=3, dilation=4 (slow global motion)
  - Branch 4: 1×1 conv (residual skip)
- Concatenate and project back to original channel dim
- Applies to all 10 ST-GCN temporal blocks
- Minimal parameter increase (~1.3×); no change to spatial GCN

**Expected gain:** +1–3pp (better temporal modeling of signs with different durations)
**Base architecture:** exp1 (GroupNorm) or exp5
**Paper:** DSTSA-GCN (Neurocomputing 2025), LMSTGCN (Big Data Mining 2025)

---

### V45_expr4: LLM Semantic Supervision (CrossGLG-style)
**Motivation:** CrossGLG (ECCV 2024), SCoPLe (CVPR 2025), and 3 other top-venue papers all converge on using LLM-generated text descriptions as auxiliary training supervision for skeleton models. Zero inference cost. Anchors features to signer-invariant semantic meaning.

**Method:**
- Generate descriptions for all 30 KSL signs using GPT-4:
  - Global: "The sign for Apple involves a rotating fist motion near the cheek"
  - Hand: "Closed fist, thumb-side knuckles pressed against cheekbone, wrist rotation"
  - Motion: "Continuous rotation from neutral position through 90 degrees and back"
  - Temporal: "Single fluid rotation cycle over ~1 second"
- Encode descriptions with CLIP text encoder (frozen ViT-L/14)
- Add auxiliary loss: cosine similarity between skeleton embedding and text embedding
  `L_sem = 1 - cosine(GCN_embed, CLIP_text_embed)` per sample
- Total loss: `CE + 0.05 * SupCon + 0.1 * L_sem`
- No text needed at inference

**Expected gain:** +2–5pp (semantic anchoring reduces within-class signer variation)
**Base architecture:** exp5
**Papers:** CrossGLG ECCV 2024, SCoPLe CVPR 2025, Language-Guided Temporal Neurocomputing 2024

---

## TIER 2 — Moderate Effort (3–5 days, medium confidence)

### V45_expr5: MASA-style Masked Autoencoder Pre-training
**Motivation:** MASA (IEEE TCSVT 2024) shows masked skeleton reconstruction pre-training gives +25pp on WLASL100. MuteMotion Kaggle dataset has pre-extracted MediaPipe skeletons for WLASL (~13K samples), compatible with our 48-joint format.

**Method:**
1. Download MuteMotion WLASL dataset from Kaggle (MediaPipe Holistic 553 landmarks)
2. Extract our 48 joints: 0–20 left hand, 21–41 right hand, 42–47 pose (map from 553 landmarks)
3. Implement masked reconstruction:
   - Randomly mask 75% of joints per frame
   - ST-GCN encoder processes visible joints
   - Lightweight GCN decoder reconstructs masked joints
   - Loss: MSE on masked joint positions + velocity prediction
4. Pre-train on WLASL (~13K samples) for 200 epochs
5. Discard decoder, fine-tune encoder on KSL 1,787 samples

**Expected gain:** +3–8pp (conservative given ASL→KSL domain gap)
**Compute:** ~4h pre-training + 1h fine-tuning on A100
**Paper:** MASA, IEEE TCSVT 2024; SkeletonMAE, ICCV 2023
**Data:** MuteMotion on Kaggle (https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

---

### V45_expr6: Self-Knowledge Distillation (SML/SGAR-DGCN)
**Motivation:** SML (Knowledge-Based Systems 2024) achieves 96.85% on AUTSL using self-KD where deeper GCN blocks teach shallower ones during training. Avoids the external-teacher failure (V33: KD with overconfident teachers). Free regularization, no separate model.

**Method:**
- Add auxiliary classification heads at GCN blocks 4, 7, 10 (early, mid, late)
- During training: CE loss at each head + KL divergence between consecutive heads
  `L_SKD = CE(head_10) + α*KL(head_7||head_10) + α*KL(head_4||head_7)`
  where α=0.3 (softer supervision from deeper→shallower blocks)
- At inference: only use final head (no overhead)
- Temperature T=4 for KL (softer distributions between heads — not the near-one-hot external teacher problem)

**Expected gain:** +1–3pp (self-distillation regularization without external teacher bias)
**Base architecture:** exp5 or exp1
**Paper:** SML/SGAR-DGCN, Knowledge-Based Systems 2024 (AUTSL 96.85%, WLASL 55.85%)

---

### V45_expr7: DABN — Diversity-Adaptive Batch Normalization
**Motivation:** DATTA (arXiv 2024) shows fixed AdaBN is suboptimal when test batch diversity varies. DABN dynamically blends source/test/instance BN stats based on per-batch diversity score. Relevant because our 140-video test set has 3 signers with very different styles.

**Method:**
- Compute diversity score for each test batch: variance of pairwise feature angles
- High diversity (batch contains multiple signer styles) → weight toward instance norm
- Low diversity (homogeneous batch) → weight toward test BN stats
- Formula: `mu_DABN = mu_source + alpha(diversity) * (mu_test - mu_source)`
  where alpha is a learned function of the diversity score
- Applied at inference only (no training change needed)
- Test on all 7 models in our ensemble

**Expected gain:** +0.5–2pp over current AdaBN (mainly helps when batch diversity mismatches)
**Implementation:** ~50 lines, inference-only change
**Paper:** DATTA, arXiv Aug 2024

---

## TIER 3 — High Effort / Research Risk (1 week+)

### V45_expr8: WLASL Supervised Pre-training + DSBN Fine-tuning
**Motivation:** Cross-dataset transfer paper (arXiv 2403.14534) shows Domain-Specific Batch Normalization (DSBN) outperforms naive fine-tuning for under-resourced SLR. Pre-train on labeled WLASL (13K samples, ASL), then DSBN fine-tune on KSL.

**Method:**
1. Same data pipeline as expr5 (MuteMotion MediaPipe skeletons)
2. Train supervised ST-GCN on WLASL-100 class subset (5K samples) — same topology as KSL
3. DSBN: separate BN layers for source (WLASL) and target (KSL) domains
   - Source BN used during WLASL training
   - Target BN initialized from source, fine-tuned on KSL with frozen backbone
4. Evaluate with target BN only

**Expected gain:** +2–5pp (supervised pre-training + domain-specific normalization)
**Paper:** Transfer Learning for Cross-dataset SLR, arXiv 2403.14534

---

### V45_expr9: SinMDM Per-Class Diffusion Augmentation
**Motivation:** SinMDM (ICLR 2024) trains a diffusion model on a SINGLE motion sequence to generate diverse variations. With ~60 samples per class, train one SinMDM per class to synthesize new signers' versions of each sign. Different from V36 (failed) because it generates motion variations, not interpolations.

**Method:**
1. For each of 30 sign classes: train SinMDM on the ~60 training samples
2. Generate 5× synthetic variants per original sample (300 synthetic per class)
3. Add synthetic samples to training set with soft labels (label smoothing α=0.2)
4. Train exp5 architecture on augmented dataset (~7K total samples)

**Expected gain:** +1–4pp if diffusion generates realistic variations (uncertain)
**Risk:** V36 synthetic signer failed (-5pp); diffusion quality on 60-sample domain is uncertain
**Paper:** SinMDM, ICLR 2024

---

## Priority Order for Implementation

| Priority | Experiment | Method | Expected Gain | Effort | Paper Confidence |
|---|---|---|---|---|---|
| 1 | V45_expr1 | Sequence-level GRL | +2–5pp combined | 1–2 days | HIGH (directly addresses V44_expr2 failure) |
| 2 | V45_expr3 | Multi-Scale TCN | +1–3pp | 1–2 days | HIGH (universal in SOTA papers) |
| 3 | V45_expr2 | Hard Negative SupCon | +1–3pp | 1–2 days | HIGH (incremental improvement) |
| 4 | V45_expr4 | LLM Semantic Supervision | +2–5pp | 2–3 days | MEDIUM (multiple papers converge) |
| 5 | V45_expr5 | MASA Pre-training | +3–8pp | 4–5 days | MEDIUM (data pipeline risk) |
| 6 | V45_expr6 | Self-KD | +1–3pp | 2 days | MEDIUM (avoids V33 failure mode) |
| 7 | V45_expr7 | DABN | +0.5–2pp | 1 day | MEDIUM (inference-only) |
| 8 | V45_expr8 | Supervised WLASL+DSBN | +2–5pp | 5–7 days | MEDIUM |
| 9 | V45_expr9 | SinMDM diffusion aug | +1–4pp | 5–7 days | LOW (V36 failed) |

---

## Key Research Insights (for paper discussion)

1. **Sequence-level vs frame-level adversarial**: GRL must be applied AFTER temporal pooling to preserve motion features. Frame-level reversal destroys word motion encoding (V44_expr2 confirmed).

2. **Multi-scale temporal is the missing piece**: Every 2024–2025 top paper uses dilated TCN. Our single kernel=9 is likely too uniform for signs of varying speed.

3. **LLM text as training signal**: CrossGLG (ECCV 2024) shows text descriptions provide domain-invariant semantic anchors that help cross-signer generalization at zero inference cost.

4. **Pre-training is the clearest path to 80%**: MASA+WLASL gives +25pp on WLASL100; even conservative transfer (+3–8pp) pushes us past 75% single model. Combined with ensemble: potentially 80%+.

5. **Diffusion augmentation risk**: V36 synthetic signers failed (-5pp); SinMDM differs by using motion variation vs interpolation, but small-domain diffusion quality is uncertain.

---

## References

- MASA: "Motion-Aware Masked Autoencoder with Semantic Alignment for SLR" — IEEE TCSVT 2024
- SML/SGAR-DGCN: "Skeleton-based Multi-feature Learning" — Knowledge-Based Systems 2024 (AUTSL 96.85%)
- DSTSA-GCN: "Dynamic Spatial-Temporal Semantic Awareness GCN" — Neurocomputing 2025
- CrossGLG: "LLM Guides One-Shot Skeleton Action Recognition" — ECCV 2024
- SCoPLe: "Semantic-guided Cross-Modal Prompt Learning" — CVPR 2025
- DATTA/DABN: "Diversity Adaptive Test-Time Adaptation" — arXiv Aug 2024
- Signer Removal Module: Zuo & Mak — ACM TOMM 2024
- Class-Aware Contrastive: Bian et al. — ACCV 2024
- SinMDM: "Single Motion Diffusion" — ICLR 2024
- Transfer Learning SLR: "Cross-dataset Isolated SLR in Under-Resourced Datasets" — arXiv 2403.14534
- SkeletonMAE: "Graph-based Masked Autoencoder for Skeleton Pre-training" — ICCV 2023
- STARS: ICLR 2025 (two-stage SSL for skeleton)
