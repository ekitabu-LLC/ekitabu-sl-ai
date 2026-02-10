# V24 Feasibility Assessment

**Date:** 2026-02-10
**Analyst:** ML Engineering Agent (codebase-analyst)
**Based on:** v22 codebase analysis, v23 failure analysis, v19-v23 version history

---

## 1. Architecture Inventory: What V22 Has

### 1.1 Model Architecture (KSLGraphNetV22)

**File:** `/scratch/alpine/hama5612/ksl-dir-2/train_ksl_v22.py` (lines 988-1111)

| Component | Details | Status |
|-----------|---------|--------|
| **ST-GCN Backbone** | 4 layers, GConv + MultiScaleTCN (kernels 3,5,7), channel progression 9->64->64->128->128 | WORKING WELL |
| **Attention Pooling** | 2-head temporal attention over node-averaged GCN output -> 256-dim embedding | WORKING WELL |
| **Auxiliary Branch** | 53 features (33 angles + 20 fingertip distances) -> MLP(128->64) -> TemporalConv1d(k=5) -> AttentionPool -> 64-dim | WORKING WELL |
| **Classifier** | Linear(320->64->nc), standard CE | WORKING |
| **GRL Signer Head** | Linear(320->64->num_signers), gradient reversal | PARTIALLY EFFECTIVE |
| **SupCon Loss** | Weight 0.1, temperature 0.07, applied to 320-dim embedding | WORKING |
| **Graph Topology** | 48 nodes: 21 LH + 21 RH + 6 pose, hand edges + pose edges, cross-hand edge (0.3 weight) | WORKING |
| **Normalization** | Wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose) | WORKING |

**Total parameters:** 776,544 (388K per split)
**Embedding dimension:** 320 (256 GCN + 64 aux)
**Input:** GCN: (B, 9, 90, 48) -- 9ch = xyz+velocity+bone; Aux: (B, 90, 53)

### 1.2 Data Pipeline

**Dataset class:** `KSLGraphDataset` (lines 609-821)
- 750 training samples, 728 validation samples across 30 classes (15 numbers + 15 words)
- After signer-group dedup: ~250-285 samples per split for training
- 4 unique signers in training data (filename pattern: `CLASS-SIGNER-REP.npy`)
- 140 real-tester samples (59 numbers + 81 words) from 3 unseen signers

**Feature computation:**
- Joint coordinates (xyz): raw after wrist normalization
- Velocity: frame difference of coordinates
- Bone vectors: child-parent coordinate differences (using 48-node PARENT_MAP)
- Joint angles: arccos of bone dot products at branching joints (33 features)
- Fingertip distances: pairwise L2 between fingertips (20 features = 10 LH + 10 RH)

### 1.3 Training Setup

- Optimizer: AdamW, lr=1e-3, wd=1e-3
- Scheduler: CosineAnnealing with 10-epoch warmup
- Loss: CE(label_smoothing=0.1) + SupCon(0.1) + GRL(signer_adversarial)
- GRL ramp: lambda from 0 to 0.3 over 100 epochs
- Signer-balanced batch sampling (custom sampler)
- 12 augmentation types (moderate intensity)
- Mixup: alpha=0.2
- Early stopping: patience=40

### 1.4 What's Working Well

1. **Wrist-centric normalization** -- eliminates global position/scale bias
2. **Bone + velocity features** -- more signer-invariant than raw xyz
3. **GRL from epoch 0** -- measurable signer accuracy reduction (31.7% at best epoch, near 25% chance)
4. **Signer-balanced sampling** -- ensures cross-signer positive pairs for SupCon
5. **Moderate augmentation** -- enough diversity without signal destruction
6. **Auxiliary branch** -- captures pose-invariant joint angle and distance features
7. **Overall simplicity** -- single-stage training, no complex scheduling

---

## 2. What Failed in V23 and Why

### 2.1 Multi-Stream Fusion: FAILED

**File:** `/scratch/alpine/hama5612/ksl-dir-2/train_ksl_v23.py`

**What was tried:** 4 independent ST-GCN models, each with 3-channel input (joint, bone, velocity, bone_velocity). Grid-searched fusion weights on validation. Total 3.1M params, 35 min training.

**Why it failed:**
- **Streams are highly correlated on OOD data.** All 4 streams face the same fundamental problem: they cannot represent signing styles from unseen signers. Different feature views of the same underlying data do not provide complementary error patterns when the core issue is signer distribution mismatch.
- **Joint stream alone (37.3% numbers) outperformed fused result (33.9%).** Fusion introduced noise rather than signal.
- **Low inter-stream agreement on real data:** Only 18.6% (numbers) and 29.6% (words) of samples had all 4 streams agree. But disagreement was random, not complementary.
- **4x compute cost for negative returns.** 35.3 min vs 7.3 min, with worse accuracy.

**Lesson for v24:** Multi-stream ensemble within the same training distribution is pointless. Diversity must come from the training DATA (more signers), not from splitting features into streams. Avoid any approach that multiplies model count without diversifying the training signal.

### 2.2 Focal Loss: NO BENEFIT

**What was tried:** gamma=2.0, per-class alpha from inverse frequency.

**Why it failed:** The sink-class problem (66, Tortoise) is not caused by class imbalance during training. It's caused by OOD samples mapping to well-learned class prototypes. Focal loss changes gradient weighting during training but doesn't change the fundamental feature-space geometry that causes prediction collapse on OOD data.

**Lesson for v24:** Don't try to fix inference-time OOD problems with training-time loss modifications. The sink-class phenomenon changes identity every version (v19: 444, v21: 268, v22: 66, v23: 444 again) -- it's a symptom, not a cause.

### 2.3 Post-Hoc Calibration: ACTIVELY HARMFUL

**What was tried:** Per-class temperature + bias optimized via L-BFGS-B to minimize NLL on validation set.

**Why it failed:** Calibration parameters learned on validation data (from training signers) encode signer-specific distributional patterns. These patterns are *opposite* to what OOD signers produce. Result: confidence calibration degraded from v22's useful Words HIGH=89% to v23's useless 48.4%. The calibration inflated confidence (81% of predictions became HIGH) while reducing accuracy.

**Lesson for v24:** Never calibrate on same-signer validation data if the goal is cross-signer deployment. Any calibration must use cross-signer data (LOSO) or be signer-agnostic.

---

## 3. Practical Constraints

### 3.1 Compute (Alpine HPC)

| Resource | Limit | V22 Usage | Implication |
|----------|-------|-----------|-------------|
| GPU | A100-PCIE-40GB MIG 3g.20gb (~20GB usable) | ~1.5GB peak | Plenty of headroom |
| Time | 1 hour (atesting_a100) | 7.3 min | ~8x headroom |
| Memory | 2GB CPU RAM | ~1.5GB | Tight, increase to 4GB for pretraining |
| CPUs | 2 | 2 (DataLoader workers) | Sufficient |
| Storage | Scratch space | ~100MB data + checkpoints | No issues |

**v24 budget:** Can comfortably run models up to ~10M params within the 1-hour limit. A pretrained backbone fine-tuning session would need ~15-20 min. Transfer learning with frozen backbone would need ~5-10 min.

### 3.2 Data

| Aspect | Value |
|--------|-------|
| Training samples (raw) | 750 total (30 classes) |
| Training samples (after dedup) | ~535 total (~250 numbers, ~285 words) |
| Validation samples | 728 total |
| Real-test samples | 140 total (59 numbers, 81 words) |
| Training signers | 4 unique |
| Test signers | 3 unseen |
| Samples per signer per class | ~3-5 repetitions |
| Landmarks | 48 nodes x 3 coordinates per frame |
| Frame count | Variable (resampled to 90) |
| Data format | `.npy` files, (frames, 225) -- 75 pose + 63 LH + 63 RH landmarks |

**The data bottleneck is signers, not samples.** With 4 signers, the ceiling is ~40% real accuracy regardless of architecture. Each additional signer is worth more than any architectural change.

### 3.3 Software Stack

- Python 3.11, PyTorch 2.5.1+cu121
- MediaPipe 0.10.14 (training data was extracted with this version)
- No pretrained weights currently available on the cluster
- `scipy` available (used in v23 calibration)
- Conda environment `ksl` has standard ML packages

---

## 4. Modification Points for V24 Approaches

### 4.1 Transfer Learning Integration Points

**Where to plug in a pretrained backbone:**

The current model is initialized from scratch at `train_ksl_v22.py:1170-1182`. To integrate a pretrained backbone:

1. **Replace STGCNBlock layers (lines 1007-1011):** Load pretrained ST-GCN/MSG3D/CTR-GCN weights. The current channel progression is `[9, 64, 64, 128, 128]`. A pretrained model would have its own channel sizes -- need an adapter layer if dimensions don't match.

2. **Freeze backbone, train classifier (line 1047-1052):** Simple modification. Add `for p in self.layers.parameters(): p.requires_grad = False` after loading weights.

3. **Input channel adapter:** V22 uses 9 channels (xyz+vel+bone). Pretrained models from WLASL/NTU-RGB-D typically use 3 channels (xyz only). Options:
   - Add a 1x1 conv to map 9->pretrained_channels at line ~1076 (before data_bn)
   - Drop velocity+bone channels and use only xyz (loses information)
   - Fine-tune first layer only to accept 9 channels

4. **Node count adapter:** V22 uses 48 nodes (21+21+6). Standard body pose datasets use 17-25 nodes. Options:
   - Subset graph: use only the 6 pose nodes as bridge to standard skeleton
   - Graph remapping: map 48 nodes to pretrained graph topology via learnable projection

**Implementation complexity:** MEDIUM-HIGH. The main challenge is graph topology mismatch -- pretrained models assume body skeleton (17-25 joints), while KSL uses hand skeleton (42 hand + 6 pose joints).

### 4.2 Prototypical Classification Integration Points

**Where to replace the FC classifier with prototype-based classification:**

1. **Remove classifier head (lines 1047-1052):** Replace `self.classifier` with a prototype memory bank.

2. **Compute class prototypes after training:** After the main training loop (line 1374-1411), run all training samples through the model to get embeddings, then compute per-class mean embeddings as prototypes.

3. **Inference by nearest prototype:** In `evaluate_real_testers_v22.py` line 450-458, replace `logits` computation with cosine similarity to prototypes: `logits = cosine_similarity(embedding, prototypes)`.

4. **Optional: learnable temperature for prototype distance:** Add a single learnable parameter `self.proto_temp = nn.Parameter(torch.tensor(1.0))` and use `logits = cosine_sim / proto_temp`.

**Specific code changes needed:**
```
# In KSLGraphNetV22.forward(), replace lines 1105-1106:
#   logits = self.classifier(embedding)
# With:
#   logits = F.cosine_similarity(
#       embedding.unsqueeze(1),    # (B, 1, 320)
#       self.prototypes.unsqueeze(0),  # (1, nc, 320)
#       dim=2
#   ) / self.proto_temp
```

**Implementation complexity:** LOW. The embedding is already computed (line 1102). The classifier is a simple swap. The main question is whether to use prototypes from training data only, or to update them during training (episodic training).

### 4.3 Augmentation Enhancement Points

**Where to add new augmentation strategies:**

1. **Synthetic signer generation (in `__getitem__`, lines 688-749):** After normalization (line 715), before feature computation. Insert a signer-style transfer augmentation that morphs hand proportions, signing speed, and gesture amplitude to simulate novel signing styles.

2. **Cross-signer mixup (in training loop, lines 1247-1265):** Currently mixup mixes random samples. A signer-aware mixup would specifically mix samples from different signers of the same class, creating interpolated signing styles.

3. **CutMix on temporal dimension:** Replace segments of one sample's temporal sequence with segments from a different signer performing the same sign. Insert at line ~787 (after temporal sampling, before GCN feature assembly).

4. **Style transfer via feature perturbation:** After computing the embedding (line 1102), add noise drawn from the distribution of cross-signer embedding differences. This would require computing per-class, cross-signer embedding statistics during a warmup phase.

**Implementation complexity:** LOW-MEDIUM. Most augmentations are straightforward additions to the existing augmentation pipeline. The dataset `__getitem__` method is already structured to accommodate additional augmentations.

### 4.4 Domain Generalization Integration Points

**Where to add domain generalization techniques:**

1. **LOSO (Leave-One-Signer-Out) validation:** Modify `train_split()` at line 1118 to split data by signer ID rather than using the fixed train/val directories. The signer IDs are already extracted (lines 650-656). This requires restructuring the data loading, not the model.

2. **Meta-learning (MAML/ProtoNet) training loop:** Replace the standard epoch loop (lines 1220-1353) with an episodic training loop. Each episode: sample support set (N classes x K signers), compute prototypes, classify query set (held-out signer). This is a significant refactor of the training loop but doesn't touch the model architecture.

3. **Domain-specific BatchNorm:** Replace `self.data_bn` (line 1001) and all `BatchNorm2d` layers with domain-conditional BN that maintains separate statistics per signer. At inference time, use running statistics from all signers (aggregated).

4. **Gradient-based signer normalization (stronger GRL):** The current GRL head (lines 1055-1059) could be replaced with a multi-task adversarial setup: add separate adversarial heads for hand-size, signing-speed, and gesture-amplitude, each with its own gradient reversal.

---

## 5. Implementation Effort Estimates

### Tier 1: LOW Effort (1-2 hours implementation, can reuse v22 infrastructure)

| Approach | Effort | Expected Impact | Risk | Notes |
|----------|--------|-----------------|------|-------|
| **Prototypical classifier** (replace FC head with nearest-prototype) | LOW | +2-5% real | LOW | Drop-in replacement for classifier; embedding already computed |
| **LOSO validation** (leave-one-signer-out instead of fixed split) | LOW | Better val-real correlation (no direct accuracy gain) | LOW | Data restructuring only; reveals true cross-signer performance |
| **Signer-aware cross-signer mixup** | LOW | +1-3% real | LOW | Modify mixup to pair same-class, different-signer samples |
| **Disable all zero-impact components** from v23 postmortem (focal, calibration) | LOW | Baseline maintenance | NONE | Keep v22 as-is, confirmed best |
| **Heavier signer augmentation** (larger bone/hand-size perturbation ranges) | LOW | +0-2% real | LOW | Just changing config values |

### Tier 2: MEDIUM Effort (4-8 hours implementation)

| Approach | Effort | Expected Impact | Risk | Notes |
|----------|--------|-----------------|------|-------|
| **Episodic/meta-learning training loop** (ProtoNet-style) | MEDIUM | +3-8% real | MEDIUM | Major refactor of training loop; needs careful episode sampling with only 4 signers |
| **Synthetic signer generation** (learned morphological perturbation) | MEDIUM | +2-5% real | MEDIUM | Need to estimate cross-signer variation statistics; may not match real variation |
| **Cross-signer domain adversarial augmentation** (stronger GRL variants) | MEDIUM | +1-3% real | LOW | Multi-head adversarial with separate attribute heads |
| **Temporal augmentation expansion** (speed randomization, segment shuffling) | MEDIUM | +1-2% real | LOW | More diverse temporal variations to simulate different signing speeds |
| **Knowledge distillation** from v22+v23 ensemble to single model | MEDIUM | +0-2% real | LOW | Use v22 and v23 as teachers; train student with soft targets |

### Tier 3: HIGH Effort (8-20 hours, may require downloading pretrained weights)

| Approach | Effort | Expected Impact | Risk | Notes |
|----------|--------|-----------------|------|-------|
| **Transfer learning from WLASL/AUTSL** (pretrained ST-GCN backbone) | HIGH | +5-10% real | HIGH | Graph topology mismatch (body vs hand); need to download/process external datasets; may not fit in 1hr training |
| **Full meta-learning framework** (MAML with inner/outer loops) | HIGH | +3-8% real | HIGH | Very complex with 4 signers; inner loop on 3 signers, outer on 1; risk of instability |
| **Video-level pretrained features** (I3D/SlowFast on raw video) | HIGH | +3-7% real | HIGH | Requires raw video access; may capture appearance cues that help but also overfit to appearance |
| **Data collection pipeline** (recruit additional signers) | HIGH (logistics, not code) | +15-25% real | LOW | Not a code task; this is the actual bottleneck |

---

## 6. Recommended V24 Architecture

### 6.1 Design Philosophy

Based on v19-v23 history, the following principles should guide v24:

1. **Do NOT increase model complexity.** V22 (776K params) outperformed v23 (3.1M params). The dataset cannot support larger models.
2. **Focus on the CLASSIFICATION HEAD, not the backbone.** The ST-GCN backbone is adequate; the problem is decision boundaries collapsing on OOD data.
3. **All optimizations must be evaluated with LOSO**, not fixed train/val split. Val accuracy is a misleading metric (v23: 95.6% val, 37.9% real).
4. **Compute budget: single model, <15 min training.** No ensembles.

### 6.2 Recommended V24: Prototypical + LOSO + Signer Augmentation

**Architecture: KSLGraphNetV24**

```
Base: v22 ST-GCN backbone (UNCHANGED: 4 layers, hidden=64, 9ch input)
  |
  v
Embedding: 320-dim (256 GCN + 64 aux) -- UNCHANGED
  |
  v
NEW: Prototypical classifier (replace FC head)
  - Compute per-class prototypes as mean embedding of training support set
  - Classify by cosine similarity to prototypes, scaled by learned temperature
  - At inference: use prototypes from ALL training signers (more robust than FC boundary)
  |
  v
KEEP: GRL signer head (unchanged from v22)
KEEP: SupCon loss (unchanged from v22)
  |
  v
NEW: LOSO validation (leave-one-signer-out cross-validation)
  - Train on 3 signers, validate on 1 (rotate 4 times)
  - Average cross-signer accuracy as primary metric
  - This gives realistic estimate of real-tester performance
  |
  v
NEW: Enhanced signer augmentation
  - Wider bone perturbation: (0.6, 1.4) instead of (0.8, 1.2)
  - Wider hand-size range: (0.6, 1.4) instead of (0.8, 1.2)
  - Cross-signer mixup: when mixing, prefer same-class different-signer pairs
  - Signing speed perturbation: 0.7x to 1.3x temporal resampling
```

**Expected parameters:** ~760K (slightly less than v22 due to removed FC classifier)
**Expected training time:** ~8-10 min per fold x 4 folds = ~35 min total (fits in 1hr)
**Expected real-tester accuracy:** 42-48% combined (based on LOSO giving more honest signal)

### 6.3 Implementation Plan

| Step | Description | Files to Modify | Effort |
|------|-------------|----------------|--------|
| 1 | Copy v22 as v24 base | Create `train_ksl_v24.py` from `train_ksl_v22.py` | LOW |
| 2 | Add LOSO data splitting | Modify dataset/dataloader creation in `train_split()` | LOW |
| 3 | Replace FC classifier with prototypical head | Modify `KSLGraphNetV22` -> `KSLGraphNetV24` class | LOW |
| 4 | Add prototype computation after training | Add post-training prototype extraction function | LOW |
| 5 | Widen signer augmentation ranges | Modify config values | LOW |
| 6 | Add cross-signer mixup | Modify mixup section in training loop | LOW |
| 7 | Create eval script for v24 | Adapt `evaluate_real_testers_v22.py` for prototype inference | LOW |
| 8 | Create SLURM script | Adapt `slurm/v22.sh` | LOW |
| 9 | Run LOSO experiments and tune | Submit jobs, analyze | MEDIUM |

**Total estimated implementation:** 4-6 hours of coding, 2-3 hours of experimentation.

### 6.4 Stretch Goals (if base v24 works)

1. **Episodic training** (ProtoNet-style): Instead of standard batches, sample episodes with support/query split per signer. Higher effort but potentially +3-5% more.

2. **Transfer learning** (if pretrained weights can be obtained): Freeze a pretrained ST-GCN backbone, only train the prototypical head + aux branch. This would require obtaining WLASL-pretrained weights and adapting the graph topology.

3. **Synthetic signer generation**: Use the cross-signer statistics from LOSO to generate synthetic signers via learned interpolation in embedding space.

### 6.5 What NOT to Do in V24

Based on v23 failure analysis:

| Approach | Reason to Avoid |
|----------|----------------|
| Multi-stream ensemble | Failed in v23; streams correlated on OOD data; 4x cost |
| Post-hoc calibration on val data | Failed in v23; val calibration doesn't transfer |
| Focal loss | No measurable benefit in v23 |
| TTA (test-time augmentation) | Consistently hurts (-1.9% combined in v22) |
| Increasing model size beyond 1M params | Dataset too small; v23's 3.1M params was worse |
| Any approach that optimizes val accuracy as primary metric | Val is unreliable (v23: 95.6% val, 37.9% real) |

---

## 7. Risk Assessment

### 7.1 Risks of Recommended V24

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Prototypical classification worse than FC on val | LOW | MEDIUM | Keep FC as fallback; prototypical should be better on OOD by design |
| LOSO too pessimistic (only 3 training signers per fold) | MEDIUM | HIGH | Use LOSO for comparison/tuning, not as final metric; real-tester eval remains the ground truth |
| 4-fold LOSO too slow for 1hr limit | LOW | LOW | Each fold trains in ~8 min; 4 folds = 32 min; well within 1hr |
| Wider augmentation ranges hurt val accuracy | LOW | MEDIUM | Val may drop, but real-tester should improve; prioritize LOSO metric |
| No improvement over v22 | MEDIUM | MEDIUM | The 4-signer ceiling is real; v24 may only gain 2-5% |

### 7.2 Ceiling Analysis

With the recommended v24 approach (no new signers):

| Scenario | Estimated Real Accuracy | Confidence |
|----------|------------------------|------------|
| Best case (everything works) | 45-48% | LOW |
| Likely case | 41-44% | MEDIUM |
| Worst case (regression) | 37-39% | LOW |
| V22 baseline | 39.8% | KNOWN |

**Honest assessment:** V24 without new signers has a realistic ceiling of ~45% combined. The expected marginal gain over v22 is 2-5 percentage points. The ONLY path to >50% is collecting more signers.

---

## 8. Summary

### What V24 Should Be

V24 should be a **disciplined, low-risk iteration** that:
1. Replaces the FC classifier with prototypical classification (addresses sink-class problem by using distribution-based decisions instead of learned boundaries)
2. Adds LOSO cross-validation (provides realistic performance estimates during development)
3. Widens signer-mimicking augmentation (simulates more signing style diversity)
4. Maintains v22's backbone, training stability, and simplicity

### What V24 Should NOT Be

V24 should NOT be another "architecturally ambitious" version like v23. The v23 postmortem conclusively showed that increased model complexity (4-stream ensemble, calibration, focal loss) yielded NEGATIVE returns. The problem is data diversity, not model capacity.

### Decision Point

After v24 results:
- **If v24 > 43% real:** Prototypical approach is working; consider episodic training (v25) for further gains
- **If v24 ~ 40% real:** 4-signer ceiling confirmed definitively; all future effort should go to data collection
- **If v24 < 38% real:** Prototypical classification hurt; revert to v22, focus exclusively on data collection

---

## Appendix A: Key Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Model definition | `train_ksl_v22.py` | 988-1111 | KSLGraphNetV22 class |
| Forward pass | `train_ksl_v22.py` | 1061-1111 | GCN + aux + classifier + GRL |
| Classifier (to replace) | `train_ksl_v22.py` | 1047-1052 | FC head: Linear(320->64->nc) |
| GRL signer head (keep) | `train_ksl_v22.py` | 1055-1059 | Adversarial: Linear(320->64->ns) |
| Training loop | `train_ksl_v22.py` | 1220-1353 | Epoch loop with mixup, SupCon, GRL |
| Dataset __getitem__ | `train_ksl_v22.py` | 667-821 | Full data pipeline + augmentation |
| Augmentation config | `train_ksl_v22.py` | 148-192 | V22_CONFIG dictionary |
| Evaluation pipeline | `evaluate_real_testers_v22.py` | 440-504 | Inference + TTA + confidence |
| SLURM template | `slurm/v22.sh` | 1-109 | Alpine job submission |
| V23 multi-stream (failed) | `train_ksl_v23.py` | 1042-1167 | KSLGraphNetV23 -- DO NOT REUSE |
| V23 fusion (failed) | `train_ksl_v23.py` | 1429-1536 | Grid-search fusion -- DO NOT REUSE |
| V23 calibration (failed) | `train_ksl_v23.py` | 1543-1685 | Per-class temp+bias -- DO NOT REUSE |

## Appendix B: Data Layout

```
data/
  train_v2/
    9/              # 15 number classes
      9-1-1.npy     # Format: CLASS-SIGNER-REP.npy
      9-1-2.npy
      ...
      9-4-5.npy     # 4 signers, ~5 reps each
    Agreement/      # 15 word classes
      Agreement-1-1.npy
      ...
  val_v2/
    9/
      9-1-1.npy     # Same signer pool as training
      ...
  checkpoints/
    v22_numbers/best_model.pt
    v22_words/best_model.pt
  results/
    v22_analysis/   # 7 plots + 2 reports
```

## Appendix C: Version Performance History

| Version | Val Combined | Real Combined (no-TTA) | Params | Time | Key Change |
|---------|-------------|----------------------|--------|------|------------|
| v19 | 87.3% | 34.2% | 4.5M | 18.4m | Baseline |
| v20 | 86.9% | 30.5% | 413K | 5.8m | Aggressive dedup (hurt) |
| v21 | 76.5% | 31.2% | 735K | 4.6m | ArcFace (catastrophic) |
| **v22** | **93.3%** | **39.8%** | **776K** | **7.3m** | **Simplification (BEST REAL)** |
| v23 | 95.6% | 37.9% | 3.1M | 35.3m | Multi-stream (failed) |
| v24 (est) | ~85-90% (LOSO) | 41-45% | ~760K | ~35m (4-fold) | Prototypical + LOSO |
