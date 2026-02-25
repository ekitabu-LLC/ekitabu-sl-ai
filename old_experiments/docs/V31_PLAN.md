# V31 Plan: Closing the Gap from 65.9% to 85%

## Current State
- **Best**: V30 Phase 1 Ensemble+Alpha-BN = 65.9% combined (Numbers 62.7%, Words 69.1%)
- **Best standalone**: V28 AdaBN Global = 58.2%
- **Target**: 85%
- **Gap**: 19.1pp

## Root Cause Analysis (from error-analyst)

### Two failure modes account for ~20pp:

**1. Signer-3 catastrophic failure (15pp potential recovery)**
- Signer 2: 93% numbers, 80% words = 85% combined (AT TARGET!)
- Signer 3: 33% numbers, 43% words = 40% combined (45pp below signer 2)
- Affects ALL models — domain shift problem

**2. Five hard number classes (5-10pp potential recovery)**
- 48 (25%), 66 (25%), 89 (25%), 91 (50%), 388 (50%)
- Persistent confusion clusters: 100↔66↔91, 48→444, 388→35
- Multi-digit numbers systematically harder than single-digit

### Key confusions:
- Numbers: 48→444 (3x), 100↔66 (4x), 388→35 (2x), 89↔9 (2x)
- Words: Tomatoes (20%), Tortoise→Sweater (50%), Apple→various (50%)

---

## V31 Experiments (prioritized, test ONE at a time)

### Exp 1: GroupNorm replacing BatchNorm (PRIORITY 1)
**Rationale**: Eliminates BN domain gap at the source. No more AdaBN/Alpha-BN tricks needed. SAR paper (ICLR 2023) showed GroupNorm is more stable than BN for test-time adaptation. GraphNorm (ICML 2021) designed specifically for GNNs.

**Implementation**:
- Replace all BatchNorm2d/1d with GroupNorm in v28 architecture
- Remove signer-adversarial head (no longer needed for domain invariance)
- Train with v28's clean ERM approach
- Test WITHOUT AdaBN (should not be needed)

**Expected**: If BN was the main domain gap source, could match signer-2 performance (85%) across all signers. Conservative estimate: +5-10pp.

**Risk**: May lose 2% training accuracy. Low implementation risk.

### Exp 2: Prototypical Loss (PRIORITY 2)
**Rationale**: "Data-Efficient ASL Recognition via Prototypical Networks" (Dec 2024) showed +13% on WLASL with small samples. Episodic training forces generalizable features.

**Implementation**:
- Keep v28 multi-stream ST-GCN architecture
- Replace cross-entropy with prototypical loss
- Episodic sampling: 5-way 5-shot per batch
- Compute class prototypes as mean embeddings, classify by distance

**Expected**: +5-10pp standalone, orthogonal to ensemble/Alpha-BN.

**Risk**: Episodic sampling with 30 classes and 895 samples may be tricky. Need at least 3 examples per class per episode.

### Exp 3: Joint Mixing Augmentation (PRIORITY 3)
**Rationale**: Skeleton-specific mixup (ACM Trans. Multimedia, 2024). Simple, proven, no architecture changes. Consistent improvements on NTU-60/120.

**Implementation**:
- During training: x_mix = λ*x_i + (1-λ)*x_j for same-class pairs
- λ ~ Beta(0.2, 0.2), probability 0.5
- Apply to all three streams (joint, bone, velocity)

**Expected**: +3-5pp. Low risk, easy to combine with other experiments.

### Exp 4: Hard-Example Mining + Focal Loss (PRIORITY 4)
**Rationale**: Error analysis shows 5 classes at 25-50%. Oversampling and focal loss focus learning on hard cases.

**Implementation**:
- Weighted sampling: P(class) ∝ 1/accuracy²
- Oversample 48, 91, 388, 66, 89 by 5x
- Focal loss γ=2 to focus on hard examples
- Add speed augmentation (0.7x-1.3x) for signer-3 adaptation

**Expected**: +5-8pp on worst classes, +3-5pp overall.

### Exp 5: Cross-Signer Contrastive Loss (PRIORITY 5)
**Rationale**: Train signer-invariant features explicitly. Multiple 2024 papers show gains for cross-signer generalization.

**Implementation**:
- Add contrastive loss: same class + different signer → pull together
- Weight 0.01-0.1, batch size 64
- Use signer ID labels already in training data

**Expected**: +2-5pp. Medium risk on small dataset.

---

## Execution Plan

### Phase 1: Architecture (1 week)
- Exp 1 (GroupNorm) — numbers only, compare to v28 baseline
- If GroupNorm works: skip AdaBN entirely in future
- If GroupNorm fails: stay with BN + AdaBN

### Phase 2: Training (1 week)
- Exp 2 (Prototypical) — numbers only
- Exp 3 (Joint Mixing) — numbers only, can run parallel with Exp 2

### Phase 3: Targeted Fixes (1 week)
- Exp 4 (Hard-example mining) on best model from Phase 1-2
- Exp 5 (Cross-signer contrastive) on best model

### Phase 4: Full Evaluation (2 days)
- Train words for best experiment(s)
- Full ensemble evaluation with Alpha-BN
- Compare to current best (65.9%)

---

## Success Criteria
- Any single experiment > 58.2% (v28 standalone baseline)
- Combined ensemble > 65.9% (current best)
- Signer-3 accuracy > 60% (currently 40%)

## What NOT to Do (Lessons from v30/v30b)
- Don't stack regularizers — test one at a time
- Don't use SWAD — incompatible with AdaBN
- Don't use label smoothing — degrades AdaBN
- Don't use MixStyle — conflicts with AdaBN
- Don't use R-Drop — unstable with batch<64
- Don't use BlockGCN — worse than ST-GCN on small data
- Don't use GANs/VAEs — need 10K+ samples

## Long-Term: Data Collection
- Biggest lever remains more signers (~1.6pp/signer)
- Priority: signer-3-like signers (diverse styles)
- Hard-class examples: 48, 91, 388, 66, 89, Tomatoes, Apple
- 20 total signers → estimated +13pp from data alone
