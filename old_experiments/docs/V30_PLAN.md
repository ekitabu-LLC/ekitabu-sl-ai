# V30 Implementation Plan: KSL Recognition

**Current best**: v29 AdaBN Global = 58.4% combined (Numbers 57.6%, Words 59.3%)
**Target**: 70%+ real-world combined accuracy
**Date**: 2026-02-13

---

## Executive Summary

V30 is structured in 3 phases with increasing effort. Phase 1 requires **zero retraining** and targets 62-66%. Phase 2 retrains with new techniques for 66-72%. Phase 3 upgrades the architecture for 72%+. Each phase builds on the previous.

### Root Causes of the 39.6pp Val-to-Real Gap
1. **Signer diversity**: 12 train signers insufficient to cover 3 unseen test signers (especially S3 at 36%)
2. **Catastrophic class failures**: 48 (0%), 66 (0%), Apple (17%), Tomatoes (20%) = 18 wrong predictions / 140 total
3. **Sink classes**: 35 absorbs numbers errors, Friend absorbs words errors
4. **Overfitting to training distribution**: deeper models overfit more (v29 baseline 50.7% < v27 baseline 53.7%)

---

## Phase 1: Quick Wins (No Retraining, 1-2 Days) -> Target 62-66%

These are inference-time or post-hoc improvements applied to existing v28/v29 models.

### 1.1 Temperature Scaling on V28 Logits
- **What**: Learn scalar T on val set, apply `softmax(logits/T)` at test time
- **Effort**: 30 minutes, ~20 lines of code
- **Expected gain**: +0.5-1pp (better calibration for fusion)
- **Risk**: ZERO (T=1 = no change)
- **Priority**: Do first

### 1.2 Alpha-BN: Source-Target BN Statistics Mixing
- **What**: Instead of fully replacing BN stats (AdaBN), mix: `mu = alpha * mu_source + (1-alpha) * mu_target`
- **Effort**: 1 hour, grid search alpha in [0, 0.1, 0.2, ..., 1.0]
- **Expected gain**: +1-3pp
- **Risk**: ZERO (alpha=0 = pure AdaBN, alpha=1 = baseline)
- **Why**: Our ~60 test samples may not cover all 15 classes well; source stats fill gaps
- **Priority**: Do second

### 1.3 TENT + AdaBN Hybrid
- **What**: After AdaBN stat collection, additionally optimize BN affine params (gamma/beta) via entropy minimization
- **Implementation**:
  1. Do AdaBN Global (existing)
  2. Freeze everything except BN weight/bias
  3. Run 1-3 epochs: loss = softmax_entropy(model(x))
  4. LR=1e-3, Adam, batch_size=all (~60)
- **Effort**: 2 hours
- **Expected gain**: +2-4pp over plain AdaBN
- **Risk**: LOW (can fall back to plain AdaBN)
- **Code reference**: github.com/DequanWang/tent
- **Priority**: High - single biggest Phase 1 win

### 1.4 V27 + V28 + V29 Model Ensemble
- **What**: Average softmax outputs from v27 (no-AdaBN, 53.7%), v28 (AdaBN, 58.2%), v29 (AdaBN, 58.4%)
- **Why**: Different architectures make complementary errors:
  - v27: single-stream, no AdaBN dependency, different confusion patterns
  - v28: 3-stream with bone-heavy numbers, velocity-heavy words
  - v29: deeper 8-layer, better on some classes (17: 100%, 91: 50%)
- **Implementation**: Run all 3 models, average softmax, argmax
- **Effort**: 1 hour
- **Expected gain**: +2-4pp
- **Risk**: ZERO
- **Variant**: Try confidence-weighted ensemble (weight by max softmax per sample)

### 1.5 Confidence-Weighted Stream Fusion (V28)
- **What**: Replace fixed grid-searched fusion weights with per-sample entropy-weighted fusion
- **Current**: Numbers = 0.65 * bone, Words = 0.7 * velocity (fixed)
- **New**: For each sample, weight stream inversely by prediction entropy
- **Effort**: 1 hour
- **Expected gain**: +1-2pp
- **Risk**: ZERO

### 1.6 T3A Prototype Classifier (V28/V29)
- **What**: Replace linear classifier with nearest-centroid at test time. Prototypes initialized from classifier weights, updated with confident test features.
- **Effort**: 2 hours
- **Expected gain**: +1-2pp
- **Risk**: VERY LOW (original classifier as fallback)
- **Caveat**: Only ~4 samples/class in test — use high confidence threshold for prototype updates

### Phase 1 Combined Estimate: 62-66% (from 58.4%)

**Implementation order**: 1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.5 -> 1.6
**Total effort**: ~8 hours of coding + evaluation

---

## Phase 2: Training Improvements (Retraining Required, 3-5 Days) -> Target 66-72%

### 2.1 MixStyle for Signer-Invariant Features [HIGH PRIORITY]
- **What**: During training, mix BN statistics between samples from different signers in early GCN layers. Synthesizes "virtual signers" in feature space.
- **Where**: Insert after first 2 GCN blocks (early layers capture signer "style")
- **Implementation**:
```python
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1):
        self.p, self.alpha = p, alpha
    def forward(self, x):
        if not self.training: return x
        if random.random() > self.p: return x
        mu, var = x.mean([2,3], keepdim=True), x.var([2,3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(x.size(0))
        lmbd = Beta(self.alpha, self.alpha).sample([x.size(0),1,1,1]).to(x.device)
        mu_mix = lmbd * mu + (1-lmbd) * mu[perm]
        sig_mix = lmbd * sig + (1-lmbd) * sig[perm]
        return x_normed * sig_mix + mu_mix
```
- **Expected gain**: +2-4pp
- **Risk**: LOW
- **Code reference**: github.com/KaiyangZhou/mixstyle-release

### 2.2 DropGraph Regularization [HIGH PRIORITY]
- **What**: Replace standard dropout with graph-aware dropout. Selects random node, drops all nodes within k-hop graph neighborhood. Forces model to use multiple body parts.
- **Where**: Replace spatial dropout in each GCN block
- **Expected gain**: +1-2pp
- **Risk**: LOW (replaces existing dropout)
- **Code reference**: github.com/kchengiva/DecoupleGCN-DropGraph

### 2.3 Confused-Pair Contrastive Loss [HIGH PRIORITY]
- **What**: Explicitly push apart confused classes in embedding space. Target pairs:
  - Numbers: 66->100, 48->444, 89->35, 388->35, 9->89
  - Words: Apple->Gift/Monday, Tomatoes->Friend, Colour->Friend, Market->Friend
- **How**: Mine hard negatives from confusion matrix. Add contrastive loss:
  `L_pair = max(0, margin - d(confused_a, confused_b) + d(same_class_a, same_class_b))`
- **Expected gain**: +2-3pp (targets our worst classes directly)
- **Risk**: LOW-MEDIUM
- **Synergy**: We already have SupCon infrastructure, extend it with hard-negative mining

### 2.4 Shap-Mix: Shapley-Guided Mixing for Weak Classes [MEDIUM PRIORITY]
- **What**: Use Shapley values to estimate body part saliency per class, guide augmentation mixing to specifically boost weak classes (Apple, 48, 66, Tomatoes)
- **Expected gain**: +2-4pp (targeted at weak classes)
- **Complexity**: MEDIUM
- **Code**: github.com/JHang2020/Shap-Mix (IJCAI 2024)
- **Why this over CutMix**: Shap-Mix preserves discriminative regions while mixing non-discriminative ones

### 2.5 JMDA: Joint Mixing Data Augmentation [MEDIUM PRIORITY]
- **What**: SpatialMix (project 3D->2D, mix spatial info) + TemporalMix (temporal resize + merge). Plug-and-play, no extra params.
- **Expected gain**: +1-3pp (complements existing CutMix)
- **Code**: github.com/aidarikako/JMDA
- **Complexity**: EASY - pure augmentation

### 2.6 Label Smoothing + R-Drop Regularization [EASY WINS]
- **Label smoothing**: epsilon=0.1, one-line change, +0.5-1pp
- **R-Drop**: Forward each sample twice with different dropout, minimize KL divergence between outputs. +0.5-1.5pp
- **Combined effort**: 2 hours
- **Risk**: VERY LOW

### 2.7 SWAD: Stochastic Weight Averaging Densely [MEDIUM PRIORITY]
- **What**: Average model weights densely during training with overfit-aware schedule. Finds flatter minima.
- **Why**: Our val-to-real gap (39.6pp) suggests sharp minima
- **Expected gain**: +1-3pp
- **Implementation**: Track val loss, start averaging when val loss increases
- **Synergy**: Combines well with MixStyle

### 2.8 More Aggressive Body Proportion Augmentation [EASY]
- **What**: Extend current bone-length perturbation from (0.8-1.2) to (0.7-1.3), add:
  - Shoulder width scaling (0.8-1.3)
  - Arm length asymmetry (left vs right)
  - Torso height variation
  - Camera distance simulation (uniform scaling 0.85-1.15)
- **Expected gain**: +1-2pp
- **Risk**: LOW (can always fall back to current range)

### 2.9 Skelbumentations: Realistic Pose Noise [EASY]
- **What**: Simulate MediaPipe estimation errors: joint jitter, occasional swaps, confidence-aware noise
- **Why**: Directly targets train-test distribution gap from real-world camera conditions
- **Expected gain**: +1-3pp (bridges noisy real-tester data gap)
- **Code**: github.com/MickaelCormier/Skelbumentations (WACV 2024)

### Phase 2 Combined Estimate: 66-72% (from Phase 1's 62-66%)

**Implementation priority**: 2.6 -> 2.1 -> 2.2 -> 2.3 -> 2.8 -> 2.9 -> 2.5 -> 2.4 -> 2.7
**Training budget**: ~5 training runs on Alpine A100 MIG (~40 min each)

---

## Phase 3: Architecture Upgrade (1-2 Weeks) -> Target 72%+

### 3.1 BlockGCN Backbone [HIGH PRIORITY]
- **What**: Replace ST-GCN backbone with BlockGCN:
  - Graph distance encoding: `B_ij = exp(-d_ij)` replaces learned adjacency
  - BlockGC: grouped feature convolutions, 40% fewer params
  - Optional: persistent homology for dynamic topology
- **Expected gain**: +2-4pp (better topology representation)
- **Params**: ~500K (down from 780K), less overfitting
- **Code**: github.com/ZhouYuxuanYX/BlockGCN (CVPR 2024)
- **Risk**: MEDIUM (new backbone, needs validation)

### 3.2 ProtoGCN: Prototype-Based Recognition [HIGH PRIORITY]
- **What**: Add prototype memory module:
  - 100 learnable motion prototypes stored in memory bank
  - Attention-based prototype assembly for each input
  - Contrastive learning on prototype responses
  - Motion Topology Enhancement (MTE)
- **Why**: **Specifically improves lowest-accuracy classes by +3.0%** (CVPR 2025 Highlight)
- **Expected gain**: +2-3pp (directly targets our worst classes)
- **Code**: github.com/firework8/ProtoGCN

### 3.3 SkeletonGCL: Graph Contrastive Learning [MEDIUM PRIORITY]
- **What**: Plug-in cross-sequence graph contrastive learning:
  - Dual memory banks (instance + semantic level)
  - Makes intra-class compact, inter-class dispersed
- **Expected gain**: +1-2pp (plug-in on top of any backbone)
- **Code**: github.com/OliverHxh/SkeletonGCL (ICLR 2023)
- **Synergy**: Combines with BlockGCN or ProtoGCN

### 3.4 SL-SLR Self-Supervised Pre-training [HIGH PRIORITY if data available]
- **What**: Self-supervised pre-training designed for sign language with MediaPipe skeletons:
  - Free-negative-pair learning (SL-FPN)
  - Temporal permutation augmentation
  - Works with only 30% labeled data
- **Pre-train on**: All KSL-alpha data (unlabeled) + potentially AUTSL skeleton data
- **Expected gain**: +3-5pp
- **Risk**: MEDIUM (pre-training pipeline complexity)
- **Key advantage**: Uses MediaPipe skeleton = direct compatibility

### 3.5 Multi-Stream with Joint Training [MEDIUM PRIORITY]
- **What**: Keep v28's 3-stream architecture but train jointly:
  - Cross-stream consistency loss (CrosSCLR)
  - Mutual distillation between streams
  - Stream dropout during training
- **Expected gain**: +1-2pp over independent training
- **Risk**: LOW-MEDIUM

### 3.6 GroupNorm Replacement [EXPERIMENTAL]
- **What**: Replace all BatchNorm with GroupNorm for batch-size-independent normalization
- **Why**: SAR paper found "batch norm is crucial factor hindering TTA stability"
- **Trade-off**: May reduce training accuracy but improve generalization
- **Expected gain**: +1-3pp (reduces AdaBN dependency)
- **Risk**: MEDIUM (may need architecture retuning)

### Phase 3 Combined Estimate: 72-78% (from Phase 2's 66-72%)

---

## SLURM Job Configuration

All training on Alpine HPC:
```bash
#SBATCH --partition=atesting_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4     # increased for data loading
#SBATCH --mem=4G              # increased for prototype memory
#SBATCH --time=02:00:00       # 2 hours for Phase 2+ (more epochs)
#SBATCH --output=slurm/v30_%j.out
#SBATCH --error=slurm/v30_%j.err

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate ksl
```

---

## A/B Testing Strategy

Each change is tested independently AND cumulatively:

| Experiment | Changes | Baseline |
|-----------|---------|----------|
| v30a | Temperature scaling only | v28 AdaBN (58.2%) |
| v30b | Alpha-BN mixing | v28 AdaBN (58.2%) |
| v30c | TENT + AdaBN | v28 AdaBN (58.2%) |
| v30d | Version ensemble (v27+v28+v29) | v29 AdaBN (58.4%) |
| v30e | All Phase 1 combined | v29 AdaBN (58.4%) |
| v30f | Phase 1 + MixStyle + DropGraph | v30e |
| v30g | Phase 1 + confused-pair loss | v30e |
| v30h | All Phase 2 combined | v30e |
| v30i | Phase 2 + BlockGCN backbone | v30h |
| v30j | Phase 2 + ProtoGCN | v30h |

**Evaluation**: Always test on real testers (3 signers) with both baseline and AdaBN Global.

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Temperature scaling | ZERO | T=1 fallback |
| Alpha-BN | ZERO | alpha grid search |
| TENT + AdaBN | LOW | Fall back to plain AdaBN |
| Version ensemble | ZERO | Use best single model |
| MixStyle | LOW | Toggle off with p=0 |
| DropGraph | LOW | Replaces existing dropout |
| Confused-pair loss | LOW-MEDIUM | Weight coefficient tuning |
| Shap-Mix | MEDIUM | Fall back to CutMix |
| BlockGCN backbone | MEDIUM | Keep v28 as fallback |
| SL-SLR pre-training | MEDIUM | Can skip if pre-training fails |
| GroupNorm | MEDIUM | A/B test vs BatchNorm |
| Diffusion generation | HIGH | Phase 3+ only, optional |

---

## Timeline

| Day | Activity | Expected Accuracy |
|-----|----------|-------------------|
| Day 1 | Phase 1: Temp scaling, Alpha-BN, TENT+AdaBN | 60-62% |
| Day 2 | Phase 1: Ensemble, confidence fusion, T3A | 62-66% |
| Day 3 | Phase 2: Label smoothing, R-Drop, MixStyle | 64-68% |
| Day 4 | Phase 2: DropGraph, confused-pair loss, body aug | 66-70% |
| Day 5 | Phase 2: JMDA, Shap-Mix, SWAD | 68-72% |
| Week 2 | Phase 3: BlockGCN or ProtoGCN backbone | 72-76% |
| Week 3 | Phase 3: SL-SLR pre-training, SkeletonGCL | 74-78% |

---

## Key Code References

| Technique | Repository | License |
|-----------|-----------|---------|
| TENT | github.com/DequanWang/tent | MIT |
| MixStyle | github.com/KaiyangZhou/mixstyle-release | MIT |
| DropGraph | github.com/kchengiva/DecoupleGCN-DropGraph | MIT |
| BlockGCN | github.com/ZhouYuxuanYX/BlockGCN | Apache 2.0 |
| ProtoGCN | github.com/firework8/ProtoGCN | MIT |
| SkeletonGCL | github.com/OliverHxh/SkeletonGCL | MIT |
| JMDA | github.com/aidarikako/JMDA | MIT |
| Shap-Mix | github.com/JHang2020/Shap-Mix | MIT |
| R-Drop | github.com/dropreg/R-Drop | MIT |
| SL-SLR | (search for latest release) | TBD |
| Skelbumentations | github.com/MickaelCormier/Skelbumentations | MIT |
| SWAD | github.com/khanrc/swad | MIT |

---

## Success Metrics

- **Primary**: Real-tester combined accuracy (no-TTA) > 70%
- **Secondary**: Per-signer minimum > 50% (especially S3)
- **Tertiary**: No zero-accuracy classes
- **Calibration**: Confidence HIGH bucket > 85% accuracy
- **Efficiency**: Training time < 2 hours on A100 MIG
