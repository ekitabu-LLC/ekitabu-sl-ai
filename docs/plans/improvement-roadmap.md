# KSL Model Improvement Roadmap

**Generated: 2026-02-08**
**Current best: v8 at 83.65% | Target: >85% | Stretch: >90%**

## Research Summary

5 parallel research agents investigated: SOTA architectures, augmentation/pretraining, training strategies, v8 hybrid design, and class confusion solutions.

---

## Phase 1: v17 Baseline (READY TO RUN)

**Goal: Fix numbers collapse, establish clean ST-GCN baseline**

Already built in `train_ksl_v17.py`:
- Match numbers capacity to words (hidden_dim=128, 6 layers, ~6.95M params)
- Simple CrossEntropy loss (removed 5 competing losses)
- Strong augmentation for both splits (hand_dropout=0.5, noise=0.015)
- SWA in last 20% of training
- Expected: numbers 65-75%, words 90%+, combined ~80%

```bash
sbatch slurm/v17.sh
```

---

## Phase 2: Multi-Stream Ensemble (Expected: +2-5%)

**Goal: Break 85% with minimal architecture changes**

### 2a. Add velocity/acceleration input streams
- Compute frame-to-frame differences: velocity (order 1), acceleration (order 2)
- Input channels: 3 (xyz) -> 9 (xyz + vel_xyz + acc_xyz)
- OR train 4 separate models: joint, bone, joint-motion, bone-motion
- Fuse with weights: joint:bone:joint-motion:bone-motion = 2:1:2:1

### 2b. Cross-architecture ensemble (v8 LSTM + ST-GCN)
- v8 LSTM already at 83.65%
- Average softmax outputs of v8 + v17 ST-GCN
- Heterogeneous ensembles reduce error by 20-30%

### 2c. Multi-seed ensemble
- Train 3-5 identical models with different seeds, average predictions
- Free 1-2% improvement

---

## Phase 3: Feature-Augmented ST-GCN (Expected: +3-5%)

**Goal: Bring v8's hand feature engineering into ST-GCN**

### 3a. Per-node feature channels (simplest)
Expand from 3 channels to 10 per node:
- Raw xyz (3)
- Distance to wrist (1)
- Velocity xyz (3)
- Acceleration magnitude (1)
- Bone angle to parent (1)
- Bone length ratio (1)

### 3b. Dual-Branch Fusion (most powerful)
Full hybrid architecture combining ST-GCN + TemporalPyramid:
- Branch A: ST-GCN (6 layers, 128 hidden) -> 512-dim features
- Branch B: v8's engineered features -> Multi-scale Conv1d(3,7,15) -> BiLSTM -> Attention pooling -> 512-dim
- Fusion: concat -> MLP -> classifier
- Auxiliary heads for each branch (deep supervision)
- ~8.5M params total

---

## Phase 4: Advanced Architecture (Expected: +2-3%)

**Goal: Upgrade from vanilla ST-GCN**

### 4a. CTR-GCN (Channel-wise Topology Refinement)
- Each channel learns its own adjacency topology
- ~1.4M params per stream, well-documented
- Code: github.com/Uason-Chen/CTR-GCN

### 4b. Multi-scale temporal convolution
- Replace fixed kernel=9 with parallel branches: kernel 3, 7, 15 (or dilated 1, 3, 9)
- Captures both local transitions and global repetition patterns

### 4c. Attention pooling
- Replace AdaptiveAvgPool with multi-head attention pooling (from v8)
- Weight important frames higher than static holds

---

## Phase 5: Temporal Confusion Solutions (Expected: +2-4% on numbers)

**Goal: Specifically fix 444/54/35/73 confusion**

### 5a. Temporal Self-Similarity Matrix (TSM) head
- Compute T x T cosine similarity of temporal features
- Feed through small CNN to detect repetition patterns
- 444 shows 3 similarity bands, 54 shows 2 blocks

### 5b. Velocity/motion features (covered in Phase 2a)
- Velocity makes repetition count trivially distinguishable

### 5c. ArcFace / Cosine classifier
- Angular margin loss pushes apart classes with high cosine similarity
- Replace final Linear with cosine similarity + angular margin

---

## Phase 6: Augmentation & Regularization (Expected: +2-3%)

### 6a. Signer-invariant augmentation
- Bone-length perturbation (0.85-1.15x per bone)
- Temporal warping (simulate different signing speeds)
- CropResize temporal augmentation

### 6b. Test-time augmentation (free accuracy)
- 5-crop temporal + spatial flip + small rotations
- Average predictions across augmented views
- Expected: +1-2% with zero retraining

### 6c. SAM optimizer
- Sharpness-Aware Minimization wrapping AdamW
- Finds flatter minima, better generalization on small datasets

---

## Phase 7: Pre-training & Transfer (Expected: +2-5%)

### 7a. Self-supervised pre-training (AimCLR)
- Contrastive learning on KSL skeleton data without labels
- Then fine-tune with classification loss
- Code: github.com/Levigty/AimCLR

### 7b. Transfer from NTU RGB+D
- Pre-train ST-GCN backbone on NTU-60 (56K sequences)
- Fine-tune on KSL
- Challenge: joint mapping (NTU 25 joints -> KSL 48 nodes)

### 7c. Knowledge distillation from v8
- Use v8 LSTM soft targets (T=8) to train ST-GCN
- Transfers v8's 83.65% knowledge into graph architecture

---

## Priority Matrix

| Phase | Expected Gain | Effort | Cumulative Target |
|-------|:---:|:---:|:---:|
| 1. v17 baseline | Recovery to ~78% | Done | 78% |
| 2. Multi-stream ensemble | +2-5% | Low (days) | 82-85% |
| 3. Feature-augmented ST-GCN | +3-5% | Medium (days) | 85-88% |
| 4. CTR-GCN + attention | +2-3% | Medium (week) | 87-90% |
| 5. Temporal confusion fixes | +2-4% numbers | Medium (days) | 88-92% |
| 6. Augmentation + TTA | +2-3% | Low (hours-days) | 90-93% |
| 7. Pre-training + KD | +2-5% | High (week) | 92-95% |

**Critical path to >85%: Phases 1 + 2 + 3 (v17 + multi-stream + v8 features)**

---

## Hardware: Alpine L40 GPU (46GB VRAM)

- Single model training: ~8 min (v16), ~15 min estimated (v17 with 300 epochs)
- 4-stream ensemble: 4 x 15 min = ~1 hour
- Dual-branch hybrid: ~30 min (8.5M params)
- All phases fit comfortably in 2-hour SLURM jobs
