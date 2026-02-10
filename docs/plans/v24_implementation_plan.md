# V24 Implementation Plan: Prototypical Classification + User Enrollment

**Date:** 2026-02-10
**Status:** APPROVED
**Based on:** 4 research documents (transfer learning, synthetic augmentation, few-shot enrollment, feasibility assessment)
**Baseline:** V22 (best real-world: 39.8% combined, 93.3% val, 776K params)

---

## Executive Summary

V24 replaces the FC classifier with **prototypical (nearest-centroid) classification**, adds **LOSO cross-validation** for honest evaluation, introduces **MixStyle + bone retargeting augmentation** for signer diversity, and enables **user enrollment** at deployment time. The v22 ST-GCN backbone remains unchanged.

**Target:** >45% real-world without enrollment, >55% with 3-shot enrollment
**Approach:** Disciplined, low-risk iteration. No architectural ambition (v23 proved that fails).

---

## 1. Design Principles (Lessons from v19-v23)

| Principle | Justification |
|-----------|---------------|
| Keep v22 backbone unchanged | V22 (776K) beat v23 (3.1M). Dataset cannot support larger models. |
| Fix the classification HEAD, not the feature extractor | The 320-dim embedding is adequate; the FC boundary collapses on OOD data. |
| Evaluate with LOSO, not fixed train/val split | V23: 95.6% val, 37.9% real. Val is a misleading metric. |
| Single model, <15 min per fold | No ensembles. 4-fold LOSO must fit in 1-hour SLURM slot. |
| Augmentation > architecture | Simple augmentation that increases signer diversity beats complex DG algorithms (confirmed by 2024 survey). |

---

## 2. Architecture: KSLGraphNetV24

### 2.1 What Changes from V22

| Component | V22 | V24 | Rationale |
|-----------|-----|-----|-----------|
| **Classifier** | FC: Linear(320->64->nc) | Prototypical: cosine similarity to class prototypes | FC boundaries collapse on OOD; prototypes adapt via enrollment |
| **Confidence** | Softmax probability | Distance margin (gap between top-1 and top-2 prototype distances) | Softmax confidence doesn't transfer to OOD (v22: HIGH numbers = 18% acc) |
| **Validation** | Fixed train/val split | LOSO (leave-one-signer-out, 4 folds) | Realistic cross-signer estimate |
| **Augmentation** | 12 types, moderate | + MixStyle (feature-level) + bone retargeting (input-level) | Synthesize virtual signer diversity |
| **Enrollment** | None | Prototype update from 3-5 user samples per class | Adapts to new signer at deployment |
| **BN at enrollment** | N/A | Update BN running stats from user samples | Adapts feature normalization to new signer |

### 2.2 What Stays the Same

- ST-GCN backbone: 4 layers, channels [9, 64, 64, 128, 128], MultiScaleTCN (kernels 3,5,7)
- Attention pooling: 2-head temporal attention -> 256-dim GCN embedding
- Auxiliary branch: 53 features (33 angles + 20 fingertip distances) -> 64-dim
- Combined embedding: 320-dim (256 + 64)
- GRL signer head: gradient reversal, lambda 0->0.3 over 100 epochs
- SupCon loss: weight 0.1, temperature 0.07
- Graph: 48 nodes (21 LH + 21 RH + 6 pose)
- Normalization: wrist-centric + palm-size (hands), mid-shoulder + shoulder-width (pose)
- Input features: 9 channels (xyz + velocity + bone)

### 2.3 Model Diagram

```
Input: skeleton (B, 9, 90, 48) + aux (B, 90, 53)
  |
  v
[ST-GCN Block 1] -- channels: 9 -> 64
  |-> NEW: MixStyle (mix feature stats across signers, p=0.5)
  v
[ST-GCN Block 2] -- channels: 64 -> 64
  |-> NEW: MixStyle (mix feature stats across signers, p=0.3)
  v
[ST-GCN Block 3] -- channels: 64 -> 128
  v
[ST-GCN Block 4] -- channels: 128 -> 128
  v
[Attention Pool] -> 256-dim     [Aux Branch] -> 64-dim
  |                                 |
  +------------ concat -------------+
  |
  v
embedding: 320-dim
  |
  +---> NEW: Cosine similarity to prototypes (nc x 320) / temperature
  |         -> classification logits
  |
  +---> GRL -> signer classifier (unchanged)
  |
  +---> SupCon loss (unchanged)
```

**Estimated parameters:** ~760K (slightly less than v22 due to removed FC head)

---

## 3. Implementation Steps

### Step 1: Copy V22 as V24 Base
- Create `train_ksl_v24.py` from `train_ksl_v22.py`
- Create `evaluate_real_testers_v24.py` from v22 eval script
- Create `slurm/v24.sh` and `slurm/eval_v24.sh`

### Step 2: Add MixStyle Module

Insert after ST-GCN blocks 1 and 2. Zero parameters, zero overhead.

```python
class MixStyle(nn.Module):
    """Mix feature statistics across samples (Zhou et al., ICLR 2021)."""
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p          # probability of applying
        self.alpha = alpha  # Beta distribution parameter

    def forward(self, x, signer_ids=None):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)    # per-channel mean
        sigma = x.std(dim=[2, 3], keepdim=True)   # per-channel std
        x_normed = (x - mu) / (sigma + 1e-6)

        # Shuffle to get mixing partners (prefer cross-signer if labels available)
        perm = torch.randperm(B)
        mu2, sigma2 = mu[perm], sigma[perm]

        # Mix statistics
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sigma_mix = lam * sigma + (1 - lam) * sigma2

        return x_normed * sigma_mix + mu_mix
```

### Step 3: Add Bone-Length Retargeting Augmentation

Insert in `__getitem__` after normalization, before feature computation.

```python
def bone_retarget(coords, signer_bone_stats):
    """Retarget skeleton to random virtual signer morphology."""
    # coords: (T, N, 3), signer_bone_stats: dict with mean/std per bone
    # Sample random bone scale factors from fitted distribution
    scale = np.random.normal(1.0, 0.15, size=(N_BONES,))  # ~15% variation
    scale = np.clip(scale, 0.7, 1.3)

    # Apply per-bone scaling to bone vectors
    for bone_idx, (child, parent) in enumerate(BONE_PAIRS):
        bone_vec = coords[:, child] - coords[:, parent]
        coords[:, child] = coords[:, parent] + bone_vec * scale[bone_idx]
    return coords
```

### Step 4: Replace FC Classifier with Prototypical Head

```python
class PrototypicalHead(nn.Module):
    def __init__(self, embed_dim, num_classes, temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # Prototypes initialized randomly, updated after training
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.prototypes.requires_grad = False  # Updated externally, not by gradient

    def forward(self, embedding):
        # L2-normalize both for cosine similarity
        emb_norm = F.normalize(embedding, dim=1)          # (B, D)
        proto_norm = F.normalize(self.prototypes, dim=1)   # (C, D)
        logits = torch.mm(emb_norm, proto_norm.t()) / self.temperature  # (B, C)
        return logits

    def update_prototypes(self, embeddings_per_class):
        """Update prototypes from mean embeddings per class."""
        for cls_idx, embs in embeddings_per_class.items():
            self.prototypes.data[cls_idx] = embs.mean(dim=0)
```

**Training approach:** During training, prototypes are computed fresh each epoch from the training set embeddings (no gradient through prototypes). The backbone is trained with cross-entropy on the cosine similarity logits. After training, final prototypes are stored.

### Step 5: LOSO Cross-Validation

```python
def get_loso_splits(data_dir, num_signers=4):
    """Generate leave-one-signer-out splits."""
    all_files = glob(f"{data_dir}/**/*.npy")
    signer_ids = sorted(set(extract_signer(f) for f in all_files))

    for held_out in signer_ids:
        train_files = [f for f in all_files if extract_signer(f) != held_out]
        val_files = [f for f in all_files if extract_signer(f) == held_out]
        yield held_out, train_files, val_files
```

**Training protocol:**
1. Run 4 folds (each holds out 1 signer)
2. Report average LOSO accuracy as primary development metric
3. Select hyperparameters that maximize average LOSO accuracy
4. For final model: train on ALL 4 signers, use LOSO accuracy as expected real-tester performance
5. Evaluate final model on real testers (ground truth)

### Step 6: Distance-Based Confidence

Replace softmax confidence with prototype distance margin:

```python
def compute_confidence(logits, prototypes, embedding):
    """Distance-based confidence: margin between top-1 and top-2 prototype distances."""
    emb_norm = F.normalize(embedding, dim=1)
    proto_norm = F.normalize(prototypes, dim=1)
    cosine_sims = torch.mm(emb_norm, proto_norm.t())  # (B, C)

    top2 = cosine_sims.topk(2, dim=1).values  # (B, 2)
    margin = top2[:, 0] - top2[:, 1]           # higher = more confident

    # Normalize to [0, 1]
    confidence = torch.sigmoid(margin * 5.0)  # calibrate with scale factor
    return confidence
```

### Step 7: User Enrollment Pipeline

At deployment time, adapt to a new signer:

```python
def enroll_user(model, user_samples_per_class, alpha=0.5):
    """
    Adapt model to new signer using calibration samples.

    Args:
        model: trained KSLGraphNetV24
        user_samples_per_class: dict {class_idx: list of (gcn, aux) tensors}
        alpha: blending weight (0=all training, 1=all user)
    """
    model.eval()

    # Step 1: BN stats adaptation (unlabeled, updates running mean/var)
    model.train()  # enable BN stat tracking
    with torch.no_grad():
        for cls_samples in user_samples_per_class.values():
            for gcn, aux in cls_samples:
                _ = model(gcn.unsqueeze(0), aux.unsqueeze(0), grl_lambda=0.0)
    model.eval()

    # Step 2: Compute user prototypes
    user_prototypes = {}
    with torch.no_grad():
        for cls_idx, samples in user_samples_per_class.items():
            embeddings = []
            for gcn, aux in samples:
                emb = model.get_embedding(gcn.unsqueeze(0), aux.unsqueeze(0))
                embeddings.append(emb)
            user_prototypes[cls_idx] = torch.cat(embeddings, dim=0).mean(dim=0)

    # Step 3: Blend with training prototypes
    for cls_idx, user_proto in user_prototypes.items():
        n_user = len(user_samples_per_class[cls_idx])
        n_prior = 5  # effective prior strength
        blend_alpha = n_user / (n_user + n_prior)
        model.proto_head.prototypes.data[cls_idx] = (
            blend_alpha * user_proto + (1 - blend_alpha) * model.proto_head.prototypes.data[cls_idx]
        )
```

### Step 8: Enhanced Signer Augmentation Config

```python
V24_CONFIG = {
    # ... keep all v22 config values ...

    # NEW: Wider signer augmentation ranges
    'bone_scale_range': (0.7, 1.3),        # was (0.8, 1.2) in v22
    'hand_size_range': (0.7, 1.3),          # was (0.8, 1.2) in v22
    'speed_warp_range': (0.8, 1.2),         # NEW: temporal speed perturbation
    'amplitude_scale_range': (0.85, 1.15),  # NEW: spatial amplitude variation

    # NEW: MixStyle
    'mixstyle_p': 0.5,         # probability of applying MixStyle
    'mixstyle_alpha': 0.1,     # Beta distribution parameter
    'mixstyle_layers': [0, 1], # apply after ST-GCN blocks 0 and 1

    # NEW: Bone retargeting
    'bone_retarget_p': 0.3,    # probability of applying bone retargeting
    'bone_retarget_std': 0.15, # std of bone scale factor

    # NEW: Prototypical classification
    'proto_temperature': 0.1,  # learnable temperature initial value
    'proto_update_freq': 5,    # recompute prototypes every N epochs
}
```

---

## 4. Training Protocol

### 4.1 LOSO Development (Hyperparameter Tuning)

```
For each fold k in {1, 2, 3, 4}:
    Train on signers {1,2,3,4} \ {k}
    Validate on signer {k}
    Record: per-class accuracy, LOSO accuracy, prototype quality

Average LOSO accuracy = primary metric for hyperparameter selection
```

**Expected LOSO accuracy:** 35-45% (much lower than fixed val, much closer to real-tester)

### 4.2 Final Model Training

```
Train on ALL 4 signers (full training set)
Compute prototypes from ALL training embeddings
Evaluate on 140 real-tester samples (3 unseen signers)
Report: accuracy, confidence calibration, per-class breakdown
```

### 4.3 Enrollment Evaluation

```
For each real tester:
    Split their samples: 3 per class for enrollment, rest for testing
    Run enrollment (BN adapt + prototype update)
    Evaluate on held-out test samples
    Report: accuracy with enrollment vs. without
```

---

## 5. What NOT to Do

Based on v23 postmortem and research findings:

| Approach | Why to Avoid |
|----------|-------------|
| Multi-stream ensemble | Failed in v23; streams correlated on OOD; 4x cost for negative returns |
| Post-hoc temperature calibration | Failed in v23; val calibration doesn't transfer to OOD signers |
| Focal loss | No measurable benefit in v23; sink classes are OOD symptom, not class imbalance |
| Test-time augmentation (TTA) | Consistently hurts (-1.9% in v22); disabled permanently |
| Model size > 1M params | V23 (3.1M) was worse than v22 (776K); dataset too small |
| MAML meta-learning | Needs 64+ tasks; we have 4 signers -- fundamentally insufficient |
| IRM (Invariant Risk Minimization) | Needs >5 environments; impractical with 4 signers |
| CVAE signer generation | High effort, high variance with 4 source signers; likely mode collapse |
| Optimizing val accuracy | V23: 95.6% val = 37.9% real. Use LOSO instead. |

---

## 6. Expected Outcomes

### 6.1 Without Enrollment (Base Model)

| Metric | V22 Baseline | V24 Target | Confidence |
|--------|-------------|------------|------------|
| Real-world combined | 39.8% | 42-48% | Medium |
| Real-world numbers | 33.9% | 38-44% | Medium |
| Real-world words | 45.7% | 48-54% | Medium |
| LOSO accuracy | N/A (new metric) | 35-45% | High |
| Parameters | 776K | ~760K | High |
| Training time (4-fold) | 7.3 min | ~35 min | High |

### 6.2 With 3-Shot Enrollment

| Metric | No Enrollment | 3-Shot Enrollment | 5-Shot Enrollment |
|--------|--------------|-------------------|-------------------|
| Real-world combined | 42-48% | 50-60% | 55-65% |
| Real-world numbers | 38-44% | 45-55% | 50-60% |
| Real-world words | 48-54% | 55-65% | 60-70% |

### 6.3 Honest Assessment

- **Without enrollment:** V24 may gain only 2-5pp over v22. The 4-signer ceiling is real.
- **With enrollment:** This is where the real gains come from. 3-5 calibration samples per class allow the model to adapt its decision boundaries to the new signer.
- **Collecting more signers remains the only reliable path to >70%.** V24 buys time with enrollment while more data is collected.

---

## 7. Decision Points After V24

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| V24 base > 43% | Prototypical approach working | Add episodic training (v25) |
| V24 base ~ 40% | 4-signer ceiling confirmed | Focus on enrollment UX + data collection |
| V24 base < 38% | Prototypical hurt base accuracy | Revert to v22, deploy with enrollment only |
| V24 + enrollment > 55% | Enrollment is the winning strategy | Invest in enrollment UX, progressive learning |
| V24 + enrollment < 45% | Enrollment insufficient | Collect more signers (no alternative) |

---

## 8. File Plan

| File | Action | Description |
|------|--------|-------------|
| `train_ksl_v24.py` | CREATE | Copy from v22, add prototypical head, MixStyle, bone retargeting, LOSO |
| `evaluate_real_testers_v24.py` | CREATE | Prototype-based inference, distance confidence, enrollment simulation |
| `slurm/v24.sh` | CREATE | SLURM job for 4-fold LOSO training |
| `slurm/eval_v24.sh` | CREATE | SLURM job for real-tester evaluation |
| `docs/reports/v24_report.md` | CREATE (after results) | Full v24 analysis report |

---

## 9. Implementation Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|---------------|
| **Phase 1: Core** | Copy v22, add prototypical head, LOSO splits | 2-3 hours |
| **Phase 2: Augmentation** | MixStyle module, bone retargeting, speed/amplitude jitter | 2-3 hours |
| **Phase 3: Training** | Run LOSO experiments, tune hyperparameters | 2-3 hours (compute) |
| **Phase 4: Enrollment** | Implement enrollment pipeline, evaluate with/without | 2-3 hours |
| **Phase 5: Evaluation** | Real-tester eval, analysis report | 1-2 hours |
| **Total** | | ~10-14 hours |

---

## 10. Key References

1. Snell et al. "Prototypical Networks for Few-shot Learning" NeurIPS 2017
2. Zhou et al. "Domain Generalization with MixStyle" ICLR 2021
3. Saad "Data-Efficient ASL Recognition via Few-Shot Prototypical Networks" arXiv 2512.10562 -- **13pp gain from prototypical on WLASL**
4. Wang et al. "Tent: Fully Test-time Adaptation by Entropy Minimization" ICLR 2021
5. Wu et al. "Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization" AAAI 2024
6. Fu et al. "Signer Diversity-driven Data Augmentation" NAACL 2024
7. Zang "TinyML-Driven On-Device Sports Command Recognition" 2025 -- last-layer FT with 15 examples/class
8. Bilge et al. "Cross-lingual few-shot sign language recognition" Pattern Recognition 2024

---

## Appendix: V24 Config Diff from V22

```diff
  # Classification
- 'classifier': 'fc'           # Linear(320->64->nc)
+ 'classifier': 'prototypical'  # Cosine similarity to prototypes
+ 'proto_temperature': 0.1
+ 'proto_update_freq': 5        # Recompute prototypes every 5 epochs

  # Validation
- 'val_split': 'fixed'         # train_v2/val_v2 directories
+ 'val_split': 'loso'          # Leave-one-signer-out, 4 folds

  # Augmentation (additions)
+ 'mixstyle_p': 0.5
+ 'mixstyle_alpha': 0.1
+ 'mixstyle_layers': [0, 1]
+ 'bone_retarget_p': 0.3
+ 'bone_retarget_std': 0.15
+ 'speed_warp_range': (0.8, 1.2)
+ 'amplitude_scale_range': (0.85, 1.15)

  # Augmentation (widened ranges)
- 'bone_scale_range': (0.8, 1.2)
+ 'bone_scale_range': (0.7, 1.3)
- 'hand_size_range': (0.8, 1.2)
+ 'hand_size_range': (0.7, 1.3)

  # Confidence
- 'confidence': 'softmax'
+ 'confidence': 'distance_margin'

  # Enrollment (new)
+ 'enrollment_bn_adapt': True
+ 'enrollment_proto_blend': True
+ 'enrollment_prior_strength': 5
```
