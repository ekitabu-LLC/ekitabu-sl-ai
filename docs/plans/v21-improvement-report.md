# V21 Comprehensive Improvement Report: Kenya Sign Language Recognition

**Date**: 2026-02-09
**Authors**: Compiled from research by signer-generalization-researcher, augmentation-researcher, architecture-researcher, report-writer
**Status**: Definitive guide for V21 development
**Target**: Close the 53-point gap between validation (87%) and real-tester (34%) accuracy

---

## 1. Executive Summary

### What Went Wrong in V16-V20

Across five model iterations (v16-v20), the KSL recognition system achieved progressively higher validation accuracy -- culminating in 87.3% (v19) -- while real-tester accuracy remained stuck at 30-34%. The fundamental problem is **training data poverty**: with only 4 unique signers (not 5 -- signers 2 and 3 are byte-identical duplicates), the model memorizes signer-specific artifacts (absolute hand position, arm length, body proportions) rather than learning generalizable sign representations. Each version introduced valuable improvements (bone features in v19, model shrinkage in v20), but also new problems: v18's SWA degraded performance, v19's 4.5M parameters enabled extreme overfitting, and v20's aggressive deduplication (750 to 240 samples, removing 68%) starved the model of training signal.

### The Fundamental Problem

With 4 unique training signers, the true Leave-One-Signer-Out (LOSO) cross-signer accuracy is 37.8% -- almost exactly what real testers achieve (34.2%). This means the model is performing at its expected capacity given the data. The AUTSL benchmark (43 signers) shows a 34-point signer-independent drop; with only 4 signers, our 53-point drop is within the expected range. **No architectural change alone will solve this; the problem is fundamentally about signer diversity.**

### What the Research Found

Three parallel research efforts surveyed 50+ papers across cross-signer generalization, data augmentation, and model architectures. Key findings:

1. **Signer-invariant features** (bone vectors, joint angles, fingertip distances) provide +10-15% cross-signer accuracy by eliminating position/scale dependence (Aiman & Ahmad 2023; SAM-SLR, CVPR 2021).
2. **Supervised contrastive learning** directly optimizes for signer invariance by pulling same-sign embeddings together across signers -- potentially the single most impactful training technique (SC2SLR, 2024; SupCon, Khosla et al.).
3. **Aggressive skeleton augmentation** (shear, joint dropout, wide rotation/speed ranges, domain randomization) can simulate 3-4x more signer diversity than current mild augmentations (AimCLR, AAAI 2022).
4. **Multi-stream ensembles** (joint + bone + velocity + bone-velocity) provide +2-5% as a near-free improvement and are the de facto standard in skeleton-based recognition.
5. **Prototypical/metric learning** substantially outperforms standard classifiers in low-data regimes (+13% on WLASL, arXiv:2512.10562).

### Recommended V21 Approach

V21 should implement a **multi-stream lightweight GCN** (0.3-0.5M params) with **signer-invariant input features** (bones, angles, velocities), trained using **supervised contrastive loss + ArcFace classification** with **aggressive domain-randomization augmentation** that simulates anatomical and behavioral variation across signers. This combination addresses all identified root causes simultaneously: feature-level invariance eliminates signer-dependent representations, contrastive learning explicitly optimizes for cross-signer consistency, augmentation expands effective signer diversity, and model shrinkage prevents memorization.

---

## 2. Version History & Lessons Learned

### Version Progression

| Version | Architecture | Val Accuracy | Real-Tester Accuracy | Params | Key Change | Key Lesson |
|---------|-------------|-------------|---------------------|--------|------------|------------|
| **v16** | ST-GCN baseline | -- | -- | -- | Initial implementation | Baseline established |
| **v17** | Larger ST-GCN | 72.6% combined | -- | Large | Larger model, simple CE loss | Bigger model != better generalization |
| **v18** | ST-GCN + SWA | 70.6% combined (53.6% numbers) | -- | 7.3M | Added SWA, found confusable pairs | **SWA consistently hurts** on small datasets; discovered class 35/22 sim=0.999 |
| **v19** | ST-GCN + bones | 87.3% (78.6% num, 95.9% words) | **34.2%** (20.3% num, 48.1% words) | 4.5M | Bone features, anti-overfitting | **BEST real-world result**; 4.5M params still too many; bone features help validation but not enough for generalization |
| **v20** | Lightweight ST-GCN | 86.9% (82.2% num, 91.6% words) | 30.5% (20.3% num, 40.7% words) | 413K | All P0-P2 fixes from gap report, aggressive dedup | **Dedup too aggressive** (750->240 removed 68%); wrist-centric norm changed patterns but didn't help; smaller model is directionally correct |

### Cumulative Lessons

1. **SWA hurts** on datasets this small -- disable it permanently.
2. **Aggressive deduplication is dangerous**: v20's 750->240 reduction removed legitimate variation. Light dedup only (remove byte-identical copies, keep ~600-650 samples).
3. **Model capacity must match data**: 375 samples cannot support 4.5M+ params. Target <500K.
4. **Bone features help validation but aren't sufficient** for cross-signer transfer -- they need to be combined with contrastive learning and augmentation.
5. **Validation accuracy is misleading**: 87% validation vs 34% real-tester means validation is not predictive of real-world performance. LOSO is the honest metric.
6. **Too many changes at once** create a whack-a-mole problem: fixing one class breaks another. Changes should be evaluated incrementally.
7. **Confidence filtering works**: at >0.6 confidence, v20 achieved 81% accuracy. The model "knows what it doesn't know" to some degree.
8. **Words are inherently easier than numbers**: Words involve more body motion and distinctive gestures; numbers involve subtle hand-shape differences that are harder to distinguish across signers.

---

## 3. Literature Review Synthesis

### 3.1 Cross-Signer Generalization

The cross-signer accuracy gap is a well-documented challenge. Benchmarks show consistent patterns:

| Dataset | Signers | Signer-Dependent | Signer-Independent | Gap |
|---------|---------|-----------------|-------------------|-----|
| AUTSL (226 signs) | 43 | 95.95% | 62.02% | -34% |
| Ethiopian SL (skeleton) | 5+2 | 94% | 73% | -21% |
| Arabic SL (23 words) | 3 users | 95%+ | 89.5% | -6% |
| **KSL (ours)** | **4** | **87%** | **34%** | **-53%** |

**Key insight**: Gap size scales inversely with signer count. AUTSL's 34-point gap with 43 signers extrapolates to a 50-60 point gap with 4 signers, which is exactly what we observe (Sincan & Keles 2020).

The literature identifies three tiers of solutions by effectiveness:

**Tier 1 -- Feature-level invariance** (highest impact): Bone features are position-invariant by construction; joint angles are scale-invariant; velocity features normalize out static pose differences. The SAM-SLR challenge winner (98.42% on signer-independent AUTSL) used 4-stream ensembles (joint/bone/velocity/bone-velocity) as its core technique (Jiang et al., CVPR 2021W).

**Tier 2 -- Training objectives**: Supervised contrastive learning (SC2SLR, CNIOT 2024) and quaternion-based contrastive learning (arXiv:2508.14574, 2025) explicitly optimize for signer invariance, achieving 16% improvement in pose keypoint accuracy from contrastive loss alone. Prototypical networks outperform standard classifiers by 13% on WLASL skeleton data (arXiv:2512.10562, 2025).

**Tier 3 -- Domain adaptation**: Signer-adversarial training via gradient reversal layers (Ganin et al. 2016) forces signer-invariant features, but with only 4 domains may be unstable. More effective with 10+ signers.

### 3.2 Data Augmentation for Small Skeleton Datasets

The AimCLR framework (Guo et al., AAAI 2022) established that "extreme" augmentation -- shear, axis masking, large rotations, joint dropout -- substantially improves cross-subject generalization in skeleton-based recognition, even without contrastive learning.

**Gap analysis for our v20 augmentations**:
| Augmentation | V20 Setting | Recommended | Status |
|-------------|-------------|-------------|--------|
| Rotation | +/-5 deg | +/-30 deg | **6x too narrow** |
| Speed perturbation | 0.9-1.1x | 0.5-2.0x | **10x too narrow** |
| Spatial jitter | sigma=0.01 | sigma=0.03-0.05 | **3-5x too small** |
| Shear | Not used | s ~ U(-0.5, 0.5) | **Missing (HIGH value)** |
| Joint dropout | Not used | 15-25% of joints | **Missing (HIGH value)** |
| Axis masking | Not used | p=0.1 per axis | **Missing (MEDIUM value)** |
| SkeletonMix | Not used | Hand-region swapping | **Missing (HIGH value)** |

Generative models (GANs, VAEs, diffusion) are **not practical** for our dataset size (<1000 samples). Instead, rule-based domain randomization -- randomizing anatomical proportions (arm length 0.8-1.2x, hand size 0.7-1.3x), behavioral properties (signing amplitude 0.7-1.5x, speed 0.5-2.0x), and sensor noise -- is more reliable and creates meaningful variation.

**Bone-space style transfer** is a practical synthetic data approach: keep bone directions from one signer but swap bone lengths from another, creating "synthetic signers" that preserve motion semantics while varying body proportions. With 4 signers, this creates 12 pairwise style-transferred versions per sample, effectively 4x the signer diversity.

### 3.3 Model Architectures and Ensembles

For datasets under 1000 samples, the literature strongly recommends:

**Lightweight GCNs**: EfficientGCN-B0 (0.29M params) achieves competitive accuracy on NTU-60/120 (Song et al. 2022). InfoGCN (Chi et al., CVPR 2022) uses an information bottleneck objective that forces compact, informative representations -- directly addressing our overfitting problem. BlockGCN (Zhou et al., CVPR 2024) preserves topological information that standard GCN training loses.

**Multi-stream ensembles**: The 4-stream approach (joint + bone + joint-velocity + bone-velocity) is the de facto standard. Late fusion via weighted sum of softmax logits provides +2-5% over any single stream. Angular features as a 5th stream provide additional +1-2% for hand gesture differentiation (Huang et al., 2024).

**Sign-language-specific architectures**: HA-GCN (Hand-Aware GCN) explicitly models hand topological relationships and achieves state-of-the-art with hand-aware graph topology. AHG-GCN (Aiman & Ahmad 2023) uses 25 angle-based features per joint and achieves 90% on DHG-14 for hand gesture recognition.

**Angular margin losses**: ArcFace (Deng et al., CVPR 2019) forces larger angular separation between classes -- directly addressing our confusable pairs problem (classes 35/22 at cosine similarity 0.999).

---

## 4. Root Cause Analysis Update

The original gap report identified 7 root causes. Research has refined and reprioritized them:

### 4.1 Primary Root Causes (Account for ~80% of the Gap)

**RC1: Only 4 unique training signers (CRITICAL)**
- True LOSO accuracy: 37.8% (excluding duplicate signer pair)
- Per-signer LOSO ranges from 16.0% (most dissimilar signer) to 68.7%
- Literature confirms: with 4 signers, a 50+ point gap is expected
- **Research update**: Even with perfect feature engineering, 4 signers cap cross-signer accuracy around 55-65%. Only more data (5-10 additional signers) can push past 70%.

**RC2: Raw XYZ features encode signer identity (CRITICAL)**
- Confusable pairs: class 35/22 cosine similarity 0.999, class 388/48 similarity 0.9998
- 8 of 15 number classes at 0% on real testers
- Class 35 at 0% across ALL versions (v16-v18)
- **Research update**: Bone features + angular features + contrastive learning can jointly address this. Angular features (wrist-to-fingertip angles) are inherently signer-invariant and proven effective for hand gesture differentiation (AHG-GCN, 90% on DHG-14).

**RC3: Prediction collapse to dominant classes (HIGH)**
- Class 444 predicted 21/59 times for numbers (35.6%)
- Model has learned degenerate decision boundary
- **Research update**: ArcFace loss with angular margin prevents collapse by enforcing minimum separation between class prototypes. Center loss adds intra-class compactness.

### 4.2 Secondary Root Causes (Account for ~15% of the Gap)

**RC4: Inadequate augmentation (HIGH)**
- Current augmentation ranges 3-10x too narrow
- Missing high-value augmentations (shear, joint dropout, axis masking)
- **Research update**: AimCLR's extreme augmentation framework is directly applicable. Domain randomization of anatomical proportions creates synthetic signer diversity.

**RC5: Model over-parameterization (HIGH)**
- v19: 4.5M params / 600 samples = 7,500 params/sample (literature recommends <100:1)
- v20 corrected to 413K params -- a good direction
- **Research update**: EfficientGCN-B0 at 0.29M params is the optimal size for 600 samples.

**RC6: Side-view video in test set (MEDIUM)**
- Signer 3's numbers all recorded from side angle
- Training data exclusively front-facing
- **Research update**: Viewpoint augmentation (rotation +/-30 deg around Y axis) can partially address this.

### 4.3 Minor Root Causes (Account for ~5% of the Gap)

**RC7: MediaPipe version mismatch** (training 0.10.14 vs eval 0.10.5)
**RC8: Frame count distribution mismatch** (training 87-127 vs real 64-188 frames)
**RC9: Per-sample normalization amplifies signer noise**

### 4.4 Confidence Filtering as Mitigation

A critical finding from v19/v20 testing: **confidence filtering works well**.
- Words at >0.6 confidence: 86.7% accurate
- Overall at >0.6 confidence: 81% accurate
- This means the model's uncertainty estimates are informative even when overall accuracy is low
- **Deployment strategy**: Use confidence thresholds to reject uncertain predictions and request re-signing

---

## 5. Recommended V21 Approach

### Priority-Ordered Technique Stack

Each technique is rated on Expected Impact, Implementation Complexity, and Risk.

#### P0: Signer-Invariant Multi-Stream Features (Expected: +10-15% cross-signer)

**Expected Impact**: HIGH (+10-15%) | **Complexity**: MEDIUM | **Risk**: LOW

Replace raw XYZ input with a multi-stream feature pipeline:

| Stream | Features | Dimension/Frame | Signer Invariance |
|--------|----------|-----------------|-------------------|
| **Bone vectors** | child - parent joint XYZ | 23 bones x 3 = 69/hand | Position-invariant |
| **Joint angles** | acos(dot(bone_a, bone_b)) | ~20 angles/hand | Position + scale invariant |
| **Fingertip distances** | pairwise L2 between 5 fingertips | 10 pairs/hand | Captures hand shape |
| **Velocities** | Frame-to-frame differences of bones | 69/hand | Motion dynamics |

**Normalization pipeline**:
```
Step 1: Center on body midpoint (mid-shoulder)
Step 2: Scale by shoulder width (normalize body size)
Step 3: For hands: center on wrist, scale by palm size (wrist-to-MCP9 distance)
Step 4: Compute bone vectors (child - parent joint)
Step 5: Compute joint angles (acos of dot product between adjacent bones)
Step 6: Compute fingertip pairwise distances
Step 7: Compute velocities (frame-to-frame bone differences)
```

**Citations**: SAM-SLR (Jiang et al. CVPR 2021W) used 4-stream ensemble to win signer-independent challenge at 98.42%. AHG-GCN (Aiman & Ahmad 2023) achieved 90% on DHG-14 using angle-based features. Dual-Stream ST-DGCN (2025, arXiv:2509.08661) demonstrated dual-reference normalization for signer invariance.

**Implementation details**:
```python
# Bone connections for MediaPipe hand (21 landmarks)
HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17)           # Palm cross-connections
]

def compute_multi_stream_features(hand_landmarks_21x3):
    # Bone vectors (position-invariant)
    bones = []
    for parent, child in HAND_BONES:
        bones.append(hand_landmarks_21x3[child] - hand_landmarks_21x3[parent])
    bones = np.array(bones)  # (23, 3)

    # Joint angles (scale-invariant)
    angles = []
    for i, (p1, c1) in enumerate(HAND_BONES):
        for j, (p2, c2) in enumerate(HAND_BONES):
            if c1 == p2:  # Connected bones
                v1 = bones[i] / (np.linalg.norm(bones[i]) + 1e-8)
                v2 = bones[j] / (np.linalg.norm(bones[j]) + 1e-8)
                angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                angles.append(angle)

    # Fingertip distances (hand shape descriptor)
    tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
    distances = []
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            d = np.linalg.norm(hand_landmarks_21x3[tips[i]] - hand_landmarks_21x3[tips[j]])
            distances.append(d)

    return bones.flatten(), np.array(angles), np.array(distances)
```

---

#### P1: Supervised Contrastive Learning (Expected: +5-10% cross-signer)

**Expected Impact**: HIGH (+5-10%) | **Complexity**: MEDIUM | **Risk**: LOW-MEDIUM

Supervised contrastive loss directly optimizes for the property we need: same sign by different signers should produce similar embeddings.

**Why this is potentially the most impactful training technique**:
1. Positive pairs = same sign class, ANY signer -- explicitly teaches signer invariance
2. Naturally handles confusable pairs by pushing different-class embeddings apart
3. Produces better-separated embeddings than cross-entropy alone
4. Works with very few signers (only needs 2+ signers per class for positive pairs)
5. Quaternion-based contrastive learning achieved 16% improvement in pose keypoint accuracy (arXiv:2508.14574)

**Training strategy**: Two-stage approach:
- Stage 1 (epochs 1-100): SupCon pretraining with temperature tau=0.07
- Stage 2 (epochs 101-200): Fine-tune with CE + SupCon (weight 0.3) + ArcFace

**Implementation sketch**:
```python
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels), device=labels.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob.mean()
```

**Citations**: SC2SLR (ACM CNIOT 2024), SL-SLR (arXiv:2509.05188, 2025), SupCon (Khosla et al. NeurIPS 2020), Quaternion contrastive (arXiv:2508.14574, 2025).

---

#### P2: Aggressive Domain-Randomization Augmentation (Expected: +5-10% cross-signer)

**Expected Impact**: HIGH (+5-10%) | **Complexity**: LOW | **Risk**: LOW

Massively increase augmentation ranges and add missing high-value augmentations.

**V21 augmentation pipeline** (applied per training sample):
```
1. Random rotation: uniform(-30, +30) degrees on Y axis, +/-15 on X/Z
2. Random shear: uniform(-0.5, +0.5) on 2 random axes
3. Random scaling: uniform(0.8, 1.2) global + uniform(0.9, 1.1) per-limb
4. Joint dropout: mask 15% of joints (set to 0)
5. Axis mask: 10% chance of zeroing X, Y, or Z axis
6. Gaussian jitter: N(0, 0.03) per coordinate
7. Speed perturbation: uniform(0.5, 2.0) via interpolation
8. Temporal crop: keep 60-100% of frames
9. Bone length perturbation: +/-15%
10. Hand size randomization: scale hands by 0.7-1.3x
11. Signing space translation: shift +/-0.1 units vertically
12. SkeletonMix: p=0.3, swap hand regions between same-class samples
```

**Progressive augmentation curriculum**:
- Epochs 1-30: Mild augmentation (50% of ranges above)
- Epochs 31-100: Full augmentation (100% of ranges)
- Epochs 101+: Extreme augmentation (120% of ranges) with hard negative mining

**Domain randomization for synthetic signer diversity**:
| Property | Randomization Range | Purpose |
|----------|-------------------|---------|
| Arm length | 0.8-1.2x | Body proportion variation |
| Hand size | 0.7-1.3x | Hand proportion variation |
| Shoulder width | 0.85-1.15x | Body frame variation |
| Finger proportions | 0.8-1.2x per finger | Fine hand shape variation |
| Signing speed | 0.5-2.0x | Temporal style variation |
| Signing amplitude | 0.7-1.5x | Signing space variation |
| Camera angle (Y) | +/-30 deg | Viewpoint variation |

**Critical**: Undo v20's aggressive deduplication. Use all ~750 training samples or light dedup to ~650 (removing only byte-identical copies).

**Citations**: AimCLR (Guo et al., AAAI 2022), SkeletonMix (IEEE 2024), JMDA (ACM TOMM 2024), SynthDa (NVIDIA 2025).

---

#### P3: ArcFace Loss + Center Loss (Expected: +3-5% on confusable pairs)

**Expected Impact**: MEDIUM (+3-5%) | **Complexity**: LOW | **Risk**: LOW

Replace cross-entropy with ArcFace angular margin loss, which forces larger angular separation between classes in the embedding space. Directly addresses our confusable pairs problem (classes 35/22 sim=0.999).

**Configuration**: s=30 (scale), m=0.3 (margin -- lower than face recognition's 0.5 since we have fewer classes). Add center loss with lambda=0.01 for intra-class compactness.

```python
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        target_logits = torch.cos(theta + self.m * one_hot)
        return self.s * target_logits
```

**Citations**: ArcFace (Deng et al., CVPR 2019), Center Loss (Wen et al., ECCV 2016), L3AM for hand gesture authentication.

---

#### P4: Signer-Adversarial Training (Expected: +3-5% cross-signer)

**Expected Impact**: MEDIUM (+3-5%) | **Complexity**: MEDIUM | **Risk**: MEDIUM

Add a signer classification head with gradient reversal layer (GRL). The feature extractor learns to produce representations that cannot predict signer identity.

**Lambda scheduling**: Start at 0.01, linearly ramp to 0.1 over training. Monitor signer classifier accuracy -- it should converge toward 25% (random for 4 signers).

**Risk**: With only 4 signers, the adversarial dynamics may be unstable. Mitigated by soft labels and low lambda values.

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# Add to model: features -> GRL -> signer_classifier (2-layer MLP)
# Loss = L_arcface + alpha * L_supcon + lambda * L_signer_adversarial
```

**Citations**: DANN (Ganin et al., JMLR 2016), Adversarial Autoencoder for CSLR (Kamal et al. 2024, DOI:10.1002/cpe.8220).

---

#### P5: Prototypical Classification Head (Expected: +5-10% vs standard classifier)

**Expected Impact**: HIGH (+5-10% over CE classifier) | **Complexity**: MEDIUM | **Risk**: LOW

Replace the standard FC classifier with prototypical network-style inference. At inference time, classify by nearest class prototype (mean embedding of all training samples per class) in the learned embedding space.

**Why this matters**: Prototypical networks outperform standard classifiers by 13% on WLASL skeleton data (arXiv:2512.10562). They naturally produce signer-invariant embeddings because the prototype averages over all signers.

**Training**: Episodic learning with LOSO-style episodes (train on 3 signers, query on 1).

**Inference**: Compute class prototypes from all training samples, classify test samples by Euclidean distance to nearest prototype.

**Citations**: Prototypical Networks (Snell et al., NeurIPS 2017), Few-shot ASL (arXiv:2512.10562, 2025), ProtoGCN (CVPR 2024).

---

#### P6: Multi-Stream Late Fusion Ensemble (Expected: +2-5%)

**Expected Impact**: MEDIUM (+2-5%) | **Complexity**: LOW (but 4x training cost) | **Risk**: LOW

Train 4 separate lightweight models on different input streams, fuse predictions via weighted sum of softmax logits.

| Stream | Input | Captures |
|--------|-------|----------|
| Joint | Normalized XYZ coordinates | Absolute spatial configuration |
| Bone | Bone direction + length vectors | Signer-invariant structure |
| Joint velocity | Frame-to-frame joint differences | Motion dynamics |
| Bone velocity | Frame-to-frame bone differences | Structural change dynamics |

**Fusion weights**: Start with equal [1,1,1,1], then tune via grid search. Literature suggests bone stream typically gets highest weight (1.2-1.5x).

**Alternative**: Early fusion (EfficientGCN MIB style) to avoid 4x training cost -- fuse streams at the input level into a single model.

**Citations**: SAM-SLR (Jiang et al., CVPR 2021W), DPGCN (Jang et al., 2025), Multi-stream angular fusion (Huang et al., 2024).

---

## 6. Implementation Plan

### Phase 1: Data Pipeline (Week 1)

**Step 1: Undo aggressive deduplication**
- Keep all ~750 samples or light dedup to ~650 (only remove byte-identical copies)
- Maintain signer labels for all samples

**Step 2: Implement multi-stream feature extraction**
```python
def extract_features(raw_landmarks_75x3):
    """Extract multi-stream features from MediaPipe landmarks."""
    pose = raw_landmarks_75x3[:33]       # 33 pose landmarks
    left_hand = raw_landmarks_75x3[33:54]  # 21 left hand landmarks
    right_hand = raw_landmarks_75x3[54:]   # 21 right hand landmarks

    # Normalize
    pose_norm = normalize_pose(pose)        # Body-center, shoulder-width scaled
    lh_norm = normalize_hand(left_hand)     # Wrist-center, palm-size scaled
    rh_norm = normalize_hand(right_hand)    # Wrist-center, palm-size scaled

    # Stream 1: Normalized joints (already done)
    joints = np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])

    # Stream 2: Bone vectors
    lh_bones = compute_bone_vectors(lh_norm, HAND_BONES)
    rh_bones = compute_bone_vectors(rh_norm, HAND_BONES)
    bones = np.concatenate([lh_bones.flatten(), rh_bones.flatten()])

    # Stream 3: Joint angles
    lh_angles = compute_joint_angles(lh_bones)
    rh_angles = compute_joint_angles(rh_bones)
    angles = np.concatenate([lh_angles, rh_angles])

    # Stream 4: Fingertip distances
    lh_dists = compute_fingertip_distances(lh_norm)
    rh_dists = compute_fingertip_distances(rh_norm)
    distances = np.concatenate([lh_dists, rh_dists])

    return joints, bones, angles, distances
```

**Step 3: Compute velocity features**
- Compute velocities on original-framerate sequences BEFORE temporal resampling
- Bone velocities = frame-to-frame bone vector differences

**Step 4: Fix MediaPipe version**
- Pin mediapipe==0.10.14 in all environments

### Phase 2: Model Architecture (Week 1-2)

**Step 5: Build lightweight multi-input model**

Target architecture: EfficientGCN-B0-inspired or slimmed InfoGCN, 0.3-0.5M total parameters.

```python
class KSLV21Model(nn.Module):
    def __init__(self, joint_dim, bone_dim, angle_dim, dist_dim,
                 hidden_dim=64, num_classes=30, num_signers=4):
        super().__init__()
        total_input = joint_dim + bone_dim + angle_dim + dist_dim

        # Temporal feature extractor (lightweight)
        self.encoder = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Temporal convolution blocks (3-4 layers)
            TemporalBlock(hidden_dim, hidden_dim, kernel=9, dropout=0.3),
            TemporalBlock(hidden_dim, hidden_dim, kernel=9, dropout=0.3),
            TemporalBlock(hidden_dim, hidden_dim, kernel=9, dropout=0.3),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

        # ArcFace classification head
        self.arcface = ArcFaceHead(128, num_classes, s=30.0, m=0.3)

        # Signer adversarial head (with GRL)
        self.signer_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_signers),
        )

    def forward(self, x, labels=None, signer_labels=None, lambda_grl=0.0):
        # x: (batch, time, features)
        features = self.encoder(x)  # (batch, hidden_dim)
        embeddings = self.projection(features)  # (batch, 128)

        # Classification
        if labels is not None:
            sign_logits = self.arcface(embeddings, labels)
        else:
            sign_logits = self.arcface.inference(embeddings)

        # Signer adversarial
        reversed_features = GradientReversalLayer.apply(features, lambda_grl)
        signer_logits = self.signer_head(reversed_features)

        return sign_logits, signer_logits, embeddings
```

**Step 6: Implement loss function**
```python
# Combined loss
L_total = L_arcface + 0.3 * L_supcon + lambda_grl * L_signer_ce + 0.01 * L_center
```

### Phase 3: Augmentation (Week 2)

**Step 7: Implement domain randomization augmentation suite**

Full augmentation pipeline as specified in P2 above, with progressive curriculum scheduling.

**Step 8: Implement bone-space style transfer**
```python
def style_transfer(source_sequence, target_signer_stats):
    """Transfer source motion to target signer's body proportions."""
    source_bones = compute_bone_vectors(source_sequence)
    source_directions = source_bones / (np.linalg.norm(source_bones, axis=-1, keepdims=True) + 1e-8)
    # Use target signer's bone lengths with source's bone directions
    synthetic = source_directions * target_signer_stats['mean_bone_lengths']
    return bones_to_joints(synthetic)
```

### Phase 4: Training (Week 2-3)

**Step 9: Training configuration**
```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-3
scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
batch_size: 32
epochs: 200
early_stopping_patience: 40
gradient_clipping: max_norm=1.0
label_smoothing: 0.1

# Stage 1 (epochs 1-50): SupCon pretraining
supcon_weight: 1.0
arcface_weight: 0.0

# Stage 2 (epochs 51-200): Joint training
supcon_weight: 0.3
arcface_weight: 1.0
signer_adversarial_lambda: 0.0 -> 0.1 (linear ramp)
center_loss_lambda: 0.01
```

**Step 10: Evaluation protocol**
```
Primary: Leave-One-Signer-Out (LOSO) accuracy (average over 3 real signers, excluding duplicates)
Secondary: Real-tester accuracy (3 unseen signers)
Tertiary: Validation accuracy (random split -- for monitoring overfitting only)

Report for each:
  - Per-class accuracy (30 classes)
  - Numbers accuracy (15 classes)
  - Words accuracy (15 classes)
  - Confusion matrix
  - Per-confidence-bucket accuracy (0-0.3, 0.3-0.6, 0.6-0.8, 0.8-1.0)
  - Per-signer accuracy
```

### Phase 5: Ensemble & Deployment (Week 3-4)

**Step 11: Multi-stream ensemble** (if single model doesn't hit target)
- Train 4 stream-specific models
- Tune fusion weights on validation set
- Evaluate ensemble on real testers

**Step 12: Test-Time Augmentation**
- Average predictions over 5-10 augmented versions
- Include: original, horizontal flip, speed 0.9x, speed 1.1x, +/-5 deg rotation

**Step 13: Confidence-based deployment**
- High confidence (>0.6): Display prediction directly
- Medium confidence (0.4-0.6): Show prediction with confirmation prompt
- Low confidence (<0.4): Request re-signing, show top-3

---

## 7. Realistic Accuracy Targets

### Given Our Data Constraints (4 unique training signers, ~600 samples, 30 classes)

| Metric | Current (v19) | V21 Target (Conservative) | V21 Target (Optimistic) | Theoretical Max (4 signers) |
|--------|--------------|---------------------------|-------------------------|-----------------------------|
| **Validation (random split)** | 87.3% | 85-90% | 88-92% | 95%+ |
| **LOSO cross-signer** | 37.8% | 45-50% | 50-60% | 60-65% |
| **Real-tester overall** | 34.2% | 42-50% | 50-58% | 55-65% |
| **Real-tester numbers** | 20.3% | 30-38% | 38-48% | 45-55% |
| **Real-tester words** | 48.1% | 55-65% | 65-72% | 70-80% |

### Per-Confidence-Bucket Targets

| Confidence Bucket | Current Accuracy | V21 Target | Coverage (% of predictions) |
|-------------------|-----------------|------------|----------------------------|
| >0.8 | ~90% | 90-95% | 15-25% |
| 0.6-0.8 | ~81% | 82-88% | 20-30% |
| 0.4-0.6 | ~50% | 55-65% | 20-25% |
| 0.2-0.4 | ~25% | 30-40% | 15-20% |
| <0.2 | ~15% | 20-25% | 10-15% |

### What "Success" Looks Like

**Minimum viable V21** (worth deploying):
- Real-tester overall: >45%
- Real-tester words: >60%
- High-confidence (>0.6) accuracy: >85%
- LOSO: >45%

**Good V21** (clear improvement):
- Real-tester overall: >50%
- Real-tester words: >65%
- High-confidence accuracy: >88%
- LOSO: >50%

**Excellent V21** (approaching theoretical limit for 4 signers):
- Real-tester overall: >55%
- Real-tester words: >70%
- High-confidence accuracy: >90%
- LOSO: >55%

### Why We Set These Targets

The literature is clear: with only 4 training signers, achieving >65% cross-signer accuracy is extremely unlikely regardless of technique. AUTSL with 43 signers achieves 62% signer-independent accuracy; our theoretical maximum with 4 signers is estimated at 55-65% based on the scaling relationship between signer count and generalization. Setting targets above this would be unrealistic and discouraging.

---

## 8. Fallback Strategies

### Plan B: If V21 multi-stream + contrastive doesn't improve over v19's 34.2%

**Diagnosis**: The augmentation and features are not creating enough signer diversity.

**Action**:
1. Implement full bone-space signer style transfer (create 12 synthetic signers from 4 real ones)
2. Add cross-lingual transfer learning: pretrain on WLASL skeleton data, fine-tune on KSL
3. Try prototypical network with episodic training (completely different training paradigm)
4. Reduce to 15-20 most distinguishable classes (drop confusable pairs)

### Plan C: If real-tester accuracy stays below 40% after Plan B

**Diagnosis**: 4 signers is fundamentally insufficient.

**Action**:
1. **Collect 5 more signers** (even 3-5 repetitions per sign per signer would help)
   - Total cost: ~2 hours of recording per signer
   - Expected impact: +15-25% from the additional signer diversity alone
2. Deploy a **confidence-gated system**: Only show predictions when confidence > 0.5
   - At >0.5, accuracy should be ~70-80%
   - Coverage will be ~40-50% of inputs (reject the rest)
3. Add **user feedback loop**: Let users confirm/correct predictions to build a per-user adaptation dataset

### Plan D: If data collection is impossible and accuracy remains below 40%

**Diagnosis**: Need to fundamentally change the problem framing.

**Action**:
1. **Reduce class count**: Drop the 8 hardest number classes (those at 0%), train only on the 22 distinguishable signs
   - Expected accuracy on remaining classes: 50-60%
2. **User-adaptive few-shot**: On first use, collect 3-5 calibration samples per sign from the new user, use for few-shot adaptation
   - Prototypical networks are naturally suited for this (just update prototypes)
3. **Hybrid keyboard**: Show top-3 predictions and let user select, reducing the task from 30-way classification to 3-way verification
   - Top-3 accuracy is likely >60% even now
4. **Transfer from larger datasets**: Use WLASL/AUTSL pretrained models and fine-tune on KSL

### Plan E: Long-term scalability

1. Build a **community data collection platform**: Allow KSL signers to contribute labeled recordings
2. Target **50+ unique signers** over 6 months
3. At 50 signers, state-of-the-art techniques should achieve >80% cross-signer accuracy
4. Implement continuous learning: automatically retrain as new data arrives

---

## 9. Data Collection Recommendations

### 9.1 Priority: More Signer Diversity (Highest Impact Possible)

The research is unambiguous: **more signer diversity is the single highest-impact intervention available**. No combination of architectural changes, augmentations, or training techniques can substitute for actual human variation.

**Minimum viable data collection** (expected +15-25%):
- 5 additional signers performing all 30 signs
- 3-5 repetitions per sign per signer
- Total: ~450-750 new samples
- Estimated recording time: 2 hours per signer
- This brings us from 4 to 9 unique signers

Based on AUTSL scaling curves, going from 4 to 9 signers should reduce the cross-signer gap by 40-50%, translating to roughly 55-70% real-tester accuracy with our improved techniques.

### 9.2 Recording Protocol for Maximum Diversity

To maximize the value of each new signer:

**Signer selection criteria**:
- Mix of male/female (different hand sizes)
- Mix of ages (different signing speeds and precision)
- Mix of experience levels (fluent vs. learner)
- At least one left-handed signer
- At least one signer with notably large/small hands

**Recording conditions**:
- **Frontal view** (primary): Camera at chest height, 1-2m distance
- **Slight angle** (secondary): 15-20 deg off-center, to build viewpoint robustness
- Consistent lighting (avoid backlighting)
- Plain background preferred but not required
- Record at 30fps minimum

**Per-sign protocol**:
- 3 natural-speed repetitions
- 1 slow repetition (for temporal variation)
- 1 fast repetition (for speed robustness)

### 9.3 If No New Signers Are Available

Maximize synthetic diversity through:
1. **Bone-space style transfer**: Create 12 synthetic "signers" from 4 real ones (4 signers x 3 target styles)
2. **Domain randomization**: Randomize anatomical proportions per training sample
3. **Temporal augmentation**: Wide speed variation (0.5x-2.0x) simulates different signing rhythms
4. **Cross-lingual pretraining**: Use WLASL or AUTSL skeleton data for pretraining the encoder

### 9.4 Data Quality Improvements

Even without new signers, improve existing data:
1. **Reprocess all data with consistent MediaPipe version** (0.10.14)
2. **Add temporal annotations**: Mark sign start/end frames to remove idle frames
3. **Quality filter**: Remove samples where MediaPipe tracking is poor (low confidence landmarks)
4. **Verify labels**: Manual review of confusable pairs (35/22, 388/48) to ensure correct labeling

---

## 10. References

### Cross-Signer Generalization
1. Ganin, Y., Ustinova, E., Ajakan, H., et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR 17(1). -- Gradient reversal layer for domain-invariant features.
2. Sincan, O.M. & Keles, H.Y. (2020). "AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset." -- 43-signer benchmark showing 34-point signer-independent gap.
3. Bani Baker, Q. et al. (2023). Arabic SL recognition. Applied Computational Intelligence and Soft Computing. DOI: 10.1155/2023/5195007.
4. Ethiopian SLR (2025). "A deep learning framework for Ethiopian sign language recognition using skeleton-based representation." Nature Scientific Reports. https://www.nature.com/articles/s41598-025-19937-0
5. Kamal, S.M., Chen, Y., & Li, S. (2024). "Adversarial autoencoder for continuous sign language recognition." Concurrency and Computation 36(22). DOI: 10.1002/cpe.8220.
6. Dual-Stream ST-DGCN (2025). arXiv:2509.08661v1. -- Dual-reference normalization for signer invariance.

### Data Augmentation & Synthetic Data
7. Guo, T. et al. (2022). "Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition" (AimCLR). AAAI 2022. arXiv:2112.03590.
8. Li, L. et al. (2023). "Cross-Stream Contrastive Learning for Self-Supervised Skeleton-Based Action Recognition" (CrosSCLR). arXiv:2305.02324.
9. Mehraban et al. (2024). "Self-supervised Tuning for 3D Action Recognition in Skeleton Sequences" (STARS). arXiv:2407.10935.
10. SkeletonMix (2024). IEEE Xplore. https://ieeexplore.ieee.org/document/10888125/
11. Joint Mixing Data Augmentation (2024). ACM TOMM. https://dl.acm.org/doi/10.1145/3700878
12. SynthDa (NVIDIA 2025). https://developer.nvidia.com/blog/improving-synthetic-data-augmentation-and-human-action-recognition-with-synthda/
13. Motion Style Transfer survey (2023). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC10007042/

### Model Architectures
14. Zhou, Y. et al. (2024). "BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition." CVPR 2024. https://github.com/ZhouYuxuanYX/BlockGCN
15. Song, Y.F. et al. (2022). EfficientGCN: Compound scaling for skeleton action recognition. -- B0 variant at 0.29M params.
16. Chi, H.G. et al. (2022). "InfoGCN: Representation Learning for Human Skeleton-Based Action Recognition." CVPR 2022. https://github.com/stnoah1/infogcn
17. Lee, J. et al. (2023). "Hierarchically Decomposed Graph Convolutional Networks" (HD-GCN). ICCV 2023.
18. SkateFormer (2024). "Skeletal-Temporal Transformer for Human Action Recognition." ECCV 2024.
19. Cheng, K. et al. ShiftGCN++: Extremely Lightweight Skeleton-Based Action Recognition.

### Sign Language-Specific
20. Jiang, H. et al. (2021). "Skeleton Aware Multi-modal Sign Language Recognition" (SAM-SLR). CVPR 2021 Workshop. https://github.com/jackyjsy/CVPR21Chal-SLR -- Challenge winner at 98.42%.
21. HA-GCN (2024). Hand-Aware GCN for skeleton-based SLR. DOI: 10.1016/j.nlp.2024.100074.
22. Aiman, U. & Ahmad, T. (2023). "Angle based hand gesture recognition using graph convolutional network" (AHG-GCN). Computer Animation and Virtual Worlds 35(1). DOI: 10.1002/cav.2207. -- 90% on DHG-14 with 25 angle features/joint.
23. SC2SLR (2024). "Skeleton-based Contrast for Sign Language Recognition." CNIOT 2024. https://dl.acm.org/doi/10.1145/3670105.3670173
24. SL-SLR (2025). "Self-Supervised Representation Learning for Sign Language Recognition." arXiv:2509.05188.

### Loss Functions & Metric Learning
25. Deng, J. et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
26. Khosla, P. et al. (2020). "Supervised Contrastive Learning." NeurIPS 2020.
27. Snell, J. et al. (2017). "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
28. ProtoGCN (2024). "Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition." CVPR 2024.
29. Quaternion-Based Pose Encoding + Contrastive Learning (2025). arXiv:2508.14574. -- 16% improvement from contrastive loss alone.

### Few-Shot & Transfer Learning
30. "Data-Efficient ASL Recognition via Few-Shot Prototypical Networks" (2025). arXiv:2512.10562. -- 13% improvement over standard classifiers on WLASL.
31. "Cross-lingual few-shot sign language recognition" (2024). Pattern Recognition. DOI: 10.1016/j.patcog.2024.110374.
32. Wei, C. et al. (2024). "Multimodal hand/finger movement sensing and fuzzy encoding for data-efficient universal sign language recognition." InfoMat 7(4). DOI: 10.1002/inf2.12642.
33. "Training Strategies for ISLR" (2024). arXiv:2412.11553. -- +6.54% from training strategy improvements.

### Knowledge Distillation
34. Structural KD (IEEE 2021). Structural Knowledge Distillation for Skeleton Action Recognition.
35. Part-Level KD (CVPR 2024). Enhancing Action Recognition from Low-Quality Skeleton Data.
36. C2VL (2024). Vision-Language Cross-Modal KD for Skeleton.

---

*This report synthesizes findings from three parallel research tracks (cross-signer generalization, data augmentation, model architectures) conducted on 2026-02-09. It incorporates analysis from the real-tester gap report and lessons learned from v16-v20 development. It serves as the definitive guide for V21 development of the KSL recognition system.*
