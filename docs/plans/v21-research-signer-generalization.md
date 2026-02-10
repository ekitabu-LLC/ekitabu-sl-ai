# V21 Research: Cross-Signer Generalization for Sign Language Recognition

## Executive Summary

Our KSL project faces a **53-point gap** between validation accuracy (87%) and real-tester accuracy (34%), with true LOSO cross-signer accuracy at 37.8%. This document synthesizes 2023-2026 research on closing such gaps in skeleton-based sign language recognition.

**Key finding**: With only 4 unique training signers, conventional supervised learning will never generalize well. The literature consistently shows 20-35% accuracy drops in signer-independent evaluations, and our gap is within the expected range for this data regime. The path forward requires a combination of: (1) signer-invariant feature engineering, (2) domain adaptation, (3) aggressive data augmentation, and (4) architectural changes.

---

## 1. SOTA Techniques for Signer-Independent SLR (2023-2026)

### 1.1 Benchmark Performance Gaps

The cross-signer generalization problem is well-documented across major benchmarks:

| Dataset | Signer-Dependent | Signer-Independent | Gap |
|---------|-----------------|-------------------|-----|
| AUTSL (226 signs, 43 signers) | 95.95% | 62.02% | -34% |
| Ethiopian SL (skeleton, 5+2 signers) | 94% | 73% | -21% |
| Arabic SL (23 words, 3 users) | 95%+ | 89.5% | -6% |
| Our KSL (30 signs, 4 signers) | 87% | 34% | -53% |

**Key insight**: Our 53-point gap is larger than typical because we have only 4 training signers. The AUTSL gap of 34 points comes from 43 signers. With more signer diversity in training, gaps typically shrink to 10-20%.

Sources:
- AUTSL benchmark: Sincan & Keles (2020), "AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset"
- Ethiopian SL: [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-19937-0)
- Arabic SL: Bani Baker et al. (2023), Applied Computational Intelligence and Soft Computing, DOI: 10.1155/2023/5195007

### 1.2 Current SOTA Approaches

**Multi-stream skeleton ensembles** remain the strongest approach for skeleton-based SLR:
- Joint stream (raw XYZ coordinates)
- Bone stream (parent-child joint differences) -- signer-invariant by design
- Joint-motion stream (frame-to-frame velocity)
- Bone-motion stream (bone velocity)
- **Angular/geometric features** -- scale and position invariant

Reference: Sign Language Recognition via Skeleton-Aware Multi-Model Ensemble (SAM-SLR, CVPR 2021W) achieved 98.42% on the ChaLearn LAP Large Scale Signer Independent Isolated SLR Challenge.

**Dual-reference normalization** (2025): A novel approach that normalizes skeleton sequences with respect to two different reference frames, providing complementary geometric constraints. This decouples input into morphological and trajectory representations.
Source: [Dual-Stream Spatio-Temporal Dynamic GCN (2025)](https://arxiv.org/html/2509.08661v1)

**Signer-invariant Conformer** (ICCVW 2025): Multi-Scale Fusion Transformer for continuous SLR that learns representations invariant to signer identity.

---

## 2. Signer-Adversarial Training (Gradient Reversal Layers)

### 2.1 How It Works

Domain Adversarial Neural Networks (DANN), introduced by Ganin & Lempitsky (2015), use a gradient reversal layer (GRL) between a feature extractor and a domain (signer) classifier. During backpropagation, the GRL multiplies gradients by -gamma, forcing the feature extractor to produce representations that cannot be distinguished across signers.

Architecture:
```
Input -> Feature Extractor -> [Sign Classifier] (minimize classification loss)
                           -> [GRL] -> [Signer Classifier] (maximize signer confusion)
```

The key equations:
- Forward pass through GRL: identity function
- Backward pass through GRL: gradient * (-gamma)
- Result: features that are discriminative for sign class but invariant to signer identity

Source: Ganin et al. (2016), "Domain-Adversarial Training of Neural Networks", JMLR 17(1).

### 2.2 Applicability to Our Problem

**Direct signer-adversarial training is highly applicable** to our problem because:
1. We can use signer ID as the domain label (we have 4 distinct signers)
2. It can be added as an auxiliary head to any existing architecture (ST-GCN, etc.)
3. It explicitly penalizes features that encode signer identity
4. Implementation is straightforward in PyTorch

**Potential issues with only 4 signers**:
- With only 4 domain classes, the discriminator may be too easy to train
- Risk of mode collapse or oscillating training
- Lambda (gradient reversal strength) scheduling is critical -- start small (0.01), ramp up

**Recommendation**: Use signer-adversarial training but with soft labels and temperature scaling. Rather than hard signer IDs, use a signer embedding that captures continuous variation.

### 2.3 Related Work in SLR

While there are no widely-cited papers specifically combining GRL with skeleton-based SLR as of 2025, the technique has been successfully applied to:
- Cross-subject EEG recognition (lightweight DANN, 2023)
- Cross-domain action recognition with skeleton data
- Speaker-independent speech recognition (analogous problem)

The SAM-SLR challenge (ChaLearn 2021) was explicitly "signer independent", and top solutions relied on multi-stream ensembles rather than adversarial training, suggesting that **feature engineering (bones, angles) may be more impactful than adversarial training** for this specific problem.

---

## 3. Domain Adaptation Techniques for Few Source Signers (4-5)

### 3.1 The Few-Source-Domain Problem

Our situation (4 training signers) is an extreme case of domain adaptation. Standard techniques assume many source domains. With so few:

**What works**:
1. **Leave-One-Signer-Out (LOSO) validation** -- essential for honest evaluation
2. **Signer-invariant features** (bones, angles, distances) -- reduces the domain gap at the feature level
3. **Data augmentation to simulate new signers** -- most impactful approach
4. **Meta-learning/few-shot approaches** -- learn to generalize from episodes

**What struggles**:
1. Adversarial training with only 4 domains -- too few for stable adversarial dynamics
2. Distribution matching (MMD, CORAL) -- hard to estimate distributions from 4 sources
3. Style transfer between 4 signers -- limited diversity to interpolate from

### 3.2 Recommended DA Strategy Stack

For our specific scenario (4 signers, 600 samples, 30 classes):

**Tier 1 -- Feature-level invariance (highest impact)**:
- Bone features (position-invariant by construction)
- Joint angles (scale-invariant, captures hand shape regardless of hand size)
- Relative distances between fingertips
- Velocity features (normalize out static pose differences)

**Tier 2 -- Augmentation-based domain expansion**:
- Spatial augmentation: random joint noise, random scaling, random rotation
- Signer mixing: interpolate skeleton sequences between signers (MixUp in skeleton space)
- Synthetic signer generation via learned style transfer

**Tier 3 -- Training-time regularization**:
- Signer-adversarial head with careful lambda scheduling
- Mixup across signers (force model to handle intermediate styles)
- Dropout on signer-specific neurons (encourages signer-agnostic features)

### 3.3 Key References

- TCA (Transfer Component Analysis): Pan et al. -- minimizes MMD between domains after feature mapping
- DAN (Deep Adaptation Network): Long et al. -- multi-kernel MMD in deep feature space
- DANN: Ganin & Lempitsky -- adversarial feature alignment
- Adversarial Autoencoder for CSLR: Kamal et al. (2024), DOI: 10.1002/cpe.8220

---

## 4. Cross-Signer Accuracy Benchmarks

### 4.1 WLASL (Word-Level ASL)

| Method | WLASL-100 | WLASL-300 | WLASL-2000 | Notes |
|--------|-----------|-----------|------------|-------|
| DSLNet (2024) | 93.70% | 89.97% | -- | SOTA, but signer-overlap likely |
| VideoMAE | 75.58% | -- | -- | Pretrained, WLASL-100 |
| Few-shot Prototypical (2025) | 43.75% (Top-1) | -- | -- | 77.10% Top-5, skeleton-only |
| MViTv2-S + pipeline (2024) | +6.54% over baseline | -- | -- | Training strategy improvements |

**Important note**: WLASL standard splits likely have signer overlap between train/test. True signer-independent numbers would be significantly lower.

Sources:
- [Few-shot prototypical on WLASL](https://arxiv.org/abs/2512.10562)
- [Training Strategies for ISLR](https://arxiv.org/abs/2412.11553)

### 4.2 AUTSL (Turkish SL)

- 226 sign classes, 43 signers, 38,336 samples
- Random split: 95.95% accuracy
- **User-independent benchmark: 62.02%** (34-point gap!)
- SAM-SLR (multi-modal): 98.42% (signer-independent challenge, but 43 training signers)

Source: [AUTSL benchmark](https://www.researchgate.net/publication/346044022_AUTSL_A_Large_Scale_Multi-Modal_Turkish_Sign_Language_Dataset_and_Baseline_Methods)

### 4.3 MSASL (MS American SL)

- Ensemble transformer (2025): 92.56% top-1 accuracy
- Multi-modal input fusion with skeleton + RGB

### 4.4 Implications for KSL

With 43 signers, AUTSL signer-independent accuracy drops to 62%. With only 4 signers, our 34% cross-signer accuracy is **within the expected range**. To reach 50%+ cross-signer:
- Need signer-invariant features (bones, angles) -- potential +10-15%
- Need aggressive augmentation simulating new signers -- potential +5-10%
- Need architectural changes (adversarial, contrastive) -- potential +3-5%

---

## 5. Few-Shot and Meta-Learning for Sign Language

### 5.1 Few-Shot Prototypical Networks for SLR

**Key paper**: "Data-Efficient ASL Recognition via Few-Shot Prototypical Networks" (Dec 2025)

Architecture: ST-GCN + Multi-Scale Temporal Aggregation (MSTA) module trained with episodic/prototypical learning.

Results:
- WLASL Top-1: 43.75%, Top-5: 77.10%
- **Outperforms standard classifier by 13%+ on same backbone**
- Zero-shot transfer to unseen SignASL: ~30% accuracy without fine-tuning

Key insight: **Prototypical learning substantially outperforms standard classification** when data is scarce. Instead of learning fixed decision boundaries that overfit to training signers, it learns a metric space where signs cluster by semantic meaning rather than signer identity.

Source: [arxiv.org/abs/2512.10562](https://arxiv.org/abs/2512.10562)

### 5.2 Cross-Lingual Few-Shot SLR

"Cross-lingual few-shot sign language recognition" (Pattern Recognition, 2024) demonstrates transfer across sign languages (ASL, German SL, Turkish SL), showing that skeleton-based representations transfer better across languages than RGB features.

Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320324001250)

### 5.3 Meta-Learning Applicability to KSL

**Highly recommended** for our scenario because:
1. With 4 signers, we can do LOSO meta-training: train on 3 signers, validate on 1
2. Prototypical networks naturally produce signer-invariant embeddings
3. They handle class imbalance well (prototype = mean of support set)
4. They are computationally cheap at inference time
5. The 13% improvement over standard classifiers on WLASL is very significant

**Implementation approach**:
- Use our ST-GCN backbone as the embedding function
- Train episodically: sample N-way K-shot episodes from training signers
- Compute class prototypes as mean embeddings
- Classify by nearest prototype in embedding space
- At test time: use all training samples per class to compute robust prototypes

---

## 6. Normalization Strategies for Signer Invariance

### 6.1 Beyond Wrist-Centric Normalization

Our v20 used wrist-centric normalization, which did not help. Research suggests more sophisticated approaches:

**1. Dual-Reference Normalization (2025)**
- Normalize with respect to TWO reference frames simultaneously
- Reference 1: body center (torso midpoint) -- captures global hand position
- Reference 2: wrist joint -- captures local hand shape
- Feed both normalized streams to dual-stream architecture
- Source: [Dual-Stream ST-DGCN](https://arxiv.org/html/2509.08661v1)

**2. Bone-Length Normalization**
- Divide all coordinates by the signer's average bone length
- Makes features invariant to hand size differences
- Critical for cross-signer: hand size varies 20-40% across signers

**3. Procrustes Alignment**
- Rigidly align skeleton to a canonical pose (remove translation, rotation, scale)
- Preserves only the relative joint positions
- Standard in gesture recognition, underused in SLR

**4. Angular Features (Position + Scale Invariant)**
- Joint angles between connected bones
- Fingertip-to-fingertip angles
- Finger curl angles (each finger independently)
- These are inherently signer-invariant: a "thumbs up" has the same angles regardless of hand size or position
- Reference: Aiman & Ahmad (2023), "Angle based hand gesture recognition using GCN", DOI: 10.1002/cav.2207 -- achieved 90% on DHG-14 using 25 angle-based features per joint

**5. Relative Coordinate Conversion**
- Convert absolute XYZ to spherical/polar coordinates relative to wrist
- Subtract wrist coordinates from first frame for all joints across all frames
- Source: HGCN (Hand Gesture-based Graph Convolutional Network)

### 6.2 Recommended Normalization Pipeline for KSL V21

```
Step 1: Center skeleton on body midpoint (hip/shoulder center)
Step 2: Scale by shoulder width or arm length (normalize body size)
Step 3: Compute bone vectors (child - parent joint)
Step 4: Compute joint angles (acos of dot product between adjacent bones)
Step 5: Compute fingertip distances (pairwise distances between all fingertips)
Step 6: Compute velocities (frame-to-frame differences)
Step 7: Multi-stream input: [bones, angles, distances, velocities]
```

---

## 7. Contrastive Learning for Skeleton-Based Sign Recognition

### 7.1 Relevant Approaches

**SC2SLR: Skeleton-based Contrast for Sign Language Recognition (2024)**
- Uses contrastive learning to enhance skeleton feature representations
- Multi-view contrastive learning augments data and improves expression capability
- Published at CNIOT 2024
- Source: [ACM Digital Library](https://dl.acm.org/doi/10.1145/3670105.3670173)

**SL-SLR: Self-Supervised Representation Learning for SLR (2025)**
- Contrastive methods build invariant representations from augmentations
- Learn to produce similar representations between a sign and its augmented variant
- Augmentation-preserving approach helps focus on discriminant parts of signs
- Source: [arxiv.org/html/2509.05188v1](https://arxiv.org/html/2509.05188v1)

**Quaternion-Based Pose Encoding + Contrastive Learning (2025)**
- Encodes poses using bone rotations in quaternion space
- Contrastive loss structures embeddings by semantic similarity
- Filters out anatomical and stylistic features (signer-specific noise)
- **16% improvement in Probability of Correct Keypoint** from contrastive loss alone
- 6% reduction in Mean Bone Angle Error when combined with quaternion encoding
- Source: [arxiv.org/abs/2508.14574](https://arxiv.org/abs/2508.14574)

**Generative Sign-Description with Multi-Positive Contrastive Learning (2025)**
- Uses sign descriptions as semantic anchors
- Multiple positive pairs per sign (different signers doing same sign = positive pair)
- Source: [arxiv.org/html/2505.02304](https://arxiv.org/html/2505.02304)

### 7.2 Contrastive Learning for Signer Invariance

The key insight for our problem: **Supervised contrastive learning can explicitly teach the model that the same sign performed by different signers should have similar embeddings**.

**SupCon loss for KSL**:
```python
# Positive pairs: same sign class, different signers
# Negative pairs: different sign class
# This directly optimizes for signer invariance

L_supcon = -sum(log(
    exp(sim(z_i, z_p) / tau) /
    sum(exp(sim(z_i, z_a) / tau) for all a != i)
)) for all positive pairs (i, p)
```

This is potentially **the single most impactful technique** for our problem because:
1. It directly optimizes for the property we want (signer invariance)
2. It works with very few signers (positive pairs = same class, any signer)
3. It produces better-separated embeddings than cross-entropy
4. It naturally handles confusable pairs (pushes them apart)

---

## 8. Practical Recommendations for KSL V21

### 8.1 Priority-Ordered Implementation Plan

**P0 -- Signer-Invariant Features (Expected: +10-15% cross-signer)**
1. Replace raw XYZ with multi-stream features:
   - Bone vectors (21 bones per hand, position-invariant)
   - Joint angles (angle between adjacent bones, ~20 per hand)
   - Fingertip pairwise distances (10 pairs per hand, scale after normalization)
   - Frame-to-frame velocities of all above
2. Proper body-size normalization (divide by shoulder width or forearm length)
3. Dual-reference normalization (body-center + wrist)

**P1 -- Supervised Contrastive Learning (Expected: +5-10% cross-signer)**
1. Add SupCon loss alongside or replacing cross-entropy
2. Positive pairs = same sign, any signer
3. Temperature tau=0.07 (standard)
4. Two-stage: pretrain with SupCon, then fine-tune with CE + SupCon

**P2 -- Prototypical/Metric Learning (Expected: +5-10% vs standard classifier)**
1. Switch from standard classifier to prototypical network
2. Episodic training with LOSO-style episodes
3. At inference: classify by nearest class prototype
4. Provides natural confidence calibration (distance to prototype)

**P3 -- Signer-Adversarial Training (Expected: +3-5% cross-signer)**
1. Add signer classification head with gradient reversal layer
2. Start lambda=0.01, ramp to 0.1 over training
3. Use alongside sign classification loss
4. Monitor: signer classifier accuracy should converge to ~25% (random for 4 signers)

**P4 -- Aggressive Skeleton Augmentation (Expected: +5-10% cross-signer)**
1. Random joint noise (sigma=0.01-0.03 of normalized coordinates)
2. Random scaling (0.8-1.2x per axis, independently)
3. Random rotation around vertical axis (+/- 15 degrees)
4. Random hand size perturbation (scale hand bones by 0.85-1.15)
5. MixUp between signers (alpha=0.2-0.4)
6. Time warping (speed variation within sequence)

### 8.2 Architecture Recommendations

Given our constraints (600 samples, 30 classes, need real-time inference):

1. **Keep ST-GCN backbone** but use multi-stream input
2. **Model size: 200K-500K parameters** (confirmed appropriate for ~600 samples)
3. **Add projection head** for contrastive learning (2-layer MLP, 128-dim output)
4. **Add signer-adversarial head** (small 2-layer MLP with GRL)
5. **Ensemble**: train 3 models with different random seeds, average predictions

### 8.3 Evaluation Protocol

1. **Primary metric**: Leave-One-Signer-Out (LOSO) accuracy
2. **Secondary**: Real-tester accuracy (new signers not in training)
3. **Always report** per-class accuracy and confusion matrix
4. **Confidence calibration**: report accuracy at different confidence thresholds

### 8.4 Expected Outcomes

With all P0-P3 implemented:
- **Optimistic**: 55-65% cross-signer accuracy (up from 34%)
- **Realistic**: 45-55% cross-signer accuracy
- **Pessimistic**: 40-45% cross-signer accuracy

**Important caveat**: With only 4 training signers, achieving >70% cross-signer accuracy is unlikely without additional data. The literature clearly shows that signer diversity is the primary driver of generalization. Our best investment beyond these techniques would be **collecting data from 5-10 more signers**.

---

## 9. Data Collection Recommendations

The research is unambiguous: **more signer diversity is the most impactful intervention**.

### 9.1 Minimum Viable Data Collection

- **5 additional signers** performing all 30 signs
- Even 3-5 repetitions per sign per signer would help enormously
- Total: ~750 new samples (5 signers x 30 signs x 5 reps)
- This would bring us from 4 to 9 signers, which based on AUTSL scaling curves, should reduce the cross-signer gap by 40-50%

### 9.2 If Data Collection Is Not Possible

Focus on:
1. Synthetic signer generation via skeleton augmentation (P4)
2. Transfer learning from larger SLR datasets (WLASL, AUTSL)
3. Self-supervised pretraining on unlabeled sign language video
4. Cross-lingual transfer from ASL/other sign language datasets

---

## 10. Key Academic References

1. Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." ICML.
2. Aiman, U., & Ahmad, T. (2023). "Angle based hand gesture recognition using graph convolutional network." Computer Animation and Virtual Worlds, 35(1). DOI: 10.1002/cav.2207
3. Kamal, S.M., Chen, Y., & Li, S. (2024). "Adversarial autoencoder for continuous sign language recognition." Concurrency and Computation, 36(22). DOI: 10.1002/cpe.8220
4. Sincan, O.M. & Keles, H.Y. (2020). "AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset."
5. Jiang et al. (2021). "Skeleton Aware Multi-modal Sign Language Recognition." CVPRW 2021.
6. "Data-Efficient ASL Recognition via Few-Shot Prototypical Networks." (2025). arXiv:2512.10562.
7. "Cross-lingual few-shot sign language recognition." (2024). Pattern Recognition. DOI: 10.1016/j.patcog.2024.110374.
8. "SC2SLR: Skeleton-based Contrast for Sign Language Recognition." (2024). CNIOT 2024.
9. "Towards Skeletal and Signer Noise Reduction via Quaternion-Based Pose Encoding and Contrastive Learning." (2025). arXiv:2508.14574.
10. "SL-SLR: Self-Supervised Representation Learning for Sign Language Recognition." (2025). arXiv:2509.05188.
11. Wei, C., Liu, S., Yuan, J., & Zhu, R. (2024). "Multimodal hand/finger movement sensing and fuzzy encoding for data-efficient universal sign language recognition." InfoMat, 7(4). DOI: 10.1002/inf2.12642.
