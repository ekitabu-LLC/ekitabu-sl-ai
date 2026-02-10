# V24 Research: Synthetic Signer Augmentation & Domain Generalization

## Context

- KSL recognition: 30 classes, ST-GCN on 48-node skeleton data
- Only **4 training signers** -- confirmed ceiling at ~40% real-world accuracy
- V22 (best real): 93.3% val / 39.8% real (55pp gap)
- V23 (multi-stream ensemble, focal loss, calibration): 95.6% val / 37.9% real -- **no improvement**
- Architecture changes alone cannot break the signer diversity barrier

The core problem is **distribution shift across signers**: each signer has unique bone proportions, signing speed, amplitude, and micro-style. With only 4 training signers, the model memorizes signer-specific patterns rather than learning signer-invariant sign representations.

---

## 1. Synthetic Signer Generation

### 1.1 Skeleton Style Transfer via Bone-Length Retargeting

**Concept**: Decompose skeleton sequences into (a) motion trajectory (normalized joint angles over time) and (b) signer morphology (bone lengths, proportions). Recombine real motion with synthetic morphologies to create "virtual signers."

**Approach**:
- Extract bone lengths from each signer's skeleton data
- Normalize all sequences to a canonical skeleton (unit bone lengths)
- Sample new bone-length vectors by interpolating between signers or drawing from a distribution fitted to the 4 known signers
- Re-apply bone lengths to normalized motion to generate new "virtual signer" sequences

**Expected Impact**: MEDIUM-HIGH. Addresses the morphological component of signer variation. Simple to implement. However, morphology may be a smaller contributor to distribution shift than signing style (speed, amplitude, spatial precision).

**Implementation Effort**: LOW (5-10 lines of skeleton transform code)

**Relevant Work**:
- Skeleton retargeting is standard in animation/motion capture (CMU MoCap conventions)
- SAM-SLR (CVPR 2021 Challenge winner) used multi-stream skeleton features (joint, bone, joint motion, bone motion) for signer-independent recognition ([GitHub](https://github.com/jackyjsy/CVPR21Chal-SLR))

### 1.2 Signer-Adaptive Normalization / Instance Normalization Mixing

**Concept**: Treat each signer's feature statistics (mean, variance in the feature space) as their "style." Mix statistics across signers to synthesize new virtual signer styles.

**Approach -- MixStyle** (Zhou et al., 2021):
- During training, randomly select pairs of samples from different signers
- Mix their feature-level statistics (channel-wise mean and std) with a mixing coefficient sampled from Beta distribution
- Apply mixed statistics to one sample's content features
- This implicitly creates "virtual signer" feature distributions

**Key Results**:
- MixStyle achieves state-of-the-art domain generalization on PACS, OfficeHome, DomainNet benchmarks
- MixStyle-Based Contrastive Test-Time Adaptation (CVPR 2024 Workshop) further improves DG accuracy
- The technique is architecture-agnostic and adds zero parameters

**Expected Impact**: HIGH. Directly addresses signer style variation in feature space. Well-validated for domain generalization. Particularly effective when domain (signer) labels are available during training.

**Implementation Effort**: LOW (insert MixStyle layer after early GCN layers; ~20 lines of code)

**Relevant Work**:
- [Domain Generalization with MixStyle](https://arxiv.org/abs/2104.02008) (ICLR 2021)
- [MixStyle Neural Networks for Domain Generalization and Adaptation](https://dl.acm.org/doi/10.1007/s11263-023-01913-8) (IJCV 2023)
- [MixStyle-Based Contrastive TTA](https://openaccess.thecvf.com/content/CVPR2024W/MAT/html/Yamashita_MixStyle-Based_Contrastive_Test-Time_Adaptation_Pathway_to_Domain_Generalization_CVPRW_2024_paper.html) (CVPR 2024W)

### 1.3 Adversarial Signer Generation

**Concept**: Use adversarial training to generate skeleton perturbations that are maximally confusing to the current model, creating "hard" virtual signers.

**Approach**:
- Compute gradients of the classification loss w.r.t. the input skeleton
- Add adversarial perturbations (FGSM-style or PGD-style) to skeleton coordinates
- Train on both clean and adversarially-perturbed samples
- Optionally: use a learned perturbation network (generator) that produces signer-style perturbations

**Key Results**:
- Fu et al. (NAACL 2024) "Signer Diversity-driven Data Augmentation" uses adversarial training with model gradients to generate adversarial examples that enrich signer diversity. Combined with diffusion-based augmentation, achieves state-of-the-art signer-independent sign language translation **without requiring signer identity labels**.
- Adversarial Vulnerability-Seeking Networks (AVSN) (IEEE 2024) jointly trains augmentation and classification for skeleton-based sign language recognition.

**Expected Impact**: MEDIUM-HIGH. Adversarial perturbations directly target model weaknesses. However, gradient-based perturbations may not correspond to realistic signer variations.

**Implementation Effort**: LOW-MEDIUM (FGSM-style: ~15 lines; learned generator: significant)

**Relevant Work**:
- [Signer Diversity-driven Data Augmentation](https://aclanthology.org/2024.findings-naacl.140/) (NAACL 2024 Findings)
- [Skeleton-Based Data Augmentation for SLR Using Adversarial Learning](https://ieeexplore.ieee.org/document/10718297/) (IEEE 2024)

### 1.4 Interpolation in Embedding/Latent Space (CVAE-based)

**Concept**: Train a conditional variational autoencoder (CVAE) on skeleton sequences conditioned on class label and signer ID. Sample from the latent space to generate new signer variants.

**Approach**:
- Train CVAE: encoder maps (skeleton_seq, class, signer) -> latent z; decoder reconstructs skeleton_seq from (z, class)
- At augmentation time: sample z from prior, condition on class label only, decode to new skeleton sequence
- The latent space naturally disentangles content (sign class) from style (signer)

**Expected Impact**: MEDIUM. Theoretically powerful but requires careful training with only 4 signers. Risk of mode collapse or generating unrealistic sequences. High variance with small source domains.

**Implementation Effort**: HIGH (full VAE architecture, careful training, hyperparameter tuning)

**Relevant Work**:
- ASMNet: Action and Style-Conditioned Motion Generative Network ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860709/))
- Content-Preserving Motion Stylization using VAE (SIGGRAPH 2023)

---

## 2. Domain Generalization Techniques

### 2.1 Meta-Learning: MLDG (Meta-Learning for Domain Generalization)

**Concept**: Simulate domain shift during training by splitting available signers into meta-train and meta-test sets. Optimize model to generalize across this simulated shift.

**Approach (MLDG / MAML-style)**:
- In each episode: sample 3 signers as "meta-train," hold out 1 signer as "meta-test"
- Inner loop: compute gradient on meta-train signers, take a step
- Outer loop: evaluate on meta-test signer, backpropagate through the inner update
- This explicitly optimizes for cross-signer generalization

**Key Results**:
- MLDG (Li et al., 2018) inspired by MAML, trains models for OOD generalization
- 2024 survey finds meta-learning is among the most effective DG approaches
- Bi-level Meta-learning for Few-Shot Domain Generalization (CVPR 2023) specifically addresses few-domain settings

**Expected Impact**: MEDIUM-HIGH. Directly optimizes for cross-signer generalization. However, with only 4 signers, the meta-train/meta-test split is very small (3/1), limiting the diversity of simulated shifts.

**Implementation Effort**: MEDIUM (requires second-order gradients or first-order approximation; careful LR tuning)

**Relevant Work**:
- [Domain Generalization through Meta-Learning: A Survey](https://arxiv.org/abs/2404.02785) (2024)
- [Bi-Level Meta-Learning for Few-Shot DG](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Bi-Level_Meta-Learning_for_Few-Shot_Domain_Generalization_CVPR_2023_paper.pdf) (CVPR 2023)

### 2.2 Distribution Alignment: Deep CORAL + MMD

**Concept**: Minimize the distance between feature distributions of different signers, forcing the model to learn signer-invariant representations.

**Approach**:
- CORAL loss: align second-order statistics (covariance matrices) of features across signer domains
- MMD loss: minimize Maximum Mean Discrepancy between signer feature distributions in a reproducing kernel Hilbert space
- Add alignment loss to standard classification loss

**Key Results**:
- Deep CORAL is simple, effective, and widely used for domain adaptation
- Works well even with few source domains since it only requires feature statistics
- Already partially implemented in v22 via GRL (gradient reversal layer) for signer invariance

**Expected Impact**: MEDIUM. V22 already uses GRL for signer invariance. Adding CORAL/MMD provides a complementary alignment signal. However, with 4 signers, alignment may be too aggressive and collapse useful variation.

**Implementation Effort**: LOW (CORAL loss is ~10 lines of code)

**Relevant Work**:
- [Deep CORAL](https://arxiv.org/abs/1607.01719) (ECCV 2016 Workshop)
- [CORAL for Unsupervised Domain Adaptation](https://arxiv.org/abs/1612.01939)

### 2.3 Invariant Risk Minimization (IRM)

**Concept**: Learn features such that the optimal classifier on top of those features is the same across all training environments (signers).

**Approach**:
- Standard ERM minimizes average loss across all signers
- IRM adds a penalty: for each signer, the gradient of the loss w.r.t. a dummy scalar classifier should be zero
- This forces features to be equally predictive in every signer's distribution

**Key Results**:
- IRM (Arjovsky et al., 2019) is theoretically elegant but practically challenging
- IRM struggles with few environments (< 5-10 typically needed)
- Recent TV-based IRM variants (ICML 2024) improve robustness but still need sufficient environment diversity
- **With only 4 signers, IRM is likely to underperform** simpler methods

**Expected Impact**: LOW. Theoretically sound but impractical with 4 source domains. IRM requires sufficient environment diversity to identify invariant features.

**Implementation Effort**: MEDIUM (IRM penalty is straightforward; tuning the penalty weight is tricky)

**Relevant Work**:
- [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- [IRM Is A Total Variation Model](https://arxiv.org/abs/2405.01389) (ICML 2024)

---

## 3. Advanced Augmentation Techniques

### 3.1 SkeletonMix / Joint Mixing Data Augmentation (JMDA)

**Concept**: Mix-based augmentation specifically designed for skeleton data, analogous to CutMix/MixUp but respecting skeleton topology.

**Approaches**:
- **SkeletonMix** (2024): Merge upper body of one sample with lower body of another. Non-learning, plug-and-play.
- **JMDA** (ACM TOMM 2024): Includes TemporalMix (mix temporal segments) and SpatialMix (mix spatial joint groups). More fine-grained than SkeletonMix.
- **Skeleton-CutMix** (IEEE 2023): Probabilistic bone exchange for supervised domain adaptation.

**Key Results**:
- SkeletonMix and JMDA improve skeleton-based action recognition accuracy on NTU-RGB+D benchmarks
- CutMix alone is too destructive for skeleton topology; SkeletonMix and JMDA preserve structure
- Skeleton-CutMix specifically targets **cross-domain** adaptation

**Expected Impact**: MEDIUM-HIGH. Low-cost augmentation that directly increases training diversity. For sign language, upper/lower body mixing is less applicable (signs are primarily upper body), but temporal mixing and spatial joint-group mixing remain valuable.

**Implementation Effort**: LOW (SkeletonMix: ~20 lines; JMDA: moderate)

**Relevant Work**:
- [SkeletonMix](https://ieeexplore.ieee.org/document/10888125/) (IEEE 2024)
- [JMDA](https://dl.acm.org/doi/10.1145/3700878) (ACM TOMM 2024)
- [Skeleton-CutMix](https://ieeexplore.ieee.org/document/10183836/) (IEEE 2023)

### 3.2 Adversarial Augmentation on Skeleton Coordinates

**Concept**: Apply targeted perturbations to skeleton coordinates based on model gradients to create harder training examples.

**Approach**:
- Forward pass: compute loss for current batch
- Compute gradient of loss w.r.t. input skeleton coordinates
- Add epsilon * sign(gradient) to skeleton coordinates (FGSM-style)
- Train on both clean and perturbed samples
- Constrain perturbation magnitude to maintain physical plausibility

**Expected Impact**: MEDIUM. Forces the model to be robust to small deformations. Acts as a regularizer. Simple and effective. May not capture realistic signer-level variations.

**Implementation Effort**: LOW (~15 lines of code, no architectural changes)

### 3.3 Temporal Speed Perturbation

**Concept**: Vary the signing speed to simulate different signers' tempos.

**Approach**:
- Time-warp: non-linearly stretch/compress temporal segments
- Speed jitter: randomly speed up or slow down signing by 0.8x-1.2x
- Temporal dropout: randomly drop frames to simulate faster signing
- Key: maintain frame count via interpolation

**Expected Impact**: MEDIUM. Speed variation is a known signer-specific trait. Easy to implement. Already partially covered by standard temporal augmentation.

**Implementation Effort**: LOW (~10 lines of code)

### 3.4 Spatial Amplitude & Offset Perturbation

**Concept**: Vary the signing amplitude (how big/small gestures are) and spatial offset (where in space signs are performed) to simulate signer body differences.

**Approach**:
- Scale all joint coordinates by a random factor per-sample (0.85-1.15x)
- Add random per-joint Gaussian noise (~1-3% of coordinate range)
- Apply random translation offset to the entire skeleton
- Apply random slight rotation (~+/-5 degrees) around vertical axis

**Expected Impact**: MEDIUM. Simulates signers with different arm lengths, signing spaces, and camera angles. Simple but may already be captured by v22's existing augmentation.

**Implementation Effort**: LOW

---

## 4. What Works with Very Small Source Domains (4 signers)?

### Key Insights from Literature

1. **Data augmentation > algorithmic sophistication**: With few source domains, simple augmentation strategies that increase training diversity tend to outperform complex DG algorithms. The 2024 survey on data augmentation in DG ([Springer](https://link.springer.com/article/10.1007/s11063-025-11747-9)) confirms this.

2. **Feature-level augmentation (MixStyle) scales well to few domains**: Unlike data-level augmentation, MixStyle creates new domains in feature space and works with as few as 2-3 source domains.

3. **Meta-learning (MLDG) works but is noisy with 4 domains**: The 3-vs-1 leave-one-out split provides limited diversity per episode. Mitigation: use multiple random splits per batch.

4. **IRM needs >5 environments typically**: Not recommended for our setting.

5. **Ensemble of simple DG methods often wins**: Combining MixStyle + adversarial augmentation + GRL typically outperforms any single complex method.

6. **The biggest gains come from more domains (signers)**, not better algorithms. The [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Galappaththige_Towards_Generalizing_to_Unseen_Domains_with_Few_Labels_CVPR_2024_paper.pdf) on "Generalizing to Unseen Domains with Few Labels" confirms that even a small increase in domain diversity has outsized impact.

### Recommended Strategy for 4 Signers

The most promising approach is a **layered augmentation stack** that addresses different aspects of signer variation:

| Layer | Technique | What it addresses | Effort |
|-------|-----------|-------------------|--------|
| Input | Bone-length retargeting | Morphological variation | Low |
| Input | Speed/amplitude perturbation | Tempo and spatial differences | Low |
| Input | Adversarial skeleton perturbation | Model-specific weaknesses | Low |
| Feature | MixStyle (after GCN layer 1-2) | Signer style distribution | Low |
| Loss | GRL (already in v22) | Signer discriminability | Done |
| Loss | CORAL or MMD (optional) | Feature distribution alignment | Low |
| Training | MLDG episodic training | Cross-signer optimization | Medium |

---

## 5. Sign-Language-Specific Research

### 5.1 SAM-SLR (CVPR 2021 Challenge Winner)
- Skeleton Aware Multi-modal Sign Language Recognition
- Won 1st place in CVPR 2021 Large Scale Signer Independent Isolated SLR Challenge
- Uses multi-stream (joint, bone, joint motion, bone motion) with ensemble
- Key insight: skeleton-based features are inherently more person-independent than RGB
- [Paper](https://arxiv.org/abs/2110.06161) | [GitHub](https://github.com/jackyjsy/CVPR21Chal-SLR)

### 5.2 Signer Diversity-Driven Data Augmentation (NAACL 2024)
- Two augmentation strategies: adversarial training + diffusion model editing
- Consistency loss and discrimination loss for signer-independent features
- Achieves SOTA signer-independent SLT **without signer ID labels**
- **Directly applicable to our problem**
- [Paper](https://aclanthology.org/2024.findings-naacl.140/)

### 5.3 AVSN: Adversarial Vulnerability-Seeking Networks (IEEE 2024)
- Jointly trains data augmentation and classification for skeleton-based SLR
- Uses adversarial learning to find and augment vulnerable skeleton patterns
- [Paper](https://ieeexplore.ieee.org/document/10718297/)

### 5.4 SC2SLR: Skeleton-based Contrast for Sign Language Recognition (2024)
- Contrastive learning framework specifically for skeleton-based SLR
- Learns representations invariant to signer-specific variations through contrastive objectives
- [Paper](https://dl.acm.org/doi/10.1145/3670105.3670173)

### 5.5 Fuzzy Encoding for Universal SLR (Wei et al., InfoMat 2024)
- Achieves 80% sentence recognition accuracy on new untrained users
- Uses "fuzzy encoding" to abstract away individual action differences
- Trained on single user's word-level data, generalizes to new users at sentence level
- [Paper](https://doi.org/10.1002/inf2.12642)

### 5.6 Cross-Dataset Skeleton Action Recognition (NeurIPS 2024)
- "Recovering Complete Actions for Cross-dataset Skeleton Action Recognition"
- Observe temporal mismatch across datasets; recover complete actions and resample
- Augmentation framework for unseen domains
- [Paper](https://papers.nips.cc/paper_files/paper/2024/file/a78f142aec481e68c75276756e0a0d91-Paper-Conference.pdf)

---

## 6. Ranked Recommendations for V24

### Tier 1: High Impact, Low Effort (implement first)

| # | Technique | Expected Gain | Effort | Notes |
|---|-----------|---------------|--------|-------|
| 1 | **MixStyle after GCN layers 1-2** | +3-8% real | 2-3 hours | Best DG technique for few domains. Insert after first 1-2 ST-GCN blocks. Mix feature statistics across signers. |
| 2 | **Bone-length retargeting augmentation** | +2-5% real | 1-2 hours | Extract bone lengths per signer, generate 10-20 virtual signer morphologies via interpolation/sampling. |
| 3 | **Adversarial skeleton perturbation (FGSM)** | +2-4% real | 2-3 hours | Add epsilon*sign(grad) perturbation to skeleton input. Combine with clean samples. |
| 4 | **Speed/amplitude jitter** | +1-3% real | 1 hour | Random speed warp (0.8-1.2x), amplitude scale (0.85-1.15x), and spatial offset per sample. |

### Tier 2: Medium Impact, Medium Effort

| # | Technique | Expected Gain | Effort | Notes |
|---|-----------|---------------|--------|-------|
| 5 | **MLDG episodic training** | +2-5% real | 4-6 hours | Leave-one-signer-out meta-learning. Requires second-order gradient (or first-order FOMAML approximation). |
| 6 | **SkeletonMix / JMDA** | +1-3% real | 3-4 hours | Temporal segment mixing and spatial joint-group mixing. Adapt for upper-body-focused signs. |
| 7 | **CORAL loss on signer features** | +1-2% real | 2 hours | Complement existing GRL with explicit covariance alignment across signers. |
| 8 | **Contrastive signer-invariant learning** | +2-4% real | 4-6 hours | Pull same-class samples from different signers together, push different classes apart. |

### Tier 3: Higher Effort, Uncertain Return

| # | Technique | Expected Gain | Effort | Notes |
|---|-----------|---------------|--------|-------|
| 9 | **CVAE signer generation** | +2-6% real? | 8-12 hours | Train conditional VAE to generate new signer sequences. High variance with 4 signers. |
| 10 | **IRM penalty** | +0-2% real | 3-4 hours | Likely insufficient environments. Try only if other methods plateau. |

### Recommended V24 Configuration

Combine Tier 1 techniques (1-4) as a **stacked augmentation pipeline**:

```
Input: skeleton sequence (T, N, C)
  |-> Bone-length retargeting (random virtual morphology)
  |-> Speed warp (0.8-1.2x random)
  |-> Amplitude scale (0.85-1.15x random)
  |-> Adversarial perturbation (FGSM, epsilon=0.01)
  |
  v
ST-GCN Block 1
  |-> MixStyle (mix feature stats across signers, p=0.5)
  |
ST-GCN Block 2
  |-> MixStyle (mix feature stats across signers, p=0.5)
  |
ST-GCN Blocks 3-N
  |
  v
Classification head + GRL signer head (from v22)
```

**Training objective**:
```
L = L_cls + lambda_grl * L_grl + lambda_coral * L_coral (optional)
```

This combines:
- **Input-level diversity**: bone retargeting + speed/amplitude + adversarial (physical signer variation)
- **Feature-level diversity**: MixStyle (statistical signer style mixing)
- **Representation constraint**: GRL + optional CORAL (signer invariance)

### Expected Outcome

Optimistic: +5-10% real accuracy (reaching ~45-50%)
Realistic: +3-7% real accuracy (reaching ~43-47%)
Pessimistic: +0-3% (diminishing returns from 4-signer ceiling)

The honest assessment: these techniques can squeeze more from the existing data, but the **4-signer ceiling** remains the fundamental bottleneck. Even the best DG methods struggle with <5 source domains. Collecting 3-6 more signers would likely provide more improvement than any algorithmic change.

---

## References

1. Zhou et al. "Domain Generalization with MixStyle" ICLR 2021. https://arxiv.org/abs/2104.02008
2. Fu et al. "Signer Diversity-driven Data Augmentation for Signer-Independent SLT" NAACL 2024. https://aclanthology.org/2024.findings-naacl.140/
3. Jiang et al. "Skeleton Aware Multi-modal Sign Language Recognition" CVPR 2021W. https://arxiv.org/abs/2103.08833
4. AVSN: Skeleton-Based Data Augmentation for SLR Using Adversarial Learning. IEEE 2024. https://ieeexplore.ieee.org/document/10718297/
5. SkeletonMix: A Mixup-Based Data Augmentation. IEEE 2024. https://ieeexplore.ieee.org/document/10888125/
6. JMDA: Joint Mixing Data Augmentation. ACM TOMM 2024. https://dl.acm.org/doi/10.1145/3700878
7. Skeleton-CutMix: Probabilistic Bone Exchange. IEEE 2023. https://ieeexplore.ieee.org/document/10183836/
8. Li et al. "Learning to Generalize: Meta-Learning for Domain Generalization" AAAI 2018.
9. Arjovsky et al. "Invariant Risk Minimization" 2019. https://arxiv.org/abs/1907.02893
10. Sun & Saenko. "Deep CORAL" ECCV 2016W. https://arxiv.org/abs/1607.01719
11. Meta-Learning for DG Survey 2024. https://arxiv.org/abs/2404.02785
12. Wei et al. "Multimodal Sensing and Fuzzy Encoding for Universal SLR" InfoMat 2024. https://doi.org/10.1002/inf2.12642
13. Recovering Complete Actions for Cross-dataset Skeleton AR. NeurIPS 2024.
14. Data Augmentation in Domain Generalization Survey 2025. https://link.springer.com/article/10.1007/s11063-025-11747-9
15. Yamashita et al. "MixStyle-Based Contrastive TTA" CVPR 2024W.
