# V21 Research: Data Augmentation & Synthetic Data for Small Skeleton Datasets

**Researcher**: augmentation-researcher
**Date**: 2026-02-09
**Context**: KSL project -- 30 classes, 4 unique signers, ~600 samples total, MediaPipe 75 keypoints
**Current augmentations (v20)**: spatial jitter N(0,0.01), speed 0.9x/1.1x, rotation +/-5deg, bone-length perturbation, temporal warping, MixUp
**V20 real-tester accuracy**: 30.5% (val 86.9%)

---

## 1. Skeleton Data Augmentation Techniques for Cross-Subject Generalization

### 1.1 Comprehensive Augmentation Taxonomy

Based on AimCLR (Guo et al., AAAI 2022) and subsequent works, skeleton augmentations fall into four categories:

**Spatial Augmentations (geometry-preserving):**
| Technique | Description | Typical Range | Cross-Subject Benefit |
|-----------|-------------|---------------|----------------------|
| **Shear** | Apply shear matrix to joint coords | s ~ U(-0.5, 0.5) | HIGH - simulates viewpoint changes |
| **Rotation** | Random 3D rotation | +/-15-30 deg per axis | HIGH - camera angle invariance |
| **Spatial Flip** | Mirror skeleton L/R | p=0.5 | MEDIUM - but be careful with handedness in sign language |
| **Axis Mask** | Zero out one coordinate axis (X/Y/Z) | p=0.33 per axis | MEDIUM - forces learning from partial info |
| **Joint Mask / Dropout** | Zero out random joints | 10-30% of joints | HIGH - occlusion robustness |
| **Scaling** | Uniform or per-limb scaling | s ~ U(0.8, 1.2) | HIGH - body proportion invariance |
| **Translation** | Random offset to all joints | t ~ N(0, 0.1) | LOW - usually already normalized |

**Temporal Augmentations:**
| Technique | Description | Typical Range | Cross-Subject Benefit |
|-----------|-------------|---------------|----------------------|
| **Temporal Crop** | Random sub-sequence selection | 50-90% of frames | HIGH - speed invariance |
| **Temporal Flip (Reversal)** | Reverse the time axis | p=0.5 | MEDIUM - careful with sequential signs |
| **Speed Perturbation** | Resample at different rate | 0.5x-2.0x | HIGH - signer speed invariance |
| **Time Warping** | Non-linear temporal distortion | sigma=0.2 | HIGH - natural speed variation |
| **Frame Dropping** | Randomly drop frames | 10-30% frames | MEDIUM - robustness to frame loss |

**Spatiotemporal Augmentations:**
| Technique | Description | Typical Range | Cross-Subject Benefit |
|-----------|-------------|---------------|----------------------|
| **Gaussian Noise** | Add noise to all coords | sigma=0.01-0.05 | MEDIUM - sensor noise invariance |
| **Gaussian Blur (temporal)** | Smooth joint trajectories | kernel=3-7 | LOW-MEDIUM |
| **Bone Length Perturbation** | Randomize limb lengths | +/-10-20% | HIGH - body shape invariance |

**Mixing Augmentations:**
| Technique | Description | Typical Range | Cross-Subject Benefit |
|-----------|-------------|---------------|----------------------|
| **MixUp** | Linear interpolation of samples+labels | alpha=0.2-1.0 | MEDIUM - regularization |
| **SkeletonMix** | Swap body parts between samples | upper/lower body | HIGH - body-part diversity |
| **CutMix (temporal)** | Swap temporal segments | random segment | MEDIUM |
| **SkeleMixCLR** | Contrastive learning with mixed skeletons | -- | HIGH (for SSL) |

### 1.2 What Our V20 is Missing (GAP ANALYSIS)

**Currently using (v20):**
- Spatial jitter N(0,0.01) -- TOO SMALL, increase to 0.02-0.05
- Speed 0.9x/1.1x -- TOO NARROW, use 0.5x-2.0x
- Rotation +/-5deg -- TOO NARROW, use +/-15-30deg
- Bone length perturbation -- GOOD, keep
- Temporal warping -- GOOD, keep
- MixUp -- GOOD, keep

**Not using but should add (HIGH PRIORITY):**
1. **Shear augmentation** -- simulates viewpoint variation, very effective for cross-subject
2. **Joint/body-part masking** -- forces model to use redundant joint information
3. **Wider rotation range** (+/-15-30deg)
4. **Wider speed range** (0.5x-2.0x)
5. **SkeletonMix** (body-part swapping between samples)
6. **Axis masking** -- zero out one coordinate dimension randomly
7. **Stronger spatial jitter** (sigma=0.03-0.05)

**Not using, MEDIUM priority:**
8. Temporal flip (careful -- some signs are direction-dependent)
9. Frame dropping (10-20%)
10. Per-limb independent scaling

### 1.3 Recommended "Extreme" Augmentation Pipeline

Following AimCLR's approach of using extreme augmentation combined with contrastive learning:

```
For each training sample:
  1. Random rotation: uniform(-30, +30) on all 3 axes
  2. Random shear: uniform(-0.5, +0.5) on 2 random axes
  3. Random scaling: uniform(0.8, 1.2) global + uniform(0.9, 1.1) per-limb
  4. Joint dropout: mask 15% of joints (set to 0)
  5. Axis mask: 10% chance of zeroing X, Y, or Z axis
  6. Gaussian jitter: N(0, 0.03) per coordinate
  7. Speed perturbation: uniform(0.5, 2.0) via interpolation
  8. Temporal crop: keep 60-100% of frames
  9. Bone length perturbation: +/-15%
  10. MixUp or SkeletonMix: p=0.3
```

---

## 2. GANs/VAEs/Diffusion Models for Synthetic Skeleton Sequences

### 2.1 Generative Approaches Overview

| Method | Pros | Cons | Practicality for KSL |
|--------|------|------|---------------------|
| **GAN (skeleton)** | High-quality samples, adversarial training | Mode collapse, training instability with <1000 samples | LOW - too little data to train a GAN |
| **VAE (skeleton)** | Stable training, latent space interpolation | Blurry outputs, may not capture fine hand details | MEDIUM - simpler to train than GAN |
| **Diffusion (MDM, MotionDiffuse)** | SOTA quality, multimodal distributions, no mode collapse | Very expensive to train, needs thousands of samples | LOW - overkill for our dataset size |
| **CVAE conditioned on class** | Class-conditional generation, moderate data needs | Still needs ~50+ samples per class | MEDIUM - we have ~20/class |

### 2.2 Practical Assessment for KSL (~600 samples, 30 classes)

**The honest assessment: Generative models are NOT practical for our dataset size.**

Reasons:
- GANs need ~1000+ samples per class for stable training; we have ~20
- Diffusion models (MDM, MotionDiffuse) are trained on datasets with 10,000+ sequences
- VAEs may work but quality will be poor with so few examples per class
- Training a generative model would likely just memorize the 4 signers

**What COULD work:**
1. **Simple interpolation-based synthesis** (like NVIDIA's SynthDa): Interpolate between existing skeleton sequences of the same class to create new samples. This is deterministic and doesn't need a trained generative model.
2. **Class-conditional noise injection**: Add structured noise to existing samples that respects skeleton topology (bone lengths, joint angle limits).
3. **Signer style transfer via affine mapping**: Learn a simple affine transform between two signers' average skeletons, then apply it to transfer one signer's samples to look like they were performed by another signer.

### 2.3 Recommended Approach: Motion Interpolation + Style Transfer

```python
# Pseudo-code for lightweight synthetic data
def generate_synthetic(sample_a, sample_b, alpha=0.5):
    """Interpolate between two samples of the same class"""
    # Temporal alignment via DTW
    aligned_a, aligned_b = dtw_align(sample_a, sample_b)
    # Interpolate in bone-angle space (not raw XYZ)
    angles_a = xyz_to_bone_angles(aligned_a)
    angles_b = xyz_to_bone_angles(aligned_b)
    synthetic_angles = alpha * angles_a + (1 - alpha) * angles_b
    # Add small random perturbation
    synthetic_angles += np.random.normal(0, 0.02, synthetic_angles.shape)
    return bone_angles_to_xyz(synthetic_angles)
```

---

## 3. Self-Supervised Pretraining Methods for Skeleton Data

### 3.1 Key Methods

**AimCLR** (Guo et al., AAAI 2022):
- Uses extreme augmentations (shear, spatial flip, rotate, axis mask, crop, temporal flip, gaussian noise/blur)
- Energy-based Attention-guided Drop Module (EADM) for diverse positive pairs
- Nearest Neighbors Mining (NNM) to expand positive samples
- Results: 68.2% on NTU-60 xsub (linear eval), significant improvement over SkeletonCLR
- **Relevance to KSL**: The augmentation pipeline is directly applicable even without the contrastive learning framework

**CrosSCLR / 3s-CrosSCLR** (Li et al., CVPR 2021):
- Cross-view contrastive learning across joint/bone/motion streams
- Single-view SkeletonCLR + Cross-View Consistency Knowledge Mining (CVC-KM)
- 3s = three-stream (joint + motion + bone) weighted averaging
- Results: ~75% on NTU-60 xsub (linear eval)
- **Relevance to KSL**: The multi-stream concept (joint/bone/motion) is valuable; the cross-stream contrastive idea can be adapted

**ISC (Inter-Skeleton Contrastive Learning)** (Thoker et al., 2022):
- Cross-contrastive learning across multiple skeleton representations
- Skeleton-specific spatial and temporal augmentations
- Learns spatio-temporal dynamics from multiple input formats
- **Relevance to KSL**: Confirms that using multiple skeleton representations (raw + bone + velocity) in a contrastive framework is beneficial

**STARS** (Mehraban et al., 2024):
- Self-supervised tuning: masked prediction stage (encoder-decoder) + nearest-neighbor contrastive tuning
- Achieves SOTA self-supervised results WITHOUT hand-crafted augmentations
- Significantly better in few-shot settings (5-shot, 10-shot)
- **Relevance to KSL**: VERY RELEVANT -- designed for low-data regimes, no augmentation engineering needed

### 3.2 Practical SSL Strategy for KSL

**Option A: Pretrain on NTU-120, fine-tune on KSL** (RECOMMENDED)
- Use a pretrained skeleton encoder (e.g., from STARS or 3s-CrosSCLR on NTU-120)
- Fine-tune the encoder on our 600 KSL samples
- This gives us the benefit of a large-scale pretrained representation
- Challenge: NTU uses 25 Kinect joints vs our 75 MediaPipe joints (need joint mapping)

**Option B: SSL on our own data**
- With only 600 samples, contrastive pretraining may not help much
- But STARS-style masked prediction could still learn useful temporal patterns
- Combine with extreme augmentations to increase effective diversity

**Option C: Cross-dataset pretraining on sign language datasets**
- Use WLASL, AUTSL, or other sign language skeleton datasets for pretraining
- More domain-relevant than NTU (action recognition)
- Then fine-tune on KSL

**Recommendation**: Option A or C, depending on available pretrained models.

---

## 4. MixUp/CutMix for Skeleton Sequences

### 4.1 Standard MixUp for Skeletons

Standard MixUp directly interpolates skeleton coordinates:
```
x_mix = lambda * x_a + (1 - lambda) * x_b
y_mix = lambda * y_a + (1 - lambda) * y_b
```
where lambda ~ Beta(alpha, alpha)

**Problem**: Direct MixUp severely deforms skeleton topology. An interpolated skeleton between two different poses creates anatomically impossible poses (e.g., arms bending the wrong way, joints in impossible positions).

### 4.2 SkeletonMix (Better for Skeletons)

SkeletonMix (2022) swaps body parts instead of interpolating:
- Take upper body from sample A and lower body from sample B
- Preserves skeleton topology within each part
- Label is weighted by the number of joints from each sample

**Improvement**: ~1-2% on NTU-60/120 when added to GCN-based models

### 4.3 Joint Mixing Data Augmentation (JMDA, 2024)

More sophisticated than SkeletonMix:
- Spatial Mixing (SM): preserves semantic information better than SkeletonMix
- Temporal Mixing (TM): swaps temporal segments instead of body parts
- Combined SM+TM achieves better results than either alone

**Improvement**: +1.5-3% on NTU benchmarks (larger gains on smaller datasets)

### 4.4 Recommendations for KSL

1. **Replace raw MixUp with SkeletonMix**: Swap hand regions between samples (more relevant for sign language than upper/lower body split)
2. **Use temporal CutMix**: Swap temporal segments to create sequences with mixed timing
3. **Sign-language-aware mixing**: Mix LEFT HAND from one sample with RIGHT HAND from another (since many KSL signs are two-handed)
4. **Alpha parameter**: Use Beta(0.4, 0.4) for MixUp -- gives more extreme mixing which helps regularization with tiny datasets

---

## 5. Extreme Augmentation Strategies

### 5.1 AimCLR's Extreme Augmentation Framework

AimCLR classifies augmentations as "normal" and "extreme":

**Normal (mild)**:
- Shear: s ~ U(-0.5, 0.5) applied as matrix multiplication
- Crop: random temporal cropping of 50-90%

**Extreme (aggressive)**:
- Spatial Flip: mirror the skeleton L/R
- Rotate: large random 3D rotations (up to +/-180 deg)
- Axis Mask: completely zero out one coordinate axis
- Temporal Flip: reverse the entire sequence
- Gaussian Noise: sigma=0.1 (10x larger than typical)
- Gaussian Blur: temporal smoothing with large kernel

**Key insight**: Extreme augmentations break the identity of the original sample. AimCLR handles this by using EADM (Energy-based Attention-guided Drop) to selectively attend to the parts that still carry discriminative information, and NNM (Nearest Neighbor Mining) to find true positive pairs even among heavily augmented samples.

### 5.2 Adaptation for KSL Sign Language

For sign language specifically, some augmentations need special care:

**SAFE to apply aggressively**:
- Shear (any magnitude)
- Scaling (0.5x-2.0x)
- Gaussian noise (up to sigma=0.05)
- Speed perturbation (0.5x-2.0x)
- Bone length perturbation (+/-20%)
- Joint dropout (up to 30%)
- Axis masking

**APPLY WITH CAUTION**:
- Spatial flip (L/R mirror) -- DANGEROUS for sign language! Many signs are hand-specific. Only use if labels are symmetric.
- Temporal flip -- Some signs have directional motion (e.g., "come" vs "go"). Avoid unless sign is symmetric in time.
- Large rotation -- OK around Y-axis (facing direction), careful on X/Z (could create upside-down hands)

**KSL-specific augmentations to ADD**:
- **Hand-focus jitter**: Apply extra noise to hand joints (sigma=0.05) since hand shape is the most variable across signers
- **Wrist-to-hand scaling**: Independently scale hand size relative to body (simulates different hand/body proportions)
- **Signing space translation**: Shift the signing space up/down/left/right (signers sign at different heights)

---

## 6. Domain Randomization for Skeleton Data

### 6.1 What to Randomize

Domain randomization creates diverse synthetic examples by randomly varying properties that differ between signers:

**Anatomical Properties (signer-specific):**
| Property | Randomization | Range |
|----------|--------------|-------|
| Arm length | Scale humerus/forearm bones | 0.8-1.2x |
| Hand size | Scale all hand bones | 0.7-1.3x |
| Shoulder width | Scale clavicle length | 0.85-1.15x |
| Torso height | Scale spine segments | 0.9-1.1x |
| Finger proportions | Per-finger scaling | 0.8-1.2x |

**Behavioral Properties (signer-style):**
| Property | Randomization | Range |
|----------|--------------|-------|
| Signing speed | Temporal scaling | 0.5-2.0x |
| Signing amplitude | Scale distance from body center | 0.7-1.5x |
| Signing height | Vertical offset of signing space | +/-0.1 units |
| Handedness emphasis | Scale dominant/non-dominant hand differently | 0.8-1.2x ratio |
| Wrist angle bias | Add constant rotation to wrist | +/-15 deg |

**Sensor/Capture Properties:**
| Property | Randomization | Range |
|----------|--------------|-------|
| Camera angle | Rotate entire skeleton around Y | +/-30 deg |
| Skeleton noise | Add per-joint gaussian noise | sigma=0.01-0.05 |
| Tracking jitter | Add temporal noise (simulates MediaPipe variance) | sigma=0.02 |
| Missing landmarks | Randomly drop joints for some frames | 5-15% |

### 6.2 Implementation Strategy

```python
def domain_randomize_skeleton(skeleton, config):
    """Apply domain randomization to a skeleton sequence."""
    s = skeleton.copy()

    # 1. Anatomical randomization
    arm_scale = np.random.uniform(0.8, 1.2)
    hand_scale = np.random.uniform(0.7, 1.3)
    s = scale_bones(s, arm_indices, arm_scale)
    s = scale_bones(s, hand_indices, hand_scale)

    # 2. Behavioral randomization
    speed = np.random.uniform(0.5, 2.0)
    s = temporal_resample(s, speed)
    amplitude = np.random.uniform(0.7, 1.5)
    s = scale_from_center(s, amplitude)

    # 3. Viewpoint randomization
    angle = np.random.uniform(-30, 30) * np.pi / 180
    s = rotate_y(s, angle)

    # 4. Sensor noise
    s += np.random.normal(0, 0.03, s.shape)

    # 5. Random joint dropout
    drop_mask = np.random.random(s.shape[1]) < 0.15
    s[:, drop_mask, :] = 0

    return s
```

---

## 7. Style Transfer / Signer Style Mixing

### 7.1 Approaches

**A. Simple Affine Style Transfer:**
1. Compute mean skeleton pose per signer (average over all their samples)
2. Learn affine transform from signer A's mean to signer B's mean
3. Apply this transform to signer A's individual samples to create "signer-B-style" versions

```python
# Compute signer means
mean_A = np.mean(signer_A_samples, axis=0)  # [T, J, 3]
mean_B = np.mean(signer_B_samples, axis=0)  # [T, J, 3]

# For each joint, learn scale+offset: B = scale * A + offset
for j in range(num_joints):
    scale[j] = std(B[:, j]) / std(A[:, j])
    offset[j] = mean(B[:, j]) - scale[j] * mean(A[:, j])

# Apply to sample from signer A
for sample in signer_A_samples:
    synthetic = scale * sample + offset  # Now "looks like" signer B
```

**B. Bone-Space Style Transfer (RECOMMENDED):**
1. Convert all skeletons to bone representation (parent-relative vectors)
2. Compute per-signer bone length statistics (mean, std)
3. To transfer style: keep bone directions from source signer, use bone lengths from target signer
4. This preserves the MOTION while changing the BODY PROPORTIONS

**C. Motion Puzzle Style Transfer:**
- Transfer style of individual body parts independently
- Can mix hand motion from one signer with arm motion from another
- Creates more diverse combinations than whole-body transfer

### 7.2 Expected Impact

With 4 signers, pairwise style transfer creates:
- 4 signers x 3 target styles = 12 style-transferred versions per sample
- Effectively 4x the signer diversity (though still synthetic)
- Combined with augmentation, this creates much more variation

### 7.3 Key Considerations for Sign Language

- **Hand shape is critical**: Style transfer must preserve hand configurations while changing body proportions
- **Bone-space transfer is preferred**: Changing bone lengths while keeping joint angles/directions preserves the actual sign semantics
- **Don't transfer between left/right dominant signers**: This could create impossible signs

---

## 8. Best Practices for Tiny Datasets (<1000 samples)

### 8.1 Data Strategy

1. **Do NOT over-deduplicate**: V20's aggressive dedup (750->240) was catastrophic. Light dedup only for truly identical frames.

2. **Maximize augmentation diversity**: Use ALL applicable augmentations simultaneously with high probability. The goal is to make every training example look different.

3. **Use augmentation progressively**: Start with mild augmentation (epoch 0-20), then increase to extreme (epoch 20+). This curriculum approach helps the model learn basic patterns first.

4. **Prefer geometric augmentations over noise**: Scaling, rotation, shear create more meaningful variation than just adding noise.

5. **Domain randomization > synthetic generation**: With <1000 samples, rule-based domain randomization is more reliable than learned generative models.

### 8.2 Model Strategy

1. **Keep models small**: <500K parameters for 600 samples (ratio ~1000:1 params-to-samples is upper limit)
2. **Use strong regularization**:
   - Dropout: 0.3-0.5
   - Weight decay: 1e-3 to 1e-2
   - Label smoothing: 0.1-0.2
3. **Early stopping on validation loss** (not accuracy)
4. **Ensemble multiple small models** rather than one large model

### 8.3 Feature Engineering (MORE IMPORTANT THAN MODEL)

For tiny datasets, feature engineering matters MORE than model architecture:
1. **Bone features** (direction + length) -- signer-invariant by nature
2. **Joint angle features** -- invariant to body proportions
3. **Velocity and acceleration** -- captures motion dynamics
4. **Relative positions** -- joint-to-joint distances (invariant to absolute position)
5. **Hand shape descriptors** -- fingertip angles, palm orientation

### 8.4 Transfer Learning

1. **Pretrain on large skeleton datasets** (NTU-120: 114K samples, BABEL, AMASS)
2. **Fine-tune only the last few layers** on KSL data
3. **Use learned representations** rather than raw coordinates
4. **Cross-lingual SLR transfer**: Pretrain on larger sign language datasets (WLASL, AUTSL) then fine-tune on KSL

### 8.5 Evaluation Strategy

1. **Leave-one-signer-out** cross-validation is essential (not random split)
2. **Report per-signer accuracy** to identify signer-specific failures
3. **Use confidence calibration** -- reject low-confidence predictions (v20 showed 81% accuracy at >0.6 confidence)

---

## 9. Priority-Ranked Recommendations for V21

### P0 - Must Implement (Expected +5-15% real-tester accuracy)

1. **Massively increase augmentation ranges**:
   - Rotation: +/-5deg -> +/-30deg
   - Speed: 0.9-1.1x -> 0.5-2.0x
   - Jitter: sigma=0.01 -> 0.03-0.05

2. **Add missing high-value augmentations**:
   - Shear augmentation (s ~ U(-0.5, 0.5))
   - Joint/body-part dropout (15-25% of joints)
   - Axis masking (p=0.1 per axis)
   - SkeletonMix (hand region swapping)

3. **Undo aggressive dedup**: Use all ~750 training samples (or light dedup to ~650)

4. **Bone-space style transfer**: Create synthetic "new signers" by swapping bone lengths between existing signers while keeping motion

### P1 - Should Implement (Expected +2-5% additional)

5. **Domain randomization pipeline**: Randomize anatomical proportions (arm length, hand size, shoulder width) for each training sample

6. **Progressive augmentation curriculum**: Mild augmentation for first 30% of training, then ramp up to extreme

7. **Motion interpolation synthetic data**: Create new samples by DTW-aligning and interpolating between same-class examples in bone-angle space

8. **Sign-language-aware MixUp**: Replace standard MixUp with hand-region SkeletonMix

### P2 - Could Implement (Expected +1-3% additional)

9. **Self-supervised pretraining**: Pretrain encoder with masked prediction (STARS-style) on our data + any available sign language skeleton data

10. **Transfer learning from large skeleton datasets**: Fine-tune pretrained NTU-120 skeleton encoder (requires joint mapping)

11. **Test-time augmentation (TTA)**: Average predictions over multiple augmented versions of each test sample

12. **Confidence thresholding**: Reject predictions below a calibrated confidence threshold (v20 showed this works)

### P3 - Future Investigation

13. **Cross-lingual pretraining**: Pretrain on WLASL/AUTSL skeleton data
14. **Adversarial training**: Use adversarial examples as augmentation (PGD-based)
15. **Feature-space augmentation**: Augment in learned feature space rather than raw coordinates

---

## 10. Key References

- AimCLR: Guo et al., "Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition," AAAI 2022. [arXiv:2112.03590](https://arxiv.org/abs/2112.03590)
- CrosSCLR: Li et al., "Cross-Stream Contrastive Learning for Self-Supervised Skeleton-Based Action Recognition," 2023. [arXiv:2305.02324](https://arxiv.org/abs/2305.02324)
- STARS: Mehraban et al., "Self-supervised Tuning for 3D Action Recognition in Skeleton Sequences," 2024. [arXiv:2407.10935](https://arxiv.org/abs/2407.10935)
- SkeletonMix: "SkeletonMix: A Mixup-Based Data Augmentation Framework for Skeleton-Based Action Recognition," IEEE 2024. [IEEE Xplore](https://ieeexplore.ieee.org/document/10888125/)
- JMDA: "Joint Mixing Data Augmentation for Skeleton-based Action Recognition," ACM TOMM 2024. [ACM](https://dl.acm.org/doi/10.1145/3700878)
- SynthDa: NVIDIA, "Improving Synthetic Data Augmentation and Human Action Recognition with SynthDa," 2025. [NVIDIA Blog](https://developer.nvidia.com/blog/improving-synthetic-data-augmentation-and-human-action-recognition-with-synthda/)
- Skeleton augmentation survey: "Enhancing Human Action Recognition with 3D Skeleton Data: A Comprehensive Study of Deep Learning and Data Augmentation," Electronics 2024. [MDPI](https://www.mdpi.com/2079-9292/13/4/747)
- SL-SLR: "Self-Supervised Representation Learning for Sign Language Recognition," 2025. [arXiv:2509.05188](https://arxiv.org/html/2509.05188v1)
- Sign Language GAN augmentation: "Skeleton-Based Data Augmentation for Sign Language Recognition Using Adversarial Learning," 2024. [ResearchGate](https://www.researchgate.net/publication/384965181)
- Ethiopian SLR: "A deep learning framework for Ethiopian sign language recognition using skeleton-based representation," Scientific Reports 2025. [Nature](https://www.nature.com/articles/s41598-025-19937-0)
- Motion Style Transfer survey: PMC 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10007042/)
