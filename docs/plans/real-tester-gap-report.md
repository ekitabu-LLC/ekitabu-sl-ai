# KSL Real-Tester Generalization Gap: Root Cause Analysis & Remediation Plan

**Date**: 2026-02-09
**Version**: 1.0
**Scope**: Analysis of the 53-percentage-point accuracy drop from validation to real-world testers

---

## 1. Executive Summary

The KSL recognition system achieves 87.3% on held-out validation signers but collapses to 34.2% on three new real-world testers -- a 53-point gap that is more than double the 15-25% cross-signer drop reported in comparable sign language recognition literature. Root cause analysis reveals this is not a single failure but a compounding of at least seven distinct issues: the training set contains only 4 unique signers (not 5) due to byte-identical duplicates, the model is over-parameterized by 120x relative to dataset size, per-sample normalization amplifies signer-specific noise, MediaPipe version mismatch changes landmark geometry, the test set includes side-view video never seen in training, prediction collapse concentrates outputs on a handful of classes, and the feature pipeline discards bone/angular features that are inherently signer-invariant. Addressing the P0 and P1 fixes outlined below -- data deduplication, wrist-centric normalization, model shrinkage to ~300K params, bone features, and signer-mimicking augmentation -- should realistically bring real-tester accuracy to 55-65%, with further gains from multi-stream ensembles and adversarial training.

---

## 2. The Problem

### 2.1 Validation vs Real-World Accuracy

| Split      | Numbers | Words | Combined |
|------------|---------|-------|----------|
| Validation | 78.6%   | 95.9% | 87.3%    |
| Real Test  | 20.3%   | 48.1% | 34.2%    |
| **Gap**    | **-58.3** | **-47.8** | **-53.1** |

### 2.2 Per-Signer Real-Test Breakdown

| Signer | Numbers | Words | Notes |
|--------|---------|-------|-------|
| S1     | 31.0%   | ~55%  | Front-facing, reasonable quality |
| S2     | 6.7%    | ~57%  | Front-facing, low number accuracy |
| S3     | 13.3%   | 32.0% | **Side-view** for all numbers |

### 2.3 Per-Class Extremes

**Robust classes** (val -> real retains >50%):
- Teach: 100%, Friend: 83%, Sweater: 83%, class 100: 72% -> 75%

**Collapsed classes** (real = 0%):
- Market: 0%, Apple: 17%, Ugali: 17%
- Numbers: 9 of 15 classes at 0% on real testers

### 2.4 Prediction Collapse

Class 444 is predicted 21 out of 59 times for numbers (35.6%), indicating the model has learned a degenerate decision boundary that maps most number inputs to a single class.

---

## 3. Root Cause Analysis

Root causes are ranked by estimated contribution to the generalization gap.

### 3.1 CRITICAL: Training Data Contains Only 4 Unique Signers

**Evidence**: Signers 2 and 3 in the training set are byte-for-byte identical across ALL classes and ALL repetitions. This means:
- Effective unique training samples: **600**, not 750
- Effective training signers: **4**, not 5
- The model sees even less signer diversity than intended

**Impact**: With only 4 unique signers for training, the model has extremely limited exposure to inter-signer variation. LOSO cross-validation confirms this:

| Left-Out Signer | Accuracy | Notes |
|-----------------|----------|-------|
| Signer 1        | 68.7%    | Moderate |
| Signer 2        | 99.3%    | Duplicate of Signer 3 -- trivially predicted |
| Signer 3        | 100.0%   | Duplicate of Signer 2 -- trivially predicted |
| Signer 4        | 28.7%    | Poor -- true cross-signer difficulty |
| Signer 5        | 16.0%    | Very poor -- most dissimilar signer |
| **Average**     | **62.5%** | Inflated by duplicate pair |

**True cross-signer accuracy** (excluding duplicates): **(68.7 + 28.7 + 16.0) / 3 = 37.8%** -- almost exactly what we see on real testers.

**Severity**: CRITICAL. This is the single largest contributor to the gap.

### 3.2 CRITICAL: Model Over-Parameterization

**Evidence**: 4.5M parameters trained on ~300 unique samples (after deduplication) = **15,000 parameters per training sample**. Literature recommends a ratio below 100:1.

**Impact**: The model memorizes signer-specific patterns rather than learning generalizable sign representations. Training accuracy reaches 99.7% while validation sits at 53-78%, a textbook overfitting signature. The high-confidence wrong predictions (mean confidence 0.261 vs 0.391 for correct) show the model is confident even when wrong -- it has memorized training-distribution artifacts.

**Severity**: CRITICAL.

### 3.3 HIGH: Per-Sample Normalization Amplifies Signer Noise

**Evidence**: Current normalization divides coordinates by `max(abs(coords))` per sample. This value varies wildly across signers based on their distance from camera, arm length, and signing space. Signer-specific coordinate distributions show left-hand wrist Y ranges from 0.64 to 0.78 across signers.

**Impact**: Instead of normalizing away signer differences, this approach amplifies them. A tall signer far from the camera and a short signer close to the camera get renormalized to the same scale, but their proportional joint positions (which carry the actual sign information) are distorted differently. The model learns these distortion patterns as class features.

**Severity**: HIGH. Directly contributes to signer-dependent representations.

### 3.4 HIGH: No Signer-Invariant Features (Bone/Angular)

**Evidence**: The model uses raw XYZ coordinates (549 features, of which only 225 are used after discarding face/lip landmarks). Training data analysis shows:
- Class 35 vs Class 22: cosine similarity 0.999 in raw XYZ
- Class 388 vs Class 48: cosine similarity 0.9998
- 1-NN accuracy on class 35: only 64% using raw features

**Impact**: Raw XYZ coordinates are inherently signer-dependent -- they encode absolute hand position, arm length, and body proportions alongside the actual sign gesture. Bone features (relative vectors between connected joints) and angular features are position-invariant and thus generalize across signers. The 0.999 cosine similarity between confusable classes in raw features means the model must rely on signer-specific artifacts to distinguish them.

**Severity**: HIGH.

### 3.5 HIGH: Side-View Video in Test Set

**Evidence**: Signer 3's number videos are ALL recorded from a side angle. The entire training set is exclusively front-facing.

**Impact**: MediaPipe's 3D landmark estimation degrades significantly with viewing angle. The projected 2D -> 3D lifting assumes roughly frontal orientation. Side-view input produces systematically different landmark geometry that the model has never seen. Signer 3's number accuracy (13.3%) is the worst in the test set.

**Severity**: HIGH for Signer 3 specifically; MEDIUM for overall accuracy since it affects 1 of 3 test signers.

### 3.6 MEDIUM: MediaPipe Version Mismatch

**Evidence**: Training data was processed with MediaPipe 0.10.14; evaluation uses 0.10.5 (9 minor versions apart).

**Impact**: Between these versions, MediaPipe changed landmark projection for non-square video and modified its kinematic path solver. This means the same video frame can produce different landmark coordinates depending on the MediaPipe version used. The magnitude of this difference is unknown but potentially significant for fine-grained classification.

**Severity**: MEDIUM. Could explain some of the noise but unlikely to be the primary driver.

### 3.7 MEDIUM: Frame Count Distribution Mismatch

**Evidence**: Training videos produce 87-127 frames; real-tester videos produce 64-188 frames.

**Impact**: The temporal resampling to a fixed frame count distorts dynamics differently depending on the input length. A 64-frame video upsampled to (e.g.) 90 frames introduces interpolation artifacts, while a 188-frame video downsampled to 90 loses temporal detail. Both cases create distribution shift the model hasn't learned to handle.

**Severity**: MEDIUM.

### 3.8 LOW-MEDIUM: Insufficient Signer-Mimicking Augmentation

**Evidence**: Current augmentation does not include bone-length perturbation, hand-size randomization, or signer-style transfer -- the techniques most effective at simulating new-signer variation.

**Impact**: Without augmentation that mimics the variation between signers (different arm lengths, signing speeds, spatial offsets), the model cannot learn to be invariant to these factors. Given only 4 unique training signers, augmentation is the primary mechanism for increasing effective signer diversity.

**Severity**: MEDIUM.

---

## 4. Literature Context

### 4.1 Comparable Cross-Signer Drop Rates

| Study | Dataset | Signer-Dep. | Signer-Indep. | Drop |
|-------|---------|-------------|----------------|------|
| Ethiopian SLR (2023) | EthSL, 5 signers | 94% | 73% | 21% |
| WLASL (2020) | 100 glosses, 100+ signers | 65.9% | -- | -- |
| SAM-SLR (CVPR 2021) | AUTSL, 43 signers | 98.4% | -- | -- |
| DSLNet (2023) | WLASL-100 | 93.7% | -- | -- |
| **KSL (ours)** | **30 classes, 4 signers** | **87.3%** | **34.2%** | **53%** |

The standard cross-signer accuracy drop in the literature is **15-25%**. Our 53% drop is 2-3x worse, indicating fundamental issues beyond normal domain shift.

### 4.2 State-of-the-Art Techniques

**Multi-Stream Ensembles**: SAM-SLR's winning approach uses 4 streams (joint, bone, joint-motion, bone-motion) with 6 modalities total. Each stream captures different aspects of the sign, and their ensemble is robust to signer-specific artifacts in any single stream.

**Dual-Reference Frame Normalization (DRFN)**: DSLNet (93.70% on WLASL-100) uses wrist-centric normalization for hands and body-centric normalization for pose, eliminating absolute position dependence.

**Signer-Adversarial Training**: Gradient reversal layers (GRL) explicitly train the model to produce features that cannot predict signer identity, forcing sign-relevant representations.

**Supervised Contrastive Learning**: Pulls same-class embeddings together and pushes different-class embeddings apart in a signer-invariant feature space.

### 4.3 Realistic Accuracy Targets

Given our dataset constraints (4 unique training signers, 30 classes, ~600 unique samples):

| Scenario | Expected Real-Test Accuracy |
|----------|-----------------------------|
| Current system | 34% |
| After P0 fixes (normalization, data cleanup) | 40-50% |
| After P0 + P1 (model shrink, augmentation, bone features) | 55-65% |
| After P0-P2 (multi-stream, TTA, contrastive loss) | 60-70% |
| With more data collection (10+ signers) | 75-85% |

---

## 5. Recommended Fixes

### P0: Critical -- Do First

#### P0.1: Remove Duplicate Signer Data

**What**: Identify and remove the duplicate signer (Signer 2 == Signer 3). Keep one copy.

**Why**: Training on duplicates inflates apparent diversity and wastes capacity learning redundant patterns. Removal forces the model to generalize from true variation.

**Implementation**:
```python
# In data loading, deduplicate by content hash
import hashlib

def get_sample_hash(data_path):
    with open(data_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Group by hash, keep only one per group
seen_hashes = set()
unique_samples = []
for sample in all_samples:
    h = get_sample_hash(sample['path'])
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_samples.append(sample)
# unique_samples now has ~600 samples instead of 750
```

**Impact**: Removes false confidence in training metrics. Reduces effective dataset to 600 samples, which correctly calibrates downstream design decisions.

#### P0.2: Wrist-Centric + Palm-Size Normalization

**What**: Replace per-sample `max(abs)` normalization with anatomically-grounded normalization:
1. Translate hand landmarks so wrist = origin
2. Scale by palm size (wrist-to-middle-finger-MCP distance)
3. Normalize pose landmarks relative to body center (mid-shoulder)
4. Scale pose by shoulder width

**Why**: This removes absolute position, camera distance, and body-size variation while preserving the relative joint geometry that encodes the actual sign.

**Implementation**:
```python
def normalize_hand(landmarks_21x3):
    """Wrist-centric, palm-size normalized hand landmarks."""
    wrist = landmarks_21x3[0]  # Wrist is landmark 0
    centered = landmarks_21x3 - wrist  # Translate to wrist origin

    # Scale by palm size (wrist to middle finger MCP, landmark 9)
    palm_size = np.linalg.norm(centered[9])
    if palm_size > 1e-6:
        centered = centered / palm_size

    return centered

def normalize_pose(landmarks_33x3):
    """Body-centric, shoulder-width normalized pose landmarks."""
    left_shoulder = landmarks_33x3[11]
    right_shoulder = landmarks_33x3[12]
    body_center = (left_shoulder + right_shoulder) / 2
    centered = landmarks_33x3 - body_center

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width > 1e-6:
        centered = centered / shoulder_width

    return centered
```

**Impact**: Expected +5-10% on real testers by removing signer-specific position/scale artifacts.

#### P0.3: Fix MediaPipe Version

**What**: Pin MediaPipe to the same version used for training data generation (0.10.14) in the evaluation/inference pipeline.

**Why**: Different MediaPipe versions produce different landmark coordinates for the same input frame. Version mismatch introduces a systematic bias in all real-tester predictions.

**Implementation**:
```bash
pip install mediapipe==0.10.14
```

If reprocessing training data is feasible, reprocess everything with the latest stable MediaPipe version instead, ensuring consistency.

**Impact**: Expected +1-3% by removing systematic landmark bias.

### P1: High Priority

#### P1.1: Reduce Model to 200K-500K Parameters

**What**: Replace the current 4.5M-parameter model with an architecture targeting 200K-500K parameters. Options:
- **EfficientGCN-B0**: ~200K params, competitive on NTU-60/120
- **Lightweight ST-GCN**: Reduce hidden dims to 32-64, reduce layers to 3-4
- **Simple temporal CNN**: 1D convolutions over joint features

**Why**: With ~600 unique training samples (after dedup), the model needs at most 600 * 100 = 60K learnable parameters for a healthy ratio. Even 300K params is conservative. The current 4.5M guarantees memorization.

**Implementation sketch**:
```python
class LightweightSTGCN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=48, num_layers=4):
        super().__init__()
        # ~300K total parameters
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Total: ~48*48*4*9 (conv) + 48*num_classes (fc) ≈ 80K-300K
```

**Impact**: Expected +5-10% by forcing the model to learn generalizable features instead of memorizing.

#### P1.2: Add Bone Features

**What**: Compute bone vectors (parent-to-child joint differences) and bone lengths as additional input channels.

**Why**: Bone features are inherently position-invariant. They encode the relative geometry of the hand/body regardless of where the signer is in the camera frame. The 0.999 cosine similarity between confusable classes in raw XYZ may be easily separable in bone space.

**Implementation**:
```python
# MediaPipe hand bone connections
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm cross-connections
]

def compute_bone_features(landmarks_21x3):
    """Compute bone vectors and lengths."""
    bones = []
    for parent, child in HAND_BONES:
        bone_vec = landmarks_21x3[child] - landmarks_21x3[parent]
        bone_len = np.linalg.norm(bone_vec)
        bones.append(np.concatenate([bone_vec, [bone_len]]))  # 4 features per bone
    return np.array(bones)  # (23, 4) = 92 features per hand per frame
```

**Impact**: Expected +3-5% by providing signer-invariant geometric features. Essential for separating confusable pairs (35/22, 388/48).

#### P1.3: Signer-Mimicking Augmentation Suite

**What**: Add augmentation transforms that simulate the variation between different signers:

1. **Bone-length perturbation** (+5-10%): Randomly scale individual bone lengths by 0.8-1.2x to simulate different hand/arm proportions
2. **Hand-size randomization** (+3-5%): Scale entire hand landmark set by a random factor
3. **Temporal warping** (+2-4%): Non-uniform temporal scaling to simulate different signing speeds
4. **Spatial jitter** (+1-2%): Small random translations and rotations of the entire signing space

**Implementation**:
```python
class SignerAugmentation:
    def __init__(self):
        self.bone_scale_range = (0.8, 1.2)
        self.hand_scale_range = (0.85, 1.15)
        self.time_warp_sigma = 0.2
        self.spatial_jitter_std = 0.02

    def bone_length_perturbation(self, landmarks, bones):
        """Perturb individual bone lengths to simulate different body proportions."""
        augmented = landmarks.copy()
        for parent, child in bones:
            scale = np.random.uniform(*self.bone_scale_range)
            bone_vec = augmented[child] - augmented[parent]
            augmented[child] = augmented[parent] + bone_vec * scale
        return augmented

    def temporal_warp(self, sequence, sigma=0.2):
        """Non-uniform temporal warping to simulate speed variation."""
        T = len(sequence)
        warp = np.cumsum(np.abs(np.random.normal(1.0, sigma, T)))
        warp = warp / warp[-1] * (T - 1)
        indices = np.clip(np.round(warp).astype(int), 0, T - 1)
        return sequence[indices]
```

**Impact**: Expected +5-10% combined by dramatically increasing effective signer diversity.

#### P1.4: Velocity Features Before Resampling

**What**: Compute velocity (frame-to-frame differences) on the original-framerate sequence, then resample to fixed length.

**Why**: Computing velocity after temporal resampling introduces interpolation artifacts. Original-framerate velocity preserves true motion dynamics.

**Implementation**:
```python
def compute_velocity(sequence):
    """Frame-to-frame differences at original framerate."""
    velocity = np.diff(sequence, axis=0)  # (T-1, features)
    velocity = np.concatenate([np.zeros((1, sequence.shape[1])), velocity], axis=0)  # Pad to T
    return velocity
```

**Impact**: Expected +1-2%. Small but important for temporal sign discrimination (e.g., 444 vs 54).

### P2: Medium Priority

#### P2.1: Multi-Stream Ensemble

**What**: Train 2-4 separate lightweight models on different feature streams and average their predictions:
- Stream 1: Joint positions (wrist-normalized)
- Stream 2: Bone vectors + bone lengths
- Stream 3: Joint velocity
- Stream 4: Bone velocity (optional)

**Why**: Each stream captures complementary information. Their errors are partially uncorrelated, so the ensemble is more robust than any individual stream. SAM-SLR (98.4% on AUTSL) uses this as a core technique.

**Impact**: Expected +3-5% over the best single stream.

#### P2.2: Test-Time Augmentation (TTA)

**What**: At inference, run the input through multiple augmented versions and average predictions:
- Horizontal flip (mirror the sign)
- Speed perturbation (0.9x, 1.0x, 1.1x)
- Small spatial jitter

**Why**: Free accuracy boost at the cost of 5-10x inference time. Reduces sensitivity to the specific test-time conditions.

**Implementation**:
```python
def predict_with_tta(model, sample, n_augments=5):
    preds = [model(sample)]  # Original
    preds.append(model(horizontal_flip(sample)))
    for speed in [0.9, 1.1]:
        preds.append(model(temporal_resample(sample, speed)))
    for _ in range(n_augments - 3):
        preds.append(model(add_jitter(sample, std=0.01)))
    return torch.stack(preds).mean(dim=0)
```

**Impact**: Expected +1-3%.

#### P2.3: Supervised Contrastive Loss

**What**: Add a contrastive learning objective that pulls same-class embeddings together and pushes different-class embeddings apart, regardless of signer identity.

**Why**: Forces the model to learn a feature space where class identity is the primary organizing principle, not signer identity.

**Implementation**:
```python
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D), labels: (B,)
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        # Mask: same class = positive, different class = negative
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        # Log-sum-exp trick for numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels), device=labels.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob.mean()
```

**Impact**: Expected +2-4%.

#### P2.4: Confusion-Pair Hard Mining

**What**: Identify the most confused class pairs from the confusion matrix and apply extra margin loss or focused sampling on those pairs.

**Why**: Classes 35/22 and 388/48 have cosine similarity >0.999 in raw features. Generic cross-entropy loss does not provide enough gradient signal to separate them. Targeted margin losses or oversampling these pairs can help.

**Impact**: Expected +1-3% on the specific confused classes.

### P3: Future / Requires Additional Resources

#### P3.1: Signer-Adversarial Training (Gradient Reversal)

**What**: Add a signer-classification head with a gradient reversal layer (GRL). The main model learns sign features that actively resist signer identification.

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class SignerAdversarialModel(nn.Module):
    def __init__(self, backbone, num_classes, num_signers):
        super().__init__()
        self.backbone = backbone
        self.sign_head = nn.Linear(backbone.feature_dim, num_classes)
        self.signer_head = nn.Sequential(
            GradientReversal.apply,
            nn.Linear(backbone.feature_dim, num_signers)
        )
```

**Impact**: Expected +3-5% but requires careful hyperparameter tuning (GRL alpha schedule).

#### P3.2: Collect More Training Data

**What**: Record 5-10 additional signers with diverse body types, signing styles, and camera angles (including side view).

**Why**: No amount of architectural cleverness can fully substitute for actual signer diversity. The literature shows that increasing from 5 to 20+ signers dramatically improves cross-signer generalization.

**Impact**: Expected +10-20% -- the single highest-impact change possible.

#### P3.3: Transfer Learning from Large SLR Datasets

**What**: Pretrain on WLASL or AUTSL (1000+ classes, 100+ signers) and fine-tune on KSL.

**Impact**: Expected +5-10% by leveraging pretrained signer-invariant representations.

---

## 6. V20 Implementation Plan

V20 should implement P0 + P1 changes. Here is the specific plan:

### Phase 1: Data & Preprocessing (change data pipeline)

```
1. Deduplicate training data
   - Hash all training samples
   - Remove Signer 2 or Signer 3 (they are identical)
   - New training set: ~600 unique samples from 4 signers

2. Replace normalization
   - Remove: per-sample max(abs) normalization
   - Add: wrist-centric hand normalization + body-centric pose normalization
   - Apply palm-size scaling for hands, shoulder-width scaling for pose

3. Fix MediaPipe version
   - Pin mediapipe==0.10.14 in requirements.txt
   - Or reprocess ALL data with a single consistent version

4. Add bone features
   - Compute bone vectors for 23 hand bones (per hand)
   - Compute bone lengths
   - Input channels: joint(63) + bone(92) + velocity(63) = 218 per hand
   - Total with pose: ~500 features per frame
```

### Phase 2: Model Architecture

```
5. Shrink model
   - Target: 200K-500K total parameters
   - Architecture: Lightweight ST-GCN or temporal CNN
   - hidden_dim=48, num_layers=4, dropout=0.3
   - Separate numbers/words models (as before)

6. Loss function
   - Primary: Cross-entropy (simple, proven)
   - Secondary: SupCon loss with weight 0.1
   - No SWA (confirmed to hurt on this dataset)
```

### Phase 3: Augmentation

```
7. Signer-mimicking augmentation
   - Bone-length perturbation (0.8-1.2x per bone)
   - Hand-size randomization (0.85-1.15x)
   - Temporal warping (sigma=0.2)
   - Spatial jitter (std=0.02)
   - Horizontal flip (with class-aware filtering for asymmetric signs)
   - Random frame dropping (up to 10%)

8. MixUp augmentation
   - Alpha=0.2, applied in feature space
```

### Phase 4: Training

```
9. Training configuration
   - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
   - Scheduler: CosineAnnealing, T_max=300
   - Batch size: 32
   - Epochs: 300 (with early stopping, patience=50)
   - Gradient clipping: max_norm=1.0

10. Evaluation
    - LOSO cross-validation on training set (excluding duplicate signer)
    - Report both val accuracy and real-tester accuracy
    - Track per-class accuracy and confusion matrix
```

### File Changes Summary

| File | Change |
|------|--------|
| `train_ksl_v20.py` | New training script with all P0+P1 changes |
| `requirements.txt` | Pin mediapipe==0.10.14 |
| `slurm/v20.sh` | SLURM job script |
| Data loading | Dedup logic, new normalization, bone features |
| Model definition | Lightweight architecture (~300K params) |
| Augmentation | Full signer-mimicking suite |

---

## 7. Expected Impact

### Projected Accuracy (Cumulative)

| Fix | Numbers (Real) | Words (Real) | Combined |
|-----|---------------|-------------|----------|
| Current | 20.3% | 48.1% | 34.2% |
| + P0.1 Dedup data | 20% | 48% | 34% (no change, but honest baseline) |
| + P0.2 Wrist-norm | 28-33% | 55-60% | 42-47% |
| + P0.3 MediaPipe fix | 30-35% | 57-62% | 44-49% |
| + P1.1 Model shrink | 35-42% | 60-67% | 48-55% |
| + P1.2 Bone features | 38-47% | 63-70% | 51-59% |
| + P1.3 Augmentation | 43-52% | 67-75% | 55-64% |
| + P1.4 Velocity | 45-54% | 68-76% | 57-65% |
| + P2 (multi-stream, TTA, SupCon) | 50-60% | 72-80% | 61-70% |

### Key Uncertainties

- Side-view robustness: Signer 3's numbers may remain poor without side-view training data
- Confusable pairs: Classes 35/22 may need dedicated solutions beyond bone features
- Numbers vs Words: Numbers inherently harder (more similar hand shapes, less body motion)

---

## 8. Deployment Recommendations

### 8.1 Confidence-Based Rejection

The model's confidence gap between correct (0.391) and incorrect (0.261) predictions is too small for reliable confidence-based filtering. However, with the improved model:

```python
def predict_with_rejection(model, sample, threshold=0.4):
    logits = model(sample)
    probs = F.softmax(logits, dim=-1)
    max_prob, pred_class = probs.max(dim=-1)

    if max_prob < threshold:
        return None, max_prob  # Rejected -- ask user to re-sign
    return pred_class, max_prob
```

**Current data shows**: At confidence threshold 0.4, words accuracy jumps from 48% to 75% (by rejecting uncertain predictions). This is a viable deployment strategy.

Recommended thresholds:
- **High confidence** (>0.6): Display prediction directly
- **Medium confidence** (0.4-0.6): Display prediction with "Did you mean...?" confirmation
- **Low confidence** (<0.4): Ask user to re-sign

### 8.2 Top-K Fallback

When confidence is low, show the top-3 predictions and let the user select:

```python
def predict_top_k(model, sample, k=3):
    logits = model(sample)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_classes = probs.topk(k, dim=-1)
    return list(zip(top_classes.tolist(), top_probs.tolist()))
```

### 8.3 View-Angle Warning

Since side-view severely degrades accuracy, detect non-frontal views and warn the user:

```python
def check_view_angle(pose_landmarks):
    """Warn if camera angle appears non-frontal."""
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulder_z_diff = abs(left_shoulder[2] - right_shoulder[2])
    if shoulder_z_diff > 0.1:  # Significant depth difference = side view
        return "Warning: Please face the camera directly for best results."
    return None
```

### 8.4 UI Recommendations

1. Show confidence indicator (color-coded bar) alongside predictions
2. For numbers, default to showing top-3 with selection (given 20% accuracy)
3. Add "Re-sign" button prominently for low-confidence predictions
4. Show camera framing guide overlay to ensure front-facing capture
5. Consider separate UI flows for numbers vs words (words are more reliable)

---

## Appendix A: Data Quality Issues Detail

### Duplicate Signer Evidence

The training set contains 5 nominal signers (1-5), but byte comparison reveals:
- Signer 2 data == Signer 3 data (100% match across all classes and repetitions)
- This means the model trains on 150 duplicate samples
- LOSO results for these signers (99.3% and 100%) confirm trivial prediction

### Feature Utilization

Current pipeline extracts 549 raw features per frame:
- **Used (225)**: Pose (33 landmarks x 3) + Left hand (21 x 3) + Right hand (21 x 3)
- **Discarded (324)**: Face mesh landmarks and lip landmarks

The discarded face/lip features could potentially help with signs that involve mouthing or facial expressions, but are not relevant for the current 30-class vocabulary which focuses on hand-dominant signs.

### Confusable Class Pairs

| Class A | Class B | Cosine Sim | Notes |
|---------|---------|-----------|-------|
| 35      | 22      | 0.999     | Nearly identical in raw XYZ |
| 388     | 48      | 0.9998    | Nearly identical in raw XYZ |
| 444     | 54      | ~0.95     | Temporal distinction needed |

Class 35 has been at 0% accuracy across all versions (v16, v17, v18), suggesting it is fundamentally inseparable from class 22 using current features.

---

## Appendix B: Literature References

- **Ethiopian SLR**: Signer-dependent 94% vs signer-independent 73%, demonstrating typical 21% cross-signer drop with 5 signers
- **SAM-SLR (CVPR 2021 Workshop winner)**: 98.42% on AUTSL with 4-stream ensemble, 43 signers, 6 modalities
- **DSLNet**: 93.70% on WLASL-100 using dual-reference frame normalization (wrist-centric + body-centric)
- **AimCLR**: Extreme augmentation strategies for skeleton-based action recognition; publicly available implementation
- **EfficientGCN**: Lightweight GCN architecture achieving competitive results with ~200K parameters

---

*This report synthesizes findings from codebase analysis, literature review, and augmentation research conducted on 2026-02-09. It serves as the primary roadmap for closing the KSL real-tester generalization gap.*
