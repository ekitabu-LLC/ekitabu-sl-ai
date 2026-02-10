# V24 Research: Few-Shot Learning, Prototypical Networks, and User Enrollment

## Context

**Problem**: KSL v22 achieves 93.3% val accuracy but only 39.8% on unseen signers (4 training signers). The 55pp val-to-real gap is dominated by signer variation, not class confusion. Architecture improvements (v23 multi-stream ensemble, focal loss, calibration) confirmed a **4-signer ceiling at ~40% real accuracy**.

**Goal**: Break the 40% barrier without collecting more signers. The most promising path: adapt the model to each new user at deployment time using a few calibration samples.

**Deployment scenario**: A new user opens the app, optionally performs 3-5 calibration signs per class, and the model adapts to their signing style.

---

## 1. Prototypical Networks for Skeleton-Based Action Recognition

### 1.1 Core Concept

Prototypical networks (Snell et al., 2017) replace the FC classifier with nearest-centroid classification in a learned embedding space. For each class k, a prototype c_k is computed as the mean embedding of support samples:

```
c_k = (1/|S_k|) * sum(f_theta(x_i))  for all (x_i, y_i) in S_k
```

Classification is via softmax over negative distances:

```
p(y=k|x) = exp(-d(f(x), c_k)) / sum_k' exp(-d(f(x), c_k'))
```

### 1.2 Relevance to KSL v22

The v22 model already produces a `self.embed_dim = gcn_embed_dim + 64` embedding (192-dim for hd=64, nl=4). The current classifier is a 2-layer MLP:

```python
self.classifier = nn.Sequential(
    nn.Linear(self.embed_dim, hd),  # 192 -> 64
    nn.ReLU(),
    nn.Dropout(dr),
    nn.Linear(hd, nc),              # 64 -> num_classes
)
```

**Prototypical replacement**: Remove the classifier entirely. During training, compute class prototypes from the batch embeddings and classify via Euclidean distance. During enrollment, compute new prototypes from user calibration samples.

### 1.3 Key Papers

- **ProtoGCN** (Xiong et al., 2024): Breaks down skeleton dynamics into learnable prototypes representing core motion patterns. Uses prototype-based decomposition for skeleton-based action recognition. Relevant for understanding how prototypes capture motion patterns rather than just class identity.

- **PGFA - Prototype-guided Feature Alignment** (2025, arxiv:2507.00566): Uses prototypical learning for zero-shot skeleton-based action recognition. Achieves +12-22% improvements over baselines on NTU-60/120 and PKU-MMD. The key insight is aligning skeleton features with semantic prototypes, which could be adapted for signer-invariant features.

- **Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition** (Zhu et al., ECCV 2022): Directly addresses few-shot skeleton recognition. Preserves spatial structure during embedding, achieving SOTA on NTU RGB+D for 5-way 1-shot and 5-shot settings.

- **CrossGLG** (ECCV 2024): LLM-guided one-shot skeleton-based 3D action recognition. Uses cross-level guidance from language models to improve one-shot performance.

- **Multimodal Prototype-Enhanced Network** (ICMR 2024): Enhances prototypes with multimodal information for few-shot action recognition.

### 1.4 Implementation for KSL v24

**Approach**: Episodic training with prototypical classification.

```python
# During training: Episodic Prototypical Training
# 1. Sample N classes (e.g., 15-way)
# 2. For each class, sample K support + Q query samples
# 3. Compute prototypes from support set
# 4. Classify query set via distance to prototypes
# 5. Backprop through the encoder (f_theta)

# At enrollment time:
# 1. User signs 3-5 samples per class
# 2. Compute prototypes: c_k = mean(f_theta(x_user_k))
# 3. Optionally blend with training prototypes:
#    c_k_final = alpha * c_k_user + (1-alpha) * c_k_train
# 4. Classify new signs by nearest prototype
```

**Complexity**: LOW. Only changes the classification head and training loop. The ST-GCN backbone and aux branch are unchanged.

**Expected improvement**: 10-25% on unseen signers (based on literature). The prototype mechanism inherently adapts to the user's embedding distribution without retraining.

---

## 2. User Enrollment: Adapting to a New Signer with 3-5 Calibration Samples

### 2.1 Enrollment Strategies (Ranked)

#### Strategy A: Prototype Update (RECOMMENDED - Simplest)

After base training, store per-class prototypes from training data. At enrollment:

1. User performs 3-5 signs per class (90-150 total signs for 30 classes)
2. Compute user prototypes in embedding space
3. Blend: `c_final = alpha * c_user + (1-alpha) * c_train`
4. Classify via nearest blended prototype

**Pros**: No gradient computation, instant, works in browser/mobile
**Cons**: Requires user to sign all 30 classes
**Fallback**: If user enrolls only a subset, use training prototypes for un-enrolled classes

#### Strategy B: Last-Layer Fine-Tuning

Freeze the backbone (ST-GCN + aux), fine-tune only the classifier head on user data.

Based on Zang (2025, "TinyML-Driven On-Device Sports Command Recognition"):
- Fine-tune only the final dense layer parameters
- Use dropout (p=0.3), early stopping (patience=3), Gaussian noise augmentation
- As few as 15 labeled examples per class sufficient for robust personalization
- 5.3% accuracy gain over non-personalized baseline

**For KSL**: Fine-tune `self.classifier` (2 layers, ~12K params for 30 classes) while freezing the 776K-param backbone.

```python
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Fine-tune with user data (3-5 samples/class)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
for epoch in range(10):  # Very few epochs
    for batch in user_dataloader:
        logits, _, _ = model(gcn, aux, grl_lambda=0.0)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

**Pros**: Uses gradient information, can adapt to class-specific biases
**Cons**: Requires ~3-5 seconds of compute, risk of overfitting with <5 samples/class

#### Strategy C: BN Statistics Adaptation

Adapt only BatchNorm statistics to the new signer's distribution.

Based on MABN (AAAI 2024, "Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization"):
- BN layers encode domain-specific information (signer style)
- Update BN running mean/variance from user's calibration data
- Keep affine parameters (gamma, beta) from training
- Only need ~30-50 unlabeled samples (1-2 per class is enough)

**For KSL v22**: The model has BN layers in:
- `self.data_bn` (input normalization)
- Each `STGCNBlock` (multiple BN layers)
- `self.aux_temporal_conv` (aux branch BN)
- `self.aux_bn` (aux input BN)

```python
# BN adaptation - no labels needed!
model.train()  # Enable BN stat tracking
with torch.no_grad():
    for batch in user_unlabeled_data:
        _ = model(gcn, aux, grl_lambda=0.0)  # Forward only, updates BN stats
model.eval()
```

**Pros**: No labels needed, very fast, minimal risk
**Cons**: May not be enough to overcome large signer differences

#### Strategy D: Combined Approach (RECOMMENDED for Maximum Impact)

1. BN adaptation (unlabeled, ~30 samples)
2. Prototype update (labeled, 3-5 per class)
3. Optional: Last-layer fine-tune if accuracy still poor

### 2.2 Enrollment UX Design

**Minimal enrollment (30 seconds)**:
- User signs 1 sample each of 10 high-confusion classes
- BN stats updated from these samples
- Prototypes updated for enrolled classes, training prototypes for rest

**Standard enrollment (2-3 minutes)**:
- User signs 3 samples each of all 30 classes
- Full prototype update + BN adaptation

**Progressive enrollment**:
- No enrollment needed to start (use training prototypes)
- App learns from user corrections over time
- Prototypes updated incrementally: `c_k = (n*c_k + f(x_new)) / (n+1)`

---

## 3. Test-Time Adaptation Methods

### 3.1 TENT (Test-time ENTropy minimization)

**Paper**: Wang et al., ICLR 2021. "Tent: Fully Test-time Adaptation by Entropy Minimization"

**Method**: At test time, minimize entropy of predictions by updating BN affine parameters (gamma, beta):

```python
# TENT adaptation
model.train()  # Enable BN tracking
for param in model.parameters():
    param.requires_grad = False
# Only update BN affine params
for m in model.modules():
    if isinstance(m, nn.BatchNorm1d):
        m.weight.requires_grad = True
        m.bias.requires_grad = True

optimizer = torch.optim.SGD(bn_params, lr=0.001)
for batch in test_data:
    logits = model(batch)
    loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    loss.backward()
    optimizer.step()
```

**Applicability to KSL**:
- TENT needs batch-level adaptation (not single-sample)
- In deployment, we process one sign at a time -- batch size 1
- TENT becomes unstable with very small batches
- **Verdict**: NOT directly suitable for single-sign inference. Could work during enrollment phase (batch of calibration signs).

### 3.2 MABN (Domain-Aware Batch Normalization)

**Paper**: Wu et al., AAAI 2024. "Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization"

**Method**: Learns domain-aware BN parameters. Key insight: updating only affine parameters (not statistics) is more stable with few samples. Statistics re-estimation from few samples is intrinsically unstable.

**For KSL**: During enrollment, adapt BN affine parameters to new signer's distribution using calibration samples. More stable than full TENT.

### 3.3 Skeleton-Cache (Training-Free TTA)

**Paper**: NeurIPS 2025. "Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation"

**Method**: Reformulates inference as lightweight retrieval over a non-parametric cache storing structured skeleton representations. No training needed. Uses both global and local descriptors with LLM-guided fusion weights.

**Results**: +4-7% accuracy improvement over base models on NTU RGB+D 60/120 without any retraining.

**Applicability to KSL**:
- The cache concept is very similar to our prototypical approach
- Storing enrollment samples as a retrieval cache
- Could combine ST-GCN embeddings with raw skeleton descriptors
- **However**: Originally designed for zero-shot (new classes), not cross-signer (same classes, new style). The distribution shift is different.

### 3.4 Test-Time Training (TTT)

**Paper**: Sun et al., ICML 2020. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts"

**Method**: Add a self-supervised auxiliary task (e.g., rotation prediction) during training. At test time, adapt the shared feature extractor by optimizing the self-supervised loss on test data.

**For KSL**:
- We already have auxiliary features (angles, distances) and a signer-adversarial head
- Could use reconstruction of skeleton coordinates as self-supervised task
- At test time, optimize the reconstruction loss on user's signs
- **Complexity**: MEDIUM. Requires adding an auxiliary decoder.

### 3.5 Recommendation for TTA

| Method | Labeled? | Compute | Stability | Expected Gain | Complexity |
|--------|----------|---------|-----------|---------------|------------|
| BN Stats Update | No | <1s | High | +5-10% | Very Low |
| TENT (BN affine) | No | ~5s | Medium | +5-15% | Low |
| MABN | No | ~5s | High | +5-15% | Low |
| Prototype Update | Yes (3-5/class) | <1s | High | +10-25% | Low |
| Last-layer FT | Yes (3-5/class) | ~10s | Medium | +10-20% | Low |
| TTT (self-sup) | No | ~30s | Medium | +5-15% | Medium |

**Best combination**: BN stats update (unlabeled) + Prototype update (labeled) = expected +15-30% over baseline.

---

## 4. Meta-Learning Approaches (MAML, ProtoMAML)

### 4.1 MAML (Model-Agnostic Meta-Learning)

**Paper**: Finn et al., ICML 2017.

**Concept**: Learn initialization parameters theta such that a few gradient steps on a new task yields good performance. The outer loop optimizes for fast adaptability across tasks.

**For KSL**: Treat each signer as a "task". With only 4 signers, we have very few tasks for meta-learning.

```
meta-train tasks: signer_1, signer_2, signer_3
meta-test task: signer_4

Inner loop: 5 gradient steps on signer_k support set
Outer loop: Optimize theta for fast adaptation
```

**Critical limitation for KSL**: MAML needs many diverse tasks (typically 64+ for mini-ImageNet). With only 4 signers, we cannot form enough meta-train/meta-test splits. **MAML is not viable with 4 signers**.

### 4.2 ProtoMAML

**Concept**: Hybrid of MAML and prototypical networks. Initialize the classifier weights from prototypes (rather than random), then take a few MAML-style gradient steps.

```python
# ProtoMAML inner loop for a new signer
# 1. Compute prototypes from support set
c_k = mean(f_theta(support_k))
# 2. Initialize classifier weights from prototypes
W = stack(c_k for all k)  # shape: (num_classes, embed_dim)
b = -0.5 * sum(c_k^2)     # bias term
# 3. Take gradient steps on support set
for step in range(5):
    logits = x @ W.T + b
    loss = CE(logits, labels)
    W, b = W - lr * grad(loss, W), b - lr * grad(loss, b)
```

**For KSL**: More promising than pure MAML because:
- Prototype initialization provides a good starting point
- Gradient steps adapt to signer-specific patterns
- Can work with as few as 3-5 samples per class
- **Still limited by 4-signer meta-training, but less sensitive than MAML**

### 4.3 Cross-Lingual Few-Shot Sign Language Recognition

**Paper**: "Cross-lingual few-shot sign language recognition" (Pattern Recognition, 2024)

Uses few-shot learning to transfer sign language recognition across different sign languages. Demonstrates that sign language-specific features can transfer with very few examples. Relevant architecture decisions for handling cross-domain (cross-signer) shift.

### 4.4 Meta-Learning Verdict for KSL

| Approach | Viable with 4 Signers? | Expected Gain | Complexity |
|----------|----------------------|---------------|------------|
| MAML | NO (too few tasks) | N/A | High |
| ProtoMAML | MARGINAL | +5-15% | Medium |
| Prototypical Net | YES | +10-25% | Low |
| Episodic Training | YES (with synthetic tasks) | +10-20% | Medium |

**Recommendation**: Use prototypical networks (not full MAML). If meta-learning is desired, create synthetic "signers" through augmentation to increase task diversity, then use episodic training.

---

## 5. Confidence Calibration Across Distribution Shifts

### 5.1 The KSL Calibration Problem

V22 analysis showed devastating miscalibration:
- Numbers HIGH confidence: only 18% accurate (worse than random!)
- Post-hoc calibration (v23): trained on val, completely fails on real testers
- Temperature scaling: optimal T on val doesn't transfer to OOD signers

### 5.2 Methods That Work Under Shift

#### Deep Ensembles (Lakshminarayanan et al., 2017)

**Finding** (Wyatt et al., 2022, Methods in Ecology and Evolution): "Ensemble methods are more stable in their calibration across dataset shifts compared with other approaches." Under extreme shift, ensembles become overconfident like single models, but degrade more gracefully.

- Train M independent models (M=3-5) with different random seeds
- Average their softmax outputs
- Disagreement between ensemble members = uncertainty signal

**For KSL**: v23 already has 4 streams. Could repurpose as an ensemble with disagreement-based confidence. However, v23 showed streams are too correlated on OOD data.

**Better approach**: Train 3-5 independent v22 models with different seeds. Use prediction entropy of the average as confidence.

#### Distance-Based OOD Detection

**Finding** (Miao et al., 2023): Distance-based methods using Mahalanobis distance or cosine similarity in the embedding space are effective for separating in-distribution from out-of-distribution samples.

**For KSL with prototypes**: After enrollment, confidence = negative distance to nearest prototype. This is inherently better calibrated because:
- Enrolled classes have prototypes in the user's embedding space
- Out-of-distribution signs are far from all prototypes
- No temperature tuning needed

```python
# Prototype-based confidence
distances = [d(embedding, c_k) for k in classes]
confidence = 1.0 / (1.0 + min(distances))  # Bounded [0, 1]
# OR: relative margin
top1_dist, top2_dist = sorted(distances)[:2]
margin = top2_dist - top1_dist  # Larger margin = more confident
```

#### ODIN (Out-of-Distribution Detector for Neural Networks)

**Method**: Apply temperature scaling and input perturbation at test time to separate in-distribution from OOD samples. Unlike static temperature scaling, ODIN computes per-sample confidence.

**Relevance**: Could help reject low-quality signs (partial hand occlusion, motion blur) rather than miscalssifying them.

### 5.3 Recommendation for Calibration

1. **Switch to prototype-based distance confidence** (replaces softmax confidence)
2. **Use embedding margin** (distance to 2nd-nearest prototype / distance to nearest) as confidence score
3. **Ensemble disagreement** as secondary signal (if multiple models available)
4. **Reject low-confidence**: Show top-3 suggestions when margin < threshold
5. **Never use post-hoc temperature scaling** -- confirmed to not transfer to OOD

---

## 6. Practical Deployment: Enrollment in a Mobile/Web App

### 6.1 Architecture for Deployment

```
+-------------------+     +------------------+     +-------------------+
|   Frontend (Web)  | --> | MediaPipe        | --> | Skeleton-to-      |
|   Camera Feed     |     | Hand/Pose Track  |     | Feature Pipeline  |
+-------------------+     +------------------+     +-------------------+
                                                            |
                                                    +-------v--------+
                                                    |  ONNX/TFLite   |
                                                    |  ST-GCN Model  |
                                                    |  (Backbone)    |
                                                    +-------+--------+
                                                            |
                                                    +-------v--------+
                                                    | Prototype Store |
                                                    | (Per-User)      |
                                                    +-------+--------+
                                                            |
                                                    +-------v--------+
                                                    | Distance-Based  |
                                                    | Classification  |
                                                    +----------------+
```

### 6.2 Enrollment Flow

```
1. App loads with DEFAULT prototypes (from training data)
   - User can start signing immediately (~40% accuracy)

2. Enrollment prompt: "Sign each word to personalize"
   - Show sign name + reference video
   - User performs the sign 3x
   - App extracts embeddings, computes per-class prototype
   - Real-time feedback: "Great!" / "Try again"

3. Prototype blending:
   - c_final = alpha * c_user + (1-alpha) * c_train
   - alpha starts at 0.5, increases with more samples
   - alpha = n_user / (n_user + n_prior), where n_prior = 5 (effective prior samples)

4. Progressive improvement:
   - After each prediction, show "Was this correct?" prompt
   - If user corrects, update prototype incrementally
   - c_k = (n * c_k + f(x_correction)) / (n + 1)
```

### 6.3 Technical Requirements

| Component | Method | Latency | Size |
|-----------|--------|---------|------|
| Skeleton extraction | MediaPipe Hands+Pose | ~30ms | Included in browser |
| Feature computation | JS/WASM | ~5ms | ~50KB |
| ST-GCN inference | ONNX.js / TFLite.js | ~20ms | ~3MB (v22: 776K params) |
| Prototype computation | JS (mean of embeddings) | <1ms | Negligible |
| Prototype storage | IndexedDB (per-user) | <1ms | ~50KB |
| Total inference | | ~55ms (~18 FPS) | ~3MB |

### 6.4 Storage and Privacy

- Prototypes are stored locally (IndexedDB), never sent to server
- Each prototype is just a 192-dim float vector (~768 bytes per class)
- 30 classes = ~23KB per user's prototype store
- Model weights are static and can be cached

### 6.5 Fallback Behavior

- **No enrollment**: Use training prototypes. Expected ~40% accuracy.
- **Partial enrollment** (10/30 classes): Blended prototypes for enrolled, training for rest. Expected ~55% overall.
- **Full enrollment** (30 classes, 3 each): Full user prototypes. Expected ~65-75% accuracy.
- **Progressive**: Accuracy improves with each corrected prediction.

---

## 7. Ranked Recommendations for V24

### Tier 1: High Impact, Low Complexity (Implement First)

| # | Approach | Expected Gain | Effort | Risk |
|---|----------|---------------|--------|------|
| 1 | **Prototypical classifier** (replace FC head with nearest-centroid) | +10-25% with enrollment | 1-2 days | Low |
| 2 | **User enrollment with prototype update** | +15-25% | 2-3 days | Low |
| 3 | **Distance-based confidence** (replace softmax confidence) | Better calibration | 0.5 days | Very Low |
| 4 | **BN statistics adaptation** at enrollment time | +5-10% | 0.5 days | Very Low |

### Tier 2: Medium Impact, Medium Complexity

| # | Approach | Expected Gain | Effort | Risk |
|---|----------|---------------|--------|------|
| 5 | **Episodic training** (simulate new signers via augmentation) | +5-15% base accuracy | 2-3 days | Medium |
| 6 | **Last-layer fine-tuning** at enrollment | +5-10% on top of prototypes | 1-2 days | Medium |
| 7 | **Progressive enrollment** (learn from corrections) | Ongoing improvement | 2-3 days | Low |
| 8 | **TENT/MABN** during enrollment batch | +5-10% | 1-2 days | Medium |

### Tier 3: Lower Priority / Higher Risk

| # | Approach | Expected Gain | Effort | Risk |
|---|----------|---------------|--------|------|
| 9 | **ProtoMAML** meta-training | +5-15% | 3-5 days | High (4 signers too few) |
| 10 | **Deep ensemble** (train 3-5 models) | Better calibration | 2-3 days (compute) | Low but expensive |
| 11 | **TTT with self-supervised head** | +5-10% | 3-5 days | Medium |
| 12 | **Skeleton-Cache** style retrieval | +4-7% | 3-5 days | Medium |

### V24 Implementation Plan

**Phase 1 (Tier 1, ~4 days)**:
1. Replace FC classifier with prototypical classification
2. Train v22 backbone with episodic/prototypical loss
3. Implement enrollment: collect user samples, compute prototypes, blend
4. Switch confidence from softmax to distance-based margin
5. Add BN stats adaptation during enrollment

**Phase 2 (Tier 2, ~5 days)**:
6. Add synthetic signer augmentation for episodic diversity
7. Implement progressive enrollment (learn from corrections)
8. Optional: TENT during enrollment batch

**Evaluation Plan**:
- Simulate enrollment using real tester data:
  - Hold out 3-5 samples per class per real tester as "enrollment set"
  - Evaluate remaining samples as "test set"
  - Compare: no enrollment vs BN-adapt vs prototype vs combined

---

## 8. Expected Accuracy Projections

| Scenario | Numbers | Words | Combined |
|----------|---------|-------|----------|
| V22 (no enrollment) | 33.9% | 45.7% | 39.8% |
| V24 + BN adapt only | ~40% | ~50% | ~45% |
| V24 + Prototype (3-shot) | ~50% | ~60% | ~55% |
| V24 + Prototype (5-shot) | ~55% | ~65% | ~60% |
| V24 + BN + Proto + FT | ~60% | ~70% | ~65% |
| V24 + Progressive (after 50 corrections) | ~65% | ~75% | ~70% |

**Note**: These are rough estimates based on literature review and the ~55pp val-real gap. The actual gains depend on:
- How much of the gap is signer variation (addressable) vs. class ambiguity (not addressable)
- Quality of calibration samples (controlled signing vs. natural)
- Number of calibration samples per class

---

## References

### Core Methods
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. NeurIPS.
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-time Adaptation by Entropy Minimization. ICLR.

### Skeleton-Based Action Recognition
- Zhu et al. (2022). Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition. ECCV.
- CrossGLG (2024). LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner. ECCV.
- PGFA (2025). Zero-Shot Skeleton-Based Action Recognition With Prototype-Guided Feature Alignment. arxiv:2507.00566.
- Skeleton-Cache (NeurIPS 2025). Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation.

### Test-Time Adaptation
- Wu et al. (AAAI 2024). Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization.
- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML.
- Mounsaveng et al. (WACV 2024). Bag of Tricks for Fully Test-Time Adaptation.

### User Personalization
- Zang, J. (2025). TinyML-Driven On-Device Sports Command Recognition. Internet Technology Letters. DOI: 10.1002/itl2.70090.
- Liu et al. (2022). Continuous Gesture Sequences Recognition Based on Few-Shot Learning. Int J Aerospace Engineering. DOI: 10.1155/2022/7868142.

### Calibration Under Shift
- Wyatt et al. (2022). Using ensemble methods to improve the robustness of deep learning for image classification in marine environments. Methods in Ecology and Evolution. DOI: 10.1111/2041-210X.13841.
- Demir et al. (2023). Subnetwork ensembling and data augmentation: Effects on calibration. Expert Systems. DOI: 10.1111/exsy.13252.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.

### Sign Language Recognition
- Kamal, S. M., Chen, Y., & Li, S. (2024). Adversarial autoencoder for continuous sign language recognition. Concurrency and Computation. DOI: 10.1002/cpe.8220.
- Cross-lingual few-shot sign language recognition. Pattern Recognition, 2024.
