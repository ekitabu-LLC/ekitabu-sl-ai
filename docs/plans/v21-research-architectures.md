# V21 Architecture & Ensemble Research Report

## Research Date: 2026-02-09
## Context: KSL Recognition - 30 classes, 4 signers, ~600 samples, current best real-tester: 34.2% (v19)

---

## 1. Lightweight GCN Architectures for Skeleton Recognition

### 1.1 BlockGCN (CVPR 2024) - RECOMMENDED
- **Paper**: Zhou et al., "BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition"
- **Key Insight**: Existing GCNs optimize adjacency matrices jointly with model weights, causing gradual decay of bone connectivity data
- **Core Technique (BlockGC)**: Divides feature dimensions into multiple groups, applying spatial aggregation and feature projection in parallel to model high-level semantics efficiently
- **Topology Encoding**: Uses graph distances and persistent homology analysis to preserve topological information that standard GCN training loses
- **Results**: SOTA on NTU RGB+D 120 with high accuracy AND lightweight design
- **Parameter Reduction**: Over 40% reduction vs original GCNs while improving performance
- **Code**: https://github.com/ZhouYuxuanYX/BlockGCN
- **Relevance to KSL**: The topology preservation is critical - our confusable class pairs (cosine sim >0.999) suggest topology information is being lost. BlockGC's grouped processing would also naturally reduce parameters.

### 1.2 EfficientGCN (Song et al., 2022)
- **Variants**: B0 (0.29M params), B2, B4 (largest)
- **Key Innovation**: Advanced separable convolutional layers + early-fused Multiple Input Branches (MIB) network
- **Input**: Three input features (joint, velocity, bone) fused early in the model
- **Scaling**: Compound scaling strategy to expand width and depth synchronously
- **B4 Results**: 91.7% on NTU 60 cross-subject, 3.15x smaller and 3.21x faster than MS-G3D
- **B0 at 0.29M params is very attractive for our 600-sample dataset** - even B0 has 290K params which is reasonable for 600 samples (~483 params/sample)
- **Relevance to KSL**: B0 with 0.29M params is ideal for our small dataset. Early fusion of joint/velocity/bone is exactly what we need.

### 1.3 InfoGCN (CVPR 2022) - HIGHLY RECOMMENDED
- **Paper**: Chi et al., "InfoGCN: Representation Learning for Human Skeleton-Based Action Recognition"
- **Key Innovation 1**: Information bottleneck-based learning objective for compact latent representations
- **Key Innovation 2**: Attention-based graph convolution capturing context-dependent intrinsic topology
- **Key Innovation 3**: Multi-modal skeleton representation using relative positions of joints
- **Results**: 93.0% NTU 60 XSub, 89.8% NTU 120 XSub, 97.0% NW-UCLA
- **Code**: https://github.com/stnoah1/infogcn
- **Relevance to KSL**: The information bottleneck objective is EXACTLY what we need - it forces the model to learn informative but COMPACT representations, which should help with our overfitting problem. The multi-modal representation using relative positions could help with signer invariance.

### 1.4 HD-GCN (ICCV 2023)
- **Paper**: Lee et al., "Hierarchically Decomposed Graph Convolutional Networks"
- **Key Technique**: Decomposes each joint node into sets extracting major structurally adjacent and distant edges
- **A-HA Module**: Attention-guided hierarchy aggregation highlights dominant hierarchical edge sets
- **Code**: https://github.com/Jho-Yonsei/HD-GCN
- **Relevance to KSL**: The hierarchical decomposition could help distinguish similar hand shapes by attending to different structural levels (finger-level vs hand-level vs arm-level).

### 1.5 SkateFormer (ECCV 2024)
- **Paper**: "Skeletal-Temporal Transformer for Human Action Recognition"
- **Key Innovation**: Partitions joints and frames based on skeletal-temporal relation types, performing self-attention within each partition
- **Approach**: Action-adaptive attention that selectively focuses on key joints and frames
- **Code**: https://github.com/KAIST-VICLab/SkateFormer
- **Relevance to KSL**: Partition-based attention could help focus on discriminative hand regions for sign language. However, transformer architectures typically need more data.

### 1.6 ShiftGCN / ShiftGCN++ (Cheng et al.)
- **Key Innovation**: Replaces heavy graph convolutions with shift-graph operations + pointwise convolutions
- **Result**: Reduces computational complexity by orders of magnitude
- **Extremely lightweight** - good candidate for our small dataset
- **Relevance to KSL**: Minimal parameters, good for small datasets

### 1.7 SGN (Semantics-Guided Network)
- **Key Innovation**: Lightweight model based on semantic information (joint type + frame index + dynamics)
- **Parameters**: ~0.69M - significantly reduced from standard GCNs
- **Relevance to KSL**: The semantic guidance (encoding joint identity) is very relevant - different joints have different importance for sign language.

### 1.8 AeS-GCN (Xu et al., 2022)
- **Paper**: "Attention-enhanced Semantic-guided GCN for Skeleton-based Action Recognition"
- **Key Innovation**: Fuses semantics of joint type, frame index, and dynamics as skeleton representation
- **Spatial Attention Block (SAB)**: Explores important spatial features with adaptive topology
- **Temporal Attention Block (TAB)**: Extracts latent temporal information
- **Very lightweight** while achieving SOTA on mainstream datasets

### Parameter Comparison for Small Datasets (<1000 samples)
| Model | Params | Notes |
|-------|--------|-------|
| EfficientGCN-B0 | 0.29M | Best lightweight option |
| SGN | 0.69M | Semantic-guided |
| PA-ResGCN-N51 | 0.77M | Residual GCN |
| AeS-GCN | ~0.5M | Attention-enhanced |
| Our v20 (ST-GCN) | 0.41M | Current model |
| Our v19 | 4.5M | Too large for 600 samples |

**RECOMMENDATION**: For 600 samples, target 0.2-0.5M parameters. EfficientGCN-B0 or a slimmed InfoGCN are the best options.

---

## 2. Multi-Stream Ensemble Strategies

### 2.1 Standard 4-Stream Ensemble (De Facto Standard)
The 4-stream approach has become the standard in skeleton-based action recognition:

1. **Joint stream**: Raw joint coordinates (x, y, z)
2. **Bone stream**: Bone vectors = child_joint - parent_joint (contains length + direction)
3. **Joint motion stream**: Temporal difference of joint positions (velocity)
4. **Bone motion stream**: Temporal difference of bone vectors

**Fusion**: Late fusion via weighted sum of softmax logits/probabilities from independently trained models.

**Why it works**: Different streams capture complementary information:
- Joint: absolute position information
- Bone: signer-invariant structural information (direction + length of bones)
- Joint motion: dynamics and speed
- Bone motion: structural change dynamics

**Key finding from literature**: "Bone data exhibited the most favorable experimental outcomes, underscoring the importance of preserving directional information among joints" - This aligns with our v19 finding that bone features help.

### 2.2 Extended Streams (5-6 streams)
Some works add additional streams:
- **Angular stream**: Angles formed between bones (Huang et al., 2024 - IET)
- **Acceleration stream**: Second-order temporal derivative
- **Relative position stream**: Joint positions relative to body center

### 2.3 Fusion Strategies
| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Late fusion (score)** | Weighted sum of logits | Simple, each model independent | Fixed weights |
| **Learned fusion (QSAW)** | Learnable weights per class | Adaptive | Needs more data |
| **Feature fusion** | Concatenate intermediate features | Richer representation | Much more parameters |
| **Early fusion (MIB)** | Fuse inputs before model | Single model, efficient | May lose stream-specific info |
| **GEM (Global Ensemble)** | SAM-SLR's late fusion approach | Proven for SLR | Needs multiple models |

### 2.4 SAM-SLR Multi-Model Ensemble (CVPR 2021 Challenge Winner)
- **Won 1st place** in CVPR 2021 Large Scale Signer Independent Isolated SLR Challenge
- **Framework**: Skeleton Aware Multi-modal SLR
- **Skeleton Models**:
  - SL-GCN (Sign Language Graph Convolution Network) - models embedded dynamics of skeleton keypoints
  - SSTCN (Separable Spatial-Temporal Convolution Network) - exploits skeleton features
- **Ensemble**: Late-fusion GEM combines skeleton predictions with RGB and depth modalities
- **Results**: 98.42% RGB, 98.53% RGB-D on AUTSL
- **Key insight for KSL**: Even skeleton-only ensemble (SL-GCN + SSTCN) provides significant gains. We should ensemble multiple skeleton models.

### 2.5 Recommendations for KSL
Given our 600 samples and 30 classes (~20 samples/class):
1. **Minimum viable ensemble**: 2-stream (joint + bone) with late fusion - proven +2-3% gain
2. **Optimal for our size**: 4-stream (joint, bone, joint_vel, bone_vel) with late fusion
3. **Weights**: Start with equal weights [1, 1, 1, 1], then tune via grid search on validation
4. **Alternative**: Early fusion (EfficientGCN-style MIB) to avoid 4x model training cost
5. **SLR-specific**: Add angular features as 5th stream (proven for hand gesture differentiation)

---

## 3. Heterogeneous Ensemble Approaches

### 3.1 GCN + Transformer Combination
- **GTC-Net**: GCN and Transformer Complementary Network allows parallel communications between GCN and Transformer domains
- **Key insight**: GCN captures local topological structure; Transformer captures global long-range dependencies
- **Combined module**: Graph convolution + self-attention perceives both local and global joint dependencies
- **Risk for KSL**: Transformers need more data, but a lightweight transformer head on GCN features could work

### 3.2 GCN + LSTM Combination
- **AGC-LSTM** (Si et al.): Embeds GCN operator into LSTM
- **Captures**: Discriminative temporal and spatial features + co-occurrence relationships
- **Key benefit**: LSTM is good at modeling sequential dependencies (important for dynamic signs)
- **Relevance to KSL**: Our legacy v8 LSTM model (83.65% val) shows LSTM works for our data. Combining with GCN features could bridge the gap.

### 3.3 CNN + GCN + LSTM (Multimodal Gesture)
- Research shows combining CNN (for image features), GCN (for skeleton topology), and LSTM (for temporal) outperforms individual architectures
- **For skeleton-only**: The GCN extracts spatial features, LSTM handles temporal patterns

### 3.4 Two-Stream Spatio-Temporal GCN-Transformer (2025)
- Latest approach combining GCN's topological modeling with Transformer's attention
- Separate spatial GCN stream and temporal Transformer stream

### 3.5 Recommendations for KSL
1. **Quick win**: Ensemble our existing v8 LSTM (649-dim features) with a new lightweight GCN model via late fusion
2. **Better approach**: GCN backbone + LSTM temporal head (hybrid architecture)
3. **Avoid**: Full transformer architectures (need too much data for 600 samples)

---

## 4. Knowledge Distillation for Skeleton Models

### 4.1 Structural Knowledge Distillation (SKD)
- **Paper**: "Structural Knowledge Distillation for Efficient Skeleton-Based Action Recognition" (IEEE 2021)
- **Approach**: Transfer structural knowledge from a large teacher model to a lightweight student
- **Preserves topology-aware features** while reducing model size

### 4.2 Part-Level Knowledge Distillation (CVPR 2024)
- **Paper**: "Enhancing Action Recognition from Low-Quality Skeleton Data via Part-Level Knowledge Distillation"
- **Key Innovation**: Teacher-student framework where teacher is trained on high-quality skeletons, student on low-quality
- **Part-level**: Distills knowledge at body-part granularity, not just global features
- **Relevance to KSL**: Could distill from a model trained on clean data to handle noisy real-world MediaPipe data

### 4.3 Self-Knowledge Distillation (SKD)
- Used in SLR to enhance model training and convergence
- The model distills knowledge from its own deeper layers to earlier layers
- **No teacher model needed** - single model self-improves
- **Very relevant for small datasets** since you don't need to train a large teacher

### 4.4 Cross-Modal Knowledge Distillation (C2VL, 2024)
- **Paper**: "Vision-Language Meets the Skeleton: Progressively Distillation with Cross-Modal Knowledge"
- Distills knowledge from vision-language models (CLIP) to skeleton models
- Could provide rich semantic features without additional skeleton data

### 4.5 Recommendations for KSL
1. **Self-Knowledge Distillation**: Add as a training technique - free improvement, no extra data needed
2. **Part-level distillation**: Relevant if we train teacher on clean lab data, student for real-world
3. **Cross-modal distillation**: Too complex for our setup, skip for now
4. **Simple ensemble distillation**: Train 4-stream ensemble, then distill into a single model for deployment

---

## 5. Angular Margin Losses (ArcFace, CosFace, SphereFace)

### 5.1 Overview
Angular margin losses were developed for face recognition but are applicable to any fine-grained classification:

| Loss | Formula Concept | Margin Type |
|------|----------------|-------------|
| **SphereFace** | multiplicative angular margin | Nonlinear angular |
| **CosFace** | additive cosine margin | Cosine space |
| **ArcFace** | additive angular margin | Linear angular (geodesic) |
| **AM-Softmax** | improved additive margin | Cosine space |

### 5.2 Why ArcFace for KSL
- **Our problem**: Confusable class pairs with cosine similarity >0.999 in feature space
- **ArcFace's solution**: Forces learned features to have larger angular separation between classes
- **Geometric interpretation**: Angular margin corresponds to geodesic distance on hypersphere
- **Key parameter**: margin `m` (typically 0.5 for face recognition, may need tuning for 30-class skeleton)
- **Scale parameter**: `s` (typically 30-64, controls feature norm)

### 5.3 Center Loss (Complementary)
- Penalizes distance between features and class centers
- **Intra-class compactness**: Pulls features of same class together
- Can be combined with ArcFace: `L = L_arcface + lambda * L_center`
- **Particularly effective for imbalanced/small datasets**

### 5.4 Contrastive / Supervised Contrastive Loss
- Recent work (SC2SLR, 2024) applies contrastive learning to skeleton-based SLR
- Multi-view contrastive learning augments data effectively
- SupCon loss: Pull same-class features together, push different-class features apart

### 5.5 Practical Implementation for KSL
```python
# ArcFace loss for skeleton recognition
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        # Add margin to target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        target_logits = torch.cos(theta + self.m * one_hot)
        logits = self.s * target_logits
        return logits
```

### 5.6 Recommendations for KSL
1. **Replace softmax with ArcFace loss** - directly addresses our confusable pairs problem
2. **Start with s=30, m=0.3** (lower margin than face recognition due to fewer classes)
3. **Add center loss with lambda=0.01** for intra-class compactness
4. **Consider SupCon loss** as a pre-training objective before fine-tuning with ArcFace

---

## 6. Domain Adversarial Neural Networks (DANN) for Cross-Subject Recognition

### 6.1 DANN Architecture
- **Three components**: Feature extractor, Label predictor, Domain classifier
- **Gradient Reversal Layer (GRL)**: Reverses gradient during backpropagation
- **Effect**: Feature extractor learns domain-invariant features that confuse the domain classifier
- **Training**: Min-max optimization - label predictor minimizes task loss, GRL maximizes domain confusion

### 6.2 Application to Skeleton Recognition
- **Source domain**: Training signers (4 signers)
- **Target domain**: Test/real-world signers (unseen)
- **Goal**: Learn features that are discriminative for sign class but invariant to signer identity
- **Key insight**: With only 4 training signers, DANN can learn features that don't encode signer-specific patterns

### 6.3 Implementation for KSL
```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class SignerAdversarialNetwork(nn.Module):
    def __init__(self, backbone, num_classes, num_signers):
        super().__init__()
        self.backbone = backbone  # Feature extractor
        self.classifier = nn.Linear(feat_dim, num_classes)  # Sign classifier
        self.signer_classifier = nn.Linear(feat_dim, num_signers)  # Signer classifier

    def forward(self, x, lambda_=1.0):
        features = self.backbone(x)
        sign_logits = self.classifier(features)
        reversed_features = GradientReversalLayer.apply(features, lambda_)
        signer_logits = self.signer_classifier(reversed_features)
        return sign_logits, signer_logits
```

### 6.4 Effectiveness Assessment
- DANN achieved 80.6% average accuracy on Amazon Reviews transfer tasks (outperforming standard NNs)
- For skeleton recognition, domain adversarial learning aligns features across different view angles and subjects
- **Critical consideration**: With only 4 training signers, we have very few "domains" - the domain classifier may not learn meaningful patterns
- **Better variant**: Use signer labels directly (not domain labels) for adversarial training

### 6.5 Recommendations for KSL
1. **Implement signer-adversarial training**: Use Leave-One-Signer-Out (LOSO) and adversarially train to confuse signer identity
2. **Lambda scheduling**: Start with lambda=0, gradually increase to 1 during training (curriculum)
3. **Combined loss**: L = L_classification + lambda * L_adversarial_signer
4. **Alternative**: Feature disentanglement - separate signer-specific features from sign-specific features
5. **Risk**: With only 4 signers, the domain classifier has very few classes to distinguish - may not provide enough gradient signal

---

## 7. Model Sizes for Very Small Datasets (<1000 samples)

### 7.1 Rules of Thumb
- **General guideline**: Parameters should not exceed 10-50x the number of training samples
- **For 600 samples**: Target 6K-30K effective parameters (but with regularization, can go up to 300K-500K)
- **Overfitting indicators**: Train accuracy >95% while val accuracy <70% (exactly our situation)

### 7.2 Strategies for Small Datasets
1. **Heavy regularization**: Dropout (0.3-0.5), weight decay (1e-3 to 1e-2), spatial/temporal dropout
2. **Data augmentation**: Random rotation, scaling, temporal crop, joint noise
3. **Early stopping**: Use validation loss, not training loss
4. **Label smoothing**: 0.1-0.2 prevents overconfident predictions
5. **Mixup / CutMix**: Virtual sample creation during training
6. **Few-shot learning**: Prototypical networks, matching networks

### 7.3 Effective Strategies from Literature
| Strategy | Typical Gain | Data Need | Implementation |
|----------|-------------|-----------|----------------|
| Multi-stream ensemble | +2-5% | Same data, 4x compute | Late fusion |
| Bone features | +1-3% | Derived from joints | Simple preprocessing |
| Angular features | +1-2% | Derived from joints | Trigonometry |
| ArcFace loss | +1-3% | Same data | Replace loss function |
| Self-knowledge distillation | +0.5-1% | Same data | Add KD loss term |
| DropGraph regularization | +1-2% | Same data | Drop nodes/edges |
| DANN (cross-subject) | +1-3% | Same data | Add adversarial head |

### 7.4 What NOT to Do with Small Datasets
- Avoid models >1M params without strong regularization
- Avoid batch normalization with very small batch sizes (<8)
- Avoid complex attention mechanisms (need many samples to learn attention patterns)
- Avoid pre-training on unrelated skeleton datasets without careful domain adaptation

---

## 8. Transformer-Based Approaches for Skeleton Sequences

### 8.1 SkateFormer (ECCV 2024)
- Partitions joints and frames based on skeletal-temporal relation types
- Performs self-attention within each partition (efficient computation)
- Action-adaptive attention focuses on key joints and frames
- **Risk for KSL**: Needs more data than GCNs

### 8.2 FreqMixFormer / FreqMixFormerV2 (2024)
- Frequency-aware mixed transformer
- Lightweight variant specifically designed for efficiency
- Combines frequency domain features with spatial-temporal attention

### 8.3 Two-Stream Spatio-Temporal GCN-Transformer (2025)
- Separate GCN spatial stream and Transformer temporal stream
- Best of both worlds: topological modeling + global attention
- Recent (January 2025) - represents current state of the art

### 8.4 MSST-RT: Multi-Stream Spatial-Temporal Relative Transformer
- Uses relative position encodings (important for sign language - relative hand positions matter)
- Multi-stream architecture with transformer backbone
- Spatial-temporal joint modeling

### 8.5 Recommendations for KSL
1. **Avoid pure transformer models** - need too much data for 600 samples
2. **Consider hybrid**: GCN backbone (spatial) + lightweight Transformer temporal head
3. **If using transformer**: Use relative position encodings (signer-invariant)
4. **FreqMixFormerV2**: Worth investigating as a lightweight transformer option

---

## 9. Prototypical Networks and Metric Learning

### 9.1 Prototypical Networks
- **Core Idea**: Learn an embedding space where classification = finding nearest class prototype (centroid)
- **Training**: Compute class prototypes from support set, classify queries by distance to prototypes
- **Key advantage for small datasets**: Simple inductive bias that is beneficial in limited-data regimes
- **Distance function**: Euclidean distance (proven superior to cosine for prototypical networks)

### 9.2 ProtoGCN (CVPR 2024) - HIGHLY RELEVANT
- **Paper**: "Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition"
- GCN-based model that decomposes skeleton dynamics into learnable prototypes
- Prototypes represent core motion patterns of action units
- SOTA on NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton, and FineGYM
- **Directly applicable to our problem** - could learn prototypical hand shapes for each sign

### 9.3 Center Loss + Prototypical Networks
- Combine prototypical classification with center loss for intra-class compactness
- Addresses class imbalance (relevant if our sign classes have uneven sample counts)
- The prototype becomes a more reliable class representative with center loss guidance

### 9.4 Metric Learning Approaches Comparison
| Approach | Key Idea | Pros for KSL | Cons |
|----------|----------|-------------|------|
| **Prototypical Networks** | Class centroids | Simple, works with few samples | Assumes spherical clusters |
| **Matching Networks** | Attention over support set | No prototype assumption | More complex |
| **Siamese Networks** | Pairwise similarity | Good with few samples | O(n^2) comparisons |
| **Triplet Loss** | Margin between pos/neg pairs | Strong embeddings | Hard mining needed |
| **SupCon** | Multi-positive contrastive | Best embeddings | Needs augmentation |

### 9.5 Recommendations for KSL
1. **Prototypical training**: Train with episodic learning (few-shot episodes) even if we have 20 samples/class
2. **ProtoGCN**: Implement the prototypical GCN approach - decomposes signs into motion primitives
3. **Combined approach**: GCN backbone + prototypical classifier head + ArcFace loss
4. **For confusable pairs**: Triplet loss with hard negative mining on the confusable pairs

---

## 10. Sign Language-Specific Architectures

### 10.1 SL-GCN (Sign Language Graph Convolution Network)
- **From SAM-SLR**: Adapted from action recognition GCN for sign language specifics
- Models embedded dynamics of skeleton keypoints
- Part of the CVPR 2021 challenge-winning ensemble

### 10.2 Hand-Aware GCN (HA-GCN)
- **Key Innovation**: Explicitly models hand topological relationships
- Two adaptive hand-graphs highlight hand topology
- Adaptive DropGraph strategy discards redundant frames and nodes
- **Critical finding**: "Training with domain-specific hand topology can single-handedly reach state-of-the-art performance"
- **Very relevant**: Our model may not be attending enough to hand structure

### 10.3 SML (Skeleton Multi-Feature Learning, 2024)
- Multiple feature types specifically designed for sign language
- Combines different skeleton representations for SLR

### 10.4 SSTCN (Separable Spatial-Temporal Convolution Network)
- Used in SAM-SLR alongside SL-GCN
- Exploits skeleton features with separable convolutions
- Lighter than full GCN approaches

### 10.5 AHG-GCN (Angle-based Hand Gesture GCN)
- **Paper**: Aiman & Ahmad, 2023 - "Angle based hand gesture recognition using graph convolutional network"
- Introduces novel edges connecting wrist to fingertips and finger bases
- 25 features per joint using angles and distances
- Results: 90%/88% on DHG 14/28 dataset, 94.05%/89.4% on SHREC 2017
- **Highly relevant**: The angle-based features between wrist and fingertips are exactly what sign language needs

### 10.6 Recommendations for KSL
1. **Adopt hand-aware topology**: Define custom graph with edges connecting wrist-to-fingertip and finger-base-to-fingertip
2. **Angular features from AHG-GCN**: Compute angles between wrist, finger base, and fingertip joints (25 features/joint)
3. **DropGraph regularization**: Randomly drop joints and temporal frames during training
4. **Hand-centric processing**: Process left hand, right hand, and body as separate subgraphs with different attention

---

## 11. Anti-Overfitting Techniques from SLR Literature

### 11.1 DropGraph (from HA-GCN)
- Spatial dropout: Randomly zero out entire joints
- Temporal dropout: Randomly skip frames
- **Removes spatial and temporal redundancy** in sign language representation
- **Specifically designed for SLR** to eliminate overfitting

### 11.2 Residual Connections
- "Residual connections from output to temporal modules can avoid overfitting during training"
- Improvements of ~3% reported when applied to SLR models

### 11.3 Contrastive Pre-Training (SC2SLR, 2024)
- Multi-view contrastive learning for skeleton-based SLR
- Augments data through contrastive augmentations before classification fine-tuning
- Addresses limited/noisy data problem common in underresourced sign languages

### 11.4 Feature Disentanglement
- Decompose features into domain-invariant (sign-specific) and domain-specific (signer-specific) components
- Only use sign-specific features for classification
- **Directly addresses our cross-signer generalization gap**

---

## 12. Comprehensive Recommendations for V21

### Priority 1 (Highest Impact, Easiest to Implement)
1. **4-stream ensemble**: Joint + Bone + Joint velocity + Bone velocity with late fusion
   - Expected gain: +2-5% over single stream
   - Implementation: Train 4 separate lightweight models, weighted sum of logits
2. **ArcFace loss**: Replace cross-entropy with ArcFace (s=30, m=0.3)
   - Expected gain: +1-3%, especially for confusable pairs
3. **Angular features as 5th stream**: Compute inter-joint angles (wrist-fingertip, finger-base-fingertip)
   - Expected gain: +1-2%, proven for hand gesture differentiation

### Priority 2 (High Impact, Moderate Effort)
4. **Hand-aware graph topology**: Custom graph connecting wrist to all fingertips
   - Expected gain: +1-2%, directly relevant for sign language
5. **Signer adversarial training**: DANN with signer labels
   - Expected gain: +1-3% on unseen signers
6. **EfficientGCN-B0 backbone** (0.29M params): Replace current ST-GCN
   - Expected gain: Better capacity-data ratio

### Priority 3 (Moderate Impact, Higher Effort)
7. **Prototypical classifier head**: Replace FC with prototypical classification
   - Expected gain: +1-2%, more robust to distribution shift
8. **Self-knowledge distillation**: Add self-KD loss term during training
   - Expected gain: +0.5-1%
9. **DropGraph regularization**: Randomly drop joints and frames
   - Expected gain: +1% through reduced overfitting

### Priority 4 (Experimental / Longer Term)
10. **Ensemble our v8 LSTM + new GCN**: Heterogeneous ensemble of existing good model + new architecture
    - High potential: Combines hand-engineered features with learned features
11. **ProtoGCN**: Decompose signs into learnable motion prototypes
12. **Hybrid GCN-LSTM architecture**: GCN spatial features fed to LSTM temporal module

### Architecture Recommendation for V21
```
Input: 30-frame skeleton sequences (63 landmarks from MediaPipe)

Feature Extraction:
  - Joint stream: [x, y, z] coordinates
  - Bone stream: child - parent joint vectors
  - Angular stream: inter-joint angles (wrist, fingertip, finger-base)
  - Velocity stream: temporal differences

Early fusion or late fusion of streams

Backbone: EfficientGCN-B0 (0.29M params) or slimmed InfoGCN
  - Hand-aware graph topology with custom edges
  - DropGraph regularization (spatial + temporal)
  - Residual connections

Classification Head: ArcFace loss (s=30, m=0.3)
  - Optional: + Center loss (lambda=0.01)
  - Optional: + Signer adversarial head with GRL

Total target params: 0.3-0.5M
Expected: ~40-50% real tester accuracy (up from 34.2%)
```

---

## References

### Architectures
- BlockGCN: Zhou et al., CVPR 2024. https://github.com/ZhouYuxuanYX/BlockGCN
- EfficientGCN: Song et al., 2022. Compound scaling for skeleton action recognition.
- InfoGCN: Chi et al., CVPR 2022. https://github.com/stnoah1/infogcn
- HD-GCN: Lee et al., ICCV 2023. https://github.com/Jho-Yonsei/HD-GCN
- SkateFormer: ECCV 2024. https://github.com/KAIST-VICLab/SkateFormer
- ShiftGCN++: Cheng et al. Extremely Lightweight Skeleton-Based Action Recognition.
- AeS-GCN: Xu et al., 2022. Attention-enhanced semantic-guided GCN.

### Sign Language Specific
- SAM-SLR: Jiang et al., CVPR 2021 Challenge Winner. https://github.com/jackyjsy/CVPR21Chal-SLR
- SAM-SLR-v2: https://github.com/jackyjsy/SAM-SLR-v2
- HA-GCN: Hand-aware GCN for skeleton-based SLR. https://doi.org/10.1016/j.nlp.2024.100074
- AHG-GCN: Aiman & Ahmad, 2023. Angle-based hand gesture recognition.
- SML: Skeleton multi-feature learning for SLR, 2024.
- SC2SLR: Skeleton contrastive learning for SLR, 2024.
- SL-GCN: Sign Language Graph Convolution Network (from SAM-SLR).

### Loss Functions
- ArcFace: Deng et al., CVPR 2019. Additive Angular Margin Loss.
- L3AM: Linear Adaptive Additive Angular Margin Loss for hand gesture authentication.
- SupCon: Khosla et al. Supervised Contrastive Learning.

### Domain Adaptation
- DANN: Ganin et al., JMLR 2016. Domain-Adversarial Training of Neural Networks.
- Feature Disentanglement for SLR: Cross-modal consistency learning.

### Few-Shot / Metric Learning
- Prototypical Networks: Snell et al., NeurIPS 2017.
- ProtoGCN: CVPR 2024. Prototypical perspective for skeleton action recognition.

### Knowledge Distillation
- Structural KD: IEEE 2021. Structural Knowledge Distillation for Skeleton Action Recognition.
- Part-Level KD: CVPR 2024. Enhancing Action Recognition from Low-Quality Skeleton Data.
- C2VL: Vision-Language Cross-Modal KD for Skeleton, 2024.

### Multi-Stream / Ensemble
- 4-stream ensemble: Standard approach (CTR-GCN, InfoGCN, etc.)
- MTSA-RGCN: Li et al., 2023. Four-stream structure for skeleton action recognition.
- DPGCN: Jang et al., 2025. Dynamic partitioning GCN with 4-stream ensemble.
- Multi-stream angular fusion: Huang et al., 2024, IET Image Processing.
