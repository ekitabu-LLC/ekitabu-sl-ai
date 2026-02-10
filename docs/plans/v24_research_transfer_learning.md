# V24 Research: Transfer Learning & Pretrained Models for Sign Language Recognition

**Date**: 2026-02-10
**Context**: KSL 30-class recognition (15 numbers + 15 words), ST-GCN on 48-node skeleton (21 left hand + 21 right hand + 6 upper body pose), 4 training signers, ~250-285 samples, v22 best real-world at 39.8%

---

## Table of Contents

1. [Pretrained ST-GCN & Skeleton-Based Models](#1-pretrained-st-gcn--skeleton-based-models)
2. [Cross-Lingual Sign Language Transfer](#2-cross-lingual-sign-language-transfer)
3. [Self-Supervised Pretraining for Sign Language](#3-self-supervised-pretraining-for-sign-language)
4. [Few-Shot & Prototypical Approaches](#4-few-shot--prototypical-approaches)
5. [Signer-Independent Recognition](#5-signer-independent-recognition)
6. [Available Pretrained Weights & Repos](#6-available-pretrained-weights--repos)
7. [Skeleton Topology Mapping Considerations](#7-skeleton-topology-mapping-considerations)
8. [Practical Recommendations for V24](#8-practical-recommendations-for-v24)

---

## 1. Pretrained ST-GCN & Skeleton-Based Models

### 1.1 SAM-SLR / SAM-SLR-v2 (CVPR 2021 Challenge Winner)

**Paper**: "Skeleton Aware Multi-modal Sign Language Recognition" (Jiang et al., CVPRW 2021)
**Repo**: https://github.com/jackyjsy/CVPR21Chal-SLR (v1), https://github.com/jackyjsy/SAM-SLR-v2 (v2)

- **Won 1st place** in CVPR 2021 Large Scale Signer Independent Isolated SLR Challenge
- Proposes **SL-GCN** (Sign Language GCN) + **SSTCN** (Separable Spatial-Temporal Convolution Network)
- Trained on **AUTSL** dataset (226 signs, 36 signers, ~36K samples)
- Achieves 98.42% (RGB) and 98.53% (RGB-D) on AUTSL
- **Pretrained models available** for all modalities including skeleton
- Processed skeleton data for AUTSL released
- **Relevance**: SL-GCN is specifically designed for sign language (not generic actions), and pretrained on a large signer-independent SLR dataset. Skeleton stream could be fine-tuned for KSL.

### 1.2 DSTA-SLR (COLING 2024)

**Paper**: "Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition" (Hu et al., LREC-COLING 2024)
**Repo**: https://github.com/hulianyuyy/DSTA-SLR

- State-of-the-art skeleton-only method on WLASL, MSASL, NMFs-CSL, SLR500
- Two concurrent spatial branches + temporal multi-scale module
- **Pretrained weights on WLASL2000 available** in `./pretrained_models`
- Outperforms RGB-based methods in most cases with much fewer computational resources
- Supports WLASL100/300/1000/2000, MLASL variants, SLR500, NMFs-CSL
- **Relevance**: WLASL2000 pretrained weights are the most directly useful for transfer to KSL since WLASL is skeleton-based isolated SLR.

### 1.3 ST-GCN-SL (Amorim et al., 2019)

**Paper**: "Spatial Temporal Graph Convolutional Networks for Sign Language Recognition"
**Repo**: https://github.com/amorim-cleison/st-gcn-sl

- Adapted original ST-GCN for sign language on ASLLVD dataset
- Pretrained weights available for download
- Based on the foundational ST-GCN architecture (Yan et al., AAAI 2018)
- **Relevance**: Direct ST-GCN adaptation for SLR, but older architecture and smaller dataset.

### 1.4 SPOTER / SPOTER2 (WACV 2022 / JSALT 2024)

**Paper**: "Sign Pose-based Transformer for Word-level Sign Language Recognition" (Bohacek et al.)
**Repo**: https://github.com/maty-bohacek/spoter (v1), https://github.com/JSALT2024/spoter2 (v2)

- Transformer-based model working on skeleton sequences
- SPOTER2 adds **encoder pretraining** on How2Sign dataset
- Open-sourced skeleton data for WLASL100 and LSA64
- "Dominance on small instance datasets" and "more suitable for applications in the wild"
- Apache 2.0 license for code
- **Relevance**: Specifically designed for small datasets and in-the-wild deployment; encoder pretraining could transfer well.

### 1.5 pyskl Toolbox

**Repo**: https://github.com/kennymckormick/pyskl

- Unified toolbox for skeleton-based action recognition
- Supports: ST-GCN, ST-GCN++, CTR-GCN, MS-G3D, AAGCN, DG-STGCN, PoseConv3D
- Pretrained models on **NTU RGB+D 60/120**, Kinetics-400, UCF101, HMDB51
- Both 2D (HRNet) and 3D (Kinect) skeleton formats
- Tools for extracting 2D skeletons from RGB videos
- **Relevance**: While action recognition pretrained (not SLR), these are robust spatial-temporal feature extractors. CTR-GCN or MS-G3D pretrained on NTU120 could provide useful initialization, especially for upper body/hand motion patterns.

---

## 2. Cross-Lingual Sign Language Transfer

### 2.1 Cross-Lingual Few-Shot SLR (Bilge et al., Pattern Recognition 2024)

**Paper**: "Cross-lingual few-shot sign language recognition" (Bilge, Ikizler-Cinbis, Cinbis, Pattern Recognition, Vol 151, 2024)

- Formulates cross-lingual FSSLR: models trained on ASL can recognize signs in other languages
- Embedding-based framework using spatio-temporal visual representation from video, hand features, and hand landmarks
- Three meta-learning FSSLR benchmarks spanning multiple languages
- Demonstrates that "pretraining is effective for sign language recognition, especially in low-resource settings"
- Shows "high crosslingual transfer from Indian-SL to few other sign languages"
- **Key finding**: Cross-lingual transfer works because many sign languages share common handshape and motion patterns
- **Relevance**: Directly applicable -- pretrain on large ASL/ISL dataset, fine-tune on KSL with few shots

### 2.2 Multilingual Skeleton-Based SLR (2024)

**Paper**: "An effective skeleton-based approach for multilingual sign language recognition" (Engineering Applications of AI, 2024)

- EfficientNet + multi-feature attention mechanism
- Transfer learning paradigm: general visual features via pretrained EfficientNet
- Represents each sign using 7 skeleton images capturing key joint positions
- Spatial (CNN) + temporal (RNN) modeling
- **Relevance**: Demonstrates skeleton images as transferable representations across languages

### 2.3 Scaling Sign Language Translation (2024)

**Paper**: "Scaling Sign Language Translation" (arXiv 2407.11855)

- Shows that adding more multilingual SLT data reduces the modality gap
- Skeleton representations benefit from cross-lingual data scaling
- Demonstrates feasibility of zero-shot SLT
- **Relevance**: More training data from other sign languages, even different ones, helps

---

## 3. Self-Supervised Pretraining for Sign Language

### 3.1 SHuBERT (ACL 2025 Oral) -- HIGHEST POTENTIAL

**Paper**: "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction" (Gueuwou et al., ACL 2025)
**Repo**: https://github.com/ShesterG/SHuBERT
**Website**: https://shubert.pals.ttic.edu/
**Demo**: Available on Hugging Face Spaces

- **Pretrained on ~1,000 hours of ASL video**
- Multi-stream: upper body pose + left hand + right hand + face
- Uses **MediaPipe landmarks + DINOv2 features** for hands/face
- Transformer encoder with masked prediction (inspired by HuBERT for speech)
- State-of-the-art on: How2Sign (+0.7 BLEU), OpenASL (+10.0 BLEU), FLEURS-ASL (+0.3 BLEU)
- For isolated SLR: ASL-Citizen (+5%), SEM-LEX (+20.6%), WLASL2000
- **Key advantage**: Self-supervised on massive unlabeled ASL data, then fine-tuned
- **Relevance**: HIGH. Pretrained representations from 1000 hours of signing could dramatically improve generalization. Uses MediaPipe like our project. Fine-tuning on 250 KSL samples with frozen/partially-frozen encoder could break the 4-signer ceiling.

### 3.2 SignBERT+ (ICCV 2021 / TPAMI 2023)

**Paper**: "SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding" (Hu et al.)
**Website**: https://signbert-zoo.github.io/
**Unofficial Implementation**: https://github.com/joshuasv/signbert_unofficial

- First self-supervised framework with hand-model-aware prior for SLR
- Views hand pose as visual token, embedded with gesture state + temporal + chirality info
- Tested on isolated SLR, continuous SLR, and SLT tasks
- **Relevance**: Hand-model-aware pretraining aligns well with our hand-centric skeleton (42 of 48 nodes are hand joints)

### 3.3 Sigma (Under Review, 2025)

**Paper**: "Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding" (arXiv 2509.21223)

- Uses 2D keypoints via **RTM-Pose**, organized into 4 part-specific groups: left hand, right hand, body, face
- Part-specific ST-GCNs to model joint interdependencies
- Sign-Aware Early Fusion + Hierarchical Alignment Learning + Sign-Grounded Text Encoder
- WLASL2000: 64.40% per-instance accuracy
- **Code promised upon acceptance**
- **Relevance**: Part-specific ST-GCN approach aligns with multi-stream ideas; semantic grounding reduces signer sensitivity

### 3.4 SC2SLR (CNIOT 2024)

**Paper**: "SC2SLR: Skeleton-based Contrast for Sign Language Recognition" (ACM CNIOT 2024)

- Multi-view contrastive learning on skeleton representations
- Addresses limited dataset sizes through contrastive data augmentation
- Skeleton keypoint-based weighting to focus on key image regions
- **Relevance**: Contrastive pretraining could help learn signer-invariant skeleton features

### 3.5 SL-SLR (2025)

**Paper**: "SL-SLR: Self-Supervised Representation Learning for Sign Language Recognition" (arXiv 2509.05188)

- Self-supervised approach specifically for SLR
- **Relevance**: Another self-supervised option, but fewer details available

---

## 4. Few-Shot & Prototypical Approaches

### 4.1 Few-Shot Prototypical Network for SLR (Saad, Dec 2025) -- MOST DIRECTLY APPLICABLE

**Paper**: "Data-Efficient American Sign Language Recognition via Few-Shot Prototypical Networks" (arXiv 2512.10562)

- **ST-GCN backbone + Multi-Scale Temporal Aggregation (MSTA)** with parallel 1D conv (kernels 3, 5, 7)
- Episodic training: K=3 shots, Q=2 queries, tested 50/100/200-way
- Embedding dimension: 768D with attention-based temporal pooling
- Pose extraction via RTMLib (RTMPose-l, COCO-WholeBody)
- **Results on WLASL**:
  - Prototypical (200-Way): **43.75% Top-1, 77.10% Top-5, 84.50% Top-10**
  - Standard Classifier: 30.37% Top-1 (13pp improvement from prototypical)
  - Zero-shot on SignASL: **29.87%** without fine-tuning
- 50,000 training episodes
- **Key insight**: Metric learning outperforms standard classification on scarce data by 13pp
- **Relevance**: EXTREMELY HIGH. This is almost our exact scenario -- small dataset, skeleton-based, need generalization. The 13pp improvement from prototypical learning alone is significant. With 250 KSL samples, episodic training could dramatically help.

### 4.2 ProtoSign: Few-Shot SLR for Low-Resource Languages

**Paper**: "Sign Language Recognition for Low Resource Languages Using Few Shot Learning" (Springer 2024)

- Prototypical network specifically for low-resource sign languages
- Designed for scenarios with very few examples per class
- **Relevance**: Directly targets our low-resource KSL scenario

### 4.3 Cross-Lingual FSSLR Benchmarks (Bilge et al., 2024)

- (See Section 2.1 above)
- Three benchmark configurations spanning multiple sign languages
- Demonstrates that ASL pretrained models generalize to unseen sign languages with few examples

---

## 5. Signer-Independent Recognition

### 5.1 Ethiopian Sign Language Framework (Scientific Reports, 2025)

**Paper**: "A deep learning framework for Ethiopian sign language recognition using skeleton-based representation"

- MediaPipe Holistic for skeleton extraction
- 5,600 annotated sign videos
- CNN-LSTM, LSTM, BiLSTM, GRU architectures
- **Signer-dependent: 94%** vs **Signer-independent: 73%** (21pp gap)
- **Key insight**: Even with more data, signer-independent gap persists at ~20pp
- **Relevance**: Shows the signer-independence challenge is universal; 73% with more signers suggests our ceiling is data-limited

### 5.2 Cross-Graph Domain Adaptation (CRV 2024)

**Paper**: "Cross-Graph Domain Adaptation for Skeleton-based Human Action Recognition" (Conference on Robots and Vision, 2024)

- Adversarial learning for skeleton domain adaptation across different graph topologies
- Source domain -> target domain with different skeleton configurations
- Action-aware and domain-agnostic knowledge extraction
- **Relevance**: Could help adapt pretrained models with different skeleton formats to our 48-node topology

### 5.3 Signer-Invariant Conformer (ICCVW 2025)

**Paper**: "Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous SLR" (ICCV Workshop 2025)

- Explicitly addresses signer invariance at the architecture level
- Multi-scale fusion for capturing sign variations across signers
- **Relevance**: Architecture-level signer invariance is exactly what we need

---

## 6. Available Pretrained Weights & Repos Summary

| Model | Dataset | Skeleton Format | Weights Available | Repo |
|-------|---------|-----------------|-------------------|------|
| **SAM-SLR/SL-GCN** | AUTSL (226 signs) | Whole body | Yes | [GitHub](https://github.com/jackyjsy/CVPR21Chal-SLR) |
| **DSTA-SLR** | WLASL2000 | Upper body + hands | Yes (./pretrained_models) | [GitHub](https://github.com/hulianyuyy/DSTA-SLR) |
| **ST-GCN-SL** | ASLLVD | Upper body + hands | Yes | [GitHub](https://github.com/amorim-cleison/st-gcn-sl) |
| **SHuBERT** | 1000h ASL video | MediaPipe + DINOv2 | Yes (checkpoint) | [GitHub](https://github.com/ShesterG/SHuBERT) |
| **SPOTER2** | How2Sign + WLASL | MediaPipe-like | Partial | [GitHub](https://github.com/JSALT2024/spoter2) |
| **pyskl (CTR-GCN)** | NTU120 | 25-joint Kinect | Yes | [GitHub](https://github.com/kennymckormick/pyskl) |
| **pyskl (MS-G3D)** | NTU120 | 25-joint Kinect | Yes | [GitHub](https://github.com/kennymckormick/pyskl) |
| **SignBERT+** | Custom SLR data | Hand model | Unofficial only | [GitHub](https://github.com/joshuasv/signbert_unofficial) |

---

## 7. Skeleton Topology Mapping Considerations

Our KSL model uses 48 nodes: 21 left hand + 21 right hand + 6 upper body pose.

### Mapping Challenges:
1. **WLASL/AUTSL pretrained models** typically use 27-67 keypoints from OpenPose or MediaPipe with different topologies
2. **NTU RGB+D models** use 25-joint Kinect format (no detailed hand joints)
3. **MediaPipe Holistic** provides 33 pose + 21 per hand + 468 face = 543 total

### Adaptation Strategies:
- **Subset mapping**: Map our 6 upper body joints to corresponding joints in pretrained model, map hand joints directly (most pretrained SLR models use 21-joint hand representation matching MediaPipe)
- **Zero-padding**: Add zero-masked dummy nodes for unmatched joints (standard practice, see BHaRNet, arXiv 2601.00369)
- **Re-initialize unmatched layers**: Keep pretrained weights for matching graph structure, randomly initialize for new topology
- **Two-stage**: Pretrain backbone on large SLR dataset with full topology, then fine-tune on KSL with our 48-node subset

### Best Compatibility:
Our 48-node setup (21+21 hands + 6 body) is most compatible with models using **MediaPipe hand landmarks** (21 per hand) which includes: DSTA-SLR, SPOTER/SPOTER2, SHuBERT (uses MediaPipe), and the Few-Shot Prototypical Network (uses RTMPose-l with similar hand topology).

---

## 8. Practical Recommendations for V24

### Tier 1: Highest Impact, Most Feasible

#### A. DSTA-SLR Pretrained Transfer (RECOMMENDED FIRST)
- **Why**: WLASL2000 pretrained weights available, skeleton-based, state-of-the-art
- **How**: Download pretrained model, map our 48-node skeleton to DSTA-SLR format, freeze early layers, fine-tune classification head + last 1-2 GCN blocks on KSL
- **Expected gain**: 10-20pp from pretrained initialization alone (based on cross-lingual transfer results)
- **Effort**: Medium (topology mapping + fine-tuning pipeline)
- **Risk**: Topology mismatch may require careful joint mapping

#### B. Prototypical Network Classification (RECOMMENDED)
- **Why**: 13pp gain over standard classification on small datasets (arXiv 2512.10562)
- **How**: Replace softmax classifier with episodic prototypical training; keep ST-GCN backbone, add MSTA module, train with 3-shot episodes
- **Expected gain**: 5-15pp from metric learning alone
- **Effort**: Low-Medium (modify training loop, keep architecture similar)
- **Risk**: Low -- can be combined with any backbone

#### C. SAM-SLR SL-GCN Pretrained Transfer
- **Why**: Specifically designed for signer-independent SLR, pretrained on AUTSL (36 signers)
- **How**: Use SL-GCN pretrained weights, adapt to our skeleton format, fine-tune on KSL
- **Expected gain**: 10-15pp (signer-independent pretraining directly addresses our weakness)
- **Effort**: Medium (need to install SAM-SLR framework, adapt data pipeline)
- **Risk**: AUTSL uses whole body + depth; skeleton-only stream may have weaker pretrained features

### Tier 2: High Impact, More Complex

#### D. SHuBERT Feature Extraction
- **Why**: 1000 hours of self-supervised ASL pretraining, uses MediaPipe (same as us)
- **How**: Extract SHuBERT embeddings for our skeleton sequences, train a lightweight classifier on top
- **Expected gain**: Potentially 15-25pp (massive pretraining data advantage)
- **Effort**: High (need to run SHuBERT inference, handle multi-stream input format)
- **Risk**: SHuBERT also uses DINOv2 visual features for hands/face which we don't have (skeleton only)

#### E. Contrastive Pretraining on Combined SLR Data
- **Why**: Self-supervised on large combined SLR datasets, then fine-tune on KSL
- **How**: Combine WLASL + AUTSL + MSASL skeleton data, contrastive pretraining (like SC2SLR), then fine-tune
- **Expected gain**: 10-20pp
- **Effort**: High (data collection, preprocessing, contrastive training pipeline)
- **Risk**: Different skeleton formats need harmonization

### Tier 3: Future Work

#### F. Cross-Lingual Few-Shot Meta-Learning
- Pretrain meta-learner on multiple sign languages, adapt to KSL with few shots
- Requires multi-language SLR data collection

#### G. SHuBERT Full Fine-Tuning
- Full model with video features (not just skeleton)
- Requires changing inference pipeline to include RGB

### Implementation Priority for V24:

1. **Start with B (Prototypical Network)** -- lowest risk, can keep v22 backbone, just change training paradigm
2. **Then add A (DSTA-SLR transfer)** -- download weights, implement skeleton mapping, fine-tune
3. **If time permits, try D (SHuBERT)** -- feature extraction as fixed encoder, train classifier on embeddings

### Key Numbers to Beat:
- V22 real-world: 39.8% (no-TTA)
- V22 val: 93.3%
- Target: >50% real-world

### Realistic Expectations:
- Prototypical training alone: ~45-50% real-world (based on 13pp gain from WLASL results)
- Pretrained transfer + prototypical: ~50-60% real-world
- SHuBERT features + prototypical: ~55-65% real-world (if compatible)
- **Note**: These are optimistic estimates. The 4-signer ceiling is fundamentally a data diversity problem. Transfer learning can help by providing priors from diverse signers in other datasets, but collecting more KSL signers remains the most reliable path to >70%.

---

## References

1. Jiang et al. "Skeleton Aware Multi-modal Sign Language Recognition" CVPRW 2021 -- https://github.com/jackyjsy/CVPR21Chal-SLR
2. Hu et al. "Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition" LREC-COLING 2024 -- https://github.com/hulianyuyy/DSTA-SLR
3. Saad. "Data-Efficient American Sign Language Recognition via Few-Shot Prototypical Networks" arXiv 2512.10562 -- https://arxiv.org/abs/2512.10562
4. Bilge et al. "Cross-lingual few-shot sign language recognition" Pattern Recognition 2024
5. Gueuwou et al. "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction" ACL 2025 -- https://github.com/ShesterG/SHuBERT
6. Hu et al. "SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding" TPAMI 2023 -- https://signbert-zoo.github.io/
7. Bohacek et al. "Sign Pose-based Transformer for Word-level Sign Language Recognition" WACV 2022 -- https://github.com/maty-bohacek/spoter
8. Amorim et al. "Spatial Temporal Graph Convolutional Networks for Sign Language Recognition" 2019 -- https://github.com/amorim-cleison/st-gcn-sl
9. pyskl toolbox -- https://github.com/kennymckormick/pyskl
10. "Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding" arXiv 2509.21223
11. "SC2SLR: Skeleton-based Contrast for Sign Language Recognition" ACM CNIOT 2024
12. Ethiopian SL framework, Scientific Reports 2025 -- https://www.nature.com/articles/s41598-025-19937-0
13. "Cross-Graph Domain Adaptation for Skeleton-based Human Action Recognition" CRV 2024
14. HA-GCN: Hand-aware Graph Convolution Network -- https://github.com/snorlaxse/HA-SLR-GCN
15. "An effective skeleton-based approach for multilingual sign language recognition" Engineering Applications of AI 2024
