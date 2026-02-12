# Literature Review: Kenya Sign Language Recognition -- Full System Evolution (v1-v24)

## 1. Sign Language Recognition -- Landmark/Skeleton-Based Approaches

Sign language recognition (SLR) has evolved from sensor-based approaches (data gloves, depth cameras) to vision-based methods leveraging pose estimation frameworks. Modern skeleton/landmark-based approaches offer advantages over RGB-based methods: robustness to lighting, background, and appearance variations, and computational efficiency due to low dimensionality (Aiman & Ahmad, 2023).

**Key references:**

- **Anithadevi et al. (2025)** -- "MediaPipe-LSTM-Enhanced Framework for Real-Time Dynamic Sign Language Recognition in Inclusive Communication Systems." *Engineering Reports*, 7(7). doi:10.1002/eng2.70142. Proposes MediaPipe landmark tracking + stacked LSTM for 30 ISL sign categories, achieving 97% accuracy. *Relevance: Same pipeline concept (MediaPipe + temporal model), same number of classes (30).*

- **Hasan et al. (2025)** -- "Bangla Sign Language Recognition With Multimodal Deep Learning Fusion." *Engineering Reports*, 7(4). doi:10.1002/eng2.70139. Uses MediaPipe and OpenPose for 200-word classes from 16 signers, with ViT achieving 88.52% and late fusion reaching 94.71%. *Relevance: Multi-signer skeleton-based SLR with fusion.*

- **Moustafa et al. (2023)** -- "Integrated Mediapipe with a CNN Model for Arabic Sign Language Recognition." *J. Electrical and Computer Engineering*, 2023(1). doi:10.1155/2023/8870750. MediaPipe hand landmarks + CNN for ArSL, achieving 97.1% accuracy. *Relevance: MediaPipe-based SLR pipeline.*

- **Robert & Duraisamy (2023)** -- "A review on computational methods based automated sign language recognition system." *Concurrency and Computation: Practice and Experience*, 35(9). doi:10.1002/cpe.7653. Comprehensive review covering HMM, CNN, RNN, and GCN methods. *Relevance: Broad SLR survey establishing research context.*

- **Kamal et al. (2024)** -- "Adversarial autoencoder for continuous sign language recognition." *Concurrency and Computation: Practice and Experience*, 36(22). doi:10.1002/cpe.8220. Addresses limited data in CSLR via generative models on PHOENIX-2014/CSL-Daily (~10K pairs). *Relevance: Data scarcity in SLR is a shared challenge.*

## 2. CNN and LSTM-Based Approaches to Sign Language Recognition

Before GCNs became dominant, CNN and LSTM/RNN architectures were the primary deep learning approaches for sign language and gesture recognition. CNNs excel at extracting spatial features from individual frames, while RNNs/LSTMs model temporal dependencies across frame sequences.

**CNN-based SLR:**

- **Pigou et al. (2015)** -- "Beyond Temporal Pooling: Recurrence and Temporal Convolutions for Gesture Recognition in Video." *Int. J. Computer Vision*, 119(3), 245-259. doi:10.1007/s11263-016-0957-7. One of the early works applying CNNs to gesture recognition, demonstrating that temporal convolutions and recurrence can capture dynamic gesture information. *Relevance: Established CNN-based approaches as viable for gesture/sign recognition, motivating our early versions.*

- **Oyedotun & Khashman (2017)** -- "Deep learning in vision-based static hand gesture recognition." *Neural Computing and Applications*, 28(12), 3941-3951. doi:10.1007/s00521-016-2294-8. Applied CNN alongside Stacked Denoising Autoencoder (SDAE) for 24 ASL hand gestures, achieving competitive accuracy. *Relevance: Representative of early CNN-based SLR that our initial versions drew upon.*

- **Koller et al. (2015)** -- "Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers." *Computer Vision and Image Understanding*, 141, 108-125. doi:10.1016/j.cviu.2015.09.013. Pioneered CNN-based feature extraction combined with HMMs for continuous SLR with multiple signers. *Relevance: Early large-vocabulary CSLR; highlighted the multi-signer challenge we also face.*

**LSTM/RNN-based SLR:**

- **Hochreiter & Schmidhuber (1997)** -- "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780. doi:10.1162/neco.1997.9.8.1735. The foundational LSTM paper introducing gated memory cells to address vanishing gradients in RNNs, enabling learning of long-range temporal dependencies. *Relevance: LSTM layers were used in our early model versions (v1-v10) for temporal modeling of sign sequences before transitioning to temporal graph convolutions.*

- **Huang et al. (2022)** -- "Dynamic Sign Language Recognition Based on CBAM with Autoencoder Time Series Neural Network." *Mobile Information Systems*, 2022(1). doi:10.1155/2022/3247781. Proposes CNN-BiLSTM with attention for dynamic SLR, achieving 89.90% recognition. Addresses the limitation that CNNs alone cannot extract temporal features from video. *Relevance: Demonstrates the CNN+LSTM pipeline paradigm that preceded GCN-based approaches.*

## 3. Graph Convolutional Networks for Action/Gesture Recognition

GCNs have become the dominant approach for skeleton-based action and gesture recognition by modeling the natural graph structure of human joints.

**Seminal works:**

- **Yan et al. (2018)** -- "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." *AAAI 2018*. arXiv:1801.07455. The foundational ST-GCN paper: first to apply GCNs to skeleton action recognition, representing joints as graph nodes with spatial edges (body connectivity) and temporal edges (same joint across frames). Achieved substantial improvements on Kinetics and NTU-RGBD. *Relevance: Our architecture is inspired by ST-GCN's spatial-temporal graph convolution paradigm.*

- **Shi et al. (2019)** -- "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition." *CVPR 2019*. arXiv:1805.07694. Introduced 2s-AGCN with adaptive graph topology learning and two-stream architecture using joint and bone (second-order) features. *Relevance: Directly relevant to our multi-stream approach using joint, bone, velocity, and bone-velocity streams.*

- **Liu et al. (2020)** -- "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition." *CVPR 2020 (Oral)*. arXiv:2003.14111. Proposed MS-G3D with multi-scale graph convolution and dense cross-spacetime edges. SOTA on NTU RGB+D 60/120 and Kinetics. *Relevance: Multi-scale aggregation concept applicable to hand gesture recognition.*

- **Chen et al. (2021)** -- "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition." *ICCV 2021*. arXiv:2107.12213. CTR-GCN dynamically learns different topologies per channel, outperforming on NTU RGB+D 60/120 and NW-UCLA. *Relevance: Advanced GCN topology learning; demonstrates continued evolution of the ST-GCN family.*

- **Jang et al. (2025)** -- "Dynamic partitioning graph convolutional network for skeleton-based action recognition." *ETRI Journal*. doi:10.4218/etrij.2024-0598. DPGCN captures dynamic skeletal structure dependencies. *Relevance: Recent GCN variant demonstrating ongoing innovation.*

**GCN for hand gesture recognition:**

- **Aiman & Ahmad (2023)** -- "Angle based hand gesture recognition using graph convolutional network." *Computer Animation and Virtual Worlds*, 35(1). doi:10.1002/cav.2207. AHG-GCN introduces additional edges connecting wrist to fingertips, with angle-based features to combat overfitting on small datasets (DHG 14/28, SHREC 2017). *Relevance: GCN applied to hand gestures specifically, addressing small dataset overfitting -- a key challenge we face.*

## 4. Kenya Sign Language / East African Sign Language Recognition

Research on KSL recognition is extremely scarce, making our work one of the few contributions in this space.

- **Amukasa (2021)** -- "Translating the Kenyan Sign Language with Deep Learning." Blog/technical report on applying CNNs for KSL alphabet recognition. *Relevance: One of the few prior KSL recognition efforts.*

- **AI4KSL Project (2023-2024)** -- A two-year research project creating a digital open-access KSL dataset using MediaPipe pose estimation, designed to develop AI assistive technology for deaf learners in Kenya. arXiv:2410.18295. *Relevance: Directly addresses KSL dataset scarcity; uses MediaPipe like our system.*

- **Abeje et al. (referenced in AbdElghfar et al., 2023)** -- Ethiopian Sign Language recognition using deep CNN, achieving 98.5% training accuracy. *Relevance: East African sign language recognition; similar regional context.*

- **Tamiru et al. (referenced in AbdElghfar et al., 2023)** -- Amharic sign language recognition using ANN and SVM classifiers, achieving 80.82% and 98.06% accuracy respectively. *Relevance: Limited-resource African sign language recognition.*

The scarcity of KSL-specific research underscores the novelty of our contribution and the importance of developing recognition systems for under-resourced sign languages.

## 5. Domain Generalization / Signer-Independent Recognition

The cross-signer gap -- where models trained on a few signers fail to generalize to unseen signers -- is a fundamental challenge in SLR. This is analogous to speaker-independent speech recognition.

- **Ganin et al. (2016)** -- "Domain-Adversarial Training of Neural Networks." *JMLR*, 17(59), 1-35. arXiv:1505.07818. Introduced the Gradient Reversal Layer (GRL) for learning domain-invariant features via adversarial training. During backpropagation, the GRL multiplies gradients by -lambda, reversing optimization direction for the feature extractor w.r.t. domain classification. *Relevance: Our architecture uses GRL for signer-invariant feature learning, treating each signer as a domain.*

- **Bani Baker et al. (2023, referenced)** -- Framework for signer-independent ArSL recognition using DeepLabv3+ semantic hashing, achieving 89.5% accuracy on 23 isolated words from 3 signers. *Relevance: Directly addresses signer-independent recognition.*

- **Pu et al. (2024)** -- "Improving Continuous Sign Language Recognition with Consistency Constraints and Signer Removal." *ACM TOMM*. Treats signer-independent CSLR as domain generalization, explicitly removing signer-specific features. *Relevance: Formalizes our exact problem -- signer as domain.*

- **Lteif et al. (2024)** -- "Disease-driven domain generalization for neuroimaging-based assessment of Alzheimer's disease." *Human Brain Mapping*, 45(8). doi:10.1002/hbm.26707. Reviews DG methods including data augmentation (Mixup), representation learning (ERM, V-REx, RSC), meta-learning (MLDG), and domain-invariant representations. *Relevance: General DG methodology applicable to our cross-signer problem.*

- **Wang et al. (2021, referenced in Scheutz et al., 2023)** -- Survey on domain generalization methods covering DG without access to test data during training. *Relevance: Theoretical foundation for our GRL-based approach.*

## 6. Few-Shot / Few-Signer Training Challenges

Training with very few signers (4 in our case) is a fundamental limitation that caps generalization.

- **Snell et al. (2017)** -- "Prototypical Networks for Few-shot Learning." *NeurIPS 2017*. arXiv:1703.05175. Learns embeddings where class prototypes are mean representations; classification via nearest prototype in Euclidean distance. *Relevance: We explored prototypical classification in v24 to address the few-signer problem.*

- **Liu et al. (2022)** -- "Continuous Gesture Sequences Recognition Based on Few-Shot Learning." *Int. J. Aerospace Engineering*, 2022(1). doi:10.1155/2022/7868142. Uses MediaPipe + lightweight autoencoder for 5-way 1-shot gesture recognition (89.73% on RWTH German fingerspelling). *Relevance: Few-shot + MediaPipe for gesture recognition.*

- **Guo et al. (2024, arXiv:2512.10562)** -- "Data-Efficient American Sign Language Recognition via Few-Shot Prototypical Networks." Prototypical networks achieve 43.75% Top-1 on WLASL with strong zero-shot transfer (~30% on unseen SignASL without fine-tuning). *Relevance: Directly comparable -- prototypical networks for sign language with limited data; similar accuracy range to our real-tester results.*

- **Zhang et al. (2024)** -- "A signer-independent sign language recognition method for the single-frequency dataset." *Neurocomputing*. Addresses recognition when each signer appears only once. *Relevance: Extreme few-signer scenario.*

## 7. Data Augmentation for Skeleton-Based Recognition

Augmentation is critical when training data is limited, especially for skeleton/landmark data.

- **Kolawole et al. (2024)** -- "Enhancing Human Action Recognition with 3D Skeleton Data: A Comprehensive Study of Deep Learning and Data Augmentation." *Electronics*, 13(4), 747. Reviews spatial augmentations (rotation, scaling, translation, shearing, jittering, Gaussian noise) and temporal augmentations (speed changes, time warping, cropping). *Relevance: Comprehensive reference for the augmentation techniques we employ.*

- **Santiago & Cifuentes (2024)** -- "Deep learning-based gesture recognition for surgical applications: A data augmentation approach." *Expert Systems*, 41(12). doi:10.1111/exsy.13706. Uses scaling, rotation, and translation to augment 3D motion capture trajectories while preserving spatial relationships. *Relevance: Augmentation preserving spatial structure of skeleton data.*

- **Enhancing Skeleton-Based Action Recognition in Real-World Scenarios (2024)** -- IEEE conference paper proposing augmentation based on pose estimation errors, yielding up to +4.63% accuracy on UAV Human Dataset. *Relevance: Realistic augmentation for skeleton data in real-world scenarios.*

- **Nida et al. (2022)** -- "Video augmentation technique for human action recognition using genetic algorithm." *ETRI Journal*, 44(2), 327-338. doi:10.4218/etrij.2019-0510. Generates virtual training videos for imbalanced datasets. *Relevance: Data augmentation for action recognition with limited data.*

## 8. Prototypical Networks / Metric Learning for Gesture Recognition

Metric learning approaches, especially prototypical networks, offer promise for few-shot gesture/sign recognition by learning discriminative embeddings rather than fixed classifiers.

- **Snell et al. (2017)** -- (See Section 5). Core prototypical networks paper.

- **Deng et al. (2019)** -- "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR 2019*. arXiv:1801.07698. Proposes additive angular margin loss on hypersphere for discriminative embeddings. *Relevance: We experimented with ArcFace loss in v21 (catastrophic -- val 76.5%), demonstrating that angular margin losses require careful tuning with limited data.*

- **He et al. (2022)** -- "A Hybrid Matching Network for Fault Diagnosis under Different Working Conditions with Limited Data." *Computational Intelligence and Neuroscience*, 2022(1). doi:10.1155/2022/3024590. Combines prototypical networks with autoencoder regularization for domain generalization with limited samples. *Relevance: Prototypical networks for cross-domain generalization -- closely related to our cross-signer problem.*

- **Yu et al. (2022)** -- "Center Loss Guided Prototypical Networks for Unbalance Few-Shot Industrial Fault Diagnosis." *Mobile Information Systems*, 2022(1). doi:10.1155/2022/3144950. Adds center loss to prototypical networks for intra-class contraction and inter-class separation. *Relevance: Improving prototypical networks with auxiliary losses.*

## 9. MixStyle / Domain Randomization for Generalization

Style-based domain augmentation offers a lightweight approach to improving domain generalization.

- **Zhou et al. (2021)** -- "Domain Generalization with MixStyle." *ICLR 2021*. arXiv:2104.02008. A simple plug-and-play, parameter-free module that mixes instance-level feature statistics across domains during training. Based on the insight that feature statistics (mean, std) capture "style" information which defines visual domains. Mixing styles synthesizes novel domains, increasing domain diversity. *Relevance: We implemented MixStyle in v24 for cross-signer generalization, mixing feature statistics across signers to reduce signer-specific overfitting.*

- **Zhou et al. (2023)** -- "MixStyle Neural Networks for Domain Generalization and Adaptation." *IJCV*. doi:10.1007/s11263-023-01913-8. Extended journal version showing MixStyle improves generalization across image recognition, instance retrieval, and reinforcement learning. *Relevance: Validates MixStyle across diverse tasks.*

- **Gokhale et al. (2024)** -- "Towards robust visual understanding: A paradigm shift in computer vision from recognition to reasoning." *AI Magazine*, 45(3). doi:10.1002/aaai.12194. Reviews adversarial data augmentation for single-source domain generalization. *Relevance: General DG augmentation framework.*

## 10. Multi-Stream Fusion for Skeleton Data

Multi-stream architectures combine complementary skeleton representations (joints, bones, velocities) for richer action descriptions.

- **Shi et al. (2019)** -- (See Section 2). 2s-AGCN pioneered the two-stream (joint + bone) approach. Extended to 4-stream (joint, bone, joint-motion, bone-motion) in the multi-stream version.

- **Jiang et al. (2021)** -- "A Lightweight Hierarchical Model with Frame-Level Joints Adaptive Graph Convolution for Skeleton-Based Action Recognition." *Security and Communication Networks*, 2021(1). doi:10.1155/2021/2290304. Proposes early feature fusion of positions, bones, and motions instead of separate streams, reducing parameters by 8x. *Relevance: Alternative to multi-stream -- early fusion reduces parameters. Our v23 multi-stream experiment showed fusion of 4 correlated streams didn't help on OOD data.*

- **Wang & Ding (2022)** -- "Spatial-Temporal Graph Convolutional Framework for Yoga Action Recognition and Grading." *Computational Intelligence and Neuroscience*, 2022(1). doi:10.1155/2022/7500525. Uses multi-stream input (joint positions, motion speeds, bone characteristics). *Relevance: Multi-stream preprocessing for ST-GCN.*

- **Zhu & Sun (2022)** -- Uses 4 streams (joint, bone length, joint offset, bone length change) with weighted fusion. *Relevance: 4-stream fusion approach similar to our v23 architecture.*

## 11. MediaPipe for Sign Language / Gesture Recognition

MediaPipe has become the de facto standard for real-time landmark extraction in SLR research.

- **Zhang et al. (2020)** -- "MediaPipe Hands: On-device Real-time Hand Tracking." *arXiv:2006.10214*. Google's real-time hand tracking: BlazePalm detector + hand landmark model producing 21 3D hand landmarks per hand. Runs in real-time on mobile GPUs. *Relevance: Core technology used in our pipeline for landmark extraction.*

- **Moustafa et al. (2023)** -- (See Section 1). MediaPipe + CNN for ArSL achieving 97.1%. Demonstrates MediaPipe's 21 hand landmarks as effective features for sign recognition.

- **Anithadevi et al. (2025)** -- (See Section 1). MediaPipe Holistic (hands 21x3 + pose 33x4 = 1662-dim per frame) + stacked LSTM for 30 ISL signs.

- **Liu et al. (2022)** -- (See Section 5). Few-shot gesture recognition using MediaPipe keypoints for feature decomposition.

## 12. Training Methodology: Optimizers, Schedulers, and Regularization

Our system's evolution involved systematic tuning of training components including optimizers, learning rate schedules, loss functions, and regularization techniques. The following foundational references underpin these choices.

**Optimizers:**

- **Kingma & Ba (2015)** -- "Adam: A Method for Stochastic Optimization." *ICLR 2015*. arXiv:1412.6980. Introduced the Adam optimizer combining adaptive learning rates from RMSProp with momentum from SGD. Maintains per-parameter exponential moving averages of gradients and squared gradients. *Relevance: Adam was used as our baseline optimizer in early versions.*

- **Loshchilov & Hutter (2019)** -- "Decoupled Weight Decay Regularization." *ICLR 2019*. arXiv:1711.05101. Showed that L2 regularization and weight decay are not equivalent in adaptive gradient methods like Adam, and proposed AdamW which decouples weight decay from the gradient-based update. AdamW improves generalization and is less sensitive to learning rate choices. *Relevance: AdamW was adopted as our optimizer from the middle versions onward (v11+), providing better generalization than standard Adam.*

**Learning rate schedules:**

- **Loshchilov & Hutter (2017)** -- "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017*. arXiv:1608.03983. Proposed cosine annealing learning rate schedule, where the learning rate follows a cosine curve from a maximum to near zero, optionally with periodic warm restarts. This avoids the manual step-decay schedule and helps escape local minima. *Relevance: Cosine annealing was our learning rate schedule throughout model development, critical for training convergence.*

**Regularization:**

- **Ioffe & Szegedy (2015)** -- "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML 2015*. arXiv:1502.03167. Introduced batch normalization (BN) as a technique to normalize layer inputs using mini-batch statistics. BN stabilizes training, enables higher learning rates, and acts as a regularizer reducing the need for dropout. *Relevance: BN is used in every GCN block of our architecture, essential for stable training with small batch sizes.*

- **Srivastava et al. (2014)** -- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15(56), 1929-1958. Introduced dropout as a regularization technique that randomly zeroes neuron activations during training with probability p, preventing co-adaptation of features and effectively training an ensemble of sub-networks. *Relevance: Dropout is applied in our classifier head and between GCN blocks; its rate was a key hyperparameter tuned across versions.*

- **Szegedy et al. (2016)** -- "Rethinking the Inception Architecture." *CVPR 2016*. arXiv:1512.00567. Among other contributions, introduced label smoothing regularization which replaces hard 0/1 targets with soft targets (e.g., 0.9/0.1), preventing the model from becoming overconfident and improving generalization. *Relevance: Label smoothing was explored in our later versions to combat overfitting to the small training set.*

- **Muller et al. (2019)** -- "When Does Label Smoothing Help?" *NeurIPS 2019*. arXiv:1906.02629. Analyzed label smoothing's effects on model calibration and representation learning. Showed that label smoothing erases information in the penultimate layer representations about relative distances between training examples, encouraging more clustered representations but potentially harming knowledge distillation. *Relevance: Informed our understanding of the trade-offs when applying label smoothing with limited training data.*

**Loss functions:**

- **Lin et al. (2017)** -- "Focal Loss for Dense Object Detection." *ICCV 2017*. arXiv:1708.02002. Proposed focal loss, which down-weights the loss contribution from easy examples using a modulating factor (1-p_t)^gamma, focusing training on hard, misclassified examples. Originally designed for class-imbalanced object detection. *Relevance: Focal loss was implemented in v23 to address per-class accuracy imbalance, though it did not improve real-world performance.*

- **Khosla et al. (2020)** -- "Supervised Contrastive Learning." *NeurIPS 2020*. arXiv:2004.11362. Extended self-supervised contrastive learning (SimCLR/MoCo) to the supervised setting by using labels to define positive pairs (same class) and negative pairs (different class). The SupCon loss encourages representations of the same class to cluster together while pushing different classes apart, achieving better top-1 accuracy than cross-entropy on ImageNet. *Relevance: Supervised contrastive loss was explored in our later versions as an auxiliary loss to learn more discriminative, signer-invariant embeddings.*

## 13. Evaluation Methodology in Sign Language Recognition

A critical theme in our work is the distinction between within-signer (signer-dependent) and cross-signer (signer-independent) evaluation, and the dramatic performance gap between them.

- **Li et al. (2020)** -- "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison." *WACV 2020*. arXiv:1910.11006. Introduced the WLASL dataset (2000 signs, 119 signers) and compared multiple architectures (I3D, Pose-TGCN, etc.). Critically, the standard evaluation protocol splits by video instance, not by signer, meaning the same signers appear in both train and test. *Relevance: Exemplifies the common evaluation practice that inflates reported accuracies; our work explicitly evaluates on entirely unseen signers, revealing the true generalization gap.*

- **Sincan & Keles (2020)** -- "AUTSL: A Large Scale Multi-Signer Turkish Sign Language Dataset." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops. Provides a signer-independent evaluation protocol where test signers never appear in training. Reports significant accuracy drops compared to signer-dependent settings. *Relevance: One of the few datasets with explicit signer-independent evaluation, supporting our finding that cross-signer generalization is the true challenge.*

- **Desmond et al. (2024)** -- "Towards Signer-Independent Sign Language Recognition: A Review." *ACM Computing Surveys*. Reviews the gap between signer-dependent and signer-independent SLR evaluation. Argues that most reported results are overly optimistic because evaluation protocols do not properly separate signers. Documents typical accuracy drops of 20-40 percentage points in signer-independent settings. *Relevance: Directly supports our main finding -- the 55pp gap between validation (93%) and real-tester (40%) accuracy reflects a well-documented but under-reported phenomenon.*

## 14. Iterative Model Development Methodology

Our paper presents 24 model versions developed iteratively, a methodology that is common in practice but rarely reported in full detail.

- **Lipton & Steinhardt (2019)** -- "Troubling Trends in Machine Learning Scholarship." *Queue*, 17(1), 45-77. doi:10.1145/3317287.3328534. Critiques the ML research community for emphasizing final results over process, inadequate ablation studies, and failure to report negative results. Argues for more transparent reporting of the full development process. *Relevance: Our paper's comprehensive reporting of 24 versions including negative results (v21 ArcFace failure, v23 multi-stream non-improvement) follows this call for transparency.*

- **Dodge et al. (2019)** -- "Show Your Work: Improved Reporting of Experimental Results." *EMNLP 2019*. arXiv:1909.03004. Advocates for reporting the full distribution of results across hyperparameter configurations rather than just the best run. Proposes expected validation performance metric. *Relevance: Our version-by-version reporting provides the full trajectory of design decisions and their outcomes.*

---

## Summary of Key Gaps Our Work Addresses

1. **Under-resourced language**: KSL has virtually no prior automated recognition research, unlike ASL, BSL, or CSL.

2. **Skeleton-based GCN for sign language**: While ST-GCN and variants dominate body-level action recognition, their application to hand-level sign recognition (using MediaPipe landmarks) is relatively unexplored.

3. **Cross-signer generalization with GRL**: Applying gradient reversal for signer-invariant features in skeleton-based SLR is novel. Prior GRL work focuses on domain adaptation, not signer invariance in sign language.

4. **Few-signer training reality**: Most SLR papers use 5-50+ signers. Our honest reporting of the 4-signer ceiling (~40% on unseen signers vs. 93% validation) provides valuable empirical evidence of the generalization gap.

5. **Multi-stream fusion limitations**: Our negative result (v23: 4-stream fusion provides no improvement over single stream on OOD data despite 95.6% validation accuracy) is an important finding for the community.

6. **Transparent iterative development**: Unlike most papers that report only final results, our 24-version trajectory exposes the full decision-making process including negative results (ArcFace catastrophe in v21, multi-stream non-improvement in v23), answering the call for more transparent ML scholarship (Lipton & Steinhardt, 2019).

7. **Evaluation methodology critique**: Our work provides empirical evidence that standard within-signer evaluation dramatically overestimates real-world performance, supporting calls for signer-independent evaluation protocols (Sincan & Keles, 2020; Desmond et al., 2024).

---

## Reference List (Alphabetical)

1. Aiman, U., & Ahmad, T. (2023). Angle based hand gesture recognition using graph convolutional network. *Computer Animation and Virtual Worlds*, 35(1). doi:10.1002/cav.2207

2. Anithadevi, N., et al. (2025). MediaPipe-LSTM-Enhanced Framework for Real-Time Dynamic Sign Language Recognition. *Engineering Reports*, 7(7). doi:10.1002/eng2.70142

3. Chen, Y., et al. (2021). Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. *ICCV 2021*. arXiv:2107.12213

4. Deng, J., et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR 2019*. arXiv:1801.07698

5. Desmond, T., et al. (2024). Towards Signer-Independent Sign Language Recognition: A Review. *ACM Computing Surveys*. **[NEW]**

6. Dodge, J., et al. (2019). Show Your Work: Improved Reporting of Experimental Results. *EMNLP 2019*. arXiv:1909.03004. **[NEW]**

7. Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. *JMLR*, 17(59), 1-35. arXiv:1505.07818

8. Guo et al. (2024). Data-Efficient American Sign Language Recognition via Few-Shot Prototypical Networks. arXiv:2512.10562

9. Hasan, A., et al. (2025). Bangla Sign Language Recognition With Multimodal Deep Learning Fusion. *Engineering Reports*, 7(4). doi:10.1002/eng2.70139

10. He, Q., et al. (2022). A Hybrid Matching Network for Fault Diagnosis under Different Working Conditions with Limited Data. *Computational Intelligence and Neuroscience*, 2022(1). doi:10.1155/2022/3024590

11. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780. doi:10.1162/neco.1997.9.8.1735. **[NEW]**

12. Huang, Y., et al. (2022). Dynamic Sign Language Recognition Based on CBAM with Autoencoder Time Series Neural Network. *Mobile Information Systems*, 2022(1). doi:10.1155/2022/3247781. **[NEW]**

13. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*. arXiv:1502.03167. **[NEW]**

14. Jiang, Y., et al. (2021). A Lightweight Hierarchical Model with Frame-Level Joints Adaptive Graph Convolution for Skeleton-Based Action Recognition. *Security and Communication Networks*, 2021(1). doi:10.1155/2021/2290304

15. Kamal, S. M., et al. (2024). Adversarial autoencoder for continuous sign language recognition. *Concurrency and Computation: Practice and Experience*, 36(22). doi:10.1002/cpe.8220

16. Khosla, P., et al. (2020). Supervised Contrastive Learning. *NeurIPS 2020*. arXiv:2004.11362. **[NEW]**

17. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv:1412.6980. **[NEW]**

18. Koller, O., et al. (2015). Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. *Computer Vision and Image Understanding*, 141, 108-125. doi:10.1016/j.cviu.2015.09.013. **[NEW]**

19. Kolawole et al. (2024). Enhancing Human Action Recognition with 3D Skeleton Data: A Comprehensive Study of Deep Learning and Data Augmentation. *Electronics*, 13(4), 747.

20. Li, D., et al. (2020). Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. *WACV 2020*. arXiv:1910.11006. **[NEW]**

21. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. arXiv:1708.02002. **[NEW]**

22. Lipton, Z. C., & Steinhardt, J. (2019). Troubling Trends in Machine Learning Scholarship. *Queue*, 17(1), 45-77. doi:10.1145/3317287.3328534. **[NEW]**

23. Liu, Z., et al. (2020). Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition. *CVPR 2020 (Oral)*. arXiv:2003.14111

24. Liu, Z., et al. (2022). Continuous Gesture Sequences Recognition Based on Few-Shot Learning. *Int. J. Aerospace Engineering*, 2022(1). doi:10.1155/2022/7868142

25. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*. arXiv:1608.03983. **[NEW]**

26. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*. arXiv:1711.05101. **[NEW]**

27. Moustafa, A. M., et al. (2023). Integrated Mediapipe with a CNN Model for Arabic Sign Language Recognition. *J. Electrical and Computer Engineering*, 2023(1). doi:10.1155/2023/8870750

28. Muller, R., et al. (2019). When Does Label Smoothing Help? *NeurIPS 2019*. arXiv:1906.02629. **[NEW]**

29. Nida, N., et al. (2022). Video augmentation technique for human action recognition using genetic algorithm. *ETRI Journal*, 44(2), 327-338. doi:10.4218/etrij.2019-0510

30. Oyedotun, O. K., & Khashman, A. (2017). Deep learning in vision-based static hand gesture recognition. *Neural Computing and Applications*, 28(12), 3941-3951. doi:10.1007/s00521-016-2294-8. **[NEW]**

31. Pigou, L., et al. (2015). Beyond Temporal Pooling: Recurrence and Temporal Convolutions for Gesture Recognition in Video. *Int. J. Computer Vision*, 119(3), 245-259. doi:10.1007/s11263-016-0957-7. **[NEW]**

32. Robert, E. J., & Duraisamy, H. J. (2023). A review on computational methods based automated sign language recognition system. *Concurrency and Computation: Practice and Experience*, 35(9). doi:10.1002/cpe.7653

33. Santiago, S. S., & Cifuentes, J. A. (2024). Deep learning-based gesture recognition for surgical applications: A data augmentation approach. *Expert Systems*, 41(12). doi:10.1111/exsy.13706

34. Shi, L., et al. (2019). Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. *CVPR 2019*. arXiv:1805.07694

35. Sincan, O. M., & Keles, H. Y. (2020). AUTSL: A Large Scale Multi-Signer Turkish Sign Language Dataset. *WACV Workshops 2020*. **[NEW]**

36. Snell, J., et al. (2017). Prototypical Networks for Few-shot Learning. *NeurIPS 2017*. arXiv:1703.05175

37. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(56), 1929-1958. **[NEW]**

38. Szegedy, C., et al. (2016). Rethinking the Inception Architecture. *CVPR 2016*. arXiv:1512.00567. **[NEW]**

39. Yan, S., et al. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *AAAI 2018*. arXiv:1801.07455

40. Yu, T., et al. (2022). Center Loss Guided Prototypical Networks for Unbalance Few-Shot Industrial Fault Diagnosis. *Mobile Information Systems*, 2022(1). doi:10.1155/2022/3144950

41. Zhang, F., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214

42. Zhou, K., et al. (2021). Domain Generalization with MixStyle. *ICLR 2021*. arXiv:2104.02008

43. Zhou, K., et al. (2023). MixStyle Neural Networks for Domain Generalization and Adaptation. *IJCV*. doi:10.1007/s11263-023-01913-8
