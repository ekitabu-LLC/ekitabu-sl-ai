# V12 Report - Pose + Hands (Full Upper Body Context)

## Version Summary

Extended the graph from hands-only (42 nodes) to hands + upper body pose (48 nodes), adding shoulder-elbow-wrist joints to provide arm context for sign recognition.

## Architecture

- **Model class:** `Model` (custom ST-GCN)
- **Type:** Spatial-Temporal Graph Convolutional Network
- **Blocks:** `Block` = GCN (linear graph conv) + BatchNorm + TCN (1D temporal conv) + residual connection
- **Layers:** 6 blocks with channel progression `[3, 128, 128, 256, 256, 512, 512]`
- **Temporal downsampling:** stride 2 at layers 2 and 4
- **Temporal kernel size:** 9
- **Input features:** 3 channels (x, y, z) per node
- **Graph nodes:** 48 (21 left hand + 21 right hand + 6 upper body pose joints)
- **Graph edges:**
  - Hand anatomical edges (same as v11)
  - Pose edges: shoulder-shoulder, shoulder-elbow, elbow-wrist, wrist-to-hand-root
  - Weak cross-hand link: 0.3 weight between hand roots (reduced from 0.5 in v11)
- **Classifier head:** Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
- **Pooling:** AdaptiveAvgPool2d(1)

## Parameters

Not recorded in results. Slightly larger than v11 due to 48 nodes vs 42 nodes (affects data_bn input dimension: 48*3=144 vs 42*3=126). Estimated ~700K params per model.

## Key Changes from Previous Version (v11)

1. **48 nodes instead of 42** -- added 6 upper body pose joints (L/R shoulder, elbow, wrist)
2. **New pose edges** -- anatomical arm connections plus wrist-to-hand-root links to connect arms to hand graphs
3. **Pose landmark extraction** -- reads MediaPipe pose indices [11,12,13,14,15,16] (upper body)
4. **Separate pose normalization** -- pose landmarks normalized relative to mid-shoulder point
5. **Weaker cross-hand link** -- reduced from 0.5 to 0.3
6. **Additional augmentation:** Random temporal shift (30% prob, shift [-5, +5] frames)
7. **Slightly stronger noise augmentation** -- std increased from 0.01 to 0.02
8. **Removed hard class weights** -- no per-class boosting, just balanced inverse-frequency weights capped at 2.0
9. **Removed confusion pair definitions** entirely
10. **Removed focal_gamma** from CONFIG

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Min LR | 1e-6 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 150 |
| Patience (early stop) | 30 |
| Warmup epochs | 10 |
| Scheduler | Cosine annealing with warmup (LambdaLR) |
| Loss | CrossEntropyLoss with class weights + label smoothing |
| Label smoothing | 0.15 |
| Dropout | 0.4 |
| Gradient clipping | max_norm=1.0 |
| Sampler | WeightedRandomSampler (class-balanced) |

## Data Pipeline

- **Data format:** `.npy` files with MediaPipe landmarks
- **Landmark extraction:**
  - Left hand: indices 99-161 (21 joints x 3 coords)
  - Right hand: indices 162-224 (21 joints x 3 coords)
  - Pose (upper body): MediaPipe indices [11,12,13,14,15,16] x 3 coords (shoulders, elbows, wrists)
- **Graph layout:** `[left_hand(0-20), right_hand(21-41), pose(42-47)]`
- **Normalization:**
  - Left hand: centered to left wrist (node 0)
  - Right hand: centered to right wrist (node 21)
  - Pose: centered to mid-shoulder point
  - Global: max-abs normalization to [-1, 1]
- **Temporal resampling:** Uniform subsampling to 90 frames (or zero-pad if shorter)
- **Augmentation (train only):**
  - Scale jitter: 50% prob, uniform [0.9, 1.1]
  - Gaussian noise: 30% prob, std=0.02
  - Temporal shift: 30% prob, roll [-5, +5] frames
- **Input tensor shape:** `(C=3, T=90, N=48)` per sample

## Results

No result JSON files found for v12.

## Platform

- **Modal** (cloud GPU)
- **GPU:** A10G
- **Python:** 3.11
- **Timeout:** 21600s (6 hours)
- **Storage:** Modal Volume (`ksl-dataset-vol`)

## Lessons / Notes

- The key hypothesis was that body pose provides context for sign meaning -- arm position/movement supplements hand shape
- Pose landmarks connect to hand roots through wrist nodes, creating a unified kinematic chain
- The pose normalization is separate from hand normalization (mid-shoulder vs wrist-relative), preserving relative spatial relationships
- Class weights simplified to balanced inverse-frequency (capped at 2x) -- no per-class hard boosting
- The temporal shift augmentation was a new addition, simulating slight timing variations in sign execution
- Still uses split models (separate numbers and words training), continuing the v11 approach
- The 6 pose nodes add overhead but are a small fraction of the 48-node graph
