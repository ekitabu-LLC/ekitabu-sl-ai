# KSL Model v10 Report

## Version Summary

Major architecture change from Temporal Pyramid CNN to Spatial-Temporal Graph Convolutional Network (ST-GCN), treating hand landmarks as graph nodes with anatomical edge connections, aiming to solve the 444 vs 54 confusion through structural hand shape modeling.

## Architecture

- **Model Class**: `STGCN`
- **Type**: Spatial-Temporal Graph Convolutional Network (ST-GCN)
- **Input**: (batch, 3, 90, 42) -- 3 channels (x,y,z), 90 frames, 42 nodes (21 left + 21 right hand landmarks)
- **Hidden Dim**: 128
- **Num Layers**: 6
- **Components**:
  - BatchNorm1d data normalization (42 * 3 = 126 features)
  - 6 STGCNBlock layers with channel progression: [3, 128, 128, 256, 256, 512, 512]
    - Each block: GraphConvolution (spatial) + BatchNorm2d + ReLU + Conv2d temporal (kernel=9) + BatchNorm2d + Dropout(0.3) + Residual connection
    - Stride=2 at layers 2 and 4 (temporal downsampling)
    - Residual connections with 1x1 conv when dimensions change
  - AdaptiveAvgPool2d(1) global pooling
  - Classifier: Linear(512->128) + ReLU + Dropout(0.3) + Linear(128->30)
- **Graph Structure**: 42 nodes (21 per hand), anatomical edges:
  - Intra-hand: 23 edges per hand following finger bones and palm connections
  - Inter-hand: Weak connection (0.5) between left wrist (node 0) and right wrist (node 21)
  - Self-loops added, degree-normalized (symmetric normalization: D^(-1/2) A D^(-1/2))

## Parameters

Printed at runtime. Estimated ~1.5-2M based on the 6-layer ST-GCN with channel progression [3, 128, 128, 256, 256, 512, 512].

## Key Changes from Previous Version (v9)

1. **Complete architecture change**: From Temporal Pyramid CNN + BiLSTM to ST-GCN (graph neural network)
2. **Graph-based input**: Hand landmarks as 42-node graph with anatomical edges, rather than flat feature vector
3. **No hand feature engineering**: Removed all 100+ engineered features (tip distances, curls, palm normal, velocities, temporal features); uses raw (x,y,z) coordinates only
4. **Wrist-relative normalization**: Each hand normalized relative to its own wrist, then globally scaled by max absolute value
5. **No confusion-aware loss**: Replaced with separate FocalLoss + ConfusionPenaltyLoss (cleaner implementation)
6. **No anti-attractor head**: Removed the binary anti-attractor classifier
7. **No mixup**: Removed mixup augmentation entirely
8. **Higher learning rate**: 1e-3 (up from 3e-4)
9. **Lower weight decay**: 1e-4 (down from 0.015)
10. **Larger batch size**: 32 (up from 24)
11. **Simpler augmentation**: Scale (0.9-1.1), noise (0.01), temporal shift (-5 to +5 frames), hand mirror flip
12. **LambdaLR scheduler** instead of manual LR setting
13. **Dual improvement criterion**: Best model saved if overall accuracy OR 444 accuracy improves

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Min LR | 1e-6 |
| LR Schedule | Warmup (10 epochs) + Cosine annealing (via LambdaLR) |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 200 |
| Early Stopping Patience | 35 |
| Loss | FocalLoss (gamma=2.0, smoothing=0.1) + ConfusionPenaltyLoss (weight=1.5) |
| Dropout | 0.3 |
| Gradient Clipping | max_norm=1.0 |
| GPU | A10G (Modal) |

## Data Pipeline

- **Data Format**: NumPy arrays (.npy), raw landmarks extracted to graph format
- **Dataset**: train_v2 / val_v2 split
- **Max Frames**: 90 (uniform sampling for longer, zero-padding for shorter)
- **Nodes**: 42 (21 left hand + 21 right hand), extracted from feature indices 99:162 (left) and 162:225 (right)
- **Channels**: 3 (x, y, z coordinates)
- **Normalization**: Wrist-relative per hand, then global max-abs scaling to [-1, 1]
- **Graph Adjacency**: 23 anatomical edges per hand + 1 inter-hand edge (wrist-to-wrist at 0.5 weight), symmetric normalization
- **Augmentation**:
  - Random scale (0.9-1.1x) at 50% probability
  - Gaussian noise (std=0.01) at 30% probability
  - Temporal shift (-5 to +5 frames) at 30% probability
  - Hand mirror flip (swap left/right, negate x-axis) at 30% probability
- **Hard Classes**: 444(2.5x), 35(2.0x), 91(1.8x), 89(1.8x), 66(1.5x)
- **Confusion Pairs**: 9 pairs, same as v9 (444->54:5.0, 444->35:3.0, etc.)
- **No face/pose features**: Only hand landmarks are used

## Results

No explicit results in the code, but the approach of using graph structure for distinguishing hand configurations (444 = repeated "4" shape vs 54 = "5" then "4" shape) was the key hypothesis. The training loop prints bottom-10 classes every 25 epochs for monitoring.

## Platform

**Modal** (cloud) -- A10G GPU, debian_slim image, Python 3.11.

## Lessons / Notes

- **Major paradigm shift**: Moving from feature-engineered flat vectors to graph-structured inputs was a significant architectural decision, motivated by the intuition that 444 vs 54 confusion is about hand shape structure, not temporal patterns
- **Simplified pipeline**: Removing all engineered features and relying on raw coordinates + graph structure was a bold simplification; the graph edges encode the anatomical prior that feature engineering was trying to capture
- **ConfusionPenaltyLoss**: Cleaner implementation than v8/v9's ConfusionAwareLoss -- operates on softmax probabilities rather than argmax, making it differentiable
- **Dual saving criterion**: Saving on either overall accuracy OR 444 accuracy improvement is a pragmatic approach to avoid losing models that improve on the critical class even if overall accuracy dips
- **Data efficiency**: Only uses hand landmarks (42 * 3 = 126 values per frame), discarding face (324 values) and pose (99 values) features; this is a trade-off between information and noise
- **Hand mirror flip**: Augmentation that swaps left/right hands and negates x-coordinates is semantically meaningful for sign language
- **No num_workers**: DataLoader uses num_workers=0, which may slow training on Modal
- **Bottom-10 monitoring**: Every 25 epochs, prints the 10 worst-performing classes -- good diagnostic practice for identifying persistent problem classes
