# KSL Model v9 Report

## Version Summary

Adding temporal segmentation features for multi-digit number detection, with updated and expanded confusion penalties, building on v8's confusion-aware framework.

## Architecture

- **Model Class**: `TemporalPyramidV9`
- **Type**: Temporal Pyramid CNN + BiLSTM + Multi-Head Attention + Anti-Attractor Head
- **Input Features**: 657 (549 base + 100 hand features + 8 temporal features)
- **Hidden Dim**: 320
- **Components**: Identical to v8's TemporalPyramidV8, except input projection accepts 657 features instead of 649:
  - LayerNorm input normalization (657)
  - Linear projection (657 -> 320) + LayerNorm + GELU + Dropout(0.2)
  - 3-scale temporal Conv1d pyramid (kernel sizes: 3, 7, 15)
  - Temporal fusion: Linear(480 -> 320) + LayerNorm + GELU + Dropout(0.35)
  - 3-layer bidirectional LSTM (hidden=320, output=640)
  - 4 attention heads
  - Pre-classifier: LayerNorm(2560) -> Linear(2560->640) + GELU
  - Classifier: Linear(640 -> 30)
  - Anti-attractor head: Linear(640->160) + GELU + Linear(160->1)

## Parameters

Nearly identical to v8 (~4-5M), with marginally more in the input projection layer (657 vs 649 input dim).

## Key Changes from Previous Version (v8)

1. **8 New Temporal Features** (`compute_temporal_features`):
   - Normalized duration (frames / 90.0)
   - 3 segment similarity scores (divide sequence into 3 segments, compute pairwise cosine similarity)
   - Right hand mean velocity
   - Right hand velocity std
   - Velocity dip ratio (fraction of frames with velocity < 0.3 * mean)
   - Gesture peak count (normalized count of hand spread peaks)
   - These features are broadcast to all frames (same value per frame, 8 features)
2. **Feature dimension**: 657 (up from 649 = +8 temporal features)
3. **Updated confusion pairs**: 10 pairs (up from 9), with higher penalties:
   - 444->54: 5.0x (up from 3.5x)
   - 444->35: 3.0x (new)
   - New pairs: 125->22 (2.0x), 89->73 (2.0x), 89->Teach (2.0x), 73->17 (2.0x)
4. **Removed**: 35->Colour, Ugali->Tomatoes, 9->89 confusion pairs from v8

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Min LR | 1e-6 |
| LR Schedule | Warmup (15 epochs) + Cosine annealing |
| Weight Decay | 0.015 |
| Batch Size | 24 |
| Max Epochs | 250 |
| Early Stopping Patience | 40 |
| Loss | ConfusionAwareLoss (focal gamma=2.0, smoothing=0.1) + AntiAttractor (BCE, weight=0.1) |
| Mixup Alpha | 0.2 (applied 30% of time) |
| Dropout | 0.35 |
| Gradient Clipping | max_norm=1.0 |
| GPU | A10G (Modal) |

All training hyperparameters are identical to v8.

## Data Pipeline

- **Data Format**: NumPy arrays (.npy), 549 base features
- **Dataset**: train_v2 / val_v2 split
- **Max Frames**: 90
- **Normalization**: Global mean/std from up to 300 training samples
- **Engineered Features**: 108 additional features total:
  - 100 hand features (same as v7/v8): tip distances, inter-tip distances, curls, spread, angle, palm normal, velocities
  - **8 NEW temporal features**: duration, 3 segment similarities, mean velocity, velocity std, dip ratio, gesture peaks
  - Temporal features are computed once per sequence and broadcast to all frames
- **Augmentation**: Identical to v8 (gentler for protected classes)
- **Hard Classes**: Same as v8 (444:2.0, 35:2.0, 91:1.8, 100:1.5, 125:1.3)
- **Protected Classes**: Same as v8 (Tortoise, 388, 9, 268)

## Results

From v10 docstring context: The temporal features approach did not significantly resolve the 444 vs 54 confusion, motivating the architecture switch to ST-GCN in v10. The issue was fundamentally about hand shape structure, not temporal patterns.

## Platform

**Modal** (cloud) -- A10G GPU, debian_slim image, Python 3.11.

## Lessons / Notes

- The temporal features (segment similarity, velocity dips, gesture peaks) were designed to detect repeated gestures in multi-digit numbers (e.g., "444" = three "4" gestures vs "54" = one "5" then one "4"), but the features being broadcast to all frames rather than being frame-specific may have limited their discriminative power
- Confusion penalty escalation (444->54 up to 5.0x) showed diminishing returns -- the penalty mechanism operates on non-differentiable argmax predictions
- The architecture was essentially identical to v8 with only 8 more input features, suggesting that adding more engineered features to the same architecture was hitting a ceiling
- This version's limitations motivated the switch to a fundamentally different architecture (graph neural networks) in v10, reasoning that structural hand configuration differences are better captured by GNNs than temporal CNNs
- The velocity-based features (mean, std, dip ratio) were a reasonable proxy for detecting multi-digit signing tempo, but the hand shape confusion was more spatial than temporal
