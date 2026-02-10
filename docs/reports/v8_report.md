# KSL Model v8 Report

## Version Summary

Fixing the class-54 attractor problem and regressions from v7 via confusion-aware loss, balanced boosting, anti-attractor regularization, and protected class handling.

## Architecture

- **Model Class**: `TemporalPyramidV8`
- **Type**: Temporal Pyramid CNN + BiLSTM + Multi-Head Attention + Anti-Attractor Head
- **Input Features**: 649 (549 base + 100 engineered hand features)
- **Hidden Dim**: 320
- **Components**:
  - LayerNorm input normalization
  - Linear projection (649 -> 320) + LayerNorm + GELU + Dropout(0.2)
  - 3-scale temporal Conv1d pyramid (kernel sizes: 3, 7, 15), each hidden_dim//2 with BatchNorm + GELU
  - Temporal fusion: Linear(480 -> 320) + LayerNorm + GELU + Dropout(0.35)
  - 3-layer bidirectional LSTM (hidden=320, output=640)
  - 4 attention heads (Linear(640->320) + Tanh + Linear(320->1))
  - Pre-classifier: LayerNorm(2560) + Dropout(0.35) + Linear(2560->640) + GELU + Dropout(0.35)
  - Classifier: Linear(640 -> 30)
  - **NEW**: Anti-attractor head: Linear(640->160) + GELU + Linear(160->1) (binary: is/isn't attractor class)

## Parameters

Not explicitly stated, but architecture is nearly identical to v7 TemporalPyramidModel plus a small anti-attractor head. Estimated ~4-5M parameters.

## Key Changes from Previous Version (v7)

1. **Confusion-Aware Loss** (`ConfusionAwareLoss`): Extra penalty when predicting commonly confused pairs. 9 explicit confusion pairs defined with penalties up to 3.5x:
   - 444->54 (3.5x), 22->54 (2.5x), 125->54 (2.5x), 100->54 (2.0x)
   - 35->Colour (2.0x), 91->73 (1.5x), Ugali->Tomatoes (1.5x), 66->17 (1.5x), 9->89 (1.5x)
2. **Balanced Hard Class Boosting**: Reduced weights from v7 (444: 2.0 down from 3.0, 100: 1.5 down from 3.0) to prevent the regressions caused by over-boosting
3. **Protected Classes**: `PROTECT_CLASSES` list (Tortoise, 388, 9, 268) with 1.2x gentle boost -- these regressed in v7 and need protection, not boosting
4. **Anti-Attractor Head**: Binary classifier predicting whether a sample is NOT from class 54; trained with BCEWithLogitsLoss, weighted at 0.1x
5. **Anti-Attractor Regularization**: Prevents class 54 from becoming the default prediction
6. **Lower learning rate**: 3e-4 (down from 5e-4)
7. **Lower dropout**: 0.35 (down from 0.4)
8. **Lower label smoothing**: 0.1 (down from 0.15)
9. **Reduced mixup**: alpha=0.2 (down from 0.3), applied at 30% (down from 40%)

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
| Loss | ConfusionAwareLoss (focal gamma=2.0, smoothing=0.1) + AntiAttractor (BCEWithLogits, weight=0.1) |
| Mixup Alpha | 0.2 (applied 30% of time) |
| Dropout | 0.35 |
| Gradient Clipping | max_norm=1.0 |
| GPU | A10G (Modal) |

## Data Pipeline

- **Data Format**: NumPy arrays (.npy), 549 base features
- **Dataset**: train_v2 / val_v2 split
- **Max Frames**: 90
- **Normalization**: Global mean/std from up to 300 training samples
- **Engineered Features**: Same 100 features as v7 (hand distances, curls, palm normal, velocities)
- **Augmentation** (gentler for protected classes):
  - Speed perturbation: 0.92-1.08x (protected) or 0.88-1.12x (others) at 40% probability
  - Gaussian noise: scale=0.005 (protected) or 0.008 (others) at 50% probability
  - Left-right hand swap at 50% probability
  - Removed: scale perturbation and frame dropout from v7
- **Sampling**: WeightedRandomSampler with hard class boost + protected class 1.2x

## Results

Per v9 docstring (problems from v8):
- Class 444 still 0% accuracy -- confused with 54 (65%) and 35 (25%)
- New confusions emerged: 66->17, 91->73, 89->73/Teach
- The anti-attractor approach reduced 54 attraction somewhat but didn't solve 444

## Platform

**Modal** (cloud) -- A10G GPU, debian_slim image, Python 3.11.

## Lessons / Notes

- The confusion-aware loss approach was conceptually sound but didn't solve the fundamental problem: similar hand shapes in 444 vs 54 vs 35
- Anti-attractor head was a creative approach (binary classifier to detect "not class 54") but 0.1 weight was too weak to make a real difference
- Reducing hard class boost weights was the right call -- prevented v7's regression pattern
- Protected class concept (Tortoise, 388, 9, 268) was good practice to avoid regression
- The confusion penalty mechanism operates on argmax predictions during training, which is not differentiable and may limit its effectiveness
- Validation tracked attractor rate (false 54 predictions) and regressed class accuracy -- good diagnostic metrics
- Overall architecture remained essentially the same as v7 TemporalPyramid, suggesting the problem was not architectural but data-level (hand shape similarity between 444/54)
