# KSL Model v7 Report

## Version Summary

Fixing critical performance issues from v5/v6 by removing contrastive loss, adding focal loss with hard class boosting, temporal pyramid CNN, reduced augmentation, and mixup.

## Architecture

- **Model Class**: `TemporalPyramidModel` (default) / `EnhancedHandFocusedModel` (alternative)
- **Type**: Temporal Pyramid CNN + BiLSTM + Multi-Head Attention
- **Input Features**: 649 (549 base landmarks + 100 engineered hand features)
- **Hidden Dim**: 320
- **Components** (TemporalPyramidModel):
  - LayerNorm input normalization
  - Linear projection (649 -> 320) with LayerNorm + GELU + Dropout(0.2)
  - 3-scale temporal Conv1d pyramid (kernel sizes: 3, 7, 15), each producing hidden_dim//2 channels with BatchNorm + GELU
  - Temporal fusion: Linear(480 -> 320) + LayerNorm + GELU + Dropout(0.4)
  - 3-layer bidirectional LSTM (hidden=320, output=640)
  - 4 attention heads (each: Linear(640->320) + Tanh + Linear(320->1))
  - Pre-classifier: LayerNorm(2560) + Dropout(0.4) + Linear(2560->640) + GELU + Dropout(0.4)
  - Classifier: Linear(640 -> 30)
- **Alternative** (`EnhancedHandFocusedModel`):
  - Separate hand encoder (226-dim) and body encoder (423-dim)
  - Cross-attention (MultiheadAttention, 8 heads) between hands and body
  - 3-layer BiLSTM on concatenated features
  - Single attention head + classifier MLP

## Parameters

Not explicitly printed in code, but estimated ~4-5M for the TemporalPyramidModel based on architecture dimensions.

## Key Changes from Previous Version (v5/v6)

1. **Removed contrastive loss** -- was hurting classification performance
2. **Focal loss with hard class boosting** -- FocalLossWithBoost (gamma=2.5) with per-class boost weights (e.g., 444: 3.0x, 35: 3.0x, 100: 3.0x)
3. **Temporal pyramid** -- Multi-scale Conv1d (3/7/15 kernels) for multi-digit number recognition (100, 444)
4. **Reduced augmentation strength** -- Previous versions were destroying features
5. **Mixup regularization** (alpha=0.3, applied 40% of the time) -- gentler alternative to contrastive
6. **Hand feature engineering** -- 50 features per hand (tip distances, inter-tip distances, finger curl, max spread, thumb-index angle, palm normal), plus velocities
7. **WeightedRandomSampler** with hard class oversampling
8. **Two model variants** -- pyramid and handfocused, can train both via `--model both`

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Min LR | 1e-6 |
| LR Schedule | Warmup (15 epochs) + Cosine annealing |
| Weight Decay | 0.02 |
| Batch Size | 24 |
| Max Epochs | 300 |
| Early Stopping Patience | 50 |
| Loss | FocalLossWithBoost (gamma=2.5, smoothing=0.15) |
| Mixup Alpha | 0.3 (applied 40% of time) |
| Dropout | 0.4 |
| Gradient Clipping | max_norm=1.0 |
| GPU | A10G (Modal) |
| Timeout | 6 hours |

## Data Pipeline

- **Data Format**: NumPy arrays (.npy), 549 base features per frame
- **Dataset**: train_v2 / val_v2 split
- **Max Frames**: 90 (random crop for train, uniform sampling for val, zero-pad if shorter)
- **Normalization**: Global mean/std computed from up to 300 training samples
- **Engineered Features**: 100 additional features (50 per hand: 5 tip distances, 10 inter-tip distances, 5 finger curls, 1 max spread, 1 thumb-index angle, 3 palm normal) + 100 velocity features (frame-to-frame deltas)
- **Augmentation** (training only):
  - Temporal speed perturbation (0.9-1.1x) at 40% probability; gentler for hard classes
  - Gaussian noise (scale=0.005 for hard, 0.01 for others) at 50% probability
  - Scale perturbation (0.98-1.02x) at 30% probability
  - Frame dropout (1-3 frames zeroed) at 30% probability (not for hard classes)
  - Left-right hand swap at 50% probability
- **Hard Classes**: 100(3.0x), 35(3.0x), 444(3.0x), 125(2.5x), 54(2.0x), 268(2.0x), Tomatoes(3.0x), Proud(2.5x), Apple(2.0x), Gift(1.5x), Market(1.5x)
- **Sampling**: WeightedRandomSampler with inverse class frequency * hard class boost

## Results

Results mentioned in v8 docstring (problems from v7):
- Class 54 became an attractor: 22->54, 444->54, 100->54, 125->54
- Regressions: Tortoise (100->60%), 100 (88->52%), 444 (50->15%), 9 (100->72%)
- Still struggling: 444 (15%), 35 (36%), 91 (48%)
- Hard class boosting caused regressions by over-boosting some classes

## Platform

**Modal** (cloud) -- A10G GPU, debian_slim image, Python 3.11, PyTorch + NumPy + scikit-learn.

## Lessons / Notes

- Contrastive loss removal was a good decision (was hurting classification)
- Hard class boosting (up to 3.0x) was too aggressive and caused class 54 to become an attractor -- classes that looked similar to 54 all collapsed to predicting 54
- The temporal pyramid architecture with multi-scale kernels was a reasonable approach for multi-digit numbers, but didn't solve the core confusion issue
- Mixup at 40% application rate was a gentler regularization strategy
- The dual-model approach (pyramid vs handfocused) allowed experimentation, but both suffered from the attractor problem
- Xavier weight initialization was applied to all Linear layers
