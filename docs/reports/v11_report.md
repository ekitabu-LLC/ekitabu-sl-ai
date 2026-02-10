# V11 Report - Split Models (Numbers + Words)

## Version Summary

Split training into two separate ST-GCN models (one for numbers, one for words) to allow domain-specific specialization, with reduced class weights to avoid mode collapse.

## Architecture

- **Model class:** `Model` (custom ST-GCN)
- **Type:** Spatial-Temporal Graph Convolutional Network
- **Blocks:** `Block` = GCN (linear graph conv) + BatchNorm + TCN (1D temporal conv) + residual connection
- **Layers:** 6 blocks with channel progression `[3, 128, 128, 256, 256, 512, 512]`
- **Temporal downsampling:** stride 2 at layers 2 and 4
- **Temporal kernel size:** 9
- **Input features:** 3 channels (x, y, z) per node
- **Graph nodes:** 42 (21 left hand + 21 right hand)
- **Graph:** Symmetric normalized adjacency with hand anatomical edges; weak 0.5 cross-hand link between wrists
- **Classifier head:** Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
- **Pooling:** AdaptiveAvgPool2d(1) over temporal and spatial dims

## Parameters

Not recorded in results. Architecture is identical to v10 per model, but each model has fewer output classes:
- Numbers model: 15 output classes
- Words model: 15 output classes

Estimated ~695K params per model based on the channel structure and 15-class head.

## Key Changes from Previous Version (v10)

1. **Two separate models** instead of one combined 30-class model -- each model specializes for its domain
2. **Reduced class weights** -- hard class weights lowered (444: 2.5->1.5, 35: 2.0->1.3, 91: 1.8->1.2, 89: 1.8->1.2) to avoid mode collapse
3. **Removed confusion pair penalty from loss** -- confusion pairs are defined but not used in the loss function (unlike v10 which used them more aggressively)
4. **No word hard classes** -- words model has empty hard class dict and no confusion pairs
5. **Focal gamma reduced** -- `focal_gamma=1.5` defined in CONFIG but not actually used in loss (loss is standard `F.cross_entropy`)

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
- **Landmark extraction:** Left hand (indices 99-161) and right hand (indices 162-224) from full 225-dim feature vector
- **Graph nodes:** 42 (left hand 0-20, right hand 21-41)
- **Normalization:** Hands centered to respective wrist roots, then global max-abs normalization to [-1, 1]
- **Temporal resampling:** Uniform subsampling to 90 frames (or zero-pad if shorter)
- **Augmentation (train only):**
  - Scale jitter: 50% prob, uniform [0.9, 1.1]
  - Gaussian noise: 30% prob, std=0.01
- **Data split:** `train_v2` / `val_v2` directories
- **Input tensor shape:** `(C=3, T=90, N=42)` per sample

## Results

No result JSON files found for v11. The v15 analysis report mentions v8 as the previous best at 83.65% combined. V11 results are not recorded in any available analysis files.

## Platform

- **Modal** (cloud GPU)
- **GPU:** A10G
- **Python:** 3.11
- **Timeout:** 21600s (6 hours)
- **Storage:** Modal Volume (`ksl-dataset-vol`)

## Lessons / Notes

- The split-model approach was motivated by the observation that numbers and words have very different characteristics (numbers = static hand shapes, words = dynamic arm movements)
- Hard class weights were significantly reduced from v10 -- v10 had weights up to 2.5x for class 444 while v11 caps at 1.5x
- `focal_gamma=1.5` is defined in CONFIG but **never used** in the actual loss computation -- the loss is plain weighted cross-entropy with label smoothing
- Confusion pairs are defined for numbers but also **not used** in the loss -- they appear to be legacy from v10
- The adjacency matrix uses symmetric normalization (D^{-1/2} A D^{-1/2}), standard for spectral GCN approaches
- No pose/body landmarks used -- hands only
