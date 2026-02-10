# V17 Full Pipeline Review (2026-02-09)

Conducted by 4-agent team reviewing code, training logs, data pipeline, and architecture.

## Executive Summary

V17 achieved 55.28% (numbers) / 89.95% (words) = ~72.6% combined, vs v8's 83.65%.
The 11% gap comes from: features (~60%), architecture (~20%), loss/training (~15%), joint-vs-split training (~5%).

---

## Critical Bugs

### 1. SWA Is Dead Code (numbers) / Harmful (words)
- `swa_start_frac=0.8` → numbers SWA starts at epoch 240, but early stopping fires at 203
- For words, SWA activated at epoch 160 but used **stale BatchNorm stats** during validation
- `update_bn()` only called AFTER training ends (line 524), so SWA model always looks worse
- Words best was 89.9% pre-SWA; SWA model scored 87.8% → saved model is pre-SWA anyway
- **Fix**: Lower `swa_start_frac` to 0.4, or disable early stopping during SWA, or call `update_bn()` each SWA epoch

### 2. Anti-Attractor Loss Removed → 444 Collapse
- V16 had explicit anti-attractor penalizing class-54 predictions on non-54 samples
- V17 removed it along with focal loss, contrastive loss, confusion penalties, and mixup
- Class 444: 45% (v16) → 10% (v17)
- **Fix**: Re-add at minimum the anti-attractor head for 444→54

### 3. Normalization Squashes Hand Features 5x
- After wrist-centering, pose joints have ~5x more variance than hand joints
- Global max normalization divides by pose range → hand values compressed to [-0.24, 0.18]
- Hand std=0.056 vs pose std=0.283 after normalization
- Gaussian noise (0.015) = ~30% of hand signal but only ~5% of pose signal
- **Fix**: Normalize hands and pose SEPARATELY, or use per-body-part scaling

### 4. Catastrophic Overfitting (Numbers)
- 99.7% train vs 55.3% val = 45-point gap
- 6.9M parameters on 375 training samples (18,500 params/sample)
- Val accuracy swings 15% between consecutive epochs (extremely unstable)
- **Fix**: Reduce capacity, add stronger regularization, or add more training data

---

## High-Severity Issues

### 5. Augmentation Miscalibrated for Numbers
- V16 numbers: hand_dropout=0.15, noise=0.006
- V17 numbers: hand_dropout=0.50, noise=0.015 (3x and 2.5x increases)
- Number signs differ in fine-grained hand configurations; aggressive dropout destroys these
- **Fix**: Moderate to hand_dropout=0.3, noise=0.010

### 6. Weight Decay Inconsistent
- Numbers: 0.003, Words: 0.001 (3x higher)
- Both have identical architecture → no reason for different regularization
- Numbers are harder → need LESS regularization, not more
- **Fix**: Equalize to 0.001

### 7. Unnecessary Class Weighting
- Training data is perfectly balanced: exactly 25 samples per class
- WeightedRandomSampler + class-weighted CE + hard-class boosting = triple rebalancing
- All inverse-frequency weights are ~1.0 → adds complexity without benefit
- **Fix**: Remove WeightedRandomSampler and class weights; keep only hard-class boosting if needed

---

## Medium-Severity Issues

### 8. Hand Swap Doesn't Flip Pose Nodes
- When L/R hands are swapped and X mirrored, pose nodes (42-47) are NOT mirrored
- Creates anatomically inconsistent samples (hands flipped, body not)

### 9. Temporal Roll Creates Discontinuities
- `np.roll` on zero-padded sequences wraps zeros into middle of sequence
- 98.4% of samples are downsampled (not padded), so mostly benign

### 10. No Random Seed
- Neither v16 nor v17 set seeds → non-reproducible results

---

## Architecture Gap Analysis (v8 vs v17)

### V8: TemporalPyramidV8 (BiLSTM)
- Input: 649-dim (225 raw + 100 engineered + 324 original)
- Multi-scale Conv1d (kernels 3, 7, 15) → BiLSTM (3 layers, 320 hidden) → 4-head attention pooling
- ConfusionAwareFocalLoss + anti-attractor head
- Joint 30-class training
- ~12-15M params

### V17: KSLGraphNet (ST-GCN)
- Input: 3 channels × 48 nodes (raw XYZ)
- 6 ST-GCN blocks [3→128→128→256→256→512→512] → AdaptiveAvgPool
- CrossEntropy with label smoothing
- Separate 15-class models
- 6.95M params

### V8's 100 Hand-Engineered Features (per hand × 2):
- 5 fingertip-to-wrist distances
- 10 pairwise fingertip distances
- 5 fingertip-to-MCP distances (finger curl)
- 1 max hand spread
- 1 thumb-index angle cosine
- 3 palm normal vector
- 25 velocity features (derivatives of above)
= 50 per hand, 100 total

---

## Data Pipeline Summary

- **Format**: NPY files, (num_frames, 549), MediaPipe Holistic output
- **V17 uses 225/549 features** (pose + hands; face discarded)
- **Training**: 750 samples, 25 per class, 5 signers × 5 reps
- **Validation**: ~720 samples, mostly 25/class (268/388/444 have 20, Teach has 19)
- **Signer-independent split**: Train signers 1-5, Val signers 6-10
- **Frame lengths**: 84-160, mean 112.5, 98.4% downsampled to 90 frames
- **Graph**: 48 nodes (21 LH + 21 RH + 6 pose), 54 edges, cross-hand edge weight 0.3

---

## Recommended V18 Fixes (Priority Order)

| # | Fix | Expected Impact | Effort |
|---|-----|:---:|:---:|
| 1 | Fix normalization: separate hand/pose scaling | +3-5% | Hours |
| 2 | Re-add anti-attractor for 444→54 | +2-3% numbers | Hours |
| 3 | Fix SWA: lower start to 0.4, call update_bn each epoch | +1-2% | Minutes |
| 4 | Moderate numbers augmentation (dropout 0.3, noise 0.01) | +1-2% | Minutes |
| 5 | Equalize weight_decay to 0.001 | +0.5-1% | Minutes |
| 6 | Remove unnecessary class weighting (data is balanced) | Cleaner code | Minutes |
| 7 | Set random seeds | Reproducibility | Minutes |
| 8 | Add velocity features as input channels (3→9 ch) | +2-4% | Hours |
| 9 | Replace AdaptiveAvgPool with attention pooling | +1-2% | Hours |
| 10 | Run on L40 (not A100 MIG) for fair comparison | Fair eval | Minutes |
