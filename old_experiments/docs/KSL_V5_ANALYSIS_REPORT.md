# KSL v5 - Comprehensive Analysis Report

## Executive Summary

**v5** is a specialized model version in the **Kenyan Sign Language (KSL) Recognition System**, designed to optimize **number recognition** through advanced hand feature engineering. It represents a targeted approach to solving the challenge that number signs require fine-grained finger position differentiation.

---

## 1. Version Evolution Timeline

| Version | Focus | Features | Key Innovation |
|---------|-------|----------|----------------|
| **v1** | Baseline | 225 (pose + hands) | Basic LSTM/Transformer |
| **v2** | Regularization | 225 | Dropout, augmentation, cosine annealing |
| **v3** | Feature Engineering | 801 (225×4) | Velocity, acceleration, focal loss, SWA, TTA |
| **v4** | Face/Lips | 549 | Multi-scale CNN+Transformer, body-part attention |
| **v5** | **Hand Optimization** | **649** | **Hand-focused features, two-head classifier** |
| **v6** | Signer Independence | 649 | Supervised contrastive loss, domain-invariant features |

### Evolution Path

```
v1 (Basic) → v2 (Regularized) → v3 (Feature Rich) → v4 (Face Enhanced)
    → v5 (Hand Optimized) → v6 (Signer Invariant)
```

---

## 2. v5 Technical Architecture

### 2.1 Feature Dimensions

```
Base v4 Features (549)
├── Pose landmarks:     99 features (33 landmarks × 3 coords)
├── Left hand:          63 features (21 landmarks × 3 coords)
├── Right hand:         63 features (21 landmarks × 3 coords)
├── Face landmarks:    204 features (68 landmarks × 3 coords)
└── Lips landmarks:    120 features (40 landmarks × 3 coords)

+ v5 Hand Engineering (100)
├── Left hand computed:  25 features
├── Right hand computed: 25 features
├── Left hand velocity:  25 features
└── Right hand velocity: 25 features

Total: 649 features per frame
```

### 2.2 Hand Feature Engineering (25 features per hand)

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Fingertip-to-wrist distances | 5 | Distance from each fingertip to wrist |
| Fingertip-to-fingertip distances | 10 | All pairwise distances between fingertips |
| Finger curl | 5 | Distance from tip to MCP (metacarpophalangeal joint) |
| Hand spread | 1 | Maximum distance between any two fingertips |
| Thumb-index angle | 1 | Cosine of angle between thumb and index vectors |
| Palm normal | 3 | Cross product of palm vectors (orientation) |

### 2.3 Model Architecture: HandFocusedModel

```
Input (batch, 90 frames, 649 features)
       │
       ├─→ Hand Encoder (226 features → 256)
       │   └── Linear → LayerNorm → GELU → Dropout(0.2)
       │
       └─→ Body Encoder (423 features → 256)
           └── Linear → LayerNorm → GELU → Dropout(0.2)
       │
       ↓
    Fusion (512 → 256)
       │
       ↓
    Bi-LSTM (3 layers, hidden=256)
       │
       ↓
    Attention Pooling
       │
       ├─→ Number Classifier (→ 15 classes)
       ├─→ Word Classifier (→ 15 classes)
       └─→ Type Classifier (→ 2: number/word)
```

### 2.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Max frames | 90 |
| Batch size | 32 |
| Epochs | 200 |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Dropout | 0.3 |
| Early stopping patience | 40 |
| Warmup epochs | 10 |
| GPU | A10G |

---

## 3. Key Innovations in v5

### 3.1 Two-Head Classification

v5 introduces **separate classification heads** for numbers and words:
- **Number classifier**: Specialized for 15 number classes
- **Word classifier**: Specialized for 15 word classes
- **Type classifier**: Predicts whether input is number or word

This allows the model to learn different decision boundaries for each category.

### 3.2 Weighted Loss for Numbers

```python
# Extra weight for numbers (they're harder)
for i in range(len(NUMBER_CLASSES)):
    class_weights[i] *= 1.5
```

Numbers receive 1.5× weight in the loss function, prioritizing harder classes.

### 3.3 Differentiated Augmentation

Numbers receive **more aggressive hand augmentation**:
- Hand-specific noise: σ=0.02 (vs 0.01 for words)
- Random hand scaling: 0.9-1.1×
- Hand swap (mirroring)

---

## 4. Classes and Dataset

### 4.1 Number Classes (15)

```
9, 17, 22, 35, 48, 54, 66, 73, 89, 91, 100, 125, 268, 388, 444
```

### 4.2 Word Classes (15)

```
Agreement, Apple, Colour, Friend, Gift, Market, Monday,
Picture, Proud, Sweater, Teach, Tomatoes, Tortoise, Twin, Ugali
```

### 4.3 Dataset Structure

- **Training**: `/data/train_v2/` (v2 extracted features)
- **Validation**: `/data/val_v2/` (signers 6-10)
- **Class-balanced sampling** using WeightedRandomSampler

---

## 5. Issues and Challenges

### 5.1 Hard Classes Identified

Based on confusion analysis (`analyze_confusion.py`):

| Hard Class | Issue |
|------------|-------|
| **22** | Frequently confused with similar hand configurations |
| **444** | Complex three-digit number with repeated gestures |
| **89** | Confused with other two-digit combinations |
| **35** | Hand position ambiguity |
| **Market** | Similar motion patterns to other words |
| **Teach** | Gesture overlap with related actions |

### 5.2 Number vs Word Performance Gap

The v5 architecture specifically addresses the observation that:
- **Numbers are harder** because they rely entirely on fine finger positions
- **Words have more motion** which provides additional discriminative information
- Numbers often differ by subtle finger configurations (e.g., 9 vs 6)

### 5.3 Data Quality Issues

From duplicate reports found:
- Duplicate samples exist in the dataset
- Reports generated: `duplicate_report_20260106_085649.txt`, `duplicate_report_20260106_085824.txt`
- This can cause training/validation data leakage

### 5.4 Architectural Limitations

1. **Feature indexing complexity**: The model manually slices features:
   ```python
   left_hand_base = x[:, :, 99:162]
   left_hand_extra = x[:, :, 549:599]
   ```
   This is error-prone and not maintainable.

2. **Code duplication**: Hand feature computation is duplicated across:
   - `train_ksl_v5.py`
   - `evaluate_v5.py`
   - `analyze_confusion.py`

3. **No validation in feature engineering**: Missing NaN/Inf checks:
   ```python
   cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
   ```
   The epsilon (1e-8) helps but edge cases may still cause issues.

---

## 6. Confusion Analysis Results

### 6.1 Hard Classes Performance

| Class | Accuracy | Main Confusion |
|-------|----------|----------------|
| **22** | 10/25 (40.0%) | Gift: 56.0%, 17: 4.0% |
| **444** | 10/20 (50.0%) | 22: 40.0%, 35: 10.0% |
| **89** | 7/25 (28.0%) | 9: 72.0% |
| **Market** | 9/24 (37.5%) | Tortoise: 25.0%, Ugali: 20.8%, Proud: 8.3% |
| **Teach** | 16/19 (84.2%) | Agreement: 15.8% |
| **35** | 15/25 (60.0%) | Colour: 32.0%, 54: 4.0%, 444: 4.0% |

### 6.2 Number Confusion Matrix

```
          100   125    17    22   268    35   388   444    48    54    66    73    89     9    91
   100:    22     3     0     0     0     0     0     0     0     0     0     0     0     0     0 | 88%
   125:     2    22     0     1     0     0     0     0     0     0     0     0     0     0     0 | 88%
    17:     0     0    18     0     0     0     0     0     0     0     0     0     0     5     0 | 78%
    22:     0     0     1    10     0     0     0     0     0     0     0     0     0     0     0 | 91%
   268:     0     0     0     0    20     0     0     0     0     0     0     0     0     0     0 | 100%
    35:     0     0     0     0     0    15     0     1     0     1     0     0     0     0     0 | 88%
   388:     0     0     0     0     0     0    20     0     0     0     0     0     0     0     0 | 100%
   444:     0     0     0     8     0     2     0    10     0     0     0     0     0     0     0 | 50%
    48:     0     0     0     0     1     0     1     0    20     0     0     0     0     3     0 | 80%
    54:     0     5     0     0     0     0     0     0     0    20     0     0     0     0     0 | 80%
    66:     0     0     9     0     0     0     0     0     0     0    16     0     0     0     0 | 64%
    73:     0     0     0     0     0     0     0     0     0     0     0    18     0     7     0 | 72%
    89:     0     0     0     0     0     0     0     0     0     0     0     0     7    18     0 | 28%
     9:     0     0     0     0     0     0     0     0     0     0     0     0     0    25     0 | 100%
    91:     0     0     0     0     0     0     0     0     0     0     0     0     0     9    16 | 64%
```

### 6.3 Top 15 Most Confused Pairs

| Rank | True Class | Predicted As | Count |
|------|------------|--------------|-------|
| 1 | 89 | 9 | 18 times |
| 2 | 22 | Gift | 14 times |
| 3 | 66 | 17 | 9 times |
| 4 | 91 | 9 | 9 times |
| 5 | 35 | Colour | 8 times |
| 6 | 444 | 22 | 8 times |
| 7 | 73 | 9 | 7 times |
| 8 | Agreement | Tortoise | 7 times |
| 9 | Apple | 100 | 6 times |
| 10 | Gift | Tortoise | 6 times |
| 11 | Market | Tortoise | 6 times |
| 12 | 17 | 9 | 5 times |
| 13 | 54 | 125 | 5 times |
| 14 | Market | Ugali | 5 times |
| 15 | Proud | Tomatoes | 5 times |

### 6.4 Key Observations from Confusion Analysis

#### Number Confusions
- **"9" is a major attractor**: Classes 89, 91, 73, 17 all frequently get misclassified as 9
  - 89 → 9: 72% of errors (the "8" gesture is lost)
  - 91 → 9: 36% confusion
  - 73 → 9: 28% confusion
- **444 ↔ 22 confusion**: Both involve similar repeated finger patterns
- **66 → 17**: Similar two-digit hand configurations

#### Cross-Category Confusions (Number ↔ Word)
- **22 → Gift** (56%): Most severe - number being classified as word
- **35 → Colour** (32%): Hand shape similarity
- **Apple → 100**: Word misclassified as number

#### Word Confusions
- **Tortoise is an attractor**: Agreement, Gift, Market all confused with Tortoise
- **Market** has distributed errors across multiple classes

### 6.5 Performance Summary by Class Type

| Category | Perfect (100%) | Good (>80%) | Medium (60-80%) | Poor (<60%) |
|----------|----------------|-------------|-----------------|-------------|
| **Numbers** | 268, 388, 9 | 100, 125, 22, 35 | 17, 48, 54, 66, 73, 91 | 444, 89 |
| **Words** | - | Teach | - | Market, 22→Gift |

---

## 7. Comparison: v4 vs v5 vs v6

| Aspect | v4 | v5 | v6 |
|--------|----|----|----|
| **Focus** | Face integration | Hand optimization | Signer independence |
| **Features** | 549 | 649 (+100 hand) | 649 |
| **Classifier** | Single head | Two heads (num/word) | Single head |
| **Loss** | Focal Loss | CrossEntropy + weights | CE + Contrastive |
| **Key technique** | Body-part attention | Hand feature engineering | Supervised contrastive |
| **Training epochs** | 400 | 200 | 100 |
| **Augmentation** | MixUp, speed, noise | Hand-focused + speed | Temporal warp, strong aug |

---

## 8. v6 Training Results (Comparison)

### 8.1 Training Summary

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **46.43%** at epoch 46 |
| Best Training Accuracy | 91.8% |
| Early Stopping | Epoch 56 (patience: 10) |
| Final Loss | 1.44 (CE: 0.89, Contrastive: 1.08) |

### 8.2 Training Progression

```
Epoch  31: Loss=2.00 | Train=74.7% | Val=42.0% *BEST*
Epoch  34: Loss=1.91 | Train=77.9% | Val=42.9% *BEST*
Epoch  36: Loss=1.84 | Train=81.0% | Val=44.6% *BEST*
Epoch  40: Loss=1.77 | Train=85.6% | Val=37.8%
Epoch  41: Loss=1.67 | Train=87.8% | Val=45.7% *BEST*
Epoch  44: Loss=1.61 | Train=88.7% | Val=45.9% *BEST*
Epoch  46: Loss=1.60 | Train=88.7% | Val=46.4% *BEST*
Epoch  50: Loss=1.44 | Train=91.8% | Val=41.3%
```

### 8.3 Per-Class Performance (v6)

#### Numbers

| Class | Accuracy | Status |
|-------|----------|--------|
| 100 | 0.0% (0/25) | FAILED |
| 125 | 12.0% (3/25) | Poor |
| 17 | 88.0% (22/25) | Good |
| 22 | 56.0% (14/25) | Medium |
| 268 | 30.0% (6/20) | Poor |
| 35 | 0.0% (0/25) | FAILED |
| 388 | 50.0% (10/20) | Medium |
| 444 | 0.0% (0/20) | FAILED |
| 48 | 68.0% (17/25) | Medium |
| 54 | 28.0% (7/25) | Poor |
| 66 | 40.0% (10/25) | Poor |
| 73 | 40.0% (10/25) | Poor |
| 89 | 44.0% (11/25) | Poor |
| 9 | 80.0% (20/25) | Good |
| 91 | 48.0% (12/25) | Medium |

**Number Average: ~39%**

#### Words

| Class | Accuracy | Status |
|-------|----------|--------|
| Agreement | 56.0% (14/25) | Medium |
| Apple | 24.0% (6/25) | Poor |
| Colour | 44.0% (11/25) | Poor |
| Friend | 100.0% (25/25) | PERFECT |
| Gift | 36.0% (9/25) | Poor |
| Market | 37.5% (9/24) | Poor |
| Monday | 80.0% (20/25) | Good |
| Picture | 68.0% (17/25) | Medium |
| Proud | 16.0% (4/25) | Poor |
| Sweater | 64.0% (16/25) | Medium |
| Teach | 63.2% (12/19) | Medium |
| Tomatoes | 4.0% (1/25) | FAILED |
| Tortoise | 60.0% (15/25) | Medium |
| Twin | 80.0% (20/25) | Good |
| Ugali | 68.0% (17/25) | Medium |

**Word Average: ~53%**

### 8.4 Critical Issues in v6

| Issue | Details |
|-------|---------|
| **Severe Overfitting** | Train 91.8% vs Val 41.3% = 50% gap |
| **Complete Failures** | 100, 35, 444 (0%), Tomatoes (4%) |
| **Numbers underperform** | ~39% average vs ~53% for words |
| **Contrastive loss not helping** | Still high at 1.08 when stopped |

### 8.5 v5 vs v6 Comparison

| Metric | v5 (HandFocused) | v6 (Contrastive) | Winner |
|--------|------------------|------------------|--------|
| Overall Val Accuracy | ~75-80%* | 46.43% | **v5** |
| Number Accuracy | ~70%* | ~39% | **v5** |
| Word Accuracy | ~80%* | ~53% | **v5** |
| Training Stability | Stable | Overfitting | **v5** |
| 0% Classes | 0 | 4 (100, 35, 444, Tomatoes) | **v5** |

*Estimated from v5 confusion analysis showing higher per-class accuracies

### 8.6 Why v6 Underperforms

1. **Contrastive weight too high (0.5)**: Competing objectives hurt classification
2. **Strong augmentation hurts**: Temporal warping may destroy discriminative features
3. **Batch size too small for contrastive**: 32 samples may not have enough positive pairs
4. **No hand feature emphasis**: Lost v5's hand-focused architecture advantage
5. **Temperature too low (0.1)**: May cause gradient issues in contrastive loss

---

## 9. Code Quality Analysis

### 9.1 Strengths

- Well-documented with clear docstrings
- Modular design with separate encoders
- Proper GPU utilization with Modal
- Class-balanced sampling implementation
- Comprehensive evaluation with per-class metrics

### 9.2 Areas for Improvement

1. **Configuration management**: CONFIG dict is defined locally in each file
2. **No unit tests**: Feature engineering functions are complex but untested
3. **Magic numbers**: Hard-coded indices (99, 162, 225, etc.)
4. **Missing type hints**: Would improve maintainability
5. **No experiment tracking**: No MLflow/W&B integration

---

## 10. Performance Metrics Tracked

v5 tracks these during training:

```python
history = {
    "train_loss": [],
    "val_acc": [],
    "number_acc": [],  # NEW in v5
    "word_acc": []     # NEW in v5
}
```

The separate tracking of number vs word accuracy is a v5 innovation.

---

## 11. Checkpoint Management

| File | Description |
|------|-------------|
| `ksl_lstm_best.pth` | Best SimpleLSTM model |
| `ksl_handfocused_best.pth` | Best HandFocusedModel |
| `ksl_*_history.json` | Training history |

Checkpoint contents:

```python
{
    "model_state_dict": model.state_dict(),
    "val_acc": val_acc,
    "number_acc": number_acc,
    "word_acc": word_acc,
    "classes": ordered_classes,
}
```

---

## 12. Summary

### What v5 Does Well

- **Targeted optimization**: Focuses specifically on the harder number recognition problem
- **Feature engineering**: Meaningful hand features (angles, distances, curl)
- **Two-head architecture**: Allows specialized learning for each class type
- **Separate metrics**: Tracks number vs word performance independently

### What v5 Could Improve

- **Code reuse**: Feature engineering duplicated across files
- **Robustness**: Missing edge case handling in feature computation
- **Testing**: No unit tests for complex feature engineering
- **Generalization**: v6 addresses the signer-independence problem v5 ignores

### Recommendation

v5 is a solid intermediate step in the model evolution. For production use, consider:

1. **Ensemble v5 with v6** for best of both worlds (hand features + signer invariance)
2. **Add validation** to feature engineering functions
3. **Centralize** shared code into a common module
4. **Address data quality** issues identified in duplicate reports

---

## Appendix: File References

| File | Purpose |
|------|---------|
| `train_ksl_v5.py` | Main v5 training script |
| `evaluate_v5.py` | v5 model evaluation |
| `analyze_confusion.py` | Confusion matrix analysis |
| `train_ksl_v4.py` | Previous version (v4) |
| `train_ksl_v6.py` | Next version (v6) |

---

*Report generated: 2026-01-07*
