# KSL v9 Implementation Plan - Temporal Segmentation for Multi-Digit Numbers

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 444→54 confusion by detecting repeated digit patterns and adding duration-aware features.

**Architecture:** Add temporal segmentation module that detects repetition patterns (for 444, 100) and duration features. Use a two-stage approach: first detect if multi-digit, then classify.

**Tech Stack:** PyTorch, NumPy, Modal (A10G GPU)

---

## Analysis Summary (from v8 results)

### Critical: 444 at 0%
- 13/20 (65%) → 54 (attractor)
- 5/20 (25%) → 35
- 2/20 (10%) → Colour

**Root Cause:** 444 and 54 have 0.9990 cosine similarity in raw features. The model cannot distinguish them with current features.

**Key Insight from analyze_class_54.py:**
- 444 duration: 101.2 frames (std: 5.7)
- 54 duration: 96.8 frames (std: 7.5)
- 444 has THREE "4" movements; 54 has ONE "54" sequence

### Other Issues to Address
| Class | Accuracy | Main Confusion | Count |
|-------|----------|----------------|-------|
| 89 | 56% | Teach, 73, 9 | 11 errors |
| 66 | 76% | 17 | 6 errors |
| 73 | 68% | 17, 89 | 6 errors |
| 91 | 72% | 73 | 5 errors |

### What's Working (keep unchanged)
- 22→54: 0% (fixed in v8)
- 100→54: 0% (fixed in v8)
- 125→54: 0% (fixed in v8)
- Protected classes: 92%+ (Tortoise, 388, 9, 268)
- Words: 88.6% average

---

## v9 Strategy

### 1. Repetition Detection Features
For 444: detect three similar hand poses in sequence
- Segment the temporal sequence into thirds
- Compute similarity between segments
- High similarity across thirds = repeated digit

### 2. Duration-Based Classification
- Add normalized duration as a feature
- 444 takes ~5% longer than 54
- Multi-digit numbers (100, 444) are longer

### 3. Transition Detection
- 444 has "reset" movements between digits
- 100 has distinct "1" then "00" pattern
- Detect velocity dips (hand returns to neutral)

### 4. Targeted Confusion Fixes
- 66 vs 17: Add distinguishing features
- 89 vs 73: Temporal pattern differences
- 91 vs 73: Similar issue

---

## Task 1: Create Feature Analysis Script

**Files:**
- Create: `analyze_444_vs_54.py`

**Purpose:** Deep dive into what makes 444 different from 54 temporally.

**Step 1: Create analysis script**

```python
"""
Deep temporal analysis of 444 vs 54 to find discriminative features.
"""
import modal
import os

app = modal.App("ksl-analyze-444")
volume = modal.Volume.from_name("ksl-dataset-vol")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy", "matplotlib")

@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def analyze_temporal():
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("TEMPORAL ANALYSIS: 444 vs 54")
    print("=" * 70)
    
    def load_class_samples(class_name, data_dir="/data/train_v2"):
        samples = []
        class_dir = f"{data_dir}/{class_name}"
        if os.path.exists(class_dir):
            for fn in os.listdir(class_dir):
                if fn.endswith(".npy"):
                    samples.append(np.load(os.path.join(class_dir, fn)))
        return samples
    
    samples_444 = load_class_samples("444")
    samples_54 = load_class_samples("54")
    
    print(f"\n444 samples: {len(samples_444)}")
    print(f"54 samples: {len(samples_54)}")
    
    # 1. Duration analysis
    print("\n" + "="*70)
    print("1. DURATION ANALYSIS")
    print("="*70)
    dur_444 = [s.shape[0] for s in samples_444]
    dur_54 = [s.shape[0] for s in samples_54]
    print(f"444: mean={np.mean(dur_444):.1f}, std={np.std(dur_444):.1f}, range=[{min(dur_444)}, {max(dur_444)}]")
    print(f"54:  mean={np.mean(dur_54):.1f}, std={np.std(dur_54):.1f}, range=[{min(dur_54)}, {max(dur_54)}]")
    
    # 2. Repetition detection - segment into thirds
    print("\n" + "="*70)
    print("2. REPETITION PATTERN (segment similarity)")
    print("="*70)
    
    def compute_segment_similarity(sample, n_segments=3):
        """Split sample into n segments, compute cosine similarity between them."""
        frames = sample.shape[0]
        segment_size = frames // n_segments
        if segment_size < 5:
            return None
        
        segments = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size
            seg_mean = sample[start:end].mean(axis=0)
            segments.append(seg_mean)
        
        # Compute pairwise similarities
        sims = []
        for i in range(n_segments):
            for j in range(i+1, n_segments):
                norm_i = np.linalg.norm(segments[i])
                norm_j = np.linalg.norm(segments[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    sim = np.dot(segments[i], segments[j]) / (norm_i * norm_j)
                    sims.append(sim)
        return np.mean(sims) if sims else None
    
    sims_444 = [compute_segment_similarity(s) for s in samples_444 if compute_segment_similarity(s) is not None]
    sims_54 = [compute_segment_similarity(s) for s in samples_54 if compute_segment_similarity(s) is not None]
    
    print(f"444 segment similarity: {np.mean(sims_444):.4f} (std: {np.std(sims_444):.4f})")
    print(f"54 segment similarity:  {np.mean(sims_54):.4f} (std: {np.std(sims_54):.4f})")
    print(f"Difference: {np.mean(sims_444) - np.mean(sims_54):.4f}")
    
    # 3. Velocity dips (transitions between digits)
    print("\n" + "="*70)
    print("3. VELOCITY DIPS (transition detection)")
    print("="*70)
    
    def count_velocity_dips(sample, threshold=0.3):
        """Count frames where hand velocity drops below threshold (potential digit transitions)."""
        right_hand = sample[:, 162:225]  # Right hand landmarks
        velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
        # Normalize
        if velocities.max() > 1e-8:
            velocities = velocities / velocities.max()
        dips = np.sum(velocities < threshold)
        return dips, len(velocities)
    
    dips_444 = [count_velocity_dips(s) for s in samples_444]
    dips_54 = [count_velocity_dips(s) for s in samples_54]
    
    dip_ratio_444 = np.mean([d[0]/d[1] for d in dips_444])
    dip_ratio_54 = np.mean([d[0]/d[1] for d in dips_54])
    
    print(f"444 velocity dip ratio: {dip_ratio_444:.4f}")
    print(f"54 velocity dip ratio:  {dip_ratio_54:.4f}")
    
    # 4. Hand position variance over time
    print("\n" + "="*70)
    print("4. POSITIONAL VARIANCE (movement range)")
    print("="*70)
    
    def compute_position_variance(sample):
        right_hand = sample[:, 162:225].reshape(-1, 21, 3)
        wrist_trajectory = right_hand[:, 0, :]  # Wrist movement
        return np.var(wrist_trajectory, axis=0).sum()
    
    var_444 = [compute_position_variance(s) for s in samples_444]
    var_54 = [compute_position_variance(s) for s in samples_54]
    
    print(f"444 position variance: {np.mean(var_444):.6f} (std: {np.std(var_444):.6f})")
    print(f"54 position variance:  {np.mean(var_54):.6f} (std: {np.std(var_54):.6f})")
    
    # 5. Peak detection (counting distinct gestures)
    print("\n" + "="*70)
    print("5. GESTURE PEAKS (distinct movements)")
    print("="*70)
    
    def count_gesture_peaks(sample):
        """Count peaks in hand spread - each digit should have a distinct peak."""
        right_hand = sample[:, 162:225].reshape(-1, 21, 3)
        # Hand spread: distance from thumb to pinky
        spread = np.linalg.norm(right_hand[:, 4] - right_hand[:, 20], axis=1)
        # Smooth
        kernel_size = 5
        spread_smooth = np.convolve(spread, np.ones(kernel_size)/kernel_size, mode='valid')
        # Find peaks (local maxima)
        peaks = 0
        for i in range(1, len(spread_smooth)-1):
            if spread_smooth[i] > spread_smooth[i-1] and spread_smooth[i] > spread_smooth[i+1]:
                if spread_smooth[i] > np.mean(spread_smooth) * 0.8:  # Significant peak
                    peaks += 1
        return peaks
    
    peaks_444 = [count_gesture_peaks(s) for s in samples_444]
    peaks_54 = [count_gesture_peaks(s) for s in samples_54]
    
    print(f"444 gesture peaks: {np.mean(peaks_444):.2f} (expected ~3 for three '4's)")
    print(f"54 gesture peaks:  {np.mean(peaks_54):.2f} (expected ~2 for '5' then '4')")
    
    # Summary
    print("\n" + "="*70)
    print("DISCRIMINATIVE FEATURES SUMMARY")
    print("="*70)
    print(f"""
    Feature                    | 444 vs 54   | Discriminative?
    ---------------------------|-------------|----------------
    Duration                   | +{np.mean(dur_444)-np.mean(dur_54):.1f} frames | {"YES" if abs(np.mean(dur_444)-np.mean(dur_54)) > 3 else "WEAK"}
    Segment similarity         | {np.mean(sims_444)-np.mean(sims_54):+.4f}     | {"YES - 444 more repetitive" if np.mean(sims_444) > np.mean(sims_54) else "CHECK"}
    Velocity dip ratio         | {dip_ratio_444-dip_ratio_54:+.4f}     | {"YES" if abs(dip_ratio_444-dip_ratio_54) > 0.02 else "WEAK"}
    Position variance          | {np.mean(var_444)-np.mean(var_54):+.6f} | {"YES" if abs(np.mean(var_444)-np.mean(var_54)) > 0.001 else "WEAK"}
    Gesture peaks              | {np.mean(peaks_444)-np.mean(peaks_54):+.2f}      | {"YES - 444 has more" if np.mean(peaks_444) > np.mean(peaks_54) else "CHECK"}
    """)
    
    return {
        "duration_diff": np.mean(dur_444) - np.mean(dur_54),
        "segment_sim_diff": np.mean(sims_444) - np.mean(sims_54),
        "peaks_diff": np.mean(peaks_444) - np.mean(peaks_54),
    }

@app.local_entrypoint()
def main():
    results = analyze_temporal.remote()
    print("\nAnalysis complete!")
```

**Step 2: Run analysis**

```bash
modal run analyze_444_vs_54.py
```

**Expected output:** Identifies which temporal features best discriminate 444 from 54.

---

## Task 2: Implement v9 Training Script

**Files:**
- Create: `train_ksl_v9.py`

**Key v9 Features:**
1. **Repetition features** - segment similarity scores
2. **Duration features** - normalized frame count  
3. **Gesture peak count** - number of distinct movements
4. **Velocity dip features** - transition detection

**Step 1: Create v9 training script**

The script should include:

```python
# NEW FEATURES FOR v9 (add to engineer_features function)

def compute_temporal_features(data):
    """Compute temporal discriminative features for multi-digit detection."""
    frames = data.shape[0]
    features = []
    
    # 1. Normalized duration (relative to expected 90 frames)
    duration_norm = frames / 90.0
    features.append(duration_norm)
    
    # 2. Segment similarity (3 segments)
    segment_size = frames // 3
    if segment_size >= 5:
        segments = [data[i*segment_size:(i+1)*segment_size].mean(axis=0) for i in range(3)]
        for i in range(3):
            for j in range(i+1, 3):
                norm_i, norm_j = np.linalg.norm(segments[i]), np.linalg.norm(segments[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    features.append(np.dot(segments[i], segments[j]) / (norm_i * norm_j))
                else:
                    features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # 3. Right hand velocity statistics
    right_hand = data[:, 162:225]
    velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
    features.append(velocities.mean())
    features.append(velocities.std())
    features.append(np.sum(velocities < velocities.mean() * 0.3) / len(velocities))  # Dip ratio
    
    # 4. Gesture peaks (hand spread peaks)
    right_hand_3d = right_hand.reshape(-1, 21, 3)
    spread = np.linalg.norm(right_hand_3d[:, 4] - right_hand_3d[:, 20], axis=1)
    # Count significant peaks
    peaks = 0
    for i in range(1, len(spread)-1):
        if spread[i] > spread[i-1] and spread[i] > spread[i+1]:
            if spread[i] > spread.mean() * 0.8:
                peaks += 1
    features.append(peaks / 10.0)  # Normalize
    
    return np.array(features, dtype=np.float32)
```

**Step 2: Update feature dimension**

Old: 649 features
New: 649 + 8 temporal features = 657 features

**Step 3: Add 444-specific loss**

```python
# Special handling for 444 class
MULTI_DIGIT_CLASSES = ["444", "100"]  # Classes with repeated/multi digits

# In loss computation, add extra penalty for 444 misclassification
if true_class == "444" and pred_class != "444":
    loss *= 4.0  # Heavy penalty for missing 444
```

**Step 4: Add confusion pairs for new issues**

```python
# Updated confusion pairs for v9
CONFUSION_PAIRS = [
    # 444 attractor (CRITICAL)
    ("444", "54", 5.0),   # Increased from 3.5
    ("444", "35", 3.0),   # New: 5 errors in v8
    
    # Existing (keep)
    ("22", "54", 2.5),
    ("125", "54", 2.5),
    ("100", "54", 2.0),
    
    # New confusions from v8
    ("66", "17", 2.5),    # 6 errors
    ("125", "22", 2.0),   # 5 errors
    ("91", "73", 2.0),    # 5 errors
    ("89", "73", 2.0),    # 3 errors
    ("89", "Teach", 2.0), # 4 errors
    ("73", "17", 2.0),    # 3 errors
]
```

---

## Task 3: Implement Two-Stage Classifier (Optional Advanced)

**Files:**
- Modify: `train_ksl_v9.py`

**Concept:** First classify if single-digit or multi-digit, then use specialized heads.

```python
class TwoStageModel(nn.Module):
    def __init__(self, num_classes, feature_dim=657, hidden_dim=320):
        super().__init__()
        # Shared backbone (same as v8)
        self.backbone = TemporalPyramidBackbone(feature_dim, hidden_dim)
        
        # Stage 1: Multi-digit detector
        self.multi_digit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Stage 2: Class-specific heads
        self.single_digit_head = nn.Linear(hidden_dim * 2, num_classes)
        self.multi_digit_head_classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Predict if multi-digit
        is_multi = self.multi_digit_head(features)
        
        # Get both predictions
        single_logits = self.single_digit_head(features)
        multi_logits = self.multi_digit_head_classifier(features)
        
        # Weighted combination based on multi-digit probability
        # This allows gradient flow through both paths
        logits = (1 - is_multi) * single_logits + is_multi * multi_logits
        
        return logits, is_multi
```

---

## Task 4: Create v9 Assessment Script

**Files:**
- Create: `assess_v9.py`

**Step 1: Copy from v8 and update**
- Change checkpoint path to `/data/checkpoints_v9/`
- Update feature_dim to 657
- Add temporal feature computation
- Track 444 specifically

---

## Task 5: Run Training and Iterate

**Step 1: Run temporal analysis first**
```bash
modal run analyze_444_vs_54.py
```

**Step 2: Train v9**
```bash
modal run train_ksl_v9.py
```

**Step 3: Assess v9**
```bash
modal run assess_v9.py
```

**Expected Improvements:**
| Class | v8 | v9 Target |
|-------|-----|-----------|
| 444 | 0% | >40% |
| 89 | 56% | >70% |
| 66 | 76% | >85% |
| 73 | 68% | >80% |
| Overall | 83.65% | >85% |

---

## Success Criteria

1. **444 accuracy > 40%** (from 0%)
2. **No regression on protected classes** (>90%)
3. **Overall accuracy > 85%**
4. **Attractor rate < 10%** (from 13.7%)

---

## Rollback Plan

If v9 causes regressions:
1. Keep v8 checkpoint as backup
2. Can ensemble v8 + v9 for best of both
3. Focus only on 444-specific fixes if general approach fails
