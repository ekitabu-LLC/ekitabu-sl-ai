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
