"""
Diagnose domain shift between training and test data.
Compare statistical properties of both distributions.

Usage:
    modal run diagnose_domain_shift.py
"""

import modal
import os

app = modal.App("ksl-diagnose")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
)


@app.function(volumes={"/data": volume}, timeout=600, image=image)
def diagnose():
    import numpy as np

    print("=" * 70)
    print("Domain Shift Diagnosis: Training vs Test Data")
    print("=" * 70)

    POSE_INDICES = [11, 12, 13, 14, 15, 16]

    def analyze_sample(data):
        """Analyze a single sample's statistics."""
        f = data.shape[0]
        stats = {}

        # Check data dimensions
        stats["frames"] = f
        stats["features"] = data.shape[1]

        if data.shape[1] >= 225:
            # Extract pose
            pose_vals = []
            for pi, idx in enumerate(POSE_INDICES):
                start = idx * 3
                pose_vals.extend(data[:, start:start+3].flatten())
            stats["pose_mean"] = np.mean(pose_vals)
            stats["pose_std"] = np.std(pose_vals)
            stats["pose_nonzero"] = np.sum(np.array(pose_vals) != 0) / len(pose_vals)

            # Left hand (99:162)
            lh = data[:, 99:162]
            stats["lh_mean"] = np.mean(lh)
            stats["lh_std"] = np.std(lh)
            stats["lh_nonzero"] = np.sum(lh != 0) / lh.size
            stats["lh_range"] = np.max(lh) - np.min(lh)

            # Right hand (162:225)
            rh = data[:, 162:225]
            stats["rh_mean"] = np.mean(rh)
            stats["rh_std"] = np.std(rh)
            stats["rh_nonzero"] = np.sum(rh != 0) / rh.size
            stats["rh_range"] = np.max(rh) - np.min(rh)

            # Hand dominance
            lh_activity = np.sum(np.abs(lh))
            rh_activity = np.sum(np.abs(rh))
            stats["hand_ratio"] = lh_activity / (rh_activity + 1e-8)

        return stats

    def summarize_stats(all_stats, name):
        """Summarize statistics across all samples."""
        if not all_stats:
            return

        print(f"\n{name} ({len(all_stats)} samples):")

        keys = ["frames", "pose_mean", "pose_std", "pose_nonzero",
                "lh_mean", "lh_std", "lh_nonzero", "lh_range",
                "rh_mean", "rh_std", "rh_nonzero", "rh_range", "hand_ratio"]

        for key in keys:
            vals = [s.get(key, 0) for s in all_stats if key in s]
            if vals:
                print(f"  {key:15s}: mean={np.mean(vals):8.4f}, std={np.std(vals):8.4f}, "
                      f"min={np.min(vals):8.4f}, max={np.max(vals):8.4f}")

    # Analyze training data
    print("\n" + "=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)

    train_stats = {"numbers": [], "words": []}
    train_dir = "/data/train_v2"

    NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
    WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                           "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])

    # Numbers
    for cls in NUMBER_CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            for fn in os.listdir(cls_dir):
                if fn.endswith(".npy"):
                    data = np.load(os.path.join(cls_dir, fn))
                    stats = analyze_sample(data)
                    stats["class"] = cls
                    train_stats["numbers"].append(stats)

    # Words
    for cls in WORD_CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            for fn in os.listdir(cls_dir):
                if fn.endswith(".npy"):
                    data = np.load(os.path.join(cls_dir, fn))
                    stats = analyze_sample(data)
                    stats["class"] = cls
                    train_stats["words"].append(stats)

    summarize_stats(train_stats["numbers"], "Training Numbers")
    summarize_stats(train_stats["words"], "Training Words")

    # Analyze test data
    print("\n" + "=" * 70)
    print("TEST DATA ANALYSIS")
    print("=" * 70)

    test_stats = {"numbers": [], "words": []}
    test_dir = "/data/testing_v2"

    # Numbers
    num_dir = os.path.join(test_dir, "Numbers")
    if os.path.exists(num_dir):
        for cls in os.listdir(num_dir):
            cls_dir = os.path.join(num_dir, cls)
            if os.path.isdir(cls_dir):
                for fn in os.listdir(cls_dir):
                    if fn.endswith(".npy"):
                        data = np.load(os.path.join(cls_dir, fn))
                        stats = analyze_sample(data)
                        stats["class"] = cls
                        test_stats["numbers"].append(stats)

    # Words
    word_dir = os.path.join(test_dir, "Words")
    if os.path.exists(word_dir):
        for cls in os.listdir(word_dir):
            cls_dir = os.path.join(word_dir, cls)
            if os.path.isdir(cls_dir):
                for fn in os.listdir(cls_dir):
                    if fn.endswith(".npy"):
                        data = np.load(os.path.join(cls_dir, fn))
                        stats = analyze_sample(data)
                        stats["class"] = cls
                        test_stats["words"].append(stats)

    summarize_stats(test_stats["numbers"], "Test Numbers")
    summarize_stats(test_stats["words"], "Test Words")

    # Per-class comparison
    print("\n" + "=" * 70)
    print("PER-CLASS COMPARISON (Train vs Test)")
    print("=" * 70)

    print("\nNUMBERS - Hand Ratio (LH/RH activity):")
    for cls in NUMBER_CLASSES:
        train_cls = [s for s in train_stats["numbers"] if s["class"] == cls]
        test_cls = [s for s in test_stats["numbers"] if s["class"] == cls]

        train_ratio = np.mean([s["hand_ratio"] for s in train_cls]) if train_cls else 0
        test_ratio = np.mean([s["hand_ratio"] for s in test_cls]) if test_cls else 0

        print(f"  {cls:5s}: Train={train_ratio:6.3f}, Test={test_ratio:6.3f}, "
              f"Diff={abs(train_ratio - test_ratio):6.3f}")

    print("\nWORDS - Hand Ratio (LH/RH activity):")
    for cls in WORD_CLASSES:
        train_cls = [s for s in train_stats["words"] if s["class"] == cls]
        test_cls = [s for s in test_stats["words"] if s["class"] == cls]

        train_ratio = np.mean([s["hand_ratio"] for s in train_cls]) if train_cls else 0
        test_ratio = np.mean([s["hand_ratio"] for s in test_cls]) if test_cls else 0

        print(f"  {cls:12s}: Train={train_ratio:6.3f}, Test={test_ratio:6.3f}, "
              f"Diff={abs(train_ratio - test_ratio):6.3f}")

    # Check class 444 specifically (the model's favorite prediction)
    print("\n" + "=" * 70)
    print("CLASS 444 ANALYSIS (Model's favorite prediction)")
    print("=" * 70)

    train_444 = [s for s in train_stats["numbers"] if s["class"] == "444"]
    test_all_nums = test_stats["numbers"]

    if train_444:
        print(f"\nTraining 444 samples: {len(train_444)}")
        print(f"  Hand ratio: {np.mean([s['hand_ratio'] for s in train_444]):.3f}")
        print(f"  LH nonzero: {np.mean([s['lh_nonzero'] for s in train_444]):.3f}")
        print(f"  RH nonzero: {np.mean([s['rh_nonzero'] for s in train_444]):.3f}")

    if test_all_nums:
        print(f"\nTest numbers samples: {len(test_all_nums)}")
        print(f"  Hand ratio: {np.mean([s['hand_ratio'] for s in test_all_nums]):.3f}")
        print(f"  LH nonzero: {np.mean([s['lh_nonzero'] for s in test_all_nums]):.3f}")
        print(f"  RH nonzero: {np.mean([s['rh_nonzero'] for s in test_all_nums]):.3f}")

    # Hypothesis: Test data might match 444's training distribution
    if train_444 and test_all_nums:
        train_444_lh = np.mean([s['lh_nonzero'] for s in train_444])
        train_444_rh = np.mean([s['rh_nonzero'] for s in train_444])
        test_avg_lh = np.mean([s['lh_nonzero'] for s in test_all_nums])
        test_avg_rh = np.mean([s['rh_nonzero'] for s in test_all_nums])

        print(f"\nSimilarity to 444:")
        print(f"  Train 444 LH/RH: {train_444_lh:.3f}/{train_444_rh:.3f}")
        print(f"  Test avg  LH/RH: {test_avg_lh:.3f}/{test_avg_rh:.3f}")

    return {"train": train_stats, "test": test_stats}


@app.local_entrypoint()
def main():
    diagnose.remote()
