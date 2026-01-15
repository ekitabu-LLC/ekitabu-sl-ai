"""
Analyze sign variations within classes.
Clusters samples within each class to identify different signing styles.

Usage:
    modal run analyze_variations.py
"""

import modal
import os

app = modal.App("ksl-variation-analysis")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
HARD_CLASSES = ["22", "444", "89", "35", "Market", "Teach"]


@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def analyze_variations():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from collections import defaultdict

    print("=" * 70)
    print("KSL Variation Analysis")
    print("Detecting multiple sign variations within classes")
    print("=" * 70)

    def extract_key_features(data):
        """Extract key features for clustering - focus on hand positions."""
        # Average hand positions across frames
        # Left hand: 99:162, Right hand: 162:225

        # Skip first/last 10% of frames (often noisy)
        n_frames = len(data)
        start = int(n_frames * 0.1)
        end = int(n_frames * 0.9)
        data = data[start:end] if end > start else data

        features = []

        # Mean hand positions
        left_hand_mean = data[:, 99:162].mean(axis=0)
        right_hand_mean = data[:, 162:225].mean(axis=0)
        features.extend(left_hand_mean)
        features.extend(right_hand_mean)

        # Std of hand positions (movement amount)
        left_hand_std = data[:, 99:162].std(axis=0).mean()
        right_hand_std = data[:, 162:225].std(axis=0).mean()
        features.append(left_hand_std)
        features.append(right_hand_std)

        # Hand activity ratio (which hand moves more)
        left_activity = np.diff(data[:, 99:162], axis=0).std()
        right_activity = np.diff(data[:, 162:225], axis=0).std()
        features.append(left_activity / (right_activity + 1e-8))

        return np.array(features)

    def analyze_class(class_name, train_dir, val_dir):
        """Analyze variations for a single class."""
        train_features = []
        val_features = []
        train_files = []
        val_files = []

        # Load training samples
        train_path = f"{train_dir}/{class_name}"
        if os.path.exists(train_path):
            for f in os.listdir(train_path):
                if f.endswith('.npy'):
                    data = np.load(os.path.join(train_path, f))
                    feat = extract_key_features(data)
                    train_features.append(feat)
                    train_files.append(f)

        # Load validation samples
        val_path = f"{val_dir}/{class_name}"
        if os.path.exists(val_path):
            for f in os.listdir(val_path):
                if f.endswith('.npy'):
                    data = np.load(os.path.join(val_path, f))
                    feat = extract_key_features(data)
                    val_features.append(feat)
                    val_files.append(f)

        if len(train_features) < 3 or len(val_features) < 1:
            return None

        train_features = np.array(train_features)
        val_features = np.array(val_features)

        # Normalize
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0) + 1e-8
        train_norm = (train_features - mean) / std
        val_norm = (val_features - mean) / std

        # Try different numbers of clusters
        best_k = 1
        best_score = -1

        for k in range(2, min(5, len(train_features))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(train_norm)
            if len(set(labels)) > 1:
                score = silhouette_score(train_norm, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        # Cluster with best k
        if best_k > 1 and best_score > 0.2:  # Only if clusters are meaningful
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            train_labels = kmeans.fit_predict(train_norm)
            val_labels = kmeans.predict(val_norm)

            # Count samples per cluster
            train_clusters = defaultdict(int)
            val_clusters = defaultdict(int)
            for l in train_labels:
                train_clusters[l] += 1
            for l in val_labels:
                val_clusters[l] += 1

            return {
                "class": class_name,
                "n_clusters": best_k,
                "silhouette": best_score,
                "train_clusters": dict(train_clusters),
                "val_clusters": dict(val_clusters),
                "train_total": len(train_features),
                "val_total": len(val_features),
            }
        else:
            return {
                "class": class_name,
                "n_clusters": 1,
                "silhouette": 0,
                "train_total": len(train_features),
                "val_total": len(val_features),
            }

    # Analyze hard classes
    print("\nAnalyzing variation patterns in hard classes...")
    print("-" * 70)

    results = []
    for class_name in HARD_CLASSES:
        result = analyze_class(class_name, "/data/train_v2", "/data/val_v2")
        if result:
            results.append(result)

    # Print results
    print("\n" + "=" * 70)
    print("VARIATION ANALYSIS RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n{r['class']}:")
        print(f"  Training samples: {r['train_total']}")
        print(f"  Validation samples: {r['val_total']}")
        print(f"  Detected variations: {r['n_clusters']}")

        if r['n_clusters'] > 1:
            print(f"  Silhouette score: {r['silhouette']:.3f}")
            print(f"  Training distribution: {r.get('train_clusters', {})}")
            print(f"  Validation distribution: {r.get('val_clusters', {})}")

            # Check for mismatch
            train_clusters = set(r.get('train_clusters', {}).keys())
            val_clusters = set(r.get('val_clusters', {}).keys())

            if val_clusters - train_clusters:
                print(f"  ⚠️  MISMATCH: Validation has clusters not seen in training!")

    # Also check train vs val distance
    print("\n" + "=" * 70)
    print("TRAIN vs VALIDATION DISTANCE")
    print("(High distance = validation signer signs differently)")
    print("=" * 70)

    for class_name in HARD_CLASSES:
        train_path = f"/data/train_v2/{class_name}"
        val_path = f"/data/val_v2/{class_name}"

        train_features = []
        val_features = []

        if os.path.exists(train_path):
            for f in os.listdir(train_path)[:20]:  # Sample
                if f.endswith('.npy'):
                    data = np.load(os.path.join(train_path, f))
                    train_features.append(extract_key_features(data))

        if os.path.exists(val_path):
            for f in os.listdir(val_path)[:20]:
                if f.endswith('.npy'):
                    data = np.load(os.path.join(val_path, f))
                    val_features.append(extract_key_features(data))

        if train_features and val_features:
            train_mean = np.mean(train_features, axis=0)
            val_mean = np.mean(val_features, axis=0)
            distance = np.linalg.norm(train_mean - val_mean)

            train_std = np.std(train_features)
            normalized_dist = distance / (train_std + 1e-8)

            status = "⚠️ HIGH" if normalized_dist > 2.0 else "✓ OK"
            print(f"  {class_name:>8}: {normalized_dist:.2f} {status}")

    return {"analyzed": HARD_CLASSES}


@app.local_entrypoint()
def main():
    analyze_variations.remote()
