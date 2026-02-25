"""
Check validation data signer distribution.

Usage:
    modal run check_val_signers.py
"""

import modal
import os

app = modal.App("ksl-check-val-signers")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    volumes={"/data": volume},
    timeout=300,
    image=image,
)
def check_signers():
    from collections import defaultdict

    print("=" * 70)
    print("Validation Data Signer Distribution")
    print("=" * 70)

    val_dir = "/data/val_v2"

    signer_counts = defaultdict(int)
    class_signer_counts = defaultdict(lambda: defaultdict(int))

    for class_name in sorted(os.listdir(val_dir)):
        class_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith('.npy'):
                # Extract signer from filename (e.g., "signer8_video.npy")
                if filename.startswith('signer'):
                    signer = filename.split('_')[0]
                    signer_counts[signer] += 1
                    class_signer_counts[class_name][signer] += 1
                else:
                    signer_counts['unknown'] += 1
                    class_signer_counts[class_name]['unknown'] += 1

    print("\nOverall signer distribution:")
    for signer in sorted(signer_counts.keys()):
        print(f"  {signer}: {signer_counts[signer]} samples")

    print(f"\nTotal: {sum(signer_counts.values())} samples")

    print("\n" + "=" * 70)
    print("Per-class signer distribution:")
    print("=" * 70)

    # Check classes with potential issues
    for class_name in sorted(class_signer_counts.keys()):
        signers = class_signer_counts[class_name]
        total = sum(signers.values())
        signer_str = ", ".join(f"{s}:{c}" for s, c in sorted(signers.items()))
        print(f"  {class_name:12s}: {total:3d} samples ({signer_str})")

    return dict(signer_counts)


@app.local_entrypoint()
def main():
    check_signers.remote()
