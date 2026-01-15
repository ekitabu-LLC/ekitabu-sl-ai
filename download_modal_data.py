"""
Download all data from Modal volume to local machine.
Run this IMMEDIATELY before credits expire!

Usage:
    modal run download_modal_data.py
"""
import modal
import os

app = modal.App("ksl-download")
volume = modal.Volume.from_name("ksl-dataset-vol")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(volumes={"/data": volume}, timeout=7200, image=image)
def list_all_files():
    """List all files in the Modal volume."""
    import os

    all_files = []
    for root, dirs, files in os.walk("/data"):
        for f in files:
            full_path = os.path.join(root, f)
            size = os.path.getsize(full_path)
            all_files.append((full_path, size))

    # Sort by path
    all_files.sort(key=lambda x: x[0])

    total_size = sum(s for _, s in all_files)
    print(f"\n{'='*70}")
    print(f"MODAL VOLUME CONTENTS")
    print(f"{'='*70}")
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nDirectories:")

    # Group by top-level directory
    dirs = {}
    for path, size in all_files:
        parts = path.split("/")
        if len(parts) >= 3:
            top_dir = parts[2]
            if top_dir not in dirs:
                dirs[top_dir] = {"count": 0, "size": 0}
            dirs[top_dir]["count"] += 1
            dirs[top_dir]["size"] += size

    for d, info in sorted(dirs.items()):
        print(f"  /data/{d}: {info['count']} files, {info['size']/1024/1024:.1f} MB")

    return all_files


@app.function(volumes={"/data": volume}, timeout=7200, image=image)
def download_directory(remote_dir: str) -> list:
    """Get all .npy files from a directory as bytes."""
    import os
    import numpy as np

    files_data = []

    if not os.path.exists(remote_dir):
        print(f"Directory not found: {remote_dir}")
        return files_data

    for root, dirs, files in os.walk(remote_dir):
        for f in files:
            if f.endswith(".npy"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, "/data")

                # Read the numpy file
                try:
                    data = np.load(full_path)
                    files_data.append({
                        "path": rel_path,
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "data": data.tobytes()
                    })
                    print(f"  Downloaded: {rel_path} - shape {data.shape}")
                except Exception as e:
                    print(f"  Error with {rel_path}: {e}")

    return files_data


@app.local_entrypoint()
def main():
    import os
    import numpy as np

    # First, list all files
    print("Listing Modal volume contents...")
    files = list_all_files.remote()

    # Ask user which directories to download
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    print("The following directories contain processed features for training:")
    print("  1. /data/train_v2 - Training features (ESSENTIAL)")
    print("  2. /data/val_v2 - Validation features (ESSENTIAL)")
    print("  3. /data/checkpoints_* - Model checkpoints (if any)")
    print("\nDownloading train_v2 and val_v2...")

    local_base = os.path.dirname(os.path.abspath(__file__))
    modal_backup_dir = os.path.join(local_base, "modal_backup")
    os.makedirs(modal_backup_dir, exist_ok=True)

    # Download train_v2
    print("\n" + "-"*70)
    print("Downloading train_v2...")
    print("-"*70)
    train_data = download_directory.remote("/data/train_v2")

    # Save locally
    for item in train_data:
        local_path = os.path.join(modal_backup_dir, item["path"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Reconstruct numpy array
        arr = np.frombuffer(item["data"], dtype=item["dtype"]).reshape(item["shape"])
        np.save(local_path, arr)
        print(f"  Saved: {local_path}")

    print(f"\nSaved {len(train_data)} files to {modal_backup_dir}/train_v2")

    # Download val_v2
    print("\n" + "-"*70)
    print("Downloading val_v2...")
    print("-"*70)
    val_data = download_directory.remote("/data/val_v2")

    for item in val_data:
        local_path = os.path.join(modal_backup_dir, item["path"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        arr = np.frombuffer(item["data"], dtype=item["dtype"]).reshape(item["shape"])
        np.save(local_path, arr)
        print(f"  Saved: {local_path}")

    print(f"\nSaved {len(val_data)} files to {modal_backup_dir}/val_v2")

    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"All data saved to: {modal_backup_dir}")
    print(f"Total files: {len(train_data) + len(val_data)}")
