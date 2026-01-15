"""
Download KSL Model Checkpoints from Modal

This script downloads trained model checkpoints from Modal volume storage
to the local checkpoints directory for use with the inference server.

Usage:
    # Download all available checkpoints
    modal run download_checkpoints.py

    # Download specific version
    modal run download_checkpoints.py --version v14
"""

import modal
import os
from pathlib import Path

app = modal.App("ksl-download-checkpoints")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = modal.Image.debian_slim(python_version="3.11")


@app.function(volumes={"/data": volume}, timeout=600, image=image)
def list_checkpoints() -> dict:
    """List all available checkpoint directories."""
    checkpoints = {}

    # Check checkpoint directories
    checkpoint_patterns = [
        "/data/checkpoints_v{}_numbers",
        "/data/checkpoints_v{}_words",
        "/data/checkpoints_v{}",
    ]

    versions = ["8", "9", "10", "11", "12", "13", "14"]

    for version in versions:
        for pattern in checkpoint_patterns:
            dir_path = pattern.format(version)
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                pt_files = [f for f in files if f.endswith(('.pt', '.pth'))]
                if pt_files:
                    checkpoints[dir_path] = pt_files
                    print(f"Found: {dir_path} -> {pt_files}")

    return checkpoints


@app.function(volumes={"/data": volume}, timeout=1800, image=image)
def download_checkpoint(remote_dir: str, filename: str) -> bytes:
    """Download a single checkpoint file."""
    filepath = os.path.join(remote_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    with open(filepath, "rb") as f:
        data = f.read()

    print(f"Downloaded {filepath} ({len(data) / 1024 / 1024:.1f} MB)")
    return data


@app.local_entrypoint()
def main(version: str = None):
    """Download checkpoints to local directory."""
    import argparse

    # Create local checkpoints directory
    local_dir = Path(__file__).parent / "checkpoints"
    local_dir.mkdir(exist_ok=True)

    print("Listing available checkpoints...")
    checkpoints = list_checkpoints.remote()

    if not checkpoints:
        print("No checkpoints found on Modal volume!")
        return

    print(f"\nFound {len(checkpoints)} checkpoint directories")

    for remote_dir, files in checkpoints.items():
        # Extract version and type from directory name
        dir_name = os.path.basename(remote_dir)

        # Parse directory name like "checkpoints_v14_numbers"
        parts = dir_name.replace("checkpoints_", "").split("_")
        if len(parts) >= 2:
            v = parts[0]  # "v14"
            model_type = parts[1] if len(parts) > 1 else "numbers"
        else:
            v = parts[0]
            model_type = "numbers"

        # Filter by version if specified
        if version and v != version:
            continue

        # Create local directory
        local_subdir = local_dir / f"{v}_{model_type}"
        local_subdir.mkdir(exist_ok=True)

        # Download each file
        for filename in files:
            local_path = local_subdir / filename
            if local_path.exists():
                print(f"Skipping {local_path} (already exists)")
                continue

            print(f"Downloading {remote_dir}/{filename}...")
            try:
                data = download_checkpoint.remote(remote_dir, filename)
                with open(local_path, "wb") as f:
                    f.write(data)
                print(f"Saved to {local_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

    print("\nDone!")
    print(f"Checkpoints saved to: {local_dir}")
