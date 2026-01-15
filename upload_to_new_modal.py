"""
Upload local data to new Modal account.
Run this AFTER setting up the new Modal account.

Usage:
    1. First run: modal token new  (to login with new account)
    2. Then run: modal run upload_to_new_modal.py
"""
import modal
import os

app = modal.App("ksl-setup")

# Create a new volume (will be created on first run)
volume = modal.Volume.from_name("ksl-dataset-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(volumes={"/data": volume}, timeout=7200, image=image)
def upload_numpy_files(files_data: list, target_dir: str):
    """Upload numpy files to Modal volume."""
    import os
    import numpy as np

    os.makedirs(f"/data/{target_dir}", exist_ok=True)

    for item in files_data:
        file_path = f"/data/{target_dir}/{item['filename']}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Reconstruct numpy array
        arr = np.frombuffer(item["data"], dtype=item["dtype"]).reshape(item["shape"])
        np.save(file_path, arr)
        print(f"  Uploaded: {file_path} - shape {arr.shape}")

    # Commit to persist
    volume.commit()
    return len(files_data)


@app.function(volumes={"/data": volume}, timeout=300, image=image)
def verify_upload():
    """Verify uploaded files."""
    import os

    print("\n" + "="*70)
    print("VERIFYING UPLOAD")
    print("="*70)

    for subdir in ["train_v2", "val_v2"]:
        path = f"/data/{subdir}"
        if os.path.exists(path):
            # Count files per class
            class_counts = {}
            for class_name in os.listdir(path):
                class_path = os.path.join(path, class_name)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) if f.endswith(".npy")])
                    class_counts[class_name] = count

            print(f"\n{subdir}: {sum(class_counts.values())} files across {len(class_counts)} classes")
            for cls, cnt in sorted(class_counts.items()):
                print(f"  {cls}: {cnt} samples")
        else:
            print(f"\n{subdir}: NOT FOUND")

    return True


@app.local_entrypoint()
def main():
    import os
    import numpy as np

    script_dir = os.path.dirname(os.path.abspath(__file__))
    modal_backup_dir = os.path.join(script_dir, "modal_backup")

    if not os.path.exists(modal_backup_dir):
        print("ERROR: modal_backup directory not found!")
        print(f"Expected at: {modal_backup_dir}")
        print("\nPlease run 'modal run download_modal_data.py' first with your old account.")
        return

    print("="*70)
    print("UPLOADING DATA TO NEW MODAL ACCOUNT")
    print("="*70)

    for subdir in ["train_v2", "val_v2"]:
        source_dir = os.path.join(modal_backup_dir, subdir)

        if not os.path.exists(source_dir):
            print(f"\nSkipping {subdir} - not found")
            continue

        print(f"\n" + "-"*70)
        print(f"Uploading {subdir}...")
        print("-"*70)

        # Gather all files
        files_data = []
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.endswith(".npy"):
                        file_path = os.path.join(class_dir, f)
                        arr = np.load(file_path)
                        files_data.append({
                            "filename": f"{class_name}/{f}",
                            "shape": arr.shape,
                            "dtype": str(arr.dtype),
                            "data": arr.tobytes()
                        })

        if files_data:
            # Upload in batches to avoid memory issues
            batch_size = 50
            total = len(files_data)

            for i in range(0, total, batch_size):
                batch = files_data[i:i+batch_size]
                count = upload_numpy_files.remote(batch, subdir)
                print(f"  Batch {i//batch_size + 1}: uploaded {count} files")

            print(f"Uploaded {total} files to {subdir}")

    # Verify
    print("\n" + "="*70)
    verify_upload.remote()

    print("\n" + "="*70)
    print("UPLOAD COMPLETE!")
    print("="*70)
    print("\nYou can now run training with:")
    print("  modal run train_ksl_v9.py")
