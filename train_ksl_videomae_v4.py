#!/usr/bin/env python3
"""
KSL VideoMAE v4 — Frame-Difference Representation for Signer-Invariant Training.

Key insight: Raw RGB frames memorize signer appearance (skin tone, clothing).
Frame differences (frame[t+1] - frame[t]) capture ONLY MOTION, which is
sign-specific and signer-invariant. Static background/clothing → diff ≈ 0.

Changes from v1:
  - Load T+1 frames, compute T temporal frame differences
  - Map differences to [0, 1] with ±64 pixel clip (typical hand motion range)
  - Apply ImageNet normalization (same as raw-frame pipeline)
  - All other hyperparams identical to v1

Uses train/ + validation-data/ from the base directory with the same
signer 1-5 (train) / 6-10 (val) split as v1.

Usage:
    python train_ksl_videomae_v4.py --model-type numbers
    python train_ksl_videomae_v4.py --model-type words
"""

import argparse
import os
import random
import time
import glob as glob_module
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUMBER_CLASSES = sorted([
    "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
    "100", "125", "268", "388", "444",
])

WORD_CLASSES = sorted([
    "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
    "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
])

BASE_DIR = "/scratch/alpine/hama5612/ksl-dir-2"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VAL_DIR = os.path.join(BASE_DIR, "validation-data")
CKPT_DIR = os.path.join(BASE_DIR, "data", "checkpoints", "videomae_v4")

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16   # number of frame-differences (need T+1 actual frames)
IMAGE_SIZE = 224

# Clip range for frame differences (pixels).
# Hand motion is typically 5-30 px per frame at 224x224.
# We clip at ±64 to handle fast motion without saturating.
DIFF_CLIP = 64.0

# ImageNet normalization (applied after mapping differences to [0, 1])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Frame-difference video loading
# ---------------------------------------------------------------------------
def load_frame_differences(video_path, num_diffs=NUM_FRAMES, size=IMAGE_SIZE, augment=False):
    """Load NUM_FRAMES+1 raw frames from a video and return NUM_FRAMES frame differences.

    Frame differences are signer-invariant: they capture hand motion only,
    not skin tone, clothing, or background appearance.

    Returns: numpy array of shape (num_diffs, 3, size, size) in ImageNet-normalized space,
             or None on failure.
    """
    num_load = num_diffs + 1  # need one extra frame to form differences

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{ts()}] WARNING: Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        total_frames = len(frames)
        if total_frames == 0:
            print(f"[{ts()}] WARNING: No frames in video: {video_path}")
            return None
    else:
        frames = None
        cap.release()

    # Sample num_load frames uniformly
    if total_frames >= num_load:
        indices = np.linspace(0, total_frames - 1, num_load, dtype=int)
    else:
        indices = np.arange(total_frames)
        indices = np.pad(indices, (0, num_load - total_frames), mode='edge')

    if frames is not None:
        sampled_raw = [frames[i] for i in indices]
    else:
        cap = cv2.VideoCapture(video_path)
        read_frames = {}
        needed = set(indices.tolist())
        frame_idx = 0
        while frame_idx <= max(needed):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in needed:
                read_frames[frame_idx] = frame
            frame_idx += 1
        cap.release()
        sampled_raw = [read_frames.get(i, read_frames.get(max(read_frames.keys()), None))
                       for i in indices.tolist()]
        if any(f is None for f in sampled_raw):
            print(f"[{ts()}] WARNING: Could not read frames from: {video_path}")
            return None

    # Augmentation (applied consistently across all frames)
    if augment:
        do_flip = random.random() < 0.5
        crop_size = int(size * 1.15)
        y_off = random.randint(0, crop_size - size)
        x_off = random.randint(0, crop_size - size)
    else:
        do_flip = False
        crop_size = int(size * 1.15)
        y_off = (crop_size - size) // 2
        x_off = (crop_size - size) // 2

    # Process all num_load frames: resize + crop + flip → float32 in [0, 255]
    processed_raw = []
    for frame in sampled_raw:
        frame = cv2.resize(frame, (crop_size, crop_size))
        frame = frame[y_off:y_off + size, x_off:x_off + size]
        if do_flip:
            frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_raw.append(frame.astype(np.float32))  # [0, 255]

    # Compute frame differences: (T+1 frames) → T differences
    diffs = []
    for i in range(num_diffs):
        diff = processed_raw[i + 1] - processed_raw[i]  # (H, W, 3) in [-255, 255]
        # Map to [0, 1]: clip at ±DIFF_CLIP, then rescale
        diff = np.clip(diff, -DIFF_CLIP, DIFF_CLIP) / (2 * DIFF_CLIP) + 0.5
        # Apply ImageNet normalization (model expects this range)
        diff = (diff - IMAGENET_MEAN) / IMAGENET_STD
        # HWC → CHW
        diff = diff.transpose(2, 0, 1)
        diffs.append(diff)

    return np.stack(diffs)  # (num_diffs, 3, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KSLFrameDiffDataset(Dataset):
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        frames = load_frame_differences(path, augment=self.augment)
        if frames is None:
            frames = np.zeros((NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data discovery (same as v1)
# ---------------------------------------------------------------------------
def discover_training_data(classes):
    paths, labels = [], []
    c2i = {c: i for i, c in enumerate(classes)}
    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[{ts()}] WARNING: Missing class dir: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(".mov") or fname.lower().endswith(".mp4"):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(c2i[cls_name])
    print(f"[{ts()}] Training: {len(paths)} videos across {len(classes)} classes")
    return paths, labels


def discover_validation_data(classes):
    paths, labels = [], []
    c2i = {c: i for i, c in enumerate(classes)}
    is_numbers = all(c.isdigit() for c in classes)

    for signer_dir in sorted(glob_module.glob(os.path.join(VAL_DIR, "*/"))):
        for class_dir in sorted(glob_module.glob(os.path.join(signer_dir, "*/"))):
            dir_name = os.path.basename(class_dir.rstrip("/"))
            cls_name = None
            if dir_name.startswith("No. "):
                cls_name = dir_name[4:].strip()
            else:
                parts = dir_name.split(". ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    cls_name = parts[1].strip()
                else:
                    cls_name = dir_name.strip()
            if cls_name not in c2i:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(".mov") or fname.lower().endswith(".mp4"):
                    paths.append(os.path.join(class_dir, fname))
                    labels.append(c2i[cls_name])

    print(f"[{ts()}] Validation: {len(paths)} videos across {len(classes)} classes")
    return paths, labels


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=frames)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct / total, total_loss / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")

    classes = NUMBER_CLASSES if args.model_type == "numbers" else WORD_CLASSES
    num_classes = len(classes)
    print(f"[{ts()}] Model type: {args.model_type}, Classes: {num_classes}")
    print(f"[{ts()}] Frame-difference representation (signer-invariant motion)")

    train_paths, train_labels = discover_training_data(classes)
    val_paths, val_labels = discover_validation_data(classes)

    if len(train_paths) == 0:
        print(f"[{ts()}] ERROR: No training videos found!")
        return

    train_ds = KSLFrameDiffDataset(train_paths, train_labels, augment=True)
    val_ds   = KSLFrameDiffDataset(val_paths,   val_labels,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"[{ts()}] Loading VideoMAE from: {MODEL_NAME}")
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Total params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_dir = os.path.join(CKPT_DIR, args.model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=frames)
            logits = outputs.logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        epoch_time = time.time() - t0

        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{ts()}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
              f"LR={current_lr:.2e} | Time={epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "classes": classes,
                "model_name": MODEL_NAME,
                "args": vars(args),
                "representation": "frame_difference",
                "diff_clip": DIFF_CLIP,
            }, save_path)
            print(f"[{ts()}]   -> New best! Saved to {save_path}")

    print(f"\n[{ts()}] Training complete!")
    print(f"[{ts()}] Best val accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")

    final_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "classes": classes,
        "model_name": MODEL_NAME,
        "args": vars(args),
        "representation": "frame_difference",
        "diff_clip": DIFF_CLIP,
    }, final_path)
    print(f"[{ts()}] Final model saved to {final_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE v4 (frame differences) for KSL")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["numbers", "words"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"[{ts()}] === KSL VideoMAE v4 Training (Frame-Difference Representation) ===")
    print(f"[{ts()}] Args: {vars(args)}")
    train(args)


if __name__ == "__main__":
    main()
