#!/usr/bin/env python3
"""
KSL VideoMAE Training — Fine-tune VideoMAE-Base on raw KSL sign language videos.

Uses HuggingFace VideoMAEForVideoClassification (MCG-NJU/videomae-base-finetuned-kinetics).
Videos loaded with OpenCV, 16 frames uniformly sampled, resized to 224x224.
Trains separately for numbers (15 classes) and words (15 classes).

Training data:  dataset/{ClassName}/{ClassName}-{SignerID}-{RepID}.mov  (signers 1-5)
Validation data: validation-data/{X}. Signer/{N}. {ClassName}/{letter}. {ClassName}.mov (signers 6-10)

Usage:
    python train_ksl_videomae.py --model-type numbers
    python train_ksl_videomae.py --model-type words
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
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

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
CKPT_DIR = os.path.join(BASE_DIR, "data", "checkpoints", "videomae")

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
IMAGE_SIZE = 224

# ImageNet normalization (used by VideoMAE)
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
# Video loading
# ---------------------------------------------------------------------------
def load_video_frames(video_path, num_frames=NUM_FRAMES, size=IMAGE_SIZE, augment=False):
    """Load video with OpenCV, sample num_frames uniformly, resize to size x size.

    Returns: numpy array of shape (num_frames, 3, H, W) normalized to [0,1] then
             with ImageNet mean/std normalization, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{ts()}] WARNING: Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Try reading all frames
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

    # Compute uniform sample indices
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat last frame if video is too short
        indices = np.arange(total_frames)
        indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')

    if frames is not None:
        # Already read all frames
        sampled = [frames[i] for i in indices]
    else:
        # Re-open and read specific frames
        cap = cv2.VideoCapture(video_path)
        sampled = []
        frame_idx = 0
        target_set = set(indices)
        # Build a mapping from index to count (for repeated indices)
        index_list = list(indices)
        read_frames = {}
        needed = set(index_list)
        while frame_idx <= max(needed):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in needed:
                read_frames[frame_idx] = frame
            frame_idx += 1
        cap.release()
        sampled = [read_frames.get(i, read_frames.get(max(read_frames.keys()), None))
                    for i in index_list]
        if any(f is None for f in sampled):
            print(f"[{ts()}] WARNING: Could not read enough frames from: {video_path}")
            return None

    # Process frames: resize, convert BGR->RGB, normalize
    processed = []
    for frame in sampled:
        # Augmentation: random horizontal flip and random crop
        if augment:
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1)
            # Random crop: resize to slightly larger, then random crop
            h, w = frame.shape[:2]
            crop_size = int(size * 1.15)
            frame = cv2.resize(frame, (crop_size, crop_size))
            y_off = random.randint(0, crop_size - size)
            x_off = random.randint(0, crop_size - size)
            frame = frame[y_off:y_off + size, x_off:x_off + size]
        else:
            # Center crop: resize to slightly larger, then center crop
            crop_size = int(size * 1.15)
            frame = cv2.resize(frame, (crop_size, crop_size))
            off = (crop_size - size) // 2
            frame = frame[off:off + size, off:off + size]

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # ImageNet normalization
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW
        frame = frame.transpose(2, 0, 1)
        processed.append(frame)

    return np.stack(processed)  # (num_frames, 3, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KSLVideoDataset(Dataset):
    """KSL video dataset for VideoMAE."""

    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        frames = load_video_frames(path, augment=self.augment)
        if frames is None:
            # Return zeros as fallback (will be a bad sample)
            frames = np.zeros((NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def discover_training_data(classes):
    """Discover training videos from dataset/{ClassName}/{ClassName}-{SignerID}-{RepID}.mov"""
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
    """Discover validation videos from validation-data/{X}. Signer/.../*.mov"""
    paths, labels = [], []
    c2i = {c: i for i, c in enumerate(classes)}

    # Numbers have dirs like "No. 100", words have dirs like "1. Friend"
    is_numbers = all(c.isdigit() for c in classes)

    for signer_dir in sorted(glob_module.glob(os.path.join(VAL_DIR, "*/"))):
        for class_dir in sorted(glob_module.glob(os.path.join(signer_dir, "*/"))):
            dir_name = os.path.basename(class_dir.rstrip("/"))
            # Extract class name from directory name
            # Numbers: "No. 100" -> "100"
            # Words: "1. Friend" -> "Friend", "3. Colour" -> "Colour"
            cls_name = None
            if dir_name.startswith("No. "):
                cls_name = dir_name[4:].strip()
            else:
                # Format: "{num}. {ClassName}" or just "{ClassName}"
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
# Training
# ---------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")

    # Select classes
    if args.model_type == "numbers":
        classes = NUMBER_CLASSES
    else:
        classes = WORD_CLASSES
    num_classes = len(classes)
    print(f"[{ts()}] Model type: {args.model_type}, Classes: {num_classes}")
    print(f"[{ts()}] Classes: {classes}")

    # Discover data
    train_paths, train_labels = discover_training_data(classes)
    val_paths, val_labels = discover_validation_data(classes)

    if len(train_paths) == 0:
        print(f"[{ts()}] ERROR: No training videos found!")
        return

    # Datasets
    train_ds = KSLVideoDataset(train_paths, train_labels, augment=True)
    val_ds = KSLVideoDataset(val_paths, val_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Load model
    print(f"[{ts()}] Loading VideoMAE from: {MODEL_NAME}")
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{ts()}] Total params: {param_count:,}, Trainable: {trainable_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Warmup + cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Checkpoint dir
    ckpt_dir = os.path.join(CKPT_DIR, args.model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)  # (B, T, C, H, W)
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

        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{ts()}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
              f"LR={current_lr:.2e} | Time={epoch_time:.1f}s")

        # Save best
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
            }, save_path)
            print(f"[{ts()}]   -> New best! Saved to {save_path}")

    print(f"\n[{ts()}] Training complete!")
    print(f"[{ts()}] Best val accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")

    # Save final model too
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
    }, final_path)
    print(f"[{ts()}] Final model saved to {final_path}")


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on a dataloader."""
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE for KSL")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["numbers", "words"],
                        help="Train on numbers or words")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"[{ts()}] === KSL VideoMAE Training ===")
    print(f"[{ts()}] Args: {vars(args)}")
    train(args)


if __name__ == "__main__":
    main()
