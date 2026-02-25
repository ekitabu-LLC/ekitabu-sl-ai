#!/usr/bin/env python3
"""
KSL VideoMAE v3 Training — VideoMAE-Base + ISLR augmentation pipeline.

Based on v2 (all 10 signers) with video augmentations from arXiv 2412.11553
(Training Strategies for Isolated Sign Language Recognition):

Video augmentations (applied to raw frame indices before sampling):
  - Speed up 2x (p=0.25): sample every 2nd frame
  - Slow down 2x (p=0.25): repeat each frame twice
  - Random frame add 30% (p=0.25): randomly duplicate frames to extend length
  - Random frame drop 10% (p=0.5): randomly drop frames

Image augmentations (applied per-frame after crop):
  - Random horizontal flip (p=0.5)
  - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  - Random erasing (p=0.25)

Batch-level augmentations:
  - CutMix (alpha=1.0, p=0.5)
  - MixUp (alpha=0.8, p=0.5)
  (One of CutMix/MixUp applied per batch, not both)

Data: 10 signers, signer 5 held out for validation.
Checkpoints saved to data/checkpoints/videomae_v3/

Usage:
    python train_ksl_videomae_v3.py --model-type numbers
    python train_ksl_videomae_v3.py --model-type words
"""

import argparse
import os
import random
import re
import time
import glob as glob_module
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
CKPT_DIR = os.path.join(BASE_DIR, "data", "checkpoints", "videomae_v3")

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
IMAGE_SIZE = 224

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
# ISLR Video Augmentations (applied to frame indices)
# ---------------------------------------------------------------------------
def augment_frame_indices(total_frames, num_frames=NUM_FRAMES):
    """Apply ISLR temporal augmentations to frame indices.

    Pipeline (applied sequentially with independent probabilities):
    1. Speed up 2x (p=0.25): select every 2nd frame
    2. Slow down 2x (p=0.25): repeat each frame twice
    3. Random add 30% (p=0.25): randomly duplicate frames
    4. Random drop 10% (p=0.5): randomly remove frames

    After all transforms, uniformly sample num_frames from the result.
    """
    if total_frames == 0:
        return np.zeros(num_frames, dtype=int)

    indices = np.arange(total_frames)

    # 1. Speed up 2x: take every 2nd frame (halves the effective length)
    if random.random() < 0.25 and len(indices) > num_frames:
        indices = indices[::2]

    # 2. Slow down 2x: repeat each frame (doubles the effective length)
    elif random.random() < 0.25:
        indices = np.repeat(indices, 2)

    # 3. Random add 30%: randomly duplicate frames to extend by 30%
    if random.random() < 0.25 and len(indices) > 0:
        n_add = max(1, int(len(indices) * 0.3))
        add_indices = np.random.choice(indices, size=n_add)
        indices = np.sort(np.concatenate([indices, add_indices]))

    # 4. Random drop 10%: randomly remove 10% of frames
    if random.random() < 0.5 and len(indices) > num_frames:
        n_keep = max(num_frames, int(len(indices) * 0.9))
        keep_mask = np.sort(np.random.choice(len(indices), size=n_keep, replace=False))
        indices = indices[keep_mask]

    # Final uniform sampling to num_frames
    n = len(indices)
    if n >= num_frames:
        sample_idx = np.linspace(0, n - 1, num_frames, dtype=int)
        indices = indices[sample_idx]
    else:
        indices = np.pad(indices, (0, num_frames - n), mode='edge')

    return indices


# ---------------------------------------------------------------------------
# ISLR Image Augmentations (applied per-frame)
# ---------------------------------------------------------------------------
def augment_frame_image(frame):
    """Apply ISLR image augmentations to a single frame (uint8 BGR).

    - ColorJitter (brightness, contrast, saturation, hue)
    - Returns augmented frame (uint8 BGR)
    """
    # ColorJitter: brightness
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-0.2, 0.2)
        frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # ColorJitter: contrast
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-0.2, 0.2)
        mean = frame.mean()
        frame = np.clip((frame.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    # ColorJitter: saturation (convert to HSV)
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-0.2, 0.2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ColorJitter: hue
    if random.random() < 0.5:
        shift = random.uniform(-0.1, 0.1) * 180
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return frame


def random_erasing(frame_chw, p=0.25, sl=0.02, sh=0.4, r1=0.3):
    """Random erasing on a normalized CHW float32 frame.

    Replaces a random rectangle with random values.
    """
    if random.random() > p:
        return frame_chw

    c, h, w = frame_chw.shape
    area = h * w

    for _ in range(10):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1.0 / r1)
        rh = int(round(np.sqrt(target_area * aspect_ratio)))
        rw = int(round(np.sqrt(target_area / aspect_ratio)))
        if rh < h and rw < w:
            y = random.randint(0, h - rh)
            x = random.randint(0, w - rw)
            frame_chw[:, y:y + rh, x:x + rw] = np.random.randn(c, rh, rw).astype(np.float32)
            break

    return frame_chw


# ---------------------------------------------------------------------------
# CutMix and MixUp (batch-level)
# ---------------------------------------------------------------------------
def cutmix_batch(frames, labels, num_classes, alpha=1.0):
    """Apply CutMix to a batch of video frames.

    frames: (B, T, C, H, W)
    labels: (B,) long tensor
    Returns: mixed frames, soft label targets (B, num_classes)
    """
    batch_size = frames.size(0)
    lam = np.random.beta(alpha, alpha)

    # Random permutation for mixing partners
    rand_index = torch.randperm(batch_size, device=frames.device)

    # Generate random box
    _, T, C, H, W = frames.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = random.randint(0, H)
    cx = random.randint(0, W)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    # Apply cut
    mixed = frames.clone()
    mixed[:, :, :, y1:y2, x1:x2] = frames[rand_index, :, :, y1:y2, x1:x2]

    # Adjust lambda for actual box area
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    # Soft labels
    targets_a = F.one_hot(labels, num_classes).float()
    targets_b = F.one_hot(labels[rand_index], num_classes).float()
    targets = lam * targets_a + (1.0 - lam) * targets_b

    return mixed, targets


def mixup_batch(frames, labels, num_classes, alpha=0.8):
    """Apply MixUp to a batch of video frames.

    frames: (B, T, C, H, W)
    labels: (B,) long tensor
    Returns: mixed frames, soft label targets (B, num_classes)
    """
    lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(frames.size(0), device=frames.device)

    mixed = lam * frames + (1.0 - lam) * frames[rand_index]

    targets_a = F.one_hot(labels, num_classes).float()
    targets_b = F.one_hot(labels[rand_index], num_classes).float()
    targets = lam * targets_a + (1.0 - lam) * targets_b

    return mixed, targets


# ---------------------------------------------------------------------------
# Video loading with ISLR augmentations
# ---------------------------------------------------------------------------
def load_video_frames(video_path, num_frames=NUM_FRAMES, size=IMAGE_SIZE, augment=False):
    """Load video with ISLR augmentations when augment=True."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{ts()}] WARNING: Cannot open video: {video_path}")
        return None

    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        print(f"[{ts()}] WARNING: No frames in video: {video_path}")
        return None

    # Get frame indices (with temporal augmentation if training)
    if augment:
        indices = augment_frame_indices(total_frames, num_frames)
    else:
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')

    # Clamp indices to valid range
    indices = np.clip(indices, 0, total_frames - 1)
    sampled = [all_frames[i] for i in indices]

    # Decide spatial augmentation params once (consistent across all frames)
    if augment:
        do_flip = random.random() < 0.5
        crop_size = int(size * 1.15)
        y_off = random.randint(0, crop_size - size)
        x_off = random.randint(0, crop_size - size)
    else:
        do_flip = False

    processed = []
    for frame in sampled:
        # Image-level augmentation (before crop, on raw BGR)
        if augment:
            frame = augment_frame_image(frame)

        if augment:
            if do_flip:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (crop_size, crop_size))
            frame = frame[y_off:y_off + size, x_off:x_off + size]
        else:
            crop_size = int(size * 1.15)
            frame = cv2.resize(frame, (crop_size, crop_size))
            off = (crop_size - size) // 2
            frame = frame[off:off + size, off:off + size]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW

        # Random erasing (on normalized frame)
        if augment:
            frame = random_erasing(frame, p=0.25)

        processed.append(frame)

    return np.stack(processed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KSLVideoDataset(Dataset):
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
            frames = np.zeros((NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data discovery (same as v2 — all 10 signers)
# ---------------------------------------------------------------------------
def discover_all_data(classes):
    """Discover videos from dataset/ (signers 1-5) and validation-data/ (signers 6-10)."""
    c2i = {c: i for i, c in enumerate(classes)}
    all_videos = []

    # Source 1: dataset/{ClassName}/{ClassName}-{SignerID}-{RepID}.mov
    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not (fname.lower().endswith(".mov") or fname.lower().endswith(".mp4")):
                continue
            parts = fname.rsplit(".", 1)[0].split("-")
            signer_id = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else -1
            all_videos.append((os.path.join(cls_dir, fname), c2i[cls_name], signer_id))

    # Source 2: validation-data/{X}. Signer/{N}. {ClassName}/{letter}. {ClassName}.mov
    for signer_dir_path in sorted(glob_module.glob(os.path.join(VAL_DIR, "*/"))):
        signer_dir_name = os.path.basename(signer_dir_path.rstrip("/"))
        m = re.match(r"(\d+)\.\s*Signer", signer_dir_name)
        if not m:
            continue
        signer_id = int(m.group(1))

        for class_dir_path in sorted(glob_module.glob(os.path.join(signer_dir_path, "*/"))):
            dir_name = os.path.basename(class_dir_path.rstrip("/"))
            if dir_name.startswith("No. "):
                cls_name = dir_name[4:].strip()
            else:
                parts = dir_name.split(". ", 1)
                cls_name = parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else dir_name.strip()

            if cls_name not in c2i:
                continue

            for fname in sorted(os.listdir(class_dir_path)):
                if fname.lower().endswith(".mov") or fname.lower().endswith(".mp4"):
                    all_videos.append((os.path.join(class_dir_path, fname), c2i[cls_name], signer_id))

    signer_counts = {}
    for _, _, sid in all_videos:
        signer_counts[sid] = signer_counts.get(sid, 0) + 1
    print(f"[{ts()}] All data: {len(all_videos)} videos across {len(classes)} classes")
    print(f"[{ts()}] Per-signer counts: {dict(sorted(signer_counts.items()))}")

    return all_videos


def split_by_signer(all_videos, val_signer_id):
    """Split videos into train/val by signer ID."""
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for path, label, signer_id in all_videos:
        if signer_id == val_signer_id:
            val_paths.append(path)
            val_labels.append(label)
        else:
            train_paths.append(path)
            train_labels.append(label)

    print(f"[{ts()}] Val signer={val_signer_id}: train={len(train_paths)}, val={len(val_paths)}")
    return train_paths, train_labels, val_paths, val_labels


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
    print(f"[{ts()}] Classes: {classes}")

    # Discover all 10 signers
    all_videos = discover_all_data(classes)
    if not all_videos:
        print(f"[{ts()}] ERROR: No videos found!")
        return

    train_paths, train_labels, val_paths, val_labels = split_by_signer(
        all_videos, args.val_signer)

    if not train_paths:
        print(f"[{ts()}] ERROR: No training videos after split!")
        return

    # Datasets
    train_ds = KSLVideoDataset(train_paths, train_labels, augment=True)
    val_ds = KSLVideoDataset(val_paths, val_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
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

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Use soft CE for CutMix/MixUp compatibility
    # label_smoothing applied via soft targets when no mix, via mixing otherwise
    base_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Checkpoint dir
    ckpt_dir = os.path.join(CKPT_DIR, args.model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Print augmentation config
    print(f"\n[{ts()}] === ISLR Augmentation Config ===")
    print(f"  Temporal: speed_up_2x(p=0.25), slow_down_2x(p=0.25), "
          f"random_add_30%(p=0.25), random_drop_10%(p=0.5)")
    print(f"  Image: color_jitter, random_erasing(p=0.25), random_crop, hflip(p=0.5)")
    print(f"  Batch: CutMix(alpha=1.0) / MixUp(alpha=0.8), p=0.5 each")
    print(f"  CutMix/MixUp: {'enabled' if args.use_mixup else 'disabled'}")
    print()

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

            # Batch-level augmentation: CutMix or MixUp
            use_soft = False
            if args.use_mixup and random.random() < 0.5:
                if random.random() < 0.5:
                    frames, soft_targets = cutmix_batch(frames, labels, num_classes, alpha=1.0)
                else:
                    frames, soft_targets = mixup_batch(frames, labels, num_classes, alpha=0.8)
                use_soft = True

            outputs = model(pixel_values=frames)
            logits = outputs.logits

            if use_soft:
                # Soft cross-entropy: -sum(target * log_softmax(logits))
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(soft_targets * log_probs).sum(dim=1).mean()
            else:
                loss = base_criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            if use_soft:
                correct += (preds == soft_targets.argmax(dim=1)).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        epoch_time = time.time() - t0

        # Validation (no augmentation)
        val_acc, val_loss = evaluate_model(model, val_loader, base_criterion, device)

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
                "val_signer": args.val_signer,
                "num_train_signers": 9,
                "augmentation": "islr_v3",
                "args": vars(args),
            }, save_path)
            print(f"[{ts()}]   -> New best! Saved to {save_path}")

    print(f"\n[{ts()}] Training complete!")
    print(f"[{ts()}] Best val accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")

    # Save final model
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "classes": classes,
        "model_name": MODEL_NAME,
        "val_signer": args.val_signer,
        "num_train_signers": 9,
        "augmentation": "islr_v3",
        "args": vars(args),
    }, final_path)
    print(f"[{ts()}] Final model saved to {final_path}")


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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE v3 (ISLR augmentation)")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["numbers", "words"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-signer", type=int, default=5)
    parser.add_argument("--use-mixup", action="store_true", default=True,
                        help="Enable CutMix/MixUp batch augmentation (default: True)")
    parser.add_argument("--no-mixup", action="store_false", dest="use_mixup",
                        help="Disable CutMix/MixUp batch augmentation")
    args = parser.parse_args()

    print(f"[{ts()}] === KSL VideoMAE v3 Training (ISLR Augmentation) ===")
    print(f"[{ts()}] Args: {vars(args)}")
    train(args)


if __name__ == "__main__":
    main()
