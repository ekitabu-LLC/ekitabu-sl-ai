#!/usr/bin/env python3
"""
VideoMAE Real-Tester Evaluation.

Loads saved VideoMAE checkpoints and evaluates on the 3 unseen real test signers.
Uses the real_testers_metadata.csv for ground truth labels and video paths.

Usage:
    python evaluate_videomae.py --category both
    python evaluate_videomae.py --category numbers
    python evaluate_videomae.py --category words
    python evaluate_videomae.py --category both --videomae-ckpt-dir data/checkpoints/videomae_v2
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
REAL_TESTERS_DIR = "/scratch/alpine/hama5612/ksl-alpha/data/real_testers"
METADATA_CSV = os.path.join(REAL_TESTERS_DIR, "real_testers_metadata.csv")
CKPT_DIR = os.path.join(BASE_DIR, "data", "checkpoints", "videomae")
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

NUM_FRAMES = 16
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Video loading (same as training, no augmentation)
# ---------------------------------------------------------------------------
def load_video_frames(video_path, num_frames=NUM_FRAMES, size=IMAGE_SIZE):
    """Load video, sample num_frames uniformly, center crop to size x size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{ts()}] WARNING: Cannot open video: {video_path}")
        return None

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

    # Uniform sampling
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')

    sampled = [frames[i] for i in indices]

    processed = []
    for frame in sampled:
        # Center crop
        crop_size = int(size * 1.15)
        frame = cv2.resize(frame, (crop_size, crop_size))
        off = (crop_size - size) // 2
        frame = frame[off:off + size, off:off + size]
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        processed.append(frame)

    return np.stack(processed)  # (T, C, H, W)


# ---------------------------------------------------------------------------
# Load real tester metadata
# ---------------------------------------------------------------------------
def load_real_testers(category="both"):
    """Parse real_testers_metadata.csv, return list of (video_path, class_name, signer)."""
    number_set = set(f"No_{c}" for c in NUMBER_CLASSES)
    word_set = set(WORD_CLASSES)

    entries = []
    with open(METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row["class_name"]
            signer = row["category"]  # "2. Signer" etc.
            video_rel = row["video_path"]
            video_path = os.path.join(REAL_TESTERS_DIR, "..",
                                       os.path.basename(os.path.dirname(video_rel)),
                                       *video_rel.split("/")[1:])
            # Resolve: the CSV paths are relative to ksl-alpha/data/
            video_path = os.path.join("/scratch/alpine/hama5612/ksl-alpha",
                                       video_rel)

            is_number = cls_name in number_set
            is_word = cls_name in word_set

            if category == "numbers" and not is_number:
                continue
            if category == "words" and not is_word:
                continue
            if not is_number and not is_word:
                continue

            entries.append({
                "video_path": video_path,
                "class_name": cls_name,
                "signer": signer,
                "is_number": is_number,
            })

    print(f"[{ts()}] Loaded {len(entries)} real tester entries (category={category})")
    return entries


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_model(model_type, device, ckpt_dir=None):
    """Load saved VideoMAE checkpoint."""
    base = ckpt_dir if ckpt_dir is not None else CKPT_DIR
    ckpt_path = os.path.join(base, model_type, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{ts()}] ERROR: Checkpoint not found: {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[{ts()}] Loaded {model_type} model from epoch {ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.1f}%, classes={num_classes}")
    return model, classes


def predict(model, classes, video_path, device):
    """Predict class for a single video. Returns (predicted_class, confidence, logits)."""
    frames = load_video_frames(video_path)
    if frames is None:
        return None, 0.0, None

    frames_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=frames_t)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_class = classes[pred_idx]
    return pred_class, confidence, probs.numpy()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")
    print(f"[{ts()}] Checkpoint dir: {args.videomae_ckpt_dir}")

    # Load models
    numbers_model, numbers_classes = None, None
    words_model, words_classes = None, None

    if args.category in ("numbers", "both"):
        numbers_model, numbers_classes = load_model("numbers", device, args.videomae_ckpt_dir)
        if numbers_model is None:
            print(f"[{ts()}] WARNING: Numbers model not available")

    if args.category in ("words", "both"):
        words_model, words_classes = load_model("words", device, args.videomae_ckpt_dir)
        if words_model is None:
            print(f"[{ts()}] WARNING: Words model not available")

    # Load test entries
    entries = load_real_testers(args.category)
    if not entries:
        print(f"[{ts()}] ERROR: No test entries found")
        return

    # Evaluate
    results_by_signer = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_category = {"numbers": {"correct": 0, "total": 0},
                           "words": {"correct": 0, "total": 0}}

    all_preds = []
    for entry in entries:
        video_path = entry["video_path"]
        true_class = entry["class_name"]
        signer = entry["signer"]
        is_number = entry["is_number"]

        if not os.path.exists(video_path):
            print(f"[{ts()}] WARNING: Video not found: {video_path}")
            continue

        # Select model
        if is_number and numbers_model is not None:
            # Map "No_100" -> index in numbers_classes
            num_val = true_class.replace("No_", "")
            if num_val not in numbers_classes:
                continue
            pred_class, confidence, probs = predict(numbers_model, numbers_classes,
                                                     video_path, device)
            if pred_class is None:
                continue
            # Compare: pred_class is like "100", true is "No_100"
            is_correct = pred_class == num_val
            cat = "numbers"
        elif not is_number and words_model is not None:
            if true_class not in words_classes:
                continue
            pred_class, confidence, probs = predict(words_model, words_classes,
                                                     video_path, device)
            if pred_class is None:
                continue
            is_correct = pred_class == true_class
            cat = "words"
        else:
            continue

        results_by_signer[signer]["total"] += 1
        results_by_class[true_class]["total"] += 1
        results_by_category[cat]["total"] += 1
        if is_correct:
            results_by_signer[signer]["correct"] += 1
            results_by_class[true_class]["correct"] += 1
            results_by_category[cat]["correct"] += 1

        all_preds.append({
            "video": os.path.basename(video_path),
            "signer": signer,
            "true": true_class,
            "pred": pred_class if is_number else pred_class,
            "correct": is_correct,
            "confidence": confidence,
        })

    # Print results
    print(f"\n{'='*60}")
    print(f"VideoMAE Real-Tester Evaluation Results")
    print(f"{'='*60}")

    # Per-category
    for cat in ["numbers", "words"]:
        r = results_by_category[cat]
        if r["total"] > 0:
            acc = 100.0 * r["correct"] / r["total"]
            print(f"\n{cat.upper()}: {r['correct']}/{r['total']} = {acc:.1f}%")

    # Combined
    total_correct = sum(r["correct"] for r in results_by_category.values())
    total_samples = sum(r["total"] for r in results_by_category.values())
    if total_samples > 0:
        combined_acc = 100.0 * total_correct / total_samples
        print(f"\nCOMBINED: {total_correct}/{total_samples} = {combined_acc:.1f}%")

    # Per-signer
    print(f"\nPer-signer breakdown:")
    for signer in sorted(results_by_signer.keys()):
        r = results_by_signer[signer]
        acc = 100.0 * r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"  {signer}: {r['correct']}/{r['total']} = {acc:.1f}%")

    # Per-class (sorted by accuracy)
    print(f"\nPer-class breakdown:")
    class_accs = []
    for cls in sorted(results_by_class.keys()):
        r = results_by_class[cls]
        acc = 100.0 * r["correct"] / r["total"] if r["total"] > 0 else 0
        class_accs.append((cls, acc, r["correct"], r["total"]))
    class_accs.sort(key=lambda x: x[1])
    for cls, acc, correct, total in class_accs:
        print(f"  {cls:15s}: {correct}/{total} = {acc:.1f}%")

    # Save predictions
    ckpt_name = os.path.basename(args.videomae_ckpt_dir.rstrip("/"))
    results_dir = os.path.join(BASE_DIR, "data", "results", ckpt_name)
    os.makedirs(results_dir, exist_ok=True)
    import json
    results_path = os.path.join(results_dir, f"real_tester_results_{args.category}.json")
    with open(results_path, "w") as f:
        json.dump({
            "category": args.category,
            "combined_accuracy": combined_acc if total_samples > 0 else 0,
            "numbers_accuracy": (100.0 * results_by_category["numbers"]["correct"] /
                                 results_by_category["numbers"]["total"])
                                if results_by_category["numbers"]["total"] > 0 else 0,
            "words_accuracy": (100.0 * results_by_category["words"]["correct"] /
                               results_by_category["words"]["total"])
                              if results_by_category["words"]["total"] > 0 else 0,
            "per_signer": {s: {"correct": r["correct"], "total": r["total"],
                               "accuracy": 100.0 * r["correct"] / r["total"] if r["total"] > 0 else 0}
                           for s, r in results_by_signer.items()},
            "predictions": all_preds,
        }, f, indent=2)
    print(f"\n[{ts()}] Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VideoMAE on KSL real testers")
    parser.add_argument("--category", type=str, default="both",
                        choices=["numbers", "words", "both"])
    parser.add_argument("--videomae-ckpt-dir", type=str, default=CKPT_DIR,
                        help="VideoMAE checkpoint directory (default: data/checkpoints/videomae)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
