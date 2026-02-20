#!/usr/bin/env python3
"""
Data Cleaning Pipeline for KSL.

Cleans the raw .npy landmark files by:
  1. Detecting frames where hands are undetected (L2 norm < threshold)
  2. Interpolating hand landmarks from nearest valid (detected) frames
  3. DBSCAN-based spatial outlier detection and interpolation per joint
  4. Flagging samples where >70% of frames have no hand detection
  5. Saving cleaned data to data/train_alpha_clean/ and data/val_alpha_clean/

Raw .npy format: (T, 549)
  cols 0:99   = 33 pose landmarks (xyz)
  cols 99:162  = left hand 21 joints (xyz)
  cols 162:225 = right hand 21 joints (xyz)
  cols 225:549 = face/world coords (preserved but not cleaned)

Usage:
    python clean_data.py
    python clean_data.py --bad-frame-threshold 0.70 --dbscan-eps 0.15
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

POSE_INDICES = [11, 12, 13, 14, 15, 16]

# Column ranges in the raw 549-col format
LH_START, LH_END = 99, 162   # 21 joints * 3
RH_START, RH_END = 162, 225  # 21 joints * 3


def extract_signer_id(filename):
    parts = os.path.splitext(filename)[0].split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def detect_bad_frames(raw, threshold=0.01):
    """Detect frames where both hands are undetected.

    Args:
        raw: (T, 549) raw landmark array
        threshold: L2 norm below which a hand is considered undetected

    Returns:
        lh_bad: (T,) bool — True where left hand is undetected
        rh_bad: (T,) bool — True where right hand is undetected
        both_bad: (T,) bool — True where BOTH hands are undetected
    """
    T = raw.shape[0]
    lh = raw[:, LH_START:LH_END].reshape(T, 21, 3)
    rh = raw[:, RH_START:RH_END].reshape(T, 21, 3)

    lh_norms = np.linalg.norm(lh, axis=(1, 2))  # (T,)
    rh_norms = np.linalg.norm(rh, axis=(1, 2))  # (T,)

    lh_bad = lh_norms < threshold
    rh_bad = rh_norms < threshold
    both_bad = lh_bad & rh_bad

    return lh_bad, rh_bad, both_bad


def interpolate_hand_frames(raw, bad_mask, col_start, col_end):
    """Interpolate missing hand frames from nearest valid neighbors.

    For each bad frame, find the nearest valid frame before and after,
    then linearly interpolate. If only one neighbor exists, copy it.
    If no valid frames exist, leave as zeros.

    Args:
        raw: (T, 549) array (modified in-place)
        bad_mask: (T,) bool — True for frames to interpolate
        col_start, col_end: column range for the hand
    """
    T = raw.shape[0]
    if not bad_mask.any():
        return

    valid_indices = np.where(~bad_mask)[0]
    if len(valid_indices) == 0:
        return  # No valid frames to interpolate from

    for t in np.where(bad_mask)[0]:
        # Find nearest valid frame before and after
        before = valid_indices[valid_indices < t]
        after = valid_indices[valid_indices > t]

        if len(before) > 0 and len(after) > 0:
            t_before = before[-1]
            t_after = after[0]
            # Linear interpolation
            alpha = (t - t_before) / (t_after - t_before)
            raw[t, col_start:col_end] = (
                (1 - alpha) * raw[t_before, col_start:col_end]
                + alpha * raw[t_after, col_start:col_end]
            )
        elif len(before) > 0:
            raw[t, col_start:col_end] = raw[before[-1], col_start:col_end]
        elif len(after) > 0:
            raw[t, col_start:col_end] = raw[after[0], col_start:col_end]
        # else: leave as zeros (no valid frames at all)


def interpolate_pose_frames(raw, both_bad_mask):
    """Interpolate pose landmarks for frames where both hands are bad.

    Pose landmarks (cols related to POSE_INDICES) may also be unreliable
    when hands are completely undetected. Only interpolate if the pose
    data itself looks suspicious (near-zero).
    """
    T = raw.shape[0]
    for pi in POSE_INDICES:
        col_start = pi * 3
        col_end = col_start + 3
        pose_norms = np.linalg.norm(raw[:, col_start:col_end], axis=1)
        pose_bad = both_bad_mask & (pose_norms < 0.01)
        if pose_bad.any():
            interpolate_hand_frames(raw, pose_bad, col_start, col_end)


def dbscan_clean_joints(raw, eps=0.15, min_samples=3):
    """Use DBSCAN to detect and interpolate spatial outlier keypoints.

    For each joint across time, run DBSCAN on its (x,y,z) trajectory.
    Points labeled as noise (-1) are replaced by interpolation from neighbors.

    Only operates on the 48 joints used by the model:
      LH: 21 joints from cols 99:162
      RH: 21 joints from cols 162:225
      Pose: 6 joints from POSE_INDICES

    Args:
        raw: (T, 549) array (modified in-place)
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        total_outliers: total number of outlier frames fixed
    """
    from sklearn.cluster import DBSCAN

    T = raw.shape[0]
    total_outliers = 0

    # Define joint column ranges: (col_start, col_end) for each joint
    joint_cols = []
    # Left hand: 21 joints
    for j in range(21):
        c = LH_START + j * 3
        joint_cols.append((c, c + 3))
    # Right hand: 21 joints
    for j in range(21):
        c = RH_START + j * 3
        joint_cols.append((c, c + 3))
    # Pose: 6 joints
    for pi in POSE_INDICES:
        c = pi * 3
        joint_cols.append((c, c + 3))

    for col_start, col_end in joint_cols:
        coords = raw[:, col_start:col_end]  # (T, 3)

        # Skip joints that are mostly zero (undetected)
        active_mask = np.linalg.norm(coords, axis=1) > 0.01
        if active_mask.sum() < max(5, min_samples + 1):
            continue

        active_coords = coords[active_mask]

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_coords)
        labels = clustering.labels_

        noise_in_active = labels == -1
        if not noise_in_active.any():
            continue

        # Map back to full time indices
        active_indices = np.where(active_mask)[0]
        noise_indices = active_indices[noise_in_active]

        # Create outlier mask for this joint
        outlier_mask = np.zeros(T, dtype=bool)
        outlier_mask[noise_indices] = True
        total_outliers += outlier_mask.sum()

        # Interpolate outlier frames from valid neighbors
        valid_indices = np.where(active_mask & ~outlier_mask)[0]
        if len(valid_indices) == 0:
            continue

        for t in noise_indices:
            before = valid_indices[valid_indices < t]
            after = valid_indices[valid_indices > t]

            if len(before) > 0 and len(after) > 0:
                t_before = before[-1]
                t_after = after[0]
                alpha = (t - t_before) / (t_after - t_before)
                raw[t, col_start:col_end] = (
                    (1 - alpha) * raw[t_before, col_start:col_end]
                    + alpha * raw[t_after, col_start:col_end]
                )
            elif len(before) > 0:
                raw[t, col_start:col_end] = raw[before[-1], col_start:col_end]
            elif len(after) > 0:
                raw[t, col_start:col_end] = raw[after[0], col_start:col_end]

    return total_outliers


def clean_sample(raw, dbscan_eps=0.15, dbscan_min_samples=3):
    """Clean a single sample. Returns (cleaned_raw, stats_dict).

    Steps:
      1. Detect bad frames (both hands undetected)
      2. Interpolate left hand from valid neighbors
      3. Interpolate right hand from valid neighbors
      4. Interpolate pose for frames where both hands were bad
      5. DBSCAN outlier detection and interpolation per joint
    """
    raw = raw.copy()  # Don't modify original
    T = raw.shape[0]

    lh_bad, rh_bad, both_bad = detect_bad_frames(raw)

    stats = {
        "num_frames": T,
        "lh_bad_frames": int(lh_bad.sum()),
        "rh_bad_frames": int(rh_bad.sum()),
        "both_bad_frames": int(both_bad.sum()),
        "both_bad_rate": float(both_bad.mean()),
    }

    # Step 1-2: Interpolate left hand
    interpolate_hand_frames(raw, lh_bad, LH_START, LH_END)

    # Step 3: Interpolate right hand
    interpolate_hand_frames(raw, rh_bad, RH_START, RH_END)

    # Step 4: Interpolate pose where both hands were missing
    interpolate_pose_frames(raw, both_bad)

    # Step 5: DBSCAN outlier cleaning
    dbscan_outliers = dbscan_clean_joints(raw, eps=dbscan_eps, min_samples=dbscan_min_samples)
    stats["dbscan_outliers_fixed"] = int(dbscan_outliers)

    # Verify improvement
    lh_bad_after, rh_bad_after, both_bad_after = detect_bad_frames(raw)
    stats["both_bad_frames_after"] = int(both_bad_after.sum())
    stats["both_bad_rate_after"] = float(both_bad_after.mean())

    return raw, stats


def process_directory(input_dir, output_dir, bad_frame_threshold=0.70,
                      dbscan_eps=0.15, dbscan_min_samples=3):
    """Process all .npy files in a directory structure.

    Args:
        input_dir: path like data/train_alpha/
        output_dir: path like data/train_alpha_clean/
        bad_frame_threshold: flag samples where >threshold frames have both hands missing
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples

    Returns:
        stats: dict of statistics
    """
    total_samples = 0
    flagged_samples = 0
    total_interpolated_frames = 0
    total_dbscan_fixes = 0
    total_frames = 0
    per_signer_stats = defaultdict(lambda: {
        "samples": 0, "flagged": 0, "bad_frames": 0, "total_frames": 0,
        "dbscan_fixes": 0,
    })
    flagged_files = []

    for cls in sorted(os.listdir(input_dir)):
        cls_in = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_in):
            continue

        cls_out = os.path.join(output_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        for fn in sorted(os.listdir(cls_in)):
            if not fn.endswith(".npy"):
                continue

            filepath = os.path.join(cls_in, fn)
            raw = np.load(filepath).astype(np.float32)

            if raw.ndim != 2 or raw.shape[1] < 225:
                # Just copy as-is
                np.save(os.path.join(cls_out, fn), raw)
                total_samples += 1
                continue

            signer = extract_signer_id(fn)
            cleaned, stats = clean_sample(raw, dbscan_eps, dbscan_min_samples)

            total_samples += 1
            total_frames += stats["num_frames"]
            total_interpolated_frames += stats["both_bad_frames"]
            total_dbscan_fixes += stats["dbscan_outliers_fixed"]

            per_signer_stats[signer]["samples"] += 1
            per_signer_stats[signer]["total_frames"] += stats["num_frames"]
            per_signer_stats[signer]["bad_frames"] += stats["both_bad_frames"]
            per_signer_stats[signer]["dbscan_fixes"] += stats["dbscan_outliers_fixed"]

            is_flagged = stats["both_bad_rate"] > bad_frame_threshold
            if is_flagged:
                flagged_samples += 1
                per_signer_stats[signer]["flagged"] += 1
                flagged_files.append({
                    "file": fn,
                    "class": cls,
                    "signer": signer,
                    "both_bad_rate": stats["both_bad_rate"],
                    "both_bad_frames": stats["both_bad_frames"],
                    "num_frames": stats["num_frames"],
                })

            np.save(os.path.join(cls_out, fn), cleaned)

        if total_samples % 200 == 0 and total_samples > 0:
            print(f"  Processed {total_samples} samples...")

    return {
        "total_samples": total_samples,
        "flagged_samples": flagged_samples,
        "total_frames": total_frames,
        "total_interpolated_frames": total_interpolated_frames,
        "total_dbscan_fixes": total_dbscan_fixes,
        "per_signer": dict(per_signer_stats),
        "flagged_files": flagged_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="KSL Data Cleaning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bad-frame-threshold", type=float, default=0.70,
                        help="Flag samples where >threshold of frames have both hands missing")
    parser.add_argument("--dbscan-eps", type=float, default=0.15,
                        help="DBSCAN eps for spatial outlier detection")
    parser.add_argument("--dbscan-min-samples", type=int, default=3,
                        help="DBSCAN min_samples parameter")
    parser.add_argument("--output-report", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/results/cleaning_report.json",
                        help="Path to save cleaning report")
    args = parser.parse_args()

    base = "/scratch/alpine/hama5612/ksl-dir-2/data"
    dirs_to_clean = [
        (os.path.join(base, "train_alpha"), os.path.join(base, "train_alpha_clean")),
        (os.path.join(base, "val_alpha"), os.path.join(base, "val_alpha_clean")),
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("KSL Data Cleaning Pipeline")
    print(f"Started: {ts}")
    print(f"Bad frame threshold: {args.bad_frame_threshold}")
    print(f"DBSCAN eps: {args.dbscan_eps}, min_samples: {args.dbscan_min_samples}")
    print("=" * 70)

    all_stats = {}

    for input_dir, output_dir in dirs_to_clean:
        if not os.path.isdir(input_dir):
            print(f"\n  SKIP: {input_dir} not found")
            continue

        dir_name = os.path.basename(input_dir)
        print(f"\n{'='*70}")
        print(f"Processing: {dir_name}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}")

        os.makedirs(output_dir, exist_ok=True)

        stats = process_directory(
            input_dir, output_dir,
            bad_frame_threshold=args.bad_frame_threshold,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )

        all_stats[dir_name] = stats

        print(f"\n  Results for {dir_name}:")
        print(f"    Total samples: {stats['total_samples']}")
        print(f"    Flagged (>{100*args.bad_frame_threshold:.0f}% bad): "
              f"{stats['flagged_samples']} samples")
        print(f"    Interpolated frames (both hands missing): "
              f"{stats['total_interpolated_frames']}/{stats['total_frames']} "
              f"({100*stats['total_interpolated_frames']/max(stats['total_frames'],1):.2f}%)")
        print(f"    DBSCAN fixes: {stats['total_dbscan_fixes']} outlier joint-frames")

        print(f"\n    Per-signer summary:")
        print(f"    {'Signer':>8s}  {'Samples':>8s}  {'Flagged':>8s}  {'Bad%':>7s}  {'DBSCAN':>7s}")
        for signer in sorted(stats["per_signer"].keys()):
            ss = stats["per_signer"][signer]
            bad_rate = 100 * ss["bad_frames"] / max(ss["total_frames"], 1)
            print(f"    {signer:>8s}  {ss['samples']:8d}  {ss['flagged']:8d}  "
                  f"{bad_rate:6.2f}%  {ss['dbscan_fixes']:7d}")

        if stats["flagged_files"]:
            print(f"\n    Flagged files ({len(stats['flagged_files'])}):")
            for ff in stats["flagged_files"][:20]:
                print(f"      {ff['class']}/{ff['file']}: "
                      f"{ff['both_bad_frames']}/{ff['num_frames']} bad "
                      f"({100*ff['both_bad_rate']:.0f}%)")
            if len(stats["flagged_files"]) > 20:
                print(f"      ... and {len(stats['flagged_files'])-20} more")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")

    total_samples = sum(s["total_samples"] for s in all_stats.values())
    total_flagged = sum(s["flagged_samples"] for s in all_stats.values())
    total_interp = sum(s["total_interpolated_frames"] for s in all_stats.values())
    total_frames = sum(s["total_frames"] for s in all_stats.values())
    total_dbscan = sum(s["total_dbscan_fixes"] for s in all_stats.values())

    print(f"  Total samples processed: {total_samples}")
    print(f"  Total flagged (>{100*args.bad_frame_threshold:.0f}% bad): {total_flagged}")
    print(f"  Total frames interpolated: {total_interp}/{total_frames} "
          f"({100*total_interp/max(total_frames,1):.2f}%)")
    print(f"  Total DBSCAN fixes: {total_dbscan}")
    print(f"\n  Output directories:")
    for _, output_dir in dirs_to_clean:
        exists = os.path.isdir(output_dir)
        count = sum(1 for c in os.listdir(output_dir)
                    for f in os.listdir(os.path.join(output_dir, c))
                    if f.endswith(".npy")) if exists else 0
        print(f"    {output_dir}: {count} files")

    # Save report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report = {
        "timestamp": ts,
        "config": {
            "bad_frame_threshold": args.bad_frame_threshold,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
        },
        "summary": {
            "total_samples": total_samples,
            "total_flagged": total_flagged,
            "total_interpolated_frames": total_interp,
            "total_frames": total_frames,
            "interpolation_rate": float(total_interp / max(total_frames, 1)),
            "total_dbscan_fixes": total_dbscan,
        },
        "per_directory": all_stats,
    }

    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nCleaning report saved to {args.output_report}")


if __name__ == "__main__":
    main()
