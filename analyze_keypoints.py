#!/usr/bin/env python3
"""
Data-Centric Keypoint Analysis for KSL.

Loads all training .npy files from data/train_alpha/ and data/val_alpha/
and computes per-keypoint statistics to identify:
  1. Outlier frames (DBSCAN-based + velocity-based detection)
  2. Per-keypoint variance across signers for the same class
  3. Zero/missing hand detection rates per signer
  4. Temporal statistics (sequence lengths, zero-frame ratios)
  5. High-variance keypoints that may be noise rather than signal

Outputs a comprehensive report to stdout and saves detailed stats to JSON.

Usage:
    python analyze_keypoints.py
    python analyze_keypoints.py --output data/results/keypoint_analysis.json
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

# Skeleton topology (from train_ksl_v37.py)
POSE_INDICES = [11, 12, 13, 14, 15, 16]

JOINT_NAMES = {
    # Left hand
    0: "LH_Wrist", 1: "LH_Thumb_CMC", 2: "LH_Thumb_MCP", 3: "LH_Thumb_IP",
    4: "LH_Thumb_Tip", 5: "LH_Index_MCP", 6: "LH_Index_PIP", 7: "LH_Index_DIP",
    8: "LH_Index_Tip", 9: "LH_Middle_MCP", 10: "LH_Middle_PIP", 11: "LH_Middle_DIP",
    12: "LH_Middle_Tip", 13: "LH_Ring_MCP", 14: "LH_Ring_PIP", 15: "LH_Ring_DIP",
    16: "LH_Ring_Tip", 17: "LH_Pinky_MCP", 18: "LH_Pinky_PIP", 19: "LH_Pinky_DIP",
    20: "LH_Pinky_Tip",
    # Right hand
    21: "RH_Wrist", 22: "RH_Thumb_CMC", 23: "RH_Thumb_MCP", 24: "RH_Thumb_IP",
    25: "RH_Thumb_Tip", 26: "RH_Index_MCP", 27: "RH_Index_PIP", 28: "RH_Index_DIP",
    29: "RH_Index_Tip", 30: "RH_Middle_MCP", 31: "RH_Middle_PIP", 32: "RH_Middle_DIP",
    33: "RH_Middle_Tip", 34: "RH_Ring_MCP", 35: "RH_Ring_PIP", 36: "RH_Ring_DIP",
    37: "RH_Ring_Tip", 38: "RH_Pinky_MCP", 39: "RH_Pinky_PIP", 40: "RH_Pinky_DIP",
    41: "RH_Pinky_Tip",
    # Pose (body)
    42: "L_Shoulder", 43: "R_Shoulder", 44: "L_Elbow", 45: "R_Elbow",
    46: "L_Wrist", 47: "R_Wrist",
}

JOINT_GROUPS = {
    "LH_fingers": list(range(1, 21)),
    "RH_fingers": list(range(22, 42)),
    "LH_wrist": [0],
    "RH_wrist": [21],
    "Body": list(range(42, 48)),
    "LH_tips": [4, 8, 12, 16, 20],
    "RH_tips": [25, 29, 33, 37, 41],
}


def extract_skeleton(raw):
    """Extract 48-joint skeleton (T, 48, 3) from raw .npy (T, 549)."""
    f = raw.shape[0]
    if raw.shape[1] < 225:
        return None

    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        pose[:, pi, :] = raw[:, idx_pose * 3:idx_pose * 3 + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)
    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)
    return h


def extract_signer_id(filename):
    parts = os.path.splitext(filename)[0].split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def compute_frame_velocity(skel):
    """Compute per-joint velocity (displacement between frames). Returns (T-1, 48)."""
    if skel.shape[0] < 2:
        return np.zeros((0, 48))
    diff = skel[1:] - skel[:-1]  # (T-1, 48, 3)
    return np.linalg.norm(diff, axis=-1)  # (T-1, 48)


def detect_outlier_frames_velocity(skel, threshold_multiplier=5.0):
    """Detect frames where any joint jumps by more than threshold_multiplier * median velocity.

    Returns:
        outlier_mask: (T,) bool — True for outlier frames
        per_joint_outliers: (48,) int — count of outlier frames per joint
    """
    T = skel.shape[0]
    outlier_mask = np.zeros(T, dtype=bool)
    per_joint_outliers = np.zeros(48, dtype=int)

    if T < 3:
        return outlier_mask, per_joint_outliers

    vel = compute_frame_velocity(skel)  # (T-1, 48)

    for j in range(48):
        jv = vel[:, j]
        # Only consider frames where the joint is not zero (detected)
        active = jv > 1e-6
        if active.sum() < 3:
            continue
        median_v = np.median(jv[active])
        if median_v < 1e-6:
            continue
        threshold = median_v * threshold_multiplier
        jumps = jv > threshold
        for t in range(len(jumps)):
            if jumps[t]:
                outlier_mask[t] = True
                outlier_mask[t + 1] = True  # The frame after the jump too
                per_joint_outliers[j] += 1

    return outlier_mask, per_joint_outliers


def detect_outlier_frames_dbscan(skel, eps_multiplier=3.0, min_samples=2):
    """Use DBSCAN to detect spatially implausible keypoints per frame.

    For each joint across time, compute displacement from its temporal median.
    Frames where the joint is far from its cluster centroid are outliers.

    Returns:
        outlier_mask: (T,) bool
        per_joint_outliers: (48,) int
    """
    from sklearn.cluster import DBSCAN

    T = skel.shape[0]
    outlier_mask = np.zeros(T, dtype=bool)
    per_joint_outliers = np.zeros(48, dtype=int)

    for j in range(48):
        coords = skel[:, j, :]  # (T, 3)
        # Skip joints that are mostly zero (undetected)
        active = np.abs(coords).sum(axis=1) > 1e-4
        if active.sum() < 5:
            continue

        # Compute pairwise distances for DBSCAN eps estimation
        active_coords = coords[active]
        dists = np.linalg.norm(active_coords - np.median(active_coords, axis=0), axis=1)
        median_dist = np.median(dists)
        if median_dist < 1e-6:
            continue

        eps = median_dist * eps_multiplier

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_coords)
        labels = clustering.labels_

        # Outliers are labeled -1
        outlier_indices = np.where(active)[0][labels == -1]
        for idx in outlier_indices:
            outlier_mask[idx] = True
            per_joint_outliers[j] += 1

    return outlier_mask, per_joint_outliers


def compute_hand_presence(skel):
    """Compute per-frame hand presence (non-zero detection)."""
    lh = skel[:, :21, :]
    rh = skel[:, 21:42, :]
    lh_present = np.abs(lh).sum(axis=(1, 2)) > 0.01
    rh_present = np.abs(rh).sum(axis=(1, 2)) > 0.01
    return lh_present, rh_present


def main():
    parser = argparse.ArgumentParser(
        description="Data-Centric Keypoint Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=str,
                        default="/scratch/alpine/hama5612/ksl-dir-2/data/results/keypoint_analysis.json",
                        help="Output JSON path")
    parser.add_argument("--velocity-threshold", type=float, default=5.0,
                        help="Multiplier for median velocity for outlier detection")
    parser.add_argument("--dbscan-eps-mult", type=float, default=3.0,
                        help="DBSCAN eps as multiplier of median displacement")
    args = parser.parse_args()

    data_dirs = [
        "/scratch/alpine/hama5612/ksl-dir-2/data/train_alpha",
        "/scratch/alpine/hama5612/ksl-dir-2/data/val_alpha",
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("Data-Centric Keypoint Analysis for KSL")
    print(f"Started: {ts}")
    print(f"Velocity outlier threshold: {args.velocity_threshold}x median")
    print(f"DBSCAN eps multiplier: {args.dbscan_eps_mult}x median dist")
    print("=" * 70)

    # =========================================================================
    # PHASE 1: Load all samples
    # =========================================================================
    print("\n--- PHASE 1: Loading all samples ---")

    samples = []  # list of (skeleton, class, signer, filename, num_frames)
    load_errors = 0

    for dd in data_dirs:
        for cls in sorted(os.listdir(dd)):
            cls_path = os.path.join(dd, cls)
            if not os.path.isdir(cls_path):
                continue
            for fn in sorted(os.listdir(cls_path)):
                if not fn.endswith(".npy"):
                    continue
                try:
                    raw = np.load(os.path.join(cls_path, fn)).astype(np.float32)
                    skel = extract_skeleton(raw)
                    if skel is None:
                        load_errors += 1
                        continue
                    signer = extract_signer_id(fn)
                    samples.append((skel, cls, signer, fn, skel.shape[0]))
                except Exception as e:
                    print(f"  ERROR loading {fn}: {e}")
                    load_errors += 1

    print(f"  Loaded {len(samples)} samples ({load_errors} errors)")

    all_signers = sorted(set(s[2] for s in samples))
    all_classes = sorted(set(s[1] for s in samples))
    print(f"  Classes: {len(all_classes)}")
    print(f"  Signers: {len(all_signers)}")

    # =========================================================================
    # PHASE 2: Temporal statistics
    # =========================================================================
    print("\n--- PHASE 2: Temporal Statistics ---")

    frame_counts = [s[4] for s in samples]
    print(f"  Sequence length: min={min(frame_counts)}, max={max(frame_counts)}, "
          f"mean={np.mean(frame_counts):.1f}, median={np.median(frame_counts):.0f}, "
          f"std={np.std(frame_counts):.1f}")

    per_signer_frames = defaultdict(list)
    for skel, cls, signer, fn, nf in samples:
        per_signer_frames[signer].append(nf)

    print(f"\n  Per-signer sequence lengths:")
    for signer in all_signers:
        frames = per_signer_frames[signer]
        print(f"    Signer {signer:>3s}: mean={np.mean(frames):.0f}, "
              f"std={np.std(frames):.0f}, range=[{min(frames)},{max(frames)}]")

    # =========================================================================
    # PHASE 3: Hand presence analysis
    # =========================================================================
    print("\n--- PHASE 3: Hand Presence Analysis ---")

    per_signer_hand_stats = defaultdict(lambda: {
        "lh_present_rate": [], "rh_present_rate": [],
        "both_present_rate": [], "neither_present_rate": [],
    })

    for skel, cls, signer, fn, nf in samples:
        lh_p, rh_p = compute_hand_presence(skel)
        total = len(lh_p)
        per_signer_hand_stats[signer]["lh_present_rate"].append(lh_p.mean())
        per_signer_hand_stats[signer]["rh_present_rate"].append(rh_p.mean())
        per_signer_hand_stats[signer]["both_present_rate"].append((lh_p & rh_p).mean())
        per_signer_hand_stats[signer]["neither_present_rate"].append((~lh_p & ~rh_p).mean())

    print(f"  Per-signer hand detection rates (mean across videos):")
    print(f"    {'Signer':>8s}  {'LH%':>6s}  {'RH%':>6s}  {'Both%':>6s}  {'None%':>6s}")
    for signer in all_signers:
        stats = per_signer_hand_stats[signer]
        print(f"    {signer:>8s}  {100*np.mean(stats['lh_present_rate']):5.1f}%  "
              f"{100*np.mean(stats['rh_present_rate']):5.1f}%  "
              f"{100*np.mean(stats['both_present_rate']):5.1f}%  "
              f"{100*np.mean(stats['neither_present_rate']):5.1f}%")

    # =========================================================================
    # PHASE 4: Velocity-based outlier detection
    # =========================================================================
    print("\n--- PHASE 4: Velocity-Based Outlier Detection ---")

    per_joint_vel_outliers = np.zeros(48, dtype=int)
    per_signer_vel_outliers = defaultdict(lambda: {"outlier_frames": 0, "total_frames": 0})
    per_class_vel_outliers = defaultdict(lambda: {"outlier_frames": 0, "total_frames": 0})
    samples_with_outliers = 0
    total_outlier_frames = 0
    total_frames = 0

    for skel, cls, signer, fn, nf in samples:
        mask, joint_outs = detect_outlier_frames_velocity(skel, args.velocity_threshold)
        n_outlier = mask.sum()
        per_joint_vel_outliers += joint_outs
        per_signer_vel_outliers[signer]["outlier_frames"] += n_outlier
        per_signer_vel_outliers[signer]["total_frames"] += nf
        per_class_vel_outliers[cls]["outlier_frames"] += n_outlier
        per_class_vel_outliers[cls]["total_frames"] += nf
        if n_outlier > 0:
            samples_with_outliers += 1
        total_outlier_frames += n_outlier
        total_frames += nf

    print(f"  Total frames: {total_frames}")
    print(f"  Outlier frames: {total_outlier_frames} ({100*total_outlier_frames/total_frames:.2f}%)")
    print(f"  Samples with any outliers: {samples_with_outliers}/{len(samples)} "
          f"({100*samples_with_outliers/len(samples):.1f}%)")

    print(f"\n  Top 15 joints by velocity outlier count:")
    ranked = np.argsort(per_joint_vel_outliers)[::-1]
    for i in range(min(15, 48)):
        j = ranked[i]
        count = per_joint_vel_outliers[j]
        if count == 0:
            break
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): {count:5d} outlier events")

    print(f"\n  Per-signer velocity outlier rate:")
    for signer in all_signers:
        s = per_signer_vel_outliers[signer]
        rate = 100 * s["outlier_frames"] / s["total_frames"] if s["total_frames"] > 0 else 0
        print(f"    Signer {signer:>3s}: {s['outlier_frames']:5d}/{s['total_frames']:6d} = {rate:.2f}%")

    # =========================================================================
    # PHASE 5: DBSCAN-based outlier detection
    # =========================================================================
    print("\n--- PHASE 5: DBSCAN-Based Outlier Detection ---")

    per_joint_dbscan_outliers = np.zeros(48, dtype=int)
    per_signer_dbscan_outliers = defaultdict(lambda: {"outlier_frames": 0, "total_frames": 0})
    dbscan_total_outlier_frames = 0
    dbscan_samples_with_outliers = 0

    for idx, (skel, cls, signer, fn, nf) in enumerate(samples):
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(samples)}...")
        mask, joint_outs = detect_outlier_frames_dbscan(skel, args.dbscan_eps_mult)
        n_outlier = mask.sum()
        per_joint_dbscan_outliers += joint_outs
        per_signer_dbscan_outliers[signer]["outlier_frames"] += n_outlier
        per_signer_dbscan_outliers[signer]["total_frames"] += nf
        if n_outlier > 0:
            dbscan_samples_with_outliers += 1
        dbscan_total_outlier_frames += n_outlier

    print(f"\n  DBSCAN outlier frames: {dbscan_total_outlier_frames} "
          f"({100*dbscan_total_outlier_frames/total_frames:.2f}%)")
    print(f"  Samples with DBSCAN outliers: {dbscan_samples_with_outliers}/{len(samples)} "
          f"({100*dbscan_samples_with_outliers/len(samples):.1f}%)")

    print(f"\n  Top 15 joints by DBSCAN outlier count:")
    ranked_db = np.argsort(per_joint_dbscan_outliers)[::-1]
    for i in range(min(15, 48)):
        j = ranked_db[i]
        count = per_joint_dbscan_outliers[j]
        if count == 0:
            break
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): {count:5d} outlier events")

    print(f"\n  Per-signer DBSCAN outlier rate:")
    for signer in all_signers:
        s = per_signer_dbscan_outliers[signer]
        rate = 100 * s["outlier_frames"] / s["total_frames"] if s["total_frames"] > 0 else 0
        print(f"    Signer {signer:>3s}: {s['outlier_frames']:5d}/{s['total_frames']:6d} = {rate:.2f}%")

    # =========================================================================
    # PHASE 6: Cross-signer variance analysis (per keypoint, per class)
    # =========================================================================
    print("\n--- PHASE 6: Cross-Signer Variance Analysis ---")
    print("  (Which keypoints vary most across signers for the same class?)")

    # For each class, compute per-signer mean position of each joint (after wrist normalization)
    # Then measure variance across signers

    # First normalize each skeleton (wrist-palm normalization, same as training)
    def normalize_wrist_palm(h):
        h = h.copy()
        lh = h[:, :21, :]
        lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
        if np.any(lh_valid):
            lh_wrist = lh[:, 0:1, :]
            lh[lh_valid] = lh[lh_valid] - lh_wrist[lh_valid]
            palm_sizes = np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True)
            palm_sizes = np.maximum(palm_sizes, 1e-6)
            lh[lh_valid] = lh[lh_valid] / palm_sizes[:, :, np.newaxis]
        h[:, :21, :] = lh

        rh = h[:, 21:42, :]
        rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
        if np.any(rh_valid):
            rh_wrist = rh[:, 0:1, :]
            rh[rh_valid] = rh[rh_valid] - rh_wrist[rh_valid]
            palm_sizes = np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True)
            palm_sizes = np.maximum(palm_sizes, 1e-6)
            rh[rh_valid] = rh[rh_valid] / palm_sizes[:, :, np.newaxis]
        h[:, 21:42, :] = rh

        pose = h[:, 42:48, :]
        pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
        if np.any(pose_valid):
            mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
            pose[pose_valid] = pose[pose_valid] - mid_shoulder[pose_valid]
            shoulder_width = np.linalg.norm(
                pose[pose_valid, 0, :] - pose[pose_valid, 1, :],
                axis=-1, keepdims=True
            )
            shoulder_width = np.maximum(shoulder_width, 1e-6)
            pose[pose_valid] = pose[pose_valid] / shoulder_width[:, :, np.newaxis]
        h[:, 42:48, :] = pose
        return h

    # Collect per-class, per-signer mean joint positions (temporally averaged)
    class_signer_means = defaultdict(lambda: defaultdict(list))  # class -> signer -> list of (48,3) means

    for skel, cls, signer, fn, nf in samples:
        norm_skel = normalize_wrist_palm(skel)
        # Temporal mean of each joint position
        mean_pos = norm_skel.mean(axis=0)  # (48, 3)
        class_signer_means[cls][signer].append(mean_pos)

    # For each class, compute the mean per signer, then variance across signers
    # per_joint_cross_signer_var[j] = average (across classes) of cross-signer variance for joint j
    per_joint_cross_signer_var = np.zeros(48)
    per_joint_class_vars = defaultdict(list)  # joint -> list of per-class variances

    for cls in all_classes:
        signer_means = {}  # signer -> (48, 3) mean
        for signer, mean_list in class_signer_means[cls].items():
            signer_means[signer] = np.mean(mean_list, axis=0)  # average across repetitions

        if len(signer_means) < 2:
            continue

        stacked = np.stack(list(signer_means.values()))  # (num_signers, 48, 3)
        # Variance of each joint position across signers
        var_per_joint = stacked.var(axis=0).sum(axis=1)  # (48,) - sum over xyz
        per_joint_cross_signer_var += var_per_joint

        for j in range(48):
            per_joint_class_vars[j].append(var_per_joint[j])

    per_joint_cross_signer_var /= len(all_classes)

    print(f"\n  Top 15 joints by CROSS-SIGNER variance (after normalization):")
    print(f"  (High variance = different signers place this joint differently for the same sign)")
    ranked_var = np.argsort(per_joint_cross_signer_var)[::-1]
    for i in range(min(15, 48)):
        j = ranked_var[i]
        v = per_joint_cross_signer_var[j]
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): var={v:.6f}")

    print(f"\n  Bottom 10 joints (most consistent across signers):")
    for i in range(min(10, 48)):
        j = ranked_var[-(i+1)]
        v = per_joint_cross_signer_var[j]
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): var={v:.6f}")

    # Group-level variance
    print(f"\n  Joint GROUP cross-signer variance:")
    for group_name, joint_indices in JOINT_GROUPS.items():
        group_var = np.mean([per_joint_cross_signer_var[j] for j in joint_indices])
        print(f"    {group_name:>15s}: mean_var={group_var:.6f}")

    # =========================================================================
    # PHASE 7: Within-class vs between-class variance (discriminability)
    # =========================================================================
    print("\n--- PHASE 7: Joint Discriminability (Between-Class / Within-Class Variance) ---")
    print("  (High ratio = joint helps distinguish classes; Low ratio = noise)")

    # Within-class variance: average variance of joint positions within each class
    within_var = np.zeros(48)
    between_var = np.zeros(48)

    class_means_all = {}  # cls -> (48, 3) grand mean
    for cls in all_classes:
        all_means = []
        for signer, mean_list in class_signer_means[cls].items():
            all_means.extend(mean_list)
        if all_means:
            stacked = np.stack(all_means)  # (N, 48, 3)
            class_means_all[cls] = stacked.mean(axis=0)  # (48, 3)
            within_var += stacked.var(axis=0).sum(axis=1)  # sum over xyz

    within_var /= len(all_classes)

    # Between-class variance
    if len(class_means_all) > 1:
        grand_mean = np.mean(list(class_means_all.values()), axis=0)  # (48, 3)
        for cls, cm in class_means_all.items():
            between_var += ((cm - grand_mean) ** 2).sum(axis=1)
        between_var /= len(class_means_all)

    # Fisher ratio: between / within
    fisher_ratio = np.zeros(48)
    for j in range(48):
        if within_var[j] > 1e-10:
            fisher_ratio[j] = between_var[j] / within_var[j]

    print(f"\n  Top 15 most DISCRIMINATIVE joints (high Fisher ratio):")
    ranked_fisher = np.argsort(fisher_ratio)[::-1]
    for i in range(min(15, 48)):
        j = ranked_fisher[i]
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): Fisher={fisher_ratio[j]:.4f} "
              f"(between={between_var[j]:.6f}, within={within_var[j]:.6f})")

    print(f"\n  Bottom 10 LEAST discriminative joints:")
    for i in range(min(10, 48)):
        j = ranked_fisher[-(i+1)]
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): Fisher={fisher_ratio[j]:.4f} "
              f"(between={between_var[j]:.6f}, within={within_var[j]:.6f})")

    # Identify joints that are high signer-variance but low discriminability (noise candidates)
    print(f"\n  NOISE CANDIDATES (high cross-signer var, low Fisher ratio):")
    print(f"  (These joints differ across signers but DON'T help distinguish classes)")
    noise_score = np.zeros(48)
    for j in range(48):
        if fisher_ratio[j] > 1e-10:
            noise_score[j] = per_joint_cross_signer_var[j] / fisher_ratio[j]
        else:
            noise_score[j] = per_joint_cross_signer_var[j] * 1000  # very noisy if no discrimination
    ranked_noise = np.argsort(noise_score)[::-1]
    for i in range(min(10, 48)):
        j = ranked_noise[i]
        print(f"    {JOINT_NAMES[j]:>20s} (j{j:2d}): noise_score={noise_score[j]:.4f} "
              f"(signer_var={per_joint_cross_signer_var[j]:.6f}, fisher={fisher_ratio[j]:.4f})")

    # =========================================================================
    # PHASE 8: Summary and Recommendations
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    print(f"\n  1. OUTLIER DETECTION:")
    print(f"     Velocity outliers: {total_outlier_frames}/{total_frames} frames "
          f"({100*total_outlier_frames/total_frames:.2f}%)")
    print(f"     DBSCAN outliers: {dbscan_total_outlier_frames}/{total_frames} frames "
          f"({100*dbscan_total_outlier_frames/total_frames:.2f}%)")
    print(f"     Affected samples: velocity={samples_with_outliers}/{len(samples)}, "
          f"DBSCAN={dbscan_samples_with_outliers}/{len(samples)}")

    # Find signers with highest outlier rates
    worst_signers_vel = sorted(
        per_signer_vel_outliers.items(),
        key=lambda x: x[1]["outlier_frames"] / max(x[1]["total_frames"], 1),
        reverse=True
    )
    print(f"\n  2. WORST SIGNERS (most outliers):")
    for signer, stats in worst_signers_vel[:3]:
        rate = 100 * stats["outlier_frames"] / stats["total_frames"]
        print(f"     Signer {signer}: {rate:.2f}% outlier frames")

    print(f"\n  3. NOISE CANDIDATE JOINTS (consider dropping or downweighting):")
    for i in range(min(5, 48)):
        j = ranked_noise[i]
        print(f"     {JOINT_NAMES[j]} (j{j})")

    print(f"\n  4. MOST DISCRIMINATIVE JOINTS (preserve these):")
    for i in range(min(5, 48)):
        j = ranked_fisher[i]
        print(f"     {JOINT_NAMES[j]} (j{j}): Fisher={fisher_ratio[j]:.4f}")

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {
        "timestamp": ts,
        "num_samples": len(samples),
        "num_classes": len(all_classes),
        "num_signers": len(all_signers),
        "temporal_stats": {
            "min_frames": int(min(frame_counts)),
            "max_frames": int(max(frame_counts)),
            "mean_frames": float(np.mean(frame_counts)),
            "median_frames": float(np.median(frame_counts)),
            "std_frames": float(np.std(frame_counts)),
        },
        "velocity_outliers": {
            "total_outlier_frames": int(total_outlier_frames),
            "total_frames": int(total_frames),
            "outlier_rate": float(total_outlier_frames / total_frames),
            "per_joint": {JOINT_NAMES[j]: int(per_joint_vel_outliers[j]) for j in range(48)},
            "per_signer": {s: {"outlier_rate": float(v["outlier_frames"] / max(v["total_frames"], 1)),
                               "outlier_frames": int(v["outlier_frames"]),
                               "total_frames": int(v["total_frames"])}
                          for s, v in per_signer_vel_outliers.items()},
        },
        "dbscan_outliers": {
            "total_outlier_frames": int(dbscan_total_outlier_frames),
            "outlier_rate": float(dbscan_total_outlier_frames / total_frames),
            "per_joint": {JOINT_NAMES[j]: int(per_joint_dbscan_outliers[j]) for j in range(48)},
        },
        "cross_signer_variance": {
            JOINT_NAMES[j]: float(per_joint_cross_signer_var[j]) for j in range(48)
        },
        "fisher_ratio": {
            JOINT_NAMES[j]: float(fisher_ratio[j]) for j in range(48)
        },
        "noise_score": {
            JOINT_NAMES[j]: float(noise_score[j]) for j in range(48)
        },
        "noise_candidates": [JOINT_NAMES[ranked_noise[i]] for i in range(min(10, 48))],
        "most_discriminative": [JOINT_NAMES[ranked_fisher[i]] for i in range(min(10, 48))],
        "hand_presence": {
            signer: {
                "lh_rate": float(np.mean(stats["lh_present_rate"])),
                "rh_rate": float(np.mean(stats["rh_present_rate"])),
                "both_rate": float(np.mean(stats["both_present_rate"])),
                "neither_rate": float(np.mean(stats["neither_present_rate"])),
            }
            for signer, stats in per_signer_hand_stats.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
