#!/usr/bin/env python3
"""
Preprocess a KSL sign language video into tensors ready for ONNX inference.

Usage:
    python preprocess_for_onnx.py video.mp4 --output-dir out/
    python preprocess_for_onnx.py video.mp4 --output-dir out/ --save-raw

Output files (NumPy .npy, dtype float32):
    out/joint.npy        (1, 3, 90, 48)  — for v28 / exp1 / exp5 / v41 / v43
    out/bone.npy         (1, 3, 90, 48)  — for v28 / exp1 / exp5 / v41 / v43
    out/velocity.npy     (1, 3, 90, 48)  — for v28 / exp1 / exp5 / v41 / v43
    out/aux.npy          (1, 90, D_aux)  — for all multistream + v27/v29 models
    out/gcn9.npy         (1, 9, 90, 48)  — for v27 / v29 only
    out/openhands.npy    (1, 2, 90, 27)  — for OpenHands model only
    out/raw.npy          (T, 549)        — raw MediaPipe landmarks (if --save-raw)

Quick inference example:
    import numpy as np, onnxruntime as ort
    joint = np.load("out/joint.npy")    # (1,3,90,48)
    bone  = np.load("out/bone.npy")
    vel   = np.load("out/velocity.npy")
    aux   = np.load("out/aux.npy")      # (1,90,D_aux)

    sess = ort.InferenceSession("onnx_models/numbers/v43/model_adapted.onnx")
    logits = sess.run(["logits"], {"gcn": joint, "aux": aux})[0]  # wrong! use fused
    # See README or evaluate_onnx.py for full ensemble fusion.
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (must match training code exactly)
# ---------------------------------------------------------------------------

MAX_FRAMES = 90
NUM_NODES  = 48

# MediaPipe pose indices used (shoulder + elbow joints)
POSE_INDICES = [11, 12, 13, 14, 15, 16]  # indices in pose_landmarks (33 total)

# Parent map for 48 joints: [lh(0-20), rh(21-41), pose(42-47)]
LH_PARENT   = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
RH_PARENT   = [-1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
POSE_PARENT = [-1, 42, 42, 43, 44, 45]
PARENT_MAP  = LH_PARENT + RH_PARENT + POSE_PARENT  # length 48

# Fingertip indices for left/right hands (within the 48-joint system)
LH_TIPS = [4, 8, 12, 16, 20]
RH_TIPS = [25, 29, 33, 37, 41]
NUM_FINGERTIP_PAIRS  = 10   # C(5,2) per hand
NUM_HAND_BODY_FEATURES = 8

# OpenHands 27-joint index → our 48-joint index (-1: approximate nose, -2: zero)
OH27_TO_OUR48 = {
    0: -1,   # nose → approximate from shoulders
    1: -2,   # l_eye_outer → zero
    2: -2,   # r_eye_outer → zero
    3: 42,   # l_shoulder
    4: 43,   # r_shoulder
    5: 44,   # l_elbow
    6: 45,   # r_elbow
    7: 0,    # lh_wrist
    8: 4,    # lh_thumb_tip
    9: 5,    # lh_index_mcp
    10: 8,   # lh_index_tip
    11: 9,   # lh_middle_mcp
    12: 12,  # lh_middle_tip
    13: 13,  # lh_ring_mcp
    14: 16,  # lh_ring_tip
    15: 17,  # lh_pinky_mcp
    16: 20,  # lh_pinky_tip
    17: 21,  # rh_wrist
    18: 25,  # rh_thumb_tip
    19: 26,  # rh_index_mcp
    20: 29,  # rh_index_tip
    21: 30,  # rh_middle_mcp
    22: 33,  # rh_middle_tip
    23: 34,  # rh_ring_mcp
    24: 37,  # rh_ring_tip
    25: 38,  # rh_pinky_mcp
    26: 41,  # rh_pinky_tip
}


# ---------------------------------------------------------------------------
# Build angle joints once (node, parent, child) triplets
# ---------------------------------------------------------------------------
def _build_angle_joints():
    from collections import defaultdict
    children = defaultdict(list)
    for child_idx, parent_idx in enumerate(PARENT_MAP):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    angle_joints = []
    for node in range(NUM_NODES):
        parent = PARENT_MAP[node]
        if parent < 0:
            continue
        for child in children[node]:
            angle_joints.append((node, parent, child))
    return angle_joints


ANGLE_JOINTS      = _build_angle_joints()
NUM_ANGLE_FEATURES = len(ANGLE_JOINTS)
D_AUX             = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES


# ===========================================================================
# STEP 1 — MediaPipe extraction
# ===========================================================================

def extract_landmarks_from_video(video_path, verbose=True):
    """Run MediaPipe Holistic on every frame.

    Returns raw (T, 549) float32 array:
        [0:99]   — pose (33 joints × 3), only indices 0-32 used by MediaPipe
        [99:162] — left hand (21 joints × 3)
        [162:225]— right hand (21 joints × 3)
        (remaining 324 cols are face landmarks, not used in preprocessing)

    Returns None if video cannot be opened or has no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}", file=sys.stderr)
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    )

    all_frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        row = []
        # Pose (33 × 3 = 99 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([0.0] * 99)
        # Left hand (21 × 3 = 63 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([0.0] * 63)
        # Right hand (21 × 3 = 63 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
        else:
            row.extend([0.0] * 63)

        all_frames.append(row)

    cap.release()
    holistic.close()

    if len(all_frames) == 0:
        print(f"  ERROR: No frames extracted from: {video_path}", file=sys.stderr)
        return None

    raw = np.array(all_frames, dtype=np.float32)
    if verbose:
        print(f"  MediaPipe: {len(all_frames)} frames extracted → shape {raw.shape}")
    return raw


# ===========================================================================
# STEP 2 — Skeleton helpers
# ===========================================================================

def _parse_48_joints(raw):
    """Parse raw MediaPipe output into (T, 48, 3) joint positions."""
    f = raw.shape[0]
    pose = np.zeros((f, 6, 3), dtype=np.float32)
    for pi, idx_pose in enumerate(POSE_INDICES):
        start = idx_pose * 3
        pose[:, pi, :] = raw[:, start:start + 3]
    lh = raw[:, 99:162].reshape(f, 21, 3)
    rh = raw[:, 162:225].reshape(f, 21, 3)
    return np.concatenate([lh, rh, pose], axis=1).astype(np.float32)


def _normalize_wrist_palm(h):
    """Root-relative + scale normalization for each hand / pose segment."""
    h = h.copy()

    lh = h[:, :21, :]
    lh_valid = np.abs(lh).sum(axis=(1, 2)) > 0.01
    if np.any(lh_valid):
        lh_wrist = lh[:, 0:1, :]
        lh[lh_valid] -= lh_wrist[lh_valid]
        palm = np.maximum(np.linalg.norm(lh[lh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        lh[lh_valid] /= palm[:, :, np.newaxis]
    h[:, :21, :] = lh

    rh = h[:, 21:42, :]
    rh_valid = np.abs(rh).sum(axis=(1, 2)) > 0.01
    if np.any(rh_valid):
        rh_wrist = rh[:, 0:1, :]
        rh[rh_valid] -= rh_wrist[rh_valid]
        palm = np.maximum(np.linalg.norm(rh[rh_valid, 9, :], axis=-1, keepdims=True), 1e-6)
        rh[rh_valid] /= palm[:, :, np.newaxis]
    h[:, 21:42, :] = rh

    pose = h[:, 42:48, :]
    pose_valid = np.abs(pose).sum(axis=(1, 2)) > 0.01
    if np.any(pose_valid):
        mid_shoulder = (pose[:, 0:1, :] + pose[:, 1:2, :]) / 2
        pose[pose_valid] -= mid_shoulder[pose_valid]
        sw = np.maximum(
            np.linalg.norm(pose[pose_valid, 0, :] - pose[pose_valid, 1, :], axis=-1, keepdims=True), 1e-6
        )
        pose[pose_valid] /= sw[:, :, np.newaxis]
    h[:, 42:48, :] = pose

    return h


def _compute_bones(h):
    """Per-joint bone vectors: joint - parent_joint."""
    bones = np.zeros_like(h)
    for child in range(NUM_NODES):
        parent = PARENT_MAP[child]
        if parent >= 0:
            bones[:, child, :] = h[:, child, :] - h[:, parent, :]
    return bones


def _compute_joint_angles(h):
    """Angle at each interior joint (between incoming and outgoing bone)."""
    T = h.shape[0]
    angles = np.zeros((T, NUM_ANGLE_FEATURES), dtype=np.float32)
    for idx, (node, parent, child) in enumerate(ANGLE_JOINTS):
        bone_in  = h[:, node, :] - h[:, parent, :]
        bone_out = h[:, child, :] - h[:, node, :]
        n_in  = np.maximum(np.linalg.norm(bone_in,  axis=-1, keepdims=True), 1e-8)
        n_out = np.maximum(np.linalg.norm(bone_out, axis=-1, keepdims=True), 1e-8)
        dot = np.sum((bone_in / n_in) * (bone_out / n_out), axis=-1)
        angles[:, idx] = np.arccos(np.clip(dot, -1.0, 1.0))
    return angles


def _compute_fingertip_distances(h):
    """Pairwise distances between fingertips on each hand."""
    T = h.shape[0]
    distances = np.zeros((T, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
    col = 0
    for tips in [LH_TIPS, RH_TIPS]:
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                diff = h[:, tips[i], :] - h[:, tips[j], :]
                distances[:, col] = np.linalg.norm(diff, axis=-1)
                col += 1
    return distances


def _compute_hand_body_features(h_raw):
    """Global hand-to-body position features (8 values per frame)."""
    T = h_raw.shape[0]
    features = np.zeros((T, 8), dtype=np.float32)
    mid_shoulder = (h_raw[:, 42, :] + h_raw[:, 43, :]) / 2
    sw = np.maximum(np.linalg.norm(h_raw[:, 42, :] - h_raw[:, 43, :], axis=-1, keepdims=True), 1e-6)
    lh_c = h_raw[:, :21, :].mean(axis=1)
    rh_c = h_raw[:, 21:42, :].mean(axis=1)
    features[:, 0] = (lh_c[:, 1] - mid_shoulder[:, 1]) / sw[:, 0]
    features[:, 1] = (rh_c[:, 1] - mid_shoulder[:, 1]) / sw[:, 0]
    features[:, 2] = (lh_c[:, 0] - mid_shoulder[:, 0]) / sw[:, 0]
    features[:, 3] = (rh_c[:, 0] - mid_shoulder[:, 0]) / sw[:, 0]
    features[:, 4] = np.linalg.norm(lh_c - rh_c, axis=-1) / sw[:, 0]
    face = mid_shoulder.copy()
    face[:, 1] -= sw[:, 0] * 0.7
    features[:, 5] = np.linalg.norm(lh_c - face, axis=-1) / sw[:, 0]
    features[:, 6] = np.linalg.norm(rh_c - face, axis=-1) / sw[:, 0]
    features[:, 7] = np.abs(lh_c[:, 1] - rh_c[:, 1]) / sw[:, 0]
    return features


def _temporal_resample(arrays, f, max_frames):
    """Uniformly subsample or zero-pad each array to max_frames along axis 0."""
    if f >= max_frames:
        idx = np.linspace(0, f - 1, max_frames, dtype=int)
        return [a[idx] for a in arrays]
    else:
        pad = max_frames - f
        return [np.concatenate([a, np.zeros((pad,) + a.shape[1:], dtype=np.float32)]) for a in arrays]


def _adapt_48_to_27(h):
    """Convert (T, 48, 3) → (T, 27, 2) for OpenHands (x,y only)."""
    T = h.shape[0]
    out = np.zeros((T, 27, 2), dtype=np.float32)
    for oh_idx, our_idx in OH27_TO_OUR48.items():
        if our_idx == -1:
            mid = (h[:, 42, :2] + h[:, 43, :2]) / 2
            sw  = np.maximum(np.linalg.norm(h[:, 42, :2] - h[:, 43, :2], axis=-1, keepdims=True), 1e-6)
            out[:, oh_idx, 0] = mid[:, 0]
            out[:, oh_idx, 1] = mid[:, 1] - 0.7 * sw[:, 0]
        elif our_idx == -2:
            pass  # leave as zero
        else:
            out[:, oh_idx, :] = h[:, our_idx, :2]
    return out


# ===========================================================================
# STEP 3 — Preprocessing functions
# ===========================================================================

def preprocess_multistream(raw, max_frames=MAX_FRAMES):
    """
    Prepare tensors for multistream models: v28, exp1, exp5, v41, v43.

    Returns:
        joint    np.ndarray (1, 3, T, 48)
        bone     np.ndarray (1, 3, T, 48)
        velocity np.ndarray (1, 3, T, 48)
        aux      np.ndarray (1, T, D_aux)   D_aux = NUM_ANGLE_FEATURES + 20 + 8

    Returns None × 4 if raw is invalid.
    """
    if raw is None or raw.shape[1] < 225:
        return None, None, None, None

    f = raw.shape[0]
    h_raw = _parse_48_joints(raw)
    hand_body = _compute_hand_body_features(h_raw)
    h = _normalize_wrist_palm(h_raw)

    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = _compute_bones(h)
    angles = _compute_joint_angles(h)
    ftdists = _compute_fingertip_distances(h)

    h, velocity, bones, angles, ftdists, hand_body = _temporal_resample(
        [h, velocity, bones, angles, ftdists, hand_body], f, max_frames
    )

    def to_gcn(arr):
        return np.clip(arr, -10, 10).transpose(2, 0, 1)[np.newaxis]  # (1,3,T,48)

    aux_raw = np.concatenate([angles, ftdists, hand_body], axis=1)
    aux = np.clip(aux_raw, -10, 10)[np.newaxis]  # (1, T, D_aux)

    return to_gcn(h), to_gcn(bones), to_gcn(velocity), aux


def preprocess_v27(raw, max_frames=MAX_FRAMES):
    """
    Prepare tensors for v27 / v29 (9-channel single model).

    Channel order: [joint(3) | velocity(3) | bone(3)]

    Returns:
        gcn9  np.ndarray (1, 9, T, 48)
        aux   np.ndarray (1, T, D_aux)

    Returns None × 2 if raw is invalid.
    """
    if raw is None or raw.shape[1] < 225:
        return None, None

    f = raw.shape[0]
    h_raw = _parse_48_joints(raw)
    hand_body = _compute_hand_body_features(h_raw)
    h = _normalize_wrist_palm(h_raw)

    velocity = np.zeros_like(h)
    velocity[1:] = h[1:] - h[:-1]
    bones = _compute_bones(h)
    angles = _compute_joint_angles(h)
    ftdists = _compute_fingertip_distances(h)

    h, velocity, bones, angles, ftdists, hand_body = _temporal_resample(
        [h, velocity, bones, angles, ftdists, hand_body], f, max_frames
    )

    # IMPORTANT: order is joint | velocity | bone (must match training)
    gcn_raw = np.concatenate([h, velocity, bones], axis=2)  # (T, 48, 9)
    gcn9 = np.clip(gcn_raw, -10, 10).transpose(2, 0, 1)[np.newaxis]  # (1,9,T,48)

    aux_raw = np.concatenate([angles, ftdists, hand_body], axis=1)
    aux = np.clip(aux_raw, -10, 10)[np.newaxis]  # (1, T, D_aux)

    return gcn9, aux


def preprocess_openhands(raw, max_frames=MAX_FRAMES):
    """
    Prepare tensor for the OpenHands model.

    Returns:
        x  np.ndarray (1, 2, T, 27)   — x/y coordinates only

    Returns None if raw is invalid.
    """
    if raw is None or raw.shape[1] < 225:
        return None

    f = raw.shape[0]
    h_raw = _parse_48_joints(raw)
    h = _normalize_wrist_palm(h_raw)

    (h,) = _temporal_resample([h], f, max_frames)
    h = np.clip(h, -10, 10)

    oh27 = _adapt_48_to_27(h)        # (T, 27, 2)
    oh27 = np.clip(oh27, -10, 10)

    x = oh27.transpose(2, 0, 1)[np.newaxis]  # (1, 2, T, 27)
    return x


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a KSL video into ONNX-ready NumPy tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Input video file (.mp4, .mov, etc.)")
    parser.add_argument("--output-dir", "-o", default="onnx_inputs",
                        help="Directory to save .npy files (default: onnx_inputs/)")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw MediaPipe landmarks as raw.npy")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                        help=f"Temporal length to resample to (default: {MAX_FRAMES})")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info output")
    args = parser.parse_args()

    verbose = not args.quiet

    if not os.path.isfile(args.video):
        print(f"ERROR: File not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. MediaPipe extraction
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[1/3] Extracting landmarks with MediaPipe...")
        print(f"  Input: {args.video}")

    raw = extract_landmarks_from_video(args.video, verbose=verbose)
    if raw is None:
        print("ERROR: Failed to extract landmarks.", file=sys.stderr)
        sys.exit(1)

    if args.save_raw:
        p = os.path.join(args.output_dir, "raw.npy")
        np.save(p, raw)
        if verbose:
            print(f"  Saved raw.npy  {raw.shape}")

    # -----------------------------------------------------------------------
    # 2. Multistream preprocessing (v28 / exp1 / exp5 / v41 / v43)
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[2/3] Preprocessing for multistream models (v28/exp1/exp5/v41/v43)...")

    joint, bone, velocity, aux = preprocess_multistream(raw, args.max_frames)
    if joint is None:
        print("ERROR: preprocess_multistream failed (not enough landmarks?).", file=sys.stderr)
        sys.exit(1)

    np.save(os.path.join(args.output_dir, "joint.npy"),    joint)
    np.save(os.path.join(args.output_dir, "bone.npy"),     bone)
    np.save(os.path.join(args.output_dir, "velocity.npy"), velocity)
    np.save(os.path.join(args.output_dir, "aux.npy"),      aux)

    if verbose:
        print(f"  joint.npy     {joint.shape}   dtype={joint.dtype}")
        print(f"  bone.npy      {bone.shape}")
        print(f"  velocity.npy  {velocity.shape}")
        print(f"  aux.npy       {aux.shape}   (D_aux={aux.shape[-1]})")

    # -----------------------------------------------------------------------
    # 3. v27 / v29 format (9-channel concatenated)
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[3/3] Preprocessing for v27/v29 (9-channel) and OpenHands...")

    gcn9, aux9 = preprocess_v27(raw, args.max_frames)
    if gcn9 is not None:
        np.save(os.path.join(args.output_dir, "gcn9.npy"), gcn9)
        if verbose:
            print(f"  gcn9.npy      {gcn9.shape}")
        # aux is identical to multistream aux — no need to save twice
    else:
        print("  WARNING: preprocess_v27 failed.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 4. OpenHands format (2-channel, 27 joints)
    # -----------------------------------------------------------------------
    x_oh = preprocess_openhands(raw, args.max_frames)
    if x_oh is not None:
        np.save(os.path.join(args.output_dir, "openhands.npy"), x_oh)
        if verbose:
            print(f"  openhands.npy {x_oh.shape}")
    else:
        print("  WARNING: preprocess_openhands failed.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"Done! Saved to: {os.path.abspath(args.output_dir)}/")
        print(f"{'='*60}")
        print(f"\nHow to use with ONNX Runtime:")
        print(f"""
  import numpy as np, onnxruntime as ort

  joint    = np.load("{args.output_dir}/joint.npy")     # (1,3,90,48)
  bone     = np.load("{args.output_dir}/bone.npy")      # (1,3,90,48)
  velocity = np.load("{args.output_dir}/velocity.npy")  # (1,3,90,48)
  aux      = np.load("{args.output_dir}/aux.npy")       # (1,90,{aux.shape[-1]})

  # Example: run v43 numbers model (static ONNX, no AdaBN)
  sess = ort.InferenceSession("onnx_models/numbers/v43/model.onnx")
  logits = sess.run(["logits"], {{"gcn": joint, "aux": aux}})[0]
  print("Predicted class index:", logits.argmax())

  # Example: run v27 numbers model
  gcn9 = np.load("{args.output_dir}/gcn9.npy")         # (1,9,90,48)
  sess = ort.InferenceSession("onnx_models/numbers/v27/model.onnx")
  logits = sess.run(["logits"], {{"gcn": gcn9, "aux": aux}})[0]

  # Example: run OpenHands model
  x = np.load("{args.output_dir}/openhands.npy")       # (1,2,90,27)
  sess = ort.InferenceSession("onnx_models/numbers/openhands/model.onnx")
  logits = sess.run(["logits"], {{"x": x}})[0]
""")


if __name__ == "__main__":
    main()
