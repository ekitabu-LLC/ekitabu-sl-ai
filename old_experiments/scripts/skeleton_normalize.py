#!/usr/bin/env python3
"""
Anchor-based skeleton normalization for KSL sign language recognition.

V44 Experiment 1: Apply global anchor normalization to raw MediaPipe landmarks
BEFORE the standard per-segment (wrist/palm/shoulder) normalization pipeline.

The existing normalize_wrist_palm normalizes each body segment independently:
  - Left hand: center on wrist, scale by palm width
  - Right hand: center on wrist, scale by palm width
  - Pose: center on shoulder midpoint, scale by shoulder width

This anchor normalization provides an ADDITIONAL pre-processing step applied
to raw landmarks that globally centers and scales the entire skeleton:
  1. Center ALL joints on the shoulder midpoint (global translation invariance)
  2. Scale ALL joints by shoulder width (global scale invariance)

This removes signer-dependent position-in-frame and body-size variation
before the per-segment normalization pipeline runs, potentially helping
cross-signer generalization for models already trained with per-segment norm.

48-joint layout (confirmed from preprocess_for_onnx.py):
  - Joints 0-20:  Left hand  (LH wrist = 0, LH middle MCP = 9)
  - Joints 21-41: Right hand (RH wrist = 21, RH middle MCP = 30)
  - Joints 42-47: Pose       (L shoulder = 42, R shoulder = 43,
                               L elbow = 44, R elbow = 45,
                               L wrist(pose) = 46, R wrist(pose) = 47)

Raw MediaPipe landmark layout (549 columns):
  - [0:99]    Pose landmarks (33 joints x 3)
  - [99:162]  Left hand (21 joints x 3)
  - [162:225] Right hand (21 joints x 3)
  - [225:549] Face (not used)

POSE_INDICES = [11, 12, 13, 14, 15, 16] map to our joints 42-47.
  Pose landmark 11 = L shoulder -> our joint 42
  Pose landmark 12 = R shoulder -> our joint 43
"""

import numpy as np


# MediaPipe pose landmark indices for shoulders
_POSE_L_SHOULDER_IDX = 11  # in 33-landmark pose
_POSE_R_SHOULDER_IDX = 12


def anchor_normalize(raw):
    """
    Apply global anchor-based normalization to raw MediaPipe landmarks.

    Parameters
    ----------
    raw : np.ndarray, shape (T, >=225)
        Raw MediaPipe output: pose(99) + left_hand(63) + right_hand(63) + ...

    Returns
    -------
    np.ndarray, same shape as input
        Normalized raw landmarks where all (x,y,z) coordinates are centered
        on shoulder midpoint and scaled by shoulder width.
    """
    raw = raw.copy()
    T = raw.shape[0]

    # Extract shoulder coordinates from pose landmarks
    ls_start = _POSE_L_SHOULDER_IDX * 3  # index 33
    rs_start = _POSE_R_SHOULDER_IDX * 3  # index 36

    l_shoulder = raw[:, ls_start:ls_start + 3]  # (T, 3)
    r_shoulder = raw[:, rs_start:rs_start + 3]  # (T, 3)

    # Compute anchor: shoulder midpoint
    midpoint = (l_shoulder + r_shoulder) / 2.0  # (T, 3)

    # Compute scale: shoulder width (per frame)
    shoulder_width = np.linalg.norm(
        l_shoulder - r_shoulder, axis=-1, keepdims=True
    )  # (T, 1)
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    # Check if pose landmarks are valid (not all zeros)
    pose_valid = np.abs(raw[:, :99]).sum(axis=1) > 0.01  # (T,)

    if not np.any(pose_valid):
        # No valid pose data, return unchanged
        return raw

    # Normalize pose landmarks (33 joints x 3 = 99 values)
    for j in range(33):
        start = j * 3
        for c in range(3):
            raw[pose_valid, start + c] = (
                (raw[pose_valid, start + c] - midpoint[pose_valid, c])
                / shoulder_width[pose_valid, 0]
            )

    # Normalize left hand (21 joints x 3 = 63 values, starting at col 99)
    lh_block = raw[:, 99:162].reshape(T, 21, 3)
    lh_valid = pose_valid & (np.abs(lh_block).sum(axis=(1, 2)) > 0.01)
    if np.any(lh_valid):
        for c in range(3):
            lh_block[lh_valid, :, c] = (
                (lh_block[lh_valid, :, c] - midpoint[lh_valid, c:c + 1])
                / shoulder_width[lh_valid]
            )
        raw[:, 99:162] = lh_block.reshape(T, 63)

    # Normalize right hand (21 joints x 3 = 63 values, starting at col 162)
    rh_block = raw[:, 162:225].reshape(T, 21, 3)
    rh_valid = pose_valid & (np.abs(rh_block).sum(axis=(1, 2)) > 0.01)
    if np.any(rh_valid):
        for c in range(3):
            rh_block[rh_valid, :, c] = (
                (rh_block[rh_valid, :, c] - midpoint[rh_valid, c:c + 1])
                / shoulder_width[rh_valid]
            )
        raw[:, 162:225] = rh_block.reshape(T, 63)

    return raw
