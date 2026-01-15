"""
KSL Feature Extraction and Preprocessing
Handles MediaPipe extraction and feature engineering for different model versions.
"""

import numpy as np
import cv2
import base64
from typing import List, Tuple, Optional
import torch

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


# ============================================================================
# Constants
# ============================================================================

POSE_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulder/elbow/wrist subset

# Hand landmark indices
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17
TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Face landmark indices for extraction
LIPS_INDICES = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
    37, 39, 40, 185, 76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 292
]))[:40]

LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]
NOSE = [1, 2, 98, 327]
FACE_INDICES = LIPS_INDICES[:40] + LEFT_EYEBROW[:5] + RIGHT_EYEBROW[:5] + LEFT_EYE[:6] + RIGHT_EYE[:6] + NOSE[:4] + [10, 152]


# ============================================================================
# MediaPipe Extraction
# ============================================================================

class MediaPipeExtractor:
    """Extract landmarks from video frames using MediaPipe."""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not available")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = None

    def __enter__(self):
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )
        return self

    def __exit__(self, *args):
        if self.holistic:
            self.holistic.close()

    def extract_frame(self, frame: np.ndarray, include_face: bool = True) -> np.ndarray:
        """Extract landmarks from a single frame.

        Returns:
            numpy array of shape (549,) if include_face else (225,)
            - Pose: 99 values (33 landmarks * 3)
            - Left hand: 63 values (21 landmarks * 3)
            - Right hand: 63 values (21 landmarks * 3)
            - Face (if included): 324 values (68 landmarks * 3 for lips + face features)
        """
        if self.holistic is None:
            raise RuntimeError("Extractor not initialized. Use 'with' statement.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        landmarks = []

        # Pose landmarks (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        # Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)

        # Face landmarks (68 selected * 3 = 204 for lips, + 120 for face = 324)
        if include_face and results.face_landmarks:
            # Lips (40 * 3 = 120)
            for idx in LIPS_INDICES[:40]:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([0.0, 0.0, 0.0])

            # Face features (68 * 3 = 204)
            for idx in FACE_INDICES:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([0.0, 0.0, 0.0])
        elif include_face:
            landmarks.extend([0.0] * 324)

        return np.array(landmarks, dtype=np.float32)

    def extract_from_frames(self, frames: List[np.ndarray], include_face: bool = True) -> np.ndarray:
        """Extract landmarks from multiple frames.

        Returns:
            numpy array of shape (num_frames, feature_dim)
        """
        all_landmarks = []
        for frame in frames:
            landmarks = self.extract_frame(frame, include_face)
            all_landmarks.append(landmarks)
        return np.array(all_landmarks, dtype=np.float32)


def decode_base64_frames(base64_frames: List[str]) -> List[np.ndarray]:
    """Decode base64 encoded frames to numpy arrays."""
    frames = []
    for b64_frame in base64_frames:
        # Remove data URL prefix if present
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',')[1]

        img_data = base64.b64decode(b64_frame)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            frames.append(frame)

    return frames


# ============================================================================
# Hand Feature Engineering
# ============================================================================

def compute_hand_features(hand_landmarks: np.ndarray) -> np.ndarray:
    """Compute engineered hand features.

    Args:
        hand_landmarks: shape (num_frames, 63) - 21 landmarks * 3 coords

    Returns:
        shape (num_frames, 25) - engineered features per frame
    """
    frames = hand_landmarks.shape[0]
    features = []

    for f in range(frames):
        hand = hand_landmarks[f].reshape(21, 3)
        frame_features = []

        wrist = hand[WRIST]
        hand_normalized = hand - wrist

        # Fingertip-to-wrist distances (5)
        for tip in TIPS:
            frame_features.append(np.linalg.norm(hand_normalized[tip]))

        # Fingertip-to-fingertip distances (10)
        for i in range(len(TIPS)):
            for j in range(i + 1, len(TIPS)):
                frame_features.append(np.linalg.norm(hand[TIPS[i]] - hand[TIPS[j]]))

        # Fingertip-to-MCP distances (finger curl) (5)
        for tip, mcp in zip(TIPS, MCPS):
            frame_features.append(np.linalg.norm(hand[tip] - hand[mcp]))

        # Max hand spread (1)
        max_spread = max(
            np.linalg.norm(hand[TIPS[i]] - hand[TIPS[j]])
            for i in range(len(TIPS))
            for j in range(i + 1, len(TIPS))
        )
        frame_features.append(max_spread)

        # Thumb-index angle (1)
        thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
        index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
        norm_product = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
        if norm_product > 1e-8:
            cos_angle = np.clip(np.dot(thumb_vec, index_vec) / norm_product, -1, 1)
        else:
            cos_angle = 0.0
        frame_features.append(cos_angle)

        # Palm normal vector (3)
        v1 = hand[INDEX_MCP] - hand[WRIST]
        v2 = hand[PINKY_MCP] - hand[WRIST]
        palm_normal = np.cross(v1, v2)
        norm = np.linalg.norm(palm_normal)
        if norm > 1e-8:
            palm_normal = palm_normal / norm
        else:
            palm_normal = np.zeros(3)
        frame_features.extend(palm_normal.tolist())

        features.append(frame_features)

    return np.array(features, dtype=np.float32)


def engineer_features_v8(data: np.ndarray) -> np.ndarray:
    """Engineer features for v8 model (649 features).

    Args:
        data: shape (num_frames, 549) - raw MediaPipe features

    Returns:
        shape (num_frames, 649)
    """
    left_hand = data[:, 99:162]
    right_hand = data[:, 162:225]

    left_features = compute_hand_features(left_hand)
    right_features = compute_hand_features(right_hand)

    # Velocity features
    left_vel = np.zeros_like(left_features)
    right_vel = np.zeros_like(right_features)
    left_vel[1:] = left_features[1:] - left_features[:-1]
    right_vel[1:] = right_features[1:] - right_features[:-1]

    return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)


def engineer_features_v9(data: np.ndarray) -> np.ndarray:
    """Engineer features for v9 model (657 features).

    Args:
        data: shape (num_frames, 549) - raw MediaPipe features

    Returns:
        shape (num_frames, 657)
    """
    # Start with v8 features
    base = engineer_features_v8(data)

    frames = data.shape[0]
    temporal = []

    # Duration feature
    temporal.append(frames / 90.0)

    # Segment similarity features
    segment_size = frames // 3
    if segment_size >= 5:
        segments = [data[i * segment_size:(i + 1) * segment_size].mean(axis=0) for i in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                norm_i = np.linalg.norm(segments[i])
                norm_j = np.linalg.norm(segments[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    temporal.append(np.dot(segments[i], segments[j]) / (norm_i * norm_j))
                else:
                    temporal.append(0.0)
    else:
        temporal.extend([0.0, 0.0, 0.0])

    # Velocity statistics
    right_hand = data[:, 162:225]
    velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
    temporal.append(velocities.mean() if len(velocities) > 0 else 0.0)
    temporal.append(velocities.std() if len(velocities) > 0 else 0.0)

    # Pause ratio
    if len(velocities) > 0:
        temporal.append(np.sum(velocities < velocities.mean() * 0.3) / len(velocities))
    else:
        temporal.append(0.0)

    # Repetition detection
    right_hand_r = right_hand.reshape(-1, 21, 3)
    spread = np.linalg.norm(right_hand_r[:, THUMB_TIP] - right_hand_r[:, PINKY_TIP], axis=1)
    kernel_size = 5
    if len(spread) >= kernel_size:
        spread_s = np.convolve(spread, np.ones(kernel_size) / kernel_size, mode='valid')
        peaks = sum(
            1 for i in range(1, len(spread_s) - 1)
            if spread_s[i] > spread_s[i - 1] and spread_s[i] > spread_s[i + 1]
            and spread_s[i] > np.mean(spread_s) * 0.8
        )
        temporal.append(peaks / 10.0)
    else:
        temporal.append(0.0)

    # Expand temporal features to all frames
    temporal_expanded = np.tile(np.array(temporal, dtype=np.float32), (frames, 1))

    return np.concatenate([base, temporal_expanded], axis=1)


# ============================================================================
# Preprocessing for ST-GCN models (v10+)
# ============================================================================

def preprocess_for_stgcn(data: np.ndarray, max_frames: int = 90) -> torch.Tensor:
    """Preprocess features for ST-GCN model (v10-v14).

    Args:
        data: shape (num_frames, 225+) - at least pose + hands
        max_frames: target number of frames

    Returns:
        torch.Tensor of shape (1, 3, max_frames, 48) ready for model
    """
    f = data.shape[0]

    # Extract skeletal data
    if data.shape[1] >= 225:
        pose = np.zeros((f, 6, 3), dtype=np.float32)
        for pi, idx_pose in enumerate(POSE_INDICES):
            start = idx_pose * 3
            pose[:, pi, :] = data[:, start:start + 3]

        lh = data[:, 99:162].reshape(f, 21, 3)
        rh = data[:, 162:225].reshape(f, 21, 3)
    else:
        pose = np.zeros((f, 6, 3), dtype=np.float32)
        lh = np.zeros((f, 21, 3), dtype=np.float32)
        rh = np.zeros((f, 21, 3), dtype=np.float32)

    # Stack: left hand (21) + right hand (21) + pose (6) = 48 nodes
    h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

    # Normalize hands relative to wrist
    lh_valid = np.abs(h[:, :21, :]).sum(axis=(1, 2)) > 0.01
    rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1, 2)) > 0.01

    if np.any(lh_valid):
        h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]  # Subtract left wrist
    if np.any(rh_valid):
        h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]  # Subtract right wrist

    # Normalize pose relative to mid-shoulder
    mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
    h[:, 42:48, :] -= mid_shoulder

    # Global normalization
    max_val = np.abs(h).max()
    if max_val > 0.01:
        h = np.clip(h / max_val, -1, 1).astype(np.float32)

    # Temporal alignment
    if f >= max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        h = h[indices]
    else:
        h = np.concatenate([h, np.zeros((max_frames - f, 48, 3), dtype=np.float32)])

    # Convert to (1, 3, T, 48) format
    return torch.FloatTensor(h).permute(2, 0, 1).unsqueeze(0)


def preprocess_for_temporal(data: np.ndarray, version: str = "v8",
                           max_frames: int = 90) -> torch.Tensor:
    """Preprocess features for TemporalPyramid model (v8/v9).

    Args:
        data: shape (num_frames, 549) - raw MediaPipe features
        version: "v8" or "v9"
        max_frames: target number of frames

    Returns:
        torch.Tensor of shape (1, max_frames, feature_dim) ready for model
    """
    # Engineer features
    if version == "v9":
        features = engineer_features_v9(data)
    else:
        features = engineer_features_v8(data)

    # Temporal alignment
    f = features.shape[0]
    if f > max_frames:
        indices = np.linspace(0, f - 1, max_frames, dtype=int)
        features = features[indices]
    else:
        padding = np.zeros((max_frames - f, features.shape[1]), dtype=np.float32)
        features = np.vstack([features, padding])

    return torch.FloatTensor(features).unsqueeze(0)


# ============================================================================
# Main Preprocessing Function
# ============================================================================

def preprocess_frames(base64_frames: List[str], version: str = "v14") -> Tuple[torch.Tensor, str]:
    """Main preprocessing pipeline.

    Args:
        base64_frames: List of base64 encoded image frames
        version: Model version (v8, v9, v10, v11, v12, v13, v14)

    Returns:
        Tuple of (preprocessed tensor, model_type)
    """
    # Decode frames
    frames = decode_base64_frames(base64_frames)

    if len(frames) < 10:
        raise ValueError(f"Too few frames ({len(frames)}). Need at least 10.")

    # Extract MediaPipe landmarks
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError("MediaPipe is required for preprocessing")

    with MediaPipeExtractor() as extractor:
        # v10+ only need pose + hands (225 features)
        # v8/v9 need face too (549 features)
        include_face = version.lower() in ["v8", "v9"]
        raw_features = extractor.extract_from_frames(frames, include_face=include_face)

    # For v8/v9, we need to add face features if using older data format
    if version.lower() in ["v8", "v9"] and raw_features.shape[1] < 549:
        # Pad to 549 features
        padding = np.zeros((raw_features.shape[0], 549 - raw_features.shape[1]), dtype=np.float32)
        raw_features = np.concatenate([raw_features, padding], axis=1)

    # Preprocess based on version
    version = version.lower()
    if version in ["v10", "v11", "v12", "v13", "v14"]:
        tensor = preprocess_for_stgcn(raw_features, max_frames=90)
        model_type = "stgcn"
    else:
        tensor = preprocess_for_temporal(raw_features, version=version, max_frames=90)
        model_type = "temporal"

    return tensor, model_type
