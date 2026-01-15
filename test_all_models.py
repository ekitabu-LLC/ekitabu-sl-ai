"""
KSL Test All Models
Tests videos from the testing folder using ALL available model checkpoints.

Usage:
    modal run test_all_models.py
"""

import modal
import os
import re

app = modal.App("ksl-test-all-models")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("torch", "numpy", "mediapipe==0.10.14", "opencv-python-headless", "scikit-learn")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Face landmark indices for extraction (for v4+ models)
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


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=600, image=image)
def list_checkpoints() -> dict:
    """List all available checkpoint files on the volume."""
    import os
    checkpoints = {}

    # Check various checkpoint directories
    dirs_to_check = [
        "/data/checkpoints",
        "/data/checkpoints_v2",
        "/data/checkpoints_v3",
        "/data/checkpoints_v4",
        "/data/checkpoints_v5",
        "/data/checkpoints_v6",
        "/data/checkpoints_v7",
        "/data/checkpoints_v8",
        "/data/checkpoints_v9",
    ]

    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            pth_files = [f for f in files if f.endswith('.pth')]
            if pth_files:
                checkpoints[dir_path] = pth_files

    # Also check root /data for any .pth files
    if os.path.exists("/data"):
        root_pth = [f for f in os.listdir("/data") if f.endswith('.pth')]
        if root_pth:
            checkpoints["/data"] = root_pth

    return checkpoints


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=7200, image=image)
def test_with_model(video_data_list: list, model_version: str, checkpoint_path: str) -> dict:
    """Test videos with a specific model version."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
    import mediapipe as mp

    results = []

    # Determine model configuration based on version
    if model_version in ["v1", "v2"]:
        feature_dim = 225
        max_frames = 120
        use_face = False
    elif model_version == "v3":
        feature_dim = 801  # 225 * 4 (original + velocity + acceleration + jerk)
        max_frames = 90
        use_face = False
    elif model_version == "v4":
        feature_dim = 549
        max_frames = 90
        use_face = True
    elif model_version in ["v5", "v6", "v7", "v8"]:
        feature_dim = 649
        max_frames = 90
        use_face = True
    elif model_version == "v9":
        feature_dim = 657
        max_frames = 90
        use_face = True
    else:
        feature_dim = 649
        max_frames = 90
        use_face = True

    # Hand feature computation
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
    THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17

    def compute_hand_features(hand_landmarks):
        frames = hand_landmarks.shape[0]
        features = []
        for f in range(frames):
            hand = hand_landmarks[f].reshape(21, 3)
            frame_features = []
            wrist = hand[WRIST]
            hand_normalized = hand - wrist
            tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
            for tip in tips:
                frame_features.append(np.linalg.norm(hand_normalized[tip]))
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    frame_features.append(np.linalg.norm(hand[tips[i]] - hand[tips[j]]))
            for tip, mcp in zip(tips, mcps):
                frame_features.append(np.linalg.norm(hand[tip] - hand[mcp]))
            max_spread = max(np.linalg.norm(hand[tips[i]] - hand[tips[j]])
                           for i in range(len(tips)) for j in range(i+1, len(tips)))
            frame_features.append(max_spread)
            thumb_vec = hand[THUMB_TIP] - hand[THUMB_MCP]
            index_vec = hand[INDEX_TIP] - hand[INDEX_MCP]
            norm_product = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
            cos_angle = np.clip(np.dot(thumb_vec, index_vec) / norm_product, -1, 1) if norm_product > 1e-8 else 0
            frame_features.append(cos_angle)
            v1 = hand[INDEX_MCP] - hand[WRIST]
            v2 = hand[PINKY_MCP] - hand[WRIST]
            palm_normal = np.cross(v1, v2)
            norm = np.linalg.norm(palm_normal)
            palm_normal = palm_normal / norm if norm > 1e-8 else np.zeros(3)
            frame_features.extend(palm_normal.tolist())
            features.append(frame_features)
        return np.array(features, dtype=np.float32)

    def engineer_features(data, version):
        """Engineer features based on model version."""
        if version in ["v1", "v2"]:
            return data  # No additional features
        elif version == "v3":
            # Add velocity, acceleration, jerk
            velocity = np.zeros_like(data)
            velocity[1:] = data[1:] - data[:-1]
            acceleration = np.zeros_like(velocity)
            acceleration[1:] = velocity[1:] - velocity[:-1]
            jerk = np.zeros_like(acceleration)
            jerk[1:] = acceleration[1:] - acceleration[:-1]
            return np.concatenate([data, velocity, acceleration, jerk], axis=1)
        elif version == "v4":
            return data  # Just base 549 features
        elif version in ["v5", "v6", "v7", "v8"]:
            # Add hand features
            left_hand, right_hand = data[:, 99:162], data[:, 162:225]
            left_features = compute_hand_features(left_hand)
            right_features = compute_hand_features(right_hand)
            left_vel = np.zeros_like(left_features)
            right_vel = np.zeros_like(right_features)
            left_vel[1:] = left_features[1:] - left_features[:-1]
            right_vel[1:] = right_features[1:] - right_features[:-1]
            return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)
        elif version == "v9":
            # v8 features + temporal features
            base = engineer_features(data, "v8")
            # Add temporal features
            frames = data.shape[0]
            temporal = []
            temporal.append(frames / 90.0)  # duration
            segment_size = frames // 3
            if segment_size >= 5:
                segments = [data[i*segment_size:(i+1)*segment_size].mean(axis=0) for i in range(3)]
                for i in range(3):
                    for j in range(i+1, 3):
                        norm_i, norm_j = np.linalg.norm(segments[i]), np.linalg.norm(segments[j])
                        if norm_i > 1e-8 and norm_j > 1e-8:
                            temporal.append(np.dot(segments[i], segments[j]) / (norm_i * norm_j))
                        else:
                            temporal.append(0.0)
            else:
                temporal.extend([0.0, 0.0, 0.0])
            right_hand = data[:, 162:225]
            velocities = np.linalg.norm(np.diff(right_hand, axis=0), axis=1)
            temporal.append(velocities.mean() if len(velocities) > 0 else 0.0)
            temporal.append(velocities.std() if len(velocities) > 0 else 0.0)
            temporal.append(np.sum(velocities < velocities.mean() * 0.3) / len(velocities) if len(velocities) > 0 else 0.0)
            right_hand_reshaped = right_hand.reshape(-1, 21, 3)
            spread = np.linalg.norm(right_hand_reshaped[:, 4] - right_hand_reshaped[:, 20], axis=1)
            kernel_size = 5
            if len(spread) >= kernel_size:
                spread_smooth = np.convolve(spread, np.ones(kernel_size)/kernel_size, mode='valid')
                peaks = sum(1 for i in range(1, len(spread_smooth)-1)
                           if spread_smooth[i] > spread_smooth[i-1] and spread_smooth[i] > spread_smooth[i+1]
                           and spread_smooth[i] > np.mean(spread_smooth) * 0.8)
                temporal.append(peaks / 10.0)
            else:
                temporal.append(0.0)
            temporal_expanded = np.tile(np.array(temporal, dtype=np.float32), (base.shape[0], 1))
            return np.concatenate([base, temporal_expanded], axis=1)
        return data

    # Model architectures
    class SimpleLSTM(nn.Module):
        """Simple LSTM for v1/v2."""
        def __init__(self, num_classes, feature_dim, hidden_dim=256):
            super().__init__()
            self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    class TemporalPyramid(nn.Module):
        """TemporalPyramid for v5+."""
        def __init__(self, num_classes, feature_dim, hidden_dim=320, dropout=0.35):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            ])
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim // 2 * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=dropout)
            self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)])
            self.pre_classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2 * 4), nn.Dropout(dropout), nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            # Anti-attractor layer for v8/v9
            self.anti_attractor = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))

        def forward(self, x):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [((lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1)) for head in self.attention_heads]
            return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2 if use_face else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=use_face
    )

    # Load model
    if not os.path.exists(checkpoint_path):
        return {"error": f"Checkpoint not found: {checkpoint_path}", "results": [], "version": model_version}

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Determine model class
        if model_version in ["v1", "v2"]:
            model = SimpleLSTM(num_classes=len(ALL_CLASSES), feature_dim=feature_dim)
        else:
            model = TemporalPyramid(num_classes=len(ALL_CLASSES), feature_dim=feature_dim)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda()
        model.train(False)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}", "results": [], "version": model_version}

    # Process each video
    for video_bytes, filename, expected_class in video_data_list:
        try:
            # Save video temporarily
            temp_path = f"/tmp/{filename}"
            with open(temp_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(temp_path)
            frames_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = holistic.process(rgb_frame)
                frame_row = []

                # Pose (99)
                if mp_results.pose_landmarks:
                    for lm in mp_results.pose_landmarks.landmark:
                        frame_row.extend([lm.x, lm.y, lm.z])
                else:
                    frame_row.extend([0.0] * 99)

                # Left hand (63)
                if mp_results.left_hand_landmarks:
                    for lm in mp_results.left_hand_landmarks.landmark:
                        frame_row.extend([lm.x, lm.y, lm.z])
                else:
                    frame_row.extend([0.0] * 63)

                # Right hand (63)
                if mp_results.right_hand_landmarks:
                    for lm in mp_results.right_hand_landmarks.landmark:
                        frame_row.extend([lm.x, lm.y, lm.z])
                else:
                    frame_row.extend([0.0] * 63)

                # Face features (for v4+)
                if use_face:
                    if mp_results.face_landmarks:
                        face_lms = mp_results.face_landmarks.landmark
                        for idx in FACE_INDICES[:68]:
                            if idx < len(face_lms):
                                lm = face_lms[idx]
                                frame_row.extend([lm.x, lm.y, lm.z])
                            else:
                                frame_row.extend([0.0, 0.0, 0.0])
                        for idx in LIPS_INDICES[:40]:
                            if idx < len(face_lms):
                                lm = face_lms[idx]
                                frame_row.extend([lm.x, lm.y, lm.z])
                            else:
                                frame_row.extend([0.0, 0.0, 0.0])
                    else:
                        frame_row.extend([0.0] * 324)  # 68*3 + 40*3

                frames_data.append(frame_row)

            cap.release()

            if len(frames_data) == 0:
                results.append({"filename": filename, "expected": expected_class, "predicted": "ERROR", "correct": False, "confidence": 0.0})
                continue

            data = np.array(frames_data, dtype=np.float32)

            # Normalize
            data = (data - data.mean()) / (data.std() + 1e-8)

            # Pad/truncate to max_frames
            if data.shape[0] > max_frames:
                data = data[np.linspace(0, data.shape[0]-1, max_frames, dtype=int)]
            elif data.shape[0] < max_frames:
                data = np.vstack([data, np.zeros((max_frames-data.shape[0], data.shape[1]), dtype=np.float32)])

            # Engineer features
            engineered = engineer_features(data, model_version)

            # Inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(engineered).unsqueeze(0).cuda()
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                predicted_class = ALL_CLASSES[pred_idx]
                confidence = probs[0, pred_idx].item()

            results.append({
                "filename": filename,
                "expected": expected_class,
                "predicted": predicted_class,
                "correct": predicted_class == expected_class,
                "confidence": confidence,
            })

        except Exception as e:
            results.append({"filename": filename, "expected": expected_class, "predicted": "ERROR", "correct": False, "confidence": 0.0, "error": str(e)})

    holistic.close()

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    return {
        "version": model_version,
        "checkpoint": checkpoint_path,
        "accuracy": correct / total * 100 if total > 0 else 0,
        "correct": correct,
        "total": total,
        "results": results,
    }


@app.local_entrypoint()
def main():
    """Test all available models on test videos."""
    print("=" * 70)
    print("KSL Multi-Model Test")
    print("=" * 70)

    # First, list available checkpoints
    print("\nScanning for available checkpoints...")
    checkpoints = list_checkpoints.remote()
    print(f"\nFound checkpoints:")
    for dir_path, files in checkpoints.items():
        print(f"  {dir_path}: {files}")

    # Collect test videos
    test_videos = []
    testing_dir = "testing"

    # Numbers folder
    numbers_dir = os.path.join(testing_dir, "Numbers")
    if os.path.exists(numbers_dir):
        for video_file in os.listdir(numbers_dir):
            if video_file.endswith((".mov", ".mp4")):
                match = re.search(r'No\.\s*(\d+)', video_file)
                if match:
                    expected_class = match.group(1)
                    video_path = os.path.join(numbers_dir, video_file)
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    test_videos.append((video_bytes, video_file, expected_class))

    # Words folder
    words_dir = os.path.join(testing_dir, "Words - Left Side")
    if os.path.exists(words_dir):
        for video_file in os.listdir(words_dir):
            if video_file.endswith((".mov", ".mp4")):
                match = re.search(r'\d+\.\s*(\w+)\.mov', video_file)
                if match:
                    expected_class = match.group(1)
                    video_path = os.path.join(words_dir, video_file)
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    test_videos.append((video_bytes, video_file, expected_class))

    print(f"\nFound {len(test_videos)} test videos")

    # Define model versions to test based on available checkpoints
    models_to_test = []

    # Map checkpoint directories to versions
    version_map = {
        "/data/checkpoints": "v1",
        "/data/checkpoints_v2": "v2",
        "/data/checkpoints_v3": "v3",
        "/data/checkpoints_v4": "v4",
        "/data/checkpoints_v5": "v5",
        "/data/checkpoints_v6": "v6",
        "/data/checkpoints_v7": "v7",
        "/data/checkpoints_v8": "v8",
        "/data/checkpoints_v9": "v9",
    }

    for dir_path, files in checkpoints.items():
        version = version_map.get(dir_path, "unknown")
        # Find best checkpoint
        best_file = None
        for f in files:
            if "best" in f.lower():
                best_file = f
                break
        if not best_file:
            best_file = files[0]  # Use first available

        models_to_test.append((version, os.path.join(dir_path, best_file)))

    print(f"\nModels to test: {len(models_to_test)}")
    for version, path in models_to_test:
        print(f"  {version}: {path}")

    # Test each model
    print("\n" + "=" * 70)
    print("TESTING ALL MODELS")
    print("=" * 70)

    all_results = []
    for version, checkpoint_path in models_to_test:
        print(f"\nTesting {version}...")
        result = test_with_model.remote(test_videos, version, checkpoint_path)
        all_results.append(result)

        if "error" in result and result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Accuracy: {result['correct']}/{result['total']} ({result['accuracy']:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL MODELS")
    print("=" * 70)
    print(f"{'Model':<10} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 40)

    for result in sorted(all_results, key=lambda x: x.get('accuracy', 0), reverse=True):
        if "error" in result and result["error"]:
            print(f"{result['version']:<10} ERROR: {result['error'][:30]}")
        else:
            print(f"{result['version']:<10} {result['accuracy']:.1f}%{'':<9} {result['correct']}/{result['total']}")

    # Best model detailed results
    best_result = max([r for r in all_results if not r.get('error')], key=lambda x: x['accuracy'], default=None)
    if best_result:
        print(f"\n{'=' * 70}")
        print(f"BEST MODEL: {best_result['version']} ({best_result['accuracy']:.1f}%)")
        print("=" * 70)

        print("\nCorrect predictions:")
        for r in best_result['results']:
            if r['correct']:
                print(f"  {r['filename']}: {r['expected']} (conf: {r['confidence']:.2f})")

        print("\nWrong predictions:")
        for r in best_result['results']:
            if not r['correct']:
                print(f"  {r['filename']}: Expected {r['expected']}, got {r['predicted']} (conf: {r['confidence']:.2f})")

    print("=" * 70)
