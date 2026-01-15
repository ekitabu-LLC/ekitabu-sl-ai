"""
KSL Test Video Inference
Tests videos from the testing folder using the best available model (v8).

Features extracted per frame (549 total):
- Pose: 33 x 3 = 99
- Left hand: 21 x 3 = 63
- Right hand: 21 x 3 = 63
- Face key points: 68 x 3 = 204
- Lips detail: 40 x 3 = 120

After hand feature engineering: 649 features

Usage:
    modal run test_videos.py
"""

import modal
import os
import re

app = modal.App("ksl-test-videos")
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

CONFIG_V8 = {"max_frames": 90, "feature_dim": 649, "hidden_dim": 320, "dropout": 0.35}

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


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def test_single_video(video_bytes: bytes, filename: str, expected_class: str) -> dict:
    """Process a single video and return prediction using v8 model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
    import mediapipe as mp

    # Extract features from video
    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )

    cap = cv2.VideoCapture(temp_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_row = []

        # 1. Pose landmarks (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0.0] * 99)

        # 2. Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0.0] * 63)

        # 3. Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0.0] * 63)

        # 4. Face key points (68 * 3 = 204)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in FACE_INDICES[:68]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_row.extend([lm.x, lm.y, lm.z])
                else:
                    frame_row.extend([0.0, 0.0, 0.0])
        else:
            frame_row.extend([0.0] * 204)

        # 5. Lips detail (40 * 3 = 120)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in LIPS_INDICES[:40]:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    frame_row.extend([lm.x, lm.y, lm.z])
                else:
                    frame_row.extend([0.0, 0.0, 0.0])
        else:
            frame_row.extend([0.0] * 120)

        frames_data.append(frame_row)

    cap.release()
    holistic.close()

    if len(frames_data) == 0:
        return {"filename": filename, "expected": expected_class, "predicted": "ERROR", "correct": False, "confidence": 0.0, "error": "No frames extracted"}

    data = np.array(frames_data, dtype=np.float32)

    # Verify we have 549 features
    if data.shape[1] != 549:
        return {"filename": filename, "expected": expected_class, "predicted": "ERROR", "correct": False, "confidence": 0.0, "error": f"Wrong feature count: {data.shape[1]} (expected 549)"}

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

    def engineer_features_v8(data):
        # data has 549 features: pose(99) + left_hand(63) + right_hand(63) + face(204) + lips(120)
        left_hand, right_hand = data[:, 99:162], data[:, 162:225]
        left_features, right_features = compute_hand_features(left_hand), compute_hand_features(right_hand)
        left_vel, right_vel = np.zeros_like(left_features), np.zeros_like(right_features)
        left_vel[1:] = left_features[1:] - left_features[:-1]
        right_vel[1:] = right_features[1:] - right_features[:-1]
        # 549 + 25 + 25 + 25 + 25 = 649
        return np.concatenate([data, left_features, right_features, left_vel, right_vel], axis=1)

    # Model architecture
    class TemporalPyramidV8(nn.Module):
        def __init__(self, num_classes, feature_dim=649, hidden_dim=320):
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.temporal_scales = nn.ModuleList([
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
                nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7), nn.BatchNorm1d(hidden_dim // 2), nn.GELU()),
            ])
            self.temporal_fusion = nn.Sequential(nn.Linear(hidden_dim // 2 * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(CONFIG_V8["dropout"]))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True, bidirectional=True, dropout=CONFIG_V8["dropout"])
            self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for _ in range(4)])
            self.pre_classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2 * 4), nn.Dropout(CONFIG_V8["dropout"]), nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2), nn.GELU(), nn.Dropout(CONFIG_V8["dropout"]))
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.anti_attractor = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))

        def forward(self, x):
            x = self.input_proj(self.input_norm(x))
            x_conv = x.transpose(1, 2)
            multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1).transpose(1, 2)
            x = self.temporal_fusion(multi_scale)
            lstm_out, _ = self.lstm(x)
            contexts = [((lstm_out * F.softmax(head(lstm_out), dim=1)).sum(dim=1)) for head in self.attention_heads]
            return self.classifier(self.pre_classifier(torch.cat(contexts, dim=-1)))

    # Load model
    checkpoint_path = "/data/checkpoints_v8/ksl_v8_best.pth"
    if not os.path.exists(checkpoint_path):
        return {"filename": filename, "expected": expected_class, "predicted": "ERROR", "correct": False, "confidence": 0.0, "error": f"Checkpoint not found: {checkpoint_path}"}

    model = TemporalPyramidV8(num_classes=len(ALL_CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.train(False)  # Set to inference mode

    def prepare_input(data):
        data = (data - data.mean()) / (data.std() + 1e-8)
        if data.shape[0] > 90:
            data = data[np.linspace(0, data.shape[0]-1, 90, dtype=int)]
        elif data.shape[0] < 90:
            data = np.vstack([data, np.zeros((90-data.shape[0], data.shape[1]), dtype=np.float32)])
        return torch.from_numpy(engineer_features_v8(data)).unsqueeze(0).cuda()

    # Run inference
    with torch.no_grad():
        logits = model(prepare_input(data.copy()))
        probs = F.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        predicted_class = ALL_CLASSES[pred_idx]
        confidence = probs[0, pred_idx].item()

        # Get top 3 predictions
        top3_probs, top3_idx = probs.topk(3, dim=1)
        top3 = [(ALL_CLASSES[idx], prob.item()) for idx, prob in zip(top3_idx[0].cpu().numpy(), top3_probs[0].cpu().numpy())]

    return {
        "filename": filename,
        "expected": expected_class,
        "predicted": predicted_class,
        "correct": predicted_class == expected_class,
        "confidence": confidence,
        "top3": top3,
        "frames": len(frames_data),
    }


@app.local_entrypoint()
def main():
    """Test all videos in the testing folder."""
    print("=" * 70)
    print("KSL Test Video Inference (v8 Model)")
    print("=" * 70)

    # Collect test videos
    test_videos = []
    testing_dir = "testing"

    # Numbers folder
    numbers_dir = os.path.join(testing_dir, "Numbers")
    if os.path.exists(numbers_dir):
        for video_file in os.listdir(numbers_dir):
            if video_file.endswith((".mov", ".mp4")):
                # Extract class from filename like "No. 9.mov" -> "9"
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
                # Extract class from filename like "1. Friend.mov" -> "Friend"
                match = re.search(r'\d+\.\s*(\w+)\.mov', video_file)
                if match:
                    expected_class = match.group(1)
                    video_path = os.path.join(words_dir, video_file)
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    test_videos.append((video_bytes, video_file, expected_class))

    print(f"\nFound {len(test_videos)} test videos")
    print("-" * 70)

    # Process videos
    results = []
    for result in test_single_video.starmap(test_videos):
        results.append(result)
        status = "CORRECT" if result["correct"] else "WRONG"
        print(f"{result['filename']:30s} | Expected: {result['expected']:12s} | Predicted: {result['predicted']:12s} | {status} | Conf: {result['confidence']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Per-category breakdown
    numbers_results = [r for r in results if r["expected"] in NUMBER_CLASSES]
    words_results = [r for r in results if r["expected"] in WORD_CLASSES]

    if numbers_results:
        num_correct = sum(1 for r in numbers_results if r["correct"])
        print(f"Numbers Accuracy: {num_correct}/{len(numbers_results)} ({num_correct/len(numbers_results)*100:.1f}%)")

    if words_results:
        word_correct = sum(1 for r in words_results if r["correct"])
        print(f"Words Accuracy: {word_correct}/{len(words_results)} ({word_correct/len(words_results)*100:.1f}%)")

    # Show wrong predictions
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nWrong predictions ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['filename']}: Expected {r['expected']}, got {r['predicted']} (conf: {r['confidence']:.2f})")
            if "top3" in r:
                print(f"    Top 3: {r['top3']}")

    print("=" * 70)
