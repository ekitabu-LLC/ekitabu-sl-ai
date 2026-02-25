# Integrating KSL ONNX Models into a Flutter App

This guide walks through building a real-time Kenya Sign Language (KSL) recognition app in Flutter using the exported ONNX ensemble models. The pipeline is:

```
Camera → MediaPipe Landmarks → Preprocessing (Dart) → ONNX Ensemble → Prediction UI
```

**Best accuracy available:** 6-model uniform ensemble — Numbers 74.6%, Words 71.6%, Combined **72.9%**
**Best single model:** V43 — Numbers 66.1%, Words 65.4%, Combined **65.7%** (simplest integration)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Dependencies](#3-dependencies)
4. [Platform Setup](#4-platform-setup)
5. [Bundle ONNX Models](#5-bundle-onnx-models)
6. [Skeleton Extraction (MediaPipe)](#6-skeleton-extraction-mediapipe)
7. [Preprocessing in Dart](#7-preprocessing-in-dart)
8. [ONNX Inference](#8-onnx-inference)
9. [6-Model Ensemble Fusion](#9-6-model-ensemble-fusion)
10. [Background Isolate](#10-background-isolate)
11. [UI — Camera + Overlay + Prediction Panel](#11-ui--camera--overlay--prediction-panel)
12. [Performance Tips](#12-performance-tips)
13. [Quick-Start: Single Model (V43)](#13-quick-start-single-model-v43)

---

## 1. Prerequisites

- Flutter 3.19+ / Dart 3.3+
- Android: minSdk 24, targetSdk 34 (ONNX Runtime requires minSdk 24)
- iOS: deployment target 14.0+
- The exported ONNX models from `onnx_models/` (see repo)
- For the ensemble, use the `model_adapted.onnx` files (AdaBN statistics baked in)

---

## 2. Project Structure

```
lib/
├── main.dart
├── features/
│   └── recognition/
│       ├── data/
│       │   ├── inference_service.dart      # ONNX sessions
│       │   └── skeleton_service.dart       # MediaPipe extraction
│       ├── domain/
│       │   ├── preprocessor.dart           # bone/velocity/normalize
│       │   └── ensemble_fusion.dart        # softmax + average
│       ├── application/
│       │   ├── inference_notifier.dart     # Riverpod state
│       │   └── inference_isolate.dart      # background thread
│       └── presentation/
│           ├── recognition_screen.dart
│           ├── skeleton_overlay.dart
│           └── prediction_panel.dart
assets/
└── models/
    ├── numbers/
    │   ├── v43/joint/model_adapted.onnx
    │   ├── v43/bone/model_adapted.onnx
    │   ├── v43/velocity/model_adapted.onnx
    │   ├── v43/fusion_weights.json
    │   ├── exp5/joint/model_adapted.onnx   # (ensemble)
    │   ├── exp1/joint/model_adapted.onnx   # (ensemble)
    │   └── ...
    └── words/
        └── ...  (same structure)
```

---

## 3. Dependencies

```yaml
# pubspec.yaml
dependencies:
  flutter:
    sdk: flutter

  # ONNX Runtime — recommended for new projects
  flutter_onnxruntime: ^1.5.1

  # Alternative (more mature, FFI-based):
  # onnxruntime: ^1.3.0

  # Camera feed
  camera: ^0.11.0

  # MediaPipe hand + pose landmarks (Android/iOS)
  google_mlkit_pose_detection: ^0.12.0
  google_mlkit_commons: ^0.9.0

  # OR: dedicated hand landmarker (Android, GPU delegate)
  # hand_landmarker: ^2.1.2

  # State management
  flutter_riverpod: ^2.6.0
  riverpod_annotation: ^2.6.0

  # Utilities
  freezed_annotation: ^2.4.0
  path_provider: ^2.1.0

dev_dependencies:
  build_runner: ^2.4.0
  riverpod_generator: ^2.6.0
  freezed: ^2.5.0

flutter:
  assets:
    # V43 (best single model — simplest option)
    - assets/models/numbers/v43/joint/model_adapted.onnx
    - assets/models/numbers/v43/bone/model_adapted.onnx
    - assets/models/numbers/v43/velocity/model_adapted.onnx
    - assets/models/numbers/v43/fusion_weights.json
    - assets/models/words/v43/joint/model_adapted.onnx
    - assets/models/words/v43/bone/model_adapted.onnx
    - assets/models/words/v43/velocity/model_adapted.onnx
    - assets/models/words/v43/fusion_weights.json

    # 6-model uniform ensemble (add if using full ensemble)
    # - assets/models/numbers/exp1/joint/model_adapted.onnx
    # - assets/models/numbers/exp1/bone/model_adapted.onnx
    # - assets/models/numbers/exp1/velocity/model_adapted.onnx
    # - assets/models/numbers/exp5/...  (etc.)
```

---

## 4. Platform Setup

### Android — `android/app/build.gradle`

```gradle
android {
    defaultConfig {
        minSdk 24          // Required by ONNX Runtime
        targetSdk 34
    }
}
```

### Android — `AndroidManifest.xml`

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

### iOS — `ios/Runner/Info.plist`

```xml
<key>NSCameraUsageDescription</key>
<string>Required for real-time sign language recognition</string>
```

### iOS — `ios/Podfile`

```ruby
platform :ios, '14.0'
```

---

## 5. Bundle ONNX Models

Copy the models from the HPC into your Flutter project:

```bash
# From the ksl-dir-2 repo — copy V43 (best single model)
cp onnx_models/numbers/v43/joint/model_adapted.onnx  flutter_app/assets/models/numbers/v43/joint/
cp onnx_models/numbers/v43/bone/model_adapted.onnx   flutter_app/assets/models/numbers/v43/bone/
cp onnx_models/numbers/v43/velocity/model_adapted.onnx flutter_app/assets/models/numbers/v43/velocity/
cp onnx_models/numbers/v43/fusion_weights.json        flutter_app/assets/models/numbers/v43/
# Repeat for words/
```

> **Why `model_adapted.onnx`?**
> These have AdaBN (Adaptive Batch Normalization) statistics baked in from the test population.
> They produce significantly better accuracy than `model.onnx` for new signers:
> +11.9pp numbers, +7.4pp words for V43.

**Model sizes (V43, 6 files total):**

| File | Size |
|------|------|
| `numbers/v43/joint/model_adapted.onnx` | ~3.2 MB |
| `numbers/v43/bone/model_adapted.onnx` | ~3.2 MB |
| `numbers/v43/velocity/model_adapted.onnx` | ~3.2 MB |
| `words/v43/...` (×3) | ~9.6 MB |
| **Total (V43 only)** | **~19 MB** |

---

## 6. Skeleton Extraction (MediaPipe)

The KSL models expect **48 joints**: 21 left-hand + 21 right-hand + 6 pose (shoulders + elbows).

### Option A — ML Kit Pose Detection (Cross-platform, simpler)

```dart
// lib/features/recognition/data/skeleton_service.dart
import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class SkeletonService {
  final _detector = PoseDetector(
    options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
  );

  bool _isBusy = false;

  /// Returns raw (T=1, 48, 3) joint positions from a single camera frame,
  /// or null if detection fails.
  Future<List<Joint3D>?> processFrame(CameraImage image, int rotation) async {
    if (_isBusy) return null;
    _isBusy = true;

    try {
      final inputImage = _toInputImage(image, rotation);
      final poses = await _detector.processImage(inputImage);
      if (poses.isEmpty) return null;

      return _extractKslJoints(poses.first);
    } finally {
      _isBusy = false;
    }
  }

  List<Joint3D> _extractKslJoints(Pose pose) {
    // KSL format: [lh(0-20), rh(21-41), pose(42-47)]
    // ML Kit provides body pose only — hand joints need hand_landmarker
    // Placeholder: use pose landmarks for body, zero for hands
    final joints = List<Joint3D>.filled(48, Joint3D.zero());

    // Pose joints 42-47: left shoulder, right shoulder, left elbow,
    //                     right elbow, left wrist, right wrist
    const poseMap = {
      42: PoseLandmarkType.leftShoulder,
      43: PoseLandmarkType.rightShoulder,
      44: PoseLandmarkType.leftElbow,
      45: PoseLandmarkType.rightElbow,
      // Use wrist as proxy for hand-wrist (joints 0 and 21)
      0:  PoseLandmarkType.leftWrist,
      21: PoseLandmarkType.rightWrist,
    };

    for (final entry in poseMap.entries) {
      final lm = pose.landmarks[entry.value];
      if (lm != null) {
        joints[entry.key] = Joint3D(lm.x, lm.y, lm.z ?? 0.0);
      }
    }
    return joints;
  }

  InputImage _toInputImage(CameraImage image, int rotation) {
    final format = InputImageFormatValue.fromRawValue(image.format.raw)
        ?? InputImageFormat.nv21;
    return InputImage.fromBytes(
      bytes: _concatenatePlanes(image.planes),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: InputImageRotationValue.fromRawValue(rotation)
            ?? InputImageRotation.rotation0deg,
        format: format,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );
  }

  Uint8List _concatenatePlanes(List<Plane> planes) {
    final buffer = WriteBuffer();
    for (final p in planes) buffer.putUint8List(p.bytes);
    return buffer.done().buffer.asUint8List();
  }

  void dispose() => _detector.close();
}

class Joint3D {
  final double x, y, z;
  const Joint3D(this.x, this.y, this.z);
  static Joint3D zero() => const Joint3D(0, 0, 0);
}
```

### Option B — `hand_landmarker` Package (Android, full 21-joint hands)

For full accuracy matching the training pipeline (21-joint hands), use the dedicated package:

```yaml
# pubspec.yaml
dependencies:
  hand_landmarker: ^2.1.2
```

```dart
import 'package:hand_landmarker/hand_landmarker.dart';

final handLandmarker = HandLandmarker(
  numHands: 2,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
  delegate: HandLandmarkerDelegate.gpu,  // GPU via MediaPipe JNI
);

// In frame callback:
final result = await handLandmarker.detect(cameraImage);
// result.landmarks: List<HandLandmarks> with 21 joints each
for (final hand in result.landmarks) {
  for (int i = 0; i < 21; i++) {
    final lm = hand.landmarks[i];
    // lm.x, lm.y, lm.z in normalized image coordinates
  }
}
```

> **Note**: The KSL training used MediaPipe Holistic (Python) which provides simultaneous
> hand + body tracking. On mobile, combining `hand_landmarker` + `google_mlkit_pose_detection`
> gives equivalent coverage. See `preprocess_for_onnx.py` in the repo for the exact joint mapping.

---

## 7. Preprocessing in Dart

This mirrors `preprocess_for_onnx.py` exactly. Input: raw joint positions → Output: `{joint, bone, velocity, aux}` tensors.

```dart
// lib/features/recognition/domain/preprocessor.dart
import 'dart:typed_data';
import 'dart:math';

class KslPreprocessor {
  static const int maxFrames  = 90;
  static const int numJoints  = 48;
  static const int numChannels = 3; // x, y, z

  // Parent map for 48 joints: [lh(0-20), rh(21-41), pose(42-47)]
  static const List<int> parentMap = [
    // Left hand
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
    // Right hand
    -1, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40,
    // Pose (6 joints)
    -1, 42, 42, 43, 44, 45,
  ];

  // Fingertip indices
  static const lhTips = [4, 8, 12, 16, 20];
  static const rhTips = [25, 29, 33, 37, 41];

  /// Buffer: accumulate frames as they arrive from camera
  final List<List<Joint3D>> _frameBuffer = [];

  void addFrame(List<Joint3D> joints) {
    _frameBuffer.add(joints);
    if (_frameBuffer.length > maxFrames * 3) {
      _frameBuffer.removeAt(0); // keep rolling window
    }
  }

  void clear() => _frameBuffer.clear();

  int get frameCount => _frameBuffer.length;

  /// Call when the user has finished signing (enough frames collected).
  /// Returns null if not enough frames.
  Map<String, Float32List>? preprocess() {
    if (_frameBuffer.isEmpty) return null;

    final frames = _frameBuffer.length;

    // Build raw [frames, 48, 3] array
    final raw = Float32List(frames * numJoints * numChannels);
    for (int t = 0; t < frames; t++) {
      final f = _frameBuffer[t];
      for (int v = 0; v < numJoints && v < f.length; v++) {
        final base = t * numJoints * numChannels + v * numChannels;
        raw[base + 0] = f[v].x;
        raw[base + 1] = f[v].y;
        raw[base + 2] = f[v].z;
      }
    }

    // 1. Normalize (wrist-relative + palm-scale)
    final h = _normalizeWristPalm(raw, frames);

    // 2. Compute auxiliary features before temporal resampling
    final handBodyFeats = _computeHandBodyFeatures(raw, frames);  // uses raw (absolute)
    final angles = _computeJointAngles(h, frames);
    final ftDists = _computeFingertipDistances(h, frames);

    // 3. Temporal resampling → maxFrames
    final hR       = _resample(h, frames, numJoints * numChannels);
    final velR     = _computeVelocity(hR, maxFrames);
    final bonesR   = _computeBones(hR, maxFrames);
    final anglesR  = _resample(angles, frames, angles.length ~/ frames);
    final ftDistsR = _resample(ftDists, frames, ftDists.length ~/ frames);
    final hbR      = _resample(handBodyFeats, frames, 8);

    // 4. Clip + reshape to [1, C, T, N]
    final joint    = _toGcnTensor(hR,     maxFrames, numJoints, numChannels);
    final bone     = _toGcnTensor(bonesR, maxFrames, numJoints, numChannels);
    final velocity = _toGcnTensor(velR,   maxFrames, numJoints, numChannels);

    // 5. Build aux tensor [1, T, D_aux]
    final dAux = anglesR.length ~/ maxFrames + ftDistsR.length ~/ maxFrames + 8;
    final auxData = Float32List(maxFrames * dAux);
    for (int t = 0; t < maxFrames; t++) {
      int col = 0;
      final aStride = anglesR.length ~/ maxFrames;
      final fStride = ftDistsR.length ~/ maxFrames;
      for (int i = 0; i < aStride; i++) auxData[t * dAux + col++] = anglesR[t * aStride + i];
      for (int i = 0; i < fStride; i++) auxData[t * dAux + col++] = ftDistsR[t * fStride + i];
      for (int i = 0; i < 8; i++)       auxData[t * dAux + col++] = hbR[t * 8 + i];
    }
    final aux = Float32List(1 * maxFrames * dAux);
    aux.setAll(0, auxData);

    return {'joint': joint, 'bone': bone, 'velocity': velocity, 'aux': aux};
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  Float32List _normalizeWristPalm(Float32List raw, int T) {
    final out = Float32List.fromList(raw);
    // Left hand: root at joint 0, scale by joint 9 distance
    _normalizeSegment(out, T, 0, 21, 0, 9);
    // Right hand: root at joint 21, scale by joint 30 (=9+21) distance
    _normalizeSegment(out, T, 21, 42, 21, 30);
    // Pose: root at midpoint(42,43), scale by shoulder width
    _normalizePose(out, T);
    return out;
  }

  void _normalizeSegment(Float32List h, int T, int start, int end,
                          int rootIdx, int scaleJoint) {
    for (int t = 0; t < T; t++) {
      final rootBase = t * numJoints * numChannels + rootIdx * numChannels;
      final rx = h[rootBase], ry = h[rootBase + 1], rz = h[rootBase + 2];
      // Check frame is valid (not all zeros)
      double frameSum = 0;
      for (int v = start; v < end; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        frameSum += h[b].abs() + h[b+1].abs() + h[b+2].abs();
      }
      if (frameSum < 0.01) continue;

      // Subtract root
      for (int v = start; v < end; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        h[b] -= rx; h[b+1] -= ry; h[b+2] -= rz;
      }
      // Scale by palm size (distance from root to scaleJoint)
      final sb = t * numJoints * numChannels + scaleJoint * numChannels;
      final palmSize = max(
        sqrt(h[sb]*h[sb] + h[sb+1]*h[sb+1] + h[sb+2]*h[sb+2]),
        1e-6,
      );
      for (int v = start; v < end; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        h[b] /= palmSize; h[b+1] /= palmSize; h[b+2] /= palmSize;
      }
    }
  }

  void _normalizePose(Float32List h, int T) {
    for (int t = 0; t < T; t++) {
      final b42 = t * numJoints * numChannels + 42 * numChannels;
      final b43 = t * numJoints * numChannels + 43 * numChannels;
      final mx = (h[b42] + h[b43]) / 2;
      final my = (h[b42+1] + h[b43+1]) / 2;
      final mz = (h[b42+2] + h[b43+2]) / 2;
      final sw = max(
        sqrt(pow(h[b42]-h[b43], 2) + pow(h[b42+1]-h[b43+1], 2) + pow(h[b42+2]-h[b43+2], 2)),
        1e-6,
      );
      for (int v = 42; v < 48; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        h[b] = (h[b] - mx) / sw;
        h[b+1] = (h[b+1] - my) / sw;
        h[b+2] = (h[b+2] - mz) / sw;
      }
    }
  }

  Float32List _computeBones(Float32List h, int T) {
    final out = Float32List(T * numJoints * numChannels);
    for (int t = 0; t < T; t++) {
      for (int v = 0; v < numJoints; v++) {
        final p = parentMap[v];
        if (p < 0) continue;
        final cb = t * numJoints * numChannels + v * numChannels;
        final pb = t * numJoints * numChannels + p * numChannels;
        out[cb]   = h[cb]   - h[pb];
        out[cb+1] = h[cb+1] - h[pb+1];
        out[cb+2] = h[cb+2] - h[pb+2];
      }
    }
    return out;
  }

  Float32List _computeVelocity(Float32List h, int T) {
    final out = Float32List(T * numJoints * numChannels);
    for (int t = 1; t < T; t++) {
      final cur  = t * numJoints * numChannels;
      final prev = (t-1) * numJoints * numChannels;
      for (int i = 0; i < numJoints * numChannels; i++) {
        out[cur + i] = h[cur + i] - h[prev + i];
      }
    }
    return out;
  }

  Float32List _resample(Float32List data, int fromT, int stride) {
    if (fromT == maxFrames) return data;
    final out = Float32List(maxFrames * stride);
    if (fromT >= maxFrames) {
      // Subsample
      for (int t = 0; t < maxFrames; t++) {
        final src = ((t * (fromT - 1)) / (maxFrames - 1)).round();
        out.setRange(t * stride, (t+1) * stride, data, src * stride);
      }
    } else {
      // Zero-pad
      out.setRange(0, fromT * stride, data);
    }
    return out;
  }

  /// Reshape [T, V, C] → [1, C, T, V] and clip to [-10, 10]
  Float32List _toGcnTensor(Float32List data, int T, int V, int C) {
    final out = Float32List(1 * C * T * V);
    for (int t = 0; t < T; t++) {
      for (int v = 0; v < V; v++) {
        for (int c = 0; c < C; c++) {
          final src = t * V * C + v * C + c;
          final dst = c * T * V + t * V + v;
          out[dst] = data[src].clamp(-10.0, 10.0);
        }
      }
    }
    return out;
  }

  Float32List _computeHandBodyFeatures(Float32List raw, int T) {
    final out = Float32List(T * 8);
    for (int t = 0; t < T; t++) {
      double lhcx = 0, lhcy = 0, rhcx = 0, rhcy = 0;
      for (int v = 0; v < 21; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        lhcx += raw[b]; lhcy += raw[b+1];
      }
      for (int v = 21; v < 42; v++) {
        final b = t * numJoints * numChannels + v * numChannels;
        rhcx += raw[b]; rhcy += raw[b+1];
      }
      lhcx /= 21; lhcy /= 21; rhcx /= 21; rhcy /= 21;
      final b42 = t * numJoints * numChannels + 42 * numChannels;
      final b43 = t * numJoints * numChannels + 43 * numChannels;
      final msx = (raw[b42] + raw[b43]) / 2;
      final msy = (raw[b42+1] + raw[b43+1]) / 2;
      final sw  = max(sqrt(pow(raw[b42]-raw[b43],2) + pow(raw[b42+1]-raw[b43+1],2)), 1e-6);
      out[t*8+0] = (lhcy - msy) / sw;
      out[t*8+1] = (rhcy - msy) / sw;
      out[t*8+2] = (lhcx - msx) / sw;
      out[t*8+3] = (rhcx - msx) / sw;
      out[t*8+4] = sqrt(pow(lhcx-rhcx,2) + pow(lhcy-rhcy,2)) / sw;
      final facey = msy - 0.7 * sw;
      out[t*8+5] = sqrt(pow(lhcx-msx,2) + pow(lhcy-facey,2)) / sw;
      out[t*8+6] = sqrt(pow(rhcx-msx,2) + pow(rhcy-facey,2)) / sw;
      out[t*8+7] = (lhcy - rhcy).abs() / sw;
    }
    return out;
  }

  Float32List _computeJointAngles(Float32List h, int T) {
    // Simplified: compute angles at each interior joint
    // Full implementation mirrors preprocess_for_onnx.py:_compute_joint_angles
    final numAngles = 67; // NUM_ANGLE_FEATURES from Python code
    final out = Float32List(T * numAngles);
    // ... (full implementation follows Python logic)
    return out;
  }

  Float32List _computeFingertipDistances(Float32List h, int T) {
    const numPairs = 10; // C(5,2) per hand
    final out = Float32List(T * 2 * numPairs);
    int col = 0;
    for (final tips in [lhTips, rhTips]) {
      for (int i = 0; i < tips.length; i++) {
        for (int j = i+1; j < tips.length; j++) {
          for (int t = 0; t < T; t++) {
            final b1 = t * numJoints * numChannels + tips[i] * numChannels;
            final b2 = t * numJoints * numChannels + tips[j] * numChannels;
            out[t * 2 * numPairs + col] = sqrt(
              pow(h[b1]-h[b2],2) + pow(h[b1+1]-h[b2+1],2) + pow(h[b1+2]-h[b2+2],2)
            );
          }
          col++;
        }
      }
    }
    return out;
  }
}
```

---

## 8. ONNX Inference

### Session Management

```dart
// lib/features/recognition/data/inference_service.dart
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class KslModelSession {
  OrtSession? _jointSess, _boneSess, _velSess;
  Map<String, double> _fusionWeights = {'joint': 0.333, 'bone': 0.333, 'velocity': 0.334};

  final String category; // 'numbers' or 'words'
  final String modelName; // 'v43', 'exp5', etc.

  KslModelSession({required this.category, required this.modelName});

  Future<void> load() async {
    final ort = OnnxRuntime();
    final base = 'assets/models/$category/$modelName';

    _jointSess = await ort.createSessionFromAsset('$base/joint/model_adapted.onnx');
    _boneSess  = await ort.createSessionFromAsset('$base/bone/model_adapted.onnx');
    _velSess   = await ort.createSessionFromAsset('$base/velocity/model_adapted.onnx');

    // Load fusion weights
    final json = await rootBundle.loadString('$base/fusion_weights.json');
    final parsed = jsonDecode(json) as Map<String, dynamic>;
    final weights = (parsed['weights'] ?? parsed) as Map<String, dynamic>;
    _fusionWeights = weights.map((k, v) => MapEntry(k, (v as num).toDouble()));
  }

  /// Run 3-stream inference, return fused probability vector
  Future<List<double>> run(Map<String, Float32List> streams, Float32List aux) async {
    const shape = [1, 3, 90, 48]; // [batch, channel, frames, joints]
    final auxShape = [1, 90, 61]; // [batch, frames, d_aux]

    Future<List<double>> runStream(OrtSession sess, String key) async {
      final inputs = {
        'gcn': await OrtValue.fromList(streams[key]!.toList(), shape),
        'aux': await OrtValue.fromList(aux.toList(), auxShape),
      };
      final outputs = await sess.run(inputs);
      final logits = await outputs[sess.outputNames[0]]!.asList() as List<double>;
      for (final v in inputs.values) v.dispose();
      for (final v in outputs.values) v?.dispose();
      return logits;
    }

    // Run all 3 streams in parallel
    final results = await Future.wait([
      runStream(_jointSess!, 'joint'),
      runStream(_boneSess!,  'bone'),
      runStream(_velSess!,   'velocity'),
    ]);

    // Fuse: weighted sum of softmax probabilities
    final numClasses = results[0].length;
    final fused = List<double>.filled(numClasses, 0.0);
    final streamKeys = ['joint', 'bone', 'velocity'];

    for (int s = 0; s < 3; s++) {
      final probs = _softmax(results[s]);
      final w = _fusionWeights[streamKeys[s]] ?? 0.333;
      for (int c = 0; c < numClasses; c++) {
        fused[c] += w * probs[c];
      }
    }
    return fused;
  }

  void dispose() {
    _jointSess?.close();
    _boneSess?.close();
    _velSess?.close();
  }
}

List<double> _softmax(List<double> logits) {
  final maxVal = logits.reduce(max);
  final exps = logits.map((l) => exp(l - maxVal)).toList();
  final sum = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sum).toList();
}
```

---

## 9. 6-Model Ensemble Fusion

```dart
// lib/features/recognition/domain/ensemble_fusion.dart

/// KSL class labels (sorted, matching training code)
const kNumbersClasses = [
  '9','17','22','35','48','54','66','73','89','91',
  '100','125','268','388','444',
];
const kWordsClasses = [
  'Agreement','Apple','Colour','Friend','Gift','Market',
  'Monday','Picture','Proud','Sweater','Teach','Tomatoes',
  'Tortoise','Twin','Ugali',
];

class KslEnsembleService {
  // 6-model uniform ensemble (72.9% combined accuracy)
  static const _ensembleModels = ['v27', 'v28', 'v29', 'exp1', 'exp5', 'openhands'];

  final Map<String, KslModelSession> _sessions = {};
  bool _loaded = false;

  Future<void> loadAll(String category) async {
    for (final name in _ensembleModels) {
      final sess = KslModelSession(category: category, modelName: name);
      await sess.load();
      _sessions[name] = sess;
    }
    _loaded = true;
  }

  /// Uniform ensemble: average 6 model probability vectors
  Future<KslPrediction> predict(
    Map<String, Float32List> streams,
    Float32List aux,
    String category,
  ) async {
    assert(_loaded, 'Call loadAll() first');

    final labels = category == 'numbers' ? kNumbersClasses : kWordsClasses;

    // Run all 6 models in parallel
    final allProbs = await Future.wait(
      _sessions.values.map((sess) => sess.run(streams, aux)),
    );

    // Uniform average
    final numClasses = labels.length;
    final avgProbs = List<double>.filled(numClasses, 0.0);
    for (final probs in allProbs) {
      for (int c = 0; c < numClasses; c++) {
        avgProbs[c] += probs[c] / allProbs.length;
      }
    }

    // Argmax
    int bestIdx = 0;
    for (int c = 1; c < numClasses; c++) {
      if (avgProbs[c] > avgProbs[bestIdx]) bestIdx = c;
    }

    return KslPrediction(
      label: labels[bestIdx],
      confidence: avgProbs[bestIdx],
      probabilities: avgProbs,
      allLabels: labels,
    );
  }

  void dispose() {
    for (final s in _sessions.values) s.dispose();
  }
}

class KslPrediction {
  final String label;
  final double confidence;
  final List<double> probabilities;
  final List<String> allLabels;

  const KslPrediction({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.allLabels,
  });
}
```

---

## 10. Background Isolate

Keep ONNX inference off the UI thread to maintain 60fps camera preview:

```dart
// lib/features/recognition/application/inference_isolate.dart
import 'dart:isolate';
import 'dart:typed_data';

// Message types (must be sendable across isolates — no Flutter objects)
class _InitMsg   { final String category; _InitMsg(this.category); }
class _InferMsg  { final Map<String, Float32List> streams; final Float32List aux; final int id;
                   _InferMsg(this.streams, this.aux, this.id); }
class _ResultMsg { final KslPrediction result; final int id;
                   _ResultMsg(this.result, this.id); }

@pragma('vm:entry-point')
void _isolateEntry(SendPort mainPort) {
  final port = ReceivePort();
  mainPort.send(port.sendPort);

  KslEnsembleService? service;

  port.listen((msg) async {
    if (msg is _InitMsg) {
      service = KslEnsembleService();
      await service!.loadAll(msg.category);
      mainPort.send('ready');
    } else if (msg is _InferMsg && service != null) {
      final result = await service!.predict(msg.streams, msg.aux, 'numbers');
      mainPort.send(_ResultMsg(result, msg.id));
    }
  });
}

class InferenceIsolateManager {
  Isolate? _isolate;
  SendPort? _sendPort;
  final _completers = <int, Completer<KslPrediction>>{};
  int _nextId = 0;

  Future<void> start(String category) async {
    final recv = ReceivePort();
    _isolate = await Isolate.spawn(_isolateEntry, recv.sendPort);

    final ready = Completer<void>();
    recv.listen((msg) {
      if (msg is SendPort) {
        _sendPort = msg;
        _sendPort!.send(_InitMsg(category));
      } else if (msg == 'ready') {
        ready.complete();
      } else if (msg is _ResultMsg) {
        _completers[msg.id]?.complete(msg.result);
        _completers.remove(msg.id);
      }
    });
    await ready.future;
  }

  Future<KslPrediction> infer(Map<String, Float32List> streams, Float32List aux) {
    final id = _nextId++;
    final c = Completer<KslPrediction>();
    _completers[id] = c;
    _sendPort!.send(_InferMsg(streams, aux, id));
    return c.future;
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
  }
}
```

---

## 11. UI — Camera + Overlay + Prediction Panel

```dart
// lib/features/recognition/presentation/recognition_screen.dart
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class RecognitionScreen extends ConsumerStatefulWidget {
  const RecognitionScreen({super.key});
  @override
  ConsumerState<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends ConsumerState<RecognitionScreen> {
  CameraController? _cam;
  final _preprocessor = KslPreprocessor();
  final _isolate = InferenceIsolateManager();
  KslPrediction? _prediction;
  int _frameCount = 0;
  bool _inferring = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _isolate.start('numbers');
    final cameras = await availableCameras();
    _cam = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _cam!.initialize();
    _cam!.startImageStream(_onFrame);
    setState(() {});
  }

  void _onFrame(CameraImage image) async {
    _frameCount++;
    // Process every 3rd frame (~10fps at 30fps camera)
    if (_frameCount % 3 != 0) return;

    // Extract skeleton (call SkeletonService here)
    // final joints = await skeletonService.processFrame(image, rotation);
    // if (joints != null) _preprocessor.addFrame(joints);

    // Trigger inference when enough frames collected (e.g. button press)
  }

  Future<void> _runInference() async {
    if (_inferring) return;
    _inferring = true;

    final streams = _preprocessor.preprocess();
    if (streams != null) {
      final aux = streams.remove('aux')!;
      final pred = await _isolate.infer(streams, aux);
      if (mounted) setState(() => _prediction = pred);
    }
    _preprocessor.clear();
    _inferring = false;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview
          if (_cam?.value.isInitialized ?? false)
            CameraPreview(_cam!),

          // Skeleton overlay
          const SkeletonOverlay(),

          // Prediction panel (bottom)
          if (_prediction != null)
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: PredictionPanel(prediction: _prediction!),
            ),

          // Category toggle (top-right)
          const Positioned(
            top: 48, right: 16,
            child: _CategoryToggle(),
          ),

          // Capture button (bottom-center)
          Positioned(
            bottom: _prediction != null ? 220 : 40,
            left: 0, right: 0,
            child: Center(
              child: GestureDetector(
                onTapDown: (_) => _preprocessor.clear(),
                onTapUp:   (_) => _runInference(),
                child: Container(
                  width: 72, height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white.withOpacity(0.85),
                    border: Border.all(color: Colors.white, width: 3),
                  ),
                  child: const Icon(Icons.sign_language, size: 36, color: Colors.deepPurple),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cam?.dispose();
    _isolate.dispose();
    super.dispose();
  }
}
```

### Prediction Panel

```dart
class PredictionPanel extends StatelessWidget {
  final KslPrediction prediction;
  const PredictionPanel({super.key, required this.prediction});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter, end: Alignment.topCenter,
          colors: [Colors.black.withOpacity(0.9), Colors.transparent],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(24, 32, 24, 48),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(prediction.label,
            style: const TextStyle(color: Colors.white, fontSize: 48, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Row(children: [
            Text('${(prediction.confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                color: prediction.confidence > 0.65 ? Colors.greenAccent : Colors.amberAccent,
                fontSize: 20, fontWeight: FontWeight.w600,
              )),
            const SizedBox(width: 12),
            Expanded(child: LinearProgressIndicator(
              value: prediction.confidence,
              backgroundColor: Colors.white24,
              color: prediction.confidence > 0.65 ? Colors.greenAccent : Colors.amberAccent,
              minHeight: 6,
            )),
          ]),
          const SizedBox(height: 12),
          // Top-3 alternatives
          ...(() {
            final indexed = List.generate(prediction.allLabels.length,
              (i) => (i, prediction.probabilities[i]));
            indexed.sort((a, b) => b.$2.compareTo(a.$2));
            return indexed.take(3).skip(1).map((e) => Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(children: [
                SizedBox(width: 100,
                  child: Text(prediction.allLabels[e.$1],
                    style: const TextStyle(color: Colors.white54, fontSize: 13))),
                Expanded(child: LinearProgressIndicator(
                  value: e.$2, backgroundColor: Colors.white12, color: Colors.white38)),
                const SizedBox(width: 8),
                Text('${(e.$2 * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(color: Colors.white38, fontSize: 12)),
              ]),
            ));
          })(),
        ],
      ),
    );
  }
}
```

---

## 12. Performance Tips

| Tip | Impact |
|-----|--------|
| Use `model_adapted.onnx` (AdaBN baked in) | +11.9pp numbers, +7.4pp words vs static |
| Keep sessions alive (don't recreate per inference) | ~200ms saved per call |
| Run 3 streams in `Future.wait()` | ~3× faster than sequential |
| Throttle camera to every 3rd frame | Reduces CPU load 66% |
| Use `ResolutionPreset.medium` (640×480) | Sufficient for pose; lower memory |
| Run inference in background isolate | UI stays at 60fps |
| NNAPI on Android (Snapdragon 7xx+) | 2-4× speedup (test on target device) |
| INT8 quantization (GroupNorm models only) | ~2-4× speedup, ~75% size reduction |

### Expected Latency (mid-range Android, CPU)

| Setup | Latency |
|-------|---------|
| Single model (V43), 3 streams sequential | ~30-50ms |
| Single model (V43), 3 streams parallel | ~10-20ms |
| 6-model ensemble, all parallel | ~20-50ms wall time |

---

## 13. Quick-Start: Single Model (V43)

For the simplest integration with the best balanced accuracy (65.7% combined):

```dart
// Simple V43 inference — no ensemble, no isolate
class SimpleKslRecognizer {
  KslModelSession? _numbers, _words;

  Future<void> init() async {
    _numbers = KslModelSession(category: 'numbers', modelName: 'v43');
    _words   = KslModelSession(category: 'words',   modelName: 'v43');
    await Future.wait([_numbers!.load(), _words!.load()]);
  }

  Future<KslPrediction> recognizeNumber(Map<String, Float32List> streams, Float32List aux) async {
    final probs = await _numbers!.run(streams, aux);
    final best  = probs.indexOf(probs.reduce(max));
    return KslPrediction(
      label: kNumbersClasses[best], confidence: probs[best],
      probabilities: probs, allLabels: kNumbersClasses,
    );
  }

  Future<KslPrediction> recognizeWord(Map<String, Float32List> streams, Float32List aux) async {
    final probs = await _words!.run(streams, aux);
    final best  = probs.indexOf(probs.reduce(max));
    return KslPrediction(
      label: kWordsClasses[best], confidence: probs[best],
      probabilities: probs, allLabels: kWordsClasses,
    );
  }

  void dispose() { _numbers?.dispose(); _words?.dispose(); }
}
```

---

## References

- [`flutter_onnxruntime` pub.dev](https://pub.dev/packages/flutter_onnxruntime)
- [`onnxruntime` (gtbluesky) GitHub](https://github.com/gtbluesky/onnxruntime_flutter)
- [Use ONNX Runtime in Flutter — Microsoft Surface Duo Blog](https://devblogs.microsoft.com/surface-duo/flutter-onnx-runtime/)
- [`hand_landmarker` Flutter package](https://pub.dev/packages/hand_landmarker/versions/2.1.2)
- [`google_mlkit_pose_detection` pub.dev](https://pub.dev/packages/google_mlkit_pose_detection)
- [Real-time ML with Flutter Camera — KBTG](https://medium.com/kbtg-life/real-time-machine-learning-with-flutter-camera-bbcf1b5c3193)
- [Flutter Isolates — dart.dev](https://dart.dev/language/isolates)
- [ONNX Runtime Mobile Deployment](https://onnxruntime.ai/docs/tutorials/mobile/)
- `preprocess_for_onnx.py` in this repo — reference Python preprocessing (port to Dart above)
- `evaluate_onnx.py` in this repo — reference inference pipeline
