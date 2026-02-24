# KSL Model API Reference

> **Version 3.0.0** -- Kenya Sign Language Recognition API

## Overview

The KSL Model API provides real-time Kenya Sign Language recognition from video frames. It uses multi-stream Spatial-Temporal Graph Convolutional Networks (ST-GCN) that operate on hand and pose landmarks extracted via MediaPipe.

The API accepts an uploaded video file, extracts frames and skeletal landmarks, and returns the top-5 predicted sign labels with confidence scores.

### Interactive Documentation

| Interface | URL |
|-----------|-----|
| Swagger UI | [http://localhost:8000/docs](http://localhost:8000/docs) |
| ReDoc | [http://localhost:8000/redoc](http://localhost:8000/redoc) |
| OpenAPI JSON | [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) |

## Base URL

```
http://localhost:8000
```

No authentication is required. CORS is enabled for all origins.

---

## Endpoints

### GET /health

Check API health and runtime configuration.

**Tags:** `system`

#### Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"healthy"` |
| `device` | string | Compute device: `"cuda"` or `"cpu"` |
| `mock_mode` | boolean | Whether the server was started with `--mock` |
| `mediapipe_available` | boolean | Whether MediaPipe and model files are loaded |

#### Example Response

```json
{
  "status": "healthy",
  "device": "cuda",
  "mock_mode": false,
  "mediapipe_available": true
}
```

---

### GET /models

List available models and which checkpoints are present on disk.

**Tags:** `system`

#### Response

| Field | Type | Description |
|-------|------|-------------|
| `models` | string[] | All supported model version identifiers |
| `available_checkpoints` | string[] | Models with checkpoint files on disk (format: `{model}_{category}`) |
| `mode` | string | `"real"` if PyTorch models are available, `"mock"` otherwise |

#### Example Response

```json
{
  "models": [
    "ensemble_6_uniform",
    "v43",
    "v41",
    "v37",
    "exp5",
    "exp1"
  ],
  "available_checkpoints": [
    "ensemble_6_uniform_numbers",
    "ensemble_6_uniform_words",
    "v43_numbers",
    "v43_words",
    "v41_numbers",
    "v41_words"
  ],
  "mode": "real"
}
```

---

### POST /predict

Run sign language recognition on an uploaded video file.

**Tags:** `inference`
**Content-Type:** `multipart/form-data`

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video` | file | Yes | -- | Video file (mp4, webm, mov, avi) — minimum ~1 second of signing |
| `model_version` | string | No | `"ensemble_6_uniform"` | Model to use (see [Models](#models) below) |
| `model_type` | string | No | `"numbers"` | `"numbers"` for numeric signs, `"words"` for word signs |

#### Response

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | Prediction[] | Top-5 predicted signs, ranked by confidence |
| `model_version` | string | Model used for inference |
| `model_type` | string | Category used (`"numbers"` or `"words"`) |
| `processing_time_ms` | float | End-to-end processing time in milliseconds |
| `mode` | string | `"real"`, `"mock"`, or `"mock (fallback)"` |

**Prediction object:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Predicted sign label |
| `confidence` | float | Softmax probability (0.0 - 1.0) |
| `rank` | int | Rank position (1 = highest confidence) |

#### Example Response

```json
{
  "predictions": [
    { "label": "54",  "confidence": 0.82, "rank": 1 },
    { "label": "48",  "confidence": 0.09, "rank": 2 },
    { "label": "73",  "confidence": 0.04, "rank": 3 },
    { "label": "66",  "confidence": 0.03, "rank": 4 },
    { "label": "100", "confidence": 0.02, "rank": 5 }
  ],
  "model_version": "ensemble_6_uniform",
  "model_type": "numbers",
  "processing_time_ms": 245.3,
  "mode": "real"
}
```

---

## Models

| Model | Architecture | Numbers Acc. | Words Acc. | Combined Acc. |
|-------|-------------|-------------|-----------|--------------|
| `ensemble_6_uniform` | 6-model uniform ensemble | 74.6% | 71.6% | **72.9%** |
| `v43` | ST-GCN + SupCon + R&R | 66.1% | 65.4% | 65.7% |
| `v41` | ST-GCN GroupNorm + R&R | 67.8% | 55.6% | 60.7% |
| `v37` | ST-GCN GroupNorm + Speed Aug | 57.9% | 57.9% | 57.9% |
| `exp5` | ST-GCN + SupCon | 61.0% | 65.4% | 63.2% |
| `exp1` | ST-GCN GroupNorm | 61.0% | 61.7% | 61.4% |

The **ensemble** averages predictions from six sub-models (v27, v28, v29, exp1, exp5, OpenHands) with equal weights. It is the recommended model for production use.

---

## Sign Classes

### Numbers (15 classes)

`9`, `17`, `22`, `35`, `48`, `54`, `66`, `73`, `89`, `91`, `100`, `125`, `268`, `388`, `444`

### Words (15 classes)

`Agreement`, `Apple`, `Colour`, `Friend`, `Gift`, `Market`, `Monday`, `Picture`, `Proud`, `Sweater`, `Teach`, `Tomatoes`, `Tortoise`, `Twin`, `Ugali`

---

## Error Codes

| Status | Condition | Detail |
|--------|-----------|--------|
| 400 | Invalid `model_version` | `"Invalid model version. Available: [...]"` |
| 400 | Empty `frames` list | `"No frames provided"` |
| 400 | Fewer than 10 frames | `"Too few frames (N). Need at least 10."` |
| 422 | Malformed request body | Standard FastAPI validation error |

All error responses follow the structure:

```json
{
  "detail": "Error message here"
}
```

---

## Inference Pipeline

1. **Frame extraction** -- OpenCV reads the uploaded video file and extracts up to 90 frames (subsampled if longer).
2. **Landmark extraction** -- MediaPipe Holistic extracts 225 landmark coordinates per frame (hands + pose, no face mesh).
3. **Stream computation** -- Raw landmarks are transformed into three streams:
   - **Joint** -- normalized (x, y, z) coordinates
   - **Bone** -- vectors between connected joints
   - **Velocity** -- frame-to-frame differences
4. **Model inference** -- Each stream is fed through its own ST-GCN branch. Logits are fused via learned weights.
   - For the ensemble, all six member models run independently and their softmax outputs are averaged.
5. **Ranking** -- The top-5 classes by probability are returned.

### Video Requirements

- **Format:** mp4, webm, mov, avi (anything OpenCV can decode)
- **Minimum length:** ~0.3 seconds (must yield at least 10 frames)
- **Recommended length:** 1-3 seconds of signing
- **Max frames:** 90 — longer videos are automatically subsampled

### Confidence Scores

- Confidence values are softmax probabilities and sum to 1.0 across all 15 classes in the chosen category.
- Only the top 5 are returned. To compute the tail probability: `1.0 - sum(top5)`.

---

## Code Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Predict from a video file
curl -X POST http://localhost:8000/predict \
  -F "video=@sign.mp4" \
  -F "model_version=ensemble_6_uniform" \
  -F "model_type=numbers"
```

### Python (requests)

```python
import requests

API_URL = "http://localhost:8000"

with open("sign.mp4", "rb") as f:
    response = requests.post(
        f"{API_URL}/predict",
        files={"video": ("sign.mp4", f, "video/mp4")},
        data={"model_version": "ensemble_6_uniform", "model_type": "words"},
    )

data = response.json()
for pred in data["predictions"]:
    print(f"  #{pred['rank']} {pred['label']}: {pred['confidence']:.1%}")

print(f"Processing time: {data['processing_time_ms']:.0f}ms")
```

### JavaScript (fetch)

```javascript
// Record a video blob from the webcam (MediaRecorder API)
async function recordAndPredict(stream, durationMs = 2000) {
  const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  const chunks = [];

  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();

  await new Promise((resolve) => setTimeout(resolve, durationMs));
  recorder.stop();
  await new Promise((resolve) => (recorder.onstop = resolve));

  const blob = new Blob(chunks, { type: "video/webm" });

  const form = new FormData();
  form.append("video", blob, "sign.webm");
  form.append("model_version", "ensemble_6_uniform");
  form.append("model_type", "numbers");

  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }

  return res.json();
}

// Usage
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const result = await recordAndPredict(stream, 2000);
console.log("Top prediction:", result.predictions[0].label);
```

---

## Running the Server

```bash
# Real mode (requires PyTorch models + MediaPipe)
python server.py

# Mock mode (no models needed -- returns random predictions)
python server.py --mock

# Custom host and port
python server.py --host 0.0.0.0 --port 9000
```
