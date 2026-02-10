"""
KSL Model Testing API Server

FastAPI backend for real-time sign language recognition inference.
Supports models v8-v14 with automatic preprocessing and model selection.

Usage:
    # With mock mode (no models required)
    python server.py --mock

    # With real models (requires checkpoints in ./checkpoints/)
    python server.py

    # Download checkpoints first
    python download_checkpoints.py
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in mock mode (default to real mode - false)
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

# Always try to import models/preprocessing
try:
    from models import create_model, NUMBER_CLASSES, WORD_CLASSES, ALL_CLASSES
    from preprocessing import preprocess_frames, MEDIAPIPE_AVAILABLE
    REAL_MODE_AVAILABLE = MEDIAPIPE_AVAILABLE
    if not MEDIAPIPE_AVAILABLE:
        logger.warning("MediaPipe not available, will use mock mode for inference")
except ImportError as e:
    logger.warning(f"Could not import models/preprocessing: {e}")
    REAL_MODE_AVAILABLE = False
    NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
    WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                           "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
    ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES


# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="KSL Model API",
    version="2.0.0",
    description="Kenya Sign Language Recognition API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = ["v8", "v9", "v10", "v11", "v12", "v13", "v14"]
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


# ============================================================================
# Model Cache
# ============================================================================

class ModelCache:
    """Cache for loaded models to avoid reloading."""

    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def get_model(self, version: str, model_type: str = "numbers") -> Optional[torch.nn.Module]:
        """Get or load a model."""
        cache_key = f"{version}_{model_type}"

        if cache_key in self.models:
            return self.models[cache_key]

        # Try to load from checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"{version}_{model_type}" / "best_model.pt"
        if not checkpoint_path.exists():
            # Try alternate naming
            checkpoint_path = CHECKPOINT_DIR / f"checkpoints_{version}_{model_type}" / "best_model.pt"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            # Determine number of classes
            num_classes = len(NUMBER_CLASSES) if model_type == "numbers" else len(WORD_CLASSES)

            # Create model
            model = create_model(version, num_classes, self.device)

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.models[cache_key] = model
            logger.info(f"Loaded model {cache_key} from {checkpoint_path}")

            return model

        except Exception as e:
            logger.error(f"Error loading model {cache_key}: {e}")
            return None

    def list_available(self) -> List[str]:
        """List available model checkpoints."""
        available = []
        for version in AVAILABLE_MODELS:
            for model_type in ["numbers", "words"]:
                checkpoint_path = CHECKPOINT_DIR / f"{version}_{model_type}" / "best_model.pt"
                if checkpoint_path.exists():
                    available.append(f"{version}_{model_type}")
        return available


model_cache = ModelCache()


# ============================================================================
# API Models
# ============================================================================

class PredictRequest(BaseModel):
    frames: List[str]  # Base64 encoded frames
    model_version: str
    model_type: str = "numbers"  # "numbers" or "words"


class Prediction(BaseModel):
    label: str
    confidence: float
    rank: int


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    model_type: str
    processing_time_ms: float
    mode: str  # "mock" or "real"


class ModelsResponse(BaseModel):
    models: List[str]
    available_checkpoints: List[str]
    mode: str


class HealthResponse(BaseModel):
    status: str
    device: str
    mock_mode: bool
    mediapipe_available: bool


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and configuration."""
    return HealthResponse(
        status="healthy",
        device=str(model_cache.device),
        mock_mode=MOCK_MODE,
        mediapipe_available=REAL_MODE_AVAILABLE
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """List available models and checkpoints."""
    return ModelsResponse(
        models=AVAILABLE_MODELS,
        available_checkpoints=model_cache.list_available() if not MOCK_MODE else [],
        mode="mock" if MOCK_MODE else "real"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on video frames."""
    start_time = time.time()

    # Validate model version
    if request.model_version not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model version. Available: {AVAILABLE_MODELS}"
        )

    # Validate frames
    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    if len(request.frames) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too few frames ({len(request.frames)}). Need at least 10 frames."
        )

    # Select class labels based on model type
    if request.model_type == "words":
        class_labels = WORD_CLASSES
    else:
        class_labels = NUMBER_CLASSES

    # Try real inference first (unless explicitly in mock mode)
    if REAL_MODE_AVAILABLE and not MOCK_MODE:
        try:
            predictions, mode = await run_real_inference(
                request.frames,
                request.model_version,
                request.model_type,
                class_labels
            )
        except Exception as e:
            logger.error(f"Real inference failed: {e}")
            logger.info("Falling back to mock predictions")
            # Fall back to mock
            predictions = generate_mock_predictions(class_labels)
            mode = "mock (fallback)"
    else:
        reason = "mock mode enabled" if MOCK_MODE else "MediaPipe/models not available"
        logger.info(f"Using mock predictions: {reason}")
        predictions = generate_mock_predictions(class_labels)
        mode = "mock"

    processing_time = (time.time() - start_time) * 1000

    return PredictResponse(
        predictions=predictions,
        model_version=request.model_version,
        model_type=request.model_type,
        processing_time_ms=processing_time,
        mode=mode
    )


# ============================================================================
# Inference Functions
# ============================================================================

async def run_real_inference(
    frames: List[str],
    version: str,
    model_type: str,
    class_labels: List[str]
) -> tuple:
    """Run actual model inference."""

    # Get model
    model = model_cache.get_model(version, model_type)
    if model is None:
        raise ValueError(f"Model {version}_{model_type} not available")

    # Preprocess frames
    input_tensor, _ = preprocess_frames(frames, version)
    input_tensor = input_tensor.to(model_cache.device)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probs[0], min(5, len(class_labels)))

    predictions = [
        Prediction(
            label=class_labels[idx.item()],
            confidence=prob.item(),
            rank=i + 1
        )
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]

    return predictions, "real"


def generate_mock_predictions(class_labels: List[str]) -> List[Prediction]:
    """Generate mock predictions for testing."""
    # Simulate some processing
    time.sleep(0.3)

    # Select random labels
    selected_labels = random.sample(class_labels, min(5, len(class_labels)))

    # Generate decreasing confidences
    confidences = sorted([random.random() for _ in selected_labels], reverse=True)

    # Normalize
    total = sum(confidences)
    confidences = [c / total for c in confidences]

    return [
        Prediction(label=label, confidence=conf, rank=i + 1)
        for i, (label, conf) in enumerate(zip(selected_labels, confidences))
    ]


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="KSL Model API Server")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "true"

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Mock mode: {MOCK_MODE}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

    uvicorn.run(app, host=args.host, port=args.port)
