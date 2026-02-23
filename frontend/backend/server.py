"""
KSL Model Testing API Server

FastAPI backend for real-time sign language recognition inference.
Supports multi-stream ST-GCN models (v27-v43), OpenHands DecoupledGCN,
and 6-model uniform ensemble.

Usage:
    # With mock mode (no models required)
    python server.py --mock

    # With real models
    python server.py

    # Specific model only
    python server.py --default-model v43
"""

import json
import os
import sys
import time
import copy
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup: project root is two levels up from frontend/backend/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in mock mode
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
REAL_MODE_AVAILABLE = False
try:
    from evaluate_real_testers_v30 import (
        preprocess_multistream,
        preprocess_v27,
        build_adj,
        adapt_bn_stats,
        KSLGraphNetV25,   # v27 (ic=9) and v28 per-stream (ic=3)
        KSLGraphNetV29,   # v29 (ic=9)
        NUMBER_CLASSES,
        WORD_CLASSES,
        NUM_ANGLE_FEATURES,
        NUM_FINGERTIP_PAIRS,
        NUM_HAND_BODY_FEATURES,
    )
    from train_ksl_v43 import KSLGraphNetV43
    from train_ksl_v41 import KSLGraphNetV41
    from train_ksl_v31_exp1 import KSLGraphNetV31Exp1
    from train_ksl_v31_exp5 import KSLGraphNetV25 as KSLGraphNetExp5
    from train_ksl_openhands import OpenHandsClassifier, OH_CONFIG
    from evaluate_openhands_realtest import preprocess_raw_for_openhands
    from preprocessing import decode_base64_frames, MediaPipeExtractor, MEDIAPIPE_AVAILABLE

    AUX_DIM = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    REAL_MODE_AVAILABLE = MEDIAPIPE_AVAILABLE
    if not MEDIAPIPE_AVAILABLE:
        logger.warning("MediaPipe not available, will use mock mode for inference")
except ImportError as e:
    logger.warning(f"Could not import project modules: {e}")
    REAL_MODE_AVAILABLE = False
    NUMBER_CLASSES = sorted([
        "9", "17", "22", "35", "48", "54", "66", "73", "89", "91",
        "100", "125", "268", "388", "444",
    ])
    WORD_CLASSES = sorted([
        "Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
        "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali",
    ])

ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="KSL Model API",
    version="3.0.0",
    description="Kenya Sign Language Recognition API — multi-stream ST-GCN models"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = ["ensemble_6_uniform", "v43", "v41", "exp5", "exp1"]

# Ensemble members (order matters for consistent logging)
ENSEMBLE_MEMBERS = ["v27", "v28", "v29", "exp1", "exp5", "openhands"]

# Models that use BatchNorm and need AdaBN at inference
ADABN_MODELS = {"v43", "exp5", "v28", "v27", "v29", "openhands"}

# Models that are multi-stream (3 separate checkpoints: joint/bone/velocity)
MULTISTREAM_MODELS = {"v43", "v41", "exp5", "exp1", "v28"}

# Models that are single-stream (1 checkpoint, ic=9)
SINGLESTREAM_9CH = {"v27", "v29"}

# Checkpoint directory name mapping
CKPT_DIR_MAP = {
    "v43": "v43",
    "v41": "v41",
    "exp1": "v31_exp1",
    "exp5": "v31_exp5",
    "v28": "v28",
    "v27": "v27",
    "v29": "v29",
    "openhands": "openhands",
}


# ============================================================================
# Model Cache
# ============================================================================

class PTModelCache:
    """Cache for loaded PyTorch models."""

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _load_multistream_model(self, version: str, category: str) -> Optional[dict]:
        """Load a multi-stream model (3 stream models + fusion weights).

        Returns dict with keys: 'joint', 'bone', 'velocity' (nn.Module each),
        'fusion_weights' (dict), 'needs_adabn' (bool).
        """
        ckpt_name = CKPT_DIR_MAP[version]
        base = PROJECT_ROOT / "data" / "checkpoints" / f"{ckpt_name}_{category}" \
            if version in ("v28",) else \
            PROJECT_ROOT / "data" / "checkpoints" / ckpt_name / category

        adj = build_adj(48).to(self.device)
        result = {"needs_adabn": version in ADABN_MODELS}
        streams_loaded = 0

        for stream in ("joint", "bone", "velocity"):
            ckpt_path = base / stream / "best_model.pt"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                return None

            ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            c = ckpt.get("config", {})
            num_signers = ckpt.get("num_signers", 12)
            nc = len(NUMBER_CLASSES) if category == "numbers" else len(WORD_CLASSES)

            model = self._create_stream_model(version, nc, num_signers, adj, c)
            if model is None:
                return None

            model.load_state_dict(ckpt["model"])
            model.eval()
            model.to(self.device)
            result[stream] = model
            streams_loaded += 1

        # Load fusion weights
        fw_path = base / "fusion_weights.json"
        if fw_path.exists():
            with open(fw_path) as f:
                fw_data = json.load(f)
            # Handle nested "weights" key
            if "weights" in fw_data and isinstance(fw_data["weights"], dict):
                result["fusion_weights"] = fw_data["weights"]
            else:
                result["fusion_weights"] = fw_data
        else:
            result["fusion_weights"] = {"joint": 1/3, "bone": 1/3, "velocity": 1/3}

        logger.info(f"Loaded {version}/{category}: {streams_loaded} streams, "
                     f"weights={result['fusion_weights']}")
        return result

    def _create_stream_model(self, version, nc, num_signers, adj, config):
        """Instantiate the correct model class for a given version."""
        if version == "v43":
            return KSLGraphNetV43(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                nn_=48, ic=3, hd=64, nl=4,
                tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=adj,
            )
        elif version == "v41":
            return KSLGraphNetV41(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                proj_dim=128, nn_=48, ic=3, hd=64, nl=4,
                tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=adj,
            )
        elif version == "exp1":
            return KSLGraphNetV31Exp1(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                nn_=48, ic=3, hd=64, nl=4,
                tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=adj,
            )
        elif version in ("exp5", "v28"):
            return KSLGraphNetExp5(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                nn_=48, ic=3, hd=64, nl=4,
                tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=adj,
            )
        return None

    def _load_singlestream_model(self, version: str, category: str) -> Optional[dict]:
        """Load a single-stream model (v27 or v29, ic=9)."""
        ckpt_name = CKPT_DIR_MAP[version]
        ckpt_path = PROJECT_ROOT / "data" / "checkpoints" / f"{ckpt_name}_{category}" / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return None

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        c = ckpt.get("config", {})
        num_signers = ckpt.get("num_signers", 12)
        nc = len(NUMBER_CLASSES) if category == "numbers" else len(WORD_CLASSES)
        adj = build_adj(48).to(self.device)

        if version == "v29":
            model = KSLGraphNetV29(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                nn_=48, ic=9, hd=64, nl=8,
                td=(1, 2, 4), dr=0.2, spatial_dropout=0.1, adj=adj,
            )
        else:  # v27
            model = KSLGraphNetV25(
                nc=nc, num_signers=num_signers, aux_dim=AUX_DIM,
                nn_=48, ic=9, hd=64, nl=4,
                tk=(3, 5, 7), dr=0.3, spatial_dropout=0.1, adj=adj,
            )

        model.load_state_dict(ckpt["model"])
        model.eval()
        model.to(self.device)

        logger.info(f"Loaded {version}/{category}: single-stream ic=9")
        return {
            "model": model,
            "needs_adabn": version in ADABN_MODELS,
            "type": "singlestream",
        }

    def _load_openhands_model(self, category: str) -> Optional[dict]:
        """Load OpenHands DecoupledGCN model."""
        ckpt_path = PROJECT_ROOT / "data" / "checkpoints" / "openhands" / category / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return None

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        config = ckpt.get("config", OH_CONFIG)
        nc = len(NUMBER_CLASSES) if category == "numbers" else len(WORD_CLASSES)

        model = OpenHandsClassifier(
            num_classes=nc,
            in_channels=config.get("in_channels", 2),
            num_nodes=config.get("num_nodes", 27),
            n_out_features=config.get("n_out_features", 256),
            cls_dropout=config.get("cls_dropout", 0.3),
        ).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        logger.info(f"Loaded openhands/{category}")
        return {
            "model": model,
            "needs_adabn": True,
            "type": "openhands",
        }

    def get_model(self, version: str, category: str) -> Optional[dict]:
        """Get or load a model (cached)."""
        cache_key = f"{version}_{category}"
        if cache_key in self.models:
            return self.models[cache_key]

        loaded = None
        if version in MULTISTREAM_MODELS:
            loaded = self._load_multistream_model(version, category)
            if loaded is not None:
                loaded["type"] = "multistream"
        elif version in SINGLESTREAM_9CH:
            loaded = self._load_singlestream_model(version, category)
        elif version == "openhands":
            loaded = self._load_openhands_model(category)

        if loaded is not None:
            self.models[cache_key] = loaded
        return loaded

    def get_ensemble(self, category: str) -> Optional[List[Tuple[str, dict]]]:
        """Load all 6 ensemble members for a category."""
        members = []
        for member_name in ENSEMBLE_MEMBERS:
            m = self.get_model(member_name, category)
            if m is None:
                logger.warning(f"Ensemble member {member_name}/{category} not available")
                return None
            members.append((member_name, m))
        return members

    def list_available(self) -> List[str]:
        """List models that have checkpoints available."""
        available = []
        for version in AVAILABLE_MODELS:
            for cat in ("numbers", "words"):
                if version == "ensemble_6_uniform":
                    # Check if all ensemble members exist
                    all_ok = True
                    for member in ENSEMBLE_MEMBERS:
                        if member in MULTISTREAM_MODELS:
                            ckpt_name = CKPT_DIR_MAP[member]
                            if member == "v28":
                                base = PROJECT_ROOT / "data" / "checkpoints" / f"{ckpt_name}_{cat}"
                            else:
                                base = PROJECT_ROOT / "data" / "checkpoints" / ckpt_name / cat
                            if not (base / "joint" / "best_model.pt").exists():
                                all_ok = False
                                break
                        elif member in SINGLESTREAM_9CH:
                            ckpt_name = CKPT_DIR_MAP[member]
                            if not (PROJECT_ROOT / "data" / "checkpoints" / f"{ckpt_name}_{cat}" / "best_model.pt").exists():
                                all_ok = False
                                break
                        elif member == "openhands":
                            if not (PROJECT_ROOT / "data" / "checkpoints" / "openhands" / cat / "best_model.pt").exists():
                                all_ok = False
                                break
                    if all_ok:
                        available.append(f"ensemble_6_uniform_{cat}")
                else:
                    m = self.get_model(version, cat)
                    if m is not None:
                        available.append(f"{version}_{cat}")
        return available


model_cache = PTModelCache()


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
        mediapipe_available=REAL_MODE_AVAILABLE,
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """List available models and checkpoints."""
    return ModelsResponse(
        models=AVAILABLE_MODELS,
        available_checkpoints=model_cache.list_available() if not MOCK_MODE else [],
        mode="mock" if MOCK_MODE else "real",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on video frames."""
    start_time = time.time()

    if request.model_version not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model version. Available: {AVAILABLE_MODELS}",
        )

    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    if len(request.frames) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too few frames ({len(request.frames)}). Need at least 10.",
        )

    class_labels = WORD_CLASSES if request.model_type == "words" else NUMBER_CLASSES

    if REAL_MODE_AVAILABLE and not MOCK_MODE:
        try:
            predictions, mode = await run_real_inference(
                request.frames,
                request.model_version,
                request.model_type,
                class_labels,
            )
        except Exception as e:
            logger.error(f"Real inference failed: {e}", exc_info=True)
            logger.info("Falling back to mock predictions")
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
        mode=mode,
    )


# ============================================================================
# Inference Functions
# ============================================================================

def extract_landmarks(frames: List[str]) -> np.ndarray:
    """Decode base64 frames and extract (T, 225) landmarks via MediaPipe."""
    decoded = decode_base64_frames(frames)
    if len(decoded) < 10:
        raise ValueError(f"Only {len(decoded)} valid frames after decoding")
    with MediaPipeExtractor() as extractor:
        raw = extractor.extract_from_frames(decoded, include_face=False)
    return raw  # shape (T, 225)


def run_single_multistream(model_dict: dict, streams: dict, aux: torch.Tensor,
                           device: torch.device, adapt_data=None) -> torch.Tensor:
    """Run inference on a multi-stream model. Returns softmax probabilities."""
    fusion_w = model_dict["fusion_weights"]
    all_logits = {}

    for stream_name in ("joint", "bone", "velocity"):
        model = model_dict[stream_name]

        # AdaBN: adapt BN stats using the current sample as adaptation data
        if model_dict["needs_adabn"] and adapt_data is not None:
            model_copy = copy.deepcopy(model)
            adapt_bn_stats(model_copy, adapt_data, device, stream_name=stream_name)
        else:
            model_copy = model

        gcn_input = streams[stream_name].unsqueeze(0).to(device)
        aux_input = aux.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_copy(gcn_input, aux_input, grl_lambda=0.0)
            logits = output[0]  # First element is always logits
        all_logits[stream_name] = logits

    # Weighted fusion of logits
    fused = sum(
        fusion_w.get(s, 1/3) * all_logits[s] for s in ("joint", "bone", "velocity")
    )
    return F.softmax(fused, dim=1)


def run_single_singlestream(model_dict: dict, raw: np.ndarray,
                            device: torch.device, adapt_data=None) -> torch.Tensor:
    """Run inference on a single-stream ic=9 model (v27/v29). Returns softmax probs."""
    gcn_tensor, aux_tensor = preprocess_v27(raw)
    if gcn_tensor is None:
        raise ValueError("Preprocessing failed for single-stream model")

    model = model_dict["model"]

    if model_dict["needs_adabn"] and adapt_data is not None:
        model_copy = copy.deepcopy(model)
        # For single-stream, adapt_data items are (gcn_tensor, aux_tensor) tuples
        adapt_bn_stats(model_copy, adapt_data, device, stream_name=None)
    else:
        model_copy = model

    with torch.no_grad():
        output = model_copy(
            gcn_tensor.unsqueeze(0).to(device),
            aux_tensor.unsqueeze(0).to(device),
            grl_lambda=0.0,
        )
        logits = output[0]
    return F.softmax(logits, dim=1)


def run_openhands_inference(model_dict: dict, raw: np.ndarray,
                            device: torch.device, adapt_data=None) -> torch.Tensor:
    """Run inference on OpenHands model. Returns softmax probs."""
    oh_tensor = preprocess_raw_for_openhands(raw)
    if oh_tensor is None:
        raise ValueError("OpenHands preprocessing failed")

    model = model_dict["model"]

    if model_dict["needs_adabn"] and adapt_data is not None:
        model_copy = copy.deepcopy(model)
        # AdaBN for openhands: pass through model in train mode
        for m in model_copy.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.num_batches_tracked.zero_()
                m.momentum = None
        model_copy.train()
        with torch.no_grad():
            for item in adapt_data:
                model_copy(item.unsqueeze(0).to(device))
        model_copy.eval()
    else:
        model_copy = model

    with torch.no_grad():
        output = model_copy(oh_tensor.unsqueeze(0).to(device))
        logits = output[0]
    return F.softmax(logits, dim=1)


def run_member_inference(member_name: str, model_dict: dict, raw: np.ndarray,
                         streams: dict, aux: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Run inference for a single ensemble member."""
    model_type = model_dict["type"]

    # AdaBN disabled for single predictions — needs 50+ samples to be effective
    if model_type == "multistream":
        return run_single_multistream(model_dict, streams, aux, device, adapt_data=None)
    elif model_type == "singlestream":
        return run_single_singlestream(model_dict, raw, device, adapt_data=None)
    elif model_type == "openhands":
        return run_openhands_inference(model_dict, raw, device, adapt_data=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


async def run_real_inference(
    frames: List[str],
    version: str,
    model_type: str,
    class_labels: List[str],
) -> tuple:
    """Run actual model inference."""
    device = model_cache.device

    # Step 1: Extract landmarks from video frames
    raw = extract_landmarks(frames)  # (T, 225)

    # Step 2: Preprocess for multi-stream models
    streams, aux = preprocess_multistream(raw)
    if streams is None:
        raise ValueError("Preprocessing failed (too few landmarks)")

    # Step 3: Dispatch based on model version
    if version == "ensemble_6_uniform":
        probs = await run_ensemble_inference(raw, streams, aux, model_type, device)
    else:
        model_dict = model_cache.get_model(version, model_type)
        if model_dict is None:
            raise ValueError(f"Model {version}/{model_type} not available")
        probs = run_member_inference(version, model_dict, raw, streams, aux, device)

    # Step 4: Get top-5 predictions
    top_probs, top_indices = torch.topk(probs[0], min(5, len(class_labels)))

    predictions = [
        Prediction(
            label=class_labels[idx.item()],
            confidence=prob.item(),
            rank=i + 1,
        )
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]

    return predictions, "real"


async def run_ensemble_inference(raw: np.ndarray, streams: dict, aux: torch.Tensor,
                                 category: str, device: torch.device) -> torch.Tensor:
    """Run 6-model uniform ensemble. Returns averaged probabilities."""
    members = model_cache.get_ensemble(category)
    if members is None:
        raise ValueError(f"Could not load all ensemble members for {category}")

    all_probs = []
    for member_name, model_dict in members:
        try:
            probs = run_member_inference(member_name, model_dict, raw, streams, aux, device)
            all_probs.append(probs)
        except Exception as e:
            logger.error(f"Ensemble member {member_name} failed: {e}")
            continue

    if not all_probs:
        raise ValueError("All ensemble members failed")

    # Uniform average
    avg_probs = sum(all_probs) / len(all_probs)
    return avg_probs


def generate_mock_predictions(class_labels: List[str]) -> List[Prediction]:
    """Generate mock predictions for testing."""
    time.sleep(0.3)
    selected_labels = random.sample(class_labels, min(5, len(class_labels)))
    confidences = sorted([random.random() for _ in selected_labels], reverse=True)
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
    parser.add_argument("--default-model", default="ensemble_6_uniform",
                        help="Default model version")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "true"

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Mock mode: {MOCK_MODE}")
    logger.info(f"Available models: {AVAILABLE_MODELS}")
    logger.info(f"Project root: {PROJECT_ROOT}")

    uvicorn.run(app, host=args.host, port=args.port)
