"""
FastAPI model server - wraps existing utils/generation functions.

Usage:
    python other/server/app.py --port 8765 --model google/gemma-2-2b-it

Endpoints:
    GET  /health              - Server status
    POST /model/load          - Load model by name
    POST /generate            - Text generation
    POST /generate/with-capture   - Generation with activation capture
    POST /generate/with-steering  - Generation with steering vectors
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch

from utils.model import load_model as _load_model
from utils.generation import generate_batch, generate_with_capture
from other.server.serialization import serialize_capture_result, deserialize_tensor

app = FastAPI(title="Model Server", description="Persistent model loading for trait extraction")

# ============ Model State ============
_model = None
_tokenizer = None
_model_name = None


def get_model():
    """Get loaded model or raise error."""
    if _model is None:
        raise HTTPException(400, "No model loaded. POST /model/load first.")
    return _model, _tokenizer


# ============ Request/Response Schemas ============
class GenerateRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 256
    temperature: float = 0.0


class CaptureRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 50
    temperature: float = 0.7
    n_layers: Optional[int] = None
    capture_mlp: bool = False


class SteeringRequest(BaseModel):
    prompts: List[str]
    vectors: Dict[str, str]  # layer (str) -> serialized vector
    coefficients: Dict[str, float]  # layer (str) -> coefficient
    max_new_tokens: int = 256
    component: str = "residual"


# ============ Endpoints ============
@app.get("/health")
def health():
    """Check server status and loaded model."""
    return {
        "status": "ok",
        "model": _model_name,
        "loaded": _model is not None,
    }


@app.post("/model/load")
def load_model_endpoint(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load a model by name. Unloads previous model if different."""
    global _model, _tokenizer, _model_name

    if _model_name == model_name:
        return {"status": "already_loaded", "model": model_name}

    # Unload previous model
    if _model is not None:
        del _model, _tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load new model
    _model, _tokenizer = _load_model(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    _model_name = model_name

    return {"status": "loaded", "model": model_name}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate text from prompts."""
    model, tokenizer = get_model()
    responses = generate_batch(
        model, tokenizer, req.prompts,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {"responses": responses}


@app.post("/generate/with-capture")
def generate_capture(req: CaptureRequest):
    """Generate text and capture activations."""
    model, tokenizer = get_model()
    results = generate_with_capture(
        model, tokenizer, req.prompts,
        n_layers=req.n_layers,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        capture_mlp=req.capture_mlp,
        show_progress=False,
    )
    return {"results": [serialize_capture_result(r) for r in results]}


@app.post("/generate/with-steering")
def generate_steering(req: SteeringRequest):
    """Generate text with steering vectors applied."""
    from core import MultiLayerSteeringHook

    model, tokenizer = get_model()

    # Deserialize vectors
    vectors = {int(l): deserialize_tensor(v) for l, v in req.vectors.items()}

    # Build steering configs: [(layer, vector, coef), ...]
    configs = [
        (layer, vectors[layer].to(model.device), req.coefficients[str(layer)])
        for layer in sorted(vectors.keys())
    ]

    with MultiLayerSteeringHook(model, configs, component=req.component):
        responses = generate_batch(
            model, tokenizer, req.prompts,
            max_new_tokens=req.max_new_tokens,
        )

    return {"responses": responses}


# ============ Entry Point ============
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, help="Pre-load model on startup")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load in 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit quantization")
    args = parser.parse_args()

    if args.model:
        print(f"Pre-loading model: {args.model}")
        load_model_endpoint(args.model, args.load_in_8bit, args.load_in_4bit)

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
