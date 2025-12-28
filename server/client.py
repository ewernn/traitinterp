"""
Model server client.

Provides `get_model_or_client()` which returns either:
- ModelClient if server is running (uses HTTP)
- (model, tokenizer) tuple if server not available (loads locally)

Usage:
    from server.client import get_model_or_client, ModelClient

    handle = get_model_or_client("google/gemma-2-2b-it")
    if isinstance(handle, ModelClient):
        # Remote mode
        results = handle.generate_with_capture(prompts, ...)
    else:
        # Local mode
        model, tokenizer = handle
"""

import sys
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from server.serialization import (
    deserialize_capture_result,
    serialize_steering_vectors,
)

SERVER_URL = "http://localhost:8765"
TIMEOUT = 300  # 5 minutes for long generation requests


def is_server_available() -> bool:
    """Check if model server is running."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=1)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


class ModelClient:
    """
    Client for remote model server.

    Provides methods that match the interface expected by scripts:
    - generate_with_capture()
    - generate_with_steering()
    """

    def __init__(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """Initialize client and ensure model is loaded on server."""
        self.model_name = model_name
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit

        # Load model on server
        r = requests.post(
            f"{SERVER_URL}/model/load",
            params={
                "model_name": model_name,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        # Get server status
        status = requests.get(f"{SERVER_URL}/health", timeout=5).json()
        self._server_model = status.get("model")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate text from prompts."""
        r = requests.post(
            f"{SERVER_URL}/generate",
            json={
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["responses"]

    def generate_with_capture(
        self,
        prompts: List[str],
        n_layers: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        capture_mlp: bool = False,
        **kwargs,  # ignore extra args like batch_size, show_progress
    ):
        """Generate text and capture activations. Returns list of CaptureResult."""
        r = requests.post(
            f"{SERVER_URL}/generate/with-capture",
            json={
                "prompts": prompts,
                "n_layers": n_layers,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "capture_mlp": capture_mlp,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return [deserialize_capture_result(d) for d in r.json()["results"]]

    def generate_with_steering(
        self,
        prompts: List[str],
        vectors: Dict[int, 'torch.Tensor'],
        coefficients: Dict[int, float],
        component: str = "residual",
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Generate text with steering vectors applied."""
        r = requests.post(
            f"{SERVER_URL}/generate/with-steering",
            json={
                "prompts": prompts,
                "vectors": serialize_steering_vectors(vectors),
                "coefficients": {str(k): v for k, v in coefficients.items()},
                "component": component,
                "max_new_tokens": max_new_tokens,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["responses"]


def get_model_or_client(
    model_name: str,
    prefer_server: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **load_kwargs,
) -> Union[ModelClient, Tuple]:
    """
    Get model handle - ModelClient if server running, else local (model, tokenizer).

    Args:
        model_name: HuggingFace model name
        prefer_server: If True, try server first (default)
        load_in_8bit: Load in 8-bit quantization
        load_in_4bit: Load in 4-bit quantization
        **load_kwargs: Additional args passed to load_model()

    Returns:
        ModelClient if server available, else (model, tokenizer) tuple
    """
    if prefer_server and is_server_available():
        return ModelClient(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    # Fall back to local loading
    from utils.model import load_model
    return load_model(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **load_kwargs)
