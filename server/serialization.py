"""
Tensor serialization for HTTP transport.

Uses safetensors + base64 for efficient, safe serialization.

Usage:
    from server.serialization import serialize_tensor, deserialize_tensor

    encoded = serialize_tensor(tensor)  # str
    tensor = deserialize_tensor(encoded)  # Tensor
"""

import base64
from typing import Dict, Any

import torch
from safetensors.torch import save as st_save, load as st_load


def serialize_tensor(tensor: torch.Tensor) -> str:
    """Tensor -> base64 string. Always moves to CPU first."""
    if tensor.numel() == 0:
        # Empty tensor: save shape and dtype only
        return f"empty:{list(tensor.shape)}:{tensor.dtype}"

    tensor = tensor.cpu().contiguous()
    raw_bytes = st_save({"t": tensor})  # returns bytes
    return base64.b64encode(raw_bytes).decode('ascii')


def deserialize_tensor(data: str) -> torch.Tensor:
    """base64 string -> Tensor."""
    if data.startswith("empty:"):
        # Parse empty tensor metadata
        parts = data.split(":", 2)
        shape = eval(parts[1])  # safe: only list of ints
        dtype_str = parts[2].split(".")[-1]
        dtype = getattr(torch, dtype_str)
        return torch.empty(shape, dtype=dtype)

    raw_bytes = base64.b64decode(data)
    return st_load(raw_bytes)["t"]


def serialize_activations(acts: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, str]]:
    """Serialize nested activation dict: {layer: {component: tensor}}."""
    return {
        str(layer): {k: serialize_tensor(v) for k, v in components.items()}
        for layer, components in acts.items()
    }


def deserialize_activations(data: Dict[str, Dict[str, str]]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Deserialize nested activation dict."""
    return {
        int(layer): {k: deserialize_tensor(v) for k, v in components.items()}
        for layer, components in data.items()
    }


def serialize_capture_result(result) -> Dict[str, Any]:
    """Serialize CaptureResult dataclass for HTTP transport."""
    return {
        "prompt_text": result.prompt_text,
        "response_text": result.response_text,
        "prompt_tokens": result.prompt_tokens,
        "response_tokens": result.response_tokens,
        "prompt_token_ids": result.prompt_token_ids,
        "response_token_ids": result.response_token_ids,
        "prompt_activations": serialize_activations(result.prompt_activations),
        "response_activations": serialize_activations(result.response_activations),
    }


def deserialize_capture_result(data: Dict[str, Any]):
    """Deserialize to CaptureResult object."""
    from utils.generation import CaptureResult
    return CaptureResult(
        prompt_text=data["prompt_text"],
        response_text=data["response_text"],
        prompt_tokens=data["prompt_tokens"],
        response_tokens=data["response_tokens"],
        prompt_token_ids=data["prompt_token_ids"],
        response_token_ids=data["response_token_ids"],
        prompt_activations=deserialize_activations(data["prompt_activations"]),
        response_activations=deserialize_activations(data["response_activations"]),
    )


def serialize_steering_vectors(vectors: Dict[int, torch.Tensor]) -> Dict[str, str]:
    """Serialize {layer: vector} for steering request."""
    return {str(l): serialize_tensor(v) for l, v in vectors.items()}


def deserialize_steering_vectors(data: Dict[str, str]) -> Dict[int, torch.Tensor]:
    """Deserialize steering vectors."""
    return {int(l): deserialize_tensor(v) for l, v in data.items()}
