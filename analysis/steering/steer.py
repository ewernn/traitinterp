"""
Steering hook for adding trait vectors during generation.

Input:
    - model: Loaded Gemma model
    - vector: Trait vector tensor (hidden_dim,)
    - layer: Layer index to hook
    - coefficient: Scaling factor for vector

Output:
    - Context manager that adds coefficient * vector to layer output

Usage:
    vector = torch.load('experiments/exp/extraction/trait/vectors/probe_layer16.pt')
    with SteeringHook(model, vector, layer=16, coefficient=1.5):
        output = model.generate(**inputs)
"""

import torch
from contextlib import contextmanager
from typing import Union, Sequence


class SteeringHook:
    """
    Context manager that adds (coefficient * vector) to a layer's output during generation.

    Handles Gemma's tuple outputs (hidden_states, ...) and adds to ALL token positions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vector: Union[torch.Tensor, Sequence[float]],
        layer: int,
        coefficient: float = 1.0,
        positions: str = "all",
    ):
        """
        Args:
            model: The transformer model (Gemma 2B)
            vector: Steering vector of shape (hidden_dim,)
            layer: Layer index to hook (0-25 for Gemma 2B)
            coefficient: Scaling factor (typical range: 0.5-2.5)
            positions: Which positions to steer:
                - "all": All token positions (recommended for evaluation)
                - "last": Only last token (for generation efficiency)
        """
        self.model = model
        self.layer = layer
        self.coefficient = float(coefficient)
        self.positions = positions.lower()
        self._handle = None

        # Convert vector to tensor on model device/dtype
        param = next(model.parameters())
        self.vector = torch.as_tensor(vector, dtype=param.dtype, device=param.device)

        if self.vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {self.vector.shape}")

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size and self.vector.numel() != hidden_size:
            raise ValueError(
                f"Vector length {self.vector.numel()} != model hidden_size {hidden_size}"
            )

        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

    def _get_layer(self) -> torch.nn.Module:
        """Get the layer module to hook."""
        # Gemma uses model.model.layers[i]
        try:
            return self.model.model.layers[self.layer]
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Could not access layer {self.layer}: {e}")

    def _hook_fn(self, module, inputs, outputs):
        """Forward hook that adds steering vector to layer output."""
        steer = self.coefficient * self.vector

        def _add_to_tensor(t: torch.Tensor) -> torch.Tensor:
            if self.positions == "all":
                return t + steer.to(t.device)
            else:  # "last"
                t_new = t.clone()
                t_new[:, -1, :] += steer.to(t.device)
                return t_new

        # Handle tuple outputs (hidden_states, attention_weights, ...)
        if torch.is_tensor(outputs):
            return _add_to_tensor(outputs)
        elif isinstance(outputs, (tuple, list)):
            if not torch.is_tensor(outputs[0]):
                return outputs  # Unknown format, don't modify
            modified = _add_to_tensor(outputs[0])
            return (modified, *outputs[1:])
        else:
            return outputs  # Unknown type, don't modify

    def __enter__(self):
        layer = self._get_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()
            self._handle = None


@contextmanager
def steer(
    model: torch.nn.Module,
    vector: torch.Tensor,
    layer: int,
    coefficient: float = 1.0,
    positions: str = "all",
):
    """
    Convenience context manager for steering.

    Usage:
        with steer(model, vector, layer=16, coefficient=1.5):
            output = model.generate(**inputs)
    """
    hook = SteeringHook(model, vector, layer, coefficient, positions)
    with hook:
        yield hook
