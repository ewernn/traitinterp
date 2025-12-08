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
    # Single-layer steering
    vector = torch.load('experiments/exp/extraction/trait/vectors/probe_layer16.pt')
    with SteeringHook(model, vector, layer=16, coefficient=1.5):
        output = model.generate(**inputs)

    # Multi-layer steering
    configs = [
        (12, vector12, 1.0),
        (14, vector14, 2.0),
        (16, vector16, 1.0),
    ]
    with MultiLayerSteeringHook(model, configs):
        output = model.generate(**inputs)
"""

import torch
from contextlib import contextmanager
from typing import Union, Sequence, List, Tuple, Dict


def orthogonalize_vectors(vectors: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """
    Make each vector orthogonal to the previous layer's vector.

    v_ℓ_orth = v_ℓ - proj(v_ℓ onto v_{ℓ-1})
             = v_ℓ - (v_ℓ · v_{ℓ-1} / ||v_{ℓ-1}||²) * v_{ℓ-1}

    Args:
        vectors: Dict mapping layer index to vector tensor

    Returns:
        Dict mapping layer index to orthogonalized vector
    """
    layers = sorted(vectors.keys())
    result = {}

    for i, layer in enumerate(layers):
        v = vectors[layer]
        if i == 0:
            # First layer: no previous vector, use as-is
            result[layer] = v
        else:
            prev_layer = layers[i - 1]
            v_prev = vectors[prev_layer]
            # Project v onto v_prev and subtract
            dot = torch.dot(v.flatten(), v_prev.flatten())
            norm_sq = torch.dot(v_prev.flatten(), v_prev.flatten())
            if norm_sq > 1e-10:
                proj = (dot / norm_sq) * v_prev
                result[layer] = v - proj
            else:
                result[layer] = v

    return result


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
        component: str = "residual",
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
            component: Which component to steer:
                - "residual": Full layer output (default)
                - "attn_out": Attention output (o_proj)
                - "mlp_out": MLP output (down_proj)
        """
        self.model = model
        self.layer = layer
        self.coefficient = float(coefficient)
        self.positions = positions.lower()
        self.component = component.lower()
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

        if self.component not in {"residual", "attn_out", "mlp_out"}:
            raise ValueError("component must be 'residual', 'attn_out', or 'mlp_out'")

    def _get_layer(self) -> torch.nn.Module:
        """Get the layer module to hook based on component."""
        try:
            layer = self.model.model.layers[self.layer]
            if self.component == "residual":
                return layer
            elif self.component == "attn_out":
                return layer.self_attn.o_proj
            elif self.component == "mlp_out":
                return layer.mlp.down_proj
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Could not access layer {self.layer} component {self.component}: {e}")

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
    component: str = "residual",
):
    """
    Convenience context manager for steering.

    Usage:
        with steer(model, vector, layer=16, coefficient=1.5):
            output = model.generate(**inputs)

        # Steer MLP output specifically
        with steer(model, vector, layer=10, coefficient=1.0, component="mlp_out"):
            output = model.generate(**inputs)
    """
    hook = SteeringHook(model, vector, layer, coefficient, positions, component)
    with hook:
        yield hook


class MultiLayerSteeringHook:
    """
    Context manager for steering multiple layers simultaneously.

    Each layer can have its own vector and coefficient.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[Tuple[int, torch.Tensor, float]],
        positions: str = "all",
        component: str = "residual",
    ):
        """
        Args:
            model: The transformer model
            configs: List of (layer_idx, vector, coefficient) tuples
            positions: Which positions to steer ("all" or "last")
            component: Which component to steer ("residual", "attn_out", "mlp_out")
        """
        self.model = model
        self.positions = positions.lower()
        self.component = component.lower()
        self._hooks: List[SteeringHook] = []

        # Create a SteeringHook for each layer config
        for layer_idx, vector, coefficient in configs:
            hook = SteeringHook(
                model=model,
                vector=vector,
                layer=layer_idx,
                coefficient=coefficient,
                positions=positions,
                component=component,
            )
            self._hooks.append(hook)

    def __enter__(self):
        # Enter all hooks
        for hook in self._hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        # Exit all hooks (in reverse order for clean teardown)
        for hook in reversed(self._hooks):
            hook.__exit__(*exc)
