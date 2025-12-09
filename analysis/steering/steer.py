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

        # Validate vector dimension based on component
        # k_cache/v_cache use head_dim * num_kv_heads (1024 for Gemma 2B)
        # residual/attn_out/mlp_out use hidden_size (2304 for Gemma 2B)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size:
            if self.component in {"k_cache", "v_cache"}:
                # KV projection size = head_dim * num_key_value_heads
                num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
                head_dim = getattr(model.config, "head_dim", hidden_size // 8)
                expected_size = num_kv_heads * head_dim
                if self.vector.numel() != expected_size:
                    raise ValueError(
                        f"Vector length {self.vector.numel()} != expected KV size {expected_size} "
                        f"(num_kv_heads={num_kv_heads}, head_dim={head_dim})"
                    )
            elif self.vector.numel() != hidden_size:
                raise ValueError(
                    f"Vector length {self.vector.numel()} != model hidden_size {hidden_size}"
                )

        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

        if self.component not in {"residual", "attn_out", "mlp_out", "k_cache", "v_cache"}:
            raise ValueError("component must be 'residual', 'attn_out', 'mlp_out', 'k_cache', or 'v_cache'")

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
            elif self.component == "k_cache":
                return layer.self_attn.k_proj
            elif self.component == "v_cache":
                return layer.self_attn.v_proj
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
            component: Which component to steer ("residual", "attn_out", "mlp_out", "k_cache", "v_cache")
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


class BatchedLayerSteeringHook:
    """
    Context manager for batched generation with different steering per batch slice.

    Each layer steers a specific slice of the batch, allowing parallel evaluation
    of multiple layers in a single forward pass.

    Example:
        # 4 layers × 5 questions = 20 batch items
        # Layer 16 steers items 0-4, layer 17 steers items 5-9, etc.
        configs = [
            (16, vector16, 100.0, (0, 5)),   # (layer, vector, coef, batch_slice)
            (17, vector17, 120.0, (5, 10)),
            (18, vector18, 80.0, (10, 15)),
            (19, vector19, 150.0, (15, 20)),
        ]
        with BatchedLayerSteeringHook(model, configs):
            outputs = model.generate(**batched_inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[Tuple[int, torch.Tensor, float, Tuple[int, int]]],
        positions: str = "all",
        component: str = "residual",
    ):
        """
        Args:
            model: The transformer model
            configs: List of (layer_idx, vector, coefficient, (batch_start, batch_end)) tuples
            positions: Which positions to steer ("all" or "last")
            component: Which component to steer
        """
        self.model = model
        self.positions = positions.lower()
        self.component = component.lower()
        self._handles = []

        # Convert vectors to model device/dtype
        param = next(model.parameters())

        # Group configs by layer (a layer might have multiple batch slices)
        self._layer_configs: Dict[int, List[Tuple[torch.Tensor, float, Tuple[int, int]]]] = {}
        for layer_idx, vector, coef, batch_slice in configs:
            vec = torch.as_tensor(vector, dtype=param.dtype, device=param.device)
            if layer_idx not in self._layer_configs:
                self._layer_configs[layer_idx] = []
            self._layer_configs[layer_idx].append((vec, float(coef), batch_slice))

    def _get_layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the layer module to hook based on component."""
        layer = self.model.model.layers[layer_idx]
        if self.component == "residual":
            return layer
        elif self.component == "attn_out":
            return layer.self_attn.o_proj
        elif self.component == "mlp_out":
            return layer.mlp.down_proj
        elif self.component == "k_cache":
            return layer.self_attn.k_proj
        elif self.component == "v_cache":
            return layer.self_attn.v_proj
        raise ValueError(f"Unknown component: {self.component}")

    def _make_hook(self, layer_configs: List[Tuple[torch.Tensor, float, Tuple[int, int]]]):
        """Create a hook function for a specific layer."""
        positions = self.positions

        def hook_fn(module, inputs, outputs):
            def _modify_tensor(t: torch.Tensor) -> torch.Tensor:
                t_new = t.clone()
                for vec, coef, (batch_start, batch_end) in layer_configs:
                    steer = coef * vec.to(t.device)
                    if positions == "all":
                        t_new[batch_start:batch_end] = t_new[batch_start:batch_end] + steer
                    else:  # "last"
                        t_new[batch_start:batch_end, -1, :] += steer
                return t_new

            if torch.is_tensor(outputs):
                return _modify_tensor(outputs)
            elif isinstance(outputs, (tuple, list)):
                if not torch.is_tensor(outputs[0]):
                    return outputs
                modified = _modify_tensor(outputs[0])
                return (modified, *outputs[1:])
            return outputs

        return hook_fn

    def __enter__(self):
        for layer_idx, layer_configs in self._layer_configs.items():
            module = self._get_layer_module(layer_idx)
            handle = module.register_forward_hook(self._make_hook(layer_configs))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def estimate_vram_gb(
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    model_size_gb: float = 5.0,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate VRAM usage for batched generation.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head
        batch_size: Total batch size
        seq_len: Maximum sequence length (prompt + generated)
        model_size_gb: Base model size in GB (default 5.0 for Gemma 2B bf16)
        dtype_bytes: Bytes per element (2 for bf16/fp16)

    Returns:
        Estimated VRAM in GB
    """
    # KV cache: 2 (K,V) × num_kv_heads × head_dim × seq_len × batch × layers × dtype
    kv_cache_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes

    # Activation buffer (rough estimate): hidden_size × batch × seq_len × dtype × multiplier
    activation_bytes = hidden_size * batch_size * seq_len * dtype_bytes * 4  # 4x for intermediate

    total_bytes = kv_cache_bytes + activation_bytes
    total_gb = total_bytes / (1024 ** 3)

    return model_size_gb + total_gb


def calculate_max_batch_size(
    model,
    available_vram_gb: float,
    seq_len: int = 160,
    model_size_gb: float = 5.0,
) -> int:
    """
    Calculate maximum batch size that fits in available VRAM.

    Args:
        model: The transformer model (to get config)
        available_vram_gb: Available VRAM in GB
        seq_len: Expected max sequence length
        model_size_gb: Base model size

    Returns:
        Maximum safe batch size
    """
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_kv_heads = getattr(config, "num_key_value_heads", 4)
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

    # Binary search for max batch size
    low, high = 1, 256
    while low < high:
        mid = (low + high + 1) // 2
        vram = estimate_vram_gb(
            num_layers, hidden_size, num_kv_heads, head_dim,
            mid, seq_len, model_size_gb
        )
        if vram <= available_vram_gb * 0.85:  # 85% safety margin
            low = mid
        else:
            high = mid - 1

    return low
