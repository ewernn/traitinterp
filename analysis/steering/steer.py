"""
Multi-layer steering composition helpers.

Uses core primitives for atomic operations.
"""

import torch
from typing import Dict, List, Tuple

from core import SteeringHook, HookManager, get_hook_path, orthogonalize, VectorSpec


def orthogonalize_vectors(vectors: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Orthogonalize each layer's vector to previous layer. Uses core.orthogonalize."""
    layers = sorted(vectors.keys())
    result = {}
    for i, layer in enumerate(layers):
        if i == 0:
            result[layer] = vectors[layer]
        else:
            result[layer] = orthogonalize(vectors[layer], vectors[layers[i - 1]])
    return result


class MultiLayerSteeringHook:
    """Steer multiple layers simultaneously. Uses core.SteeringHook instances."""

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[Tuple[int, torch.Tensor, float]],
        component: str = "residual",
    ):
        """
        Args:
            model: The transformer model
            configs: List of (layer, vector, coefficient) tuples
            component: "residual", "attn_out", etc.
        """
        self._hooks = [
            SteeringHook(model, vector, get_hook_path(layer, component, model=model), coefficient)
            for layer, vector, coefficient in configs
        ]

    @classmethod
    def from_vector_specs(
        cls,
        model: torch.nn.Module,
        specs: List[VectorSpec],
        vectors: Dict[Tuple[int, str], torch.Tensor],
    ) -> "MultiLayerSteeringHook":
        """
        Create from VectorSpecs and pre-loaded vectors.

        Args:
            model: The transformer model
            specs: List of VectorSpec (layer, component, position, method, weight)
            vectors: Dict mapping (layer, component) to loaded vector tensors

        Returns:
            MultiLayerSteeringHook instance

        Example:
            specs = [VectorSpec(9, 'residual', 'response[:]', 'probe', 0.9)]
            vectors = {(9, 'residual'): loaded_vector}
            hook = MultiLayerSteeringHook.from_vector_specs(model, specs, vectors)
        """
        # Group by component (all specs should have same component for now)
        components = set(s.component for s in specs)
        if len(components) > 1:
            raise ValueError("All VectorSpecs must have same component for MultiLayerSteeringHook")
        component = specs[0].component

        configs = []
        for spec in specs:
            vec = vectors.get((spec.layer, spec.component))
            if vec is None:
                raise KeyError(f"No vector loaded for layer={spec.layer}, component={spec.component}")
            configs.append((spec.layer, vec, spec.weight))

        return cls(model, configs, component)

    def __enter__(self):
        for hook in self._hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in reversed(self._hooks):
            hook.__exit__(*exc)


class BatchedLayerSteeringHook:
    """
    Batched steering with different vectors per batch slice.

    Use when you need different steering for different items in a batch
    (e.g., A/B testing steering vs no-steering in same forward pass).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[Tuple[int, torch.Tensor, float, Tuple[int, int]]],
        component: str = "residual",
    ):
        """
        Args:
            model: The transformer model
            configs: List of (layer, vector, coefficient, (batch_start, batch_end)) tuples
            component: "residual", "attn_out", etc.
        """
        self.model = model
        self.component = component.lower()
        self._manager = None

        param = next(model.parameters())
        self._layer_configs: Dict[int, List[Tuple[torch.Tensor, float, Tuple[int, int]]]] = {}
        for layer_idx, vector, coef, batch_slice in configs:
            vec = torch.as_tensor(vector, dtype=param.dtype, device=param.device)
            if layer_idx not in self._layer_configs:
                self._layer_configs[layer_idx] = []
            self._layer_configs[layer_idx].append((vec, float(coef), batch_slice))

    def _make_hook(self, layer_configs: List[Tuple[torch.Tensor, float, Tuple[int, int]]]):
        def hook_fn(module, inputs, outputs):
            t = outputs[0] if isinstance(outputs, tuple) else outputs
            t_new = t.clone()
            for vec, coef, (batch_start, batch_end) in layer_configs:
                t_new[batch_start:batch_end] = t_new[batch_start:batch_end] + coef * vec.to(t.device)
            if isinstance(outputs, tuple):
                return (t_new, *outputs[1:])
            return t_new
        return hook_fn

    def __enter__(self):
        self._manager = HookManager(self.model)
        for layer_idx, layer_configs in self._layer_configs.items():
            path = get_hook_path(layer_idx, self.component, model=self.model)
            self._manager.add_forward_hook(path, self._make_hook(layer_configs))
        return self

    def __exit__(self, *exc):
        if self._manager:
            self._manager.remove_all()
            self._manager = None
