"""
Hook management for transformer models.

HookManager: base for all hook registration (single source of truth)
LayerHook: single-layer hook base class (uses HookManager)
CaptureHook: capture activations from a layer
SteeringHook: add vectors to layer outputs
MultiLayerCapture: capture one component across many layers
"""

import torch
from typing import Any, Callable, List, Sequence, Union


# =============================================================================
# Path utilities
# =============================================================================

def detect_contribution_paths(model) -> dict:
    """Auto-detect where attention/MLP contributions come from based on model architecture.

    Key insight: Mistral's 'post_attention_layernorm' is actually a pre-norm for MLP!
    Only Gemma-2 has TRUE post-sublayer norms. Detection: check for 'pre_feedforward_layernorm'.

    Returns:
        Dict mapping 'attn_contribution' and 'mlp_contribution' to their submodule paths.
    """
    layer = model.model.layers[0]
    children = dict(layer.named_children())

    if 'pre_feedforward_layernorm' in children:
        # Gemma-2 pattern: has BOTH pre and post norms
        # The post norms scale outputs before residual addition
        return {
            'attn_contribution': 'post_attention_layernorm',
            'mlp_contribution': 'post_feedforward_layernorm',
        }
    else:
        # Standard pre-norm only (Llama, Mistral, Qwen, etc.)
        # Attention/MLP outputs go directly to residual without post-scaling
        return {
            'attn_contribution': 'self_attn.o_proj',
            'mlp_contribution': 'mlp.down_proj',
        }


def get_hook_path(layer: int, component: str = "residual", prefix: str = None, model=None) -> str:
    """
    Convert layer + component to string path for hooking.

    Args:
        layer: Layer index (0-indexed)
        component: One of:
            - "residual": Layer output (accumulated state)
            - "attn_out": Raw attention output (before any post-norm)
            - "mlp_out": Raw MLP output (before any post-norm)
            - "attn_contribution": What attention actually adds to residual (requires model)
            - "mlp_contribution": What MLP actually adds to residual (requires model)
            - "k_proj", "v_proj": Key/value projections
        prefix: Path prefix to layers (auto-detected if model provided, else "model.layers")
        model: Used for auto-detecting prefix and architecture

    Returns:
        String path like "model.layers.16" or "model.layers.16.self_attn.o_proj"

    Example:
        >>> get_hook_path(16)
        'model.layers.16'
        >>> get_hook_path(16, "attn_out")
        'model.layers.16.self_attn.o_proj'
        >>> get_hook_path(16, "attn_contribution", model=model)  # Auto-detects post-norm if present
        'model.layers.16.post_attention_layernorm'  # Gemma-2
        'model.layers.16.self_attn.o_proj'          # Llama/Mistral/Qwen
    """
    # Auto-detect prefix if model provided and prefix not specified
    if prefix is None:
        if model is not None:
            from utils.model import get_layer_path_prefix
            prefix = get_layer_path_prefix(model)
        else:
            prefix = "model.layers"
    # Static paths (don't require model)
    paths = {
        'residual': f"{prefix}.{layer}",
        'attn_out': f"{prefix}.{layer}.self_attn.o_proj",
        'mlp_out': f"{prefix}.{layer}.mlp.down_proj",
        'k_proj': f"{prefix}.{layer}.self_attn.k_proj",
        'v_proj': f"{prefix}.{layer}.self_attn.v_proj",
    }

    # Dynamic paths (require model for architecture detection)
    if component in ('attn_contribution', 'mlp_contribution'):
        if model is None:
            raise ValueError(f"Component '{component}' requires model parameter for architecture detection")
        contrib_paths = detect_contribution_paths(model)
        paths['attn_contribution'] = f"{prefix}.{layer}.{contrib_paths['attn_contribution']}"
        paths['mlp_contribution'] = f"{prefix}.{layer}.{contrib_paths['mlp_contribution']}"

    if component not in paths:
        raise ValueError(f"Unknown component: {component}. Valid: {list(paths.keys())}")
    return paths[component]


# =============================================================================
# HookManager - base for all hook registration
# =============================================================================

class HookManager:
    """
    Base for all hook registration. Single source of truth for path navigation.

    All other hook classes use HookManager internally.

    Usage:
        with HookManager(model) as hooks:
            hooks.add_forward_hook("model.layers.16", my_hook_fn)
            hooks.add_forward_hook("model.embed_tokens", another_hook_fn)
            output = model.generate(input_ids)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _navigate_path(self, path: str) -> torch.nn.Module:
        """Navigate to module using dot-separated path."""
        module = self.model
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def add_forward_hook(
        self,
        path: str,
        hook_fn: Callable[[torch.nn.Module, Any, Any], Any],
    ) -> torch.utils.hooks.RemovableHandle:
        """Add forward hook to module at dot-separated path."""
        module = self._navigate_path(path)
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> 'HookManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_all()


# =============================================================================
# LayerHook - single-layer hook base class
# =============================================================================

class LayerHook:
    """
    Base class for hooks on a single layer. Uses HookManager internally.

    Takes a string path (e.g., "model.layers.16") and handles:
    - Registering/removing the hook via HookManager
    - Context manager protocol

    Subclasses implement _hook_fn to define what happens when the hook fires.
    """

    def __init__(self, model: torch.nn.Module, path: str):
        """
        Args:
            model: The transformer model
            path: Dot-separated path like "model.layers.16"
        """
        self.path = path
        self._manager = HookManager(model)

    def _hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any) -> Any:
        """
        Called when the hooked module runs.

        Args:
            module: The module that fired
            inputs: Tuple of inputs to the module
            outputs: Output from the module (tensor or tuple)

        Returns:
            Modified outputs, or None to leave unchanged
        """
        raise NotImplementedError("Subclasses must implement _hook_fn")

    def __enter__(self):
        self._manager.add_forward_hook(self.path, self._hook_fn)
        return self

    def __exit__(self, *exc):
        self._manager.remove_all()


# =============================================================================
# CaptureHook - capture from single layer
# =============================================================================

class CaptureHook(LayerHook):
    """
    Capture activations from a single layer.

    Usage:
        with CaptureHook(model, "model.layers.16") as hook:
            model(**inputs)
        activations = hook.get()  # [batch, seq, hidden]

        # Or with helper:
        with CaptureHook(model, get_hook_path(16, "attn_out")) as hook:
            model(**inputs)
        activations = hook.get()
    """

    def __init__(self, model: torch.nn.Module, path: str, keep_on_gpu: bool = False):
        super().__init__(model, path)
        self.captured: List[torch.Tensor] = []
        self.keep_on_gpu = keep_on_gpu

    def _hook_fn(self, module, inputs, outputs):
        """Capture output tensor, don't modify."""
        if isinstance(outputs, tuple):
            tensor = outputs[0]
        else:
            tensor = outputs
        captured = tensor.detach() if self.keep_on_gpu else tensor.detach().cpu()
        self.captured.append(captured)
        return None  # don't modify

    def get(self, concat: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get captured activations.

        Args:
            concat: If True, concatenate along batch dim. If False, return list.

        Returns:
            Tensor [total_batch, seq, hidden] or list of tensors
        """
        if not self.captured:
            raise ValueError(f"No activations captured for path '{self.path}'")
        if concat:
            return torch.cat(self.captured, dim=0)
        return self.captured

    def clear(self):
        """Clear captured activations."""
        self.captured = []


# =============================================================================
# SteeringHook - add vector to layer output
# =============================================================================

class SteeringHook(LayerHook):
    """
    Add (coefficient * vector) to a layer's output during forward pass.

    Usage:
        vector = torch.load('vectors/probe_layer16.pt')

        # Explicit path
        with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
            output = model.generate(**inputs)

        # Or with helper
        with SteeringHook(model, vector, get_hook_path(16), coefficient=1.5):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vector: Union[torch.Tensor, Sequence[float]],
        path: str,
        coefficient: float = 1.0,
    ):
        super().__init__(model, path)
        self.coefficient = float(coefficient)

        # Convert vector to tensor on model device/dtype
        param = next(model.parameters())
        self.vector = torch.as_tensor(vector, dtype=param.dtype, device=param.device)

        if self.vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {self.vector.shape}")

    def _hook_fn(self, module, inputs, outputs):
        """Add steering vector to output. Moves vector to output device for multi-GPU."""
        device = outputs[0].device if isinstance(outputs, tuple) else outputs.device
        steer = (self.coefficient * self.vector).to(device)

        if torch.is_tensor(outputs):
            return outputs + steer
        elif isinstance(outputs, tuple) and torch.is_tensor(outputs[0]):
            return (outputs[0] + steer, *outputs[1:])
        return outputs


# =============================================================================
# MultiLayerCapture - one component across many layers
# =============================================================================

class MultiLayerCapture:
    """
    Capture activations from multiple layers in one forward pass.

    Uses HookManager internally to register CaptureHooks.

    Usage:
        # Specific layers
        with MultiLayerCapture(model, layers=[14, 15, 16]) as capture:
            model(**inputs)
        acts_16 = capture.get(16)

        # All layers
        with MultiLayerCapture(model) as capture:  # layers=None means all
            model(**inputs)
        all_acts = capture.get_all()  # {0: tensor, 1: tensor, ...}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int] = None,
        component: str = "residual",
        prefix: str = None,
        keep_on_gpu: bool = False,
    ):
        """
        Args:
            model: The transformer model
            layers: List of layer indices, or None for all layers
            component: "residual", "attn_out", "mlp_out", "attn_contribution", "mlp_contribution", etc.
            prefix: Path prefix (auto-detected if None)
            keep_on_gpu: If True, keep captured tensors on GPU (faster for batch processing)
        """
        # Auto-detect prefix for different model types
        if prefix is None:
            from utils.model import get_layer_path_prefix
            prefix = get_layer_path_prefix(model)

        if layers is None:
            # Handle nested text_config for multimodal models (e.g., Gemma 3)
            config = model.config
            if hasattr(config, 'text_config'):
                config = config.text_config
            layers = list(range(config.num_hidden_layers))

        self._hooks = {
            layer: CaptureHook(model, get_hook_path(layer, component, prefix, model=model), keep_on_gpu=keep_on_gpu)
            for layer in layers
        }

    def get(self, layer: int) -> torch.Tensor:
        """Get activations for one layer."""
        if layer not in self._hooks:
            raise KeyError(f"Layer {layer} not captured. Available: {list(self._hooks.keys())}")
        return self._hooks[layer].get()

    def get_all(self) -> dict:
        """Get dict of all layers: {layer: tensor}"""
        return {layer: hook.get() for layer, hook in self._hooks.items()}

    def clear(self):
        """Clear all captured activations."""
        for hook in self._hooks.values():
            hook.clear()

    def __enter__(self):
        for hook in self._hooks.values():
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in self._hooks.values():
            hook.__exit__(*exc)
