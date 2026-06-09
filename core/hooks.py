"""
Hook management for transformer models.

HookManager: base for all hook registration (single source of truth for path navigation)
LayerHook: single-layer hook base class (uses HookManager)
CaptureHook: capture activations from a layer (shape-agnostic)
SteeringHook: add vectors to layer outputs (residual-shape: [batch, seq, hidden])
MultiLayerCapture: capture one component across many layers (uses Architecture registry)
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import torch

from core.architectures import UnsupportedComponentError, get_architecture


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

    def add_forward_hook(
        self,
        path: str,
        hook_fn: Callable[[torch.nn.Module, Any, Any], Any],
    ) -> torch.utils.hooks.RemovableHandle:
        """Add forward hook to module at dot-separated path."""
        module = self.model.get_submodule(path)
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
# Path resolution (one-shot helper; if you have many lookups, hoist arch yourself)
# =============================================================================

def resolve_hook_path(model: torch.nn.Module, layer: int, component: str) -> str:
    """One-shot path lookup. Equivalent to get_architecture(model).path(component, layer, model=model).

    Raises UnsupportedComponentError if the component is not exposed at this layer.
    For multiple lookups, hoist the architecture once and call arch.path(...) directly.
    """
    return get_architecture(model).path(component, layer, model=model)


# =============================================================================
# CaptureHook - capture from single layer (shape-agnostic)
# =============================================================================

class CaptureHook(LayerHook):
    """
    Capture activations from a single layer.

    Shape-agnostic: stores whatever tensor the hooked module emits. The standard
    case is residual-stream shape [batch, seq, hidden], but the same machinery
    works for per-head [batch, seq, n_heads, d_head], attention patterns
    [batch, n_heads, q, k], or any other tensor shape. Downstream consumers
    should assert their own shape requirements.

    Usage:
        with CaptureHook(model, "model.layers.16") as hook:
            model(**inputs)
        activations = hook.get()  # whatever shape the module emitted
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
            Tensor of whatever shape the module emitted (concatenated along dim 0)
            or list of per-call tensors.
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
# SteeringHook - add vector to layer output (residual-shape only)
# =============================================================================

def _assert_residual_shape(tensor: torch.Tensor, hook_name: str) -> None:
    """Transform hooks (Steering/Ablation/Capping) require [batch, seq, hidden].

    For per-head or attention-pattern hooks, use a different hook class - the
    additive math here only makes sense for residual-shape activations.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"{hook_name} requires [batch, seq, hidden] activations; got shape "
            f"{tuple(tensor.shape)}. Per-head and attention-pattern components are "
            f"capture-only - use a separate hook class for transforms on those."
        )


def _norm_match_scaled(vector: torch.Tensor, target: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    """Rescale a 1-D steering vector to match per-token L2 norm of target.

    Returns a tensor with shape [batch, seq, hidden] that, when added to target,
    contributes magnitude `||target_t||` along the unit-direction of `vector`
    at each token position t. Norm computed in float32 for stability, output
    cast back to target dtype.

    Args:
        vector: 1-D steering direction, shape [hidden].
        target: residual-shape tensor [batch, seq, hidden] to match norms against.
        eps: guard against zero-norm vector denominator (residual zero is fine -
            just contributes zero).
    """
    r_norm = target.float().norm(dim=-1, keepdim=True)        # [batch, seq, 1]
    v_norm = vector.float().norm()                            # scalar
    scale = r_norm / (v_norm + eps)                           # [batch, seq, 1]
    return (vector.float() * scale).to(dtype=target.dtype)    # [batch, seq, hidden]


class SteeringHook(LayerHook):
    """
    Add (coefficient * vector) to a layer's output during forward pass.

    Residual-shape only ([batch, seq, hidden]). Raises if the hooked tensor is
    not 3D (e.g., if you accidentally hook a per-head component).

    Usage:
        vector = torch.load('vectors/probe_layer16.pt')
        with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vector: Union[torch.Tensor, Sequence[float]],
        path: str,
        coefficient: float = 1.0,
        norm_match: bool = False,
    ):
        super().__init__(model, path)
        self.coefficient = float(coefficient)
        self.norm_match = bool(norm_match)

        # Keep vector in float32 for precision, cast to model dtype after scaling
        param = next(model.parameters())
        self.vector = torch.as_tensor(vector, dtype=torch.float32, device=param.device)

        if self.vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {self.vector.shape}")

    def _hook_fn(self, module, inputs, outputs):
        """Add steering vector to output. Multiplies in float32 for precision, then casts to output dtype."""
        out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        _assert_residual_shape(out_tensor, "SteeringHook")
        if self.norm_match:
            # Per-token: scale vector to match ||residual_t|| before applying coefficient.
            scaled = _norm_match_scaled(self.vector, out_tensor)
            steer = (self.coefficient * scaled.float()).to(dtype=out_tensor.dtype)
        else:
            # Multiply in float32, then cast to output dtype for the addition
            steer = (self.coefficient * self.vector).to(device=out_tensor.device, dtype=out_tensor.dtype)

        if torch.is_tensor(outputs):
            return outputs + steer
        elif isinstance(outputs, tuple) and torch.is_tensor(outputs[0]):
            return (outputs[0] + steer, *outputs[1:])
        return outputs


class PerPositionSteeringHook(SteeringHook):
    """Steer only at specific token positions within the sequence.

    Useful for experiments that need to steer on a subset of tokens
    (e.g., activity-description tokens in preference experiments).

    Usage:
        with PerPositionSteeringHook(model, vector, path, coefficient=1.5, token_range=(12, 18)):
            output = model.generate(**inputs)
    """

    def __init__(self, model, vector, path, coefficient=1.0, token_range=None,
                 norm_match=False):
        super().__init__(model, vector, path, coefficient, norm_match=norm_match)
        self.token_range = token_range  # (start, end) or None for all positions

    def _hook_fn(self, module, inputs, outputs):
        if self.token_range is None:
            return super()._hook_fn(module, inputs, outputs)

        out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        _assert_residual_shape(out_tensor, "PerPositionSteeringHook")

        start, end = self.token_range
        result = out_tensor.clone()
        if self.norm_match:
            sliced = result[:, start:end]
            scaled = _norm_match_scaled(self.vector, sliced)
            result[:, start:end] = sliced + (self.coefficient * scaled.float()).to(dtype=out_tensor.dtype)
        else:
            steer = (self.coefficient * self.vector).to(device=out_tensor.device, dtype=out_tensor.dtype)
            result[:, start:end] = result[:, start:end] + steer

        if torch.is_tensor(outputs):
            return result
        elif isinstance(outputs, tuple):
            return (result, *outputs[1:])
        return outputs


# =============================================================================
# AblationHook - project out direction from layer output (residual-shape only)
# =============================================================================

class AblationHook(LayerHook):
    """
    Project out a direction from a layer's output during forward pass.

    Implements directional ablation: x' = x - (x · r̂) * r̂.
    Residual-shape only ([batch, seq, hidden]).

    Usage:
        direction = torch.load('vectors/mean_diff_layer16.pt')
        with AblationHook(model, direction, "model.layers.16"):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        direction: Union[torch.Tensor, Sequence[float]],
        path: str,
    ):
        super().__init__(model, path)

        param = next(model.parameters())
        direction = torch.as_tensor(direction, dtype=torch.float32, device=param.device)

        if direction.ndim != 1:
            raise ValueError(f"Direction must be 1-D, got shape {direction.shape}")

        # Normalize to unit vector (fail fast on zero vector)
        norm = direction.norm()
        if norm < 1e-8:
            raise ValueError("Direction vector has near-zero norm, cannot normalize")
        self.direction = direction / norm

    def _hook_fn(self, module, inputs, outputs):
        """Project out direction from output: x' = x - (x · r̂) * r̂"""
        out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        _assert_residual_shape(out_tensor, "AblationHook")

        # Cast direction to output dtype for computation
        r_hat = self.direction.to(device=out_tensor.device, dtype=out_tensor.dtype)

        # Compute projection: (x · r̂) gives scalar per position, then scale by r̂
        # out_tensor: [batch, seq, hidden], r_hat: [hidden]
        proj_coef = out_tensor @ r_hat  # [batch, seq]
        proj = proj_coef.unsqueeze(-1) * r_hat  # [batch, seq, hidden]

        ablated = out_tensor - proj

        if torch.is_tensor(outputs):
            return ablated
        elif isinstance(outputs, tuple):
            return (ablated, *outputs[1:])
        return outputs


# =============================================================================
# MultiLayerAblation - ablate direction across all layers
# =============================================================================

class MultiLayerAblation:
    """
    Ablate a direction across multiple layers simultaneously.

    Usage:
        direction = torch.load('vectors/mean_diff_layer16.pt')
        with MultiLayerAblation(model, direction):                    # all layers
            output = model.generate(**inputs)
        with MultiLayerAblation(model, direction, layers=[10, 11, 12]):  # specific layers
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        direction: Union[torch.Tensor, Sequence[float]],
        layers: List[int] = None,
        component: str = "residual",
    ):
        arch = get_architecture(model)
        if layers is None:
            layers = list(range(len(arch.layers(model))))

        self._hooks = [
            AblationHook(model, direction, arch.path(component, layer, model=model))
            for layer in layers
        ]

    def __enter__(self):
        for hook in self._hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in reversed(self._hooks):
            hook.__exit__(*exc)


# =============================================================================
# MultiLayerCapture - one component across many layers
# =============================================================================

class MultiLayerCapture:
    """
    Capture activations from multiple layers in one forward pass.

    Path resolution goes through the Architecture registry. Layers that don't
    expose the requested component (e.g., k_proj on Qwen3.5 linear-attn layers,
    k_proj on DeepSeek V3 MLA, mlp.down_proj on a MoE layer) are skipped with a
    note rather than raising - the alternative is forcing every caller to filter
    layers up front.

    Usage:
        with MultiLayerCapture(model, layers=[14, 15, 16]) as capture:
            model(**inputs)
        acts_16 = capture.get(16)

        with MultiLayerCapture(model) as capture:  # layers=None means all
            model(**inputs)
        all_acts = capture.get_all()  # {0: tensor, 1: tensor, ...}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int] = None,
        component: str = "residual",
        keep_on_gpu: bool = False,
    ):
        arch = get_architecture(model)
        if layers is None:
            layers = list(range(len(arch.layers(model))))

        self._hooks: Dict[int, CaptureHook] = {}
        self._skipped_layers: List[int] = []
        for layer in layers:
            try:
                path = arch.path(component, layer, model=model)
            except UnsupportedComponentError:
                self._skipped_layers.append(layer)
                continue
            self._hooks[layer] = CaptureHook(model, path, keep_on_gpu=keep_on_gpu)

        if not self._hooks:
            raise ValueError(
                f"Component {component!r} not available on any of the requested layers "
                f"(architecture: {type(arch).__name__}). "
                f"This may happen with hybrid architectures (Qwen3.5 linear-attn layers "
                f"don't have k_proj/v_proj) or MLA architectures (DeepSeek V3 / Kimi K2 "
                f"have no standard k_proj/v_proj)."
            )
        if self._skipped_layers:
            print(f"  [note] Skipped {len(self._skipped_layers)} layers without {component}: {self._skipped_layers}")

    @property
    def available_layers(self) -> List[int]:
        """Layers that have active hooks (excludes skipped layers)."""
        return list(self._hooks.keys())

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


# =============================================================================
# ProjectionHook - project onto trait vectors on GPU (no full capture)
# =============================================================================


class ProjectionHook(LayerHook):
    """Project activations onto pre-stacked trait vectors inside the hook.

    Instead of capturing full [batch, seq, hidden_dim] tensors and transferring
    them to CPU, this hook computes projections on-device and stores only the
    small score arrays. Eliminates the GPU-CPU transfer bottleneck.

    Shape contract: matmul-based, so the captured tensor's last dim must match
    `vectors.shape[-1]`. Works on any rank ≥ 2 (residual [batch, seq, hidden],
    flattened per-head [batch, seq, hidden], etc.).

    Usage:
        # Stack vectors for this layer: [n_vectors, hidden_dim]
        vectors = torch.stack([v1, v2, v3]).to(device)
        with ProjectionHook(model, "model.layers.16", vectors) as hook:
            model(**inputs)
        scores = hook.get_projections()  # [batch, seq, n_vectors]
        norms = hook.get_norms()         # [batch, seq]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        path: str,
        vectors: torch.Tensor,
        compute_norms: bool = True,
    ):
        """
        Args:
            model: The transformer model
            path: Hook path (e.g., "model.layers.16")
            vectors: Tensor[n_vectors, hidden_dim] - will be L2-normalized
            compute_norms: Also compute per-token ||h|| activation norms
        """
        super().__init__(model, path)
        # Normalize vectors for cosine-like projection
        norms = vectors.float().norm(dim=-1, keepdim=True)
        self._vectors = (vectors.float() / norms).to(next(model.parameters()).device)
        self._compute_norms = compute_norms
        self.projections: List[torch.Tensor] = []
        self.norms: List[torch.Tensor] = []

    def _hook_fn(self, module, inputs, outputs):
        tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        # Project on GPU: [..., hidden] @ [n_vectors, hidden].T → [..., n_vectors]
        scores = torch.matmul(tensor.float(), self._vectors.T)
        self.projections.append(scores.cpu())
        if self._compute_norms:
            self.norms.append(tensor.float().norm(dim=-1).cpu())
        return None

    def get_projections(self, concat: bool = True) -> torch.Tensor:
        if concat:
            return torch.cat(self.projections, dim=0)
        return self.projections

    def get_norms(self, concat: bool = True) -> torch.Tensor:
        if concat:
            return torch.cat(self.norms, dim=0)
        return self.norms

    def clear(self):
        self.projections = []
        self.norms = []


class MultiLayerProjection:
    """Project activations onto trait vectors across multiple layers in one pass.

    Groups vectors by layer, stacks them, and uses ProjectionHook per layer.
    Only the small score arrays cross the GPU-CPU boundary.

    Usage:
        vectors_by_layer = {
            16: torch.stack([vec_a, vec_b]),  # [2, hidden_dim]
            20: torch.stack([vec_c]),          # [1, hidden_dim]
        }
        with MultiLayerProjection(model, vectors_by_layer) as proj:
            model(**inputs)
        scores = proj.get_all()       # {16: [batch, seq, 2], 20: [batch, seq, 1]}
        norms = proj.get_all_norms()  # {16: [batch, seq], 20: [batch, seq]}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vectors_by_layer: Dict[int, torch.Tensor],
        component: str = "residual",
        compute_norms: bool = True,
    ):
        arch = get_architecture(model)
        self._hooks = {}
        for layer, vectors in vectors_by_layer.items():
            path = arch.path(component, layer, model=model)
            self._hooks[layer] = ProjectionHook(
                model, path, vectors, compute_norms=compute_norms,
            )

    def get_all(self) -> Dict[int, torch.Tensor]:
        """Get projection scores: {layer: [batch, seq, n_vectors]}"""
        return {layer: hook.get_projections() for layer, hook in self._hooks.items()}

    def get_all_norms(self) -> Dict[int, torch.Tensor]:
        """Get activation norms: {layer: [batch, seq]}"""
        return {layer: hook.get_norms() for layer, hook in self._hooks.items()}

    def clear(self):
        for hook in self._hooks.values():
            hook.clear()

    def __enter__(self):
        for hook in self._hooks.values():
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in self._hooks.values():
            hook.__exit__(*exc)


# =============================================================================
# MultiLayerSteering - steer multiple layers simultaneously
# =============================================================================

class MultiLayerSteering:
    """
    Steer multiple layers simultaneously with different vectors/coefficients.

    Usage:
        # Same component for all layers
        configs = [(14, vec_14, 1.2), (16, vec_16, 0.8)]
        with MultiLayerSteering(model, configs, component="residual"):
            output = model.generate(**inputs)

        # Per-config components (4-tuples)
        configs = [(14, vec_14, 1.2, "attn_contribution"), (16, vec_16, 0.8, "mlp_contribution")]
        with MultiLayerSteering(model, configs):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[tuple],  # (layer, vector, coef) or (layer, vector, coef, component)
        component: str = "residual",
        norm_match: bool = False,
    ):
        # Validate components before arch lookup so the error is clear even if
        # arch resolution fails for other reasons.
        if norm_match:
            for config in configs:
                comp = config[3] if len(config) == 4 else component
                if comp != "residual":
                    raise ValueError(
                        f"norm_match=True is only valid for component='residual'; got '{comp}'. "
                        f"Sub-component activations (attn/mlp) have different L2 scale than the "
                        f"residual stream and norm-matching against them would silently under-steer."
                    )

        arch = get_architecture(model)
        self._hooks = []
        for config in configs:
            if len(config) == 4:
                layer, vector, coefficient, comp = config
            else:
                layer, vector, coefficient = config
                comp = component
            self._hooks.append(
                SteeringHook(model, vector, arch.path(comp, layer, model=model),
                             coefficient, norm_match=norm_match)
            )

    def __enter__(self):
        for hook in self._hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in reversed(self._hooks):
            hook.__exit__(*exc)


class ActivationCappingHook(LayerHook):
    """
    Activation capping: ensure projections onto a direction stay within bounds.

    Floor mode (default): forces `<h, v_hat>` to be at least `effective_tau`.
    Ceiling mode: forces `<h, v_hat>` to be at most `effective_tau`.

    The hook unit-normalizes `direction` internally, then computes
    `proj = h @ v_hat` per token, then clamps. The orthogonal component of
    `h` is preserved. From Lu et al. (2026), "The Assistant Axis."

    Residual-shape only ([batch, seq, hidden]). Fires per-token at every
    sequence position; no response/prompt-token masking.

    tau_mode (what `tau` actually means)
    ------------------------------------
    Three modes, picked smartly by default:

      - "cosine" (default when `mean_activation_norm` is NOT given):
        `tau` is a per-token cosine fraction in [-1, +1]. Internally the hook
        rescales to raw per-token via `effective_tau_t = tau * ||h_t||`. So
        `tau = 0.4` means "force this token's projection to be at least 0.4
        times its own L2 norm." Model-independent and intuitive.

      - "calibrated" (default when `mean_activation_norm` IS given):
        `tau` is a fraction of a precomputed per-layer mean activation norm.
        Internally `effective_tau = tau * mean_activation_norm` (a constant
        per layer, not per token). So `tau = 1.0` means "one typical residual
        norm at this layer." This matches how steering coefficients are
        expressed elsewhere in this codebase (see `coefficient_search.py`
        `base_coef = mean_activation_norm`).

      - "raw" (must be requested explicitly):
        `tau` is the absolute raw projection value. No rescaling. Use this
        only when interoperating with externally-provided raw thresholds
        (e.g., Lu et al.'s `lu-christina/assistant-axis-vectors` capping
        config.pt, which stores per-layer p25 raw projections).

    Sign convention
    ---------------
    For `axis = default_mean - role_mean` (the Lu et al. convention),
    positive projection means assistant-mode, negative means persona-mode.

      - Floor mode with positive tau forces the residual TOWARD assistant.
        Used for safety capping: keep the model in assistant mode under
        jailbreak pressure.
      - Ceiling mode with negative tau forces the residual TOWARD persona.
        Used for persona elicitation: force the model into a character.

    If the user supplied `axis = role - default` instead, both directions
    flip silently. Verify the sign of `axis @ default_activation` before use.

    Usage:
        # Cosine mode (default, no calibration needed)
        with ActivationCappingHook(model, axis, "model.layers.60", tau=0.5):
            output = model.generate(**inputs)

        # Calibrated mode (recommended when you have per-layer norms)
        norms = load_cached_activation_norms("quant-sensitivity/llama-70b-nf4", "residual")
        with ActivationCappingHook(model, axis, "model.layers.60", tau=1.0,
                                   mean_activation_norm=norms[60]):
            output = model.generate(**inputs)

        # Raw mode (interop with Lu et al. precomputed thresholds)
        with ActivationCappingHook(model, axis, "model.layers.60",
                                   tau=16.99, tau_mode="raw"):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        direction: Union[torch.Tensor, Sequence[float]],
        path: str,
        tau: float,
        mode: str = "floor",
        tau_mode: Optional[Literal["cosine", "calibrated", "raw"]] = None,
        mean_activation_norm: Optional[float] = None,
    ):
        if mode not in ("floor", "ceiling"):
            raise ValueError(f"mode must be 'floor' or 'ceiling', got {mode!r}")

        # Resolve default tau_mode: calibrated if mean_norm provided, else cosine.
        if tau_mode is None:
            tau_mode = "calibrated" if mean_activation_norm is not None else "cosine"
        if tau_mode not in ("cosine", "calibrated", "raw"):
            raise ValueError(
                f"tau_mode must be 'cosine', 'calibrated', or 'raw', got {tau_mode!r}"
            )
        if tau_mode == "calibrated" and mean_activation_norm is None:
            raise ValueError(
                "tau_mode='calibrated' requires `mean_activation_norm` to be provided. "
                "Pass the precomputed mean residual norm at this layer (e.g. from "
                "`load_cached_activation_norms(experiment, 'residual')[layer]`)."
            )

        super().__init__(model, path)
        self.tau = float(tau)
        self.mode = mode
        self.tau_mode = tau_mode
        self.mean_activation_norm = (
            float(mean_activation_norm) if mean_activation_norm is not None else None
        )

        param = next(model.parameters())
        direction = torch.as_tensor(direction, dtype=torch.float32, device=param.device)

        if direction.ndim != 1:
            raise ValueError(f"Direction must be 1-D, got shape {direction.shape}")

        norm = direction.norm()
        if norm < 1e-8:
            raise ValueError("Direction vector has near-zero norm, cannot normalize")
        self.direction = direction / norm

    def _hook_fn(self, module, inputs, outputs):
        """Clamp projection onto direction within bounds."""
        out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        _assert_residual_shape(out_tensor, "ActivationCappingHook")
        v_hat = self.direction.to(device=out_tensor.device, dtype=out_tensor.dtype)

        proj = out_tensor @ v_hat  # [batch, seq]

        # effective_tau resolves tau_mode to a raw-projection-scale threshold
        if self.tau_mode == "raw":
            effective_tau = self.tau
        elif self.tau_mode == "calibrated":
            effective_tau = self.tau * self.mean_activation_norm
        else:  # "cosine"
            # Per-token rescaling. effective_tau is now a tensor [batch, seq].
            token_norms = out_tensor.norm(dim=-1)
            effective_tau = self.tau * token_norms

        if self.mode == "floor":
            delta = torch.clamp(effective_tau - proj, min=0)
        else:
            delta = torch.clamp(effective_tau - proj, max=0)

        capped = out_tensor + delta.unsqueeze(-1) * v_hat

        if torch.is_tensor(outputs):
            return capped
        elif isinstance(outputs, tuple):
            return (capped, *outputs[1:])
        return outputs


class MultiLayerActivationCapping:
    """
    Apply activation capping across multiple layers with per-layer thresholds.

    Each layer needs its own unit-axis direction (typically the assistant axis
    extracted at that layer) and its own tau value. See `ActivationCappingHook`
    docstring for tau units (cosine / calibrated / raw) and sign convention.

    The same tau_mode applies to every layer. If `mean_norm_per_layer` is
    provided, the default mode is "calibrated" and each layer uses its own
    norm. Otherwise the default is "cosine" (per-token rescaling).

    Tau values can vary per layer because residual norms grow with depth.
    For the most consistent semantic across layers, use cosine or calibrated
    mode with the same tau scalar across all layers.

    Usage:
        # Cosine mode (default, no calibration needed)
        axis = torch.load('axis.pt')  # {layer: 1D tensor}
        layers = list(range(56, 72))
        directions = {l: axis[l] for l in layers}
        tau_per_layer = {l: 0.5 for l in layers}  # 0.5 cosine, all layers
        with MultiLayerActivationCapping(model, directions, tau_per_layer):
            output = model.generate(**inputs)

        # Calibrated mode (recommended when per-layer norms are available)
        from utils.vectors import load_cached_activation_norms
        norms = load_cached_activation_norms("quant-sensitivity/llama-70b-nf4", "residual")
        with MultiLayerActivationCapping(
            model, directions, tau_per_layer={l: 1.0 for l in layers},
            mean_norm_per_layer={l: norms[l] for l in layers},
        ):
            output = model.generate(**inputs)

        # Raw mode (interop with Lu et al.'s precomputed thresholds)
        raw_taus = {56: 21.98, 57: 22.45, ..., 71: 31.59}
        with MultiLayerActivationCapping(model, directions, raw_taus, tau_mode="raw"):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: dict,  # {layer: vector} - per-layer direction vectors
        tau_per_layer: dict,  # {layer: tau_value}
        component: str = "residual",
        mode: str = "floor",
        tau_mode: Optional[Literal["cosine", "calibrated", "raw"]] = None,
        mean_norm_per_layer: Optional[Dict[int, float]] = None,
    ):
        # Resolve default tau_mode: calibrated if norms provided, else cosine.
        if tau_mode is None:
            tau_mode = "calibrated" if mean_norm_per_layer is not None else "cosine"
        if tau_mode == "calibrated" and mean_norm_per_layer is None:
            raise ValueError(
                "tau_mode='calibrated' requires `mean_norm_per_layer` to be provided."
            )

        arch = get_architecture(model)
        self._hooks = []
        for layer, tau in tau_per_layer.items():
            vec = directions[layer]
            layer_mean_norm = (
                mean_norm_per_layer[layer]
                if mean_norm_per_layer is not None and layer in mean_norm_per_layer
                else None
            )
            self._hooks.append(
                ActivationCappingHook(
                    model, vec, arch.path(component, layer, model=model), tau,
                    mode=mode,
                    tau_mode=tau_mode,
                    mean_activation_norm=layer_mean_norm,
                )
            )

    def __enter__(self):
        for hook in self._hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in reversed(self._hooks):
            hook.__exit__(*exc)


# =============================================================================
# PerSampleSteering - different steering per batch slice
# =============================================================================

class PerSampleSteering:
    """
    Per-slice steering: applies vec * coef to batch[start:end] for each config.

    Configs are grouped by layer internally - one hook registered per unique layer.
    Each hook applies all its configs to their respective batch slices.

    Config format: (layer, vector, coefficient, (batch_start, batch_end))

    Example - independent coefficient evaluation:
        configs = [
            (14, vec, 1.0, (0, 10)),   # Coef 1.0 on batch[0:10]
            (14, vec, 2.0, (10, 20)),  # Coef 2.0 on batch[10:20]
        ]

    Example - multi-layer ensemble:
        configs = [
            (11, vec11, 0.5, (0, 10)),  # L11 on batch[0:10]
            (13, vec13, 0.8, (0, 10)),  # L13 on batch[0:10] (ensemble)
        ]

    Usage:
        with PerSampleSteering(model, configs, component="residual"):
            output = model.generate(**batched_inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs: List[tuple],  # (layer, vector, coefficient, (batch_start, batch_end)) or
                               # (layer, vector, coefficient, (batch_start, batch_end), norm_match)
        component: str = "residual",
        norm_match: bool = False,
    ):
        self.model = model
        self.component = component.lower()
        self._manager = None
        if norm_match and self.component != "residual":
            raise ValueError(
                f"norm_match=True is only valid for component='residual'; got '{self.component}'."
            )

        param = next(model.parameters())
        # Vectors stored in float32 so norm-match computation is stable;
        # cast back to model dtype inside the hook before adding.
        self._layer_configs: dict = {}  # layer_idx -> List[(vector_f32, coef, batch_slice, norm_match)]
        for config in configs:
            if len(config) == 5:
                layer_idx, vector, coef, batch_slice, cfg_norm_match = config
            else:
                layer_idx, vector, coef, batch_slice = config
                cfg_norm_match = norm_match
            vec = torch.as_tensor(vector, dtype=torch.float32, device=param.device)
            if layer_idx not in self._layer_configs:
                self._layer_configs[layer_idx] = []
            self._layer_configs[layer_idx].append(
                (vec, float(coef), batch_slice, bool(cfg_norm_match))
            )

    def _make_hook(self, layer_configs: list):
        def hook_fn(module, inputs, outputs):
            t = outputs[0] if isinstance(outputs, tuple) else outputs
            _assert_residual_shape(t, "PerSampleSteering")
            t_new = t.clone()
            for vec, coef, (batch_start, batch_end), cfg_norm_match in layer_configs:
                slice_view = t_new[batch_start:batch_end]
                if cfg_norm_match:
                    scaled = _norm_match_scaled(vec, slice_view)
                    t_new[batch_start:batch_end] = slice_view + (coef * scaled.float()).to(dtype=t.dtype)
                else:
                    t_new[batch_start:batch_end] = slice_view + (coef * vec).to(dtype=t.dtype, device=t.device)
            if isinstance(outputs, tuple):
                return (t_new, *outputs[1:])
            return t_new
        return hook_fn

    def __enter__(self):
        arch = get_architecture(self.model)
        self._manager = HookManager(self.model)
        for layer_idx, layer_configs in self._layer_configs.items():
            path = arch.path(self.component, layer_idx, model=self.model)
            self._manager.add_forward_hook(path, self._make_hook(layer_configs))
        return self

    def __exit__(self, *exc):
        if self._manager:
            self._manager.remove_all()
            self._manager = None
