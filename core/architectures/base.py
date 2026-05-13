"""
Architecture descriptors: per-arch hook paths and module trees.

Input: HuggingFace model
Output: Architecture instance with hook paths, supported components, and a curated
    module_tree of named submodules under one block.
Usage:
    from core.architectures import get_architecture
    arch = get_architecture(model)
    path = arch.path("attn_contribution", layer=16, model=model)  # canonical hook path
    paths = arch.paths_for_layer(16, model=model)                  # all components for one layer
    supported = arch.supported_components(16)                       # set of valid components
    blocks = arch.layers(model)                                     # nn.ModuleList of blocks
    inner = arch.inner_model(model)                                 # unwrap PeftModel/multimodal
    arch.validate(model)                                            # raises if live tree disagrees
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


COMPONENTS = (
    "residual",
    "attn_contribution",
    "mlp_contribution",
    "k_proj",
    "v_proj",
)


class UnsupportedComponentError(ValueError):
    """Raised when a component is requested for a layer that doesn't expose it.

    Example: requesting `k_proj` on a Qwen3-Next linear-attn layer, on DeepSeek V3
    (MLA has no standard k_proj), or any standard transformer component on a Mamba block.
    """


class ArchitectureMismatchError(RuntimeError):
    """Raised when the live model's module tree disagrees with the adapter.

    Common causes: device_map="auto" inserted quantization wrappers that changed
    module paths, a transformers version refactored module names, or the wrong
    adapter was selected.
    """


@dataclass(frozen=True)
class ModuleSpec:
    """One named submodule under a block.

    Used by Architecture.module_tree to advertise hookable points to tooling
    (the dashboard, ad-hoc analysis scripts) and to validate the live model.
    """
    path: str            # "self_attn.q_proj" relative to one block
    kind: str            # "linear" | "rmsnorm" | "attention" | "mlp" | "moe" | "mla" | ...
    description: str     # human-readable; surfaced in tooltips
    output_shape: str = ""  # symbolic, documentation only


@dataclass(frozen=True)
class LayerPaths:
    """Resolved hook paths for one layer.

    Fields are absolute dot-paths into the model. None means the component does
    not exist for that layer (k_proj on MLA, attn_* on a Mamba block, etc.).
    """
    residual: Optional[str]
    attn_contribution: Optional[str]
    mlp_contribution: Optional[str]
    k_proj: Optional[str]
    v_proj: Optional[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "residual": self.residual,
            "attn_contribution": self.attn_contribution,
            "mlp_contribution": self.mlp_contribution,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
        }


@dataclass(frozen=True)
class Architecture:
    """Declarative description of a transformer-shaped architecture.

    One instance per HF model_type. Holds canonical relative paths from each
    block to its components, plus a module_tree of named submodules.

    For per-layer dispatch (Qwen3-Next, DeepSeek V3), use HybridArchitecture
    which overrides _layer_overrides_for() to introspect the live block.
    """

    # Path within an unwrapped model to the block list, e.g. "model.layers"
    # or "model.language_model.layers" for Gemma3-multimodal.
    layer_prefix_path: str = "model.layers"

    # Relative suffixes from a block to each component. Empty string = the block
    # itself (used for residual). None = component does not exist.
    residual_suffix: str = ""
    attn_contribution_suffix: Optional[str] = "self_attn.o_proj"
    mlp_contribution_suffix: Optional[str] = "mlp.down_proj"
    k_proj_suffix: Optional[str] = "self_attn.k_proj"
    v_proj_suffix: Optional[str] = "self_attn.v_proj"

    # Curated tree of hookable submodules under one block.
    module_tree: Dict[str, ModuleSpec] = field(default_factory=dict)

    # ---- public API ----

    def path(self, component: str, layer: int, model=None) -> str:
        """Absolute hook path for one (component, layer) pair.

        If `model` is given, the prefix is LoRA-aware and (for HybridArchitecture)
        the live block is introspected to handle per-layer dispatch correctly.
        Without `model`, uses the static layer_prefix_path and falls back to a
        model-free best guess for hybrid arches (suitable for tests only).

        Raises UnsupportedComponentError if the component is not exposed.
        """
        if component not in COMPONENTS:
            raise ValueError(
                f"Unknown component {component!r}. Valid: {COMPONENTS}. "
                f"For arbitrary submodules, pass the full dot-path directly to the hook."
            )
        block = self.layers(model)[layer] if model is not None else None
        suffix = self.suffix_for(component, layer, block=block)
        if suffix is None:
            raise UnsupportedComponentError(
                f"{type(self).__name__} does not expose {component!r} at layer {layer}"
            )
        prefix = self.layer_prefix(model) if model is not None else self.layer_prefix_path
        return f"{prefix}.{layer}" if suffix == "" else f"{prefix}.{layer}.{suffix}"

    def paths_for_layer(self, layer: int, model=None) -> LayerPaths:
        """All canonical paths for one layer. None means component absent."""
        prefix = self.layer_prefix(model) if model is not None else self.layer_prefix_path
        block = self.layers(model)[layer] if model is not None else None

        def resolve(name: str) -> Optional[str]:
            suffix = self.suffix_for(name, layer, block=block)
            if suffix is None:
                return None
            return f"{prefix}.{layer}" if suffix == "" else f"{prefix}.{layer}.{suffix}"

        return LayerPaths(
            residual=resolve("residual"),
            attn_contribution=resolve("attn_contribution"),
            mlp_contribution=resolve("mlp_contribution"),
            k_proj=resolve("k_proj"),
            v_proj=resolve("v_proj"),
        )

    def supported_components(self, layer: int) -> set:
        """Components that have a hook path at this layer."""
        return {
            name for name, value in self.paths_for_layer(layer).as_dict().items()
            if value is not None
        }

    def suffix_for(self, component: str, layer: int, block=None) -> Optional[str]:
        """Resolve the per-layer suffix for one component.

        Base implementation returns the static suffix. HybridArchitecture overrides
        to consult per-layer rules (and the live block when given).
        """
        return getattr(self, f"{component}_suffix")

    # ---- model interaction ----

    def inner_model(self, model):
        """Return the model with PeftModel and multimodal wrappers stripped."""
        # PeftModel: recurse to handle multimodal-under-LoRA
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            if type(model).__name__ != type(model.base_model).__name__:
                return self.inner_model(model.base_model.model)
        # Multimodal: model.model.language_model is the LLM (vision encoder is a sibling)
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        if hasattr(model, "model"):
            return model.model
        return model

    def layer_prefix(self, model) -> str:
        """Absolute dot-path to the block list, accounting for wrappers."""
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            if type(model).__name__ != type(model.base_model).__name__:
                return f"base_model.model.{self.layer_prefix_path}"
        return self.layer_prefix_path

    def layers(self, model):
        """Return the nn.ModuleList of blocks, accounting for wrappers."""
        return model.get_submodule(self.layer_prefix(model))

    def validate(self, model) -> None:
        """Raise ArchitectureMismatchError if the live model diverges from module_tree.

        Walks every block once and verifies every path in module_tree resolves
        somewhere. For hybrid architectures (where a path may exist on some blocks
        but not others), a path is only considered missing if it resolves on no block.
        """
        if not self.module_tree:
            return
        blocks = self.layers(model)
        missing = [
            spec.path for spec in self.module_tree.values()
            if not any(_path_exists(b, spec.path) for b in blocks)
        ]
        if missing:
            raise ArchitectureMismatchError(
                f"{type(self).__name__} module_tree declares paths that don't resolve "
                f"on this {type(model).__name__}. Missing: {missing}. "
                f"Common causes: device_map='auto' inserted wrappers, transformers "
                f"version refactored module names, or wrong adapter selected for "
                f"model_type={getattr(getattr(model, 'config', None), 'model_type', '?')}."
            )

    def discover(self, model, layer: int = 0) -> List[str]:
        """Runtime enumeration of every named submodule under one block.

        Use for exploration ("show me everything") or when module_tree is incomplete.
        Returns dot-paths relative to the block.
        """
        block = self.layers(model)[layer]
        return [name for name, _ in block.named_modules() if name]


@dataclass(frozen=True)
class HybridArchitecture(Architecture):
    """Architecture with per-layer dispatch (Qwen3-Next: linear-attn vs full-attn;
    DeepSeek V3: dense MLP vs MoE).

    Subclasses override _layer_overrides_for(layer, block) to dispatch based on
    layer index and (when given) live block introspection. The block kwarg is
    passed when a model is in scope; it is None for model-free static lookups.
    """

    def _layer_overrides_for(
        self, layer: int, block: Optional[object] = None
    ) -> Optional[Dict[str, Optional[str]]]:
        """Override in subclasses. Default: no overrides."""
        return None

    def suffix_for(self, component: str, layer: int, block=None) -> Optional[str]:
        overrides = self._layer_overrides_for(layer, block=block)
        if overrides is not None and component in overrides:
            return overrides[component]
        return getattr(self, f"{component}_suffix")


def _path_exists(root, path: str) -> bool:
    """True if the dot-path resolves to a real submodule under root."""
    try:
        root.get_submodule(path)
        return True
    except AttributeError:
        return False
