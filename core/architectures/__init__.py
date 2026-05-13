"""
Architecture registry: model_type -> Architecture instance.

Input: HuggingFace model
Output: Architecture instance describing hookable components and module tree.
Usage:
    from core.architectures import get_architecture, layers, layer_prefix, inner_model

    arch = get_architecture(model)
    path = arch.path("attn_contribution", layer=16)

    # Standalone helpers for one-shot lookups when you don't need the arch object
    blocks = layers(model)
    prefix = layer_prefix(model)
    inner = inner_model(model)
"""

from typing import Dict

from core.architectures.base import (
    Architecture,
    ArchitectureMismatchError,
    COMPONENTS,
    HybridArchitecture,
    LayerPaths,
    ModuleSpec,
    UnsupportedComponentError,
)


# model_type -> Architecture instance. Populated by per-arch modules at import.
ARCHITECTURE_REGISTRY: Dict[str, Architecture] = {}

# model_type aliases. Architectural truth, not back-compat. Examples:
#   kimi_k2     -> deepseek_v3  (Kimi K2 reuses the DeepSeek V3 architecture)
#   gemma3_text -> gemma3       (HF's Gemma3TextConfig.model_type for the LLM half
#                                of multimodal Gemma3, where the wrapper is "gemma3")
ARCHITECTURE_ALIASES: Dict[str, str] = {
    "kimi_k2": "deepseek_v3",
    "gemma3_text": "gemma3",
    "qwen3_5": "qwen3_next",
}


def register(model_type: str, arch: Architecture) -> None:
    """Register an Architecture for a HuggingFace model_type."""
    if model_type in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Architecture for model_type={model_type!r} already registered. "
            f"Update the existing registration in core/architectures/{model_type}.py "
            f"rather than registering twice."
        )
    if model_type in ARCHITECTURE_ALIASES:
        raise ValueError(
            f"Cannot register {model_type!r}: it is an alias for "
            f"{ARCHITECTURE_ALIASES[model_type]!r}. Update the alias target instead."
        )
    ARCHITECTURE_REGISTRY[model_type] = arch


def get_architecture(model) -> Architecture:
    """Resolve the Architecture for a model from its config.model_type.

    Handles PeftModel and multimodal wrappers by reading the inner model's config.
    """
    config_obj = _config_of(model)
    model_type = getattr(config_obj, "model_type", None)
    if model_type is None:
        raise ValueError(
            f"Cannot resolve architecture: {type(model).__name__} has no model_type "
            f"in its config. Pass `path=` directly to bypass the registry."
        )
    resolved = ARCHITECTURE_ALIASES.get(model_type, model_type)
    if resolved not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"No Architecture registered for model_type={model_type!r}. "
            f"Registered: {sorted(ARCHITECTURE_REGISTRY)}. "
            f"Add a new file in core/architectures/{model_type}.py and register it."
        )
    return ARCHITECTURE_REGISTRY[resolved]


def layers(model):
    """Return the nn.ModuleList of transformer blocks for any registered model."""
    return get_architecture(model).layers(model)


def layer_prefix(model) -> str:
    """Return the dot-path prefix to the block list (LoRA-aware)."""
    return get_architecture(model).layer_prefix(model)


def inner_model(model):
    """Strip PeftModel and multimodal wrappers; return the inner transformer."""
    return get_architecture(model).inner_model(model)


def _config_of(model):
    """Return the config object, walking through PeftModel/multimodal wrappers."""
    # PeftModel: recurse into base model so multimodal-under-LoRA also resolves
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        if type(model).__name__ != type(model.base_model).__name__:
            return _config_of(model.base_model.model)
    if hasattr(model, "config"):
        cfg = model.config
        # Multimodal: prefer text_config so we get the LLM's model_type, not the wrapper's
        # (HF Gemma3 ships gemma3_text on the inner config; aliases handle the mapping)
        if hasattr(cfg, "text_config") and getattr(cfg.text_config, "model_type", None):
            return cfg.text_config
        return cfg
    return None


# Import per-arch modules to trigger registration. Order matters only insofar as
# duplicate registrations raise; alphabetical is a stable convention.
from core.architectures import (  # noqa: E402,F401  (side-effect imports)
    deepseek_v3,
    gemma2,
    gemma3,
    gpt_oss,
    llama,
    mistral,
    olmo2,
    qwen2,
    qwen3,
    qwen3_next,
)


__all__ = [
    "ARCHITECTURE_ALIASES",
    "ARCHITECTURE_REGISTRY",
    "Architecture",
    "ArchitectureMismatchError",
    "COMPONENTS",
    "HybridArchitecture",
    "LayerPaths",
    "ModuleSpec",
    "UnsupportedComponentError",
    "get_architecture",
    "inner_model",
    "layer_prefix",
    "layers",
    "register",
]
