"""Qwen3 Architecture.

Llama-shaped at the contribution level. Adds q_norm/k_norm (per-head RMSNorms)
inside self_attn - declared in module_tree so they're discoverable, but they
don't affect the canonical contribution paths.
"""

from core.architectures import register
from core.architectures.base import Architecture, ModuleSpec
from core.architectures.llama import _LLAMA_TREE


_QWEN3_TREE = {
    **_LLAMA_TREE,
    "self_attn.q_norm": ModuleSpec("self_attn.q_norm", "rmsnorm", "Per-head Q normalization (Qwen3-specific)"),
    "self_attn.k_norm": ModuleSpec("self_attn.k_norm", "rmsnorm", "Per-head K normalization (Qwen3-specific)"),
}


register("qwen3", Architecture(
    layer_prefix_path="model.layers",
    module_tree=_QWEN3_TREE,
))
