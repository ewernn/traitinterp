"""Gemma-3 (multimodal) Architecture.

Wrapped in a multimodal container that exposes the LLM at
model.language_model.layers (not model.layers).

Despite being a different family, Gemma-3 inherits Gemma-2's TRUE post-norm
structure: post_attention_layernorm scales the attention output and
post_feedforward_layernorm scales the MLP output before they're added to the
residual (verified in transformers.models.gemma3.modeling_gemma3.Gemma3DecoderLayer).

Note: Gemma3Config.model_type is "gemma3" but Gemma3TextConfig.model_type is
"gemma3_text" - the registry handles both via an alias entry.
"""

from core.architectures import register
from core.architectures.base import Architecture
from core.architectures.gemma2 import _GEMMA2_TREE


_GEMMA3_ARCH = Architecture(
    layer_prefix_path="model.language_model.layers",
    attn_contribution_suffix="post_attention_layernorm",
    mlp_contribution_suffix="post_feedforward_layernorm",
    module_tree=_GEMMA2_TREE,
)

register("gemma3", _GEMMA3_ARCH)
