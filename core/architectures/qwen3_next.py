"""Qwen3-Next Architecture (hybrid linear-attention + full attention).

HF model_type: `qwen3_next`. Layers alternate between standard self-attention and
a Qwen3NextGatedDeltaNet linear-attention variant per `config.layer_types[i]`.
All MLPs are MoE blocks (no Llama-style `mlp.down_proj`).

Linear-attn layers expose `linear_attn.out_proj`; full-attn layers expose
`self_attn.{q,k,v,o}_proj`. Hooks with the model in scope go through
`arch.suffix_for(...)`, which introspects the live block and is correct for any
layer ordering. The model-free fallback uses the published 3:1 ratio
(full-attn at `i % FULL_ATTN_LAYER_PERIOD == FULL_ATTN_LAYER_OFFSET`),
which holds for the current 80B checkpoint but is only a best guess for tests.

Aliased: `qwen3_5 -> qwen3_next` (anticipated future Qwen3.5 release using the
same hybrid architecture).
"""

from typing import Dict, Optional

from core.architectures import register
from core.architectures.base import HybridArchitecture, ModuleSpec


# Checkpoint-specific layer pattern (current Qwen3-Next-80B). Runtime path
# (suffix_for with a live block) doesn't depend on these.
FULL_ATTN_LAYER_PERIOD = 4
FULL_ATTN_LAYER_OFFSET = 3


_LINEAR_ATTN_OVERRIDES: Dict[str, Optional[str]] = {
    "attn_contribution": "linear_attn.out_proj",
    "k_proj": None,
    "v_proj": None,
    "mlp_contribution": "mlp",  # all layers are MoE; no Llama-style down_proj
}

_FULL_ATTN_OVERRIDES: Dict[str, Optional[str]] = {
    "mlp_contribution": "mlp",
}


_QWEN3_NEXT_TREE = {
    "input_layernorm":          ModuleSpec("input_layernorm",          "rmsnorm",          "Pre-attention RMS norm"),
    "self_attn":                ModuleSpec("self_attn",                "attention",        "Standard self-attention (full-attn layers)"),
    "self_attn.q_proj":         ModuleSpec("self_attn.q_proj",         "linear",           "Query projection"),
    "self_attn.k_proj":         ModuleSpec("self_attn.k_proj",         "linear",           "Key projection"),
    "self_attn.v_proj":         ModuleSpec("self_attn.v_proj",         "linear",           "Value projection"),
    "self_attn.o_proj":         ModuleSpec("self_attn.o_proj",         "linear",           "Attention output"),
    "linear_attn":              ModuleSpec("linear_attn",              "linear_attention", "Gated DeltaNet (linear-attn layers)"),
    "linear_attn.out_proj":     ModuleSpec("linear_attn.out_proj",     "linear",           "Linear-attn output"),
    "post_attention_layernorm": ModuleSpec("post_attention_layernorm", "rmsnorm",          "Pre-MoE norm"),
    "mlp":                      ModuleSpec("mlp",                      "moe",              "Sparse MoE block (residual contribution)"),
    "mlp.gate":                 ModuleSpec("mlp.gate",                 "linear",           "Expert routing logits"),
}


class _Qwen3NextArchitecture(HybridArchitecture):
    """Per-layer dispatch between linear_attn and self_attn; all layers are MoE."""

    def _layer_overrides_for(self, layer, block=None):
        if block is not None:
            if hasattr(block, "linear_attn") and not hasattr(block, "self_attn"):
                return _LINEAR_ATTN_OVERRIDES
            return _FULL_ATTN_OVERRIDES
        if layer % FULL_ATTN_LAYER_PERIOD != FULL_ATTN_LAYER_OFFSET:
            return _LINEAR_ATTN_OVERRIDES
        return _FULL_ATTN_OVERRIDES


register("qwen3_next", _Qwen3NextArchitecture(
    layer_prefix_path="model.layers",
    module_tree=_QWEN3_NEXT_TREE,
))
