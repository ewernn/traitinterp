"""OLMo-2 Architecture.

True post-norm: post_attention_layernorm scales attention output and
post_feedforward_layernorm scales MLP output before they're added to the
residual. (Older `detect_contribution_paths` mis-classified OLMo-2 as Llama-style
because OLMo-2 lacks Gemma-2's `pre_feedforward_layernorm` - see docs/other/olmo.md.)
"""

from core.architectures import register
from core.architectures.base import Architecture, ModuleSpec


_OLMO2_TREE = {
    "self_attn":                    ModuleSpec("self_attn",                    "attention", "Multi-head self-attention"),
    "self_attn.q_proj":             ModuleSpec("self_attn.q_proj",             "linear",    "Query projection"),
    "self_attn.k_proj":             ModuleSpec("self_attn.k_proj",             "linear",    "Key projection"),
    "self_attn.v_proj":             ModuleSpec("self_attn.v_proj",             "linear",    "Value projection"),
    "self_attn.o_proj":             ModuleSpec("self_attn.o_proj",             "linear",    "Attention output (raw, pre post-norm)"),
    "post_attention_layernorm":     ModuleSpec("post_attention_layernorm",     "rmsnorm",   "TRUE post-attn norm (residual contribution point)"),
    "mlp":                          ModuleSpec("mlp",                          "mlp",       "Gated MLP"),
    "mlp.gate_proj":                ModuleSpec("mlp.gate_proj",                "linear",    "MLP gate projection"),
    "mlp.up_proj":                  ModuleSpec("mlp.up_proj",                  "linear",    "MLP up projection"),
    "mlp.down_proj":                ModuleSpec("mlp.down_proj",                "linear",    "MLP output (raw, pre post-norm)"),
    "post_feedforward_layernorm":   ModuleSpec("post_feedforward_layernorm",   "rmsnorm",   "TRUE post-MLP norm (residual contribution point)"),
}


register("olmo2", Architecture(
    layer_prefix_path="model.layers",
    attn_contribution_suffix="post_attention_layernorm",
    mlp_contribution_suffix="post_feedforward_layernorm",
    module_tree=_OLMO2_TREE,
))
