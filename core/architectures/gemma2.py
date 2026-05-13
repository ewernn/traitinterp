"""Gemma-2 Architecture.

Has TRUE post-sublayer norms: post_attention_layernorm scales the attention
output before it's added to the residual, and post_feedforward_layernorm scales
the MLP output. Contribution components hook those post-norm outputs (not the
raw sublayer outputs as in Llama/Mistral/Qwen).
"""

from core.architectures import register
from core.architectures.base import Architecture, ModuleSpec


_GEMMA2_TREE = {
    "input_layernorm":              ModuleSpec("input_layernorm",              "rmsnorm",   "Pre-attention RMS norm"),
    "self_attn":                    ModuleSpec("self_attn",                    "attention", "Multi-head self-attention"),
    "self_attn.q_proj":             ModuleSpec("self_attn.q_proj",             "linear",    "Query projection"),
    "self_attn.k_proj":             ModuleSpec("self_attn.k_proj",             "linear",    "Key projection"),
    "self_attn.v_proj":             ModuleSpec("self_attn.v_proj",             "linear",    "Value projection"),
    "self_attn.o_proj":             ModuleSpec("self_attn.o_proj",             "linear",    "Attention output (raw, pre post-norm)"),
    "post_attention_layernorm":     ModuleSpec("post_attention_layernorm",     "rmsnorm",   "TRUE post-attn norm (scales o_proj before residual add)"),
    "pre_feedforward_layernorm":    ModuleSpec("pre_feedforward_layernorm",    "rmsnorm",   "Pre-MLP norm"),
    "mlp":                          ModuleSpec("mlp",                          "mlp",       "Gated MLP"),
    "mlp.gate_proj":                ModuleSpec("mlp.gate_proj",                "linear",    "MLP gate projection"),
    "mlp.up_proj":                  ModuleSpec("mlp.up_proj",                  "linear",    "MLP up projection"),
    "mlp.down_proj":                ModuleSpec("mlp.down_proj",                "linear",    "MLP output (raw, pre post-norm)"),
    "post_feedforward_layernorm":   ModuleSpec("post_feedforward_layernorm",   "rmsnorm",   "TRUE post-MLP norm (scales mlp.down_proj before residual add)"),
}


register("gemma2", Architecture(
    layer_prefix_path="model.layers",
    attn_contribution_suffix="post_attention_layernorm",
    mlp_contribution_suffix="post_feedforward_layernorm",
    module_tree=_GEMMA2_TREE,
))
