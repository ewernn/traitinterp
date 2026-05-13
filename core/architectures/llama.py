"""Llama-family Architecture (Llama 3.x, AWQ variants).

Pre-norm transformer with standard self-attention. Contribution to the residual
is the raw sublayer output (no post-norm to fold in).
"""

from core.architectures import register
from core.architectures.base import Architecture, ModuleSpec


_LLAMA_TREE = {
    "input_layernorm":         ModuleSpec("input_layernorm",         "rmsnorm",   "Pre-attention RMS norm"),
    "self_attn":               ModuleSpec("self_attn",               "attention", "Multi-head self-attention (GQA)"),
    "self_attn.q_proj":        ModuleSpec("self_attn.q_proj",        "linear",    "Query projection",  "[batch, seq, n_heads * d_head]"),
    "self_attn.k_proj":        ModuleSpec("self_attn.k_proj",        "linear",    "Key projection",    "[batch, seq, n_kv_heads * d_head]"),
    "self_attn.v_proj":        ModuleSpec("self_attn.v_proj",        "linear",    "Value projection",  "[batch, seq, n_kv_heads * d_head]"),
    "self_attn.o_proj":        ModuleSpec("self_attn.o_proj",        "linear",    "Attention output (residual contribution)"),
    "post_attention_layernorm": ModuleSpec("post_attention_layernorm", "rmsnorm",  "Pre-MLP RMS norm (NOT a true post-norm despite the name)"),
    "mlp":                     ModuleSpec("mlp",                     "mlp",       "Gated MLP"),
    "mlp.gate_proj":           ModuleSpec("mlp.gate_proj",           "linear",    "MLP gate projection"),
    "mlp.up_proj":             ModuleSpec("mlp.up_proj",             "linear",    "MLP up projection"),
    "mlp.down_proj":           ModuleSpec("mlp.down_proj",           "linear",    "MLP output (residual contribution)"),
}


register("llama", Architecture(
    layer_prefix_path="model.layers",
    module_tree=_LLAMA_TREE,
))
