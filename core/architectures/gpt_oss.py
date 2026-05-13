"""GPT-OSS Architecture (OpenAI's open-source 120B MoE).

Standard self-attention with MoE FFN. Attention paths are Llama-shaped; the MLP
is fused into a single MoE module without a per-expert .down_proj exposed in HF.

mlp_contribution falls back to hooking `mlp` (the whole MoE module's output) -
that's what gets added to the residual.
"""

from core.architectures import register
from core.architectures.base import Architecture, ModuleSpec


_GPT_OSS_TREE = {
    "input_layernorm":          ModuleSpec("input_layernorm",          "rmsnorm",   "Pre-attention RMS norm"),
    "self_attn":                ModuleSpec("self_attn",                "attention", "Multi-head self-attention (GQA)"),
    "self_attn.q_proj":         ModuleSpec("self_attn.q_proj",         "linear",    "Query projection"),
    "self_attn.k_proj":         ModuleSpec("self_attn.k_proj",         "linear",    "Key projection"),
    "self_attn.v_proj":         ModuleSpec("self_attn.v_proj",         "linear",    "Value projection"),
    "self_attn.o_proj":         ModuleSpec("self_attn.o_proj",         "linear",    "Attention output (residual contribution)"),
    "post_attention_layernorm": ModuleSpec("post_attention_layernorm", "rmsnorm",   "Pre-MoE norm"),
    "mlp":                      ModuleSpec("mlp",                      "moe",       "MoE FFN (residual contribution)"),
    "mlp.router":               ModuleSpec("mlp.router",               "linear",    "Expert routing logits"),
}


register("gpt_oss", Architecture(
    layer_prefix_path="model.layers",
    mlp_contribution_suffix="mlp",  # MoE block as a whole, no .down_proj
    module_tree=_GPT_OSS_TREE,
))
