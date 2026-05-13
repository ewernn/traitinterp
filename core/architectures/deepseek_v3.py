"""DeepSeek V3 Architecture (also serves Kimi K2 via the kimi_k2 alias).

Multi-head Latent Attention (MLA): there is no standard k_proj/v_proj. Q and KV
are decompressed from low-rank latents, so we expose attn_contribution as o_proj
and mark k_proj/v_proj as unsupported (callers get UnsupportedComponentError
instead of silently hooking the wrong tensor).

Mixed dense/MoE layers: layers `< config.first_k_dense_replace` use a dense MLP
with `mlp.down_proj`; later layers use a MoE block whose residual contribution
is the whole `mlp` module (no .down_proj). Per-layer dispatch is config-driven:
prefer runtime introspection of the live block (hasattr `gate`/`experts`).

Note: DeepSeekV3Attention conditionally creates either `q_proj` (when
`q_lora_rank is None`) or `q_a_proj`/`q_a_layernorm`/`q_b_proj` (V3 default,
q_lora_rank=1536). The module_tree lists the LoRA-Q variant since that's V3.

For latent-space analysis (TLens hook_q_latent / hook_kv_latent), use the path=
API directly with self_attn.q_a_layernorm or self_attn.kv_a_layernorm.
"""

from typing import Dict, Optional

from core.architectures import register
from core.architectures.base import HybridArchitecture, ModuleSpec


_MOE_LAYER_OVERRIDES: Dict[str, Optional[str]] = {
    "mlp_contribution": "mlp",  # Whole MoE module's output is the residual contribution
}


_MLA_TREE = {
    "input_layernorm":              ModuleSpec("input_layernorm",              "rmsnorm",   "Pre-attention RMS norm"),
    "self_attn":                    ModuleSpec("self_attn",                    "mla",       "Multi-head latent attention"),
    "self_attn.q_a_proj":           ModuleSpec("self_attn.q_a_proj",           "linear",    "Q down-projection (to latent; only when q_lora_rank is set)"),
    "self_attn.q_a_layernorm":      ModuleSpec("self_attn.q_a_layernorm",      "rmsnorm",   "Q latent norm (TLens hook_q_latent)"),
    "self_attn.q_b_proj":           ModuleSpec("self_attn.q_b_proj",           "linear",    "Q up-projection (latent → full Q)"),
    "self_attn.kv_a_proj_with_mqa": ModuleSpec("self_attn.kv_a_proj_with_mqa", "linear",    "KV down-projection + per-head K-rope"),
    "self_attn.kv_a_layernorm":     ModuleSpec("self_attn.kv_a_layernorm",     "rmsnorm",   "KV latent norm (TLens hook_kv_latent)"),
    "self_attn.kv_b_proj":          ModuleSpec("self_attn.kv_b_proj",          "linear",    "KV up-projection (latent → full K, V)"),
    "self_attn.o_proj":             ModuleSpec("self_attn.o_proj",             "linear",    "Attention output (residual contribution)"),
    "post_attention_layernorm":     ModuleSpec("post_attention_layernorm",     "rmsnorm",   "Pre-MLP norm"),
    "mlp":                          ModuleSpec("mlp",                          "moe",       "MoE on most layers (DeepseekV3MoE), dense MLP on first_k_dense_replace layers"),
}



class _DeepSeekV3Architecture(HybridArchitecture):
    """DeepSeek V3 / Kimi K2: MoE layers expose `mlp` as a whole, dense layers expose `mlp.down_proj`."""

    def _layer_overrides_for(self, layer, block=None):
        if block is not None:
            # Runtime: dense MLPs have .down_proj, MoE blocks don't
            mlp = getattr(block, "mlp", None)
            if mlp is not None and not hasattr(mlp, "down_proj"):
                return _MOE_LAYER_OVERRIDES
            return None
        # Model-free fallback: V3/Kimi-K2 use first_k_dense_replace=1 (layer 0 dense, rest MoE).
        # Static path queries should default to the common case (MoE).
        if layer == 0:
            return None  # dense: use default mlp.down_proj suffix
        return _MOE_LAYER_OVERRIDES


register("deepseek_v3", _DeepSeekV3Architecture(
    layer_prefix_path="model.layers",
    k_proj_suffix=None,  # MLA has no standard k_proj
    v_proj_suffix=None,  # MLA has no standard v_proj
    module_tree=_MLA_TREE,
))
