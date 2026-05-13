"""Pure-Python tests for the architecture registry.

No HuggingFace models loaded - just verifies the dataclass logic, registry
resolution, and per-arch declarations. Real-model verification lives in
test_architectures_real_models.py.
"""

import pytest

from core.architectures import (
    ARCHITECTURE_ALIASES,
    ARCHITECTURE_REGISTRY,
    Architecture,
    ArchitectureMismatchError,
    COMPONENTS,
    HybridArchitecture,
    LayerPaths,
    ModuleSpec,
    UnsupportedComponentError,
    get_architecture,
    inner_model,
    layer_prefix,
    layers,
    register,
)


# ----------------------------------------------------------------------------
# Path resolution per architecture (no model - uses static layer_prefix_path)
# ----------------------------------------------------------------------------

PATH_CASES = [
    # (model_type, component, layer, expected_path)
    ("llama",       "residual",          5, "model.layers.5"),
    ("llama",       "attn_contribution", 5, "model.layers.5.self_attn.o_proj"),
    ("llama",       "mlp_contribution",  5, "model.layers.5.mlp.down_proj"),
    ("llama",       "k_proj",            5, "model.layers.5.self_attn.k_proj"),
    ("llama",       "v_proj",            5, "model.layers.5.self_attn.v_proj"),
    ("mistral",     "attn_contribution", 0, "model.layers.0.self_attn.o_proj"),
    ("qwen2",       "attn_contribution", 3, "model.layers.3.self_attn.o_proj"),
    ("qwen3",       "attn_contribution", 3, "model.layers.3.self_attn.o_proj"),
    ("gemma2",      "attn_contribution", 5, "model.layers.5.post_attention_layernorm"),
    ("gemma2",      "mlp_contribution",  5, "model.layers.5.post_feedforward_layernorm"),
    ("gemma3",      "attn_contribution", 5, "model.language_model.layers.5.post_attention_layernorm"),
    ("gemma3",      "mlp_contribution",  5, "model.language_model.layers.5.post_feedforward_layernorm"),
    ("olmo2",       "attn_contribution", 5, "model.layers.5.post_attention_layernorm"),
    ("olmo2",       "mlp_contribution",  5, "model.layers.5.post_feedforward_layernorm"),
    ("gpt_oss",     "attn_contribution", 7, "model.layers.7.self_attn.o_proj"),
    ("gpt_oss",     "mlp_contribution",  7, "model.layers.7.mlp"),
    ("deepseek_v3", "attn_contribution", 2, "model.layers.2.self_attn.o_proj"),
    ("deepseek_v3", "mlp_contribution",  0, "model.layers.0.mlp.down_proj"),  # dense
    ("deepseek_v3", "mlp_contribution",  5, "model.layers.5.mlp"),            # MoE
]


@pytest.mark.parametrize("model_type,component,layer,expected", PATH_CASES)
def test_path_resolution(model_type, component, layer, expected):
    arch = ARCHITECTURE_REGISTRY[model_type]
    assert arch.path(component, layer) == expected


# ----------------------------------------------------------------------------
# OLMo2 contribution-path bug fix (regression guard)
# ----------------------------------------------------------------------------

def test_olmo2_post_norm_contribution_paths():
    """OLMo2 was previously misclassified as Llama-style by detect_contribution_paths.
    The adapter now declares the true post-norm paths."""
    arch = ARCHITECTURE_REGISTRY["olmo2"]
    assert arch.path("attn_contribution", 0).endswith("post_attention_layernorm")
    assert arch.path("mlp_contribution", 0).endswith("post_feedforward_layernorm")


# ----------------------------------------------------------------------------
# Unsupported components raise loud errors
# ----------------------------------------------------------------------------

def test_mla_k_proj_unsupported():
    arch = ARCHITECTURE_REGISTRY["deepseek_v3"]
    with pytest.raises(UnsupportedComponentError):
        arch.path("k_proj", 0)
    with pytest.raises(UnsupportedComponentError):
        arch.path("v_proj", 0)


def test_qwen35_k_proj_only_on_full_attn_layers():
    arch = ARCHITECTURE_REGISTRY["qwen3_next"]
    # layer % 4 != 3 -> linear-attn, no k_proj
    with pytest.raises(UnsupportedComponentError):
        arch.path("k_proj", 0)
    with pytest.raises(UnsupportedComponentError):
        arch.path("k_proj", 2)
    # layer % 4 == 3 -> full-attn, k_proj present
    assert arch.path("k_proj", 3) == "model.layers.3.self_attn.k_proj"


def test_unknown_component_raises():
    arch = ARCHITECTURE_REGISTRY["llama"]
    with pytest.raises(ValueError, match="Unknown component"):
        arch.path("attn_out", 0)
    with pytest.raises(ValueError, match="Unknown component"):
        arch.path("not_a_thing", 0)


# ----------------------------------------------------------------------------
# Hybrid architecture: runtime block introspection
# ----------------------------------------------------------------------------

def test_qwen35_runtime_introspection_full_attn_block():
    """A full-attn block at any layer index should resolve to self_attn.o_proj."""
    arch = ARCHITECTURE_REGISTRY["qwen3_next"]

    class FullAttnBlock:
        self_attn = object()  # has self_attn, no linear_attn

    suffix = arch.suffix_for("attn_contribution", layer=0, block=FullAttnBlock())
    assert suffix == "self_attn.o_proj"


def test_qwen35_runtime_introspection_linear_attn_block():
    arch = ARCHITECTURE_REGISTRY["qwen3_next"]

    class LinearAttnBlock:
        linear_attn = object()  # has linear_attn, no self_attn

    suffix = arch.suffix_for("attn_contribution", layer=99, block=LinearAttnBlock())
    assert suffix == "linear_attn.out_proj"
    assert arch.suffix_for("k_proj", 99, block=LinearAttnBlock()) is None


def test_deepseek_v3_runtime_introspection_dense_vs_moe():
    arch = ARCHITECTURE_REGISTRY["deepseek_v3"]

    class DenseMLP:
        down_proj = object()

    class DenseBlock:
        mlp = DenseMLP()

    class MoEBlock:
        mlp = object()  # no down_proj

    assert arch.suffix_for("mlp_contribution", 99, block=DenseBlock()) == "mlp.down_proj"
    assert arch.suffix_for("mlp_contribution", 0, block=MoEBlock()) == "mlp"


# ----------------------------------------------------------------------------
# Registry: aliases, registration, error messages
# ----------------------------------------------------------------------------

def test_kimi_k2_alias_resolves_to_deepseek_v3():
    class FakeCfg:
        model_type = "kimi_k2"

    class FakeModel:
        config = FakeCfg()

    arch = get_architecture(FakeModel())
    assert arch is ARCHITECTURE_REGISTRY["deepseek_v3"]


def test_gemma3_text_alias_resolves_to_gemma3():
    """HF Gemma3TextConfig.model_type is 'gemma3_text'; the alias maps it."""
    class TextCfg:
        model_type = "gemma3_text"

    class WrapperCfg:
        model_type = "gemma3"
        text_config = TextCfg()

    class FakeModel:
        config = WrapperCfg()

    arch = get_architecture(FakeModel())
    assert arch is ARCHITECTURE_REGISTRY["gemma3"]


def test_register_blocks_alias_names():
    with pytest.raises(ValueError, match="alias"):
        register("kimi_k2", ARCHITECTURE_REGISTRY["llama"])


def test_register_blocks_double_registration():
    with pytest.raises(ValueError, match="already registered"):
        register("llama", ARCHITECTURE_REGISTRY["llama"])


def test_unknown_model_type_raises_with_diagnostic():
    class FakeCfg:
        model_type = "made_up_arch"

    class FakeModel:
        config = FakeCfg()

    with pytest.raises(ValueError, match="No Architecture registered"):
        get_architecture(FakeModel())


# ----------------------------------------------------------------------------
# Frozen dataclasses
# ----------------------------------------------------------------------------

def test_architecture_is_frozen():
    arch = ARCHITECTURE_REGISTRY["llama"]
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        arch.layer_prefix_path = "mutated"


def test_layer_paths_is_frozen():
    paths = LayerPaths(
        residual="x", attn_contribution=None, mlp_contribution=None,
        k_proj=None, v_proj=None,
    )
    with pytest.raises(Exception):
        paths.residual = "y"


# ----------------------------------------------------------------------------
# paths_for_layer + supported_components
# ----------------------------------------------------------------------------

def test_paths_for_layer_returns_all_components():
    paths = ARCHITECTURE_REGISTRY["llama"].paths_for_layer(5)
    assert isinstance(paths, LayerPaths)
    assert paths.residual == "model.layers.5"
    assert paths.attn_contribution == "model.layers.5.self_attn.o_proj"
    assert paths.k_proj == "model.layers.5.self_attn.k_proj"


def test_supported_components_excludes_unsupported():
    arch = ARCHITECTURE_REGISTRY["deepseek_v3"]
    supported = arch.supported_components(0)
    assert "residual" in supported
    assert "attn_contribution" in supported
    assert "k_proj" not in supported
    assert "v_proj" not in supported


def test_supported_components_qwen35_per_layer():
    arch = ARCHITECTURE_REGISTRY["qwen3_next"]
    # Linear-attn layer
    s0 = arch.supported_components(0)
    assert "k_proj" not in s0
    assert "v_proj" not in s0
    # Full-attn layer
    s3 = arch.supported_components(3)
    assert "k_proj" in s3
    assert "v_proj" in s3


# ----------------------------------------------------------------------------
# Standalone helpers
# ----------------------------------------------------------------------------

def test_standalone_helpers_delegate_to_registry():
    """Standalone layers/layer_prefix/inner_model helpers go through get_architecture."""
    import torch.nn as nn

    class LlamaCfg:
        model_type = "llama"

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            self.config = LlamaCfg()

    m = FakeModel()
    assert layers(m) is m.model.layers
    assert layer_prefix(m) == "model.layers"
    assert inner_model(m) is m.model


# ----------------------------------------------------------------------------
# Module tree validation
# ----------------------------------------------------------------------------

def test_validate_passes_when_tree_matches_live_block():
    """Walking module_tree against a block exposing the right names should pass."""
    import torch.nn as nn

    arch = ARCHITECTURE_REGISTRY["llama"]

    # Build a fake block with every submodule referenced in module_tree.
    block = nn.Module()
    for spec in arch.module_tree.values():
        cur = block
        parts = spec.path.split(".")
        for part in parts[:-1]:
            if not hasattr(cur, part):
                cur.add_module(part, nn.Module())
            cur = getattr(cur, part)
        if not hasattr(cur, parts[-1]):
            cur.add_module(parts[-1], nn.Module())

    class LlamaCfg:
        model_type = "llama"

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([block])
            self.config = LlamaCfg()

    arch.validate(FakeModel())  # should not raise


def test_validate_raises_when_tree_diverges_from_live_block():
    import torch.nn as nn

    arch = ARCHITECTURE_REGISTRY["llama"]

    class LlamaCfg:
        model_type = "llama"

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])  # empty block
            self.config = LlamaCfg()

    with pytest.raises(ArchitectureMismatchError, match="don't resolve"):
        arch.validate(FakeModel())


# ----------------------------------------------------------------------------
# Inner model / multimodal handling
# ----------------------------------------------------------------------------

def test_inner_model_unwraps_gemma3_multimodal():
    """For Gemma3 multimodal, inner_model should return model.model.language_model
    (where .layers actually lives), not model.model (where .vision_tower is too)."""
    class FakeLanguageModel:
        layers = []

    class Gemma3Inner:
        language_model = FakeLanguageModel()
        vision_tower = object()

    class TextCfg:
        model_type = "gemma3_text"

    class WrapperCfg:
        model_type = "gemma3"
        text_config = TextCfg()

    class FakeModel:
        config = WrapperCfg()
        model = Gemma3Inner()

    inner = inner_model(FakeModel())
    assert inner is FakeModel().model.language_model.__class__ or hasattr(inner, "layers")


def test_inner_model_unwraps_peft_then_multimodal():
    """LoRA-wrapped multimodal Gemma3 should still resolve to language_model."""
    class FakeLanguageModel:
        layers = []

    class Gemma3Inner:
        language_model = FakeLanguageModel()

    class TextCfg:
        model_type = "gemma3_text"

    class WrapperCfg:
        model_type = "gemma3"
        text_config = TextCfg()

    class InnerHF:
        config = WrapperCfg()
        model = Gemma3Inner()

    class PeftBaseModel:
        model = InnerHF()

    class PeftModel:
        base_model = PeftBaseModel()

    arch = get_architecture(PeftModel())
    inner = arch.inner_model(PeftModel())
    assert hasattr(inner, "layers")


# ----------------------------------------------------------------------------
# COMPONENTS contract
# ----------------------------------------------------------------------------

def test_components_tuple_is_canonical_set():
    assert set(COMPONENTS) == {
        "residual", "attn_contribution", "mlp_contribution", "k_proj", "v_proj",
    }


