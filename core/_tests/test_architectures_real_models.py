"""Empty-weights parity tests against real HuggingFace architectures.

Loads each registered model_type's HF config (no weights, no GPU) via
init_empty_weights() + AutoConfig + from_config, then verifies the adapter's
declared paths and module_tree resolve to real submodules on the live model.

Catches the bug class where an adapter declares a path that doesn't exist on
the actual HF model — usually because HF refactored a module name in a newer
transformers release, or because the adapter was written by hand without
checking.

Tests skip cleanly when:
- accelerate is not installed (init_empty_weights unavailable)
- the model_type isn't registered in the installed transformers version
- the model is gated and HF_TOKEN isn't set (some Llama/Gemma checkpoints)

Run manually before merging changes to core/architectures/. Cheap (no GPU,
no weights downloaded — just the config JSON), fast (a few seconds total).
"""

import os
import warnings

import pytest

from core.architectures import (
    ARCHITECTURE_REGISTRY,
    HybridArchitecture,
    UnsupportedComponentError,
    get_architecture,
)


# Per-arch test models: pick the smallest publicly available config per family.
# We use AutoConfig.from_pretrained(...) to fetch just the JSON; no weights.
ARCH_TEST_MODELS = {
    "llama":       "meta-llama/Llama-3.2-1B",
    "mistral":     "mistralai/Mistral-7B-v0.1",
    "qwen2":       "Qwen/Qwen2.5-0.5B",
    "qwen3":       "Qwen/Qwen3-4B",
    "qwen3_next": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "gemma2":      "google/gemma-2-2b",
    "gemma3":      "google/gemma-3-4b-it",
    "olmo2":       "allenai/OLMo-2-1124-7B",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "gpt_oss":     "openai/gpt-oss-20b",
}


def _model_class_available(model_type: str) -> bool:
    """True if this model_type can be constructed by the installed transformers version."""
    try:
        from transformers import CONFIG_MAPPING
        return model_type in CONFIG_MAPPING
    except ImportError:
        return False


def _accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


def _build_empty_model(model_id: str, model_type: str):
    """Build a HF model with no weights for the given checkpoint id.

    Returns the empty-weights model, or skips the test if the build fails for
    reasons unrelated to architecture correctness (gated repo, network, etc.).
    """
    if not _accelerate_available():
        pytest.skip("accelerate not installed; init_empty_weights unavailable")
    if not _model_class_available(model_type):
        pytest.skip(f"transformers does not register model_type={model_type!r}")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"AutoConfig.from_pretrained({model_id!r}) failed: {e}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with init_empty_weights():
            try:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            except Exception as e:
                pytest.skip(f"AutoModelForCausalLM.from_config({model_id!r}) failed: {e}")
    return model


@pytest.mark.parametrize("model_type,model_id", sorted(ARCH_TEST_MODELS.items()))
def test_adapter_validates_against_live_model(model_type, model_id):
    """The adapter's module_tree resolves on the actual HF model (no missing paths)."""
    model = _build_empty_model(model_id, model_type)
    arch = get_architecture(model)
    arch.validate(model)  # raises ArchitectureMismatchError if any path doesn't resolve


@pytest.mark.parametrize("model_type,model_id", sorted(ARCH_TEST_MODELS.items()))
def test_adapter_paths_resolve_per_layer(model_type, model_id):
    """For every block, the paths declared by paths_for_layer(idx) navigate to real modules.

    Hybrid architectures dispatch per-layer (different paths on different blocks);
    we use suffix_for when available so the resolved suffix matches what the
    hook layer would actually use at capture time.
    """
    from core.hooks import resolve_hook_path

    model = _build_empty_model(model_id, model_type)
    arch = get_architecture(model)
    blocks = arch.layers(model)
    failures = []
    for idx in range(len(blocks)):
        # resolve_hook_path is what production hooks call; matches their behavior exactly
        for component in ("residual", "attn_contribution", "mlp_contribution", "k_proj", "v_proj"):
            try:
                path = resolve_hook_path(model, idx, component)
            except UnsupportedComponentError:
                continue
            try:
                model.get_submodule(path)
            except AttributeError as e:
                failures.append(f"layer {idx} {component}={path!r}: {e}")

    if failures:
        pytest.fail(f"{model_type}: paths did not resolve on live model:\n  " + "\n  ".join(failures[:10]))


@pytest.mark.parametrize("model_type,model_id", sorted(ARCH_TEST_MODELS.items()))
def test_layer_count_matches_config(model_type, model_id):
    """arch.layers(model) returns the same number of layers as model.config reports."""
    model = _build_empty_model(model_id, model_type)
    arch = get_architecture(model)
    actual = len(arch.layers(model))

    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    expected = config.num_hidden_layers

    assert actual == expected, (
        f"{model_type}: arch.layers(model) returned {actual} blocks but "
        f"config.num_hidden_layers = {expected}"
    )


def test_kimi_k2_alias_resolves_via_real_config():
    """Kimi K2 ships with model_type='kimi_k2' but reuses the deepseek_v3 architecture."""
    if not _accelerate_available():
        pytest.skip("accelerate not installed")
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained("moonshotai/Kimi-K2-Instruct", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not fetch Kimi K2 config: {e}")

    # The remote config defines model_type via its custom class; verify it resolves
    # through the alias to the deepseek_v3 architecture.
    class _Stub:
        pass
    stub = _Stub()
    stub.config = config
    arch = get_architecture(stub)
    assert arch is ARCHITECTURE_REGISTRY["deepseek_v3"]


@pytest.mark.parametrize("model_type,model_id", [
    ("deepseek_v3", "deepseek-ai/DeepSeek-V3"),
])
def test_hybrid_runtime_dispatch_against_live_blocks(model_type, model_id):
    """For HybridArchitecture, runtime block introspection picks the right path on real layers."""
    model = _build_empty_model(model_id, model_type)
    arch = get_architecture(model)
    if not isinstance(arch, HybridArchitecture):
        pytest.skip(f"{model_type} is not a HybridArchitecture")

    blocks = arch.layers(model)

    # DeepSeek V3: layer 0 is dense (has mlp.down_proj), later layers are MoE (no down_proj).
    # Verify runtime dispatch picks the right suffix on each.
    suffix_dense = arch.suffix_for("mlp_contribution", layer=0, block=blocks[0])
    suffix_moe = arch.suffix_for("mlp_contribution", layer=5, block=blocks[5])

    assert suffix_dense == "mlp.down_proj", (
        f"{model_type}: layer 0 (dense) should resolve to mlp.down_proj, got {suffix_dense!r}"
    )
    assert suffix_moe == "mlp", (
        f"{model_type}: layer 5 (MoE) should resolve to mlp (whole module), got {suffix_moe!r}"
    )
