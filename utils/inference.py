"""
Inference pipeline stages.

Stage orchestrators called by inference/run_inference_pipeline.py. Each stage
takes (config, shared_backend) and writes outputs to the canonical paths.

When the pipeline can share a single HF backend across stages (default path:
generate + project), init_hf_backend() loads it once and each stage reuses it.
vLLM, server, and --from-activations modes don't share (different constraints),
so init_hf_backend returns None and each stage loads its own if needed.

Stages:
    generate                  Generate model responses for a prompt set
    capture                   Save raw .pt activations (no projection)
    project_from_saved        Project from saved .pt files (CPU)
    project_stream_through    Prefill with projection hooks (needs HF backend)
"""

from pathlib import Path

from core.kwargs_configs import InferenceConfig
from utils.backends import LocalBackend
from utils.paths import (
    get as get_path, get_model_variant, get_default_variant,
    discover_extracted_traits,
)
from utils.distributed import flush_cuda
from utils.vector_selection import load_trait_vectors


def init_hf_backend(config: InferenceConfig):
    """Pre-load a single HF backend only when `--backend local` + stages can share it.

    Returns None for:
    - `--backend vllm`: vllm owns its own lifecycle inside generate_responses
    - `--backend auto` / `server`: generate may resolve to a remote ModelClient;
      pre-loading a LocalBackend here would silently override server detection.
      Each stage handles its own model loading in those modes (matches today).
    - `--from-activations`: CPU-only projection, no HF model needed

    When None is returned, project_stream_through falls back to loading its own
    LocalBackend internally (via the `owned = shared is None` pattern).
    """
    if config.backend != 'local' or config.from_activations:
        return None
    variant = get_model_variant(config.experiment, config.model_variant, mode='application')
    return LocalBackend.from_experiment(
        config.experiment, variant=variant.name,
        load_in_4bit=config.load_in_4bit,
    )


def generate(config: InferenceConfig, shared: LocalBackend = None) -> int:
    """Stage: generate model responses for the prompt set."""
    from inference.generate_responses import generate_responses
    variant = get_model_variant(config.experiment, config.model_variant, mode='application')
    return generate_responses(
        experiment=config.experiment,
        prompt_set=config.prompt_set,
        model_variant=variant.name,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        force=config.regenerate,
        load_in_4bit=config.load_in_4bit,
        no_server=config.no_server,
        backend=config.backend,
        model=shared.model if shared is not None else None,
        tokenizer=shared.tokenizer if shared is not None else None,
    )


def capture(config: InferenceConfig, shared: LocalBackend = None) -> int:
    """Stage: capture raw activations to .pt files (no projection)."""
    from utils.capture_activations import capture_raw_activations
    variant = get_model_variant(config.experiment, config.model_variant, mode='application')
    return capture_raw_activations(
        experiment=config.experiment,
        prompt_set=config.prompt_set,
        model_variant=variant.name,
        layers=config.layers,
        force=config.force,
        load_in_4bit=config.load_in_4bit,
        model=shared.model if shared is not None else None,
        tokenizer=shared.tokenizer if shared is not None else None,
    )


def project_from_saved(config: InferenceConfig):
    """Stage: project from saved .pt files. CPU-only, no HF backend needed."""
    from utils.project_activations import project_from_saved as _project_from_saved
    variant = get_model_variant(config.experiment, config.model_variant, mode='application')
    model_variant = variant.name
    model_name = variant.model
    extraction_variant = config.extraction_variant or get_default_variant(config.experiment, mode='extraction')
    inference_dir = Path(get_path('inference.variant', experiment=config.experiment, model_variant=model_variant))
    return _project_from_saved(
        inference_dir, config.prompt_set, model_name, model_variant,
        extraction_variant, config.experiment, None,
        experiment=config.experiment,
        component=config.component,
        layers=config.layers,
        force=config.force,
        centered=config.centered,
        traits=','.join(config.traits) if config.traits else None,
        score_mode=config.score_mode,
    )


def project_stream_through(config: InferenceConfig, shared: LocalBackend) -> int:
    """Stage: prefill forward pass with projection hooks. Requires a loaded HF backend."""
    from utils.project_activations import stream_through_project

    variant = get_model_variant(config.experiment, config.model_variant, mode='application')
    model_variant = variant.name
    extraction_variant = config.extraction_variant or get_default_variant(config.experiment, mode='extraction')
    inference_dir = Path(get_path('inference.variant', experiment=config.experiment, model_variant=model_variant))

    # Resolve traits — bail early if none
    if config.traits:
        traits = config.traits
    else:
        trait_tuples = discover_extracted_traits(config.experiment, extraction_variant)
        traits = [f"{cat}/{name}" for cat, name in trait_tuples]
    if not traits:
        print("  No traits found — nothing to project")
        return 0

    # Bail if no responses
    responses_dir = inference_dir / "responses" / config.prompt_set
    response_files = sorted(responses_dir.glob("*.json")) if responses_dir.exists() else []
    if not response_files:
        print(f"  No responses at {responses_dir}")
        return 0

    # Bail if no vectors
    trait_vectors, vectors_by_layer, hook_index = load_trait_vectors(
        config.experiment, extraction_variant, traits,
        config.component, config.layers,
    )
    if not vectors_by_layer:
        print("  No vectors loaded — nothing to project")
        return 0

    # Own the backend if caller didn't pass one (e.g., dev script calling directly).
    owned = shared is None
    if owned:
        shared = LocalBackend.from_experiment(
            config.experiment, variant=model_variant, load_in_4bit=config.load_in_4bit,
        )

    try:
        return stream_through_project(
            shared.model, shared.tokenizer, response_files,
            trait_vectors, vectors_by_layer, hook_index,
            config.component, inference_dir, config.prompt_set, config.experiment,
            force=config.force, centered=config.centered,
            score_mode=config.score_mode,
        )
    finally:
        if owned:
            del shared
            flush_cuda()
