#!/usr/bin/env python3
"""
Inference pipeline: generate responses, project onto trait vectors.

Three projection modes:
    default         Generate (if needed) → prefill with projection hooks → save scores
    --capture       Generate (if needed) → save raw .pt activations (no projection)
    --from-activations  Project from saved .pt files (after --capture)

Responses are saved as individual JSON files and serve as the checkpoint.
Re-running without --regenerate skips generation and re-projects existing responses.

Input:  datasets/inference/{prompt_set}.json + extracted trait vectors
Output:
    responses:   experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json
    projections: experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json
    raw (capture): experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{id}.pt

Usage:
    # Generate + project (default)
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main

    # Re-project existing responses with different vectors/layers
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main --layers best

    # Force re-generate responses
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main --regenerate

    # Capture raw activations (for later re-projection)
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main --capture

    # Project from saved .pt files
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main --from-activations
"""

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from core.kwargs_configs import InferenceConfig
from utils.paths import (
    get as get_path, get_model_variant, get_default_variant,
    discover_extracted_traits,
)
from utils.backends import LocalBackend, add_backend_args
from utils.vector_selection import load_trait_vectors
from utils.distributed import flush_cuda
from utils.vram import format_duration


# =============================================================================
# Recipe
# =============================================================================

def run_pipeline(config: InferenceConfig):
    """Generate → project (or capture)."""
    if config.backend == 'vllm':
        # vLLM supports generation only — projection & capture need HF hooks.
        if config.capture:
            raise ValueError("--backend vllm cannot --capture activations; use --backend local.")
        if not config.from_activations:
            raise ValueError(
                "--backend vllm supports generation only; default projection (stream-through) "
                "requires --backend local. Generate with --backend vllm separately, then project "
                "via --from-activations with --backend local."
            )

    variant_info = get_model_variant(config.experiment, config.model_variant, mode='application')
    model_variant = variant_info.name
    model_name = variant_info.model

    if config.extraction_variant is None:
        config.extraction_variant = get_default_variant(config.experiment, mode='extraction')

    inference_dir = Path(get_path('inference.variant', experiment=config.experiment, model_variant=model_variant))

    # Generate responses if needed. generate_responses() skips per-prompt when
    # output JSON already exists, so we always dispatch when in a mode that
    # needs responses — a partial output dir gets its missing files filled in
    # instead of being silently skipped.
    if config.regenerate or not config.from_activations:
        generate(config, model_variant)

    # Capture or project
    if config.capture:
        capture(config, model_variant)
    elif config.from_activations:
        project_from_saved_activations(config, inference_dir, model_name, model_variant)
    else:
        project_stream_through(config, inference_dir, model_variant)


# =============================================================================
# Stage implementations
# =============================================================================

def generate(config: InferenceConfig, model_variant: str) -> int:
    """Generate model responses for the prompt set."""
    from inference.generate_responses import generate_responses
    return generate_responses(
        experiment=config.experiment,
        prompt_set=config.prompt_set,
        model_variant=model_variant,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        force=config.regenerate, load_in_4bit=config.load_in_4bit,
        no_server=config.no_server,
        backend=config.backend,
    )


def capture(config: InferenceConfig, model_variant: str) -> int:
    """Capture raw activations to .pt files (no projection)."""
    from utils.capture_activations import capture_raw_activations
    return capture_raw_activations(
        experiment=config.experiment,
        prompt_set=config.prompt_set,
        model_variant=model_variant,
        layers=config.layers,
        force=config.force, load_in_4bit=config.load_in_4bit,
    )


def project_from_saved_activations(config: InferenceConfig, inference_dir: Path,
                                    model_name: str, model_variant: str):
    """Project from saved .pt files."""
    from utils.project_activations import project_from_saved

    project_from_saved(
        inference_dir, config.prompt_set, model_name, model_variant,
        config.extraction_variant, config.experiment, None,
        experiment=config.experiment,
        component=config.component,
        layers=config.layers,
        force=config.force,
        centered=config.centered,
        traits=','.join(config.traits) if config.traits else None,
        score_mode=config.score_mode,
    )


def project_stream_through(config: InferenceConfig, inference_dir: Path,
                            model_variant: str) -> int:
    """Prefill forward pass with projection hooks (default mode)."""
    from utils.project_activations import stream_through_project

    # Resolve inputs — bail early if anything's missing
    if config.traits:
        traits = config.traits
    else:
        trait_tuples = discover_extracted_traits(config.experiment, config.extraction_variant)
        traits = [f"{cat}/{name}" for cat, name in trait_tuples]
    if not traits:
        print("  No traits found — nothing to project")
        return 0

    responses_dir = inference_dir / "responses" / config.prompt_set
    response_files = sorted(responses_dir.glob("*.json")) if responses_dir.exists() else []
    if not response_files:
        print(f"  No responses at {responses_dir}")
        return 0

    trait_vectors, vectors_by_layer, hook_index = load_trait_vectors(
        config.experiment, config.extraction_variant, traits,
        config.component, config.layers,
    )
    if not vectors_by_layer:
        print("  No vectors loaded — nothing to project")
        return 0

    # All inputs ready — load model and project
    # TODO: pass model from generation stage to avoid double load (~3-6 min wasted per variant on 70B)
    backend = LocalBackend.from_experiment(
        config.experiment, variant=model_variant, load_in_4bit=config.load_in_4bit,
    )

    try:
        n = stream_through_project(
            backend.model, backend.tokenizer, response_files,
            trait_vectors, vectors_by_layer, hook_index,
            config.component, inference_dir, config.prompt_set, config.experiment,
            force=config.force, centered=config.centered,
            score_mode=config.score_mode,
        )
    finally:
        del backend
        flush_cuda()

    return n


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline: generate → project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--model-variant", default=None)

    # Pipeline control
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture", action="store_true",
                      help="Save raw .pt activations instead of projecting")
    mode.add_argument("--from-activations", action="store_true",
                      help="Project from saved .pt files (after --capture)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force re-generate responses (default: skip if responses exist)")

    # Projection
    parser.add_argument("--traits", type=str, default=None)
    parser.add_argument("--layers", type=str, default="best,best+5")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--force", action="store_true")

    # Scoring
    parser.add_argument("--score-mode", default="normalized",
                        choices=["raw", "normalized", "cosine"],
                        help="Projection score normalization: raw (no divisor), "
                             "normalized (÷ mean ||h|| over response, default), "
                             "cosine (÷ per-token ||h||, true cosine similarity)")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Model
    parser.add_argument("--load-in-4bit", action="store_true")
    add_backend_args(parser)

    args = parser.parse_args()

    config = InferenceConfig(
        experiment=args.experiment,
        prompt_set=args.prompt_set,
        model_variant=args.model_variant,
        regenerate=args.regenerate,
        capture=args.capture,
        from_activations=args.from_activations,
        traits=args.traits.split(',') if args.traits else None,
        layers=args.layers,
        component=args.component,
        centered=args.centered,
        force=args.force,
        score_mode=args.score_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        no_server=(args.backend == 'local'),
        backend=args.backend,
        load_in_4bit=args.load_in_4bit,
    )

    t = time.time()
    run_pipeline(config)
    print(f"\nComplete ({format_duration(time.time() - t)})")


if __name__ == "__main__":
    main()
