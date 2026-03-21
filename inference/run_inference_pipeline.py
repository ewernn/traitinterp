#!/usr/bin/env python3
"""
Inference pipeline: generate responses, project onto trait vectors.

Default mode (stream-through): projects during capture — no intermediate .pt files.
Use --from-activations to project from saved .pt files instead.

Stages:
    1: generate    --max-new-tokens, --temperature    Generate model responses
    2: project     --layers, --traits, --component    Capture + project (stream-through)

Usage:
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main --skip-generate
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
    """Generate → project."""
    variant_info = get_model_variant(config.experiment, config.model_variant, mode='application')
    model_variant = variant_info.name
    model_name = variant_info.model

    if config.extraction_variant is None:
        config.extraction_variant = get_default_variant(config.experiment, mode='extraction')

    inference_dir = Path(get_path('inference.base', experiment=config.experiment, model_variant=model_variant))

    if not config.skip_generate and not config.from_activations:
        generate(config, model_variant)

    if config.from_activations:
        project_from_saved(config, inference_dir, model_name, model_variant)
    else:
        project_stream_through(config, inference_dir, model_variant)


# =============================================================================
# Stage implementations
# =============================================================================

def generate(config: InferenceConfig, model_variant: str) -> int:
    """Stage 1: Generate model responses for the prompt set."""
    from utils.inference_generation import generate_responses
    return generate_responses(
        experiment=config.experiment,
        prompt_set=config.prompt_set,
        model_variant=model_variant,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        skip_existing=config.skip_existing,
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
        no_server=config.no_server,
    )


def project_from_saved(config: InferenceConfig, inference_dir: Path,
                       model_name: str, model_variant: str):
    """Stage 2 (alt): Project from saved .pt files."""
    from utils.process_activations import process_prompt_set

    process_prompt_set(
        inference_dir, config.prompt_set, model_name, model_variant,
        config.extraction_variant, config.experiment, None,
        experiment=config.experiment,
        component=config.component,
        layers=config.layers,
        skip_existing=config.skip_existing,
        centered=config.centered,
        traits=','.join(config.traits) if config.traits else None,
    )


def project_stream_through(config: InferenceConfig, inference_dir: Path,
                            model_variant: str) -> int:
    """Stage 2 (default): Capture activations and project in one forward pass."""
    from utils.process_activations import stream_through_project

    # Resolve inputs — bail early if anything's missing
    traits = config.traits or discover_extracted_traits(config.experiment, config.extraction_variant)
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
    backend = LocalBackend.from_experiment(
        config.experiment, variant=model_variant,
        load_in_8bit=config.load_in_8bit, load_in_4bit=config.load_in_4bit,
    )

    try:
        n = stream_through_project(
            backend.model, backend.tokenizer, response_files,
            trait_vectors, vectors_by_layer, hook_index,
            config.component, inference_dir, config.prompt_set, config.experiment,
            skip_existing=config.skip_existing, centered=config.centered,
        )
    finally:
        del backend
        flush_cuda()

    return n


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference pipeline: generate → project")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--model-variant", default=None)

    # Pipeline control
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--from-activations", action="store_true",
                        help="Project from saved .pt files (no GPU capture)")

    # Projection
    parser.add_argument("--traits", type=str, default=None)
    parser.add_argument("--layers", type=str, default="best,best+5")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Model
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    add_backend_args(parser)

    args = parser.parse_args()

    config = InferenceConfig(
        experiment=args.experiment,
        prompt_set=args.prompt_set,
        model_variant=args.model_variant,
        skip_generate=args.skip_generate,
        from_activations=args.from_activations,
        traits=args.traits.split(',') if args.traits else None,
        layers=args.layers,
        component=args.component,
        centered=args.centered,
        skip_existing=args.skip_existing,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        no_server=(args.backend == 'local'),
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    t = time.time()
    run_pipeline(config)
    print(f"\nComplete ({format_duration(time.time() - t)})")


if __name__ == "__main__":
    main()
