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
import gc
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from core.kwargs_configs import InferenceConfig
from utils.paths import (
    get as get_path, get_model_variant, get_default_variant,
    discover_extracted_traits,
)
from utils.backends import LocalBackend, add_backend_args
from utils.vector_selection import load_trait_vectors
from utils.vram import format_duration


# =============================================================================
# Recipe
# =============================================================================

def run_pipeline(config: InferenceConfig):
    """The recipe: generate → project."""
    variant_info = get_model_variant(config.experiment, config.model_variant, mode='application')
    model_variant = variant_info['name']
    model_name = variant_info['model']

    if config.extraction_variant is None:
        config.extraction_variant = get_default_variant(config.experiment, mode='extraction')

    inference_dir = Path(get_path('inference.base', experiment=config.experiment, model_variant=model_variant))

    print("=" * 60)
    print(f"INFERENCE PIPELINE | {config.experiment}")
    print(f"Model: {model_name} | Variant: {model_variant}")
    print(f"Prompt set: {config.prompt_set}")
    print("=" * 60)

    pipeline_start = time.time()

    # stage 1: generate responses
    if not config.skip_generate and not config.from_activations:
        print(f"\n[1] Generating responses...")
        t = time.time()
        n = generate(config, model_variant)
        print(f"    Generated {n} responses ({format_duration(time.time() - t)})")

    # stage 2: project onto trait vectors
    if config.from_activations:
        print(f"\n[2] Projecting from saved activations...")
        project_from_saved(config, inference_dir, model_name, model_variant)
    else:
        print(f"\n[2] Stream-through capture + projection...")
        t = time.time()
        n = project_stream_through(config, inference_dir, model_variant)
        print(f"    Projected {n} prompts ({format_duration(time.time() - t)})")

    print(f"\nPipeline complete ({format_duration(time.time() - pipeline_start)})")


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

    proj_args = argparse.Namespace(
        experiment=config.experiment,
        component=config.component,
        layers=config.layers,
        layer=None,
        position='response[:5]',
        method=None,
        multi_vector=None,
        logit_lens=False,
        skip_existing=config.skip_existing,
        centered=config.centered,
        traits=','.join(config.traits) if config.traits else None,
        vectors_experiment=None,
        steering_variant=None,
    )
    process_prompt_set(
        proj_args, inference_dir, config.prompt_set,
        model_name, model_variant, config.extraction_variant,
        config.experiment, None,
    )


def project_stream_through(config: InferenceConfig, inference_dir: Path,
                            model_variant: str) -> int:
    """Stage 2 (default): Capture activations and project in one forward pass."""
    from utils.process_activations import stream_through_project

    traits = config.traits or discover_extracted_traits(config.experiment, config.extraction_variant)
    if not traits:
        print("    No traits found — nothing to project")
        return 0

    backend = LocalBackend.from_experiment(
        config.experiment, variant=model_variant,
        load_in_8bit=config.load_in_8bit, load_in_4bit=config.load_in_4bit,
    )

    trait_vectors, vectors_by_layer, hook_index = load_trait_vectors(
        config.experiment, config.extraction_variant, traits,
        config.component, config.layers,
    )
    if not vectors_by_layer:
        print("    No vectors loaded — nothing to project")
        del backend
        return 0

    n_vecs = sum(v.shape[0] for v in vectors_by_layer.values())
    print(f"    Loaded {n_vecs} vectors across {len(vectors_by_layer)} layers")

    responses_dir = inference_dir / "responses" / config.prompt_set
    if not responses_dir.exists():
        print(f"    No responses at {responses_dir}")
        del backend
        return 0

    response_files = sorted(responses_dir.glob("*.json"))
    if not response_files:
        print("    No response files found")
        del backend
        return 0

    n = stream_through_project(
        backend.model, backend.tokenizer, response_files,
        trait_vectors, vectors_by_layer, hook_index,
        config.component, inference_dir, config.prompt_set, config.experiment,
        skip_existing=config.skip_existing, centered=config.centered,
    )

    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    run_pipeline(config)


if __name__ == "__main__":
    main()
