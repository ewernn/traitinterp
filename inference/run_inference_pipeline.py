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
from utils.backends import add_backend_args
from utils.distributed import flush_cuda
from utils.vram import format_duration
from utils.inference import (
    init_hf_backend, generate, capture, project_from_saved, project_stream_through,
)


# =============================================================================
# Recipe
# =============================================================================

def run_pipeline(config: InferenceConfig):
    """Generate → capture / project (mode-dependent)."""
    _check_vllm_compatibility(config)
    shared = init_hf_backend(config)
    try:
        if not config.from_activations:
            generate(config, shared)
        if config.capture:
            capture(config, shared)
        elif config.from_activations:
            project_from_saved(config)
        else:
            project_stream_through(config, shared)
    finally:
        if shared is not None:
            del shared
            flush_cuda()


def _check_vllm_compatibility(config: InferenceConfig):
    """vLLM supports generation only — hooks (projection / capture) need HF."""
    if config.backend != 'vllm':
        return
    if config.capture:
        raise ValueError("--backend vllm cannot --capture activations; use --backend local.")
    if not config.from_activations:
        raise ValueError(
            "--backend vllm supports generation only; default projection (stream-through) "
            "requires --backend local. Generate with --backend vllm separately, then project "
            "via --from-activations with --backend local."
        )


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
    parser.set_defaults(backend='local')

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
