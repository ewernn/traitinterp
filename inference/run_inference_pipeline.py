#!/usr/bin/env python3
"""
Inference pipeline: generate responses, capture activations, project onto trait vectors.

Default mode (stream-through): projects during capture — no intermediate .pt files.
Use --from-activations to skip capture and project from saved .pt files.

Stages:
    1. generate    - Generate model responses (or skip with --skip-generate)
    2. project     - Capture activations and project onto trait vectors

Output:
    experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json
    experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json

Usage:
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main_prompts
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main_prompts --skip-generate
    python inference/run_inference_pipeline.py --experiment my_exp --prompt-set main_prompts --from-activations
"""

import sys
import gc
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datetime import datetime
from tqdm import tqdm

from utils.paths import (
    get as get_path, get_model_variant, load_experiment_config,
    get_default_variant, discover_extracted_traits,
)
from utils.backends import LocalBackend, add_backend_args
from utils.model import tokenize
from utils.json import dump_compact
from utils.vector_selection import load_trait_vectors
from utils.vram import format_duration
from core import MultiLayerProjection
from inference.process_activations import load_massive_dims_from_analysis


def stream_through_project(
    model, tokenizer, response_files, trait_vectors, vectors_by_layer, hook_index,
    component, inference_dir, prompt_set, experiment,
    skip_existing=False, centered=False,
):
    """Capture activations and project onto trait vectors in one forward pass.

    Uses MultiLayerProjection to project on GPU inside hooks. Only small
    score arrays cross the GPU-CPU boundary.
    """
    massive_dims_info = None
    analysis_massive_dims, top_dims_by_layer = load_massive_dims_from_analysis(experiment)
    if analysis_massive_dims:
        massive_dims_info = (analysis_massive_dims, top_dims_by_layer)

    layers = sorted(vectors_by_layer.keys())
    n_projected = 0

    # Pre-compute reverse lookup: (category, trait_name) -> list of (vec_list_idx, layer, slot)
    trait_to_slots = {}
    for (layer, slot), (cat, trait_name, vec_list_idx) in hook_index.items():
        key = (cat, trait_name)
        if key not in trait_to_slots:
            trait_to_slots[key] = []
        trait_to_slots[key].append((vec_list_idx, layer, slot))

    for response_file in tqdm(response_files, desc="Projecting"):
        prompt_id = response_file.stem

        with open(response_file) as f:
            resp_data = json.load(f)

        prompt_end = resp_data.get('prompt_end', 0)
        all_tokens = resp_data.get('tokens', [])
        n_prompt = len(all_tokens[:prompt_end])
        n_response = len(all_tokens[prompt_end:])

        # Tokenize and forward pass with projection hooks
        full_text = resp_data.get('prompt', '') + resp_data.get('response', '')
        inputs = tokenize(full_text, tokenizer)
        input_ids = inputs.input_ids.to(next(model.parameters()).device)
        attention_mask = inputs.attention_mask.to(input_ids.device)

        with MultiLayerProjection(
            model, vectors_by_layer, component=component, compute_norms=True,
        ) as proj:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            all_scores = proj.get_all()      # {layer: [1, seq, n_vectors]}
            all_norms = proj.get_all_norms()  # {layer: [1, seq]}

        # Pre-compute per-layer mean norms (same for all traits)
        prompt_norms_list = []
        response_norms_list = []
        for layer in sorted(all_norms.keys()):
            layer_n = all_norms[layer][0]
            prompt_norms_list.append(layer_n[:n_prompt].mean().item() if n_prompt > 0 else 0.0)
            response_norms_list.append(layer_n[n_prompt:].mean().item())

        # Build per-trait projection output
        for (category, trait_name), vector_list in trait_vectors.items():
            trait_path = f"{category}/{trait_name}"
            out_dir = inference_dir / "projections" / trait_path / prompt_set
            out_file = out_dir / f"{prompt_id}.json"

            if skip_existing and out_file.exists():
                continue

            slots = trait_to_slots.get((category, trait_name), [])

            all_projections = []
            for vec_list_idx, layer, slot in slots:
                vec_entry = vector_list[vec_list_idx]
                _, method, _, _, _, selection_source, baseline, position = vec_entry

                layer_scores = all_scores[layer][0, :, slot]
                prompt_proj = layer_scores[:n_prompt].tolist()
                response_proj = layer_scores[n_prompt:].tolist()

                if centered and baseline != 0.0:
                    prompt_proj = [s - baseline for s in prompt_proj]
                    response_proj = [s - baseline for s in response_proj]

                layer_norms = all_norms[layer][0]
                all_projections.append({
                    'method': method,
                    'layer': layer,
                    'selection_source': selection_source,
                    'baseline': baseline,
                    'prompt': prompt_proj,
                    'response': response_proj,
                    'token_norms': {
                        'prompt': layer_norms[:n_prompt].tolist(),
                        'response': layer_norms[n_prompt:].tolist(),
                    },
                })

            first_position = vector_list[0][7] if vector_list else 'response[:5]'

            proj_data = {
                'metadata': {
                    'prompt_id': prompt_id,
                    'prompt_set': prompt_set,
                    'n_prompt_tokens': n_prompt,
                    'n_response_tokens': n_response,
                    'multi_vector': True,
                    'n_vectors': len(all_projections),
                    'component': component,
                    'position': first_position,
                    'centered': centered,
                    'projection_date': datetime.now().isoformat(),
                },
                'projections': all_projections,
                'activation_norms': {
                    'prompt': prompt_norms_list,
                    'response': response_norms_list,
                },
            }

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(proj_data, f)

        n_projected += 1
        del all_scores, all_norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n_projected


def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline: generate → capture → project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--model-variant", default=None)

    # Pipeline control
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--from-activations", action="store_true",
                        help="Project from saved .pt files (no GPU capture)")

    # Projection options
    parser.add_argument("--traits", type=str, default=None)
    parser.add_argument("--layers", type=str, default="best,best+5")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    # Generation options
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Model options
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    add_backend_args(parser)

    args = parser.parse_args()

    # Resolve experiment config
    config = load_experiment_config(args.experiment)
    model_variant = args.model_variant or get_default_variant(args.experiment, mode='application')
    variant_info = get_model_variant(args.experiment, model_variant)
    model_name = variant_info['model']
    extraction_variant = get_default_variant(args.experiment, mode='extraction')

    # Resolve traits
    if args.traits:
        traits = [t.strip() for t in args.traits.split(',')]
    else:
        traits = discover_extracted_traits(args.experiment, extraction_variant)
    if not traits:
        print("No extracted traits found")
        return

    inference_dir = Path(get_path('inference.base', experiment=args.experiment, model_variant=model_variant))

    print(f"{'=' * 60}")
    print(f"INFERENCE PIPELINE | {args.experiment}")
    print(f"Model: {model_name} | Variant: {model_variant}")
    print(f"Prompt set: {args.prompt_set} | Traits: {len(traits)}")
    print(f"{'=' * 60}")

    pipeline_start = time.time()

    # Stage 1: Generate responses
    if not args.skip_generate and not args.from_activations:
        from inference.generate_responses import generate_responses
        print(f"\n[1] Generating responses...")
        gen_start = time.time()
        n_generated = generate_responses(
            experiment=args.experiment,
            prompt_set=args.prompt_set,
            model_variant=model_variant,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            skip_existing=args.skip_existing,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            no_server=args.backend == 'local',
        )
        print(f"    Generated {n_generated} responses ({format_duration(time.time() - gen_start)})")

    # Stage 2: Project
    if args.from_activations:
        # Project from saved .pt files via process_prompt_set
        from inference.process_activations import process_prompt_set
        print(f"\n[2] Projecting from saved activations...")

        # Build a minimal args namespace for process_prompt_set
        proj_args = argparse.Namespace(
            experiment=args.experiment,
            component=args.component,
            layers=args.layers,
            layer=None,
            position='response[:5]',
            method=None,
            multi_vector=None,
            logit_lens=False,
            skip_existing=args.skip_existing,
            centered=args.centered,
            traits=args.traits,
            vectors_experiment=None,
            steering_variant=None,
        )
        process_prompt_set(
            proj_args, inference_dir, args.prompt_set,
            model_name, model_variant, extraction_variant,
            args.experiment, None,
        )
    else:
        # Stream-through: capture + project in one pass
        print(f"\n[2] Stream-through capture + projection...")
        proj_start = time.time()

        backend = LocalBackend.from_experiment(
            args.experiment, variant=model_variant,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )

        trait_vectors, vectors_by_layer, hook_index = load_trait_vectors(
            args.experiment, extraction_variant, traits, args.component, args.layers,
        )

        if not vectors_by_layer:
            print("No vectors loaded — nothing to project")
            return

        print(f"    Loaded {sum(v.shape[0] for v in vectors_by_layer.values())} vectors "
              f"across {len(vectors_by_layer)} layers")

        responses_dir = inference_dir / "responses" / args.prompt_set
        if not responses_dir.exists():
            print(f"No responses found at {responses_dir}")
            return

        response_files = sorted(responses_dir.glob("*.json"))
        if not response_files:
            print("No response files found")
            return

        n_projected = stream_through_project(
            backend.model, backend.tokenizer, response_files,
            trait_vectors, vectors_by_layer, hook_index,
            args.component, inference_dir, args.prompt_set, args.experiment,
            skip_existing=args.skip_existing,
            centered=args.centered,
        )
        print(f"    Projected {n_projected} prompts ({format_duration(time.time() - proj_start)})")

        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - pipeline_start
    print(f"\nPipeline complete ({format_duration(total)})")


if __name__ == "__main__":
    main()
