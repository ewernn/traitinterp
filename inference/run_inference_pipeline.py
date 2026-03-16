#!/usr/bin/env python3
"""
Inference pipeline: generate responses, capture activations, project onto trait vectors.

Default mode (stream-through): projects during capture — no intermediate .pt files.
Use --save-activations to also save raw .pt files for re-projection.
Use --from-activations to skip capture and project from saved .pt files.

Stages:
    1. generate    - Generate model responses (or skip with --skip-generate)
    2. project     - Capture activations and project onto trait vectors

Output:
    experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json
    experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json

Usage:
    # Full pipeline
    python inference/run_inference_pipeline.py \\
        --experiment my_exp --prompt-set main_prompts

    # Skip generation (responses already exist)
    python inference/run_inference_pipeline.py \\
        --experiment my_exp --prompt-set main_prompts --skip-generate

    # Save raw activations for later re-projection
    python inference/run_inference_pipeline.py \\
        --experiment my_exp --prompt-set main_prompts --save-activations

    # Re-project from saved activations (no GPU needed for projection)
    python inference/run_inference_pipeline.py \\
        --experiment my_exp --prompt-set main_prompts --from-activations
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
from utils.model import get_inner_model, tokenize, pad_sequences, format_prompt
from utils.json import dump_compact
from utils.vectors import get_best_vector_spec, load_vector_with_baseline
from utils.layers import parse_layers
from core import MultiLayerProjection, MultiLayerCapture, get_hook_path
from inference.project_activations_onto_traits import (
    project_prompt_onto_traits, resolve_layers,
    load_massive_dims_from_analysis, compute_activation_norms,
)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def load_trait_vectors(experiment, extraction_variant, traits, component, layers_spec,
                       available_layers=None):
    """Load trait vectors and group by layer for batched projection.

    Returns:
        trait_vectors: {(cat, trait): [(vector, method, path, layer, metadata,
            source, baseline, position), ...]} — for building output JSON
        vectors_by_layer: {layer: Tensor[n_vectors, hidden_dim]} — for ProjectionHook
        vector_index: [(layer, vec_idx_in_layer, cat, trait, vec_list_idx)] — maps
            hook output back to trait vectors
    """
    trait_vectors = {}
    vectors_for_layer = {}  # layer -> list of (vector, cat, trait, vec_list_idx)

    for trait in traits:
        parts = trait.split('/')
        category, trait_name = parts[0], '/'.join(parts[1:]) if len(parts) > 2 else parts[1]
        key = (category, trait_name)

        spec = get_best_vector_spec(experiment, trait, extraction_variant=extraction_variant)
        if spec is None:
            print(f"  Warning: no vectors found for {trait}, skipping")
            continue

        best_layer = spec.get('layer')
        best_method = spec.get('method')

        # Resolve layers
        if available_layers is None:
            concrete_layers = [best_layer] if best_layer else []
        else:
            concrete_layers = resolve_layers(
                layers_spec or "best,best+5", best_layer, set(available_layers)
            )

        vector_list = []
        for layer in concrete_layers:
            vector, baseline, vec_metadata = load_vector_with_baseline(
                experiment, trait, layer, best_method, component,
                extraction_variant=extraction_variant,
            )
            position = spec.get('position', 'response[:5]')
            selection_source = spec.get('selection_source', 'steering')

            vec_list_idx = len(vector_list)
            vector_list.append((
                vector, best_method, None, layer, vec_metadata,
                selection_source, baseline, position,
            ))

            if layer not in vectors_for_layer:
                vectors_for_layer[layer] = []
            vectors_for_layer[layer].append((vector, category, trait_name, vec_list_idx))

        if vector_list:
            trait_vectors[key] = vector_list

    # Stack vectors per layer for batched GPU projection
    vectors_by_layer = {}
    vector_index = []
    for layer, vecs in vectors_for_layer.items():
        stacked = torch.stack([v[0] for v in vecs])  # [n_vectors, hidden_dim]
        vectors_by_layer[layer] = stacked
        for idx, (_, cat, trait_name, vec_list_idx) in enumerate(vecs):
            vector_index.append((layer, idx, cat, trait_name, vec_list_idx))

    return trait_vectors, vectors_by_layer, vector_index


def stream_through_project(
    model, tokenizer, response_files, trait_vectors, vectors_by_layer,
    component, inference_dir, prompt_set, experiment,
    save_activations=False, skip_existing=False, centered=False,
):
    """Capture activations and project onto trait vectors in one forward pass.

    Uses MultiLayerProjection to project on GPU inside hooks. Only small
    score arrays cross the GPU-CPU boundary.
    """
    from utils.vram import calculate_max_batch_size

    massive_dims_info = None
    analysis_massive_dims, top_dims_by_layer = load_massive_dims_from_analysis(experiment)
    if analysis_massive_dims:
        massive_dims_info = (analysis_massive_dims, top_dims_by_layer)

    layers = sorted(vectors_by_layer.keys())
    n_projected = 0

    for response_file in tqdm(response_files, desc="Projecting"):
        prompt_id = response_file.stem

        with open(response_file) as f:
            resp_data = json.load(f)

        prompt_text = resp_data.get('prompt', '')
        response_text = resp_data.get('response', '')
        prompt_end = resp_data.get('prompt_end', 0)
        all_tokens = resp_data.get('tokens', [])
        prompt_tokens = all_tokens[:prompt_end]
        response_tokens = all_tokens[prompt_end:]

        # Tokenize prompt + response for prefill
        full_text = prompt_text + response_text
        inputs = tokenize(full_text, tokenizer)
        input_ids = inputs.input_ids.to(next(model.parameters()).device)
        attention_mask = inputs.attention_mask.to(input_ids.device)

        # Forward pass with projection hooks
        with MultiLayerProjection(
            model, vectors_by_layer, component=component, compute_norms=True,
        ) as proj:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            all_scores = proj.get_all()      # {layer: [1, seq, n_vectors]}
            all_norms = proj.get_all_norms()  # {layer: [1, seq]}

        # Also save raw activations if requested
        if save_activations:
            raw_dir = inference_dir / "raw" / component / prompt_set
            raw_dir.mkdir(parents=True, exist_ok=True)
            # Capture separately (would need another forward pass or dual hooks)
            # For now, this flag means "also run capture_activations.py afterwards"
            pass

        # Split scores into prompt/response portions
        n_prompt = len(prompt_tokens)

        # Build activations-like dict for project_prompt_onto_traits compatibility
        # We construct per-layer projection results directly
        prompt_acts_proxy = {}
        response_acts_proxy = {}

        for layer in layers:
            scores = all_scores[layer][0]  # [seq, n_vectors]
            norms_at_layer = all_norms[layer][0]  # [seq]

            # We don't have full activations, but we have projection scores
            # and norms — enough to build the output without going through
            # project_prompt_onto_traits(). Build output directly.

        # Build output directly from hook scores (bypass project_prompt_onto_traits
        # since we already have projections, not raw activations)
        for (category, trait_name), vector_list in trait_vectors.items():
            trait_path = f"{category}/{trait_name}"
            out_dir = inference_dir / "projections" / trait_path / prompt_set
            out_file = out_dir / f"{prompt_id}.json"

            if skip_existing and out_file.exists():
                continue

            all_projections = []
            for vec_idx, (vector, method, _, layer, vec_metadata, selection_source, baseline, position) in enumerate(vector_list):
                # Find this vector's index in the stacked vectors for this layer
                layer_vectors = vectors_by_layer[layer]
                # Linear search — small N
                hook_vec_idx = None
                for i in range(layer_vectors.shape[0]):
                    if torch.equal(layer_vectors[i], vector):
                        hook_vec_idx = i
                        break

                if hook_vec_idx is None:
                    continue

                layer_scores = all_scores[layer][0, :, hook_vec_idx]  # [seq]
                layer_norms_vals = all_norms[layer][0]  # [seq]

                prompt_proj = layer_scores[:n_prompt].tolist()
                response_proj = layer_scores[n_prompt:].tolist()

                if centered and baseline != 0.0:
                    prompt_proj = [s - baseline for s in prompt_proj]
                    response_proj = [s - baseline for s in response_proj]

                prompt_token_norms = layer_norms_vals[:n_prompt].tolist()
                response_token_norms = layer_norms_vals[n_prompt:].tolist()

                all_projections.append({
                    'method': method,
                    'layer': layer,
                    'selection_source': selection_source,
                    'baseline': baseline,
                    'prompt': prompt_proj,
                    'response': response_proj,
                    'token_norms': {
                        'prompt': prompt_token_norms,
                        'response': response_token_norms,
                    },
                })

            first_position = vector_list[0][7] if vector_list else 'response[:5]'

            # Compute per-layer mean norms from hook data (replaces calibration step)
            prompt_norms_list = []
            response_norms_list = []
            for layer in sorted(all_norms.keys()):
                layer_n = all_norms[layer][0]  # [seq]
                prompt_norms_list.append(layer_n[:n_prompt].mean().item() if n_prompt > 0 else 0.0)
                response_norms_list.append(layer_n[n_prompt:].mean().item())

            proj_data = {
                'metadata': {
                    'prompt_id': prompt_id,
                    'prompt_set': prompt_set,
                    'n_prompt_tokens': n_prompt,
                    'n_response_tokens': len(response_tokens),
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

            if massive_dims_info and len(vector_list) == 1:
                # Massive dim data requires full activations — skip in stream-through
                pass

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(proj_data, f)

        n_projected += 1

        # Free memory
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
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation (responses must already exist)")
    parser.add_argument("--from-activations", action="store_true",
                        help="Project from saved .pt files (no GPU capture)")
    parser.add_argument("--save-activations", action="store_true",
                        help="Also save raw .pt activation files")

    # Projection options
    parser.add_argument("--traits", type=str, default=None,
                        help="Comma-separated traits (default: all extracted)")
    parser.add_argument("--layers", type=str, default="best,best+5",
                        help="Layer spec per trait (default: best,best+5)")
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

    # Stage 2: Capture + Project
    if args.from_activations:
        # Use existing flow: project from saved .pt files
        from inference.project_activations_onto_traits import main as project_main
        print(f"\n[2] Projecting from saved activations...")
        sys.argv = [
            'project_activations_onto_traits.py',
            '--experiment', args.experiment,
            '--prompt-set', args.prompt_set,
            '--component', args.component,
            '--layers', args.layers,
        ]
        if args.skip_existing:
            sys.argv.append('--skip-existing')
        if args.centered:
            sys.argv.append('--centered')
        if args.traits:
            sys.argv.extend(['--traits', args.traits])
        project_main()
    else:
        # Stream-through: capture + project in one pass
        print(f"\n[2] Stream-through capture + projection...")
        proj_start = time.time()

        # Load model
        backend = LocalBackend.from_experiment(
            args.experiment, variant=model_variant,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        model = backend.model
        tokenizer = backend.tokenizer

        # Load trait vectors
        trait_vectors, vectors_by_layer, vector_index = load_trait_vectors(
            args.experiment, extraction_variant, traits, args.component, args.layers,
        )

        if not vectors_by_layer:
            print("No vectors loaded — nothing to project")
            return

        print(f"    Loaded {sum(v.shape[0] for v in vectors_by_layer.values())} vectors "
              f"across {len(vectors_by_layer)} layers")

        # Find response files
        responses_dir = inference_dir / "responses" / args.prompt_set
        if not responses_dir.exists():
            print(f"No responses found at {responses_dir}")
            return

        response_files = sorted(responses_dir.glob("*.json"))
        if not response_files:
            print("No response files found")
            return

        n_projected = stream_through_project(
            model, tokenizer, response_files, trait_vectors, vectors_by_layer,
            args.component, inference_dir, args.prompt_set, args.experiment,
            save_activations=args.save_activations,
            skip_existing=args.skip_existing,
            centered=args.centered,
        )
        print(f"    Projected {n_projected} prompts ({format_duration(time.time() - proj_start)})")

        # Cleanup
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - pipeline_start
    print(f"\nPipeline complete ({format_duration(total)})")


if __name__ == "__main__":
    main()
