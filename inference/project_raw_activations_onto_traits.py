#!/usr/bin/env python3
"""
Post-hoc projection: compute trait projections from saved raw activations.

This allows re-projecting onto different traits or with different vectors
without re-running model inference.

Layer selection (default: best+5 per trait):
- best+5 = best steering layer + 5 (robustness offset)
- Use --layers best for steering-optimal layer
- Use --layer N or --layers spec to override

Storage:
  Responses (shared)   : experiments/{exp}/inference/{model_variant}/responses/{prompt_set}/{id}.json
  Projections          : experiments/{exp}/inference/{model_variant}/projections/{trait}/{prompt_set}/{id}.json

Format (always multi-vector):
  {
    "metadata": {
      "multi_vector": true,
      "n_vectors": 1,
      ...
    },
    "projections": [
      {
        "method": "probe",
        "layer": 16,
        "selection_source": "steering",
        "baseline": 0.0,
        "prompt": [0.5, -0.3, ...],
        "response": [2.1, 1.8, ...],
        "token_norms": {"prompt": [...], "response": [...]}
      }
    ],
    "activation_norms": {"prompt": [...], "response": [...]}
  }

Usage:
    # Project onto all traits (auto-selects best layer per trait)
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # Multi-layer projection (robustness check)
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --layers best,best+5

    # Override with fixed layer for all traits
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --layer 16

    # Project onto specific traits
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --traits behavioral/refusal,cognitive/retrieval

    # Project attn_contribution activations onto attn_contribution vectors
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set harmful \\
        --component attn_contribution \\
        --layer 8
"""

import gc
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from core import projection, VectorSpec
from utils.vectors import (
    load_vector_metadata, load_vector_with_baseline, find_vector_method,
    get_best_vector, get_top_N_vectors, get_best_vector_spec, load_vector_from_spec,
    _discover_vectors
)
from utils.paths import get as get_path, get_vector_path, discover_extracted_traits, get_model_variant, load_experiment_config
from utils.model_registry import get_model_slug
from utils.json import dump_compact

LOGIT_LENS_LAYERS = [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25]


# ============================================================================
# Layer Resolution
# ============================================================================

def resolve_layers(layers_spec: str, best_layer: Optional[int], available_layers: set) -> List[int]:
    """Resolve layer specs like 'best,best+5' into concrete layer numbers.

    Args:
        layers_spec: Comma-separated layer specs (e.g., "best,best+5", "20,25")
        best_layer: Best steering layer for current trait (can be None)
        available_layers: Set of available layer numbers in raw activations

    Returns:
        List of resolved, validated layer numbers
    """
    result = []
    for spec in layers_spec.split(','):
        spec = spec.strip()
        if 'best' in spec:
            if best_layer is None:
                print(f"    Warning: cannot resolve '{spec}' (no steering data), skipping")
                continue
            if spec == 'best':
                layer = best_layer
            else:
                m = re.match(r'best\s*([+-])\s*(\d+)', spec)
                if not m:
                    raise ValueError(f"Invalid layer spec: '{spec}'. Use: best, best+N, best-N, or integer")
                op, val = m.group(1), int(m.group(2))
                layer = best_layer + val if op == '+' else best_layer - val
        else:
            layer = int(spec)

        if layer in available_layers:
            if layer not in result:  # Deduplicate
                result.append(layer)
        elif available_layers:
            # Snap to closest available layer
            closest = min(available_layers, key=lambda l: abs(l - layer))
            print(f"    Layer {layer} not captured, snapping to nearest: {closest}")
            if closest not in result:
                result.append(closest)
        else:
            print(f"    Warning: layer {layer} not in raw activations, skipping")

    return result


# ============================================================================
# Massive Dims
# ============================================================================

def load_massive_dims_from_analysis(experiment: str) -> Tuple[List[int], Dict]:
    """
    Load massive dims from calibration.json.

    Returns:
        (dims_list, top_dims_by_layer) - union of all dims, and per-layer top dims
    """
    inference_dir = Path(get_path('inference.base', experiment=experiment))
    calibration_path = inference_dir / 'massive_activations' / 'calibration.json'

    if not calibration_path.exists():
        return [], {}

    with open(calibration_path) as f:
        data = json.load(f)

    top_dims_by_layer = data.get('aggregate', {}).get('top_dims_by_layer', {})
    if not top_dims_by_layer:
        return [], {}

    # Union of all dims across all layers
    all_dims = set()
    for layer_dims in top_dims_by_layer.values():
        all_dims.update(layer_dims)

    return sorted(all_dims), top_dims_by_layer


def extract_massive_dim_values(
    activations: Dict,
    dims: List[int],
    layer: int,
    component: str = "residual"
) -> Dict[int, List[float]]:
    """Extract activation values at massive dims for each token."""
    if not dims or layer not in activations:
        return {}

    act = activations[layer].get(component)
    if act is None:
        return {}

    result = {}
    for dim in dims:
        if dim < act.shape[-1]:
            result[dim] = act[:, dim].tolist()
    return result


# ============================================================================
# Projection
# ============================================================================

def project_onto_vector(activations: Dict, vector: torch.Tensor, layer: int,
                        component: str = "residual") -> torch.Tensor:
    """Project activations onto trait vector at a specific layer.

    Args:
        activations: Dict of layer -> component -> tensor
        vector: Trait vector
        layer: Layer number to project at
        component: 'residual' or 'attn_contribution'

    Returns:
        Projection tensor [n_tokens] - one value per token
    """
    if component == "attn_contribution":
        # Project attn_contribution activations at specified layer
        if 'attn_contribution' in activations[layer] and activations[layer]['attn_contribution'].numel() > 0:
            act = activations[layer]['attn_contribution']
            return projection(act, vector.to(act.dtype), normalize_vector=True)
        else:
            # Return zeros if attn_contribution not available
            n_tokens = activations[layer]['residual'].shape[0]
            return torch.zeros(n_tokens)
    else:
        # Project residual at specified layer
        act = activations[layer]['residual']
        return projection(act, vector.to(act.dtype), normalize_vector=True)


def compute_activation_norms(activations: Dict, n_layers: int) -> List[float]:
    """Compute activation norms per layer (averaged across tokens and components).

    Returns [n_layers] array of ||h|| values showing activation magnitude by layer.
    """
    components = ['attn_contribution', 'residual']
    norms = []

    for layer in sorted(activations.keys()):
        layer_norms = []
        for component in components:
            if component in activations[layer] and activations[layer][component].numel() > 0:
                # Compute L2 norm per token, then average across tokens
                h = activations[layer][component]  # [n_tokens, hidden_dim]
                token_norms = h.norm(dim=-1)  # [n_tokens]
                layer_norms.append(token_norms.mean().item())
        # Average across available components
        if layer_norms:
            norms.append(sum(layer_norms) / len(layer_norms))
        else:
            norms.append(0.0)

    return norms


def compute_token_norms(activations: Dict, layer: int) -> List[float]:
    """Compute activation norm per token at a specific layer.

    Returns [n_tokens] array of ||h|| values for comparing token magnitudes.
    Uses residual component.
    """
    h = activations[layer]['residual']  # [n_tokens, hidden_dim]
    token_norms = h.norm(dim=-1)  # [n_tokens]
    return token_norms.tolist()


# ============================================================================
# Logit Lens (requires model)
# ============================================================================

def compute_logit_lens_from_raw(activations: Dict, model, tokenizer, n_layers: int) -> Dict:
    """Compute logit lens from saved activations (requires model for unembed)."""
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.detach()
    else:
        unembed = model.model.embed_tokens.weight.detach()

    result = {}
    for layer in LOGIT_LENS_LAYERS:
        if layer >= n_layers:
            continue

        residual = activations[layer]['residual']
        if len(residual.shape) == 1:
            residual = residual.unsqueeze(0)

        logits = residual.to(unembed.device).to(unembed.dtype) @ unembed.T
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(3, dim=-1)

        top_tokens = []
        for token_idx in range(top_ids.shape[0]):
            tokens = [tokenizer.decode([tid.item()]) for tid in top_ids[token_idx]]
            top_tokens.append(tokens)

        result[f'layer_{layer}'] = {
            'tokens': top_tokens,
            'probs': top_probs.cpu().tolist()
        }

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-hoc projection from raw activations")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", help="Prompt set name (or use --all-prompt-sets)")
    parser.add_argument("--all-prompt-sets", action="store_true", help="Process all prompt sets")
    parser.add_argument("--traits", help="Comma-separated traits (category/name format)")
    parser.add_argument("--layer", type=int,
                       help="Override layer for all traits (default: auto-select best per trait)")
    parser.add_argument("--layers",
                       help="Layer spec: 'best', 'best,best+5', '20,25' (default: best+5)")
    parser.add_argument("--method", help="Vector method (default: auto-detect or use best from evaluation)")
    parser.add_argument("--component", choices=["residual", "attn_contribution"], default="residual",
                       help="Activation component to project (default: residual)")
    parser.add_argument("--position", default=None,
                       help="Token position for vectors (default: auto-detect from available vectors)")
    parser.add_argument("--logit-lens", action="store_true", help="Compute logit lens")
    parser.add_argument("--centered", action="store_true",
                       help="Subtract training baseline from projections (centers around 0)")
    parser.add_argument("--multi-vector", type=int, metavar="N",
                       help="Project onto top N vectors per trait (enables multi-vector mode)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-calibration", action="store_true",
                       help="Skip massive dims calibration check (disables interactive cleaning in viz)")
    parser.add_argument("--model-variant", default=None,
                       help="Model variant for projections (default: from experiment defaults.application)")
    parser.add_argument("--extraction-variant", default=None,
                       help="Model variant for vectors (default: from experiment defaults.extraction)")

    args = parser.parse_args()

    if not args.prompt_set and not args.all_prompt_sets:
        parser.error("Either --prompt-set or --all-prompt-sets is required")

    if args.layer is not None and args.layers is not None:
        parser.error("--layer and --layers are mutually exclusive")

    # Resolve model variants: application for inference, extraction for vectors
    app_variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = app_variant['name']  # For inference paths
    model_name = app_variant['model']

    ext_variant = get_model_variant(args.experiment, args.extraction_variant, mode="extraction")
    extraction_variant = ext_variant['name']  # For vector loading

    # Set inference_dir using application variant
    inference_dir = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant)

    # Check for calibration (required for interactive viz cleaning)
    # Calibration is always in default inference dir (shared across models)
    default_inference_dir = get_path('inference.base', experiment=args.experiment)
    calibration_path = default_inference_dir / 'massive_activations' / 'calibration.json'
    if not calibration_path.exists() and not args.no_calibration:
        print(f"Error: Massive dims calibration not found at {calibration_path}")
        print(f"\nRun calibration first:")
        print(f"  python analysis/massive_activations.py --experiment {args.experiment}")
        print(f"\nOr skip with --no-calibration (disables interactive cleaning in viz)")
        return

    raw_residual_dir = inference_dir / "raw" / "residual"

    # Discover prompt sets if --all-prompt-sets
    if args.all_prompt_sets:
        if not raw_residual_dir.exists():
            print(f"Raw residual directory not found: {raw_residual_dir}")
            return
        prompt_sets = [d.name for d in raw_residual_dir.iterdir() if d.is_dir()]
        print(f"Found {len(prompt_sets)} prompt sets: {', '.join(prompt_sets)}")
    else:
        prompt_sets = [args.prompt_set]

    for prompt_set in prompt_sets:
        print(f"\n{'='*60}\nProcessing prompt set: {prompt_set}\n{'='*60}")
        process_prompt_set(args, inference_dir, prompt_set, model_name, model_variant, extraction_variant)


def process_prompt_set(args, inference_dir, prompt_set, model_name, model_variant, extraction_variant):
    """Process a single prompt set."""
    raw_dir = inference_dir / "raw" / "residual" / prompt_set

    if not raw_dir.exists():
        print(f"Raw activations not found: {raw_dir}")
        print("Run 'python inference/capture_raw_activations.py' first to capture raw activations.")
        return

    # Find raw activation files (new format: {id}.pt)
    raw_files = sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0)
    if not raw_files:
        print(f"No raw activation files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw activation files")

    # Load massive dims from calibration (for interactive visualization)
    analysis_massive_dims, top_dims_by_layer = load_massive_dims_from_analysis(args.experiment)
    if analysis_massive_dims:
        print(f"Loaded {len(analysis_massive_dims)} massive dims from calibration for visualization")

    # Get traits to project onto
    if args.traits:
        trait_list = [tuple(t.split('/')) for t in args.traits.split(',')]
    else:
        trait_list = discover_extracted_traits(args.experiment)

    if not trait_list:
        print("No traits found")
        return

    print(f"Projecting onto {len(trait_list)} traits")

    # Determine layer mode
    use_top_n = args.multi_vector is not None and args.multi_vector > 0

    # Unify --layer N into layers_spec; default to best+5
    layers_spec = args.layers
    if args.layer is not None and layers_spec is None:
        layers_spec = str(args.layer)
    elif layers_spec is None and not use_top_n:
        layers_spec = 'best+5'

    # Discover available raw layers (for --layers validation)
    available_raw_layers = set()
    if layers_spec:
        sample = torch.load(raw_files[0], weights_only=False)
        available_raw_layers = set(sample['response']['activations'].keys())
        del sample

    if use_top_n:
        print(f"Multi-vector mode: projecting onto top {args.multi_vector} vectors per trait")
    else:
        print(f"Layers spec: {layers_spec}")

    # ========================================================================
    # Load trait vectors — always stored as list of tuples
    # [(vector, method, path, layer, meta, source, baseline, position), ...]
    # ========================================================================

    trait_vectors = {}
    for category, trait_name in trait_list:
        trait_path = f"{category}/{trait_name}"

        if use_top_n:
            # Top N vectors from steering (existing multi-vector mode)
            top_vectors = get_top_N_vectors(
                args.experiment, trait_path, extraction_variant=extraction_variant,
                component=args.component, position=args.position, N=args.multi_vector
            )
            if not top_vectors:
                print(f"  Skip {trait_path}: no vectors found")
                continue

            loaded_vectors = []
            for v in top_vectors:
                layer = v['layer']
                method = v['method']
                selection_source = v['source']
                vector_path = get_vector_path(
                    args.experiment, trait_path, method, layer, extraction_variant, args.component, args.position
                )

                if not vector_path.exists():
                    continue

                try:
                    vector, baseline, per_vec_metadata = load_vector_with_baseline(
                        args.experiment, trait_path, method, layer, extraction_variant, args.component, args.position
                    )
                    vector = vector.to(torch.float16)
                    vec_metadata = load_vector_metadata(
                        args.experiment, trait_path, method, extraction_variant, args.component, args.position
                    )
                    loaded_vectors.append((vector, method, vector_path, layer, vec_metadata, selection_source, baseline, args.position))
                except FileNotFoundError:
                    continue

            if loaded_vectors:
                trait_vectors[(category, trait_name)] = loaded_vectors
                methods_str = ', '.join([f"L{v[3]} {v[1]}" for v in loaded_vectors])
                print(f"  {trait_path}: [{methods_str}]")
        else:
            # Resolve layers for this trait
            # First, get best vector info for 'best' resolution and default layer
            best_layer = None
            best_method = None
            best_position = args.position
            best_source = None
            best_score = None

            try:
                spec, spec_meta = get_best_vector_spec(
                    args.experiment, trait_path, extraction_variant=extraction_variant,
                    component=args.component, position=args.position
                )
                best_layer = spec.layer
                best_method = spec.method
                best_source = spec_meta['source']
                best_score = spec_meta['score']
                if best_position is None:
                    best_position = spec.position
            except FileNotFoundError as e:
                if layers_spec and 'best' in layers_spec:
                    print(f"  Skip {trait_path}: {e}")
                    continue
                # May still work with explicit numeric layers

            # Determine which layers to project at
            layers = resolve_layers(layers_spec, best_layer, available_raw_layers)

            if not layers:
                print(f"  Skip {trait_path}: no valid layers resolved")
                continue

            # Load vector at each layer
            loaded_vectors = []
            for layer in layers:
                # Determine method for this layer
                method = args.method
                if method is None:
                    if layer == best_layer and best_method:
                        method = best_method
                    else:
                        method = find_vector_method(
                            args.experiment, trait_path, layer, extraction_variant,
                            args.component, best_position or 'response[:]'
                        )

                if not method:
                    print(f"    Skip L{layer}: no vector found")
                    continue

                # Determine position for this layer
                position = best_position
                if position is None:
                    candidates = _discover_vectors(
                        args.experiment, trait_path, extraction_variant,
                        component=args.component, layer=layer
                    )
                    if candidates:
                        position = candidates[0]['position']
                    else:
                        print(f"    Skip L{layer}: no vectors found")
                        continue

                try:
                    vector, baseline, per_vec_meta = load_vector_with_baseline(
                        args.experiment, trait_path, method, layer, extraction_variant,
                        args.component, position
                    )
                    vector = vector.to(torch.float16)
                except FileNotFoundError:
                    print(f"    Skip L{layer}: vector file not found")
                    continue

                vec_metadata = load_vector_metadata(
                    args.experiment, trait_path, method, extraction_variant,
                    args.component, position
                )
                source = best_source if layer == best_layer else 'layers_arg'
                loaded_vectors.append((vector, method, None, layer, vec_metadata, source, baseline, position))

            if loaded_vectors:
                trait_vectors[(category, trait_name)] = loaded_vectors
                desc = ', '.join([f"L{v[3]} {v[1]}" for v in loaded_vectors])
                source_info = f"(from {best_source}: {best_score:.2f})" if best_source and best_score else ''
                print(f"  {trait_path}: [{desc}] {source_info}")

    print(f"Loaded vectors for {len(trait_vectors)} traits")

    # Load model only if logit lens requested
    model = None
    tokenizer = None
    if args.logit_lens:
        print(f"\nLoading model for logit lens: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ========================================================================
    # Process each raw file — unified multi-vector output
    # ========================================================================

    for raw_file in tqdm(raw_files, desc="Projecting"):
        # Extract prompt ID from filename (new format: {id}.pt)
        prompt_id = raw_file.stem  # e.g., "1", "2", etc.

        # Load raw activations
        data = torch.load(raw_file, weights_only=False)
        # Handle --response-only .pt files (no prompt activations)
        prompt_acts = data['prompt'].get('activations', {})
        response_acts = data['response']['activations']
        n_layers = len(response_acts) or len(prompt_acts)

        # Ensure response JSON exists (trait-independent, stored once)
        # For rollout data, convert_rollout.py creates this file first with
        # turn_boundaries and tags — we preserve it by not overwriting.
        responses_dir = inference_dir / "responses" / prompt_set
        response_file = responses_dir / f"{prompt_id}.json"
        if not response_file.exists():
            responses_dir.mkdir(parents=True, exist_ok=True)
            # Flatten to unified schema (no metadata wrapper)
            prompt_tokens = data['prompt']['tokens']
            response_tokens = data['response']['tokens']
            prompt_token_ids = data['prompt'].get('token_ids', [])
            response_token_ids = data['response'].get('token_ids', [])
            response_data = {
                'prompt': data['prompt']['text'],
                'response': data['response']['text'],
                'system_prompt': None,
                'tokens': prompt_tokens + response_tokens,
                'token_ids': prompt_token_ids + response_token_ids,
                'prompt_end': len(prompt_tokens),
                'inference_model': model_name,
                'capture_date': datetime.now().isoformat(),
                'tags': []
            }
            with open(response_file, 'w') as f:
                dump_compact(response_data, f)

        # Compute activation norms (trait-independent, computed once per prompt)
        prompt_norms = compute_activation_norms(prompt_acts, n_layers) if prompt_acts else []
        response_norms = compute_activation_norms(response_acts, n_layers)

        # Compute logit lens if requested
        logit_lens_data = None
        if args.logit_lens and model is not None:
            logit_lens_data = {
                'prompt': compute_logit_lens_from_raw(prompt_acts, model, tokenizer, n_layers) if prompt_acts else {},
                'response': compute_logit_lens_from_raw(response_acts, model, tokenizer, n_layers)
            }

        # Project onto each trait
        for (category, trait_name), vector_list in trait_vectors.items():
            # Path: projections/{trait}/{prompt_set}/{id}.json
            trait_path = f"{category}/{trait_name}"
            out_dir = inference_dir / "projections" / trait_path / prompt_set
            out_file = out_dir / f"{prompt_id}.json"

            if args.skip_existing and out_file.exists():
                continue

            # Build projections list (one entry per vector/layer)
            all_projections = []
            for (vector, method, vector_path, layer, vec_metadata, selection_source, baseline, position) in vector_list:
                prompt_proj = project_onto_vector(prompt_acts, vector, layer, component=args.component) if prompt_acts else []
                response_proj = project_onto_vector(response_acts, vector, layer, component=args.component)

                if args.centered and baseline != 0.0:
                    if prompt_proj:
                        prompt_proj = prompt_proj - baseline
                    response_proj = response_proj - baseline

                prompt_token_norms = compute_token_norms(prompt_acts, layer) if prompt_acts else []
                response_token_norms = compute_token_norms(response_acts, layer)

                all_projections.append({
                    'method': method,
                    'layer': layer,
                    'selection_source': selection_source,
                    'baseline': baseline,
                    'prompt': prompt_proj.tolist() if hasattr(prompt_proj, 'tolist') else prompt_proj,
                    'response': response_proj.tolist(),
                    'token_norms': {
                        'prompt': prompt_token_norms,
                        'response': response_token_norms,
                    }
                })

            # Determine position for metadata (from first vector)
            first_position = vector_list[0][7]  # position field

            proj_data = {
                'metadata': {
                    'prompt_id': prompt_id,
                    'prompt_set': prompt_set,
                    'n_prompt_tokens': len(data['prompt']['tokens']),
                    'n_response_tokens': len(data['response']['tokens']),
                    'multi_vector': True,
                    'n_vectors': len(all_projections),
                    'component': args.component,
                    'position': first_position,
                    'centered': args.centered,
                    'projection_date': datetime.now().isoformat()
                },
                'projections': all_projections,
                'activation_norms': {
                    'prompt': prompt_norms,
                    'response': response_norms
                },
            }

            # Add massive dim data for single-vector case (backward compat for viz)
            if len(vector_list) == 1 and analysis_massive_dims:
                vector = vector_list[0][0]
                layer = vector_list[0][3]
                vec_norm = vector.norm().item()
                vec_components = {dim: vector[dim].item() for dim in analysis_massive_dims if dim < len(vector)}

                prompt_dim_vals = extract_massive_dim_values(
                    prompt_acts, analysis_massive_dims, layer, args.component) if prompt_acts else {}
                response_dim_vals = extract_massive_dim_values(
                    response_acts, analysis_massive_dims, layer, args.component)

                proj_data['massive_dim_data'] = {
                    'dims': analysis_massive_dims,
                    'top_dims_by_layer': top_dims_by_layer,
                    'vec_norm': vec_norm,
                    'vec_components': vec_components,
                    'activation_values': {
                        'prompt': prompt_dim_vals,
                        'response': response_dim_vals
                    }
                }

            if logit_lens_data:
                proj_data['logit_lens'] = logit_lens_data

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(proj_data, f)

        # Free memory after each prompt
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nProjections saved to: {inference_dir}/projections/{{trait}}/{prompt_set}/")


if __name__ == "__main__":
    main()
