"""
Project activations onto trait vectors to get per-token trait expression scores.

Two modes:
    stream-through  - Prefill forward pass with projection hooks (default, no .pt files)
    from-saved      - Load .pt activation files and project on CPU

The core math (project_prompt_onto_traits) is shared by both modes.

Input: Response JSONs + trait vectors (from extraction pipeline)
Output: experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json

Usage:
    from utils.project_activations import stream_through_project, project_from_saved
    from utils.project_activations import project_prompt_onto_traits  # core math
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
from tqdm import tqdm

from core import projection
from core.types import ResponseRecord, ProjectionEntry, ProjectionRecord
from utils.vector_selection import select_vectors, get_best_vector_spec
from utils.vectors import load_vector_metadata, load_vector_with_baseline, find_vector_method, discover_vectors
from utils.paths import (
    get as get_path, get_vector_path, discover_extracted_traits,
    get_model_variant,
)
from utils.model import tokenize
from utils.layers import resolve_layers
from utils.json_utils import dump_compact


# =============================================================================
# Projection Helpers
# =============================================================================

def project_onto_vector(activations: Dict, vector: torch.Tensor, layer: int,
                        component: str = "residual") -> torch.Tensor:
    """Project activations onto trait vector at a specific layer.

    Returns projection tensor [n_tokens].
    """
    if component == "attn_contribution":
        if 'attn_contribution' in activations[layer] and activations[layer]['attn_contribution'].numel() > 0:
            act = activations[layer]['attn_contribution']
            return projection(act, vector.to(act.dtype), normalize_vector=True)
        else:
            n_tokens = activations[layer]['residual'].shape[0]
            return torch.zeros(n_tokens)
    else:
        act = activations[layer]['residual']
        return projection(act, vector.to(act.dtype), normalize_vector=True)


def compute_token_norms(activations: Dict, layer: int) -> List[float]:
    """Compute activation norm per token at a specific layer."""
    h = activations[layer]['residual']
    return h.norm(dim=-1).tolist()


def _to_list(x):
    """Convert tensor or iterable to plain list."""
    return x.tolist() if hasattr(x, 'tolist') else list(x) if x else []


# =============================================================================
# Core Projection (shared by stream-through and from-saved)
# =============================================================================

def project_prompt_onto_traits(
    prompt_activations: Dict,
    response_activations: Dict,
    trait_vectors: Dict,
    component: str = "residual",
    centered: bool = False,
    n_prompt_tokens: int = 0,
    n_response_tokens: int = 0,
    prompt_set: str = "",
    prompt_id: str = "",
) -> Dict[str, dict]:
    """Project one prompt's activations onto all loaded trait vectors.

    This is the core computation, separated from disk I/O for reuse in
    stream-through mode (inference/run_inference_pipeline.py).

    Returns {trait_path: proj_data_dict}.
    """
    results = {}
    for (category, trait_name), vector_list in trait_vectors.items():
        trait_path = f"{category}/{trait_name}"

        all_projections = []
        for (vector, method, vector_path, layer, vec_metadata, selection_source, baseline, position) in vector_list:
            prompt_proj = project_onto_vector(prompt_activations, vector, layer, component=component) if prompt_activations else []
            response_proj = project_onto_vector(response_activations, vector, layer, component=component)

            if centered and baseline != 0.0:
                if hasattr(prompt_proj, '__sub__'):
                    prompt_proj = prompt_proj - baseline
                response_proj = response_proj - baseline

            prompt_token_norms = compute_token_norms(prompt_activations, layer) if prompt_activations else []
            response_token_norms = compute_token_norms(response_activations, layer)

            all_projections.append(ProjectionEntry(
                method=method, layer=layer, selection_source=selection_source,
                baseline=baseline,
                prompt=_to_list(prompt_proj), response=_to_list(response_proj),
                prompt_token_norms=prompt_token_norms,
                response_token_norms=response_token_norms,
            ))

        first_position = vector_list[0][7]

        record = ProjectionRecord(
            prompt_id=prompt_id, prompt_set=prompt_set,
            n_prompt_tokens=n_prompt_tokens, n_response_tokens=n_response_tokens,
            component=component, position=first_position, centered=centered,
            projections=all_projections,
        )

        results[trait_path] = record.to_dict()

    return results


# =============================================================================
# Stream-through: prefill forward pass with projection hooks
# =============================================================================

def stream_through_project(
    model, tokenizer, response_files, trait_vectors, vectors_by_layer, hook_index,
    component, inference_dir, prompt_set, experiment,
    skip_existing=False, centered=False,
):
    """Project onto trait vectors via prefill forward pass with GPU hooks.

    Reads existing response JSONs, runs a forward pass with MultiLayerProjection
    hooks that compute dot products on GPU. Only small score arrays cross the
    GPU-CPU boundary. No intermediate .pt files.
    """
    from core import MultiLayerProjection

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

        full_text = resp_data.get('prompt', '') + resp_data.get('response', '')
        inputs = tokenize(full_text, tokenizer)
        input_ids = inputs.input_ids.to(next(model.parameters()).device)
        attention_mask = inputs.attention_mask.to(input_ids.device)

        with MultiLayerProjection(
            model, vectors_by_layer, component=component, compute_norms=True,
        ) as proj:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            all_scores = proj.get_all()
            all_norms = proj.get_all_norms()

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
                all_projections.append(ProjectionEntry(
                    method=method, layer=layer, selection_source=selection_source,
                    baseline=baseline,
                    prompt=prompt_proj, response=response_proj,
                    prompt_token_norms=layer_norms[:n_prompt].tolist(),
                    response_token_norms=layer_norms[n_prompt:].tolist(),
                ))

            first_position = vector_list[0][7] if vector_list else 'response[:5]'

            record = ProjectionRecord(
                prompt_id=prompt_id, prompt_set=prompt_set,
                n_prompt_tokens=n_prompt, n_response_tokens=n_response,
                component=component, position=first_position, centered=centered,
                projections=all_projections,
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(record.to_dict(), f)

        n_projected += 1
        del all_scores, all_norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n_projected


# =============================================================================
# From-saved: project from .pt activation files
# =============================================================================

def project_from_saved(inference_dir, prompt_set, model_name, model_variant,
                       extraction_variant, vectors_experiment, steering_variant,
                       *, experiment, traits=None, multi_vector=None, layers=None,
                       layer=None, component='residual', position=None,
                       method=None, centered=False,
                       skip_existing=False):
    """Project saved .pt activations onto trait vectors.

    Loads raw activation files captured by capture_raw_activations(), then
    projects onto trait vectors on CPU. Slower than stream-through but allows
    re-projecting with different vectors/layers without re-running the model.
    """
    raw_dir = inference_dir / "raw" / "residual" / prompt_set

    if not raw_dir.exists():
        print(f"Raw activations not found: {raw_dir}")
        print("Run with --capture first, or use the pipeline for stream-through mode.")
        return

    raw_files = sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0)
    if not raw_files:
        print(f"No raw activation files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw activation files")

    if traits:
        trait_list = []
        for t in traits.split(','):
            parts = t.strip().split('/')
            trait_list.append((parts[0], '/'.join(parts[1:])) if len(parts) > 1 else (parts[0], parts[0]))
    else:
        trait_list = discover_extracted_traits(vectors_experiment)

    if not trait_list:
        print("No traits found")
        return

    print(f"Projecting onto {len(trait_list)} traits")

    use_top_n = multi_vector is not None and multi_vector > 0

    layers_spec = layers
    if layer is not None and layers_spec is None:
        layers_spec = str(layer)
    elif layers_spec is None and not use_top_n:
        layers_spec = 'best+5'

    available_raw_layers = set()
    if layers_spec:
        sample = torch.load(raw_files[0], weights_only=False)
        available_raw_layers = set(sample['response']['activations'].keys())
        del sample

    if use_top_n:
        print(f"Multi-vector mode: projecting onto top {multi_vector} vectors per trait")
    else:
        print(f"Layers spec: {layers_spec}")

    # Load trait vectors
    trait_vectors = {}
    for category, trait_name in trait_list:
        trait_path = f"{category}/{trait_name}"

        if use_top_n:
            top_vectors = select_vectors(
                vectors_experiment, trait_path, extraction_variant=extraction_variant,
                component=component, position=position, n=multi_vector
            )
            if not top_vectors:
                print(f"  Skip {trait_path}: no vectors found")
                continue

            loaded_vectors = []
            for vec_info in top_vectors:
                vec_layer = vec_info.layer
                vec_method = vec_info.method
                selection_source = vec_info.source
                vector_path = get_vector_path(
                    vectors_experiment, trait_path, vec_method, vec_layer, extraction_variant, component, position
                )

                if not vector_path.exists():
                    continue

                try:
                    vector, baseline, per_vec_metadata = load_vector_with_baseline(
                        vectors_experiment, trait_path, vec_method, vec_layer, extraction_variant, component, position
                    )
                    vector = vector.to(torch.float16)
                    vec_metadata = load_vector_metadata(
                        vectors_experiment, trait_path, vec_method, extraction_variant, component, position
                    )
                    loaded_vectors.append((vector, vec_method, vector_path, vec_layer, vec_metadata, selection_source, baseline, position))
                except FileNotFoundError:
                    continue

            if loaded_vectors:
                trait_vectors[(category, trait_name)] = loaded_vectors
                methods_str = ', '.join([f"L{v[3]} {v[1]}" for v in loaded_vectors])
                print(f"  {trait_path}: [{methods_str}]")
        else:
            best_layer = None
            best_method = None
            best_position = position
            best_source = None
            best_score = None

            try:
                spec, spec_meta = get_best_vector_spec(
                    vectors_experiment, trait_path, extraction_variant=extraction_variant,
                    steering_variant=steering_variant,
                    component=component, position=position
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

            resolved_layers = resolve_layers(layers_spec, best_layer, available_raw_layers)

            if not resolved_layers:
                print(f"  Skip {trait_path}: no valid layers resolved")
                continue

            loaded_vectors = []
            for candidate_layer in resolved_layers:
                vec_method = method
                if vec_method is None:
                    if candidate_layer == best_layer and best_method:
                        vec_method = best_method
                    else:
                        vec_method = find_vector_method(
                            vectors_experiment, trait_path, candidate_layer, extraction_variant,
                            component, best_position or 'response[:]'
                        )

                if not vec_method:
                    print(f"    Skip L{candidate_layer}: no vector found")
                    continue

                vec_position = best_position
                if vec_position is None:
                    candidates = discover_vectors(
                        vectors_experiment, trait_path, extraction_variant,
                        component=component, layer=candidate_layer
                    )
                    if candidates:
                        vec_position = candidates[0]['position']
                    else:
                        print(f"    Skip L{candidate_layer}: no vectors found")
                        continue

                try:
                    vector, baseline, per_vec_meta = load_vector_with_baseline(
                        vectors_experiment, trait_path, vec_method, candidate_layer, extraction_variant,
                        component, vec_position
                    )
                    vector = vector.to(torch.float16)
                except FileNotFoundError:
                    print(f"    Skip L{candidate_layer}: vector file not found")
                    continue

                vec_metadata = load_vector_metadata(
                    vectors_experiment, trait_path, vec_method, extraction_variant,
                    component, vec_position
                )
                source = best_source if candidate_layer == best_layer else 'layers_arg'
                loaded_vectors.append((vector, vec_method, None, candidate_layer, vec_metadata, source, baseline, vec_position))

            loaded_layers = {v[3] for v in loaded_vectors}
            skipped_layers = sorted(set(resolved_layers) - loaded_layers)
            if skipped_layers:
                print(f"\n  WARNING: {trait_path} — skipped {len(skipped_layers)}/{len(resolved_layers)} layers: {skipped_layers}")
                print(f"    Vectors missing. Extract them first: python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {trait_path} --only-stage 3,4 --layers {','.join(str(l) for l in skipped_layers)}\n")

            if loaded_vectors:
                trait_vectors[(category, trait_name)] = loaded_vectors
                desc = ', '.join([f"L{v[3]} {v[1]}" for v in loaded_vectors])
                source_info = f"(from {best_source}: {best_score:.2f})" if best_source and best_score else ''
                print(f"  {trait_path}: [{desc}] {source_info}")

    print(f"Loaded vectors for {len(trait_vectors)} traits")

    # Process each raw file
    for raw_file in tqdm(raw_files, desc="Projecting"):
        prompt_id = raw_file.stem

        data = torch.load(raw_file, weights_only=False)
        prompt_acts = data['prompt'].get('activations', {})
        response_acts = data['response']['activations']
        n_layers = len(response_acts) or len(prompt_acts)

        # Ensure response JSON exists
        responses_dir = inference_dir / "responses" / prompt_set
        response_file = responses_dir / f"{prompt_id}.json"
        if not response_file.exists():
            responses_dir.mkdir(parents=True, exist_ok=True)
            prompt_tokens = data['prompt']['tokens']
            response_tokens = data['response']['tokens']
            prompt_token_ids = data['prompt'].get('token_ids', [])
            response_token_ids = data['response'].get('token_ids', [])
            record = ResponseRecord(
                prompt=data['prompt']['text'],
                response=data['response']['text'],
                tokens=prompt_tokens + response_tokens,
                token_ids=prompt_token_ids + response_token_ids,
                prompt_end=len(prompt_tokens),
                inference_model=model_name,
                capture_date=datetime.now().isoformat(),
            )
            with open(response_file, 'w') as f:
                dump_compact(record.to_dict(), f)

        results = project_prompt_onto_traits(
            prompt_activations=prompt_acts,
            response_activations=response_acts,
            trait_vectors=trait_vectors,
            component=component,
            centered=centered,
            n_prompt_tokens=len(data['prompt']['tokens']),
            n_response_tokens=len(data['response']['tokens']),
            prompt_set=prompt_set,
            prompt_id=prompt_id,
        )

        for trait_path, proj_data in results.items():
            out_dir = inference_dir / "projections" / trait_path / prompt_set
            out_file = out_dir / f"{prompt_id}.json"
            if skip_existing and out_file.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(proj_data, f)

        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nProjections saved to: {inference_dir}/projections/{{trait}}/{prompt_set}/")
