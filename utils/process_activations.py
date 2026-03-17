#!/usr/bin/env python3
"""
Capture activations and/or project onto trait vectors.

Two standalone modes:
    capture  - Save raw activations as .pt files (for re-projection later)
    project  - Project saved .pt activations onto trait vectors (default)

The inference pipeline (run_inference_pipeline.py) uses stream-through mode
instead, which projects during capture with no intermediate .pt files.

Input:  Response JSONs from generate_responses.py
Output:
    capture: experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{id}.pt
    project: experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json

Usage:
    # Project from saved .pt files (default)
    python inference/process_activations.py --experiment my_exp --prompt-set main

    # Capture raw activations (save .pt)
    python inference/process_activations.py --experiment my_exp --prompt-set main --capture

    # Multi-layer projection
    python inference/process_activations.py --experiment my_exp --prompt-set main --layers best,best+5
"""

import gc
import re
import sys
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from core import projection, VectorSpec, MultiLayerCapture
from utils.vector_selection import get_best_vector, get_top_N_vectors, get_best_vector_spec, _discover_vectors
from utils.vectors import load_vector_metadata, load_vector_with_baseline, find_vector_method, load_vector_from_spec
from utils.paths import (
    get as get_path, get_vector_path, discover_extracted_traits,
    get_model_variant, load_experiment_config,
)
from utils.model import load_model_with_lora, get_inner_model, tokenize, pad_sequences
from utils.backends import add_backend_args
from utils.distributed import is_tp_mode, is_rank_zero
from utils.vram import calculate_max_batch_size
from utils.layers import parse_layers
from utils.model_registry import get_model_slug
from utils.json import dump_compact

LOGIT_LENS_LAYERS = [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25]


# ============================================================================
# Layer Resolution
# ============================================================================

def resolve_layers(layers_spec: str, best_layer: Optional[int], available_layers: set) -> List[int]:
    """Resolve layer specs like 'best,best+5' into concrete layer numbers."""
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
            if layer not in result:
                result.append(layer)
        elif available_layers:
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
    """Load massive dims from calibration.json.

    Returns (dims_list, top_dims_by_layer).
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

    all_dims = set()
    for layer_dims in top_dims_by_layer.values():
        all_dims.update(layer_dims)

    return sorted(all_dims), top_dims_by_layer


def extract_massive_dim_values(
    activations: Dict, dims: List[int], layer: int, component: str = "residual"
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
# Projection Helpers
# ============================================================================

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


def compute_activation_norms(activations: Dict, n_layers: int) -> List[float]:
    """Compute activation norms per layer (averaged across tokens and components)."""
    components = ['attn_contribution', 'residual']
    norms = []

    for layer in sorted(activations.keys()):
        layer_norms = []
        for component in components:
            if component in activations[layer] and activations[layer][component].numel() > 0:
                h = activations[layer][component]
                token_norms = h.norm(dim=-1)
                layer_norms.append(token_norms.mean().item())
        if layer_norms:
            norms.append(sum(layer_norms) / len(layer_norms))
        else:
            norms.append(0.0)

    return norms


def compute_token_norms(activations: Dict, layer: int) -> List[float]:
    """Compute activation norm per token at a specific layer."""
    h = activations[layer]['residual']
    token_norms = h.norm(dim=-1)
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
# Core Projection Function (used by stream-through and from-activations)
# ============================================================================

def project_prompt_onto_traits(
    prompt_activations: Dict,
    response_activations: Dict,
    trait_vectors: Dict,
    component: str = "residual",
    centered: bool = False,
    massive_dims_info: Optional[Tuple] = None,
    logit_lens_data: Optional[Dict] = None,
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
    n_layers = len(response_activations) or len(prompt_activations)

    prompt_norms = compute_activation_norms(prompt_activations, n_layers) if prompt_activations else []
    response_norms = compute_activation_norms(response_activations, n_layers)

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

        first_position = vector_list[0][7]

        proj_data = {
            'metadata': {
                'prompt_id': prompt_id,
                'prompt_set': prompt_set,
                'n_prompt_tokens': n_prompt_tokens,
                'n_response_tokens': n_response_tokens,
                'multi_vector': True,
                'n_vectors': len(all_projections),
                'component': component,
                'position': first_position,
                'centered': centered,
                'projection_date': datetime.now().isoformat(),
            },
            'projections': all_projections,
            'activation_norms': {
                'prompt': prompt_norms,
                'response': response_norms,
            },
        }

        # Massive dim data for single-vector case (backward compat for viz)
        if len(vector_list) == 1 and massive_dims_info:
            analysis_massive_dims, top_dims_by_layer = massive_dims_info
            if analysis_massive_dims:
                vector = vector_list[0][0]
                layer = vector_list[0][3]
                vec_norm = vector.norm().item()
                vec_components = {dim: vector[dim].item() for dim in analysis_massive_dims if dim < len(vector)}

                prompt_dim_vals = extract_massive_dim_values(
                    prompt_activations, analysis_massive_dims, layer, component) if prompt_activations else {}
                response_dim_vals = extract_massive_dim_values(
                    response_activations, analysis_massive_dims, layer, component)

                proj_data['massive_dim_data'] = {
                    'dims': analysis_massive_dims,
                    'top_dims_by_layer': top_dims_by_layer,
                    'vec_norm': vec_norm,
                    'vec_components': vec_components,
                    'activation_values': {
                        'prompt': prompt_dim_vals,
                        'response': response_dim_vals,
                    }
                }

        if logit_lens_data:
            proj_data['logit_lens'] = logit_lens_data

        results[trait_path] = proj_data

    return results


# ============================================================================
# Stream-through: capture + project in one forward pass (used by inference pipeline)
# ============================================================================

def stream_through_project(
    model, tokenizer, response_files, trait_vectors, vectors_by_layer, hook_index,
    component, inference_dir, prompt_set, experiment,
    skip_existing=False, centered=False,
):
    """Capture activations and project onto trait vectors in one forward pass.

    Uses MultiLayerProjection to project on GPU inside hooks. Only small
    score arrays cross the GPU-CPU boundary.
    """
    from core import MultiLayerProjection
    from utils.json import dump_compact

    massive_dims_info = None
    analysis_massive_dims, top_dims_by_layer = load_massive_dims_from_analysis(experiment)
    if analysis_massive_dims:
        massive_dims_info = (analysis_massive_dims, top_dims_by_layer)

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

        prompt_norms_list = []
        response_norms_list = []
        for layer in sorted(all_norms.keys()):
            layer_n = all_norms[layer][0]
            prompt_norms_list.append(layer_n[:n_prompt].mean().item() if n_prompt > 0 else 0.0)
            response_norms_list.append(layer_n[n_prompt:].mean().item())

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


# ============================================================================
# Capture: Save raw activations as .pt files
# ============================================================================

def _save_pt_data(data: Dict, prompt_id, raw_dir: Path, response_only: bool = False):
    """Save captured activation data as .pt file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_data = data
    if response_only:
        save_data = {
            'prompt': {k: v for k, v in data['prompt'].items() if k != 'activations'},
            'response': data['response'],
        }
        save_data['prompt']['activations'] = {}
    torch.save(save_data, raw_dir / f"{prompt_id}.pt")


def capture_raw_activations(
    experiment: str,
    prompt_set: str,
    model_variant: str = None,
    components: str = "residual",
    layers: str = None,
    response_only: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    responses_from: str = None,
    skip_existing: bool = False,
    limit: int = None,
    output_suffix: str = None,
    model=None,
    tokenizer=None,
    prompt_ids: list = None,
) -> int:
    """Capture raw activations from pre-generated responses.

    Returns number of prompts captured.
    """
    exp_dir = get_path('experiments.base', experiment=experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return 0

    config = load_experiment_config(experiment)
    variant = get_model_variant(experiment, model_variant, mode="application")
    variant_name = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    output_set_name = prompt_set
    if output_suffix:
        output_set_name = f"{output_set_name}_{output_suffix}"

    responses_variant = responses_from or variant_name
    responses_dir = get_path('inference.responses',
                             experiment=experiment,
                             model_variant=responses_variant,
                             prompt_set=prompt_set)
    if not responses_dir.exists():
        print(f"Response JSONs not found: {responses_dir}")
        print(f"Run generate_responses.py first, or check --responses-from variant.")
        return 0

    response_files = sorted(responses_dir.glob("*.json"))
    response_files = [f for f in response_files if not f.stem.endswith('_annotations')]
    if not response_files:
        print(f"No response JSON files found in {responses_dir}")
        return 0

    if prompt_ids is not None:
        prompt_ids_set = set(str(pid) for pid in prompt_ids)
        response_files = [f for f in response_files if f.stem in prompt_ids_set]
        if not response_files:
            print(f"No response files matching prompt_ids: {prompt_ids}")
            return 0

    if limit is not None:
        response_files = response_files[:limit]

    print(f"Found {len(response_files)} response JSONs in {responses_variant}/{prompt_set}")
    if responses_from:
        print(f"Reading responses from variant: {responses_variant}")

    inference_dir = get_path('inference.variant', experiment=experiment, model_variant=variant_name)
    raw_dir = inference_dir / "raw" / "residual" / output_set_name

    if skip_existing:
        original_count = len(response_files)
        response_files = [f for f in response_files if not (raw_dir / f"{f.stem}.pt").exists()]
        skipped = original_count - len(response_files)
        if skipped:
            print(f"Skipping {skipped} already captured")
    if not response_files:
        print("All responses already captured, nothing to do.")
        return 0

    # Load model if not provided
    should_cleanup = model is None
    if model is None:
        model, tokenizer = load_model_with_lora(
            model_name, lora_adapter=lora,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
        )

    n_layers = len(get_inner_model(model).layers)
    print(f"Model has {n_layers} layers")

    comp_list = [c.strip() for c in components.split(',')]
    print(f"Components: {comp_list}")

    capture_layers = None
    if layers:
        capture_layers = parse_layers(layers, n_layers)
        print(f"Capturing {len(capture_layers)} of {n_layers} layers: {capture_layers}")

    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Pre-load and pre-tokenize
    items = []
    for response_file in response_files:
        with open(response_file) as f:
            rj = json.load(f)
        prompt_text = rj['prompt']
        response_text = rj['response']

        if not response_text and rj.get('token_ids'):
            all_ids = torch.tensor(rj['token_ids'])
            prompt_end = rj.get('prompt_end', len(all_ids))
            p_ids = all_ids[:prompt_end]
            r_ids = all_ids[prompt_end:]
        else:
            for eos in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
                if response_text.endswith(eos):
                    response_text = response_text[:-len(eos)]
                    break
            p_ids = tokenize(prompt_text, tokenizer)['input_ids'][0]
            r_ids = tokenize(response_text, tokenizer, add_special_tokens=False)['input_ids'][0]

        items.append((response_file.stem, prompt_text, response_text, p_ids, r_ids))

    max_seq_len = max(len(it[3]) + len(it[4]) for it in items)
    batch_size = calculate_max_batch_size(model, max_seq_len, mode='extraction')

    if is_tp_mode():
        import torch.distributed as dist
        bs_tensor = torch.tensor([batch_size], device='cuda')
        dist.all_reduce(bs_tensor, op=dist.ReduceOp.MIN)
        batch_size = int(bs_tensor.item())

    layer_indices = capture_layers if capture_layers is not None else list(range(n_layers))

    print(f"\n{'='*60}")
    print(f"Capturing {len(items)} prompts → {variant_name}/raw/residual/{output_set_name}/")
    print(f"Batch size: {batch_size} (max_seq_len={max_seq_len})")
    print(f"{'='*60}")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    i = 0
    pbar = tqdm(total=len(items), desc="Prefill capture")
    while i < len(items):
        batch_items = items[i:i + batch_size]
        oom = False

        try:
            full_sequences = [torch.cat([it[3], it[4]]) for it in batch_items]
            batch = pad_sequences(full_sequences, pad_token_id, padding_side='left')
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pad_offsets = batch['pad_offsets']

            with ExitStack() as stack:
                captures = {}
                for component in comp_list:
                    cap = stack.enter_context(MultiLayerCapture(
                        model, component=component, layers=capture_layers, keep_on_gpu=False
                    ))
                    captures[component] = cap

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                component_acts_all = {}
                for component, cap in captures.items():
                    component_acts_all[component] = cap.get_all()

            for b, (prompt_id, prompt_text, response_text, p_ids, r_ids) in enumerate(batch_items):
                pad_offset = pad_offsets[b]
                n_prompt = len(p_ids)
                n_response = len(r_ids)
                prompt_start = pad_offset
                prompt_end = pad_offset + n_prompt
                response_end = pad_offset + n_prompt + n_response

                prompt_acts = {}
                response_acts = {}
                for layer_idx in layer_indices:
                    prompt_acts[layer_idx] = {}
                    response_acts[layer_idx] = {}

                    for component in comp_list:
                        acts = component_acts_all.get(component, {})
                        if layer_idx in acts:
                            full = acts[layer_idx]
                            prompt_acts[layer_idx][component] = full[b, prompt_start:prompt_end, :].cpu()
                            response_acts[layer_idx][component] = full[b, prompt_end:response_end, :].cpu()

                prompt_token_ids = p_ids.tolist()
                response_token_ids = r_ids.tolist()
                data = {
                    'prompt': {
                        'text': prompt_text,
                        'tokens': [tokenizer.decode([tid]) for tid in prompt_token_ids],
                        'token_ids': prompt_token_ids,
                        'activations': prompt_acts,
                        'attention': {},
                    },
                    'response': {
                        'text': response_text,
                        'tokens': [tokenizer.decode([tid]) for tid in response_token_ids],
                        'token_ids': response_token_ids,
                        'activations': response_acts,
                        'attention': [],
                    },
                }
                if is_rank_zero():
                    _save_pt_data(data, prompt_id, raw_dir, response_only=response_only)

            pbar.update(len(batch_items))
            i += batch_size

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            if is_tp_mode():
                raise RuntimeError(
                    f"OOM during TP forward pass (batch_size={batch_size}). "
                    f"NCCL state is corrupted and cannot recover. "
                    f"Re-run with fewer layers or a larger GPU."
                )
            import traceback as tb_mod
            if e.__traceback__:
                tb_mod.clear_frames(e.__traceback__)
            e.__traceback__ = None
            for chained in (e.__context__, e.__cause__):
                if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
                    tb_mod.clear_frames(chained.__traceback__)
                    chained.__traceback__ = None
            del e
            oom = True

        if oom:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError("OOM even with batch_size=1")
            batch_size = max(1, batch_size // 2)
            print(f"\nOOM, reducing batch_size to {batch_size}")

    pbar.close()

    if should_cleanup:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nOutput: {raw_dir}")
    return len(items)


# ============================================================================
# Post-hoc Projection from Saved .pt Files
# ============================================================================

def process_prompt_set(args, inference_dir, prompt_set, model_name, model_variant, extraction_variant, vectors_experiment, steering_variant):
    """Process a single prompt set: load .pt files and project onto trait vectors."""
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

    analysis_massive_dims, top_dims_by_layer = load_massive_dims_from_analysis(args.experiment)
    if analysis_massive_dims:
        print(f"Loaded {len(analysis_massive_dims)} massive dims from calibration for visualization")

    if args.traits:
        trait_list = [tuple(t.split('/')) for t in args.traits.split(',')]
    else:
        trait_list = discover_extracted_traits(vectors_experiment)

    if not trait_list:
        print("No traits found")
        return

    print(f"Projecting onto {len(trait_list)} traits")

    use_top_n = args.multi_vector is not None and args.multi_vector > 0

    layers_spec = args.layers
    if args.layer is not None and layers_spec is None:
        layers_spec = str(args.layer)
    elif layers_spec is None and not use_top_n:
        layers_spec = 'best+5'

    available_raw_layers = set()
    if layers_spec:
        sample = torch.load(raw_files[0], weights_only=False)
        available_raw_layers = set(sample['response']['activations'].keys())
        del sample

    if use_top_n:
        print(f"Multi-vector mode: projecting onto top {args.multi_vector} vectors per trait")
    else:
        print(f"Layers spec: {layers_spec}")

    # Load trait vectors
    trait_vectors = {}
    for category, trait_name in trait_list:
        trait_path = f"{category}/{trait_name}"

        if use_top_n:
            top_vectors = get_top_N_vectors(
                vectors_experiment, trait_path, extraction_variant=extraction_variant,
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
                    vectors_experiment, trait_path, method, layer, extraction_variant, args.component, args.position
                )

                if not vector_path.exists():
                    continue

                try:
                    vector, baseline, per_vec_metadata = load_vector_with_baseline(
                        vectors_experiment, trait_path, method, layer, extraction_variant, args.component, args.position
                    )
                    vector = vector.to(torch.float16)
                    vec_metadata = load_vector_metadata(
                        vectors_experiment, trait_path, method, extraction_variant, args.component, args.position
                    )
                    loaded_vectors.append((vector, method, vector_path, layer, vec_metadata, selection_source, baseline, args.position))
                except FileNotFoundError:
                    continue

            if loaded_vectors:
                trait_vectors[(category, trait_name)] = loaded_vectors
                methods_str = ', '.join([f"L{v[3]} {v[1]}" for v in loaded_vectors])
                print(f"  {trait_path}: [{methods_str}]")
        else:
            best_layer = None
            best_method = None
            best_position = args.position
            best_source = None
            best_score = None

            try:
                spec, spec_meta = get_best_vector_spec(
                    vectors_experiment, trait_path, extraction_variant=extraction_variant,
                    steering_variant=steering_variant,
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

            layers = resolve_layers(layers_spec, best_layer, available_raw_layers)

            if not layers:
                print(f"  Skip {trait_path}: no valid layers resolved")
                continue

            loaded_vectors = []
            for layer in layers:
                method = args.method
                if method is None:
                    if layer == best_layer and best_method:
                        method = best_method
                    else:
                        method = find_vector_method(
                            vectors_experiment, trait_path, layer, extraction_variant,
                            args.component, best_position or 'response[:]'
                        )

                if not method:
                    print(f"    Skip L{layer}: no vector found")
                    continue

                position = best_position
                if position is None:
                    candidates = _discover_vectors(
                        vectors_experiment, trait_path, extraction_variant,
                        component=args.component, layer=layer
                    )
                    if candidates:
                        position = candidates[0]['position']
                    else:
                        print(f"    Skip L{layer}: no vectors found")
                        continue

                try:
                    vector, baseline, per_vec_meta = load_vector_with_baseline(
                        vectors_experiment, trait_path, method, layer, extraction_variant,
                        args.component, position
                    )
                    vector = vector.to(torch.float16)
                except FileNotFoundError:
                    print(f"    Skip L{layer}: vector file not found")
                    continue

                vec_metadata = load_vector_metadata(
                    vectors_experiment, trait_path, method, extraction_variant,
                    args.component, position
                )
                source = best_source if layer == best_layer else 'layers_arg'
                loaded_vectors.append((vector, method, None, layer, vec_metadata, source, baseline, position))

            loaded_layers = {v[3] for v in loaded_vectors}
            skipped_layers = sorted(set(layers) - loaded_layers)
            if skipped_layers:
                print(f"\n  WARNING: {trait_path} — skipped {len(skipped_layers)}/{len(layers)} layers: {skipped_layers}")
                print(f"    Vectors missing. Extract them first: python extraction/run_extraction_pipeline.py --experiment {args.experiment} --traits {trait_path} --only-stage 3,4 --layers {','.join(str(l) for l in skipped_layers)}\n")

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

        logit_lens_data = None
        if args.logit_lens and model is not None:
            logit_lens_data = {
                'prompt': compute_logit_lens_from_raw(prompt_acts, model, tokenizer, n_layers) if prompt_acts else {},
                'response': compute_logit_lens_from_raw(response_acts, model, tokenizer, n_layers)
            }

        results = project_prompt_onto_traits(
            prompt_activations=prompt_acts,
            response_activations=response_acts,
            trait_vectors=trait_vectors,
            component=args.component,
            centered=args.centered,
            massive_dims_info=(analysis_massive_dims, top_dims_by_layer) if analysis_massive_dims else None,
            logit_lens_data=logit_lens_data,
            n_prompt_tokens=len(data['prompt']['tokens']),
            n_response_tokens=len(data['response']['tokens']),
            prompt_set=prompt_set,
            prompt_id=prompt_id,
        )

        for trait_path, proj_data in results.items():
            out_dir = inference_dir / "projections" / trait_path / prompt_set
            out_file = out_dir / f"{prompt_id}.json"
            if args.skip_existing and out_file.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                dump_compact(proj_data, f)

        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nProjections saved to: {inference_dir}/projections/{{trait}}/{prompt_set}/")


# ============================================================================
# CLI
# ============================================================================

def main():
    import builtins
    _original_print = builtins.print
    if is_tp_mode():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        if not is_rank_zero():
            builtins.print = lambda *a, **k: None

    parser = argparse.ArgumentParser(
        description="Capture activations and/or project onto trait vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)

    # Mode
    parser.add_argument("--capture", action="store_true",
                        help="Capture mode: save raw activations as .pt files")

    # Capture-specific options
    capture_group = parser.add_argument_group("capture options")
    capture_group.add_argument("--components", type=str, default="residual",
                               help="Components to capture (default: residual)")
    capture_group.add_argument("--response-only", action="store_true",
                               help="Only save response token activations")
    capture_group.add_argument("--responses-from", type=str, default=None,
                               help="Read responses from a different variant")
    capture_group.add_argument("--limit", type=int, default=None)
    capture_group.add_argument("--output-suffix", type=str, default=None)
    capture_group.add_argument("--prompt-ids", type=str, default=None,
                               help="Comma-separated prompt IDs")

    # Projection-specific options
    proj_group = parser.add_argument_group("projection options")
    proj_group.add_argument("--all-prompt-sets", action="store_true")
    proj_group.add_argument("--traits", help="Comma-separated traits")
    proj_group.add_argument("--layer", type=int, help="Override layer for all traits")
    proj_group.add_argument("--layers", help="Layer spec: 'best', 'best,best+5', '20,25'")
    proj_group.add_argument("--method", help="Vector method")
    proj_group.add_argument("--component", choices=["residual", "attn_contribution"],
                            default="residual")
    proj_group.add_argument("--position", default=None)
    proj_group.add_argument("--logit-lens", action="store_true")
    proj_group.add_argument("--centered", action="store_true")
    proj_group.add_argument("--multi-vector", type=int, metavar="N")
    proj_group.add_argument("--no-calibration", action="store_true")
    proj_group.add_argument("--extraction-variant", default=None)
    proj_group.add_argument("--vectors-experiment", default=None)

    # Shared options
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    add_backend_args(parser)

    args = parser.parse_args()

    try:
        if args.capture:
            # Capture mode
            capture_raw_activations(
                experiment=args.experiment,
                prompt_set=args.prompt_set,
                model_variant=args.model_variant,
                components=args.components,
                layers=args.layers,
                response_only=args.response_only,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                responses_from=args.responses_from,
                skip_existing=args.skip_existing,
                limit=args.limit,
                output_suffix=args.output_suffix,
                prompt_ids=args.prompt_ids.split(",") if args.prompt_ids else None,
            )
        else:
            # Projection mode
            vectors_experiment = args.vectors_experiment or args.experiment

            if args.layer is not None and args.layers is not None:
                parser.error("--layer and --layers are mutually exclusive")

            app_variant = get_model_variant(args.experiment, args.model_variant, mode="application")
            model_variant = app_variant['name']
            model_name = app_variant['model']

            ext_variant = get_model_variant(vectors_experiment, args.extraction_variant, mode="extraction")
            extraction_variant = ext_variant['name']

            if vectors_experiment != args.experiment:
                steering_variant = get_model_variant(vectors_experiment, None, mode="application")['name']
            else:
                steering_variant = model_variant

            inference_dir = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant)

            default_inference_dir = get_path('inference.base', experiment=args.experiment)
            calibration_path = default_inference_dir / 'massive_activations' / 'calibration.json'
            if not calibration_path.exists() and not args.no_calibration:
                print(f"Error: Massive dims calibration not found at {calibration_path}")
                print(f"\nRun calibration first:")
                print(f"  python analysis/massive_activations.py --experiment {args.experiment}")
                print(f"\nOr skip with --no-calibration")
                return

            if args.all_prompt_sets:
                raw_residual_dir = inference_dir / "raw" / "residual"
                if not raw_residual_dir.exists():
                    print(f"Raw residual directory not found: {raw_residual_dir}")
                    return
                prompt_sets = [d.name for d in raw_residual_dir.iterdir() if d.is_dir()]
                print(f"Found {len(prompt_sets)} prompt sets: {', '.join(prompt_sets)}")
            else:
                prompt_sets = [args.prompt_set]

            for prompt_set in prompt_sets:
                print(f"\n{'='*60}\nProcessing prompt set: {prompt_set}\n{'='*60}")
                process_prompt_set(args, inference_dir, prompt_set, model_name, model_variant,
                                   extraction_variant, vectors_experiment, steering_variant)
    finally:
        builtins.print = _original_print
        if is_tp_mode():
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
