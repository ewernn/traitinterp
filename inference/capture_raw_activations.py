#!/usr/bin/env python3
"""
Unified activation capture for trait analysis.

Captures raw activations and computes projections onto trait vectors.
Residual stream is ALWAYS captured (baseline). Optional add-ons for visualization.
Uses batched generation for efficient processing of large prompt sets.

Baseline (always):
  - Capture residual stream at all layers
  - Save raw .pt file
  - Project onto all trait vectors (unless --no-project)

Optional add-ons:
  --attention       : Also save attention patterns as JSON (~12MB per prompt)
  --logit-lens      : Also save logit lens predictions as JSON (~4MB per prompt)
  --layer-internals : Also save full internals as .pt files (~175MB per prompt)
  --capture-attn    : Also capture raw attn_out (for attn_out vector projections)

Performance:
  --batch-size N    : Batch size for capture (auto-detect from VRAM if not set)
  --limit N         : Limit number of prompts to process (for testing)

Post-hoc extraction:
  --replay          : Load saved tokens from .pt, extract attention/logit-lens
                      (no new generation, uses exact same tokens)

Storage:
  Raw activations      : experiments/{exp}/inference/raw/residual/{prompt_set}/{id}.pt
  Responses (shared)   : experiments/{exp}/inference/responses/{prompt_set}/{id}.json
  Projections (slim)   : experiments/{exp}/inference/{category}/{trait}/residual_stream/{prompt_set}/{id}.json
  Layer internals      : experiments/{exp}/inference/raw/internals/{prompt_set}/{id}_L{layer}.pt
  Attention JSON       : experiments/{exp}/analysis/per_token/{prompt_set}/{id}_attention.json
  Logit lens JSON      : experiments/{exp}/analysis/per_token/{prompt_set}/{id}_logit_lens.json

Note: Responses are stored once per prompt (trait-independent). Projections are slim JSONs
containing only projections + dynamics, referencing the shared response data.

Usage:
    # Basic: capture residual stream + project onto all traits
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # Add attention patterns for visualization
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set dynamic \\
        --attention

    # Full mechanistic capture (projections + attention + logit lens + internals)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set dynamic \\
        --layer-internals all

    # Just capture raw activations, no projections
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --no-project

    # Extract attention/logit-lens from existing captures (no new generation)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set single_trait \\
        --replay --attention --logit-lens

    # Capture with attn_out for attn_out vector projections
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set harmful \\
        --capture-attn
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from traitlens import HookManager, projection
from traitlens.compute import compute_derivative, compute_second_derivative
from utils.model import format_prompt, load_experiment_config
from utils.vectors import load_vector_metadata
from utils.generation import generate_with_capture, get_available_vram_gb, calculate_max_batch_size


MODEL_NAME = "google/gemma-2-2b-it"



# ============================================================================
# Trait Discovery
# ============================================================================

def discover_traits(experiment_name: str) -> List[Tuple[str, str]]:
    """Discover all traits with vectors in an experiment."""
    from utils.paths import get
    extraction_dir = get('extraction.base', experiment=experiment_name)

    if not extraction_dir.exists():
        raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            vectors_dir = trait_dir / "vectors"
            if vectors_dir.exists() and list(vectors_dir.glob('*.pt')):
                traits.append((category_dir.name, trait_dir.name))

    if not traits:
        raise ValueError(f"No traits with vectors found in {extraction_dir}")
    return traits


def find_vector_method(vectors_dir: Path, layer: int) -> Optional[str]:
    """Auto-detect best vector method for a layer."""
    for method in ["probe", "mean_diff", "gradient"]:
        if (vectors_dir / f"{method}_layer{layer}.pt").exists():
            return method
    return None


# ============================================================================
# Dynamics Analysis
# ============================================================================

def analyze_dynamics(trajectory: torch.Tensor) -> Dict:
    """Compute velocity, acceleration, commitment point, and persistence."""
    if len(trajectory) < 2:
        return {
            'commitment_point': None,
            'peak_velocity': 0.0,
            'avg_velocity': 0.0,
            'persistence': 0,
            'velocity': [],
            'acceleration': [],
        }

    velocity = compute_derivative(trajectory.unsqueeze(-1)).squeeze(-1)

    if len(trajectory) >= 3:
        acceleration = compute_second_derivative(trajectory.unsqueeze(-1)).squeeze(-1)
    else:
        acceleration = torch.tensor([])

    # Commitment point: where acceleration drops below threshold
    commitment = None
    if len(acceleration) > 0:
        candidates = (acceleration.abs() < 0.1).nonzero()
        if len(candidates) > 0:
            commitment = candidates[0].item()

    # Persistence: tokens above threshold after peak
    persistence = 0
    if len(trajectory) > 0:
        peak_idx = trajectory.abs().argmax().item()
        peak_value = trajectory[peak_idx].abs().item()
        if peak_idx < len(trajectory) - 1:
            threshold = peak_value * 0.5
            persistence = (trajectory[peak_idx + 1:].abs() > threshold).sum().item()

    return {
        'commitment_point': commitment,
        'peak_velocity': velocity.abs().max().item() if len(velocity) > 0 else 0.0,
        'avg_velocity': velocity.abs().mean().item() if len(velocity) > 0 else 0.0,
        'persistence': persistence,
        'velocity': velocity.tolist(),
        'acceleration': acceleration.tolist() if len(acceleration) > 0 else [],
    }


# ============================================================================
# CaptureResult Conversion
# ============================================================================

def capture_result_to_data(result, n_layers: int) -> Dict:
    """Convert CaptureResult from utils.generation to legacy data format."""
    prompt_acts = {}
    response_acts = {}

    for layer_idx in range(n_layers):
        prompt_acts[layer_idx] = result.prompt_activations.get(layer_idx, {})
        response_acts[layer_idx] = result.response_activations.get(layer_idx, {})

    return {
        'prompt': {
            'text': result.prompt_text,
            'tokens': result.prompt_tokens,
            'token_ids': result.prompt_token_ids,
            'activations': prompt_acts,
            'attention': {}  # Not captured in batched mode
        },
        'response': {
            'text': result.response_text,
            'tokens': result.response_tokens,
            'token_ids': result.response_token_ids,
            'activations': response_acts,
            'attention': []  # Not captured in batched mode
        }
    }


# ============================================================================
# Residual Stream Capture (Legacy - for layer_internals mode)
# ============================================================================

def create_residual_storage(n_layers: int, capture_attn: bool = False) -> Dict:
    """Create storage for residual stream capture."""
    base = {'residual_in': [], 'after_attn': [], 'residual_out': []}
    if capture_attn:
        base['attn_out'] = []
    return {i: {k: [] for k in base} for i in range(n_layers)}


def setup_residual_hooks(hook_manager: HookManager, storage: Dict, n_layers: int, mode: str,
                         capture_attn: bool = False):
    """Register hooks for residual stream at all layers."""
    for i in range(n_layers):
        def make_layer_hook(layer_idx):
            def hook(module, inp, out):
                inp_t = inp[0] if isinstance(inp, tuple) else inp
                out_t = out[0] if isinstance(out, tuple) else out
                if mode == 'response':
                    storage[layer_idx]['residual_in'].append(inp_t[:, -1, :].detach().cpu())
                    storage[layer_idx]['residual_out'].append(out_t[:, -1, :].detach().cpu())
                else:
                    storage[layer_idx]['residual_in'].append(inp_t.detach().cpu())
                    storage[layer_idx]['residual_out'].append(out_t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{i}", make_layer_hook(i))

        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                inp_t = inp[0] if isinstance(inp, tuple) else inp
                if mode == 'response':
                    storage[layer_idx]['after_attn'].append(inp_t[:, -1, :].detach().cpu())
                else:
                    storage[layer_idx]['after_attn'].append(inp_t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{i}.mlp", make_mlp_hook(i))

        # Optional: capture raw attention output (before residual addition)
        if capture_attn:
            def make_attn_hook(layer_idx):
                def hook(module, inp, out):
                    out_t = out[0] if isinstance(out, tuple) else out
                    if mode == 'response':
                        storage[layer_idx]['attn_out'].append(out_t[:, -1, :].detach().cpu())
                    else:
                        storage[layer_idx]['attn_out'].append(out_t.detach().cpu())
                return hook
            hook_manager.add_forward_hook(f"model.layers.{i}.self_attn", make_attn_hook(i))


def capture_residual_stream(model, tokenizer, prompt_text: str, n_layers: int,
                            max_new_tokens: int, temperature: float,
                            capture_attn: bool = False) -> Dict:
    """Capture residual stream activations at all layers."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Prompt capture
    prompt_storage = create_residual_storage(n_layers, capture_attn=capture_attn)
    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, prompt_storage, n_layers, 'prompt', capture_attn=capture_attn)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    prompt_attention = {f'layer_{i}': attn[0].mean(dim=0).detach().cpu()
                       for i, attn in enumerate(outputs.attentions)}

    prompt_acts = {}
    for i in range(n_layers):
        prompt_acts[i] = {k: v[0].squeeze(0) for k, v in prompt_storage[i].items()}

    # Response capture
    response_storage = create_residual_storage(n_layers, capture_attn=capture_attn)
    context = inputs['input_ids'].clone()
    generated_ids = []
    response_attention = []

    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, response_storage, n_layers, 'response', capture_attn=capture_attn)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=context, output_attentions=True, return_dict=True)

            step_attn = {f'layer_{i}': attn[0].mean(dim=0)[-1, :].detach().cpu()
                        for i, attn in enumerate(outputs.attentions)}
            response_attention.append(step_attn)

            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            context = torch.cat([context, torch.tensor([[next_id]], device=model.device)], dim=1)
            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

    response_tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    response_acts = {}
    for i in range(n_layers):
        response_acts[i] = {}
        for k, v in response_storage[i].items():
            if v:
                response_acts[i][k] = torch.stack([a.squeeze(0) for a in v], dim=0)
            else:
                response_acts[i][k] = torch.empty(0, model.config.hidden_size)

    return {
        'prompt': {'text': prompt_text, 'tokens': tokens, 'token_ids': token_ids,
                   'activations': prompt_acts, 'attention': prompt_attention},
        'response': {'text': response_text, 'tokens': response_tokens, 'token_ids': generated_ids,
                     'activations': response_acts, 'attention': response_attention}
    }


# ============================================================================
# Layer Internals Capture
# ============================================================================

def create_internals_storage() -> Dict:
    """Create storage for single-layer deep capture."""
    return {
        'attention': {'q_proj': [], 'k_proj': [], 'v_proj': [], 'attn_weights': []},
        'mlp': {'up_proj': [], 'gelu': [], 'down_proj': []},
        'residual': {'input': [], 'after_attn': [], 'output': []}
    }


def setup_internals_hooks(hook_manager: HookManager, storage: Dict, layer_idx: int, mode: str):
    """Register hooks for single layer internals."""
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['attention'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{layer_idx}.self_attn.{proj}", make_hook(proj))

    for proj, path in [('up_proj', 'up_proj'), ('gelu', 'act_fn'), ('down_proj', 'down_proj')]:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['mlp'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{layer_idx}.mlp.{path}", make_hook(proj))

    def layer_hook(module, inp, out):
        inp_t = inp[0] if isinstance(inp, tuple) else inp
        out_t = out[0] if isinstance(out, tuple) else out
        if mode == 'response':
            storage['residual']['input'].append(inp_t[:, -1, :].detach().cpu())
            storage['residual']['output'].append(out_t[:, -1, :].detach().cpu())
        else:
            storage['residual']['input'].append(inp_t.detach().cpu())
            storage['residual']['output'].append(out_t.detach().cpu())
    hook_manager.add_forward_hook(f"model.layers.{layer_idx}", layer_hook)

    def mlp_input_hook(module, inp, out):
        inp_t = inp[0] if isinstance(inp, tuple) else inp
        t = inp_t[:, -1, :] if mode == 'response' else inp_t
        storage['residual']['after_attn'].append(t.detach().cpu())
    hook_manager.add_forward_hook(f"model.layers.{layer_idx}.mlp", mlp_input_hook)


def capture_multiple_layer_internals(model, tokenizer, prompt_text: str, layer_indices: list,
                                     max_new_tokens: int, temperature: float) -> Dict[int, Dict]:
    """Capture internals for multiple layers in a SINGLE forward pass."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Storage for all requested layers
    all_layer_data = {}

    # PROMPT PHASE - Single forward pass for all layers
    prompt_storages = {idx: create_internals_storage() for idx in layer_indices}

    with HookManager(model) as hooks:
        # Set up hooks for ALL requested layers
        for layer_idx in layer_indices:
            setup_internals_hooks(hooks, prompt_storages[layer_idx], layer_idx, 'prompt')

        # Single forward pass captures all layers!
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    # Extract data for each layer
    for layer_idx in layer_indices:
        prompt_internals = {
            'attention': {k: v[0].squeeze(0) for k, v in prompt_storages[layer_idx]['attention'].items() if v},
            'mlp': {k: v[0].squeeze(0) for k, v in prompt_storages[layer_idx]['mlp'].items() if v},
            'residual': {k: v[0].squeeze(0) for k, v in prompt_storages[layer_idx]['residual'].items() if v}
        }
        prompt_internals['attention']['attn_weights'] = outputs.attentions[layer_idx][0].detach().cpu()
        all_layer_data[layer_idx] = {'prompt': prompt_internals}

    # RESPONSE PHASE - Generate once, capture all layers
    response_storages = {idx: create_internals_storage() for idx in layer_indices}
    context = inputs['input_ids'].clone()
    generated_ids = []

    for step in range(max_new_tokens):
        with HookManager(model) as hooks:
            # Set up hooks for ALL layers
            for layer_idx in layer_indices:
                setup_internals_hooks(hooks, response_storages[layer_idx], layer_idx, 'response')

            # Single forward pass
            with torch.no_grad():
                outputs = model(input_ids=context, output_attentions=True, return_dict=True)

        # Save attention for all layers
        for layer_idx in layer_indices:
            response_storages[layer_idx]['attention']['attn_weights'].append(
                outputs.attentions[layer_idx][0].detach().cpu()
            )

        # Generate next token (same for all layers)
        logits = outputs.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        context = torch.cat([context, torch.tensor([[next_id]], device=model.device)], dim=1)
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    # Package response data for all layers
    response_tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    for layer_idx in layer_indices:
        response_internals = {
            'attention': {},
            'mlp': {},
            'residual': {}
        }

        # Handle attention separately (variable size due to growing context)
        for k, v in response_storages[layer_idx]['attention'].items():
            if k == 'attn_weights':
                # Keep as list of tensors with different sizes
                response_internals['attention'][k] = v
            elif v:
                # Other attention tensors should be same size, can stack
                response_internals['attention'][k] = torch.stack(v) if len(v) > 0 else torch.tensor([])

        # MLP tensors are all same size, can stack
        for k, v in response_storages[layer_idx]['mlp'].items():
            if v:
                response_internals['mlp'][k] = torch.stack(v) if len(v) > 0 else torch.tensor([])

        # Residual tensors are all same size, can stack
        for k, v in response_storages[layer_idx]['residual'].items():
            if v:
                response_internals['residual'][k] = torch.stack(v) if len(v) > 0 else torch.tensor([])

        all_layer_data[layer_idx]['response'] = response_internals
        all_layer_data[layer_idx]['prompt_text'] = prompt_text
        all_layer_data[layer_idx]['prompt_tokens'] = tokens
        all_layer_data[layer_idx]['response_text'] = response_text
        all_layer_data[layer_idx]['response_tokens'] = response_tokens

    return all_layer_data



# ============================================================================
# Attention Extraction for Visualization
# ============================================================================

def extract_attention_for_visualization(all_layer_data: Dict[int, Dict], n_layers: int) -> Dict:
    """
    Format captured attention data into visualization JSON structure.

    Output structure matches what layer-deep-dive.js expects:
    {
        "tokens": [...],
        "n_prompt_tokens": N,
        "n_response_tokens": M,
        "attention": [
            {"token_idx": 0, "context_size": 1, "by_layer": [[head0], [head1], ...]},
            ...
        ]
    }
    """
    # Get token info from any layer's data
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_tokens = layer_data['prompt_tokens']
    response_tokens = layer_data['response_tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)
    n_total = len(all_tokens)

    result = {
        "prompt_id": None,  # Will be set by caller
        "prompt_set": None,  # Will be set by caller
        "n_layers": n_layers,
        "n_heads": 8,  # Gemma 2B
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "attention": []
    }

    # Extract attention for prompt tokens
    for token_idx in range(n_prompt_tokens):
        token_attention = {
            "token_idx": token_idx,
            "context_size": token_idx + 1,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_attention["by_layer"].append([[0.0] * (token_idx + 1)] * 8)
                continue

            attn_weights = all_layer_data[layer]['prompt']['attention'].get('attn_weights')
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                # Prompt attention is [8, seq, seq]
                if attn_weights.dim() == 3 and attn_weights.shape[1] > token_idx:
                    heads_attn = attn_weights[:, token_idx, :token_idx+1].tolist()
                else:
                    heads_attn = [[0.0] * (token_idx + 1)] * 8
            else:
                heads_attn = [[0.0] * (token_idx + 1)] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    # Extract attention for response tokens
    for resp_idx in range(n_response_tokens):
        token_idx = n_prompt_tokens + resp_idx
        context_size = token_idx + 1

        token_attention = {
            "token_idx": token_idx,
            "context_size": context_size,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_attention["by_layer"].append([[0.0] * context_size] * 8)
                continue

            attn_weights = all_layer_data[layer]['response']['attention'].get('attn_weights', [])

            if resp_idx < len(attn_weights):
                attn = attn_weights[resp_idx]
                if isinstance(attn, torch.Tensor) and attn.dim() == 3:
                    # Response attention: [8, seq, seq] - last row is current token's attention
                    last_row_idx = attn.shape[1] - 1
                    key_len = attn.shape[2]
                    heads_attn = attn[:, last_row_idx, :key_len].tolist()
                    # Pad if needed
                    if len(heads_attn[0]) < context_size:
                        for h in range(len(heads_attn)):
                            heads_attn[h] = heads_attn[h] + [0.0] * (context_size - len(heads_attn[h]))
                else:
                    heads_attn = [[0.0] * context_size] * 8
            else:
                heads_attn = [[0.0] * context_size] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    return result


# ============================================================================
# Logit Lens Extraction for Visualization
# ============================================================================

def extract_logit_lens_for_visualization(all_layer_data: Dict[int, Dict], model, tokenizer,
                                          n_layers: int, top_k: int = 50) -> Dict:
    """
    Apply logit lens to residual stream and format for visualization.

    Output structure matches what layer-deep-dive.js expects:
    {
        "tokens": [...],
        "n_prompt_tokens": N,
        "predictions": [
            {
                "token_idx": 0,
                "actual_next_token": "...",
                "by_layer": [
                    {"layer": 0, "top_k": [...], "actual_rank": N, "actual_prob": 0.xx},
                    ...
                ]
            },
            ...
        ]
    }
    """
    # Get token info
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_tokens = layer_data['prompt_tokens']
    response_tokens = layer_data['response_tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)
    n_total = len(all_tokens)

    # Get model components for logit lens
    norm = model.model.norm
    lm_head = model.lm_head

    result = {
        "prompt_id": None,  # Will be set by caller
        "prompt_set": None,  # Will be set by caller
        "n_layers": n_layers,
        "top_k": top_k,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "predictions": []
    }

    # Process each token (predicting next token)
    for token_idx in range(n_total - 1):
        actual_next_token = all_tokens[token_idx + 1] if token_idx + 1 < n_total else None
        # Get actual next token ID by encoding just that token
        if actual_next_token:
            actual_next_ids = tokenizer.encode(actual_next_token, add_special_tokens=False)
            actual_next_id = actual_next_ids[0] if actual_next_ids else None
        else:
            actual_next_id = None

        token_predictions = {
            "token_idx": token_idx,
            "actual_next_token": actual_next_token,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })
                continue

            # Get residual for this token at this layer
            if token_idx < n_prompt_tokens:
                # Prompt token
                residual_data = all_layer_data[layer]['prompt']['residual'].get('output')
                if residual_data is not None and token_idx < residual_data.shape[0]:
                    residual = residual_data[token_idx]
                else:
                    residual = None
            else:
                # Response token
                resp_idx = token_idx - n_prompt_tokens
                residual_data = all_layer_data[layer]['response']['residual'].get('output')
                if residual_data is not None and resp_idx < residual_data.shape[0]:
                    residual = residual_data[resp_idx]
                else:
                    residual = None

            if residual is not None:
                # Apply logit lens: norm -> lm_head -> softmax
                with torch.no_grad():
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)
                    residual = residual.squeeze()
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)

                    normed = norm(residual.float().to(model.device))
                    logits = lm_head(normed.half())
                    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)

                    # Get top-k
                    top_k_probs, top_k_ids = torch.topk(probs, top_k)

                    top_k_list = []
                    for i in range(top_k):
                        token_id = top_k_ids[i].item()
                        token_str = tokenizer.decode([token_id])
                        top_k_list.append({
                            "token": token_str,
                            "prob": round(top_k_probs[i].item(), 6)
                        })

                    # Find actual token's rank and prob
                    actual_rank = None
                    actual_prob = None
                    if actual_next_id is not None:
                        actual_prob = round(probs[actual_next_id].item(), 6)
                        sorted_indices = torch.argsort(probs, descending=True)
                        rank_tensor = (sorted_indices == actual_next_id).nonzero()
                        if len(rank_tensor) > 0:
                            actual_rank = rank_tensor[0].item() + 1  # 1-indexed

                    token_predictions["by_layer"].append({
                        "layer": layer,
                        "top_k": top_k_list,
                        "actual_rank": actual_rank,
                        "actual_prob": actual_prob
                    })
            else:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })

        result["predictions"].append(token_predictions)

    return result


# ============================================================================
# Data Extraction Helpers
# ============================================================================

def extract_residual_from_internals(all_layer_data: Dict[int, Dict], n_layers: int) -> Dict:
    """
    Extract residual stream data from layer internals for projection.

    Returns structure compatible with capture_residual_stream output:
    {
        'prompt': {'text': ..., 'tokens': ..., 'token_ids': ..., 'activations': {layer: {...}}},
        'response': {'text': ..., 'tokens': ..., 'token_ids': ..., 'activations': {layer: {...}}}
    }
    """
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_tokens = layer_data['prompt_tokens']
    response_tokens = layer_data['response_tokens']
    prompt_text = layer_data['prompt_text']
    response_text = layer_data['response_text']

    # Build activations dict in the format expected by project_onto_vector
    prompt_acts = {}
    response_acts = {}

    for layer_idx in range(n_layers):
        if layer_idx in all_layer_data:
            layer = all_layer_data[layer_idx]
            prompt_acts[layer_idx] = {
                'residual_in': layer['prompt']['residual'].get('input', torch.empty(0)),
                'after_attn': layer['prompt']['residual'].get('after_attn', torch.empty(0)),
                'residual_out': layer['prompt']['residual'].get('output', torch.empty(0))
            }
            response_acts[layer_idx] = {
                'residual_in': layer['response']['residual'].get('input', torch.empty(0)),
                'after_attn': layer['response']['residual'].get('after_attn', torch.empty(0)),
                'residual_out': layer['response']['residual'].get('output', torch.empty(0))
            }
        else:
            # Layer not captured - placeholder empty tensors
            prompt_acts[layer_idx] = {
                'residual_in': torch.empty(0),
                'after_attn': torch.empty(0),
                'residual_out': torch.empty(0)
            }
            response_acts[layer_idx] = {
                'residual_in': torch.empty(0),
                'after_attn': torch.empty(0),
                'residual_out': torch.empty(0)
            }

    return {
        'prompt': {
            'text': prompt_text,
            'tokens': prompt_tokens,
            'token_ids': [],  # Not stored in internals, but not needed for projection
            'activations': prompt_acts
        },
        'response': {
            'text': response_text,
            'tokens': response_tokens,
            'token_ids': [],  # Not stored in internals
            'activations': response_acts
        }
    }


def extract_attention_from_residual_capture(data: Dict, n_layers: int) -> Dict:
    """
    Extract attention patterns from residual stream capture data.

    The residual capture already has attention via output_attentions=True.
    Format for visualization JSON.
    """
    prompt_tokens = data['prompt']['tokens']
    response_tokens = data['response']['tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "n_heads": 8,  # Gemma 2B
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "attention": []
    }

    # Extract prompt attention
    prompt_attention = data['prompt'].get('attention', {})
    for token_idx in range(n_prompt_tokens):
        token_attention = {
            "token_idx": token_idx,
            "context_size": token_idx + 1,
            "by_layer": []
        }

        for layer in range(n_layers):
            layer_key = f'layer_{layer}'
            if layer_key in prompt_attention:
                attn = prompt_attention[layer_key]
                if isinstance(attn, torch.Tensor) and attn.dim() == 2:
                    # [seq, seq] head-averaged attention
                    if attn.shape[0] > token_idx:
                        # Expand to per-head format (all heads same since head-averaged)
                        row = attn[token_idx, :token_idx+1].tolist()
                        heads_attn = [row] * 8
                    else:
                        heads_attn = [[0.0] * (token_idx + 1)] * 8
                else:
                    heads_attn = [[0.0] * (token_idx + 1)] * 8
            else:
                heads_attn = [[0.0] * (token_idx + 1)] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    # Extract response attention
    response_attention = data['response'].get('attention', [])
    for resp_idx in range(n_response_tokens):
        token_idx = n_prompt_tokens + resp_idx
        context_size = token_idx + 1

        token_attention = {
            "token_idx": token_idx,
            "context_size": context_size,
            "by_layer": []
        }

        for layer in range(n_layers):
            layer_key = f'layer_{layer}'
            if resp_idx < len(response_attention):
                step_attn = response_attention[resp_idx]
                if layer_key in step_attn:
                    attn = step_attn[layer_key]
                    if isinstance(attn, torch.Tensor):
                        # [context] attention for current token
                        row = attn.tolist()
                        # Pad to context_size if needed
                        if len(row) < context_size:
                            row = row + [0.0] * (context_size - len(row))
                        heads_attn = [row] * 8
                    else:
                        heads_attn = [[0.0] * context_size] * 8
                else:
                    heads_attn = [[0.0] * context_size] * 8
            else:
                heads_attn = [[0.0] * context_size] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    return result


def extract_logit_lens_from_residual_capture(data: Dict, model, tokenizer, n_layers: int, top_k: int = 50) -> Dict:
    """
    Apply logit lens to residual stream capture data.
    """
    prompt_tokens = data['prompt']['tokens']
    response_tokens = data['response']['tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)
    n_total = len(all_tokens)

    # Get model components for logit lens
    norm = model.model.norm
    lm_head = model.lm_head

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "top_k": top_k,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "predictions": []
    }

    prompt_acts = data['prompt']['activations']
    response_acts = data['response']['activations']

    # Process each token (predicting next token)
    for token_idx in range(n_total - 1):
        actual_next_token = all_tokens[token_idx + 1] if token_idx + 1 < n_total else None
        if actual_next_token:
            actual_next_ids = tokenizer.encode(actual_next_token, add_special_tokens=False)
            actual_next_id = actual_next_ids[0] if actual_next_ids else None
        else:
            actual_next_id = None

        token_predictions = {
            "token_idx": token_idx,
            "actual_next_token": actual_next_token,
            "by_layer": []
        }

        for layer in range(n_layers):
            # Get residual for this token at this layer
            if token_idx < n_prompt_tokens:
                residual_data = prompt_acts.get(layer, {}).get('residual_out')
                if residual_data is not None and len(residual_data) > 0 and token_idx < residual_data.shape[0]:
                    residual = residual_data[token_idx]
                else:
                    residual = None
            else:
                resp_idx = token_idx - n_prompt_tokens
                residual_data = response_acts.get(layer, {}).get('residual_out')
                if residual_data is not None and len(residual_data) > 0 and resp_idx < residual_data.shape[0]:
                    residual = residual_data[resp_idx]
                else:
                    residual = None

            if residual is not None:
                with torch.no_grad():
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)
                    residual = residual.squeeze()
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)

                    normed = norm(residual.float().to(model.device))
                    logits = lm_head(normed.half())
                    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)

                    top_k_probs, top_k_ids = torch.topk(probs, top_k)

                    top_k_list = []
                    for i in range(top_k):
                        token_id = top_k_ids[i].item()
                        token_str = tokenizer.decode([token_id])
                        top_k_list.append({
                            "token": token_str,
                            "prob": round(top_k_probs[i].item(), 6)
                        })

                    actual_rank = None
                    actual_prob = None
                    if actual_next_id is not None:
                        actual_prob = round(probs[actual_next_id].item(), 6)
                        sorted_indices = torch.argsort(probs, descending=True)
                        rank_tensor = (sorted_indices == actual_next_id).nonzero()
                        if len(rank_tensor) > 0:
                            actual_rank = rank_tensor[0].item() + 1

                    token_predictions["by_layer"].append({
                        "layer": layer,
                        "top_k": top_k_list,
                        "actual_rank": actual_rank,
                        "actual_prob": actual_prob
                    })
            else:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })

        result["predictions"].append(token_predictions)

    return result


# ============================================================================
# Projection
# ============================================================================

def project_onto_vector(activations: Dict, vector: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Project activations onto trait vector. Returns [n_tokens, n_layers, 3]."""
    n_tokens = activations[0]['residual_in'].shape[0]
    result = torch.zeros(n_tokens, n_layers, 3)
    sublayers = ['residual_in', 'after_attn', 'residual_out']

    for layer in range(n_layers):
        for s_idx, sublayer in enumerate(sublayers):
            result[:, layer, s_idx] = projection(activations[layer][sublayer], vector, normalize_vector=True)

    return result


# ============================================================================
# Data Saving Helper
# ============================================================================

def _save_capture_data(
    data: Dict, prompt_item: Dict, set_name: str, inference_dir: Path,
    trait_vectors: Dict, n_layers: int, args, get_path,
    all_layer_data=None, layer_indices=None, model=None, tokenizer=None
):
    """Save captured data: raw .pt, response JSON, projections, and optionals."""
    prompt_id = prompt_item['id']
    prompt_note = prompt_item.get('note', '')

    # Save raw residual .pt
    raw_dir = inference_dir / "raw" / "residual" / set_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, raw_dir / f"{prompt_id}.pt")

    # Save response JSON (shared, trait-independent)
    responses_dir = inference_dir / "responses" / set_name
    responses_dir.mkdir(parents=True, exist_ok=True)
    response_data = {
        'prompt': {
            'text': data['prompt']['text'],
            'tokens': data['prompt']['tokens'],
            'token_ids': data['prompt'].get('token_ids', []),
            'n_tokens': len(data['prompt']['tokens'])
        },
        'response': {
            'text': data['response']['text'],
            'tokens': data['response']['tokens'],
            'token_ids': data['response'].get('token_ids', []),
            'n_tokens': len(data['response']['tokens'])
        },
        'metadata': {
            'inference_model': MODEL_NAME,
            'inference_experiment': args.experiment,
            'prompt_set': set_name,
            'prompt_id': prompt_id,
            'prompt_note': prompt_note,
            'capture_date': datetime.now().isoformat()
        }
    }
    with open(responses_dir / f"{prompt_id}.json", 'w') as f:
        json.dump(response_data, f, indent=2)

    # Run projections (unless --no-project)
    if not args.no_project:
        for (category, trait_name), (vector, method, vector_path, vec_metadata) in trait_vectors.items():
            prompt_proj = project_onto_vector(data['prompt']['activations'], vector, n_layers)
            response_proj = project_onto_vector(data['response']['activations'], vector, n_layers)

            # Compute dynamics on layer-averaged scores
            prompt_scores_avg = prompt_proj.mean(dim=(1, 2))
            response_scores_avg = response_proj.mean(dim=(1, 2))
            all_scores = torch.cat([prompt_scores_avg, response_scores_avg])

            proj_data = {
                'projections': {
                    'prompt': prompt_proj.tolist(),
                    'response': response_proj.tolist()
                },
                'dynamics': analyze_dynamics(all_scores),
                'metadata': {
                    'prompt_id': prompt_id,
                    'prompt_set': set_name,
                    'n_prompt_tokens': len(data['prompt']['tokens']),
                    'n_response_tokens': len(data['response']['tokens']),
                    'vector_source': {
                        'model': vec_metadata.get('extraction_model', 'unknown'),
                        'experiment': args.experiment,
                        'trait': f"{category}/{trait_name}",
                        'method': method,
                        'layer': args.layer,
                        'component': 'residual',
                    },
                    'projection_date': datetime.now().isoformat()
                }
            }

            out_dir = inference_dir / category / trait_name / "residual_stream" / set_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{prompt_id}.json", 'w') as f:
                json.dump(proj_data, f, indent=2)

    # Optional: Save attention JSON
    if args.attention and model is not None:
        analysis_dir = get_path('analysis.per_token', experiment=args.experiment, prompt_set=set_name)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if all_layer_data is not None:
            attention_data = extract_attention_for_visualization(all_layer_data, n_layers)
        else:
            attention_data = extract_attention_from_residual_capture(data, n_layers)

        attention_data["prompt_id"] = str(prompt_id)
        attention_data["prompt_set"] = set_name
        with open(analysis_dir / f"{prompt_id}_attention.json", 'w') as f:
            json.dump(attention_data, f)

    # Optional: Save logit lens JSON
    if args.logit_lens and model is not None and tokenizer is not None:
        analysis_dir = get_path('analysis.per_token', experiment=args.experiment, prompt_set=set_name)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if all_layer_data is not None:
            logit_lens_data = extract_logit_lens_for_visualization(all_layer_data, model, tokenizer, n_layers)
        else:
            logit_lens_data = extract_logit_lens_from_residual_capture(data, model, tokenizer, n_layers)

        logit_lens_data["prompt_id"] = str(prompt_id)
        logit_lens_data["prompt_set"] = set_name
        with open(analysis_dir / f"{prompt_id}_logit_lens.json", 'w') as f:
            json.dump(logit_lens_data, f)

    # Optional: Save layer internals .pt files
    if args.layer_internals is not None and all_layer_data is not None and layer_indices is not None:
        internals_dir = inference_dir / "raw" / "internals" / set_name
        internals_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layer_indices:
            torch.save(all_layer_data[layer_idx], internals_dir / f"{prompt_id}_L{layer_idx}.pt")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified activation capture")
    parser.add_argument("--experiment", required=True, help="Experiment name")

    # Optional add-ons (residual stream is always captured)
    parser.add_argument("--attention", action="store_true",
                       help="Save attention patterns as JSON for visualization (~12MB per prompt)")
    parser.add_argument("--logit-lens", action="store_true",
                       help="Save logit lens predictions as JSON for visualization (~4MB per prompt)")
    parser.add_argument("--layer-internals", metavar="N", action='append',
                       help="Save full internals as .pt files (Q/K/V, MLP, residual). "
                            "Can use 'all' for all layers or specify layer indices.")
    parser.add_argument("--capture-attn", action="store_true",
                       help="Capture raw attn_out (for attn_out vector projections)")
    parser.add_argument("--no-project", action="store_true",
                       help="Skip projection computation")
    parser.add_argument("--replay", action="store_true",
                       help="Load saved tokens from .pt file and extract attention/logit-lens "
                            "(no new generation, uses exact same tokens)")

    # Prompt input
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-set", help="Prompt set from datasets/inference/{name}.json")
    prompt_group.add_argument("--prompt", help="Single prompt string")
    prompt_group.add_argument("--all-prompt-sets", action="store_true")

    # Options
    parser.add_argument("--layer", type=int, default=16,
                       help="Layer for projection vectors (default: 16)")
    parser.add_argument("--method", help="Vector method (auto-detect if not set)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for capture (auto-detect from VRAM if not set)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of prompts to process (for testing)")

    args = parser.parse_args()

    from utils.paths import get as get_path

    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=args.experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    inference_dir = get_path('inference.base', experiment=args.experiment)

    # Get prompts from JSON files
    prompts_source = get_path('datasets.inference')
    if not prompts_source.exists():
        print(f"Prompts directory not found: {prompts_source}")
        return

    if args.prompt:
        # Ad-hoc single prompt - use id=1
        prompt_sets = [("adhoc", [{"id": 1, "text": args.prompt, "note": "ad-hoc prompt"}])]
    elif args.prompt_set:
        prompt_file = prompts_source / f"{args.prompt_set}.json"
        if not prompt_file.exists():
            print(f"Prompt set not found: {prompt_file}")
            return
        with open(prompt_file) as f:
            data = json.load(f)
        prompt_sets = [(args.prompt_set, data['prompts'])]
        print(f"Loaded {len(data['prompts'])} prompts from {args.prompt_set}")
    else:
        # Load all JSON prompt sets
        prompt_sets = []
        for f in sorted(prompts_source.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
            if 'prompts' in data and data['prompts']:
                prompt_sets.append((f.stem, data['prompts']))
        print(f"Found {len(prompt_sets)} prompt sets")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        attn_implementation='eager'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers")

    # Calculate batch size
    if args.batch_size is None:
        args.batch_size = min(8, calculate_max_batch_size(model, get_available_vram_gb()))
    print(f"Batch size: {args.batch_size}")

    # Load experiment config for chat template setting
    config = load_experiment_config(args.experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {use_chat_template}")

    # Load trait vectors (unless --no-project)
    trait_vectors = {}
    if not args.no_project:
        all_traits = discover_traits(args.experiment)
        print(f"Found {len(all_traits)} traits")

        for category, trait_name in all_traits:
            vectors_dir = get_path('extraction.vectors', experiment=args.experiment,
                                   trait=f"{category}/{trait_name}")
            method = args.method or find_vector_method(vectors_dir, args.layer)
            if not method:
                print(f"  Skip {category}/{trait_name}: no vector at layer {args.layer}")
                continue

            vector_path = vectors_dir / f"{method}_layer{args.layer}.pt"
            vector = torch.load(vector_path, weights_only=True).to(torch.float16)

            # Load vector metadata for source info
            trait_path = f"{category}/{trait_name}"
            vec_metadata = load_vector_metadata(args.experiment, trait_path)

            trait_vectors[(category, trait_name)] = (vector, method, vector_path, vec_metadata)

        print(f"Loaded {len(trait_vectors)} trait vectors")

    # Process prompts
    for set_name, prompts in prompt_sets:
        # Apply --limit if set
        if args.limit is not None:
            prompts = prompts[:args.limit]

        print(f"\n{'='*60}")
        print(f"Processing: {set_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        # ================================================================
        # REPLAY MODE: Sequential processing (no batching)
        # ================================================================
        if args.replay:
            for prompt_item in tqdm(prompts, desc="Replaying"):
                prompt_id = prompt_item['id']

                raw_pt_path = inference_dir / "raw" / "residual" / set_name / f"{prompt_id}.pt"
                if not raw_pt_path.exists():
                    print(f"  Skip {prompt_id}: no saved .pt file for replay")
                    continue

                print(f"  Loading saved tokens from {raw_pt_path}...")
                data = torch.load(raw_pt_path, weights_only=False)

                # Extract attention/logit-lens from loaded data
                analysis_dir = get_path('analysis.per_token', experiment=args.experiment, prompt_set=set_name)
                analysis_dir.mkdir(parents=True, exist_ok=True)

                if args.attention:
                    print(f"  Extracting attention patterns...")
                    attention_data = extract_attention_from_residual_capture(data, n_layers)
                    attention_data["prompt_id"] = str(prompt_id)
                    attention_data["prompt_set"] = set_name
                    with open(analysis_dir / f"{prompt_id}_attention.json", 'w') as f:
                        json.dump(attention_data, f)
                    print(f"  Saved: {analysis_dir}/{prompt_id}_attention.json")

                if args.logit_lens:
                    print(f"  Extracting logit lens predictions...")
                    logit_lens_data = extract_logit_lens_from_residual_capture(data, model, tokenizer, n_layers)
                    logit_lens_data["prompt_id"] = str(prompt_id)
                    logit_lens_data["prompt_set"] = set_name
                    with open(analysis_dir / f"{prompt_id}_logit_lens.json", 'w') as f:
                        json.dump(logit_lens_data, f)
                    print(f"  Saved: {analysis_dir}/{prompt_id}_logit_lens.json")

            continue  # Done with this prompt set

        # ================================================================
        # LAYER INTERNALS MODE: Sequential processing (specialized)
        # ================================================================
        if args.layer_internals is not None:
            # Parse layer indices once
            layer_indices = []
            for layer_spec in args.layer_internals:
                if layer_spec == 'all':
                    layer_indices = list(range(n_layers))
                    break
                else:
                    layer_indices.append(int(layer_spec))

            for prompt_item in tqdm(prompts, desc="Capturing internals"):
                prompt_id = prompt_item['id']
                raw_prompt = prompt_item['text']
                prompt_note = prompt_item.get('note', '')
                prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template)

                # Heavy capture: layer internals (includes residual)
                all_layer_data = capture_multiple_layer_internals(
                    model, tokenizer, prompt_text, layer_indices,
                    args.max_new_tokens, args.temperature
                )

                # Extract residual data from internals for projections
                data = extract_residual_from_internals(all_layer_data, n_layers)

                # Save data using the helper below
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, trait_vectors,
                    n_layers, args, get_path,
                    all_layer_data=all_layer_data, layer_indices=layer_indices,
                    model=model, tokenizer=tokenizer
                )

            continue  # Done with this prompt set

        # ================================================================
        # BATCHED CAPTURE MODE: Standard residual stream
        # ================================================================
        # Prepare prompts for batching
        prompt_texts = []
        prompt_items_filtered = []

        for prompt_item in prompts:
            prompt_id = prompt_item['id']

            # Skip existing if requested
            if args.skip_existing:
                raw_pt_path = inference_dir / "raw" / "residual" / set_name / f"{prompt_id}.pt"
                if raw_pt_path.exists():
                    continue

            raw_prompt = prompt_item['text']
            prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template)
            prompt_texts.append(prompt_text)
            prompt_items_filtered.append(prompt_item)

        if not prompt_texts:
            print("  All prompts already captured, skipping...")
            continue

        print(f"  Capturing {len(prompt_texts)} prompts in batches of {args.batch_size}...")

        # Pre-batch prompt_items to match generator output
        prompt_item_batches = [
            prompt_items_filtered[i:i+args.batch_size]
            for i in range(0, len(prompt_items_filtered), args.batch_size)
        ]

        # Run batched capture with incremental saving (generator mode)
        batch_generator = generate_with_capture(
            model, tokenizer, prompt_texts,
            n_layers=n_layers,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            capture_attn=args.capture_attn,
            show_progress=True,
            yield_per_batch=True
        )

        # Process and save after each batch (crash-resilient)
        for (batch_results, _batch_prompts), batch_items in zip(batch_generator, prompt_item_batches):
            for result, prompt_item in zip(batch_results, batch_items):
                # Convert CaptureResult to legacy data format
                data = capture_result_to_data(result, n_layers)

                # Save data immediately
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, trait_vectors,
                    n_layers, args, get_path,
                    all_layer_data=None, layer_indices=None,
                    model=model, tokenizer=tokenizer
                )

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"\nOutput locations:")
    print(f"  Raw activations: {inference_dir}/raw/residual/{{prompt_set}}/")
    print(f"  Responses:       {inference_dir}/responses/{{prompt_set}}/")
    print(f"  Projections:     {inference_dir}/{{category}}/{{trait}}/residual_stream/")


if __name__ == "__main__":
    main()
