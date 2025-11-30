#!/usr/bin/env python3
"""
Unified activation capture for trait analysis.

Captures raw activations and/or computes projections onto trait vectors.

Modes:
  --residual-stream    : Capture residual stream at all layers (default)
  --capture-attn       : Also capture raw attn_out (attention output before residual addition)
  --layer-internals N  : Capture full internals for layer N (Q/K/V, MLP, attention)
                         Automatically extracts attention + logit lens for visualization

Storage:
  Raw activations      : experiments/{exp}/inference/raw/residual/{prompt_set}/{id}.pt
  Layer internals      : experiments/{exp}/inference/raw/internals/{prompt_set}/{id}_L{layer}.pt
  Attention JSON       : experiments/{exp}/analysis/per_token/{prompt_set}/{id}_attention.json
  Logit lens JSON      : experiments/{exp}/analysis/per_token/{prompt_set}/{id}_logit_lens.json
  Residual stream JSON : experiments/{exp}/inference/{category}/{trait}/residual_stream/{prompt_set}/{id}.json

Usage:
    # Capture residual stream + project onto all traits
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # Capture ALL layer internals + attention + logit lens (for visualization)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set dynamic \\
        --layer-internals all

    # Capture specific layers only
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt "How do I make a bomb?" \\
        --layer-internals 0 --layer-internals 16 --layer-internals 25

    # Just capture raw activations, no projections
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --no-project

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
# Residual Stream Capture
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
# Serialization
# ============================================================================

def tensor_to_list(obj):
    """Recursively convert tensors to lists."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, list):
        return [tensor_to_list(x) for x in obj]
    if isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    return obj


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified activation capture")
    parser.add_argument("--experiment", required=True, help="Experiment name")

    # What to capture
    parser.add_argument("--residual-stream", action="store_true", default=True,
                       help="Capture residual stream at all layers (default)")
    parser.add_argument("--capture-attn", action="store_true",
                       help="Also capture raw attn_out (attention output before residual addition)")
    parser.add_argument("--layer-internals", metavar="N", action='append',
                       help="Capture full internals for layer N (can be used multiple times, or 'all' for all layers). "
                            "Automatically extracts attention + logit lens for visualization.")
    parser.add_argument("--no-project", action="store_true",
                       help="Skip projection computation")

    # Prompt input
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-set", help="Prompt set from inference/prompts/{name}.txt")
    prompt_group.add_argument("--prompt", help="Single prompt string")
    prompt_group.add_argument("--all-prompt-sets", action="store_true")

    # Options
    parser.add_argument("--layer", type=int, default=16,
                       help="Layer for projection vectors (default: 16)")
    parser.add_argument("--method", help="Vector method (auto-detect if not set)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    from utils.paths import get as get_path

    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=args.experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    inference_dir = get_path('inference.base', experiment=args.experiment)

    # Get prompts from JSON files
    prompts_source = Path(__file__).parent / "prompts"
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
            trait_vectors[(category, trait_name)] = (vector, method, vector_path)

        print(f"Loaded {len(trait_vectors)} trait vectors")

    # Process prompts
    for set_name, prompts in prompt_sets:
        print(f"\n{'='*60}")
        print(f"Processing: {set_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        for prompt_item in tqdm(prompts, desc="Capturing"):
            prompt_id = prompt_item['id']
            prompt_text = prompt_item['text']
            prompt_note = prompt_item.get('note', '')

            # Capture layer internals (can be multiple)
            if args.layer_internals is not None:
                raw_dir = inference_dir / "raw" / "internals" / set_name
                raw_dir.mkdir(parents=True, exist_ok=True)

                # Parse layer indices: handle "all" or specific numbers
                layer_indices = []
                for layer_spec in args.layer_internals:
                    if layer_spec == 'all':
                        layer_indices = list(range(n_layers))
                        break
                    else:
                        layer_indices.append(int(layer_spec))

                # Capture all requested layers in a SINGLE forward pass!
                print(f"  Capturing {len(layer_indices)} layers in single pass...")
                all_layer_data = capture_multiple_layer_internals(
                    model, tokenizer, prompt_text, layer_indices,
                    args.max_new_tokens, args.temperature
                )

                # Save each layer's data as .pt
                for layer_idx in layer_indices:
                    torch.save(all_layer_data[layer_idx], raw_dir / f"{prompt_id}_L{layer_idx}.pt")

                print(f"  Saved internals for {len(layer_indices)} layers: {raw_dir}/{prompt_id}_L*.pt")

                # Extract attention and logit lens for visualization (automatic)
                analysis_dir = get_path('analysis.per_token', experiment=args.experiment, prompt_set=set_name)
                analysis_dir.mkdir(parents=True, exist_ok=True)

                # Extract attention
                print(f"  Extracting attention patterns...")
                attention_data = extract_attention_for_visualization(all_layer_data, n_layers)
                attention_data["prompt_id"] = str(prompt_id)
                attention_data["prompt_set"] = set_name
                with open(analysis_dir / f"{prompt_id}_attention.json", 'w') as f:
                    json.dump(attention_data, f)

                # Extract logit lens
                print(f"  Extracting logit lens predictions...")
                logit_lens_data = extract_logit_lens_for_visualization(all_layer_data, model, tokenizer, n_layers)
                logit_lens_data["prompt_id"] = str(prompt_id)
                logit_lens_data["prompt_set"] = set_name
                with open(analysis_dir / f"{prompt_id}_logit_lens.json", 'w') as f:
                    json.dump(logit_lens_data, f)

                print(f"  Saved: {analysis_dir}/{prompt_id}_attention.json, {prompt_id}_logit_lens.json")

            # Capture residual stream
            elif args.residual_stream or not args.layer_internals:
                data = capture_residual_stream(model, tokenizer, prompt_text, n_layers,
                                               args.max_new_tokens, args.temperature,
                                               capture_attn=args.capture_attn)

                # Save raw residual as .pt
                raw_dir = inference_dir / "raw" / "residual" / set_name
                raw_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data, raw_dir / f"{prompt_id}.pt")

                # Project onto each trait (unless --no-project)
                if not args.no_project:
                    for (category, trait_name), (vector, method, vector_path) in trait_vectors.items():
                        prompt_proj = project_onto_vector(data['prompt']['activations'], vector, n_layers)
                        response_proj = project_onto_vector(data['response']['activations'], vector, n_layers)

                        # Compute dynamics on layer-averaged scores
                        prompt_scores_avg = prompt_proj.mean(dim=(1, 2))  # Average across layers and sublayers
                        response_scores_avg = response_proj.mean(dim=(1, 2))
                        all_scores = torch.cat([prompt_scores_avg, response_scores_avg])

                        proj_data = {
                            'prompt': {
                                'text': data['prompt']['text'],
                                'tokens': data['prompt']['tokens'],
                                'token_ids': data['prompt']['token_ids'],
                                'n_tokens': len(data['prompt']['tokens'])
                            },
                            'response': {
                                'text': data['response']['text'],
                                'tokens': data['response']['tokens'],
                                'token_ids': data['response']['token_ids'],
                                'n_tokens': len(data['response']['tokens'])
                            },
                            'projections': {
                                'prompt': prompt_proj.tolist(),
                                'response': response_proj.tolist()
                            },
                            'dynamics': analyze_dynamics(all_scores),
                            'attention_weights': {
                                'prompt': tensor_to_list(data['prompt']['attention']),
                                'response': tensor_to_list(data['response']['attention'])
                            },
                            'metadata': {
                                'prompt_id': prompt_id,
                                'prompt_set': set_name,
                                'prompt_note': prompt_note,
                                'trait': trait_name,
                                'category': category,
                                'method': method,
                                'layer': args.layer,
                                'vector_path': str(vector_path),
                                'model': MODEL_NAME,
                                'capture_date': datetime.now().isoformat()
                            }
                        }

                        # Save projection JSON to residual_stream/{prompt_set}/{id}.json
                        out_dir = inference_dir / category / trait_name / "residual_stream" / set_name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_dir / f"{prompt_id}.json", 'w') as f:
                            json.dump(proj_data, f, indent=2)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"\nOutput locations:")
    print(f"  Raw:            {inference_dir}/raw/")
    print(f"  Residual stream: {inference_dir}/{{category}}/{{trait}}/residual_stream/")


if __name__ == "__main__":
    main()
