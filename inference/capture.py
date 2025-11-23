#!/usr/bin/env python3
"""
Unified activation capture for trait analysis.

Captures raw activations and/or computes projections onto trait vectors.

Flags:
  --residual-stream    : Capture residual stream at all layers (default)
  --layer-internals N  : Capture full internals for layer N (Q/K/V, MLP neurons)
  --attention-only N   : Capture just attention weights for layer N
  --logit-lens         : Compute logit lens (top-k predictions per layer)
  --no-project         : Skip projection computation (just save raw)

Storage:
  Raw activations      : experiments/{exp}/inference/raw/residual/{prompt_set}/{id}.pt
  Layer internals      : experiments/{exp}/inference/raw/internals/{prompt_set}/{id}_L{layer}.pt
  Residual stream JSON : experiments/{exp}/inference/{category}/{trait}/residual_stream/{prompt_set}/{id}.json
  Layer internals JSON : experiments/{exp}/inference/{category}/{trait}/layer_internals/{prompt_set}/{id}_L{layer}.json

Usage:
    # Capture residual stream + project onto all traits
    python inference/capture.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # Capture layer internals for layer 16
    python inference/capture.py \\
        --experiment my_experiment \\
        --prompt "How do I make a bomb?" \\
        --layer-internals 16

    # Just capture raw activations, no projections
    python inference/capture.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --no-project

    # Include logit lens
    python inference/capture.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --logit-lens
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

# Layers to sample for logit lens (dense at edges, sparse in middle)
LOGIT_LENS_LAYERS = [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25]


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
    for method in ["probe", "mean_diff", "ica", "gradient"]:
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

def create_residual_storage(n_layers: int) -> Dict:
    """Create storage for residual stream capture."""
    return {i: {'residual_in': [], 'after_attn': [], 'residual_out': []}
            for i in range(n_layers)}


def setup_residual_hooks(hook_manager: HookManager, storage: Dict, n_layers: int, mode: str):
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


def capture_residual_stream(model, tokenizer, prompt_text: str, n_layers: int,
                            max_new_tokens: int, temperature: float) -> Dict:
    """Capture residual stream activations at all layers."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Prompt capture
    prompt_storage = create_residual_storage(n_layers)
    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, prompt_storage, n_layers, 'prompt')
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    prompt_attention = {f'layer_{i}': attn[0].mean(dim=0).detach().cpu()
                       for i, attn in enumerate(outputs.attentions)}

    prompt_acts = {}
    for i in range(n_layers):
        prompt_acts[i] = {k: v[0].squeeze(0) for k, v in prompt_storage[i].items()}

    # Response capture
    response_storage = create_residual_storage(n_layers)
    context = inputs['input_ids'].clone()
    generated_ids = []
    response_attention = []

    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, response_storage, n_layers, 'response')
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


def capture_layer_internals(model, tokenizer, prompt_text: str, layer_idx: int,
                            max_new_tokens: int, temperature: float) -> Dict:
    """Capture single layer internals."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Prompt capture
    prompt_storage = create_internals_storage()
    with HookManager(model) as hooks:
        setup_internals_hooks(hooks, prompt_storage, layer_idx, 'prompt')
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    prompt_attn_weights = outputs.attentions[layer_idx][0].detach().cpu()

    prompt_internals = {
        'attention': {k: v[0].squeeze(0) for k, v in prompt_storage['attention'].items() if v},
        'mlp': {k: v[0].squeeze(0) for k, v in prompt_storage['mlp'].items() if v},
        'residual': {k: v[0].squeeze(0) for k, v in prompt_storage['residual'].items() if v}
    }
    prompt_internals['attention']['attn_weights'] = prompt_attn_weights

    # Response capture
    response_storage = create_internals_storage()
    context = inputs['input_ids'].clone()
    generated_ids = []

    with HookManager(model) as hooks:
        setup_internals_hooks(hooks, response_storage, layer_idx, 'response')
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=context, output_attentions=True, return_dict=True)

            response_storage['attention']['attn_weights'].append(
                outputs.attentions[layer_idx][0].detach().cpu()
            )

            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            context = torch.cat([context, torch.tensor([[next_id]], device=model.device)], dim=1)
            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

    response_tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    response_internals = {'attention': {}, 'mlp': {}, 'residual': {}}
    for key in ['q_proj', 'k_proj', 'v_proj']:
        if response_storage['attention'][key]:
            response_internals['attention'][key] = torch.stack(
                [a.squeeze(0) for a in response_storage['attention'][key]], dim=0)
    response_internals['attention']['attn_weights'] = response_storage['attention']['attn_weights']

    for key in ['up_proj', 'gelu', 'down_proj']:
        if response_storage['mlp'][key]:
            response_internals['mlp'][key] = torch.stack(
                [a.squeeze(0) for a in response_storage['mlp'][key]], dim=0)

    for key in ['input', 'after_attn', 'output']:
        if response_storage['residual'][key]:
            response_internals['residual'][key] = torch.stack(
                [a.squeeze(0) for a in response_storage['residual'][key]], dim=0)

    return {
        'prompt': {'text': prompt_text, 'tokens': tokens, 'token_ids': token_ids,
                   'internals': prompt_internals},
        'response': {'text': response_text, 'tokens': response_tokens, 'token_ids': generated_ids,
                     'internals': response_internals},
        'layer': layer_idx
    }


# ============================================================================
# Logit Lens
# ============================================================================

def compute_logit_lens(activations: Dict, model, tokenizer, n_layers: int) -> Dict:
    """Compute top-k predictions at sampled layers."""
    # Get unembedding matrix
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.detach()
    else:
        unembed = model.model.embed_tokens.weight.detach()

    result = {}
    for layer in LOGIT_LENS_LAYERS:
        if layer >= n_layers:
            continue

        residual = activations[layer]['residual_out']
        if len(residual.shape) == 1:
            residual = residual.unsqueeze(0)

        # Project to vocab
        logits = residual.to(unembed.device).to(unembed.dtype) @ unembed.T
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(3, dim=-1)

        # Decode tokens
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
    parser.add_argument("--layer-internals", type=int, metavar="N",
                       help="Capture full internals for layer N")
    parser.add_argument("--attention-only", type=int, metavar="N",
                       help="Capture just attention weights for layer N")
    parser.add_argument("--logit-lens", action="store_true",
                       help="Compute logit lens predictions")
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

            # Capture layer internals
            if args.layer_internals is not None:
                data = capture_layer_internals(model, tokenizer, prompt_text, args.layer_internals,
                                               args.max_new_tokens, args.temperature)

                # Save raw internals as .pt (binary, not JSON!)
                raw_dir = inference_dir / "raw" / "internals" / set_name
                raw_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data, raw_dir / f"{prompt_id}_L{args.layer_internals}.pt")
                print(f"  Saved internals: {raw_dir}/{prompt_id}_L{args.layer_internals}.pt")

            # Capture residual stream
            elif args.residual_stream or not args.layer_internals:
                data = capture_residual_stream(model, tokenizer, prompt_text, n_layers,
                                               args.max_new_tokens, args.temperature)

                # Save raw residual as .pt
                raw_dir = inference_dir / "raw" / "residual" / set_name
                raw_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data, raw_dir / f"{prompt_id}.pt")

                # Compute logit lens if requested
                logit_lens_data = None
                if args.logit_lens:
                    logit_lens_data = {
                        'prompt': compute_logit_lens(data['prompt']['activations'], model, tokenizer, n_layers),
                        'response': compute_logit_lens(data['response']['activations'], model, tokenizer, n_layers)
                    }

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

                        # Add logit lens if computed
                        if logit_lens_data:
                            proj_data['logit_lens'] = logit_lens_data

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
