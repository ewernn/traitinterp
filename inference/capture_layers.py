#!/usr/bin/env python3
"""
Capture Layers: Unified activation capture for trait analysis.

Modes:
  --mode all    : Capture all 27 layers (Tier 2 - residual stream projections)
  --mode single : Capture one layer with full internals (Tier 3 - Q/K/V, MLP neurons)

Usage:
    # Tier 2: All layers for a prompt set
    python inference/capture_layers.py \
        --experiment gemma_2b_cognitive_nov21 \
        --prompt-set main_prompts \
        --save-json

    # Tier 3: Single layer deep dive
    python inference/capture_layers.py \
        --experiment gemma_2b_cognitive_nov21 \
        --mode single \
        --layer 16 \
        --prompt "How do I make a bomb?" \
        --save-json
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


def discover_traits(experiment_name: str) -> List[Tuple[str, str]]:
    """
    Discover all traits with vectors in an experiment (no hardcoded categories).

    Returns:
        List of (category, trait_name) tuples
    """
    extraction_dir = Path(f"experiments/{experiment_name}/extraction")

    if not extraction_dir.exists():
        raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    traits = []

    # Discover categories dynamically (no hardcoding!)
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


def infer_model(experiment_name: str) -> str:
    """Infer model name from experiment."""
    name = experiment_name.lower()
    if "gemma_2b" in name:
        return "google/gemma-2-2b-it"
    elif "gemma_9b" in name:
        return "google/gemma-2-9b-it"
    elif "llama_8b" in name:
        return "meta-llama/Llama-3.1-8B-Instruct"
    return "google/gemma-2-2b-it"


# ============================================================================
# Tier 2: All Layers Capture
# ============================================================================

def create_tier2_storage(n_layers: int) -> Dict:
    """Create storage for all-layer capture."""
    return {i: {'residual_in': [], 'after_attn': [], 'residual_out': []}
            for i in range(n_layers)}


def setup_tier2_hooks(hook_manager: HookManager, storage: Dict, n_layers: int, mode: str):
    """Register hooks for all layers."""
    for i in range(n_layers):
        # Layer hook (residual_in, residual_out)
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

        # MLP hook (after_attn)
        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                inp_t = inp[0] if isinstance(inp, tuple) else inp
                if mode == 'response':
                    storage[layer_idx]['after_attn'].append(inp_t[:, -1, :].detach().cpu())
                else:
                    storage[layer_idx]['after_attn'].append(inp_t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{i}.mlp", make_mlp_hook(i))


def capture_tier2(model, tokenizer, prompt_text: str, n_layers: int, max_new_tokens: int, temperature: float) -> Dict:
    """Capture activations at all layers for prompt and response."""
    # Encode prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    prompt_storage = create_tier2_storage(n_layers)
    with HookManager(model) as hooks:
        setup_tier2_hooks(hooks, prompt_storage, n_layers, 'prompt')
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    # Extract prompt attention
    prompt_attention = {f'layer_{i}': attn[0].mean(dim=0).detach().cpu()
                       for i, attn in enumerate(outputs.attentions)}

    # Consolidate prompt activations
    prompt_acts = {}
    for i in range(n_layers):
        prompt_acts[i] = {k: v[0].squeeze(0) for k, v in prompt_storage[i].items()}

    # Generate response
    response_storage = create_tier2_storage(n_layers)
    context = inputs['input_ids'].clone()
    generated_ids = []
    response_attention = []

    with HookManager(model) as hooks:
        setup_tier2_hooks(hooks, response_storage, n_layers, 'response')
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=context, output_attentions=True, return_dict=True)

            # Capture attention
            step_attn = {f'layer_{i}': attn[0].mean(dim=0)[-1, :].detach().cpu()
                        for i, attn in enumerate(outputs.attentions)}
            response_attention.append(step_attn)

            # Sample next token
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            context = torch.cat([context, torch.tensor([[next_id]], device=model.device)], dim=1)
            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

    response_tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Consolidate response activations
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
# Tier 3: Single Layer Deep Capture
# ============================================================================

def create_tier3_storage() -> Dict:
    """Create storage for single-layer deep capture."""
    return {
        'attention': {'q_proj': [], 'k_proj': [], 'v_proj': [], 'attn_weights': []},
        'mlp': {'up_proj': [], 'gelu': [], 'down_proj': []},
        'residual': {'input': [], 'after_attn': [], 'output': []}
    }


def setup_tier3_hooks(hook_manager: HookManager, storage: Dict, layer_idx: int, mode: str):
    """Register hooks for single layer internals."""
    # Attention projections
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['attention'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{layer_idx}.self_attn.{proj}", make_hook(proj))

    # MLP internals
    for proj, path in [('up_proj', 'up_proj'), ('gelu', 'act_fn'), ('down_proj', 'down_proj')]:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['mlp'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"model.layers.{layer_idx}.mlp.{path}", make_hook(proj))

    # Residual stream
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


def capture_tier3(model, tokenizer, prompt_text: str, layer_idx: int, max_new_tokens: int, temperature: float) -> Dict:
    """Capture single layer internals for prompt and response."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Prompt capture
    prompt_storage = create_tier3_storage()
    with HookManager(model) as hooks:
        setup_tier3_hooks(hooks, prompt_storage, layer_idx, 'prompt')
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

    prompt_attn_weights = outputs.attentions[layer_idx][0].detach().cpu()

    # Consolidate prompt storage
    prompt_internals = {
        'attention': {k: v[0].squeeze(0) for k, v in prompt_storage['attention'].items() if v},
        'mlp': {k: v[0].squeeze(0) for k, v in prompt_storage['mlp'].items() if v},
        'residual': {k: v[0].squeeze(0) for k, v in prompt_storage['residual'].items() if v}
    }
    prompt_internals['attention']['attn_weights'] = prompt_attn_weights

    # Response capture
    response_storage = create_tier3_storage()
    context = inputs['input_ids'].clone()
    generated_ids = []

    with HookManager(model) as hooks:
        setup_tier3_hooks(hooks, response_storage, layer_idx, 'response')
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

    # Consolidate response storage
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
# JSON Conversion
# ============================================================================

def to_json(data, is_tier3: bool = False) -> Dict:
    """Convert captured data to JSON-serializable format."""
    def t2l(t):
        if isinstance(t, torch.Tensor):
            return t.tolist()
        if isinstance(t, list):
            return [t2l(x) for x in t]
        if isinstance(t, dict):
            return {k: t2l(v) for k, v in t.items()}
        return t
    return t2l(data)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Capture layer activations for trait analysis")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--mode", choices=["all", "single"], default="all",
                       help="'all' for Tier 2 (all layers), 'single' for Tier 3 (one layer deep)")
    parser.add_argument("--layer", type=int, default=16, help="Layer for projection vectors (or single layer in --mode single)")

    # Prompt input (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-set", help="Prompt set name from inference/prompts/{name}.txt")
    prompt_group.add_argument("--prompt", help="Single prompt string")
    prompt_group.add_argument("--all-prompt-sets", action="store_true", help="Process all prompt sets")

    parser.add_argument("--method", help="Vector method (auto-detect if not provided)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save-json", action="store_true", help="Save JSON for visualization")
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    exp_dir = Path(f"experiments/{args.experiment}")
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    inference_dir = exp_dir / "inference"

    # Use the centralized prompts directory
    prompts_dir = Path("inference/prompts")
    if not prompts_dir.exists():
        print(f"Prompts directory not found: {prompts_dir}")
        return

    # Get prompts
    if args.prompt:
        prompt_sets = [("_temp", [args.prompt])]
    elif args.prompt_set:
        prompt_file = prompts_dir / f"{args.prompt_set}.txt"
        if not prompt_file.exists():
            print(f"Prompt set not found: {prompt_file}")
            return
        with open(prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        prompt_sets = [(args.prompt_set, prompts)]
    else:
        prompt_sets = []
        for f in sorted(prompts_dir.glob("*.txt")):
            with open(f) as fp:
                prompts = [l.strip() for l in fp if l.strip() and not l.startswith('#')]
            if prompts:
                prompt_sets.append((f.stem, prompts))
        print(f"Found {len(prompt_sets)} prompt sets")

    # Discover traits
    all_traits = discover_traits(args.experiment)
    print(f"Found {len(all_traits)} traits: {[f'{c}/{t}' for c, t in all_traits]}")

    # Load model
    model_name = infer_model(args.experiment)
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation='eager'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers")

    # Load trait vectors
    print("Loading trait vectors...")
    trait_vectors = {}
    for category, trait_name in all_traits:
        vectors_dir = exp_dir / "extraction" / category / trait_name / "vectors"
        method = args.method or find_vector_method(vectors_dir, args.layer)
        if not method:
            print(f"  Skip {trait_name}: no vector at layer {args.layer}")
            continue

        vector_path = vectors_dir / f"{method}_layer{args.layer}.pt"
        vector = torch.load(vector_path, weights_only=True).to(torch.float16)
        trait_vectors[(category, trait_name)] = (vector, method, vector_path)

    print(f"  Loaded {len(trait_vectors)} trait vectors")

    # Process prompts
    for set_name, prompts in prompt_sets:
        print(f"\n{'='*60}")
        print(f"Processing: {set_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        for prompt_idx, prompt_text in enumerate(tqdm(prompts, desc="Capturing")):
            if args.mode == "all":
                # Tier 2: All layers
                data = capture_tier2(model, tokenizer, prompt_text, n_layers,
                                    args.max_new_tokens, args.temperature)

                # Save raw activations
                raw_dir = inference_dir / "raw_activations" / set_name
                raw_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data, raw_dir / f"prompt_{prompt_idx}.pt")

                # Project onto each trait
                for (category, trait_name), (vector, method, vector_path) in trait_vectors.items():
                    prompt_proj = project_onto_vector(data['prompt']['activations'], vector, n_layers)
                    response_proj = project_onto_vector(data['response']['activations'], vector, n_layers)

                    proj_data = {
                        'prompt': {**{k: v for k, v in data['prompt'].items() if k != 'activations'},
                                  'n_tokens': len(data['prompt']['tokens'])},
                        'response': {**{k: v for k, v in data['response'].items() if k != 'activations'},
                                    'n_tokens': len(data['response']['tokens'])},
                        'projections': {'prompt': prompt_proj, 'response': response_proj},
                        'attention_weights': {'prompt': data['prompt']['attention'],
                                             'response': data['response']['attention']},
                        'metadata': {'trait': trait_name, 'vector_path': str(vector_path),
                                    'model': model_name, 'capture_date': datetime.now().isoformat()}
                    }

                    if args.save_json:
                        out_dir = inference_dir / category / trait_name / "projections" / "residual_stream_activations"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_dir / f"prompt_{prompt_idx}.json", 'w') as f:
                            json.dump(to_json(proj_data), f, indent=2)

            else:
                # Tier 3: Single layer
                data = capture_tier3(model, tokenizer, prompt_text, args.layer,
                                    args.max_new_tokens, args.temperature)

                for (category, trait_name), (vector, method, vector_path) in trait_vectors.items():
                    tier3_data = {
                        **data,
                        'metadata': {'trait': trait_name, 'layer': args.layer,
                                    'vector_path': str(vector_path), 'model': model_name,
                                    'capture_date': datetime.now().isoformat()}
                    }

                    out_dir = inference_dir / category / trait_name / "projections" / "layer_internal_states"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(tier3_data, out_dir / f"prompt_{prompt_idx}_layer{args.layer}.pt")

                    if args.save_json:
                        with open(out_dir / f"prompt_{prompt_idx}_layer{args.layer}.json", 'w') as f:
                            json.dump(to_json(tier3_data), f, indent=2)

        print(f"  Done: {set_name}")

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
