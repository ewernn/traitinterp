#!/usr/bin/env python3
"""
Capture raw activations from model inference.

Runs model inference and saves raw activations as .pt files. Projections onto trait
vectors are computed separately via project_raw_activations_onto_traits.py.

Output files:
  Raw activations : experiments/{exp}/inference/raw/residual/{prompt_set}/{id}.pt
  Responses       : experiments/{exp}/inference/responses/{prompt_set}/{id}.json
  Layer internals : experiments/{exp}/inference/raw/internals/{prompt_set}/{id}_L{layer}.pt

.pt file structure (model-agnostic):
  {
    'prompt': {
      'text': str,
      'tokens': List[str],
      'token_ids': List[int],
      'activations': {
        layer_idx: {
          'residual': Tensor[n_tokens, hidden_dim],  # layer output
          'attn_out': Tensor[n_tokens, hidden_dim],  # attention contribution
          'mlp_out': Tensor[n_tokens, hidden_dim],   # if --capture-mlp
        }
      }
    },
    'response': { ... }  # same structure
  }

Usage:
    # Basic capture (residual + attn_out)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # With mlp_out for component decomposition
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set harmful \\
        --capture-mlp

    # Deep capture with full layer internals (Q/K/V, MLP)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set dynamic \\
        --layer-internals all

    # Model-diff: run same responses through different model
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set dynamic \\
        --replay-responses other_prompt_set

    # Then compute projections
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts
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

from core import HookManager
from utils.model import format_prompt, tokenize_prompt, load_experiment_config, load_model_with_lora, get_inner_model, get_layer_path_prefix
from utils.generation import generate_with_capture, calculate_max_batch_size, create_residual_storage, setup_residual_hooks
from utils.paths import get as get_path
from server.client import get_model_or_client, ModelClient


def capture_result_to_data(result, n_layers: int) -> Dict:
    """Convert CaptureResult from utils.generation to dict format for saving."""
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
        },
        'response': {
            'text': result.response_text,
            'tokens': result.response_tokens,
            'token_ids': result.response_token_ids,
            'activations': response_acts,
        }
    }


def extract_residual_from_internals(all_layer_data: Dict[int, Dict], n_layers: int) -> Dict:
    """Convert layer internals format to standard residual format for saving."""
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_acts = {}
    response_acts = {}
    for layer_idx in range(n_layers):
        if layer_idx in all_layer_data:
            layer = all_layer_data[layer_idx]
            prompt_acts[layer_idx] = {
                'attn_out': layer['prompt']['residual'].get('attn_out', torch.empty(0)),
                'residual': layer['prompt']['residual'].get('residual', torch.empty(0))
            }
            response_acts[layer_idx] = {
                'attn_out': layer['response']['residual'].get('attn_out', torch.empty(0)),
                'residual': layer['response']['residual'].get('residual', torch.empty(0))
            }
        else:
            prompt_acts[layer_idx] = {'attn_out': torch.empty(0), 'residual': torch.empty(0)}
            response_acts[layer_idx] = {'attn_out': torch.empty(0), 'residual': torch.empty(0)}

    return {
        'prompt': {
            'text': layer_data['prompt_text'],
            'tokens': layer_data['prompt_tokens'],
            'token_ids': [],
            'activations': prompt_acts
        },
        'response': {
            'text': layer_data['response_text'],
            'tokens': layer_data['response_tokens'],
            'token_ids': [],
            'activations': response_acts
        }
    }


def capture_residual_stream_prefill(model, tokenizer, prompt_text: str, response_text: str,
                                     n_layers: int, capture_mlp: bool = False) -> Dict:
    """
    Capture residual stream activations with prefilled response (single forward pass).

    Used for model-diff analysis: run same text through different models.
    """
    # Tokenize prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_prompt_tokens = prompt_inputs['input_ids'].shape[1]
    prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (without special tokens)
    response_inputs = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    response_token_ids = response_inputs['input_ids'][0].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    # Concatenate for single forward pass
    full_input_ids = torch.cat([prompt_inputs['input_ids'], response_inputs['input_ids']], dim=1)

    # Capture all activations in one pass
    layer_prefix = get_layer_path_prefix(model)
    storage = create_residual_storage(n_layers, capture_mlp=capture_mlp)
    with HookManager(model) as hooks:
        setup_residual_hooks(hooks, storage, n_layers, 'prompt', capture_mlp=capture_mlp, layer_prefix=layer_prefix)
        with torch.no_grad():
            outputs = model(input_ids=full_input_ids, output_attentions=True, return_dict=True)

    # Split activations into prompt/response portions
    prompt_acts = {}
    response_acts = {}
    for i in range(n_layers):
        prompt_acts[i] = {}
        response_acts[i] = {}
        for k, v in storage[i].items():
            full_acts = v[0].squeeze(0)  # [seq_len, hidden_dim]
            prompt_acts[i][k] = full_acts[:n_prompt_tokens]
            response_acts[i][k] = full_acts[n_prompt_tokens:]

    # Split attention patterns
    prompt_attention = {}
    response_attention = []
    for i, attn in enumerate(outputs.attentions):
        attn_avg = attn[0].mean(dim=0).detach().cpu()  # [seq_len, seq_len]
        prompt_attention[f'layer_{i}'] = attn_avg[:n_prompt_tokens, :n_prompt_tokens]
        # Response attention: each token's attention over full context
        for t in range(n_prompt_tokens, attn_avg.shape[0]):
            if i == 0:
                response_attention.append({})
            response_attention[t - n_prompt_tokens][f'layer_{i}'] = attn_avg[t, :t+1]

    return {
        'prompt': {'text': prompt_text, 'tokens': prompt_tokens, 'token_ids': prompt_token_ids,
                   'activations': prompt_acts, 'attention': prompt_attention},
        'response': {'text': response_text, 'tokens': response_tokens, 'token_ids': response_token_ids,
                     'activations': response_acts, 'attention': response_attention}
    }


def create_internals_storage() -> Dict:
    """Create storage for single-layer deep capture."""
    return {
        'attention': {'q_proj': [], 'k_proj': [], 'v_proj': [], 'attn_weights': []},
        'mlp': {'up_proj': [], 'gelu': [], 'down_proj': []},
        'residual': {'input': [], 'attn_out': [], 'residual': []}
    }


def setup_internals_hooks(hook_manager: HookManager, storage: Dict, layer_idx: int, mode: str,
                          layer_prefix: str = "model.layers"):
    """Register hooks for single layer internals."""
    # Attention projections (Q/K/V)
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['attention'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"{layer_prefix}.{layer_idx}.self_attn.{proj}", make_hook(proj))

    # MLP stages
    for proj, path in [('up_proj', 'up_proj'), ('gelu', 'act_fn'), ('down_proj', 'down_proj')]:
        def make_hook(key):
            def hook(module, inp, out):
                t = out[:, -1, :] if mode == 'response' else out
                storage['mlp'][key].append(t.detach().cpu())
            return hook
        hook_manager.add_forward_hook(f"{layer_prefix}.{layer_idx}.mlp.{path}", make_hook(proj))

    # Layer input/output (residual stream)
    def layer_hook(module, inp, out):
        inp_t = inp[0] if isinstance(inp, tuple) else inp
        out_t = out[0] if isinstance(out, tuple) else out
        if mode == 'response':
            storage['residual']['input'].append(inp_t[:, -1, :].detach().cpu())
            storage['residual']['residual'].append(out_t[:, -1, :].detach().cpu())
        else:
            storage['residual']['input'].append(inp_t.detach().cpu())
            storage['residual']['residual'].append(out_t.detach().cpu())
    hook_manager.add_forward_hook(f"{layer_prefix}.{layer_idx}", layer_hook)

    # Attention output (o_proj)
    def attn_out_hook(module, inp, out):
        t = out[:, -1, :] if mode == 'response' else out
        storage['residual']['attn_out'].append(t.detach().cpu())
    hook_manager.add_forward_hook(f"{layer_prefix}.{layer_idx}.self_attn.o_proj", attn_out_hook)


def capture_multiple_layer_internals(model, tokenizer, prompt_text: str, layer_indices: list,
                                     max_new_tokens: int, temperature: float,
                                     use_chat_template: bool = None) -> Dict[int, Dict]:
    """Capture internals for multiple layers in a SINGLE forward pass."""
    inputs = tokenize_prompt(prompt_text, tokenizer, use_chat_template).to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    layer_prefix = get_layer_path_prefix(model)

    # Storage for all requested layers
    all_layer_data = {}

    # PROMPT PHASE - Single forward pass for all layers
    prompt_storages = {idx: create_internals_storage() for idx in layer_indices}

    with HookManager(model) as hooks:
        # Set up hooks for ALL requested layers
        for layer_idx in layer_indices:
            setup_internals_hooks(hooks, prompt_storages[layer_idx], layer_idx, 'prompt', layer_prefix)

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
                setup_internals_hooks(hooks, response_storages[layer_idx], layer_idx, 'response', layer_prefix)

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
        # Skip first capture (duplicates prompt[-1], captured before any token generated)
        for k, v in response_storages[layer_idx]['attention'].items():
            if k == 'attn_weights':
                # Keep as list of tensors with different sizes
                response_internals['attention'][k] = v[1:] if len(v) > 1 else []
            elif len(v) > 1:
                # Other attention tensors should be same size, can stack
                response_internals['attention'][k] = torch.stack(v[1:])
            else:
                response_internals['attention'][k] = torch.tensor([])

        # MLP tensors are all same size, can stack (skip first capture)
        for k, v in response_storages[layer_idx]['mlp'].items():
            if len(v) > 1:
                response_internals['mlp'][k] = torch.stack(v[1:])
            else:
                response_internals['mlp'][k] = torch.tensor([])

        # Residual tensors are all same size, can stack (skip first capture)
        for k, v in response_storages[layer_idx]['residual'].items():
            if len(v) > 1:
                response_internals['residual'][k] = torch.stack(v[1:])
            else:
                response_internals['residual'][k] = torch.tensor([])

        all_layer_data[layer_idx]['response'] = response_internals
        all_layer_data[layer_idx]['prompt_text'] = prompt_text
        all_layer_data[layer_idx]['prompt_tokens'] = tokens
        all_layer_data[layer_idx]['response_text'] = response_text
        all_layer_data[layer_idx]['response_tokens'] = response_tokens

    return all_layer_data



def _save_capture_data(
    data: Dict, prompt_item: Dict, set_name: str, inference_dir: Path,
    args, model_name: str = None, lora_adapter: str = None,
    all_layer_data=None, layer_indices=None
):
    """Save captured data: raw .pt and response JSON.

    Projections are computed separately via project_raw_activations_onto_traits.py.
    """
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
            'inference_model': model_name or 'unknown',
            'lora_adapter': lora_adapter,
            'inference_experiment': args.experiment,
            'prompt_set': set_name,
            'prompt_id': prompt_id,
            'prompt_note': prompt_note,
            'capture_date': datetime.now().isoformat()
        }
    }
    with open(responses_dir / f"{prompt_id}.json", 'w') as f:
        json.dump(response_data, f, indent=2)

    # Optional: Save layer internals .pt files
    if args.layer_internals is not None and all_layer_data is not None and layer_indices is not None:
        internals_dir = inference_dir / "raw" / "internals" / set_name
        internals_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layer_indices:
            torch.save(all_layer_data[layer_idx], internals_dir / f"{prompt_id}_L{layer_idx}.pt")


def main():
    parser = argparse.ArgumentParser(description="Capture raw activations from model inference")
    parser.add_argument("--experiment", required=True, help="Experiment name")

    # Prompt input
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-set", help="Prompt set from datasets/inference/{name}.json")
    prompt_group.add_argument("--prompt", help="Single prompt string")
    prompt_group.add_argument("--all-prompt-sets", action="store_true")

    # Options
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for capture (auto-detect from VRAM if not set)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of prompts to process (for testing)")

    # Optional deep capture
    parser.add_argument("--layer-internals", metavar="N", action='append',
                       help="Save full layer internals as .pt files (Q/K/V, MLP, residual). "
                            "Use 'all' for all layers or specify layer indices.")
    parser.add_argument("--capture-mlp", action="store_true",
                       help="Also capture mlp_out activations (down_proj output)")

    # Model options
    parser.add_argument("--model", type=str, default=None,
                       help="Model to use (overrides experiment config application_model)")
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA adapter to apply on top of model (HuggingFace path)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit quantization (for 70B+ models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix to append to output directory names (e.g., 'sycophant' -> prompt_set_sycophant/)")
    parser.add_argument("--prefill", type=str, default=None,
                       help="Prefill string to force model to start with (for prefill attack testing). "
                            "Appended after assistant turn marker, counted as prompt tokens.")
    parser.add_argument("--replay-responses", type=str, default=None,
                       help="Load responses from another prompt set's .pt files for prefill capture. "
                            "Used for model-diff: run same text through different models. "
                            "Example: --replay-responses rm_sycophancy_train_100_sycophant")
    parser.add_argument("--no-server", action="store_true",
                       help="Force local model loading (skip model server check)")

    args = parser.parse_args()

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

    # Load experiment config
    config = load_experiment_config(args.experiment)

    # Determine model to use
    if args.model:
        model_name = args.model
    else:
        model_name = config.get('application_model')
        if not model_name:
            print("Error: No model specified. Use --model or set application_model in experiment config.")
            return

    # Load model (with optional LoRA and quantization)
    # Try server first unless --no-server or --lora (LoRA requires local model)
    is_remote = False
    if not args.no_server and not args.lora:
        handle = get_model_or_client(model_name, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            model = handle
            # Still need tokenizer locally for prompt formatting
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            is_remote = True
            n_layers = None  # Server handles this
        else:
            model, tokenizer = handle
    elif args.lora or args.load_in_8bit or args.load_in_4bit:
        model, tokenizer = load_model_with_lora(
            model_name,
            lora_adapter=args.lora,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if not is_remote:
        n_layers = len(get_inner_model(model).layers)
        print(f"Model has {n_layers} layers")

    # Calculate batch size (only for local mode)
    if not is_remote:
        if args.batch_size is None:
            # For inference capture, estimate max_seq_len as prompt + generated
            max_seq_len = 512  # Conservative estimate for inference
            args.batch_size = calculate_max_batch_size(model, max_seq_len)
        print(f"Batch size: {args.batch_size}")

    # Get chat template setting from config
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {use_chat_template}")
    if args.prefill:
        print(f"Prefill: '{args.prefill}' (will be appended to each prompt)")

    # Process prompts
    for set_name, prompts in prompt_sets:
        # Apply --output-suffix if set
        if args.output_suffix:
            set_name = f"{set_name}_{args.output_suffix}"

        # Apply --limit if set
        if args.limit is not None:
            prompts = prompts[:args.limit]

        print(f"\n{'='*60}")
        print(f"Processing: {set_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        # ================================================================
        # LAYER INTERNALS MODE: Sequential deep capture
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
                # Append prefill if provided (for prefill attack testing)
                if args.prefill:
                    prompt_text = prompt_text + args.prefill

                # Heavy capture: layer internals (includes residual)
                all_layer_data = capture_multiple_layer_internals(
                    model, tokenizer, prompt_text, layer_indices,
                    args.max_new_tokens, args.temperature,
                    use_chat_template=use_chat_template
                )

                # Extract residual data from internals for projections
                data = extract_residual_from_internals(all_layer_data, n_layers)

                # Save raw .pt and response JSON
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, args,
                    model_name=model_name, lora_adapter=args.lora,
                    all_layer_data=all_layer_data, layer_indices=layer_indices
                )

            continue  # Done with this prompt set

        # ================================================================
        # REPLAY-RESPONSES MODE: Prefill capture from another prompt set
        # ================================================================
        if args.replay_responses:
            source_dir = inference_dir / "raw" / "residual" / args.replay_responses
            if not source_dir.exists():
                print(f"  ERROR: Source prompt set not found: {source_dir}")
                continue

            print(f"  Prefill mode: loading responses from {args.replay_responses}")

            for prompt_item in tqdm(prompts, desc="Prefill capture"):
                prompt_id = prompt_item['id']

                # Skip existing if requested
                if args.skip_existing:
                    raw_pt_path = inference_dir / "raw" / "residual" / set_name / f"{prompt_id}.pt"
                    if raw_pt_path.exists():
                        continue

                # Load response from source prompt set
                source_pt = source_dir / f"{prompt_id}.pt"
                if not source_pt.exists():
                    print(f"  Warning: {prompt_id}.pt not found in source, skipping")
                    continue

                source_data = torch.load(source_pt, weights_only=True)
                response_text = source_data['response']['text']

                # Format prompt
                raw_prompt = prompt_item['text']
                prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template)

                # Capture with prefill
                data = capture_residual_stream_prefill(
                    model, tokenizer, prompt_text, response_text,
                    n_layers, capture_mlp=args.capture_mlp
                )

                # Save raw .pt and response JSON
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, args,
                    model_name=model_name, lora_adapter=args.lora
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
            # Append prefill if provided (for prefill attack testing)
            if args.prefill:
                prompt_text = prompt_text + args.prefill
            prompt_texts.append(prompt_text)
            prompt_items_filtered.append(prompt_item)

        if not prompt_texts:
            print("  All prompts already captured, skipping...")
            continue

        if is_remote:
            # Remote: single call returns all results
            print(f"  Capturing {len(prompt_texts)} prompts via server...")
            all_results = model.generate_with_capture(
                prompt_texts,
                n_layers=n_layers,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                capture_mlp=args.capture_mlp,
            )
            # Infer n_layers from first result's activations
            if all_results and n_layers is None:
                n_layers = len(all_results[0].prompt_activations)
            for result, prompt_item in zip(all_results, prompt_items_filtered):
                data = capture_result_to_data(result, n_layers)
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, args,
                    model_name=model_name, lora_adapter=args.lora
                )
        else:
            # Local: batched generator for crash resilience
            print(f"  Capturing {len(prompt_texts)} prompts in batches of {args.batch_size}...")

            # Pre-batch prompt_items to match generator output
            prompt_item_batches = [
                prompt_items_filtered[i:i+args.batch_size]
                for i in range(0, len(prompt_items_filtered), args.batch_size)
            ]

            # Run batched capture with incremental saving (generator mode)
            # Chat template already includes BOS, so don't add special tokens again
            batch_generator = generate_with_capture(
                model, tokenizer, prompt_texts,
                n_layers=n_layers,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                capture_mlp=args.capture_mlp,
                show_progress=True,
                yield_per_batch=True,
                add_special_tokens=not use_chat_template
            )

            # Process and save after each batch (crash-resilient)
            for (batch_results, _batch_prompts), batch_items in zip(batch_generator, prompt_item_batches):
                for result, prompt_item in zip(batch_results, batch_items):
                    # Convert CaptureResult to dict and save
                    data = capture_result_to_data(result, n_layers)
                    _save_capture_data(
                        data, prompt_item, set_name, inference_dir, args,
                        model_name=model_name, lora_adapter=args.lora
                    )

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"\nOutput locations:")
    print(f"  Raw activations: {inference_dir}/raw/residual/{{prompt_set}}/")
    print(f"  Responses:       {inference_dir}/responses/{{prompt_set}}/")
    print(f"\nTo compute projections, run:")
    print(f"  python inference/project_raw_activations_onto_traits.py --experiment {args.experiment} --prompt-set <prompt_set>")


if __name__ == "__main__":
    main()
