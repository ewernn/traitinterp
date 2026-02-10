#!/usr/bin/env python3
"""
Capture raw activations from model inference.

Runs model inference and saves raw activations as .pt files. Projections onto trait
vectors are computed separately via project_raw_activations_onto_traits.py.

Output files:
  Raw activations : experiments/{exp}/inference/raw/residual/{prompt_set}/{id}.pt
  Responses       : experiments/{exp}/inference/responses/{prompt_set}/{id}.json

.pt file structure (model-agnostic):
  {
    'prompt': {
      'text': str,
      'tokens': List[str],
      'token_ids': List[int],
      'activations': {
        layer_idx: {
          'residual': Tensor[n_tokens, hidden_dim],           # layer output
          'attn_contribution': Tensor[n_tokens, hidden_dim],  # what attention adds to residual
          'mlp_contribution': Tensor[n_tokens, hidden_dim],   # if --capture-mlp
        }
      }
    },
    'response': { ... }  # same structure
  }

Usage:
    # Basic capture (residual + attn_contribution)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # With mlp_out for component decomposition
    python inference/capture_raw_activations.py \\
        --experiment my_experiment \\
        --prompt-set harmful \\
        --capture-mlp

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
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model import format_prompt, load_model_with_lora, get_inner_model, get_layer_path_prefix, tokenize, tokenize_batch
from utils.json import dump_compact
from utils.generation import generate_with_capture, calculate_max_batch_size
from utils.paths import get as get_path, get_model_variant, get_inference_raw_dir, load_experiment_config
from utils.model_registry import get_model_slug
from other.server.client import get_model_or_client, ModelClient


def normalize_prompt_item(item: Dict) -> Dict:
    """Normalize prompt item to have 'text' key (handles 'prompt' as fallback)."""
    if 'text' not in item and 'prompt' in item:
        item = {**item, 'text': item['prompt']}
    return item


def normalize_prompts(prompts: List[Dict]) -> List[Dict]:
    """Normalize list of prompt items."""
    return [normalize_prompt_item(p) for p in prompts]


def parse_layers(layers_str: str, n_layers: int) -> List[int]:
    """Parse layer specification string into list of layer indices.

    Accepts: '0,10,20,30' or '0-75:5' (range with step) or '0-25' (range).
    """
    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if ':' in part:
            # Range with step: '0-75:5'
            range_part, step = part.split(':')
            start, end = range_part.split('-')
            layers.extend(range(int(start), int(end) + 1, int(step)))
        elif '-' in part and not part.startswith('-'):
            # Range: '0-25'
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    # Filter to valid range
    layers = sorted(set(l for l in layers if 0 <= l < n_layers))
    return layers


def capture_result_to_data(result, n_layers: int, layers: List[int] = None) -> Dict:
    """Convert CaptureResult from utils.generation to dict format for saving.

    Args:
        layers: If provided, only include these layers (post-capture filtering for batch mode).
    """
    layer_indices = layers if layers is not None else list(range(n_layers))
    prompt_acts = {}
    response_acts = {}
    for layer_idx in layer_indices:
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


def capture_residual_stream_prefill(model, tokenizer, prompt_text: str, response_text: str,
                                     n_layers: int, capture_mlp: bool = False,
                                     capture_attention: bool = False,
                                     layers: List[int] = None) -> Dict:
    """
    Capture residual stream activations with prefilled response (single forward pass).

    Used for model-diff analysis: run same text through different models.
    Uses MultiLayerCapture primitive for clean hook management.
    """
    from core import MultiLayerCapture

    # Tokenize prompt
    prompt_inputs = tokenize(prompt_text, tokenizer).to(model.device)
    n_prompt_tokens = prompt_inputs['input_ids'].shape[1]
    prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (without special tokens - it's appended to prompt)
    response_inputs = tokenize(response_text, tokenizer, add_special_tokens=False).to(model.device)
    response_token_ids = response_inputs['input_ids'][0].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    # Concatenate for single forward pass
    full_input_ids = torch.cat([prompt_inputs['input_ids'], response_inputs['input_ids']], dim=1)

    # Capture all components in one forward pass using MultiLayerCapture
    # attn_contribution/mlp_contribution auto-detect architecture (post-norm for Gemma-2, o_proj for Llama)
    with MultiLayerCapture(model, component='residual', layers=layers) as cap_residual:
        with MultiLayerCapture(model, component='attn_contribution', layers=layers) as cap_attn:
            if capture_mlp:
                with MultiLayerCapture(model, component='mlp_contribution') as cap_mlp:
                    with torch.no_grad():
                        outputs = model(input_ids=full_input_ids, output_attentions=capture_attention, return_dict=True)
                mlp_acts_full = cap_mlp.get_all()
            else:
                with torch.no_grad():
                    outputs = model(input_ids=full_input_ids, output_attentions=capture_attention, return_dict=True)
                mlp_acts_full = {}
        attn_acts_full = cap_attn.get_all()
    residual_acts_full = cap_residual.get_all()

    # Split activations into prompt/response portions
    prompt_acts = {}
    response_acts = {}
    layer_indices = layers if layers is not None else list(range(n_layers))
    for layer_idx in layer_indices:
        prompt_acts[layer_idx] = {}
        response_acts[layer_idx] = {}

        # Residual: [batch, seq_len, hidden] -> [seq_len, hidden]
        if layer_idx in residual_acts_full:
            full = residual_acts_full[layer_idx].squeeze(0)
            prompt_acts[layer_idx]['residual'] = full[:n_prompt_tokens]
            response_acts[layer_idx]['residual'] = full[n_prompt_tokens:]

        # Attention contribution (what actually adds to residual)
        if layer_idx in attn_acts_full:
            full = attn_acts_full[layer_idx].squeeze(0)
            prompt_acts[layer_idx]['attn_contribution'] = full[:n_prompt_tokens]
            response_acts[layer_idx]['attn_contribution'] = full[n_prompt_tokens:]

        # MLP contribution (optional)
        if capture_mlp and layer_idx in mlp_acts_full:
            full = mlp_acts_full[layer_idx].squeeze(0)
            prompt_acts[layer_idx]['mlp_contribution'] = full[:n_prompt_tokens]
            response_acts[layer_idx]['mlp_contribution'] = full[n_prompt_tokens:]

    # Split attention patterns (only if captured)
    prompt_attention = {}
    response_attention = []
    if capture_attention and outputs.attentions is not None:
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


def _save_capture_data(
    data: Dict, prompt_item: Dict, set_name: str, inference_dir: Path,
    args, model_name: str = None, lora_adapter: str = None,
):
    """Save captured data: raw .pt and response JSON.

    Projections are computed separately via project_raw_activations_onto_traits.py.
    """
    prompt_id = prompt_item['id']
    prompt_note = prompt_item.get('note', '')

    # Save raw residual .pt
    raw_dir = inference_dir / "raw" / "residual" / set_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_data = data
    if getattr(args, 'response_only', False):
        # Strip prompt activations to save space (keep metadata for downstream)
        save_data = {
            'prompt': {k: v for k, v in data['prompt'].items() if k != 'activations'},
            'response': data['response'],
        }
        save_data['prompt']['activations'] = {}
    torch.save(save_data, raw_dir / f"{prompt_id}.pt")

    # Save response JSON (shared, trait-independent)
    responses_dir = inference_dir / "responses" / set_name
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
        'inference_model': model_name or 'unknown',
        'prompt_note': prompt_note if prompt_note else None,
        'capture_date': datetime.now().isoformat(),
        'tags': []
    }
    with open(responses_dir / f"{prompt_id}.json", 'w') as f:
        dump_compact(response_data, f)


def main():
    parser = argparse.ArgumentParser(description="Capture raw activations from model inference")
    parser.add_argument("--experiment", required=True, help="Experiment name")

    # Prompt input
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-set", help="Prompt set from datasets/inference/{name}.json")
    prompt_group.add_argument("--prompts-file", help="Direct path to prompts JSON file (requires --prompt-set-name)")
    prompt_group.add_argument("--prompt", help="Single prompt string")
    prompt_group.add_argument("--all-prompt-sets", action="store_true")

    parser.add_argument("--prompt-set-name", help="Name for --prompts-file output directory")

    # Options
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for capture (auto-detect from VRAM if not set)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of prompts to process (for testing)")

    # Optional capture
    parser.add_argument("--capture-mlp", action="store_true",
                       help="Also capture mlp_out activations (down_proj output)")
    parser.add_argument("--layers", type=str, default=None,
                       help="Only capture specific layers. Comma-separated (e.g., '0,10,20,30') "
                            "or range with step (e.g., '0-75:5' for every 5th layer). "
                            "Default: all layers. Reduces CPU hook overhead and storage.")
    parser.add_argument("--response-only", action="store_true",
                       help="Only save response token activations (skip prompt tokens). "
                            "Dramatically reduces .pt file size for long prompts.")

    # Model options
    parser.add_argument("--model-variant", default=None,
                       help="Model variant for inference (default: from experiment defaults.application)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit quantization (for 70B+ models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit quantization)")
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix to append to output directory names (e.g., 'sycophant' -> prompt_set_sycophant/)")
    parser.add_argument("--prefill", type=str, default=None,
                       help="Prefill string to force model to start with (for prefill attack testing). "
                            "Appended after assistant turn marker, counted as prompt tokens.")
    parser.add_argument("--replay-responses", type=str, default=None,
                       help="Load responses from another prompt set's .pt files for prefill capture. "
                            "Used for model-diff: run same text through different models. "
                            "Example: --replay-responses rm_syco/train_100")
    parser.add_argument("--replay-from-variant", type=str, default=None,
                       help="Model variant to load replay responses from (default: same as --model-variant). "
                            "Use with --replay-responses for cross-variant model-diff. "
                            "Example: --replay-from-variant rm_lora")
    parser.add_argument("--no-server", action="store_true",
                       help="Force local model loading (skip model server check)")

    args = parser.parse_args()

    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=args.experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    # Get prompts from JSON files
    prompts_source = get_path('datasets.inference')
    if not prompts_source.exists():
        print(f"Prompts directory not found: {prompts_source}")
        return

    if args.prompt:
        # Ad-hoc single prompt - use id=1
        prompt_sets = [("adhoc", [{"id": 1, "text": args.prompt, "note": "ad-hoc prompt"}])]
    elif args.prompts_file:
        # Direct path to prompts JSON
        if not args.prompt_set_name:
            print("Error: --prompts-file requires --prompt-set-name")
            return
        prompt_file = Path(args.prompts_file)
        if not prompt_file.exists():
            print(f"Prompts file not found: {prompt_file}")
            return
        with open(prompt_file) as f:
            data = json.load(f)
        # Handle both {"prompts": [...]} and bare [...] formats
        prompts = data.get('prompts', data) if isinstance(data, dict) else data
        prompts = normalize_prompts(prompts)
        prompt_sets = [(args.prompt_set_name, prompts)]
        print(f"Loaded {len(prompts)} prompts from {prompt_file}")
    elif args.prompt_set:
        prompt_file = prompts_source / f"{args.prompt_set}.json"
        if not prompt_file.exists():
            print(f"Prompt set not found: {prompt_file}")
            return
        with open(prompt_file) as f:
            data = json.load(f)
        prompts = normalize_prompts(data['prompts'])
        prompt_sets = [(args.prompt_set, prompts)]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_set}")
    else:
        # Load all JSON prompt sets
        prompt_sets = []
        for f in sorted(prompts_source.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
            if 'prompts' in data and data['prompts']:
                prompts = normalize_prompts(data['prompts'])
                prompt_sets.append((f.stem, prompts))
        print(f"Found {len(prompt_sets)} prompt sets")

    # Load experiment config
    config = load_experiment_config(args.experiment)

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    # Set inference_dir using model_variant
    inference_dir = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant)
    print(f"Saving to: inference/{model_variant}/")

    # Load model (with optional LoRA and quantization)
    # Try server first unless --no-server or lora (LoRA requires local model)
    is_remote = False
    if not args.no_server and not lora:
        handle = get_model_or_client(model_name, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})" + (f" + LoRA: {lora}" if lora else ""))
            model = handle
            # Still need tokenizer locally for prompt formatting
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            is_remote = True
            n_layers = None  # Server handles this
        else:
            model, tokenizer = handle
    elif lora or args.load_in_8bit or args.load_in_4bit:
        model, tokenizer = load_model_with_lora(
            model_name,
            lora_adapter=lora,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        print(f"Loading model: {model_name}" + (f" + LoRA: {lora}" if lora else ""))
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    if not is_remote:
        n_layers = len(get_inner_model(model).layers)
        print(f"Model has {n_layers} layers")

    # Parse --layers flag
    capture_layers = None
    if args.layers and not is_remote:
        capture_layers = parse_layers(args.layers, n_layers)
        print(f"Capturing {len(capture_layers)} layers: {capture_layers}")

    # Get chat template setting from config (needed for batch size calculation)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {use_chat_template}")

    # Calculate batch size (only for local mode)
    if not is_remote:
        if args.batch_size is None:
            # Calculate max_seq_len from actual tokenized prompts
            # Build all formatted prompts across all sets
            all_formatted_prompts = []
            for set_name, prompts in prompt_sets:
                for prompt_item in prompts:
                    raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
                    prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template)

                    # Account for prefill if set
                    if args.prefill:
                        prompt_text = prompt_text + args.prefill

                    all_formatted_prompts.append(prompt_text)

            if not all_formatted_prompts:
                raise ValueError("No prompts found to tokenize. Cannot calculate batch size.")

            # Tokenize all prompts in one batch call
            batch_result = tokenize_batch(all_formatted_prompts, tokenizer)
            max_prompt_len = max(batch_result['lengths'])

            max_seq_len = max_prompt_len + args.max_new_tokens
            args.batch_size = calculate_max_batch_size(model, max_seq_len, mode='generation')
            print(f"Batch size: {args.batch_size} (max_prompt_len={max_prompt_len}, max_new_tokens={args.max_new_tokens})")
        else:
            print(f"Batch size: {args.batch_size} (user-specified)")
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
        # REPLAY-RESPONSES MODE: Prefill capture from another prompt set
        # ================================================================
        if args.replay_responses:
            # Determine source variant (default: same as current model variant)
            source_variant = args.replay_from_variant or model_variant
            source_dir = get_path('inference.raw_residual',
                                  experiment=args.experiment,
                                  model_variant=source_variant,
                                  prompt_set=args.replay_responses)
            if not source_dir.exists():
                print(f"  ERROR: Source prompt set not found: {source_dir}")
                continue

            print(f"  Prefill mode: loading responses from {source_variant}/{args.replay_responses}")

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
                raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
                prompt_text = format_prompt(raw_prompt, tokenizer, use_chat_template=use_chat_template)

                # Capture with prefill
                data = capture_residual_stream_prefill(
                    model, tokenizer, prompt_text, response_text,
                    n_layers, capture_mlp=args.capture_mlp,
                    layers=capture_layers
                )

                # Save raw .pt and response JSON
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, args,
                    model_name=model_name, lora_adapter=lora
                )

            continue  # Done with this prompt set

        # ================================================================
        # BATCHED CAPTURE MODE: Standard residual stream
        # ================================================================
        if capture_layers is not None:
            print(f"  --layers: hooking {len(capture_layers)} of {n_layers} layers")
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

            raw_prompt = prompt_item.get('text') or prompt_item.get('prompt')
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
                data = capture_result_to_data(result, n_layers, layers=capture_layers)
                _save_capture_data(
                    data, prompt_item, set_name, inference_dir, args,
                    model_name=model_name, lora_adapter=lora
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
            # tokenize_batch auto-detects BOS from text content
            batch_generator = generate_with_capture(
                model, tokenizer, prompt_texts,
                n_layers=n_layers,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                capture_mlp=args.capture_mlp,
                show_progress=True,
                yield_per_batch=True,
                layers=capture_layers,
            )

            # Process and save after each batch (crash-resilient)
            for (batch_results, _batch_prompts), batch_items in zip(batch_generator, prompt_item_batches):
                for result, prompt_item in zip(batch_results, batch_items):
                    # Convert CaptureResult to dict and save
                    data = capture_result_to_data(result, n_layers, layers=capture_layers)
                    _save_capture_data(
                        data, prompt_item, set_name, inference_dir, args,
                        model_name=model_name, lora_adapter=lora
                    )

    # Cleanup GPU memory
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
