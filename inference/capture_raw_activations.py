#!/usr/bin/env python3
"""
Capture raw activations from pre-generated responses via prefill.

Reads response JSONs (from generate_responses.py), runs prefill forward passes
to capture activations, and saves .pt files. Projections onto trait vectors are
computed separately via project_raw_activations_onto_traits.py.

Output:
    experiments/{exp}/inference/{model_variant}/raw/residual/{prompt_set}/{id}.pt

.pt file structure (model-agnostic):
  {
    'prompt': {
      'text': str,
      'tokens': List[str],
      'token_ids': List[int],
      'activations': {
        layer_idx: {
          'residual': Tensor[n_tokens, hidden_dim],
          'attn_contribution': Tensor[n_tokens, hidden_dim],
          'mlp_contribution': Tensor[n_tokens, hidden_dim],  # if --capture-mlp
        }
      }
    },
    'response': { ... }  # same structure
  }

Usage:
    # 1. Generate responses first
    python inference/generate_responses.py \\
        --experiment my_experiment --prompt-set main_prompts

    # 2. Capture activations (reads from response JSONs)
    python inference/capture_raw_activations.py \\
        --experiment my_experiment --prompt-set main_prompts

    # Model-diff: capture through different model using another variant's responses
    python inference/capture_raw_activations.py \\
        --experiment my_experiment --prompt-set main_prompts \\
        --model-variant instruct --responses-from rm_lora

    # Then compute projections
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment --prompt-set main_prompts
"""

import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model import format_prompt, load_model_with_lora, get_inner_model
from utils.paths import get as get_path, get_model_variant, load_experiment_config
from utils.capture import capture_residual_stream_prefill  # canonical location

# Re-export for backward compatibility — callers that import from here still work
__all__ = ['capture_residual_stream_prefill']


def parse_layers(layers_str: str, n_layers: int) -> List[int]:
    """Parse layer specification string into list of layer indices.

    Accepts: '0,10,20,30' or '0-75:5' (range with step) or '0-25' (range).
    """
    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if ':' in part:
            range_part, step = part.split(':')
            start, end = range_part.split('-')
            layers.extend(range(int(start), int(end) + 1, int(step)))
        elif '-' in part and not part.startswith('-'):
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(l for l in layers if 0 <= l < n_layers))
    return layers


def _save_pt_data(
    data: Dict, prompt_id, raw_dir: Path, response_only: bool = False,
):
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


def main():
    parser = argparse.ArgumentParser(description="Capture raw activations from pre-generated responses")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name")

    # Capture options
    parser.add_argument("--capture-mlp", action="store_true",
                       help="Also capture mlp_contribution activations")
    parser.add_argument("--layers", type=str, default=None,
                       help="Only capture specific layers. Comma-separated (e.g., '0,10,20,30') "
                            "or range with step (e.g., '0-75:5'). Default: all layers.")
    parser.add_argument("--response-only", action="store_true",
                       help="Only save response token activations (skip prompt tokens)")

    # Model options
    parser.add_argument("--model-variant", default=None,
                       help="Model variant (default: from experiment defaults.application)")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-server", action="store_true",
                       help="Force local model loading")

    # Response source
    parser.add_argument("--responses-from", type=str, default=None,
                       help="Read responses from a different variant's response JSONs. "
                            "For model-diff: capture through current model using another variant's responses.")

    # Common options
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix for output directory name")

    args = parser.parse_args()

    # Validate experiment
    exp_dir = get_path('experiments.base', experiment=args.experiment)
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return

    # Load experiment config
    config = load_experiment_config(args.experiment)

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    # Resolve prompt set name for output
    set_name = args.prompt_set
    if args.output_suffix:
        set_name = f"{set_name}_{args.output_suffix}"

    # Resolve response source
    responses_variant = args.responses_from or model_variant
    responses_dir = get_path('inference.responses',
                             experiment=args.experiment,
                             model_variant=responses_variant,
                             prompt_set=set_name)
    if not responses_dir.exists():
        print(f"Response JSONs not found: {responses_dir}")
        print(f"Run generate_responses.py first, or check --responses-from variant.")
        return

    # Discover available response JSONs
    response_files = sorted(responses_dir.glob("*.json"))
    # Filter out annotation files (files ending with _annotations.json)
    response_files = [f for f in response_files if not f.stem.endswith('_annotations')]
    if not response_files:
        print(f"No response JSON files found in {responses_dir}")
        return

    if args.limit is not None:
        response_files = response_files[:args.limit]

    print(f"Found {len(response_files)} response JSONs in {responses_variant}/{set_name}")
    if args.responses_from:
        print(f"Reading responses from variant: {responses_variant}")

    # Output directory for .pt files
    inference_dir = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant)
    raw_dir = inference_dir / "raw" / "residual" / set_name

    # Filter to non-existing if --skip-existing
    if args.skip_existing:
        original_count = len(response_files)
        response_files = [f for f in response_files if not (raw_dir / f"{f.stem}.pt").exists()]
        skipped = original_count - len(response_files)
        if skipped:
            print(f"Skipping {skipped} already captured")
    if not response_files:
        print("All responses already captured, nothing to do.")
        return

    # Load model (capture requires local model — server doesn't support prefill capture)
    if lora or args.load_in_8bit or args.load_in_4bit:
        model, tokenizer = load_model_with_lora(
            model_name,
            lora_adapter=lora,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        # Try server check — but error clearly since capture needs local model
        if not args.no_server:
            try:
                from other.server.client import is_server_available
                if is_server_available():
                    print("WARNING: Model server detected but capture requires local model. Loading locally.")
            except ImportError:
                pass

        print(f"Loading model: {model_name}" + (f" + LoRA: {lora}" if lora else ""))
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    n_layers = len(get_inner_model(model).layers)
    print(f"Model has {n_layers} layers")

    # Parse --layers flag
    capture_layers = None
    if args.layers:
        capture_layers = parse_layers(args.layers, n_layers)
        print(f"Capturing {len(capture_layers)} of {n_layers} layers: {capture_layers}")

    # Chat template setting
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Capture activations from response JSONs
    print(f"\n{'='*60}")
    print(f"Capturing {len(response_files)} prompts → {model_variant}/raw/residual/{set_name}/")
    print(f"{'='*60}")

    for response_file in tqdm(response_files, desc="Prefill capture"):
        prompt_id = response_file.stem

        # Load response JSON
        with open(response_file) as f:
            response_json = json.load(f)

        prompt_text = response_json['prompt']
        response_text = response_json['response']

        # Capture with prefill
        data = capture_residual_stream_prefill(
            model, tokenizer, prompt_text, response_text,
            n_layers, capture_mlp=args.capture_mlp,
            layers=capture_layers
        )

        # Save .pt file
        _save_pt_data(data, prompt_id, raw_dir, response_only=args.response_only)

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
    print(f"\nOutput: {raw_dir}")
    print(f"\nTo compute projections, run:")
    print(f"  python inference/project_raw_activations_onto_traits.py --experiment {args.experiment} --prompt-set {args.prompt_set}")


if __name__ == "__main__":
    main()
