#!/usr/bin/env python3
"""
Stage 2: Extract activations from generated responses or prefill.

Input:
    - experiments/{experiment}/extraction/{category}/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{category}/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{category}/{trait}/vetting/response_scores.json (optional)
    OR (for --prefill-only):
    - experiments/{experiment}/extraction/{category}/{trait}/positive.txt
    - experiments/{experiment}/extraction/{category}/{trait}/negative.txt

Output:
    - experiments/{experiment}/extraction/{category}/{trait}/activations/all_layers.pt
    - experiments/{experiment}/extraction/{category}/{trait}/activations/metadata.json
    - experiments/{experiment}/extraction/{category}/{trait}/val_activations/ (if --val-split)

Usage:
    # Single trait (with vetting filter by default)
    python extraction/extract_activations.py --experiment my_exp --trait category/my_trait

    # With validation split (last 20% for validation)
    python extraction/extract_activations.py --experiment my_exp --trait category/my_trait --val-split 0.2

    # All traits
    python extraction/extract_activations.py --experiment my_exp --trait all

    # Disable vetting filter
    python extraction/extract_activations.py --experiment my_exp --trait all --no-vetting-filter

    # Prefill-only extraction (last token of prompt, no generation)
    python extraction/extract_activations.py --experiment my_exp --trait category/my_trait --prefill-only
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, load_experiment_config
from utils.model_registry import is_base_model
from traitlens.hooks import HookManager
from traitlens.activations import ActivationCapture


def get_hook_path(layer: int, component: str) -> str:
    """Get the hook path for a given layer and component."""
    if component == 'residual':
        return f"model.layers.{layer}"
    elif component == 'attn_out':
        return f"model.layers.{layer}.self_attn.o_proj"
    elif component == 'mlp_out':
        return f"model.layers.{layer}.mlp.down_proj"
    elif component == 'k_cache':
        return f"model.layers.{layer}.self_attn.k_proj"
    elif component == 'v_cache':
        return f"model.layers.{layer}.self_attn.v_proj"
    else:
        raise ValueError(f"Unknown component: {component}")


def load_vetting_filter(experiment: str, trait: str) -> dict:
    """
    Load failed indices from response vetting.
    Returns dict with 'positive' and 'negative' lists of indices to EXCLUDE.
    """
    vetting_file = get_path('extraction.trait', experiment=experiment, trait=trait) / 'vetting' / 'response_scores.json'
    if not vetting_file.exists():
        return {'positive': [], 'negative': []}

    with open(vetting_file) as f:
        data = json.load(f)

    return data.get('failed_indices', {'positive': [], 'negative': []})


def discover_traits_with_responses(experiment: str) -> list[str]:
    """Find all traits that have response files (Stage 1 complete)."""
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            responses_dir = trait_dir / 'responses'
            if (responses_dir / 'pos.json').exists() and (responses_dir / 'neg.json').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def discover_traits_with_prompts(experiment: str) -> list[str]:
    """Find all traits that have prompt .txt files (for prefill-only extraction)."""
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            if (trait_dir / 'positive.txt').exists() and (trait_dir / 'negative.txt').exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)


def extract_prefill_activations_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_split: float = 0.2,
    token_position: str = 'last',
    component: str = 'residual',
) -> int:
    """
    Extract activations from prefill tokens (no generation).

    Reads prompts from positive.txt and negative.txt, runs forward pass,
    and extracts hidden states at specified token position for all layers.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "category/trait_name").
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        val_split: Fraction of prompts for validation (0.2 = last 20%). 0 = no split.
        token_position: Which token to extract - 'last', 'first', or 'mean'.

    Returns:
        Number of layers extracted.
    """
    print(f"  [Stage 2] Extracting prefill activations for '{trait}' (token: {token_position}, component: {component})...")

    n_layers = model.config.num_hidden_layers
    trait_dir = get_path('extraction.trait', experiment=experiment, trait=trait)
    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts from .txt files
    pos_file = trait_dir / 'positive.txt'
    neg_file = trait_dir / 'negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        print(f"    ERROR: Prompt files not found. Need positive.txt and negative.txt in {trait_dir}")
        return 0

    with open(pos_file, 'r') as f:
        pos_prompts = [line.strip() for line in f if line.strip()]
    with open(neg_file, 'r') as f:
        neg_prompts = [line.strip() for line in f if line.strip()]

    print(f"    Loaded {len(pos_prompts)} positive, {len(neg_prompts)} negative prompts")

    # Split into train/val if requested
    train_pos, train_neg = pos_prompts, neg_prompts
    val_pos, val_neg = [], []
    if val_split > 0:
        pos_split_idx = int(len(pos_prompts) * (1 - val_split))
        neg_split_idx = int(len(neg_prompts) * (1 - val_split))
        train_pos, val_pos = pos_prompts[:pos_split_idx], pos_prompts[pos_split_idx:]
        train_neg, val_neg = neg_prompts[:neg_split_idx], neg_prompts[neg_split_idx:]
        print(f"    Split: train={len(train_pos)}+{len(train_neg)}, val={len(val_pos)}+{len(val_neg)}")

    def extract_token_activations(prompts: list[str], label: str) -> dict[int, torch.Tensor]:
        """Extract hidden states at specified token position for each prompt."""
        all_activations = {layer: [] for layer in range(n_layers)}

        for prompt in tqdm(prompts, desc=f"    Extracting {label}", leave=False):
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

            capture = ActivationCapture()
            with HookManager(model) as hooks:
                for layer in range(n_layers):
                    hook_path = get_hook_path(layer, component)
                    hooks.add_forward_hook(hook_path, capture.make_hook(f"layer_{layer}"))
                with torch.no_grad():
                    model(**inputs)

            for layer in range(n_layers):
                acts = capture.get(f"layer_{layer}")
                if acts is not None:
                    # Extract based on token_position
                    if token_position == 'last':
                        token_act = acts[0, -1, :].cpu()  # [hidden_dim]
                    elif token_position == 'first':
                        token_act = acts[0, 0, :].cpu()  # [hidden_dim]
                    elif token_position == 'mean':
                        token_act = acts[0, :, :].mean(dim=0).cpu()  # [hidden_dim]
                    all_activations[layer].append(token_act)

        for layer in range(n_layers):
            if all_activations[layer]:
                all_activations[layer] = torch.stack(all_activations[layer])
            else:
                all_activations[layer] = torch.empty(0)

        return all_activations

    # Extract training activations
    pos_activations = extract_token_activations(train_pos, 'train_positive')
    neg_activations = extract_token_activations(train_neg, 'train_negative')

    # Combine and save training activations
    pos_all_layers = torch.stack([pos_activations[l] for l in range(n_layers)], dim=1)
    neg_all_layers = torch.stack([neg_activations[l] for l in range(n_layers)], dim=1)
    all_acts = torch.cat([pos_all_layers, neg_all_layers], dim=0)

    filename = "all_layers.pt" if component == 'residual' else f"{component}_all_layers.pt"
    torch.save(all_acts, activations_dir / filename)
    print(f"    Saved train activations: {all_acts.shape} -> {filename}")

    # Extract and save validation activations if val_split > 0
    n_val_pos, n_val_neg = 0, 0
    if val_split > 0 and (val_pos or val_neg):
        val_activations_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)
        val_activations_dir.mkdir(parents=True, exist_ok=True)

        val_pos_acts = extract_token_activations(val_pos, 'val_positive')
        val_neg_acts = extract_token_activations(val_neg, 'val_negative')

        # Save per-layer format for extraction_evaluation.py compatibility
        prefix = "" if component == 'residual' else f"{component}_"
        for layer in range(n_layers):
            torch.save(val_pos_acts[layer], val_activations_dir / f"{prefix}val_pos_layer{layer}.pt")
            torch.save(val_neg_acts[layer], val_activations_dir / f"{prefix}val_neg_layer{layer}.pt")

        n_val_pos, n_val_neg = len(val_pos), len(val_neg)
        print(f"    Saved val activations: {n_val_pos} pos, {n_val_neg} neg ({n_layers} layers each)")

    # Compute per-layer activation norms (mean L2 norm across examples)
    # all_acts shape: [n_examples, n_layers, hidden_dim]
    activation_norms = {}
    for layer in range(n_layers):
        layer_acts = all_acts[:, layer, :]
        norms = layer_acts.norm(dim=-1)  # L2 norm per example
        activation_norms[layer] = round(norms.mean().item(), 2)
    print(f"    Computed activation norms for {n_layers} layers")

    # Save metadata (explicit model tracking)
    model_hf_id = model.config.name_or_path
    metadata = {
        'model': model_hf_id,
        'experiment': experiment,
        'trait': trait,
        'n_layers': n_layers,
        'hidden_dim': all_acts.shape[-1],
        'n_examples_pos': len(train_pos),
        'n_examples_neg': len(train_neg),
        'vetting_filter_used': False,
        'n_filtered_pos': 0,
        'n_filtered_neg': 0,
        'val_split': val_split,
        'n_val_pos': n_val_pos,
        'n_val_neg': n_val_neg,
        'base_model': True,
        'extraction_mode': f'prefill_{token_position}_token',
        'token_position': token_position,
        'component': component,
        'activation_norms': activation_norms,
        'timestamp': datetime.now().isoformat(),
    }
    metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    with open(activations_dir / metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_layers


def extract_activations_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    use_vetting_filter: bool = True,
    val_split: float = 0.2,
    base_model: bool = False,
    component: str = 'residual',
    max_completion_tokens: int = None,
) -> int:
    """
    Extract activations from generated responses.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "category/trait_name").
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        use_vetting_filter: If True, exclude responses that failed vetting.
        val_split: Fraction of scenarios for validation (0.2 = last 20%). 0 = no split.
        base_model: If True, extract from completion tokens only (after prefix).
        max_completion_tokens: If set, limit extraction to first N completion tokens.

    Returns:
        Number of layers extracted.
    """
    print(f"  [Stage 2] Extracting activations for '{trait}' (component: {component})...")

    n_layers = model.config.num_hidden_layers
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load responses
    try:
        with open(responses_dir / 'pos.json', 'r') as f:
            pos_data = json.load(f)
        with open(responses_dir / 'neg.json', 'r') as f:
            neg_data = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: Response files not found in {responses_dir}. Run Stage 1 first.")
        return 0

    # Filter based on vetting results
    n_filtered_pos = 0
    n_filtered_neg = 0
    if use_vetting_filter:
        failed_indices = load_vetting_filter(experiment, trait)
        pos_failed = set(failed_indices.get('positive', []))
        neg_failed = set(failed_indices.get('negative', []))

        if pos_failed or neg_failed:
            pos_data_filtered = [r for i, r in enumerate(pos_data) if i not in pos_failed]
            neg_data_filtered = [r for i, r in enumerate(neg_data) if i not in neg_failed]
            n_filtered_pos = len(pos_data) - len(pos_data_filtered)
            n_filtered_neg = len(neg_data) - len(neg_data_filtered)
            print(f"    Filtered {n_filtered_pos + n_filtered_neg} responses based on vetting ({n_filtered_pos} pos, {n_filtered_neg} neg)")
            pos_data = pos_data_filtered
            neg_data = neg_data_filtered

    # Split into train/val if requested
    train_pos, train_neg = pos_data, neg_data
    val_pos, val_neg = [], []
    if val_split > 0:
        pos_split_idx = int(len(pos_data) * (1 - val_split))
        neg_split_idx = int(len(neg_data) * (1 - val_split))
        train_pos, val_pos = pos_data[:pos_split_idx], pos_data[pos_split_idx:]
        train_neg, val_neg = neg_data[:neg_split_idx], neg_data[neg_split_idx:]
        print(f"    Split: train={len(train_pos)}+{len(train_neg)}, val={len(val_pos)}+{len(val_neg)}")

    def extract_from_responses(responses: list[dict], label: str) -> dict[int, torch.Tensor]:
        all_activations = {layer: [] for layer in range(n_layers)}
        for item in tqdm(responses, desc=f"    Extracting {label}", leave=False):
            full_text = item.get('full_text')
            if not full_text:
                print(f"    WARNING: Missing 'full_text' in response. Skipping.")
                continue

            inputs = tokenizer(full_text, return_tensors='pt').to(model.device)
            seq_len = inputs['input_ids'].shape[1]

            # Determine which tokens to extract from
            if base_model:
                # For base model: extract only from completion tokens (after prefix)
                # Use stored prompt_token_count if available, otherwise re-tokenize prompt
                prompt_token_count = item.get('prompt_token_count')
                if prompt_token_count is None:
                    # Fallback: re-tokenize the prompt to get length
                    prompt = item.get('prompt') or item.get('question', '')
                    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
                    prompt_token_count = prompt_ids.shape[1]

                start_idx = prompt_token_count
                end_idx = seq_len

                # Optionally limit to first N completion tokens
                if max_completion_tokens is not None:
                    end_idx = min(end_idx, start_idx + max_completion_tokens)

                if start_idx >= end_idx:
                    print(f"    WARNING: No completion tokens (prefix={start_idx}, total={end_idx}). Skipping.")
                    continue
            else:
                # For IT model: use all tokens (current behavior)
                start_idx = 0
                end_idx = seq_len

            capture = ActivationCapture()
            with HookManager(model) as hooks:
                for layer in range(n_layers):
                    hook_path = get_hook_path(layer, component)
                    hooks.add_forward_hook(hook_path, capture.make_hook(f"layer_{layer}"))
                with torch.no_grad():
                    model(**inputs)

            for layer in range(n_layers):
                acts = capture.get(f"layer_{layer}")
                if acts is not None:
                    # Extract only the relevant tokens and average
                    relevant_acts = acts[0, start_idx:end_idx, :]  # [completion_len, hidden_dim]
                    acts_mean = relevant_acts.mean(dim=0).cpu()  # [hidden_dim]
                    all_activations[layer].append(acts_mean)

        for layer in range(n_layers):
            if all_activations[layer]:
                all_activations[layer] = torch.stack(all_activations[layer])
            else:
                all_activations[layer] = torch.empty(0)

        return all_activations

    # Extract training activations
    pos_activations = extract_from_responses(train_pos, 'train_positive')
    neg_activations = extract_from_responses(train_neg, 'train_negative')

    # Combine and save training activations
    pos_all_layers = torch.stack([pos_activations[l] for l in range(n_layers)], dim=1)
    neg_all_layers = torch.stack([neg_activations[l] for l in range(n_layers)], dim=1)
    all_acts = torch.cat([pos_all_layers, neg_all_layers], dim=0)

    filename = "all_layers.pt" if component == 'residual' else f"{component}_all_layers.pt"
    torch.save(all_acts, activations_dir / filename)
    print(f"    Saved train activations: {all_acts.shape} -> {filename}")

    # Extract and save validation activations if val_split > 0
    n_val_pos, n_val_neg = 0, 0
    if val_split > 0 and (val_pos or val_neg):
        val_activations_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)
        val_activations_dir.mkdir(parents=True, exist_ok=True)

        val_pos_acts = extract_from_responses(val_pos, 'val_positive')
        val_neg_acts = extract_from_responses(val_neg, 'val_negative')

        # Save per-layer format for extraction_evaluation.py compatibility
        prefix = "" if component == 'residual' else f"{component}_"
        for layer in range(n_layers):
            torch.save(val_pos_acts[layer], val_activations_dir / f"{prefix}val_pos_layer{layer}.pt")
            torch.save(val_neg_acts[layer], val_activations_dir / f"{prefix}val_neg_layer{layer}.pt")

        n_val_pos, n_val_neg = len(val_pos), len(val_neg)
        print(f"    Saved val activations: {n_val_pos} pos, {n_val_neg} neg ({n_layers} layers each)")

    # Compute per-layer activation norms (mean L2 norm across examples)
    # all_acts shape: [n_examples, n_layers, hidden_dim]
    activation_norms = {}
    for layer in range(n_layers):
        layer_acts = all_acts[:, layer, :]
        norms = layer_acts.norm(dim=-1)  # L2 norm per example
        activation_norms[layer] = round(norms.mean().item(), 2)
    print(f"    Computed activation norms for {n_layers} layers")

    # Save metadata (explicit model tracking)
    model_hf_id = model.config.name_or_path
    metadata = {
        'model': model_hf_id,
        'experiment': experiment,
        'trait': trait,
        'n_layers': n_layers,
        'hidden_dim': all_acts.shape[-1],
        'n_examples_pos': len(train_pos),
        'n_examples_neg': len(train_neg),
        'vetting_filter_used': use_vetting_filter,
        'n_filtered_pos': n_filtered_pos,
        'n_filtered_neg': n_filtered_neg,
        'val_split': val_split,
        'n_val_pos': n_val_pos,
        'n_val_neg': n_val_neg,
        'base_model': base_model,
        'extraction_mode': 'completion_only' if base_model else 'full_response',
        'max_completion_tokens': max_completion_tokens,
        'component': component,
        'activation_norms': activation_norms,
        'timestamp': datetime.now().isoformat(),
    }
    metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    with open(activations_dir / metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_layers


def main():
    parser = argparse.ArgumentParser(description='Extract activations from responses or prefill.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait name (e.g., "category/my_trait") or "all" for all traits')
    parser.add_argument('--model', type=str, default=None, help='Model (default: from experiment config)')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu, mps)')
    parser.add_argument('--no-vetting-filter', action='store_true', help='Disable filtering based on vetting results')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of scenarios for validation. 0 = no split.')
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument('--base-model', action='store_true', dest='base_model_override',
                            help='Force base model mode: extract from completion tokens only')
    model_mode.add_argument('--it-model', action='store_true', dest='it_model_override',
                            help='Force IT model mode: extract from full response')
    parser.add_argument('--prefill-only', action='store_true',
                        help='Prefill-only mode: extract from last token of prompt (no generation). '
                             'Reads from positive.txt/negative.txt instead of responses/*.json')
    parser.add_argument('--token-position', type=str, default='last', choices=['last', 'first', 'mean'],
                        help='For prefill-only: which token position to extract (default: last)')
    parser.add_argument('--component', type=str, default='residual',
                        help='Which component(s) to extract: residual, attn_out, mlp_out, k_cache, v_cache, or comma-separated')
    parser.add_argument('--max-completion-tokens', type=int, default=None,
                        help='For base-model mode: limit extraction to first N completion tokens')

    args = parser.parse_args()

    # Resolve model: CLI > config
    config = load_experiment_config(args.experiment, warn_missing=False)
    model_name = args.model or config.get('extraction_model')
    if not model_name:
        parser.error(f"No model specified. Use --model or add 'extraction_model' to experiments/{args.experiment}/config.json")

    # Determine traits to process
    if args.trait.lower() == 'all':
        if args.prefill_only:
            traits = discover_traits_with_prompts(args.experiment)
            if not traits:
                print(f"No traits with prompt files found in experiment '{args.experiment}'")
                return
        else:
            traits = discover_traits_with_responses(args.experiment)
            if not traits:
                print(f"No traits with responses found in experiment '{args.experiment}'")
                return
        print(f"Found {len(traits)} traits to process")
    else:
        traits = [args.trait]

    # Auto-detect base model from config if not explicitly set
    if args.base_model_override:
        base_model = True
    elif args.it_model_override:
        base_model = False
    else:
        base_model = is_base_model(model_name)

    mode_source = "(auto-detected)" if base_model == is_base_model(model_name) else "(override)"

    print("=" * 80)
    print("EXTRACT ACTIVATIONS")
    print(f"Experiment: {args.experiment}")
    print(f"Traits: {len(traits)}")
    print(f"Model: {model_name}")
    if args.prefill_only:
        print(f"Mode: PREFILL-ONLY ({args.token_position} token, no generation)")
    elif base_model:
        max_tok_str = f", max {args.max_completion_tokens} tokens" if args.max_completion_tokens else ""
        print(f"Mode: BASE MODEL {mode_source} (completion tokens only{max_tok_str})")
    else:
        print(f"Mode: IT MODEL {mode_source} (full response)")
    # Parse components
    components = [c.strip() for c in args.component.split(',')]
    valid_components = {'residual', 'attn_out', 'mlp_out', 'k_cache', 'v_cache'}
    for c in components:
        if c not in valid_components:
            print(f"ERROR: Invalid component '{c}'. Must be one of: {valid_components}")
            return
    if components != ['residual']:
        print(f"Components: {', '.join(components)}")
    if args.val_split > 0:
        print(f"Val split: {args.val_split:.0%} (last {args.val_split:.0%} of scenarios)")
    print("=" * 80)

    # Load model and tokenizer once
    model, tokenizer = load_model(model_name, device=args.device)

    # Process each trait and component
    total_layers = 0
    for trait in traits:
        for component in components:
            if args.prefill_only:
                n_layers = extract_prefill_activations_for_trait(
                    experiment=args.experiment,
                    trait=trait,
                    model=model,
                    tokenizer=tokenizer,
                    val_split=args.val_split,
                    token_position=args.token_position,
                    component=component,
                )
            else:
                n_layers = extract_activations_for_trait(
                    experiment=args.experiment,
                    trait=trait,
                    model=model,
                    tokenizer=tokenizer,
                    use_vetting_filter=not args.no_vetting_filter,
                    val_split=args.val_split,
                    base_model=base_model,
                    component=component,
                    max_completion_tokens=args.max_completion_tokens,
                )
            if n_layers > 0:
                total_layers = n_layers

    print(f"\nDONE: Extracted activations from {total_layers} layers for {len(traits)} traits, {len(components)} components.")


if __name__ == '__main__':
    main()
