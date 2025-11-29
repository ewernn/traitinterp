#!/usr/bin/env python3
"""
Stage 2: Extract activations from generated responses.

Input:
    - experiments/{experiment}/extraction/{category}/{trait}/responses/pos.json
    - experiments/{experiment}/extraction/{category}/{trait}/responses/neg.json
    - experiments/{experiment}/extraction/{category}/{trait}/vetting/response_scores.json (optional)

Output:
    - experiments/{experiment}/extraction/{category}/{trait}/activations/all_layers.pt
    - experiments/{experiment}/extraction/{category}/{trait}/activations/metadata.json

Usage:
    # Single trait (with vetting filter by default)
    python extraction/extract_activations.py --experiment my_exp --trait category/my_trait

    # All traits
    python extraction/extract_activations.py --experiment my_exp --trait all

    # Disable vetting filter
    python extraction/extract_activations.py --experiment my_exp --trait all --no-vetting-filter
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from traitlens.hooks import HookManager
from traitlens.activations import ActivationCapture


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


def extract_activations_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    use_vetting_filter: bool = True
) -> int:
    """
    Extract activations from generated responses.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "category/trait_name").
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        use_vetting_filter: If True, exclude responses that failed vetting.

    Returns:
        Number of layers extracted.
    """
    print(f"  [Stage 2] Extracting activations for '{trait}'...")

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

    def extract_from_responses(responses: list[dict], label: str) -> dict[int, torch.Tensor]:
        all_activations = {layer: [] for layer in range(n_layers)}
        for item in tqdm(responses, desc=f"    Extracting {label}", leave=False):
            full_text = item.get('full_text')
            if not full_text:
                print(f"    WARNING: Missing 'full_text' in response. Skipping.")
                continue

            inputs = tokenizer(full_text, return_tensors='pt').to(model.device)

            capture = ActivationCapture()
            with HookManager(model) as hooks:
                for layer in range(n_layers):
                    hooks.add_forward_hook(f"model.layers.{layer}", capture.make_hook(f"layer_{layer}"))
                with torch.no_grad():
                    model(**inputs)

            for layer in range(n_layers):
                acts = capture.get(f"layer_{layer}")
                if acts is not None:
                    acts_mean = acts.mean(dim=1).squeeze(0).cpu()
                    all_activations[layer].append(acts_mean)

        for layer in range(n_layers):
            if all_activations[layer]:
                all_activations[layer] = torch.stack(all_activations[layer])
            else:
                all_activations[layer] = torch.empty(0)

        return all_activations

    pos_activations = extract_from_responses(pos_data, 'positive')
    neg_activations = extract_from_responses(neg_data, 'negative')

    # Combine and save
    pos_all_layers = torch.stack([pos_activations[l] for l in range(n_layers)], dim=1)
    neg_all_layers = torch.stack([neg_activations[l] for l in range(n_layers)], dim=1)
    all_acts = torch.cat([pos_all_layers, neg_all_layers], dim=0)

    torch.save(all_acts, activations_dir / "all_layers.pt")

    print(f"    Saved activations: {all_acts.shape}")

    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model': model.config.name_or_path,
        'n_layers': n_layers,
        'n_examples_pos': len(pos_data),
        'n_examples_neg': len(neg_data),
        'hidden_dim': all_acts.shape[-1],
        'vetting_filter_used': use_vetting_filter,
        'n_filtered_pos': n_filtered_pos,
        'n_filtered_neg': n_filtered_neg
    }
    with open(activations_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_layers


def main():
    parser = argparse.ArgumentParser(description='Extract activations from responses.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True,
                        help='Trait name (e.g., "category/my_trait") or "all" for all traits')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it', help='Model name')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu, mps)')
    parser.add_argument('--no-vetting-filter', action='store_true', help='Disable filtering based on vetting results')

    args = parser.parse_args()

    # Determine traits to process
    if args.trait.lower() == 'all':
        traits = discover_traits_with_responses(args.experiment)
        if not traits:
            print(f"No traits with responses found in experiment '{args.experiment}'")
            return
        print(f"Found {len(traits)} traits to process")
    else:
        traits = [args.trait]

    print("=" * 80)
    print("EXTRACT ACTIVATIONS")
    print(f"Experiment: {args.experiment}")
    print(f"Traits: {len(traits)}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model and tokenizer once
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("Model loaded.\n")

    # Process each trait
    total_layers = 0
    for trait in traits:
        n_layers = extract_activations_for_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            use_vetting_filter=not args.no_vetting_filter
        )
        if n_layers > 0:
            total_layers = n_layers

    print(f"\nDONE: Extracted activations from {total_layers} layers for {len(traits)} traits.")


if __name__ == '__main__':
    main()
