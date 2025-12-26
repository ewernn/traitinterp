"""
Extract activations from generated responses.

Called by run_pipeline.py (stage 3).
"""

import json
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path
from core import MultiLayerCapture


def load_vetting_filter(experiment: str, trait: str) -> dict:
    """Load failed indices from response vetting. Returns dict of indices to EXCLUDE."""
    vetting_file = get_path('extraction.trait', experiment=experiment, trait=trait) / 'vetting' / 'response_scores.json'
    if not vetting_file.exists():
        return {'positive': [], 'negative': []}
    with open(vetting_file) as f:
        data = json.load(f)
    return data.get('failed_indices', {'positive': [], 'negative': []})


def extract_activations_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_split: float = 0.2,
    base_model: bool = False,
    component: str = 'residual',
    use_vetting_filter: bool = True,
    max_completion_tokens: int = 128,
) -> int:
    """
    Extract activations from generated responses. Returns number of layers extracted.
    """
    print(f"  [3] Extracting activations for '{trait}' (component: {component})...")

    n_layers = model.config.num_hidden_layers
    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load responses
    try:
        with open(responses_dir / 'pos.json') as f:
            pos_data = json.load(f)
        with open(responses_dir / 'neg.json') as f:
            neg_data = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: Response files not found. Run stage 1 first.")
        return 0

    # Filter based on vetting results
    n_filtered_pos, n_filtered_neg = 0, 0
    if use_vetting_filter:
        failed = load_vetting_filter(experiment, trait)
        pos_failed, neg_failed = set(failed.get('positive', [])), set(failed.get('negative', []))
        if pos_failed or neg_failed:
            pos_data = [r for i, r in enumerate(pos_data) if i not in pos_failed]
            neg_data = [r for i, r in enumerate(neg_data) if i not in neg_failed]
            n_filtered_pos, n_filtered_neg = len(pos_failed), len(neg_failed)
            print(f"    Filtered {n_filtered_pos + n_filtered_neg} responses based on vetting")

    # Split into train/val
    train_pos, train_neg, val_pos, val_neg = pos_data, neg_data, [], []
    if val_split > 0:
        pos_split = int(len(pos_data) * (1 - val_split))
        neg_split = int(len(neg_data) * (1 - val_split))
        train_pos, val_pos = pos_data[:pos_split], pos_data[pos_split:]
        train_neg, val_neg = neg_data[:neg_split], neg_data[neg_split:]

    def extract_from_responses(responses: list[dict], label: str) -> dict[int, torch.Tensor]:
        all_activations = {layer: [] for layer in range(n_layers)}
        for item in tqdm(responses, desc=f"    {label}", leave=False):
            full_text = item.get('full_text')
            if not full_text:
                continue

            inputs = tokenizer(full_text, return_tensors='pt').to(model.device)
            seq_len = inputs['input_ids'].shape[1]

            if base_model:
                prompt_token_count = item.get('prompt_token_count') or len(tokenizer(item.get('prompt', ''))['input_ids'])
                start_idx, end_idx = prompt_token_count, min(seq_len, prompt_token_count + max_completion_tokens)
                if start_idx >= end_idx:
                    continue
            else:
                start_idx, end_idx = 0, seq_len

            with MultiLayerCapture(model, component=component) as capture:
                with torch.no_grad():
                    model(**inputs)

            for layer in range(n_layers):
                acts = capture.get(layer)
                if acts is not None:
                    acts_mean = acts[0, start_idx:end_idx, :].mean(dim=0).cpu()
                    all_activations[layer].append(acts_mean)

        for layer in range(n_layers):
            all_activations[layer] = torch.stack(all_activations[layer]) if all_activations[layer] else torch.empty(0)
        return all_activations

    # Extract training activations
    pos_acts = extract_from_responses(train_pos, 'train_positive')
    neg_acts = extract_from_responses(train_neg, 'train_negative')

    # Combine and save
    pos_all = torch.stack([pos_acts[l] for l in range(n_layers)], dim=1)
    neg_all = torch.stack([neg_acts[l] for l in range(n_layers)], dim=1)
    all_acts = torch.cat([pos_all, neg_all], dim=0)

    filename = "all_layers.pt" if component == 'residual' else f"{component}_all_layers.pt"
    torch.save(all_acts, activations_dir / filename)
    print(f"    Saved: {all_acts.shape}")

    # Save validation activations
    n_val_pos, n_val_neg = 0, 0
    if val_split > 0 and (val_pos or val_neg):
        val_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)
        val_dir.mkdir(parents=True, exist_ok=True)

        val_pos_acts = extract_from_responses(val_pos, 'val_positive')
        val_neg_acts = extract_from_responses(val_neg, 'val_negative')

        prefix = "" if component == 'residual' else f"{component}_"
        for layer in range(n_layers):
            torch.save(val_pos_acts[layer], val_dir / f"{prefix}val_pos_layer{layer}.pt")
            torch.save(val_neg_acts[layer], val_dir / f"{prefix}val_neg_layer{layer}.pt")
        n_val_pos, n_val_neg = len(val_pos), len(val_neg)

    # Compute activation norms
    activation_norms = {layer: round(all_acts[:, layer, :].norm(dim=-1).mean().item(), 2) for layer in range(n_layers)}

    # Save metadata
    metadata = {
        'model': model.config.name_or_path,
        'trait': trait,
        'n_layers': n_layers,
        'hidden_dim': all_acts.shape[-1],
        'n_examples_pos': len(train_pos),
        'n_examples_neg': len(train_neg),
        'n_filtered_pos': n_filtered_pos,
        'n_filtered_neg': n_filtered_neg,
        'val_split': val_split,
        'n_val_pos': n_val_pos,
        'n_val_neg': n_val_neg,
        'base_model': base_model,
        'component': component,
        'activation_norms': activation_norms,
        'timestamp': datetime.now().isoformat(),
    }
    metadata_filename = "metadata.json" if component == 'residual' else f"{component}_metadata.json"
    with open(activations_dir / metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_layers
