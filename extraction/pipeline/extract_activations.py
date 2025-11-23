"""
Core logic for Stage 2: Extract activations from generated responses.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path
from traitlens.hooks import HookManager
from traitlens.activations import ActivationCapture

def extract_activations_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer
) -> int:
    """
    Extracts activations from generated responses using a pre-loaded model.

    Args:
        experiment: Experiment name.
        trait: Trait name (e.g., "defensiveness").
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.

    Returns:
        Number of layers extracted.
    """
    print(f"  [Stage 2] Extracting activations for '{trait}'...")
    
    n_layers = model.config.num_hidden_layers

    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    activations_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    activations_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(responses_dir / 'pos.json', 'r') as f:
            pos_data = json.load(f)
        with open(responses_dir / 'neg.json', 'r') as f:
            neg_data = json.load(f)
    except FileNotFoundError:
        print(f"    ERROR: Response files for trait '{trait}' not found in {responses_dir}. Run Stage 1 first.")
        return 0

    def extract_from_responses(responses: list[dict], label: str) -> dict[int, torch.Tensor]:
        all_activations = {layer: [] for layer in range(n_layers)}
        for item in tqdm(responses, desc=f"    Extracting {label}", leave=False):
            full_text = item.get('full_text')
            if not full_text:
                print(f"    WARNING: Missing 'full_text' in response item. Skipping.")
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
            else: # Handle case where no activations were captured
                all_activations[layer] = torch.empty(0)

        return all_activations

    pos_activations_by_layer = extract_from_responses(pos_data, 'positive')
    neg_activations_by_layer = extract_from_responses(neg_data, 'negative')

    # Reorganize and save in the original format (all_layers.pt)
    # This is more efficient for storage and loading in the next stage.
    pos_all_layers = torch.stack([pos_activations_by_layer[l] for l in range(n_layers)], dim=1)
    neg_all_layers = torch.stack([neg_activations_by_layer[l] for l in range(n_layers)], dim=1)
    all_acts = torch.cat([pos_all_layers, neg_all_layers], dim=0)

    # Save combined activations
    torch.save(all_acts, activations_dir / "all_layers.pt")
    
    print(f"    Saved combined activations for {n_layers} layers.")
    print(f"    Shape: {all_acts.shape}")
    
    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model': model.config.name_or_path,
        'n_layers': n_layers,
        'n_examples_pos': len(pos_data),
        'n_examples_neg': len(neg_data),
        'hidden_dim': all_acts.shape[-1]
    }

    with open(activations_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_layers
