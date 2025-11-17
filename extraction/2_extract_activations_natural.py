#!/usr/bin/env python3
"""
Extract activations for natural elicitation responses
Reads JSON format (not CSV) from 1_generate_natural.py output
"""

import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from traitlens.hooks import HookManager
from traitlens.activations import ActivationCapture

def extract_activations_natural(experiment, trait, model_name='google/gemma-2-2b-it'):
    """Extract activations from natural elicitation responses"""

    print("="*80)
    print("EXTRACTING ACTIVATIONS - NATURAL ELICITATION")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print("="*80)

    # Setup paths
    base_dir = Path('experiments') / experiment / trait / 'extraction'
    responses_dir = base_dir / 'responses'
    activations_dir = base_dir / 'activations'
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load responses
    print(f"\nLoading responses...")
    with open(responses_dir / 'pos.json', 'r') as f:
        pos_data = json.load(f)
    with open(responses_dir / 'neg.json', 'r') as f:
        neg_data = json.load(f)

    print(f"  Positive: {len(pos_data)} responses")
    print(f"  Negative: {len(neg_data)} responses")

    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    # Get number of layers
    if 'gemma-2' in model_name:
        n_layers = 26  # Gemma 2B has 26 layers (0-25)
    elif 'Llama' in model_name:
        n_layers = 32  # Llama has 32 layers (0-31)
    else:
        n_layers = 26  # Default

    print(f"  Model loaded with {n_layers} layers")

    def extract_from_responses(responses, label):
        """Extract activations for a set of responses"""

        # Storage: {layer: [activations for each example]}
        all_activations = {layer: [] for layer in range(n_layers)}

        for item in tqdm(responses, desc=f"Extracting {label}"):
            # Get full text (prompt + response)
            full_text = item['full_text']

            # Tokenize
            inputs = tokenizer(full_text, return_tensors='pt').to(model.device)

            # Setup hooks for all layers
            capture = ActivationCapture()
            with HookManager(model) as hooks:
                # Add hooks for all layers
                for layer in range(n_layers):
                    hooks.add_forward_hook(
                        f"model.layers.{layer}",
                        capture.make_hook(f"layer_{layer}")
                    )

                # Forward pass
                with torch.no_grad():
                    _ = model(**inputs)

            # Extract and average activations for each layer
            for layer in range(n_layers):
                acts = capture.get(f"layer_{layer}")  # [1, seq_len, hidden_dim]
                acts_mean = acts.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]
                all_activations[layer].append(acts_mean)

        # Stack into tensors [n_examples, hidden_dim] for each layer
        for layer in range(n_layers):
            all_activations[layer] = torch.stack(all_activations[layer])

        return all_activations

    # Extract activations
    print(f"\n{'='*80}")
    print("EXTRACTING ACTIVATIONS")
    print(f"{'='*80}\n")

    pos_activations = extract_from_responses(pos_data, 'positive')
    neg_activations = extract_from_responses(neg_data, 'negative')

    # Save activations
    print(f"\n{'='*80}")
    print("SAVING ACTIVATIONS")
    print(f"{'='*80}\n")

    for layer in range(n_layers):
        # Save positive
        pos_file = activations_dir / f'pos_layer{layer}.pt'
        torch.save(pos_activations[layer], pos_file)

        # Save negative
        neg_file = activations_dir / f'neg_layer{layer}.pt'
        torch.save(neg_activations[layer], neg_file)

    print(f"✅ Saved activations for {n_layers} layers")
    print(f"   Positive shape: {pos_activations[0].shape}")
    print(f"   Negative shape: {neg_activations[0].shape}")
    print(f"   Location: {activations_dir}")

    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model': model_name,
        'n_layers': n_layers,
        'n_positive': len(pos_data),
        'n_negative': len(neg_data),
        'hidden_dim': pos_activations[0].shape[1]
    }

    with open(activations_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Next step:")
    print(f"  python extraction/3_extract_vectors.py --experiment {experiment} --trait {trait}")

    return n_layers

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--trait', type=str, required=True)
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it')

    args = parser.parse_args()

    n_layers = extract_activations_natural(
        experiment=args.experiment,
        trait=args.trait,
        model_name=args.model
    )

    print(f"\n✅ SUCCESS: Extracted activations from {n_layers} layers")
