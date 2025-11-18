#!/usr/bin/env python3
"""
Extract ALL natural vectors for emotional_valence
All 4 methods × 26 layers = 104 vectors
"""
import sys
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append('/Users/ewern/Desktop/code/trait-interp')
from traitlens.methods import MeanDifferenceMethod, ProbeMethod, ICAMethod, GradientMethod

def extract_all_layers(experiment='gemma_2b_cognitive_nov20', trait='emotional_valence'):
    """Extract vectors for all 26 layers using all 4 methods"""

    print("="*80)
    print("EXTRACTING ALL NATURAL VECTORS")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Methods: mean_diff, probe, ica, gradient")
    print(f"Layers: 0-25 (26 total)")
    print("="*80)

    acts_dir = Path(f'experiments/{experiment}/{trait}/extraction/activations')
    vectors_dir = Path(f'experiments/{experiment}/{trait}/extraction/vectors')
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Initialize methods
    methods = {
        'mean_diff': MeanDifferenceMethod(),
        'probe': ProbeMethod(),
        'ica': ICAMethod(),
        'gradient': GradientMethod()
    }

    print(f"\nProcessing all 26 layers...")

    for layer in tqdm(range(26), desc="Layers"):
        # Load activations for this layer
        pos_file = acts_dir / f'pos_layer{layer}.pt'
        neg_file = acts_dir / f'neg_layer{layer}.pt'

        if not pos_file.exists() or not neg_file.exists():
            print(f"\n⚠️  Layer {layer}: activations not found, skipping")
            continue

        pos_acts = torch.load(pos_file)
        neg_acts = torch.load(neg_file)

        # Extract with each method
        for method_name, method in methods.items():
            try:
                result = method.extract(pos_acts, neg_acts)
                vector = result['vector']

                # Save vector
                vector_file = vectors_dir / f'{method_name}_layer{layer}.pt'
                torch.save(vector, vector_file)

                # Save metadata
                metadata = {
                    'layer': layer,
                    'method': method_name,
                    'vector_norm': vector.norm().item(),
                    'n_pos': pos_acts.shape[0],
                    'n_neg': neg_acts.shape[0]
                }

                if method_name == 'probe':
                    metadata['train_acc'] = result['train_acc']
                elif method_name == 'gradient':
                    metadata['final_separation'] = result.get('final_separation', 0)
                elif method_name == 'ica':
                    metadata['component_idx'] = result.get('component_idx', -1)

                import json
                metadata_file = vectors_dir / f'{method_name}_layer{layer}_metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"\n⚠️  Layer {layer}, {method_name}: {str(e)}")
                continue

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nVectors saved to: {vectors_dir}")
    print(f"Total: 4 methods × 26 layers = 104 vector files")

    # Verify
    vector_files = list(vectors_dir.glob('*_layer*.pt'))
    print(f"Found: {len(vector_files)} vector files")

    return vectors_dir

if __name__ == '__main__':
    extract_all_layers()
