#!/usr/bin/env python3
"""
Validate natural elicitation vectors have correct polarity
Test that harmful prompts score positive, benign prompts score negative
"""

import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from traitlens.hooks import HookManager
from traitlens.activations import ActivationCapture

def load_test_scenarios(experiment, trait):
    """Load test scenarios for validation from experiment-specific prompts"""
    prompts_dir = Path('experiments') / experiment / trait / 'extraction' / 'prompts'

    if trait == 'refusal':
        pos_file = prompts_dir / 'harmful.txt'
        neg_file = prompts_dir / 'benign.txt'
    elif trait == 'uncertainty_calibration':
        pos_file = prompts_dir / 'uncertain.txt'
        neg_file = prompts_dir / 'certain.txt'
    elif trait == 'sycophancy':
        pos_file = prompts_dir / 'agreeable.txt'
        neg_file = prompts_dir / 'disagreeable.txt'
    else:
        raise ValueError(f"Unknown trait: {trait}")

    def load_file(path):
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()][:10]  # First 10 for quick test

    return load_file(pos_file), load_file(neg_file)

def validate_vector_polarity(
    experiment,
    trait,
    layer=16,
    method='probe',
    model_name='google/gemma-2-2b-it'
):
    """Test if vector has correct polarity on natural prompts"""

    print("="*80)
    print("NATURAL VECTOR VALIDATION")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Method: {method}, Layer: {layer}")
    print("="*80)

    # Load vector
    vector_dir = Path('experiments') / experiment / trait / 'extraction' / 'vectors'
    vector_file = vector_dir / f'{method}_layer{layer}.pt'

    if not vector_file.exists():
        print(f"❌ ERROR: Vector not found at {vector_file}")
        print(f"   Have you run extraction/3_extract_vectors.py?")
        return None

    vector = torch.load(vector_file, map_location='cpu')
    print(f"\n✅ Loaded vector: {vector.shape}, norm: {vector.norm():.2f}")

    # Load test scenarios
    print(f"\nLoading test scenarios...")
    pos_scenarios, neg_scenarios = load_test_scenarios(experiment, trait)
    print(f"  Positive (trait-eliciting): {len(pos_scenarios)} prompts")
    print(f"  Negative (trait-avoiding): {len(neg_scenarios)} prompts")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    # Get activations and project
    def get_projection(prompt):
        """Get trait projection for a single prompt"""
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        capture = ActivationCapture()
        with HookManager(model) as hooks:
            hooks.add_forward_hook(f"model.layers.{layer}", capture.make_hook(f"layer_{layer}"))
            with torch.no_grad():
                _ = model(**inputs)

        # Get activation (mean over sequence)
        acts = capture.get(f"layer_{layer}")  # [1, seq_len, hidden_dim]
        acts_mean = acts.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]

        # Project onto vector
        vector_norm = vector / vector.norm()
        projection = (acts_mean @ vector_norm).item()

        return projection

    print(f"\nComputing projections...")

    pos_scores = []
    for prompt in tqdm(pos_scenarios, desc="Positive"):
        score = get_projection(prompt)
        pos_scores.append(score)

    neg_scores = []
    for prompt in tqdm(neg_scenarios, desc="Negative"):
        score = get_projection(prompt)
        neg_scores.append(score)

    # Analyze results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")

    pos_mean = np.mean(pos_scores)
    pos_std = np.std(pos_scores)
    neg_mean = np.mean(neg_scores)
    neg_std = np.std(neg_scores)
    separation = pos_mean - neg_mean

    print(f"Positive scenarios (should be HIGH):")
    print(f"  Mean: {pos_mean:+.3f}")
    print(f"  Std:  {pos_std:.3f}")
    print(f"  Range: [{min(pos_scores):+.3f}, {max(pos_scores):+.3f}]")

    print(f"\nNegative scenarios (should be LOW):")
    print(f"  Mean: {neg_mean:+.3f}")
    print(f"  Std:  {neg_std:.3f}")
    print(f"  Range: [{min(neg_scores):+.3f}, {max(neg_scores):+.3f}]")

    print(f"\nSeparation: {separation:+.3f}")

    # Verdict
    print(f"\n{'='*80}")
    if separation > 0:
        print("✅ CORRECT POLARITY: Positive scenarios score higher than negative")
        if separation > 0.5:
            print("✅ STRONG SEPARATION: Vector works well")
        elif separation > 0.2:
            print("⚠️  MODERATE SEPARATION: Vector works but could be stronger")
        else:
            print("⚠️  WEAK SEPARATION: Vector might need improvement")
    else:
        print("❌ INVERTED POLARITY: Negative scenarios score higher!")
        print("   This means the vector has the WRONG sign.")
        print("   The instruction-based vectors had this problem.")

    print(f"{'='*80}\n")

    # Comparison
    print("Sample scores:")
    print("\nPositive scenarios:")
    for i in range(min(3, len(pos_scenarios))):
        print(f"  [{i+1}] {pos_scenarios[i][:60]}...")
        print(f"      Score: {pos_scores[i]:+.3f}")

    print("\nNegative scenarios:")
    for i in range(min(3, len(neg_scenarios))):
        print(f"  [{i+1}] {neg_scenarios[i][:60]}...")
        print(f"      Score: {neg_scores[i]:+.3f}")

    return {
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'separation': separation,
        'correct_polarity': separation > 0
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--trait', type=str, required=True)
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--method', type=str, default='probe')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it')

    args = parser.parse_args()

    result = validate_vector_polarity(
        experiment=args.experiment,
        trait=args.trait,
        layer=args.layer,
        method=args.method,
        model_name=args.model
    )

    if result:
        print(f"\n{'='*80}")
        if result['correct_polarity']:
            print(f"✅ VALIDATION PASSED: Natural elicitation works!")
            print(f"   Separation: {result['separation']:+.3f}")
        else:
            print(f"❌ VALIDATION FAILED: Vector has wrong polarity")
            print(f"   Separation: {result['separation']:+.3f} (should be positive)")
        print(f"{'='*80}")
