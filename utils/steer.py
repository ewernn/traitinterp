#!/usr/bin/env python3
"""
Quick Steering Script

Test steering with any vector.

Usage:
    # Basic steering (finds vector automatically)
    python steer.py \
        --experiment my_experiment \
        --trait my_trait \
        --prompt "Test prompt" \
        --strength 3

    # Test multiple strengths
    python steer.py \
        --experiment my_experiment \
        --trait my_trait \
        --prompt "Test prompt" \
        --strengths -5,-3,-1,0,1,3,5

    # Use specific vector file
    python steer.py \
        --vector-path experiments/my_exp/my_trait/vectors/probe_layer16.pt \
        --prompt "Test prompt" \
        --strength 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_vector(experiment, trait, method='probe', layer=16):
    """Find vector in common directory structures."""
    possible_paths = [
        Path(f"experiments/{experiment}/{trait}/extraction/vectors/{method}_layer{layer}.pt"),
        Path(f"experiments/{experiment}/{trait}-extraction/extraction/vectors/{method}_layer{layer}.pt"),
        Path(f"experiments/{experiment}/{trait}/vectors/{method}_layer{layer}.pt"),
    ]

    for vector_path in possible_paths:
        if vector_path.exists():
            print(f"✅ Found vector: {vector_path}")
            return torch.load(vector_path).float(), vector_path

    print(f"❌ Vector not found. Tried:")
    for p in possible_paths:
        print(f"   {p}")
    return None, None

def load_vector(vector_path):
    """Load vector from path."""
    vector_path = Path(vector_path)
    if not vector_path.exists():
        print(f"❌ Vector not found: {vector_path}")
        return None
    print(f"✅ Loaded vector: {vector_path}")
    return torch.load(vector_path).float()

def steer(model, tokenizer, prompt, vector, layer, strength, max_tokens=100):
    """Generate with steering."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if vector is None or strength == 0:
        # Baseline
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Steering
    vector = vector.to(model.device)

    def hook(module, input, output):
        hidden_states = output[0]
        norm_vec = vector / (vector.norm() + 1e-8)
        hidden_states[:, -1, :] += strength * norm_vec
        return (hidden_states,)

    handle = model.model.layers[layer].register_forward_hook(hook)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        handle.remove()

    return response

def main():
    parser = argparse.ArgumentParser(description='Quick steering test')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (auto-finds vector)')
    parser.add_argument('--trait', type=str, default=None,
                       help='Trait name (auto-finds vector)')
    parser.add_argument('--vector-path', type=str, default=None,
                       help='Direct path to vector .pt file')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--strength', type=float, default=3.0)
    parser.add_argument('--strengths', type=str, default=None,
                       help='Comma-separated strengths, e.g., -3,0,3')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--method', type=str, default='probe')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--max-tokens', type=int, default=100)

    args = parser.parse_args()

    print("="*80)
    print("STEERING TEST")
    print("="*80)

    # Load vector
    if args.vector_path:
        vector = load_vector(args.vector_path)
    elif args.experiment and args.trait:
        vector, _ = find_vector(args.experiment, args.trait, args.method, args.layer)
    else:
        print("❌ Must provide either --vector-path OR (--experiment + --trait)")
        return

    if vector is None:
        return

    print(f"Prompt: {args.prompt}")

    # Load model
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    if args.strengths:
        # Test multiple strengths
        strengths = [float(s) for s in args.strengths.split(',')]

        print(f"\n{'='*80}")
        print("TESTING MULTIPLE STRENGTHS")
        print(f"{'='*80}")

        for strength in strengths:
            response = steer(model, tokenizer, args.prompt, vector, args.layer, strength, args.max_tokens)
            print(f"\nStrength {strength:+.1f}:")
            print(f"  {response}")

    else:
        # Single strength test
        print(f"\nStrength: {args.strength:+.1f}")

        # Baseline (strength 0)
        print(f"\n{'='*80}")
        print("BASELINE (no steering)")
        print(f"{'='*80}")
        baseline = steer(model, tokenizer, args.prompt, vector, args.layer, 0, args.max_tokens)
        print(baseline)

        # Steered
        print(f"\n{'='*80}")
        print(f"STEERED (strength {args.strength:+.1f})")
        print(f"{'='*80}")
        steered = steer(model, tokenizer, args.prompt, vector, args.layer, args.strength, args.max_tokens)
        print(steered)

        # Comparison
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"Baseline length: {len(baseline.split())} words")
        print(f"Steered length:  {len(steered.split())} words")

        if baseline == steered:
            print("\n⚠️  WARNING: Responses are identical - steering may not be working")
        else:
            print(f"\n✅ Steering effect observed")

if __name__ == '__main__':
    main()
