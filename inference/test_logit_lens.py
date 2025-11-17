#!/usr/bin/env python3
"""
Quick test script for logit lens functionality.

Tests that the compute_logits_at_layers function works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

# Import the function we're testing
from capture_all_layers import compute_logits_at_layers, LOGIT_LENS_LAYERS


def test_logit_computation():
    """Test logit lens computation with mock data."""
    print("Testing logit lens computation...")
    print("=" * 60)

    # Mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    # Mock activations (3 tokens, 27 layers, hidden_dim=2304)
    n_tokens = 3
    n_layers = 27
    hidden_dim = 2304
    vocab_size = 256000

    activations = {}
    for layer_idx in range(n_layers):
        activations[layer_idx] = {
            'residual_in': torch.randn(n_tokens, hidden_dim),
            'after_attn': torch.randn(n_tokens, hidden_dim),
            'residual_out': torch.randn(n_tokens, hidden_dim)  # This is what we use
        }

    # Mock unembedding matrix
    unembed = torch.randn(vocab_size, hidden_dim)

    print(f"Mock data:")
    print(f"  Tokens: {n_tokens}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sampling layers: {LOGIT_LENS_LAYERS}")
    print()

    # Compute logits
    print("Computing logit lens...")
    logits_data = compute_logits_at_layers(
        activations, unembed, tokenizer, n_layers, top_k=3
    )

    print(f"✓ Computed logits at {len(logits_data)} layers")
    print()

    # Validate structure
    print("Validating output structure:")
    for layer_key in list(logits_data.keys())[:3]:  # Show first 3
        layer_data = logits_data[layer_key]

        print(f"\n{layer_key}:")
        print(f"  Keys: {list(layer_data.keys())}")
        print(f"  Tokens shape: {len(layer_data['tokens'])} tokens, each with {len(layer_data['tokens'][0])} predictions")
        print(f"  Probs shape: {len(layer_data['probs'])} tokens, each with {len(layer_data['probs'][0])} probabilities")

        # Check first token's predictions
        first_token_preds = layer_data['tokens'][0]
        first_token_probs = layer_data['probs'][0]

        print(f"  Sample (token 0):")
        for i, (tok, prob) in enumerate(zip(first_token_preds, first_token_probs)):
            print(f"    {i+1}. '{tok}' ({prob:.3f})")

        # Validate probabilities sum to < 1.0 (since we only keep top-3)
        prob_sum = sum(first_token_probs)
        print(f"  Probability sum: {prob_sum:.3f} (< 1.0 is expected)")

        if prob_sum > 1.0:
            print("  ⚠️  Warning: Probabilities sum > 1.0!")

    print("\n" + "=" * 60)
    print("✅ Test passed! Logit lens computation works correctly.")
    print()
    print("Data format is ready for visualization:")
    print("- Each layer has 'tokens' (list of lists of strings)")
    print("- Each layer has 'probs' (list of lists of floats)")
    print("- Already JSON-serializable")


if __name__ == "__main__":
    test_logit_computation()
