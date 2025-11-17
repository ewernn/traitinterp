#!/usr/bin/env python3
"""
Verify Tier 2 Data Format

Loads a captured Tier 2 file and verifies it matches the specification.

Usage:
    python inference/verify_tier2_format.py experiments/gemma_2b_cognitive_nov20/refusal/inference/residual_stream_activations/prompt_0.pt
"""

import torch
import sys
from pathlib import Path


def verify_tier2_format(file_path: str) -> bool:
    """
    Verify a Tier 2 .pt file matches the specification.

    Returns:
        True if valid, False otherwise
    """
    print(f"Loading: {file_path}")
    print(f"{'='*60}")

    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False

    # Check top-level keys
    required_keys = ['prompt', 'response', 'projections', 'attention_weights', 'metadata']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"❌ Missing top-level keys: {missing_keys}")
        return False
    print(f"✓ Top-level keys: {list(data.keys())}")

    # Check prompt structure
    prompt_keys = ['text', 'tokens', 'token_ids', 'n_tokens']
    missing = [k for k in prompt_keys if k not in data['prompt']]
    if missing:
        print(f"❌ Missing prompt keys: {missing}")
        return False
    print(f"✓ Prompt keys: {list(data['prompt'].keys())}")
    print(f"  - Text: {data['prompt']['text'][:80]}...")
    print(f"  - Tokens: {data['prompt']['n_tokens']}")

    # Check response structure
    response_keys = ['text', 'tokens', 'token_ids', 'n_tokens']
    missing = [k for k in response_keys if k not in data['response']]
    if missing:
        print(f"❌ Missing response keys: {missing}")
        return False
    print(f"✓ Response keys: {list(data['response'].keys())}")
    print(f"  - Text: {data['response']['text'][:80]}...")
    print(f"  - Tokens: {data['response']['n_tokens']}")

    # Check projections structure
    proj_keys = ['prompt', 'response']
    missing = [k for k in proj_keys if k not in data['projections']]
    if missing:
        print(f"❌ Missing projections keys: {missing}")
        return False

    prompt_proj = data['projections']['prompt']
    response_proj = data['projections']['response']

    # Infer n_layers from data (Gemma 2B has 26, others may differ)
    n_layers = prompt_proj.shape[1] if prompt_proj.ndim >= 2 else None

    print(f"✓ Projections:")
    print(f"  - Prompt shape: {prompt_proj.shape} (expected: [n_tokens, n_layers, 3])")
    print(f"  - Response shape: {response_proj.shape} (expected: [n_tokens, n_layers, 3])")

    # Verify shapes
    if prompt_proj.ndim != 3 or prompt_proj.shape[2] != 3:
        print(f"❌ Invalid prompt projections shape (should be [n_tokens, n_layers, 3])")
        return False

    if response_proj.ndim != 3 or response_proj.shape[2] != 3:
        print(f"❌ Invalid response projections shape (should be [n_tokens, n_layers, 3])")
        return False

    # Verify both have same number of layers
    if prompt_proj.shape[1] != response_proj.shape[1]:
        print(f"❌ Layer count mismatch: prompt={prompt_proj.shape[1]}, response={response_proj.shape[1]}")
        return False

    print(f"  - Layers: {n_layers} (auto-detected from model)")

    # Check token count matches
    if prompt_proj.shape[0] != data['prompt']['n_tokens']:
        print(f"❌ Prompt projection tokens ({prompt_proj.shape[0]}) != prompt n_tokens ({data['prompt']['n_tokens']})")
        return False

    if response_proj.shape[0] != data['response']['n_tokens']:
        print(f"❌ Response projection tokens ({response_proj.shape[0]}) != response n_tokens ({data['response']['n_tokens']})")
        return False

    # Check attention weights
    attn_keys = ['prompt', 'response']
    missing = [k for k in attn_keys if k not in data['attention_weights']]
    if missing:
        print(f"❌ Missing attention_weights keys: {missing}")
        return False

    prompt_attn = data['attention_weights']['prompt']
    response_attn = data['attention_weights']['response']

    print(f"✓ Attention weights:")
    print(f"  - Prompt: {len(prompt_attn)} layers")

    # Check prompt attention has all layers (use n_layers from projections)
    expected_layers = set([f'layer_{i}' for i in range(n_layers)])
    actual_layers = set(prompt_attn.keys())
    if expected_layers != actual_layers:
        print(f"❌ Prompt attention missing layers: {expected_layers - actual_layers}")
        return False

    # Check response attention is a list of dicts (one per token)
    if not isinstance(response_attn, list):
        print(f"❌ Response attention should be a list, got {type(response_attn)}")
        return False

    print(f"  - Response: {len(response_attn)} tokens")

    # Verify each response token has all layers
    if response_attn:  # If not empty
        first_token_attn = response_attn[0]
        if not isinstance(first_token_attn, dict):
            print(f"❌ Response attention items should be dicts, got {type(first_token_attn)}")
            return False

        expected_layers = set([f'layer_{i}' for i in range(n_layers)])
        actual_layers = set(first_token_attn.keys())
        if expected_layers != actual_layers:
            print(f"❌ Response token attention missing layers: {expected_layers - actual_layers}")
            return False

        print(f"  - Each response token has all {n_layers} layers ✓")

    # Check metadata
    metadata_keys = ['trait', 'trait_display_name', 'vector_path', 'model', 'capture_date', 'temperature']
    missing = [k for k in metadata_keys if k not in data['metadata']]
    if missing:
        print(f"❌ Missing metadata keys: {missing}")
        return False

    print(f"✓ Metadata:")
    print(f"  - Trait: {data['metadata']['trait']} ({data['metadata']['trait_display_name']})")
    print(f"  - Model: {data['metadata']['model']}")
    print(f"  - Vector: {data['metadata']['vector_path']}")
    print(f"  - Temperature: {data['metadata']['temperature']}")
    print(f"  - Capture date: {data['metadata']['capture_date']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ All checks passed! File is valid Tier 2 format.")
    print(f"{'='*60}")
    print(f"\nData Summary:")
    print(f"  Prompt: {data['prompt']['n_tokens']} tokens")
    print(f"  Response: {data['response']['n_tokens']} tokens")
    print(f"  Trait: {data['metadata']['trait']}")
    print(f"  Projections captured at: 27 layers × 3 sublayers = 81 checkpoints")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference/verify_tier2_format.py <path_to_prompt_N.pt>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    success = verify_tier2_format(file_path)
    sys.exit(0 if success else 1)
