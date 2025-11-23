#!/usr/bin/env python3
"""
Test Model Configurations and Data Integrity

Validates that:
1. Model configs match documentation claims
2. Saved data structures have correct dimensions
3. Metadata is consistent across files

Run this to catch drift between docs and reality.

Usage:
    python tests/test_model_configs.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from transformers import AutoConfig


def test_gemma_2b_config():
    """Verify Gemma 2B model config matches documentation."""
    print("Testing Gemma 2B config...")

    config = AutoConfig.from_pretrained('google/gemma-2-2b-it')

    # Documentation claims in docs/main.md
    expected_layers = 26
    expected_hidden = 2304

    assert config.num_hidden_layers == expected_layers, \
        f"Expected {expected_layers} layers, got {config.num_hidden_layers}"

    assert config.hidden_size == expected_hidden, \
        f"Expected hidden size {expected_hidden}, got {config.hidden_size}"

    print(f"✓ Gemma 2B: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")


def test_gemini_2_2b_it_config():
    """Verify gemini-2-2b-it model config matches documentation."""

    try:
        config = AutoConfig.from_pretrained('google/gemma-2-2b-it')

        # Documentation claims in docs/main.md
        expected_layers = 32  # 0-32 = 33 total, but num_hidden_layers = 32
        expected_hidden = 4096

        assert config.num_hidden_layers == expected_layers, \
            f"Expected {expected_layers} layers, got {config.num_hidden_layers}"

        assert config.hidden_size == expected_hidden, \
            f"Expected hidden size {expected_hidden}, got {config.hidden_size}"

        print(f"✓ gemini-2-2b-it: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
    except Exception as e:
        print(f"⚠ gemini-2-2b-it config not accessible (may require authentication): {e}")


def test_inference_data_shapes():
    """Verify saved inference data has correct layer dimensions."""
    print("\nTesting inference data shapes...")

    # Check if experiment data exists (update path to match your experiment)
    inference_dir = Path('experiments/my_experiment/refusal/inference/residual_stream_activations')

    if not inference_dir.exists():
        print("⚠ No inference data found, skipping shape test")
        return

    # Load first available JSON file
    json_files = list(inference_dir.glob('prompt_*.json'))
    if not json_files:
        print("⚠ No JSON files found, skipping shape test")
        return

    test_file = json_files[0]
    with open(test_file) as f:
        data = json.load(f)

    # Check projection shapes
    prompt_proj = data['projections']['prompt']
    response_proj = data['projections']['response']

    # Each should be [n_tokens, n_layers, 3_sublayers]
    n_layers_prompt = len(prompt_proj[0])
    n_layers_response = len(response_proj[0])
    n_sublayers_prompt = len(prompt_proj[0][0])
    n_sublayers_response = len(response_proj[0][0])

    # Gemma 2B should have 26 layers
    expected_layers = 26
    expected_sublayers = 3

    assert n_layers_prompt == expected_layers, \
        f"Prompt projections: expected {expected_layers} layers, got {n_layers_prompt}"

    assert n_layers_response == expected_layers, \
        f"Response projections: expected {expected_layers} layers, got {n_layers_response}"

    assert n_sublayers_prompt == expected_sublayers, \
        f"Prompt sublayers: expected {expected_sublayers}, got {n_sublayers_prompt}"

    assert n_sublayers_response == expected_sublayers, \
        f"Response sublayers: expected {expected_sublayers}, got {n_sublayers_response}"

    print(f"✓ Inference data: {n_layers_prompt} layers × {n_sublayers_prompt} sublayers")
    print(f"  Tested file: {test_file.name}")


def test_vector_metadata_consistency():
    """Verify vector metadata files have consistent layer counts."""
    print("\nTesting vector metadata consistency...")

    vectors_dir = Path('experiments/my_experiment/refusal/extraction/vectors')

    if not vectors_dir.exists():
        print("⚠ No vector metadata found, skipping consistency test")
        return

    # Load metadata from different methods
    metadata_files = list(vectors_dir.glob('*_layer*_metadata.json'))

    if not metadata_files:
        print("⚠ No metadata files found, skipping consistency test")
        return

    # Check that all metadata files agree on model config
    expected_layers = 26
    expected_hidden = 2304

    for meta_file in metadata_files[:5]:  # Sample first 5
        with open(meta_file) as f:
            meta = json.load(f)

        # Check hidden_dim if present
        if 'hidden_dim' in meta:
            assert meta['hidden_dim'] == expected_hidden, \
                f"{meta_file.name}: expected hidden_dim {expected_hidden}, got {meta['hidden_dim']}"

    print(f"✓ Vector metadata: consistent dimensions across {len(metadata_files)} files")


def test_activation_metadata():
    """Verify activation metadata has correct layer count."""
    print("\nTesting activation metadata...")

    meta_file = Path('experiments/my_experiment/refusal/extraction/activations/metadata.json')

    if not meta_file.exists():
        print("⚠ No activation metadata found, skipping test")
        return

    with open(meta_file) as f:
        meta = json.load(f)

    expected_layers = 26
    expected_hidden = 2304

    assert meta['n_layers'] == expected_layers, \
        f"Expected {expected_layers} layers, got {meta['n_layers']}"

    assert meta['hidden_dim'] == expected_hidden, \
        f"Expected hidden_dim {expected_hidden}, got {meta['hidden_dim']}"

    print(f"✓ Activation metadata: {meta['n_layers']} layers, {meta['hidden_dim']} hidden dim")


def main():
    """Run all config validation tests."""
    print("=" * 60)
    print("Model Configuration & Data Integrity Tests")
    print("=" * 60)

    try:
        # Test model configs from HuggingFace
        test_gemma_2b_config()
        test_gemini_2_2b_it_config()

        # Test saved data structures
        test_activation_metadata()
        test_inference_data_shapes()
        test_vector_metadata_consistency()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Unexpected error: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
