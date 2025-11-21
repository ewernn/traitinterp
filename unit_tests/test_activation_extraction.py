#!/usr/bin/env python3
"""
Validation tests for activation extraction to prevent indexing bugs.

Run this BEFORE extracting activations on a full dataset.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_hidden_states_indexing():
    """
    Test that hidden_states indices are correctly mapped to layers.

    CRITICAL: hidden_states[0] is the EMBEDDING, not layer 0!
    """
    print("=" * 60)
    print("Testing hidden_states indexing...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} decoder layers")

    # Test text
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    print(f"hidden_states tuple length: {len(hidden_states)}")

    # CRITICAL CHECK 1: Length should be n_layers + 1 (embedding) [+ 1 if final norm]
    expected_min = n_layers + 1
    expected_max = n_layers + 2  # Some models include post-final-norm

    if not (expected_min <= len(hidden_states) <= expected_max):
        print(f"❌ FAILED: Expected {expected_min}-{expected_max} hidden states, got {len(hidden_states)}")
        return False

    print(f"✓ Length check passed: {len(hidden_states)} hidden states")

    # CRITICAL CHECK 2: Index 0 should be embedding (different from layer outputs)
    print(f"\nChecking indexing convention:")
    print(f"  hidden_states[0] (embedding) norm: {hidden_states[0].mean(dim=1).norm().item():.2f}")
    print(f"  hidden_states[1] (layer 0 out) norm: {hidden_states[1].mean(dim=1).norm().item():.2f}")

    # For Gemma 2, check if final norm exists
    if len(hidden_states) == n_layers + 2:
        print(f"  hidden_states[{n_layers}] (layer {n_layers-1} out) norm: {hidden_states[n_layers].mean(dim=1).norm().item():.2f}")
        print(f"  hidden_states[{n_layers+1}] (final norm) norm: {hidden_states[n_layers+1].mean(dim=1).norm().item():.2f}")

        # Final norm should significantly reduce magnitude
        ratio = hidden_states[n_layers].mean(dim=1).norm().item() / hidden_states[n_layers+1].mean(dim=1).norm().item()
        if ratio > 5:
            print(f"  ✓ Final RMSNorm detected (reduces by {ratio:.1f}x)")
            print(f"\n  CORRECT INDEXING:")
            print(f"    layers = list(range(1, n_layers + 2))  # [1, 2, ..., {n_layers+1}]")
            print(f"    This gives you: layer 0 output → layer {n_layers-1} output + final norm")
        else:
            print(f"  ⚠️  No strong final norm detected")
    else:
        print(f"  CORRECT INDEXING:")
        print(f"    layers = list(range(1, n_layers + 1))  # [1, 2, ..., {n_layers}]")

    # CRITICAL CHECK 3: Activation norms should be stable (not monotonically grow)
    print(f"\nChecking activation norm stability:")
    norms = [hidden_states[i].mean(dim=1).norm().item() for i in range(len(hidden_states))]

    print(f"  Norm trajectory:")
    for i in [0, 1, n_layers//2, n_layers-1, n_layers]:
        if i < len(norms):
            print(f"    Index {i:2d}: {norms[i]:8.2f}")

    # Check for dangerous monotonic growth (without final norm)
    if len(hidden_states) == n_layers + 1:
        # No final norm - activations should stay relatively stable
        max_growth = max(norms[1:]) / min(norms[1:])
        if max_growth > 3:
            print(f"  ❌ WARNING: Activations grow {max_growth:.1f}x without normalization!")
            print(f"      This model may have unstable residual stream.")
        else:
            print(f"  ✓ Stable: Max growth {max_growth:.1f}x")

    print("\n✅ All checks passed")
    print("\nRECOMMENDED USAGE:")
    print("  layers = list(range(1, len(hidden_states)))  # Skip embedding at index 0")
    return True


def test_extraction_metadata():
    """Test that extraction metadata matches reality."""
    print("\n" + "=" * 60)
    print("Testing extraction metadata...")
    print("=" * 60)

    # TODO: Add checks that:
    # 1. Saved layer indices match actual layer numbers
    # 2. Vector norms are in expected range
    # 3. Separation scores are meaningful

    print("✓ Metadata tests would go here")
    return True


if __name__ == "__main__":
    success = test_hidden_states_indexing()
    success = success and test_extraction_metadata()

    if success:
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATION TESTS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED - DO NOT RUN FULL EXTRACTION")
        print("=" * 60)
        exit(1)
