"""
Test that attn_contribution + mlp_contribution ≈ residual delta.

Verifies the new auto-detected contribution components work correctly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import get_hook_path, detect_contribution_paths, MultiLayerCapture, CaptureHook, HookManager


def test_additivity(model, tokenizer, test_layers=[10, 15, 20, 24]):
    """Test that contributions sum to residual delta."""

    # Check what architecture was detected
    contrib_paths = detect_contribution_paths(model)
    print(f"Detected architecture:")
    print(f"  attn_contribution -> {contrib_paths['attn_contribution']}")
    print(f"  mlp_contribution -> {contrib_paths['mlp_contribution']}")
    print()

    # Simple test input
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Testing additivity on layers {test_layers}...")
    print(f"{'Layer':<6} {'Residual Δ':<12} {'Attn+MLP':<12} {'Diff':<12} {'Ratio':<8}")
    print("-" * 52)

    errors = []

    for layer in test_layers:
        # We need residual at layer-1 and layer, plus contributions at layer
        with HookManager(model) as hooks:
            storage = {
                'residual_prev': [],
                'residual_curr': [],
                'attn_contrib': [],
                'mlp_contrib': [],
            }

            # Hook residual at layer-1
            def make_residual_prev_hook():
                def hook(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    storage['residual_prev'].append(t.detach())
                return hook
            hooks.add_forward_hook(get_hook_path(layer - 1, 'residual'), make_residual_prev_hook())

            # Hook residual at layer
            def make_residual_curr_hook():
                def hook(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    storage['residual_curr'].append(t.detach())
                return hook
            hooks.add_forward_hook(get_hook_path(layer, 'residual'), make_residual_curr_hook())

            # Hook attn contribution
            def make_attn_hook():
                def hook(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    storage['attn_contrib'].append(t.detach())
                return hook
            hooks.add_forward_hook(get_hook_path(layer, 'attn_contribution', model=model), make_attn_hook())

            # Hook mlp contribution
            def make_mlp_hook():
                def hook(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    storage['mlp_contrib'].append(t.detach())
                return hook
            hooks.add_forward_hook(get_hook_path(layer, 'mlp_contribution', model=model), make_mlp_hook())

            # Forward pass
            with torch.no_grad():
                model(**inputs)

        # Compute
        residual_prev = storage['residual_prev'][0]
        residual_curr = storage['residual_curr'][0]
        attn_contrib = storage['attn_contrib'][0]
        mlp_contrib = storage['mlp_contrib'][0]

        # Residual delta
        residual_delta = residual_curr - residual_prev

        # Sum of contributions
        contrib_sum = attn_contrib + mlp_contrib

        # Compare (use mean over all positions and hidden dims)
        delta_norm = residual_delta.norm().item()
        contrib_norm = contrib_sum.norm().item()
        diff = (residual_delta - contrib_sum).norm().item()
        ratio = contrib_norm / delta_norm if delta_norm > 0 else float('inf')

        print(f"{layer:<6} {delta_norm:<12.2f} {contrib_norm:<12.2f} {diff:<12.4f} {ratio:<8.4f}")

        errors.append(diff / delta_norm if delta_norm > 0 else 0)

    print()
    avg_error = sum(errors) / len(errors)
    print(f"Average relative error: {avg_error:.6f}")

    if avg_error < 0.05:  # 5% tolerance for bfloat16 precision
        print("✓ Additivity verified! Contributions sum to residual delta.")
    else:
        print("✗ Additivity NOT verified. Check hook points.")

    return avg_error


def compare_raw_vs_contribution(model, tokenizer, test_layers=[10, 15, 20, 24]):
    """Compare raw attn_out/mlp_out vs contribution components."""

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\nComparing raw outputs vs contributions...")
    print(f"{'Layer':<6} {'Raw Δ':<12} {'Contrib Δ':<12} {'Actual Δ':<12} {'Raw Err%':<10} {'Contrib Err%':<10}")
    print("-" * 70)

    for layer in test_layers:
        with HookManager(model) as hooks:
            storage = {
                'residual_prev': [],
                'residual_curr': [],
                'attn_raw': [],
                'mlp_raw': [],
                'attn_contrib': [],
                'mlp_contrib': [],
            }

            # Residual hooks
            def make_hook(key):
                def hook(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    storage[key].append(t.detach())
                return hook

            hooks.add_forward_hook(get_hook_path(layer - 1, 'residual'), make_hook('residual_prev'))
            hooks.add_forward_hook(get_hook_path(layer, 'residual'), make_hook('residual_curr'))
            hooks.add_forward_hook(get_hook_path(layer, 'attn_out'), make_hook('attn_raw'))
            hooks.add_forward_hook(get_hook_path(layer, 'mlp_out'), make_hook('mlp_raw'))
            hooks.add_forward_hook(get_hook_path(layer, 'attn_contribution', model=model), make_hook('attn_contrib'))
            hooks.add_forward_hook(get_hook_path(layer, 'mlp_contribution', model=model), make_hook('mlp_contrib'))

            with torch.no_grad():
                model(**inputs)

        actual_delta = (storage['residual_curr'][0] - storage['residual_prev'][0]).norm().item()
        raw_sum = (storage['attn_raw'][0] + storage['mlp_raw'][0]).norm().item()
        contrib_sum = (storage['attn_contrib'][0] + storage['mlp_contrib'][0]).norm().item()

        raw_err = abs(raw_sum - actual_delta) / actual_delta * 100
        contrib_err = abs(contrib_sum - actual_delta) / actual_delta * 100

        print(f"{layer:<6} {raw_sum:<12.1f} {contrib_sum:<12.1f} {actual_delta:<12.1f} {raw_err:<10.1f} {contrib_err:<10.1f}")

    print()


def main():
    print("Loading google/gemma-2-2b...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    print(f"Model loaded on {model.device}")
    print()

    test_additivity(model, tokenizer)
    compare_raw_vs_contribution(model, tokenizer)


if __name__ == "__main__":
    main()
