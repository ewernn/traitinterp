"""
Test extraction methods on synthetic data.
"""

import torch
from traitlens import (
    MeanDifferenceMethod,
    ICAMethod,
    ProbeMethod,
    GradientMethod,
    get_method
)


def generate_synthetic_data(n_pos=100, n_neg=100, hidden_dim=512, separation=2.0):
    """Generate synthetic positive and negative activations."""
    # Positive examples: centered at +separation
    pos_acts = torch.randn(n_pos, hidden_dim) + separation

    # Negative examples: centered at -separation
    neg_acts = torch.randn(n_neg, hidden_dim) - separation

    return pos_acts, neg_acts


def test_method(method, name, pos_acts, neg_acts):
    """Test a single extraction method."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        result = method.extract(pos_acts, neg_acts)

        print(f"✅ {name} succeeded")
        print(f"   Vector shape: {result['vector'].shape}")
        print(f"   Vector norm: {result['vector'].norm():.4f}")

        # Test projection
        pos_proj = pos_acts @ result['vector']
        neg_proj = neg_acts @ result['vector']
        separation = (pos_proj.mean() - neg_proj.mean()).item()

        print(f"   Pos projection: {pos_proj.mean():.4f} ± {pos_proj.std():.4f}")
        print(f"   Neg projection: {neg_proj.mean():.4f} ± {neg_proj.std():.4f}")
        print(f"   Separation: {separation:.4f}")

        # Print method-specific outputs
        for key, value in result.items():
            if key != 'vector' and not key.endswith('_proj'):
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        print(f"   {key}: {value.item():.4f}")
                    elif value.ndim == 1 and value.shape[0] <= 10:
                        print(f"   {key}: {value.tolist()}")
                else:
                    print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return False


def main():
    """Run all method tests."""
    print("Generating synthetic data...")
    pos_acts, neg_acts = generate_synthetic_data(
        n_pos=100,
        n_neg=100,
        hidden_dim=512,
        separation=2.0
    )

    print(f"Positive acts: {pos_acts.shape}")
    print(f"Negative acts: {neg_acts.shape}")

    # Test all methods
    methods = [
        (MeanDifferenceMethod(), "Mean Difference"),
        (ICAMethod(), "ICA"),
        (ProbeMethod(), "Linear Probe"),
        (GradientMethod(), "Gradient Optimization"),
    ]

    results = []
    for method, name in methods:
        success = test_method(method, name, pos_acts, neg_acts)
        results.append((name, success))

    # Test get_method function
    print(f"\n{'='*60}")
    print("Testing get_method() convenience function")
    print(f"{'='*60}")

    for method_name in ['mean_diff', 'ica', 'probe', 'gradient']:
        try:
            method = get_method(method_name)
            print(f"✅ get_method('{method_name}') -> {method.__class__.__name__}")
        except Exception as e:
            print(f"❌ get_method('{method_name}') failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All methods working!")
    else:
        print("\n⚠️  Some methods failed (likely missing dependencies)")
        print("   Install scikit-learn: pip install scikit-learn")


if __name__ == "__main__":
    main()
