"""
Tests for core/methods.py - extraction methods.

Run: pytest core/_tests/test_methods.py -v
"""

import pytest
import torch

from core.methods import (
    MeanDifferenceMethod,
    ProbeMethod,
    GradientMethod,
    RandomBaselineMethod,
    get_method,
)


# =============================================================================
# MeanDifferenceMethod tests
# =============================================================================

class TestMeanDifferenceMethod:
    """Tests for MeanDifferenceMethod."""

    def test_returns_unit_vector(self, pos_activations, neg_activations):
        """Output vector is unit normalized."""
        method = MeanDifferenceMethod()
        result = method.extract(pos_activations, neg_activations)
        norm = result['vector'].norm().item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_direction_correct(self, hidden_dim):
        """Vector points from neg mean to pos mean."""
        # Create activations where pos is shifted in dim 0
        pos = torch.zeros(10, hidden_dim)
        pos[:, 0] = 10.0
        neg = torch.zeros(10, hidden_dim)
        neg[:, 0] = -10.0

        method = MeanDifferenceMethod()
        result = method.extract(pos, neg)

        # Vector should have positive component in dim 0
        assert result['vector'][0].item() > 0.5

    def test_dtype_preservation(self, hidden_dim):
        """Output dtype matches input dtype."""
        pos = torch.randn(10, hidden_dim, dtype=torch.bfloat16)
        neg = torch.randn(10, hidden_dim, dtype=torch.bfloat16)

        method = MeanDifferenceMethod()
        result = method.extract(pos, neg)
        assert result['vector'].dtype == torch.bfloat16

    def test_returns_means(self, pos_activations, neg_activations):
        """Returns pos_mean and neg_mean in result dict."""
        method = MeanDifferenceMethod()
        result = method.extract(pos_activations, neg_activations)
        assert 'pos_mean' in result
        assert 'neg_mean' in result
        assert result['pos_mean'].shape == (pos_activations.shape[-1],)

    def test_large_values_stable(self, hidden_dim):
        """Large activation values don't cause overflow."""
        pos = torch.randn(10, hidden_dim) * 1e4
        neg = torch.randn(10, hidden_dim) * 1e4

        method = MeanDifferenceMethod()
        result = method.extract(pos, neg)
        assert torch.isfinite(result['vector']).all()


# =============================================================================
# ProbeMethod tests
# =============================================================================

class TestProbeMethod:
    """Tests for ProbeMethod."""

    def test_returns_unit_vector(self, pos_activations, neg_activations):
        """Output vector is unit normalized."""
        method = ProbeMethod()
        result = method.extract(pos_activations, neg_activations)
        norm = result['vector'].norm().item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_separates_easy_data(self, pos_activations, neg_activations):
        """Achieves high accuracy on well-separated data."""
        method = ProbeMethod()
        result = method.extract(pos_activations, neg_activations)
        # pos/neg_activations are shifted by +/-2, should be easy
        assert result['train_acc'] > 0.9

    def test_returns_bias(self, pos_activations, neg_activations):
        """Returns bias in result dict."""
        method = ProbeMethod()
        result = method.extract(pos_activations, neg_activations)
        assert 'bias' in result
        assert result['bias'].ndim == 0  # scalar

    def test_dtype_preservation(self, hidden_dim):
        """Output dtype matches input dtype."""
        pos = torch.randn(20, hidden_dim, dtype=torch.bfloat16) + 2.0
        neg = torch.randn(20, hidden_dim, dtype=torch.bfloat16) - 2.0

        method = ProbeMethod()
        result = method.extract(pos, neg)
        assert result['vector'].dtype == torch.bfloat16

    def test_different_scales_work(self, hidden_dim):
        """Row normalization handles different activation scales."""
        # Simulate Gemma 3 scale (much larger activations)
        pos = (torch.randn(20, hidden_dim) + 2.0) * 1000
        neg = (torch.randn(20, hidden_dim) - 2.0) * 1000

        method = ProbeMethod()
        result = method.extract(pos, neg)
        # Should still separate well
        assert result['train_acc'] > 0.8


# =============================================================================
# GradientMethod tests
# =============================================================================

class TestGradientMethod:
    """Tests for GradientMethod."""

    def test_returns_unit_vector(self, pos_activations, neg_activations):
        """Output vector is unit normalized."""
        method = GradientMethod()
        result = method.extract(pos_activations, neg_activations, num_steps=50)
        norm = result['vector'].norm().item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_positive_separation(self, pos_activations, neg_activations):
        """Optimization achieves positive separation."""
        method = GradientMethod()
        result = method.extract(pos_activations, neg_activations, num_steps=50)
        assert result['final_separation'] > 0

    def test_dtype_preservation(self, hidden_dim):
        """Output dtype matches input dtype."""
        pos = torch.randn(20, hidden_dim, dtype=torch.bfloat16) + 2.0
        neg = torch.randn(20, hidden_dim, dtype=torch.bfloat16) - 2.0

        method = GradientMethod()
        result = method.extract(pos, neg, num_steps=20)
        assert result['vector'].dtype == torch.bfloat16

    def test_more_steps_better_separation(self, overlapping_pos, overlapping_neg):
        """More optimization steps improve separation."""
        method = GradientMethod()
        result_few = method.extract(overlapping_pos, overlapping_neg, num_steps=10)
        result_many = method.extract(overlapping_pos, overlapping_neg, num_steps=100)
        # More steps should give equal or better separation
        assert result_many['final_separation'] >= result_few['final_separation'] * 0.9


# =============================================================================
# RandomBaselineMethod tests
# =============================================================================

class TestRandomBaselineMethod:
    """Tests for RandomBaselineMethod."""

    def test_returns_unit_vector(self, pos_activations, neg_activations):
        """Output vector is unit normalized."""
        method = RandomBaselineMethod()
        result = method.extract(pos_activations, neg_activations)
        norm = result['vector'].norm().item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_seed_reproducible(self, pos_activations, neg_activations):
        """Same seed gives same vector."""
        method = RandomBaselineMethod()
        result1 = method.extract(pos_activations, neg_activations, seed=42)
        result2 = method.extract(pos_activations, neg_activations, seed=42)
        assert torch.equal(result1['vector'], result2['vector'])

    def test_different_seeds_different(self, pos_activations, neg_activations):
        """Different seeds give different vectors."""
        method = RandomBaselineMethod()
        result1 = method.extract(pos_activations, neg_activations, seed=42)
        result2 = method.extract(pos_activations, neg_activations, seed=123)
        assert not torch.equal(result1['vector'], result2['vector'])

    def test_dtype_preservation(self, hidden_dim):
        """Output dtype matches input dtype."""
        pos = torch.randn(10, hidden_dim, dtype=torch.bfloat16)
        neg = torch.randn(10, hidden_dim, dtype=torch.bfloat16)

        method = RandomBaselineMethod()
        result = method.extract(pos, neg)
        assert result['vector'].dtype == torch.bfloat16


# =============================================================================
# get_method() tests
# =============================================================================

class TestGetMethod:
    """Tests for get_method() factory."""

    def test_valid_names(self):
        """All valid names return correct types."""
        assert isinstance(get_method('mean_diff'), MeanDifferenceMethod)
        assert isinstance(get_method('probe'), ProbeMethod)
        assert isinstance(get_method('gradient'), GradientMethod)
        assert isinstance(get_method('random_baseline'), RandomBaselineMethod)

    def test_invalid_raises(self):
        """Invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            get_method('invalid_method_name')

    def test_returns_instance(self):
        """Returns an instance, not a class."""
        method = get_method('mean_diff')
        # Should be able to call extract directly
        assert hasattr(method, 'extract')
        assert callable(method.extract)
