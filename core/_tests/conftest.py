"""
Shared fixtures for core/ tests.
"""

import pytest
import torch


@pytest.fixture
def hidden_dim():
    """Standard hidden dimension for tests."""
    return 256


@pytest.fixture
def sample_activations(hidden_dim):
    """Random activations [batch=8, hidden_dim]."""
    return torch.randn(8, hidden_dim)


@pytest.fixture
def sample_vector(hidden_dim):
    """Random vector [hidden_dim]."""
    return torch.randn(hidden_dim)


@pytest.fixture
def unit_vector(hidden_dim):
    """Unit-normalized random vector."""
    v = torch.randn(hidden_dim)
    return v / v.norm()


@pytest.fixture
def zero_vector(hidden_dim):
    """Zero vector."""
    return torch.zeros(hidden_dim)


@pytest.fixture
def pos_activations(hidden_dim):
    """Positive class activations - shifted positive."""
    return torch.randn(20, hidden_dim) + 2.0


@pytest.fixture
def neg_activations(hidden_dim):
    """Negative class activations - shifted negative."""
    return torch.randn(20, hidden_dim) - 2.0


@pytest.fixture
def overlapping_pos(hidden_dim):
    """Positive activations with overlap."""
    return torch.randn(20, hidden_dim) + 0.5


@pytest.fixture
def overlapping_neg(hidden_dim):
    """Negative activations with overlap."""
    return torch.randn(20, hidden_dim) - 0.5
