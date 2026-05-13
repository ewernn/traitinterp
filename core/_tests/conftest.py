"""
Shared fixtures for core/ tests.
"""

import pytest
import torch
import torch.nn as nn


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


# =============================================================================
# Mock models for hook testing
# =============================================================================

class MockAttn(nn.Module):
    """Mock attention block exposing o_proj/k_proj/v_proj for path resolution."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, x):
        return self.o_proj(x)


class MockMLP(nn.Module):
    """Mock MLP exposing down_proj for path resolution."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(x)


class MockTransformerLayer(nn.Module):
    """Minimal Llama-shaped transformer layer for hook testing."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockAttn(hidden_dim)
        self.mlp = MockMLP(hidden_dim)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class MockGemma2Layer(nn.Module):
    """Transformer layer with Gemma-2 architecture markers (post-sublayer norms)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockAttn(hidden_dim)
        self.mlp = MockMLP(hidden_dim)
        self.pre_feedforward_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.post_feedforward_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out = self.post_attention_layernorm(self.self_attn(x))
        mlp_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x + attn_out)))
        return x + attn_out + mlp_out


class MockConfig:
    """Minimal config object - carries model_type so the Architecture registry resolves."""

    def __init__(self, n_layers, model_type="llama"):
        self.num_hidden_layers = n_layers
        self.model_type = model_type


class MockModel(nn.Module):
    """Minimal model for hook testing (Llama-style architecture)."""

    def __init__(self, hidden_dim, n_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(n_layers)])
        self.config = MockConfig(n_layers)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockGemma2Model(nn.Module):
    """Mock model with Gemma-2 architecture (post-sublayer norms)."""

    def __init__(self, hidden_dim, n_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockGemma2Layer(hidden_dim) for _ in range(n_layers)])
        self.config = MockConfig(n_layers, model_type="gemma2")

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockUnknownArchLayer(nn.Module):
    """Layer with unrecognized architecture (no self_attn or pre_feedforward_layernorm)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.weird_module = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.weird_module(x)


class MockUnknownModel(nn.Module):
    """Model with unknown architecture for testing fail-fast validation."""

    def __init__(self, hidden_dim, n_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockUnknownArchLayer(hidden_dim) for _ in range(n_layers)])
        self.config = MockConfig(n_layers, model_type="not_a_real_arch")

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.fixture
def mock_model(hidden_dim):
    """Llama-style mock model for hook testing."""
    return MockModel(hidden_dim, n_layers=4)


@pytest.fixture
def mock_gemma2_model(hidden_dim):
    """Gemma-2 style mock model for hook testing."""
    return MockGemma2Model(hidden_dim, n_layers=4)


@pytest.fixture
def mock_unknown_model(hidden_dim):
    """Unknown architecture mock model for testing fail-fast."""
    return MockUnknownModel(hidden_dim, n_layers=4)
