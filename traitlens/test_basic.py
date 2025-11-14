#!/usr/bin/env python3
"""
Basic unit tests for traitlens v0.1

Tests core functionality without requiring actual models.
"""

import torch
import torch.nn as nn

# Import from local files since not installed as package
from hooks import HookManager
from activations import ActivationCapture
from compute import (
    mean_difference,
    compute_derivative,
    compute_second_derivative,
    projection,
    cosine_similarity,
    normalize_vectors
)


class TinyModel(nn.Module):
    """Minimal model for testing."""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        for layer in self.layers:
            x = layer(x)
        return x


def test_hook_manager():
    """Test HookManager functionality."""
    print("Testing HookManager...")

    model = TinyModel(hidden_dim=128)
    capture = ActivationCapture()

    # Test context manager
    with HookManager(model) as hooks:
        # Add a hook
        hooks.add_forward_hook("layer1", capture.make_hook("test"))

        # Forward pass
        x = torch.randn(2, 10, 128)  # [batch, seq, hidden]
        output = model(x)

        assert len(hooks) == 1, "Should have 1 hook"

    # Hooks should be removed after context
    assert "test" in capture.keys(), "Should have captured activations"
    acts = capture.get("test")
    assert acts.shape == (2, 10, 128), f"Wrong shape: {acts.shape}"

    print("✅ HookManager tests passed")


def test_activation_capture():
    """Test ActivationCapture functionality."""
    print("Testing ActivationCapture...")

    model = TinyModel(hidden_dim=64)
    capture = ActivationCapture()

    with HookManager(model) as hooks:
        # Hook multiple locations
        hooks.add_forward_hook("layer1", capture.make_hook("l1"))
        hooks.add_forward_hook("layer2", capture.make_hook("l2"))
        hooks.add_forward_hook("layers.0", capture.make_hook("l3"))

        # Multiple forward passes
        for _ in range(3):
            x = torch.randn(1, 5, 64)
            model(x)

    # Check captures
    assert len(capture) == 3, "Should have 3 named captures"
    assert set(capture.keys()) == {"l1", "l2", "l3"}

    # Check concatenation
    l1_acts = capture.get("l1", concat=True)
    assert l1_acts.shape[0] == 3, "Should have 3 batches concatenated"

    # Check memory usage
    usage = capture.memory_usage
    assert all(v > 0 for v in usage.values()), "Should have positive memory usage"

    # Test clear
    capture.clear("l1")
    assert "l1" not in capture.keys()

    capture.clear()
    assert len(capture) == 0

    print("✅ ActivationCapture tests passed")


def test_compute_functions():
    """Test compute functions."""
    print("Testing compute functions...")

    # Test mean_difference
    pos_acts = torch.randn(10, 20, 768)  # [batch, seq, hidden]
    neg_acts = torch.randn(10, 20, 768)
    vector = mean_difference(pos_acts, neg_acts)
    assert vector.shape == (768,), f"Wrong vector shape: {vector.shape}"

    # Test derivatives
    trajectory = torch.randn(50, 768)  # [seq_len, hidden]
    velocity = compute_derivative(trajectory)
    assert velocity.shape == (49, 768), f"Wrong velocity shape: {velocity.shape}"

    acceleration = compute_second_derivative(trajectory)
    assert acceleration.shape == (48, 768), f"Wrong acceleration shape: {acceleration.shape}"

    # Test projection
    activations = torch.randn(10, 50, 768)  # [batch, seq, hidden]
    trait_vector = torch.randn(768)
    scores = projection(activations, trait_vector)
    assert scores.shape == (10, 50), f"Wrong projection shape: {scores.shape}"

    # Test cosine similarity
    vec1 = torch.randn(768)
    vec2 = torch.randn(768)
    sim = cosine_similarity(vec1, vec2)
    assert -1 <= sim <= 1, f"Cosine similarity out of range: {sim}"

    # Test normalize
    vectors = torch.randn(10, 768)
    normalized = normalize_vectors(vectors)
    norms = normalized.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Vectors not normalized"

    print("✅ Compute function tests passed")


def test_multi_layer_extraction():
    """Test extracting from multiple layers."""
    print("Testing multi-layer extraction...")

    model = TinyModel(hidden_dim=32)
    capture = ActivationCapture()

    with HookManager(model) as hooks:
        # Hook all layers in ModuleList
        for i in range(3):
            hooks.add_forward_hook(f"layers.{i}", capture.make_hook(f"layer_{i}"))

        x = torch.randn(2, 5, 32)
        model(x)

    # Check all captures
    for i in range(3):
        assert f"layer_{i}" in capture.keys()
        acts = capture.get(f"layer_{i}")
        assert acts.shape == (2, 5, 32)

    print("✅ Multi-layer extraction tests passed")


def test_temporal_analysis():
    """Test temporal dynamics analysis."""
    print("Testing temporal analysis...")

    # Simulate per-token trajectory
    seq_len = 20
    hidden_dim = 256
    trajectory = torch.randn(seq_len, hidden_dim)

    # Add some structure (exponential decay)
    for i in range(seq_len):
        trajectory[i] *= (0.9 ** i)

    # Compute derivatives
    velocity = compute_derivative(trajectory)
    acceleration = compute_second_derivative(trajectory)

    # Velocity should generally decrease (negative acceleration)
    mean_accel = acceleration.mean(dim=0).norm()
    assert mean_accel > 0, "Should have non-zero acceleration"

    # Test normalization
    velocity_norm = compute_derivative(trajectory, normalize=True)
    norms = velocity_norm.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Normalized velocity should have unit norm"

    print("✅ Temporal analysis tests passed")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running traitlens v0.1 tests")
    print("=" * 50 + "\n")

    test_hook_manager()
    test_activation_capture()
    test_compute_functions()
    test_multi_layer_extraction()
    test_temporal_analysis()

    print("\n" + "=" * 50)
    print("All tests passed! traitlens v0.1 is ready.")
    print("=" * 50)


if __name__ == "__main__":
    main()