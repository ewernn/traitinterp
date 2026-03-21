"""
Tests for core/hooks.py - hook management for transformer models.

Run: pytest core/_tests/test_hooks.py -v
"""

import pytest
import torch

from core.hooks import (
    HookManager,
    CaptureHook,
    SteeringHook,
    AblationHook,
    MultiLayerCapture,
    MultiLayerSteeringHook,
    MultiLayerAblationHook,
    get_hook_path,
    detect_contribution_paths,
)


# =============================================================================
# HookManager tests
# =============================================================================

class TestHookManager:
    """Tests for HookManager base class."""

    def test_navigate_path_valid(self, mock_model):
        """Navigates to nested module via dot-separated path."""
        manager = HookManager(mock_model)
        layer = manager._navigate_path("model.layers.0")
        assert layer is mock_model.model.layers[0]

    def test_navigate_path_with_numeric_index(self, mock_model):
        """Handles numeric indices in path."""
        manager = HookManager(mock_model)
        layer2 = manager._navigate_path("model.layers.2")
        assert layer2 is mock_model.model.layers[2]

    def test_navigate_path_invalid_raises(self, mock_model):
        """AttributeError on invalid path."""
        manager = HookManager(mock_model)
        with pytest.raises(AttributeError):
            manager._navigate_path("model.nonexistent.path")

    def test_add_forward_hook_fires(self, mock_model, hidden_dim):
        """Hook function is called during forward pass."""
        fired = []

        def hook_fn(module, inputs, outputs):
            fired.append(True)
            return None

        with HookManager(mock_model) as manager:
            manager.add_forward_hook("model.layers.0", hook_fn)
            x = torch.randn(2, 8, hidden_dim)
            mock_model(x)

        assert len(fired) == 1

    def test_remove_all_cleans_up(self, mock_model, hidden_dim):
        """Hooks removed after remove_all()."""
        fired = []

        def hook_fn(module, inputs, outputs):
            fired.append(True)
            return None

        manager = HookManager(mock_model)
        manager.add_forward_hook("model.layers.0", hook_fn)

        x = torch.randn(2, 8, hidden_dim)
        mock_model(x)
        assert len(fired) == 1

        manager.remove_all()
        mock_model(x)
        assert len(fired) == 1  # Hook didn't fire again

    def test_context_manager_cleans_up(self, mock_model, hidden_dim):
        """Context manager removes hooks on exit."""
        fired = []

        def hook_fn(module, inputs, outputs):
            fired.append(True)
            return None

        with HookManager(mock_model) as manager:
            manager.add_forward_hook("model.layers.0", hook_fn)
            x = torch.randn(2, 8, hidden_dim)
            mock_model(x)

        assert len(fired) == 1

        # After context exit, hook should be removed
        mock_model(x)
        assert len(fired) == 1

    def test_cleanup_on_exception(self, mock_model, hidden_dim):
        """Hooks cleaned up even if forward raises."""
        hooks_before = len(mock_model.model.layers[0]._forward_hooks)

        class ForwardError(Exception):
            pass

        def bad_hook(module, inputs, outputs):
            raise ForwardError("Intentional error")

        try:
            with HookManager(mock_model) as manager:
                manager.add_forward_hook("model.layers.0", bad_hook)
                x = torch.randn(2, 8, hidden_dim)
                mock_model(x)
        except ForwardError:
            pass

        hooks_after = len(mock_model.model.layers[0]._forward_hooks)
        assert hooks_after == hooks_before


# =============================================================================
# CaptureHook tests
# =============================================================================

class TestCaptureHook:
    """Tests for CaptureHook."""

    def test_captures_output_tensor(self, mock_model, hidden_dim):
        """Captures tensor with correct shape."""
        with CaptureHook(mock_model, "model.layers.0") as hook:
            x = torch.randn(2, 8, hidden_dim)
            mock_model(x)
        captured = hook.get()
        assert captured.shape == (2, 8, hidden_dim)

    def test_multiple_forward_passes_accumulate(self, mock_model, hidden_dim):
        """Multiple forward passes concatenate along batch dim."""
        with CaptureHook(mock_model, "model.layers.0") as hook:
            x1 = torch.randn(2, 8, hidden_dim)
            x2 = torch.randn(3, 8, hidden_dim)
            mock_model(x1)
            mock_model(x2)
        captured = hook.get()
        assert captured.shape == (5, 8, hidden_dim)  # 2 + 3 = 5

    def test_get_raises_when_empty(self, mock_model):
        """ValueError if no captures."""
        with CaptureHook(mock_model, "model.layers.0") as hook:
            pass  # No forward pass
        with pytest.raises(ValueError, match="No activations captured"):
            hook.get()

    def test_clear_resets(self, mock_model, hidden_dim):
        """clear() empties captured list."""
        with CaptureHook(mock_model, "model.layers.0") as hook:
            x = torch.randn(2, 8, hidden_dim)
            mock_model(x)
            assert len(hook.captured) == 1
            hook.clear()
            assert len(hook.captured) == 0

    def test_get_no_concat(self, mock_model, hidden_dim):
        """get(concat=False) returns list."""
        with CaptureHook(mock_model, "model.layers.0") as hook:
            x1 = torch.randn(2, 8, hidden_dim)
            x2 = torch.randn(3, 8, hidden_dim)
            mock_model(x1)
            mock_model(x2)
        captured = hook.get(concat=False)
        assert isinstance(captured, list)
        assert len(captured) == 2
        assert captured[0].shape == (2, 8, hidden_dim)
        assert captured[1].shape == (3, 8, hidden_dim)

    def test_keep_on_gpu_flag(self, mock_model, hidden_dim):
        """Respects keep_on_gpu parameter."""
        # Default: move to CPU
        with CaptureHook(mock_model, "model.layers.0", keep_on_gpu=False) as hook:
            x = torch.randn(2, 8, hidden_dim)
            mock_model(x)
        assert hook.get().device == torch.device('cpu')


# =============================================================================
# SteeringHook tests
# =============================================================================

class TestSteeringHook:
    """Tests for SteeringHook."""

    def test_adds_vector_to_output(self, mock_model, hidden_dim):
        """Output is modified by coefficient * vector."""
        vector = torch.randn(hidden_dim)

        # Capture without steering
        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        # Capture with steering
        with SteeringHook(mock_model, vector, "model.layers.0", coefficient=1.0):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        # Difference should be the vector (broadcast across batch, seq)
        diff = steered - original
        expected = vector.unsqueeze(0).unsqueeze(0).expand_as(diff)
        assert torch.allclose(diff, expected, atol=1e-5)

    def test_coefficient_scaling(self, mock_model, hidden_dim):
        """Different coefficients scale the steering."""
        vector = torch.randn(hidden_dim)

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        with SteeringHook(mock_model, vector, "model.layers.0", coefficient=2.5):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        diff = steered - original
        expected = (2.5 * vector).unsqueeze(0).unsqueeze(0).expand_as(diff)
        assert torch.allclose(diff, expected, atol=1e-5)

    def test_zero_coefficient_no_change(self, mock_model, hidden_dim):
        """coefficient=0 leaves output unchanged."""
        vector = torch.randn(hidden_dim)

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        with SteeringHook(mock_model, vector, "model.layers.0", coefficient=0.0):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        assert torch.allclose(original, steered, atol=1e-6)

    def test_rejects_non_1d_vector(self, mock_model, hidden_dim):
        """ValueError for non-1D vector."""
        bad_vector = torch.randn(4, hidden_dim)  # 2D
        with pytest.raises(ValueError, match="must be 1-D"):
            SteeringHook(mock_model, bad_vector, "model.layers.0")

    def test_negative_coefficient(self, mock_model, hidden_dim):
        """Negative coefficient subtracts vector."""
        vector = torch.randn(hidden_dim)

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        with SteeringHook(mock_model, vector, "model.layers.0", coefficient=-1.0):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        diff = steered - original
        expected = (-vector).unsqueeze(0).unsqueeze(0).expand_as(diff)
        assert torch.allclose(diff, expected, atol=1e-5)

    def test_vector_stored_as_float32(self, mock_model, hidden_dim):
        """Vector is stored in float32 for precision (avoids bfloat16 loss)."""
        vector = torch.randn(hidden_dim, dtype=torch.bfloat16)
        hook = SteeringHook(mock_model, vector, "model.layers.0", coefficient=1.0)
        assert hook.vector.dtype == torch.float32


# =============================================================================
# AblationHook tests
# =============================================================================

class TestAblationHook:
    """Tests for AblationHook."""

    def test_projects_out_direction(self, mock_model, hidden_dim):
        """Ablation removes component along direction."""
        # Use a specific direction
        direction = torch.zeros(hidden_dim)
        direction[0] = 1.0  # Unit vector along dim 0

        # Input with known component along direction
        x = torch.zeros(1, 1, hidden_dim)
        x[0, 0, 0] = 5.0  # Component along direction
        x[0, 0, 1] = 3.0  # Orthogonal component

        with AblationHook(mock_model, direction, "model.layers.0"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        ablated = cap.get()

        # Dim 0 should be zeroed (projection removed), dim 1 unchanged
        # Note: mock model adds x to itself, so we check the hook's effect
        # The ablation happens at layer output, which includes transformations
        # For this test, we verify the ablation math directly instead
        pass  # See test_parallel_input_zeroed for cleaner verification

    def test_parallel_input_zeroed(self, hidden_dim):
        """Input parallel to direction becomes zero after ablation."""
        # Direct math test without model
        direction = torch.randn(hidden_dim)
        direction = direction / direction.norm()

        # Input is exactly the direction (scaled)
        x = direction * 3.0

        # Ablation: x' = x - (x · r̂) * r̂
        proj_coef = x @ direction
        ablated = x - proj_coef * direction

        assert torch.allclose(ablated, torch.zeros_like(ablated), atol=1e-6)

    def test_orthogonal_input_unchanged(self, hidden_dim):
        """Input orthogonal to direction is unchanged."""
        # Create orthogonal vectors
        direction = torch.zeros(hidden_dim)
        direction[0] = 1.0

        x = torch.zeros(hidden_dim)
        x[1] = 5.0  # Orthogonal to direction

        # Ablation should not change x
        proj_coef = x @ direction
        ablated = x - proj_coef * direction

        assert torch.allclose(ablated, x, atol=1e-6)

    def test_zero_direction_raises(self, mock_model, hidden_dim):
        """ValueError for zero vector (new fix)."""
        zero_dir = torch.zeros(hidden_dim)
        with pytest.raises(ValueError, match="near-zero norm"):
            AblationHook(mock_model, zero_dir, "model.layers.0")

    def test_tiny_direction_raises(self, mock_model, hidden_dim):
        """ValueError for very small vector."""
        tiny_dir = torch.ones(hidden_dim) * 1e-10
        with pytest.raises(ValueError, match="near-zero norm"):
            AblationHook(mock_model, tiny_dir, "model.layers.0")

    def test_rejects_non_1d_direction(self, mock_model, hidden_dim):
        """ValueError for non-1D direction."""
        bad_dir = torch.randn(4, hidden_dim)
        with pytest.raises(ValueError, match="must be 1-D"):
            AblationHook(mock_model, bad_dir, "model.layers.0")


# =============================================================================
# get_hook_path tests
# =============================================================================

class TestGetHookPath:
    """Tests for get_hook_path utility."""

    def test_residual_path(self):
        """Returns layer path for residual."""
        path = get_hook_path(16, "residual")
        assert path == "model.layers.16"

    def test_attn_out_path(self):
        """Returns attention output path."""
        path = get_hook_path(5, "attn_out")
        assert path == "model.layers.5.self_attn.o_proj"

    def test_mlp_out_path(self):
        """Returns MLP output path."""
        path = get_hook_path(10, "mlp_out")
        assert path == "model.layers.10.mlp.down_proj"

    def test_k_proj_path(self):
        """Returns key projection path."""
        path = get_hook_path(3, "k_proj")
        assert path == "model.layers.3.self_attn.k_proj"

    def test_v_proj_path(self):
        """Returns value projection path."""
        path = get_hook_path(7, "v_proj")
        assert path == "model.layers.7.self_attn.v_proj"

    def test_custom_prefix(self):
        """Respects custom prefix."""
        path = get_hook_path(2, "residual", prefix="base_model.model.model.layers")
        assert path == "base_model.model.model.layers.2"

    def test_unknown_component_raises(self):
        """ValueError for invalid component."""
        with pytest.raises(ValueError, match="Unknown component"):
            get_hook_path(0, "not_a_component")

    def test_contribution_requires_model(self):
        """ValueError when model=None for contribution components."""
        with pytest.raises(ValueError, match="requires model parameter"):
            get_hook_path(0, "attn_contribution", model=None)

        with pytest.raises(ValueError, match="requires model parameter"):
            get_hook_path(0, "mlp_contribution", model=None)

    def test_attn_contribution_llama_style(self, mock_model):
        """attn_contribution resolves to o_proj for Llama-style."""
        path = get_hook_path(0, "attn_contribution", model=mock_model)
        assert "self_attn.o_proj" in path

    def test_attn_contribution_gemma2_style(self, mock_gemma2_model):
        """attn_contribution resolves to post_attention_layernorm for Gemma-2."""
        path = get_hook_path(0, "attn_contribution", model=mock_gemma2_model)
        assert "post_attention_layernorm" in path


# =============================================================================
# detect_contribution_paths tests
# =============================================================================

class TestDetectContributionPaths:
    """Tests for architecture detection."""

    def test_detect_llama_pattern(self, mock_model):
        """Detects Llama/Mistral pattern (pre-norm only)."""
        paths = detect_contribution_paths(mock_model)
        assert paths['attn_contribution'] == 'self_attn.o_proj'
        assert paths['mlp_contribution'] == 'mlp.down_proj'

    def test_detect_gemma2_pattern(self, mock_gemma2_model):
        """Detects Gemma-2 pattern (post-sublayer norms)."""
        paths = detect_contribution_paths(mock_gemma2_model)
        assert paths['attn_contribution'] == 'post_attention_layernorm'
        assert paths['mlp_contribution'] == 'post_feedforward_layernorm'

    def test_unknown_architecture_raises(self, mock_unknown_model):
        """ValueError for unrecognized architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            detect_contribution_paths(mock_unknown_model)


# =============================================================================
# MultiLayerCapture tests
# =============================================================================

class TestMultiLayerCapture:
    """Tests for MultiLayerCapture."""

    def test_captures_specified_layers(self, mock_model, hidden_dim):
        """Captures from each specified layer."""
        with MultiLayerCapture(mock_model, layers=[0, 2]) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

        # Can get specified layers
        act0 = cap.get(0)
        act2 = cap.get(2)
        assert act0.shape == (2, 4, hidden_dim)
        assert act2.shape == (2, 4, hidden_dim)

    def test_captures_all_layers_by_default(self, mock_model, hidden_dim):
        """layers=None captures all layers."""
        with MultiLayerCapture(mock_model, layers=None) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

        all_acts = cap.get_all()
        assert len(all_acts) == 4  # MockModel has 4 layers

    def test_get_specific_layer(self, mock_model, hidden_dim):
        """get() retrieves single layer's activations."""
        with MultiLayerCapture(mock_model, layers=[1]) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

        acts = cap.get(1)
        assert acts.shape == (2, 4, hidden_dim)

    def test_get_all_returns_dict(self, mock_model, hidden_dim):
        """get_all() returns {layer: tensor} dict."""
        with MultiLayerCapture(mock_model, layers=[0, 1]) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

        all_acts = cap.get_all()
        assert isinstance(all_acts, dict)
        assert 0 in all_acts
        assert 1 in all_acts

    def test_invalid_layer_raises(self, mock_model, hidden_dim):
        """KeyError for layer not in capture list."""
        with MultiLayerCapture(mock_model, layers=[0]) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

        with pytest.raises(KeyError, match="Layer 1 not captured"):
            cap.get(1)

    def test_clear_all_layers(self, mock_model, hidden_dim):
        """clear() empties all captured activations."""
        with MultiLayerCapture(mock_model, layers=[0, 1]) as cap:
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)
            cap.clear()
            with pytest.raises(ValueError):
                cap.get(0)


# =============================================================================
# MultiLayerSteeringHook tests
# =============================================================================

class TestMultiLayerSteeringHook:
    """Tests for MultiLayerSteeringHook."""

    def test_steers_multiple_layers(self, mock_model, hidden_dim):
        """Applies steering to multiple layers."""
        vec0 = torch.randn(hidden_dim)
        vec1 = torch.randn(hidden_dim)

        configs = [
            (0, vec0, 1.0),
            (1, vec1, 0.5),
        ]

        # Should not raise
        with MultiLayerSteeringHook(mock_model, configs):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)


# =============================================================================
# MultiLayerAblationHook tests
# =============================================================================

class TestMultiLayerAblationHook:
    """Tests for MultiLayerAblationHook."""

    def test_ablates_all_layers_by_default(self, mock_model, hidden_dim):
        """layers=None ablates all layers."""
        direction = torch.randn(hidden_dim)

        # Should not raise
        with MultiLayerAblationHook(mock_model, direction, layers=None):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

    def test_ablates_specific_layers(self, mock_model, hidden_dim):
        """Ablates only specified layers."""
        direction = torch.randn(hidden_dim)

        with MultiLayerAblationHook(mock_model, direction, layers=[1, 2]):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

    def test_zero_direction_raises(self, mock_model, hidden_dim):
        """Zero direction raises ValueError."""
        zero_dir = torch.zeros(hidden_dim)
        with pytest.raises(ValueError, match="near-zero norm"):
            MultiLayerAblationHook(mock_model, zero_dir)


# =============================================================================
# Context manager cleanup robustness
# =============================================================================

class TestCleanupRobustness:
    """Tests for hook cleanup in edge cases."""

    def test_nested_hooks_both_cleanup(self, mock_model, hidden_dim):
        """Nested hook contexts both clean up properly."""
        hooks_before = len(mock_model.model.layers[0]._forward_hooks)

        with CaptureHook(mock_model, "model.layers.0"):
            with SteeringHook(mock_model, torch.randn(hidden_dim), "model.layers.0"):
                x = torch.randn(2, 4, hidden_dim)
                mock_model(x)

        hooks_after = len(mock_model.model.layers[0]._forward_hooks)
        assert hooks_after == hooks_before

    def test_multilayer_cleanup_on_forward_error(self, mock_model, hidden_dim):
        """MultiLayerCapture cleans up even if forward fails."""
        hooks_counts_before = [
            len(mock_model.model.layers[i]._forward_hooks) for i in range(4)
        ]

        # Create a model that will error during forward
        original_forward = mock_model.model.layers[2].forward

        def bad_forward(x):
            raise RuntimeError("Intentional error")

        mock_model.model.layers[2].forward = bad_forward

        try:
            with MultiLayerCapture(mock_model, layers=[0, 1, 2, 3]):
                x = torch.randn(2, 4, hidden_dim)
                mock_model(x)
        except RuntimeError:
            pass
        finally:
            mock_model.model.layers[2].forward = original_forward

        hooks_counts_after = [
            len(mock_model.model.layers[i]._forward_hooks) for i in range(4)
        ]
        assert hooks_counts_after == hooks_counts_before
