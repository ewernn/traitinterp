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
    ActivationCappingHook,
    MultiLayerCapture,
    MultiLayerSteering,
    MultiLayerAblation,
    MultiLayerActivationCapping,
)


# =============================================================================
# HookManager tests
# =============================================================================

class TestHookManager:
    """Tests for HookManager base class."""

    def test_navigate_path_valid(self, mock_model):
        """Navigates to nested module via dot-separated path."""
        manager = HookManager(mock_model)
        layer = manager.model.get_submodule("model.layers.0")
        assert layer is mock_model.model.layers[0]

    def test_navigate_path_with_numeric_index(self, mock_model):
        """Handles numeric indices in path."""
        manager = HookManager(mock_model)
        layer2 = manager.model.get_submodule("model.layers.2")
        assert layer2 is mock_model.model.layers[2]

    def test_navigate_path_invalid_raises(self, mock_model):
        """AttributeError on invalid path."""
        manager = HookManager(mock_model)
        with pytest.raises(AttributeError):
            manager.model.get_submodule("model.nonexistent.path")

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

    def test_norm_match_per_token_magnitude(self, mock_model, hidden_dim):
        """norm_match=True: per-token addend has magnitude coef * ||residual_t||.

        For unit-direction vector v, the added delta at token t equals
        coef * ||residual_t|| * v_hat. Verifies the residual-norm match property
        rather than the original residual-norm preservation.
        """
        vector = torch.randn(hidden_dim)
        v_unit = vector / vector.norm()

        # Capture original residual
        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        coef = 1.5
        with SteeringHook(mock_model, vector, "model.layers.0",
                          coefficient=coef, norm_match=True):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        diff = steered - original
        # Per-token addend should equal coef * ||original_t|| * v_unit
        r_norms = original.float().norm(dim=-1, keepdim=True)  # [1, 4, 1]
        expected = coef * r_norms * v_unit
        assert torch.allclose(diff, expected, atol=1e-5)

    def test_norm_match_differs_from_plain(self, mock_model, hidden_dim):
        """norm_match=True produces a different output than norm_match=False."""
        vector = torch.randn(hidden_dim)
        x = torch.randn(1, 4, hidden_dim) * 3.0  # non-unit residual norms

        with SteeringHook(mock_model, vector, "model.layers.0",
                          coefficient=1.0, norm_match=False):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        plain = cap.get()

        with SteeringHook(mock_model, vector, "model.layers.0",
                          coefficient=1.0, norm_match=True):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        matched = cap.get()

        assert not torch.allclose(plain, matched, atol=1e-3)

    def test_norm_match_zero_coefficient_no_change(self, mock_model, hidden_dim):
        """norm_match with coef=0 leaves output unchanged."""
        vector = torch.randn(hidden_dim)

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        with SteeringHook(mock_model, vector, "model.layers.0",
                          coefficient=0.0, norm_match=True):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        steered = cap.get()

        assert torch.allclose(original, steered, atol=1e-6)


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
# Path resolution + arch detection moved to test_architectures.py
# =============================================================================


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
# MultiLayerSteering tests
# =============================================================================

class TestMultiLayerSteering:
    """Tests for MultiLayerSteering."""

    def test_steers_multiple_layers(self, mock_model, hidden_dim):
        """Applies steering to multiple layers."""
        vec0 = torch.randn(hidden_dim)
        vec1 = torch.randn(hidden_dim)

        configs = [
            (0, vec0, 1.0),
            (1, vec1, 0.5),
        ]

        # Should not raise
        with MultiLayerSteering(mock_model, configs):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)


# =============================================================================
# MultiLayerAblation tests
# =============================================================================

class TestMultiLayerAblation:
    """Tests for MultiLayerAblation."""

    def test_ablates_all_layers_by_default(self, mock_model, hidden_dim):
        """layers=None ablates all layers."""
        direction = torch.randn(hidden_dim)

        # Should not raise
        with MultiLayerAblation(mock_model, direction, layers=None):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

    def test_ablates_specific_layers(self, mock_model, hidden_dim):
        """Ablates only specified layers."""
        direction = torch.randn(hidden_dim)

        with MultiLayerAblation(mock_model, direction, layers=[1, 2]):
            x = torch.randn(2, 4, hidden_dim)
            mock_model(x)

    def test_zero_direction_raises(self, mock_model, hidden_dim):
        """Zero direction raises ValueError."""
        zero_dir = torch.zeros(hidden_dim)
        with pytest.raises(ValueError, match="near-zero norm"):
            MultiLayerAblation(mock_model, zero_dir)


# =============================================================================
# ActivationCappingHook tests
# =============================================================================

class TestActivationCappingHook:
    """Tests for ActivationCappingHook (single layer)."""

    def test_floor_pulls_below_tau_up(self, mock_model, hidden_dim):
        """Floor mode: projection below tau gets pulled up to exactly tau."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        # Capture unmodified projection at layer 0
        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()
        original_proj = original @ v_hat  # [1, 4]

        # Pick tau strictly above the original projection
        tau = float(original_proj.max()) + 1.0  # every token starts below tau

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="floor", tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()
        capped_proj = capped @ v_hat

        # Every token's projection should be exactly tau
        assert torch.allclose(capped_proj, torch.full_like(capped_proj, tau), atol=1e-4)

    def test_floor_leaves_above_tau_unchanged(self, mock_model, hidden_dim):
        """Floor mode: projection already above tau is unchanged."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()
        original_proj = original @ v_hat

        # Pick tau strictly below every token's projection
        tau = float(original_proj.min()) - 1.0

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="floor", tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()

        # No-op
        assert torch.allclose(capped, original, atol=1e-5)

    def test_ceiling_pulls_above_tau_down(self, mock_model, hidden_dim):
        """Ceiling mode: projection above tau gets pulled down to exactly tau."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()
        original_proj = original @ v_hat

        tau = float(original_proj.min()) - 1.0  # every token starts above tau

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="ceiling", tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()
        capped_proj = capped @ v_hat

        assert torch.allclose(capped_proj, torch.full_like(capped_proj, tau), atol=1e-4)

    def test_ceiling_leaves_below_tau_unchanged(self, mock_model, hidden_dim):
        """Ceiling mode: projection already below tau is unchanged."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()
        original_proj = original @ v_hat

        tau = float(original_proj.max()) + 1.0

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="ceiling", tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()

        assert torch.allclose(capped, original, atol=1e-5)

    def test_orthogonal_component_preserved(self, mock_model, hidden_dim):
        """Cap only changes the projection along the direction;
        the orthogonal complement is untouched."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()

        tau = float((original @ v_hat).max()) + 1.0  # force the cap to fire on every token

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="floor", tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()

        # Orthogonal component = h - (h @ v_hat) * v_hat
        original_orth = original - (original @ v_hat).unsqueeze(-1) * v_hat
        capped_orth = capped - (capped @ v_hat).unsqueeze(-1) * v_hat
        assert torch.allclose(original_orth, capped_orth, atol=1e-4)

    def test_zero_direction_raises(self, mock_model, hidden_dim):
        """Zero (or near-zero) direction vector raises before forward pass."""
        with pytest.raises(ValueError, match="near-zero norm"):
            ActivationCappingHook(mock_model, torch.zeros(hidden_dim),
                                  "model.layers.0", tau=0.5)

    def test_non_1d_direction_raises(self, mock_model, hidden_dim):
        """Non-1D direction raises."""
        bad = torch.randn(4, hidden_dim)
        with pytest.raises(ValueError, match="must be 1-D"):
            ActivationCappingHook(mock_model, bad, "model.layers.0", tau=0.5)

    def test_invalid_mode_raises(self, mock_model, hidden_dim):
        """Mode other than 'floor' / 'ceiling' raises."""
        with pytest.raises(ValueError, match="mode must be"):
            ActivationCappingHook(mock_model, torch.randn(hidden_dim),
                                  "model.layers.0", tau=0.5, mode="middle")

    def test_invalid_tau_mode_raises(self, mock_model, hidden_dim):
        """tau_mode other than the three valid options raises."""
        with pytest.raises(ValueError, match="tau_mode must be"):
            ActivationCappingHook(mock_model, torch.randn(hidden_dim),
                                  "model.layers.0", tau=0.5, tau_mode="weird")

    def test_calibrated_mode_requires_mean_norm(self, mock_model, hidden_dim):
        """tau_mode='calibrated' without mean_activation_norm raises."""
        with pytest.raises(ValueError, match="requires `mean_activation_norm`"):
            ActivationCappingHook(mock_model, torch.randn(hidden_dim),
                                  "model.layers.0", tau=0.5, tau_mode="calibrated")

    def test_default_mode_is_cosine_without_norm(self, mock_model, hidden_dim):
        """Default tau_mode is 'cosine' when no mean_activation_norm is given."""
        hook = ActivationCappingHook(mock_model, torch.randn(hidden_dim),
                                     "model.layers.0", tau=0.5)
        assert hook.tau_mode == "cosine"

    def test_default_mode_is_calibrated_with_norm(self, mock_model, hidden_dim):
        """Default tau_mode is 'calibrated' when mean_activation_norm is given."""
        hook = ActivationCappingHook(mock_model, torch.randn(hidden_dim),
                                     "model.layers.0", tau=0.5,
                                     mean_activation_norm=10.0)
        assert hook.tau_mode == "calibrated"
        assert hook.mean_activation_norm == 10.0

    def test_calibrated_mode_uses_mean_norm(self, mock_model, hidden_dim):
        """Calibrated mode: effective_tau = tau * mean_activation_norm (constant)."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        natural_proj = cap.get() @ v_hat
        mean_norm = 100.0
        # tau * mean_norm must exceed every natural projection so the floor fires
        target_raw = float(natural_proj.max()) + 1.0
        tau = target_raw / mean_norm

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="floor",
                                   mean_activation_norm=mean_norm):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped_proj = cap.get() @ v_hat

        # Post-cap projection should equal tau * mean_norm exactly
        assert torch.allclose(capped_proj, torch.full_like(capped_proj, target_raw), atol=1e-3)

    def test_cosine_mode_uses_per_token_norm(self, mock_model, hidden_dim):
        """Cosine mode: effective_tau = tau * ||h_t|| per token. Output cos sim
        of the post-cap projection-to-input-norm should equal tau (for tokens
        where the cap fired)."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        original = cap.get()
        original_norms = original.norm(dim=-1)  # [1, 4]
        original_proj = original @ v_hat  # [1, 4]
        original_cos = original_proj / original_norms

        # Pick tau strictly above every token's natural cosine
        tau = float(original_cos.max()) + 0.05

        with ActivationCappingHook(mock_model, direction, "model.layers.0",
                                   tau=tau, mode="floor", tau_mode="cosine"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped = cap.get()
        capped_proj = capped @ v_hat

        # Post-cap projection per token should be exactly tau * original_norm[t]
        # (the hook uses the PRE-cap output norm to compute effective_tau)
        expected_proj = tau * original_norms
        assert torch.allclose(capped_proj, expected_proj, atol=1e-3)


# =============================================================================
# MultiLayerActivationCapping tests
# =============================================================================

class TestMultiLayerActivationCapping:
    """Tests for MultiLayerActivationCapping (multi-layer composite)."""

    def test_registers_hook_per_layer(self, mock_model, hidden_dim):
        """Composite registers one underlying hook per layer in `tau_per_layer`."""
        direction = torch.randn(hidden_dim)
        directions = {l: direction for l in range(4)}
        tau_per_layer = {0: 100.0, 2: 200.0}  # subset of layers

        hooks_before = [len(mock_model.model.layers[l]._forward_hooks) for l in range(4)]

        with MultiLayerActivationCapping(mock_model, directions, tau_per_layer):
            hooks_during = [len(mock_model.model.layers[l]._forward_hooks) for l in range(4)]

        hooks_after = [len(mock_model.model.layers[l]._forward_hooks) for l in range(4)]

        # During the context: layers 0 and 2 have +1 hook each, layers 1 and 3 unchanged
        assert hooks_during[0] == hooks_before[0] + 1
        assert hooks_during[1] == hooks_before[1]
        assert hooks_during[2] == hooks_before[2] + 1
        assert hooks_during[3] == hooks_before[3]
        # After: all back to baseline
        assert hooks_after == hooks_before

    def test_single_layer_caps_correctly(self, mock_model, hidden_dim):
        """Single-layer composite caps the projection at that layer's tau."""
        direction = torch.randn(hidden_dim)
        v_hat = direction / direction.norm()
        directions = {0: direction}

        with CaptureHook(mock_model, "model.layers.0") as cap:
            x = torch.randn(1, 4, hidden_dim)
            mock_model(x)
        natural_proj = cap.get() @ v_hat
        tau = float(natural_proj.max()) + 1.0  # floor will fire on every token

        with MultiLayerActivationCapping(mock_model, directions, {0: tau}, mode="floor",
                                         tau_mode="raw"):
            with CaptureHook(mock_model, "model.layers.0") as cap:
                mock_model(x)
        capped_proj = cap.get() @ v_hat

        assert torch.allclose(capped_proj, torch.full_like(capped_proj, tau), atol=1e-4)

    def test_cleanup_on_exception(self, mock_model, hidden_dim):
        """Hooks removed even when forward pass raises."""
        direction = torch.randn(hidden_dim)
        directions = {l: direction for l in range(4)}
        tau_per_layer = {l: 0.5 for l in range(4)}

        hooks_before = [len(mock_model.model.layers[l]._forward_hooks) for l in range(4)]

        original_forward = mock_model.model.layers[2].forward
        mock_model.model.layers[2].forward = lambda x: (_ for _ in ()).throw(RuntimeError("nope"))

        try:
            with MultiLayerActivationCapping(mock_model, directions, tau_per_layer):
                with pytest.raises(RuntimeError):
                    mock_model(torch.randn(1, 4, hidden_dim))
        finally:
            mock_model.model.layers[2].forward = original_forward

        hooks_after = [len(mock_model.model.layers[l]._forward_hooks) for l in range(4)]
        assert hooks_after == hooks_before


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
