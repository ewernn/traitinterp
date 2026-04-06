"""Tests for fused MoE forward (batched dequantize + grouped_mm).

Validates correctness of INT4 dequantization, routing logic, and grouped_mm
against a naive per-expert Python loop reference implementation.

Usage:
    pytest core/_tests/test_fused_moe.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.moe import _batch_dequantize_int4, _fuse_expert_weights

# Skip all tests if no CUDA available (grouped_mm requires CUDA)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused MoE tests require CUDA for grouped_mm"
)


def _pack_int4_to_int32(int8_tensor):
    """Pack signed int8 values (range [-8, 7]) into int32, 8 values per int32.

    Mimics compressed_tensors packing: add offset 8 to make unsigned [0, 15],
    then pack 8 nibbles per int32.
    """
    rows, cols = int8_tensor.shape
    pack_factor = 8
    assert cols % pack_factor == 0
    packed_cols = cols // pack_factor
    unsigned = (int8_tensor.to(torch.int32) + 8)  # [0, 15]
    packed = torch.zeros(rows, packed_cols, dtype=torch.int32, device=int8_tensor.device)
    for i in range(pack_factor):
        packed |= (unsigned[:, i::pack_factor] << (4 * i))
    return packed


def _naive_moe(hidden_states, topk_indices, topk_weights, gate_w, up_w, down_w):
    """Reference MoE implementation: Python loop over experts."""
    N, H = hidden_states.shape
    K = topk_indices.shape[1]
    output = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    for t in range(N):
        for k in range(K):
            eid = topk_indices[t, k].item()
            w = topk_weights[t, k]
            x = hidden_states[t:t+1].to(gate_w.dtype)
            g = x @ gate_w[eid].T
            u = x @ up_w[eid].T
            d = (F.silu(g) * u) @ down_w[eid].T
            output[t] += d.squeeze(0) * w
    return output.to(hidden_states.dtype)


def _run_fused_moe(hidden_states, topk_indices, topk_weights, gate_w, up_w, down_w):
    """Run the fused MoE forward path with pre-dequantized BF16 weights."""
    N, H = hidden_states.shape
    K = topk_indices.shape[1]

    flat_expert_ids = topk_indices.reshape(-1)
    flat_token_ids = torch.arange(N, device=hidden_states.device) \
        .unsqueeze(1).expand(-1, K).reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    order = flat_expert_ids.argsort(stable=True)
    sorted_expert_ids = flat_expert_ids[order]
    sorted_token_ids = flat_token_ids[order]
    sorted_weights = flat_weights[order]

    unique_experts, counts = sorted_expert_ids.unique_consecutive(return_counts=True)
    offs = counts.cumsum(0).to(torch.int32)

    idx = unique_experts.long()
    gw = gate_w[idx]
    uw = up_w[idx]
    dw = down_w[idx]

    x = hidden_states[sorted_token_ids].to(torch.bfloat16)

    gate_out = F.grouped_mm(x, gw.transpose(-1, -2), offs=offs)
    up_out = F.grouped_mm(x, uw.transpose(-1, -2), offs=offs)
    act_out = F.silu(gate_out) * up_out
    down_out = F.grouped_mm(act_out, dw.transpose(-1, -2), offs=offs)

    down_out = down_out * sorted_weights.unsqueeze(-1)
    output = torch.zeros(N, H, device=hidden_states.device, dtype=down_out.dtype)
    output.index_add_(0, sorted_token_ids, down_out)
    return output.to(hidden_states.dtype)


class TestBatchDequantizeInt4:
    """Test _batch_dequantize_int4 correctness."""

    def test_basic(self):
        """Single expert, known values."""
        N, out_f, in_f = 1, 4, 32
        group_size = 32

        torch.manual_seed(42)
        original_int8 = torch.randint(-8, 8, (N * out_f, in_f), dtype=torch.int8)
        scales = torch.randn(N * out_f, in_f // group_size, dtype=torch.bfloat16).abs() * 0.1

        expected = original_int8.to(torch.bfloat16) * scales.repeat_interleave(group_size, dim=-1)

        packed = _pack_int4_to_int32(original_int8)
        packed_3d = packed.reshape(N, out_f, -1)
        scale_3d = scales.reshape(N, out_f, -1)

        result = _batch_dequantize_int4(packed_3d, scale_3d, (out_f, in_f))
        assert result.shape == (N, out_f, in_f)
        assert torch.allclose(result, expected.reshape(N, out_f, in_f), atol=1e-3)

    def test_multiple_experts(self):
        """Multiple experts batched together."""
        N, out_f, in_f = 8, 16, 64
        group_size = 32
        n_groups = in_f // group_size

        torch.manual_seed(123)
        original_int8 = torch.randint(-8, 8, (N * out_f, in_f), dtype=torch.int8)
        scales = torch.randn(N * out_f, n_groups, dtype=torch.bfloat16).abs() * 0.05

        expected = original_int8.to(torch.bfloat16) * scales.repeat_interleave(group_size, dim=-1)

        packed = _pack_int4_to_int32(original_int8)
        result = _batch_dequantize_int4(
            packed.reshape(N, out_f, -1),
            scales.reshape(N, out_f, -1),
            (out_f, in_f),
        )
        assert result.shape == (N, out_f, in_f)
        assert torch.allclose(result, expected.reshape(N, out_f, in_f), atol=1e-3)

    def test_different_group_sizes(self):
        """Group size 16 instead of 32."""
        N, out_f, in_f = 2, 8, 64
        group_size = 16
        n_groups = in_f // group_size  # 4

        torch.manual_seed(99)
        original_int8 = torch.randint(-8, 8, (N * out_f, in_f), dtype=torch.int8)
        scales = torch.randn(N * out_f, n_groups, dtype=torch.bfloat16).abs() * 0.1

        expected = original_int8.to(torch.bfloat16) * scales.repeat_interleave(group_size, dim=-1)

        packed = _pack_int4_to_int32(original_int8)
        result = _batch_dequantize_int4(
            packed.reshape(N, out_f, -1),
            scales.reshape(N, out_f, -1),
            (out_f, in_f),
        )
        assert torch.allclose(result, expected.reshape(N, out_f, in_f), atol=1e-3)


class TestFusedMoeRouting:
    """Test the fused MoE routing + grouped_mm against naive loop."""

    @pytest.fixture
    def device(self):
        return 'cuda'

    def test_basic_routing(self, device):
        """Standard case: 4 tokens, top-2 from 8 experts."""
        n_experts, H, I = 8, 64, 32
        torch.manual_seed(42)

        gate_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        up_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        down_w = torch.randn(n_experts, H, I, dtype=torch.bfloat16, device=device)

        hidden = torch.randn(4, H, dtype=torch.bfloat16, device=device)
        indices = torch.tensor([[0, 3], [1, 5], [2, 7], [0, 6]], device=device)
        weights = torch.randn(4, 2, dtype=torch.bfloat16, device=device).softmax(dim=-1)

        ref = _naive_moe(hidden, indices, weights, gate_w, up_w, down_w)
        fused = _run_fused_moe(hidden, indices, weights, gate_w, up_w, down_w)

        rel_diff = (fused - ref).abs().max() / ref.abs().max()
        assert rel_diff < 0.02, f"Routing mismatch: relative diff {rel_diff:.4f}"

    def test_single_token(self, device):
        """Edge case: only 1 token."""
        n_experts, H, I = 4, 32, 16
        torch.manual_seed(7)

        gate_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        up_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        down_w = torch.randn(n_experts, H, I, dtype=torch.bfloat16, device=device)

        hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device)
        indices = torch.tensor([[1, 3]], device=device)
        weights = torch.tensor([[0.6, 0.4]], dtype=torch.bfloat16, device=device)

        ref = _naive_moe(hidden, indices, weights, gate_w, up_w, down_w)
        fused = _run_fused_moe(hidden, indices, weights, gate_w, up_w, down_w)

        rel_diff = (fused - ref).abs().max() / ref.abs().max()
        assert rel_diff < 0.02

    def test_all_same_expert(self, device):
        """Edge case: all tokens routed to the same expert pair."""
        n_experts, H, I = 4, 32, 16
        torch.manual_seed(11)

        gate_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        up_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        down_w = torch.randn(n_experts, H, I, dtype=torch.bfloat16, device=device)

        hidden = torch.randn(8, H, dtype=torch.bfloat16, device=device)
        # All tokens go to experts 0 and 2
        indices = torch.tensor([[0, 2]] * 8, device=device)
        weights = torch.randn(8, 2, dtype=torch.bfloat16, device=device).softmax(dim=-1)

        ref = _naive_moe(hidden, indices, weights, gate_w, up_w, down_w)
        fused = _run_fused_moe(hidden, indices, weights, gate_w, up_w, down_w)

        rel_diff = (fused - ref).abs().max() / ref.abs().max()
        assert rel_diff < 0.02

    def test_top8_many_experts(self, device):
        """Larger case: top-8 from 64 experts, 16 tokens."""
        n_experts, H, I, K = 64, 128, 64, 8
        torch.manual_seed(55)

        gate_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        up_w = torch.randn(n_experts, I, H, dtype=torch.bfloat16, device=device)
        down_w = torch.randn(n_experts, H, I, dtype=torch.bfloat16, device=device)

        hidden = torch.randn(16, H, dtype=torch.bfloat16, device=device)
        # Random top-8 assignments (no duplicates per token)
        indices = torch.stack([torch.randperm(n_experts, device=device)[:K] for _ in range(16)])
        weights = torch.randn(16, K, dtype=torch.bfloat16, device=device).softmax(dim=-1)

        ref = _naive_moe(hidden, indices, weights, gate_w, up_w, down_w)
        fused = _run_fused_moe(hidden, indices, weights, gate_w, up_w, down_w)

        rel_diff = (fused - ref).abs().max() / ref.abs().max()
        assert rel_diff < 0.02, f"Top-8 mismatch: relative diff {rel_diff:.4f}"


class TestFuseExpertWeights:
    """Test _fuse_expert_weights stacking and cleanup."""

    def _make_mock_compressed_linear(self, out_f, in_f, group_size=32):
        """Create a mock module with CompressedLinear-like attributes."""
        module = nn.Module()
        n_groups = in_f // group_size
        pack_factor = 8
        packed_dim = in_f // pack_factor

        # Register as parameters (matching CompressedLinear behavior)
        module.register_parameter(
            'weight_packed',
            nn.Parameter(torch.randint(0, 2**31, (out_f, packed_dim), dtype=torch.int32), requires_grad=False),
        )
        module.register_parameter(
            'weight_scale',
            nn.Parameter(torch.randn(out_f, n_groups, dtype=torch.bfloat16).abs() * 0.1, requires_grad=False),
        )
        module.register_buffer(
            'weight_shape',
            torch.tensor([out_f, in_f], dtype=torch.int32),
        )
        return module

    def _make_mock_moe(self, n_experts, hidden_size, intermediate_size, group_size=32):
        """Create a mock MoE module with CompressedLinear experts."""
        moe = nn.Module()
        experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = nn.Module()
            expert.gate_proj = self._make_mock_compressed_linear(intermediate_size, hidden_size, group_size)
            expert.up_proj = self._make_mock_compressed_linear(intermediate_size, hidden_size, group_size)
            expert.down_proj = self._make_mock_compressed_linear(hidden_size, intermediate_size, group_size)
            experts.append(expert)
        moe.experts = experts
        return moe

    def test_stacking_shapes(self):
        """Verify stacked tensors have correct shapes."""
        n_experts, H, I = 4, 64, 32
        moe = self._make_mock_moe(n_experts, H, I)

        _fuse_expert_weights(moe)

        # Check stacked shapes
        assert moe._gate_packed.shape == (n_experts, I, H // 8)
        assert moe._gate_scale.shape == (n_experts, I, H // 32)
        assert moe._gate_shape == (I, H)

        assert moe._up_packed.shape == (n_experts, I, H // 8)
        assert moe._up_scale.shape == (n_experts, I, H // 32)
        assert moe._up_shape == (I, H)

        assert moe._down_packed.shape == (n_experts, H, I // 8)
        assert moe._down_scale.shape == (n_experts, H, I // 32)
        assert moe._down_shape == (H, I)

    def test_individual_weights_freed(self):
        """Verify individual expert weight tensors are deleted after stacking."""
        n_experts, H, I = 4, 64, 32
        moe = self._make_mock_moe(n_experts, H, I)

        _fuse_expert_weights(moe)

        for i in range(n_experts):
            for proj_name in ('gate_proj', 'up_proj', 'down_proj'):
                proj = getattr(moe.experts[i], proj_name)
                assert 'weight_packed' not in proj._parameters
                assert 'weight_packed' not in proj._buffers
                assert 'weight_scale' not in proj._parameters
                assert 'weight_scale' not in proj._buffers
                assert 'weight_shape' not in proj._buffers

    def test_stacked_data_matches_originals(self):
        """Verify stacked tensors contain the exact same data as originals."""
        n_experts, H, I = 4, 64, 32
        moe = self._make_mock_moe(n_experts, H, I)

        # Save original data for comparison
        originals = {}
        for i in range(n_experts):
            originals[i] = {
                'gate_packed': moe.experts[i].gate_proj.weight_packed.data.clone(),
                'gate_scale': moe.experts[i].gate_proj.weight_scale.data.clone(),
                'up_packed': moe.experts[i].up_proj.weight_packed.data.clone(),
                'down_packed': moe.experts[i].down_proj.weight_packed.data.clone(),
            }

        _fuse_expert_weights(moe)

        for i in range(n_experts):
            assert torch.equal(moe._gate_packed[i], originals[i]['gate_packed'])
            assert torch.equal(moe._gate_scale[i], originals[i]['gate_scale'])
            assert torch.equal(moe._up_packed[i], originals[i]['up_packed'])
            assert torch.equal(moe._down_packed[i], originals[i]['down_packed'])
