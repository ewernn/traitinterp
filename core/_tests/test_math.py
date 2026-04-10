"""
Tests for core/math.py - projection, similarity, and separation functions.

Run: pytest core/_tests/test_math.py -v
"""

import pytest
import torch

from core.math import (
    projection,
    cosine_similarity,
    batch_cosine_similarity,
    accuracy,
    effect_size,
    remove_massive_dims,
    orthogonalize,
    pairwise_cosine_matrix,
    pca,
    project_out_subspace,
)


# =============================================================================
# projection() tests
# =============================================================================

class TestProjection:
    """Tests for projection()."""

    def test_basic_shape(self, sample_activations, sample_vector):
        """Output shape matches input batch dimensions."""
        result = projection(sample_activations, sample_vector)
        assert result.shape == (sample_activations.shape[0],)

    def test_parallel_vectors_positive(self, hidden_dim):
        """Projecting vector onto itself gives positive result."""
        v = torch.randn(hidden_dim)
        v_normalized = v / v.norm()
        # Activation = the vector itself
        acts = v_normalized.unsqueeze(0)
        result = projection(acts, v)
        # Should be close to 1.0 (unit projection onto unit vector)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_zero(self, hidden_dim):
        """Projecting orthogonal vectors gives zero."""
        # Create two orthogonal vectors
        v1 = torch.zeros(hidden_dim)
        v1[0] = 1.0
        v2 = torch.zeros(hidden_dim)
        v2[1] = 1.0

        acts = v1.unsqueeze(0)
        result = projection(acts, v2)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_negative(self, hidden_dim):
        """Projecting opposite vectors gives negative result."""
        v = torch.randn(hidden_dim)
        v_normalized = v / v.norm()
        acts = (-v_normalized).unsqueeze(0)
        result = projection(acts, v)
        assert result.item() == pytest.approx(-1.0, abs=1e-5)

    def test_normalization_effect(self, sample_activations, hidden_dim):
        """normalize_vector=True gives consistent scale."""
        v = torch.randn(hidden_dim) * 100  # Large magnitude
        result_normalized = projection(sample_activations, v, normalize_vector=True)
        result_unnormalized = projection(sample_activations, v, normalize_vector=False)
        # Unnormalized should be ||v|| times larger (||v|| ≈ sqrt(dim) * 100)
        expected_ratio = v.norm().item()
        ratio = result_unnormalized.abs().mean() / result_normalized.abs().mean()
        assert ratio == pytest.approx(expected_ratio, rel=0.1)

    def test_dimension_mismatch_raises(self, sample_activations):
        """Mismatched dimensions raise AssertionError."""
        wrong_dim_vector = torch.randn(sample_activations.shape[-1] + 1)
        with pytest.raises(AssertionError, match="Hidden dim mismatch"):
            projection(sample_activations, wrong_dim_vector)

    def test_dtype_conversion(self, hidden_dim):
        """bfloat16 inputs produce float32 internally."""
        acts = torch.randn(4, hidden_dim, dtype=torch.bfloat16)
        vec = torch.randn(hidden_dim, dtype=torch.bfloat16)
        result = projection(acts, vec)
        # Result should be float32 (from internal conversion)
        assert result.dtype == torch.float32

    def test_zero_vector_no_nan(self, sample_activations, zero_vector):
        """Zero vector doesn't produce NaN (epsilon prevents div by zero)."""
        result = projection(sample_activations, zero_vector)
        assert not torch.isnan(result).any()

    def test_multidim_batch(self, hidden_dim):
        """Works with [batch, seq, hidden] shaped inputs."""
        acts = torch.randn(2, 5, hidden_dim)  # batch=2, seq=5
        vec = torch.randn(hidden_dim)
        result = projection(acts, vec)
        assert result.shape == (2, 5)

    def test_large_values_stable(self, hidden_dim):
        """Large activation values don't cause overflow."""
        acts = torch.randn(4, hidden_dim) * 1e6
        vec = torch.randn(hidden_dim)
        result = projection(acts, vec)
        assert torch.isfinite(result).all()


# =============================================================================
# cosine_similarity() tests
# =============================================================================

class TestCosineSimilarity:
    """Tests for cosine_similarity()."""

    def test_identical_is_one(self, sample_vector):
        """Identical vectors have similarity 1.0."""
        result = cosine_similarity(sample_vector, sample_vector)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_opposite_is_negative_one(self, sample_vector):
        """Opposite vectors have similarity -1.0."""
        result = cosine_similarity(sample_vector, -sample_vector)
        assert result.item() == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_is_zero(self, hidden_dim):
        """Orthogonal vectors have similarity 0.0."""
        v1 = torch.zeros(hidden_dim)
        v1[0] = 1.0
        v2 = torch.zeros(hidden_dim)
        v2[1] = 1.0
        result = cosine_similarity(v1, v2)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_scale_invariant(self, sample_vector):
        """Scaling doesn't affect similarity."""
        v_scaled = sample_vector * 1000
        result = cosine_similarity(sample_vector, v_scaled)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_no_nan(self, sample_vector, zero_vector):
        """Zero vector doesn't produce NaN."""
        result = cosine_similarity(sample_vector, zero_vector)
        assert not torch.isnan(result)


# =============================================================================
# batch_cosine_similarity() tests
# =============================================================================

class TestBatchCosineSimilarity:
    """Tests for batch_cosine_similarity()."""

    def test_basic_shape(self, sample_activations, sample_vector):
        """Output shape matches input batch dimensions."""
        result = batch_cosine_similarity(sample_activations, sample_vector)
        assert result.shape == (sample_activations.shape[0],)

    def test_range_bounded(self, sample_activations, sample_vector):
        """All outputs in [-1, 1]."""
        result = batch_cosine_similarity(sample_activations, sample_vector)
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_identical_batch_all_one(self, sample_vector):
        """Batch of identical vectors all give 1.0."""
        acts = sample_vector.unsqueeze(0).repeat(10, 1)
        result = batch_cosine_similarity(acts, sample_vector)
        assert torch.allclose(result, torch.ones(10), atol=1e-5)

    def test_multidim_shape(self, hidden_dim):
        """Works with [batch, seq, hidden] and returns [batch, seq]."""
        acts = torch.randn(3, 7, hidden_dim)
        vec = torch.randn(hidden_dim)
        result = batch_cosine_similarity(acts, vec)
        assert result.shape == (3, 7)

    def test_zero_activations_no_nan(self, zero_vector, sample_vector):
        """Zero activations don't produce NaN."""
        acts = zero_vector.unsqueeze(0).repeat(5, 1)
        result = batch_cosine_similarity(acts, sample_vector)
        assert not torch.isnan(result).any()


# =============================================================================
# effect_size() and related tests
# =============================================================================

class TestEffectSize:
    """Tests for effect_size() and accuracy()."""

    def test_effect_size_large(self):
        """Well-separated data has large Cohen's d."""
        pos = torch.randn(100) + 5.0
        neg = torch.randn(100) - 5.0
        d = effect_size(pos, neg)
        assert d > 2.0  # Very large effect

    def test_effect_size_small(self):
        """Overlapping data has small Cohen's d."""
        pos = torch.randn(100) + 0.1
        neg = torch.randn(100) - 0.1
        d = effect_size(pos, neg)
        assert d < 0.5  # Small effect

    def test_accuracy_perfect(self):
        """Perfect separation gives 100% accuracy."""
        pos = torch.tensor([10.0, 11.0, 12.0])
        neg = torch.tensor([0.0, 1.0, 2.0])
        acc = accuracy(pos, neg)
        assert acc == pytest.approx(1.0, abs=0.01)

    def test_accuracy_random(self):
        """Identical distributions give ~50% accuracy."""
        torch.manual_seed(42)
        pos = torch.randn(1000)
        neg = torch.randn(1000)
        acc = accuracy(pos, neg)
        assert acc == pytest.approx(0.5, abs=0.1)


# =============================================================================
# remove_massive_dims() tests
# =============================================================================

class TestRemoveMassiveDims:
    """Tests for remove_massive_dims()."""

    def test_zeros_correct_indices(self, sample_activations):
        """Specified dims are zeroed."""
        dims = [0, 5, 10]
        result = remove_massive_dims(sample_activations, dims)
        for dim in dims:
            assert (result[..., dim] == 0).all()

    def test_other_dims_unchanged(self, sample_activations):
        """Non-specified dims are unchanged."""
        dims = [0, 1]
        original = sample_activations.clone()
        result = remove_massive_dims(sample_activations, dims)
        # Check dim 2 is unchanged
        assert torch.equal(result[..., 2], original[..., 2])

    def test_clone_true_preserves_original(self, sample_activations):
        """clone=True doesn't modify original."""
        original = sample_activations.clone()
        _ = remove_massive_dims(sample_activations, [0, 1], clone=True)
        assert torch.equal(sample_activations, original)

    def test_clone_false_modifies_inplace(self, sample_activations):
        """clone=False modifies in-place."""
        _ = remove_massive_dims(sample_activations, [0], clone=False)
        assert (sample_activations[..., 0] == 0).all()

    def test_out_of_bounds_ignored(self, sample_activations):
        """Dims >= hidden_dim are silently ignored."""
        hidden_dim = sample_activations.shape[-1]
        dims = [0, hidden_dim + 100, hidden_dim + 200]
        # Should not raise
        result = remove_massive_dims(sample_activations, dims)
        assert (result[..., 0] == 0).all()

    def test_empty_dims_unchanged(self, sample_activations):
        """Empty dims list returns input unchanged."""
        original = sample_activations.clone()
        result = remove_massive_dims(sample_activations, [])
        assert torch.equal(result, original)


# =============================================================================
# orthogonalize() tests
# =============================================================================

class TestOrthogonalize:
    """Tests for orthogonalize()."""

    def test_removes_component(self, hidden_dim):
        """Result is orthogonal to 'onto' vector."""
        v = torch.randn(hidden_dim)
        onto = torch.randn(hidden_dim)
        result = orthogonalize(v, onto)
        # Dot product should be ~0
        dot = (result @ onto).item()
        assert dot == pytest.approx(0.0, abs=1e-5)

    def test_zero_onto_unchanged(self, sample_vector, zero_vector):
        """Orthogonalizing onto zero vector returns original."""
        result = orthogonalize(sample_vector, zero_vector)
        assert torch.allclose(result, sample_vector, atol=1e-5)

    def test_parallel_gives_zero(self, sample_vector):
        """Orthogonalizing parallel vectors gives zero."""
        result = orthogonalize(sample_vector, sample_vector)
        assert result.norm().item() == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# pairwise_cosine_matrix() tests
# =============================================================================

class TestPairwiseCosineMatrix:
    """Tests for pairwise_cosine_matrix()."""

    def test_shape(self):
        """Output is N×N for N vectors."""
        vecs = torch.randn(10, 64)
        result = pairwise_cosine_matrix(vecs)
        assert result.shape == (10, 10)

    def test_diagonal_is_one(self):
        """Diagonal entries are 1.0 for non-zero vectors."""
        vecs = torch.randn(5, 32)
        result = pairwise_cosine_matrix(vecs)
        assert torch.allclose(result.diag(), torch.ones(5), atol=1e-5)

    def test_symmetric(self):
        """Matrix is symmetric."""
        vecs = torch.randn(8, 64)
        result = pairwise_cosine_matrix(vecs)
        assert torch.allclose(result, result.T, atol=1e-6)

    def test_range_bounded(self):
        """All entries in [-1, 1]."""
        vecs = torch.randn(10, 64)
        result = pairwise_cosine_matrix(vecs)
        assert (result >= -1.0 - 1e-5).all() and (result <= 1.0 + 1e-5).all()

    def test_consistent_with_cosine_similarity(self, hidden_dim):
        """Entries match cosine_similarity for individual pairs."""
        vecs = torch.randn(5, hidden_dim)
        matrix = pairwise_cosine_matrix(vecs)
        for i in range(5):
            for j in range(5):
                expected = cosine_similarity(vecs[i], vecs[j]).item()
                assert matrix[i, j].item() == pytest.approx(expected, abs=1e-5)

    def test_single_vector(self):
        """N=1 returns [[1.0]]."""
        vecs = torch.randn(1, 32)
        result = pairwise_cosine_matrix(vecs)
        assert result.shape == (1, 1)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_no_nan(self):
        """Zero vectors produce 0.0 similarity, not NaN."""
        vecs = torch.zeros(3, 32)
        vecs[1] = torch.randn(32)
        result = pairwise_cosine_matrix(vecs)
        assert not torch.isnan(result).any()


# =============================================================================
# pca() tests
# =============================================================================

class TestPCA:
    """Tests for pca()."""

    def test_output_shapes(self):
        """Components, variance, projections have correct shapes."""
        vecs = torch.randn(20, 64)
        components, var_ratio, projections = pca(vecs, n_components=5)
        assert components.shape == (5, 64)
        assert var_ratio.shape == (5,)
        assert projections.shape == (20, 5)

    def test_variance_ratio_sums_below_one(self):
        """Explained variance ratios sum to <= 1."""
        vecs = torch.randn(50, 32)
        _, var_ratio, _ = pca(vecs, n_components=10)
        assert var_ratio.sum().item() <= 1.0 + 1e-5

    def test_components_orthogonal(self):
        """Principal components are orthogonal."""
        vecs = torch.randn(30, 64)
        components, _, _ = pca(vecs, n_components=5)
        gram = components @ components.T
        expected = torch.eye(5)
        assert torch.allclose(gram, expected, atol=1e-4)

    def test_n_components_exceeds_n(self):
        """Requesting more components than vectors works (returns min)."""
        vecs = torch.randn(3, 64)
        components, var_ratio, projections = pca(vecs, n_components=10)
        assert components.shape[0] == 3  # min(10, 3)
        assert var_ratio.shape[0] == 3
        assert projections.shape == (3, 3)

    def test_single_vector_raises(self):
        """PCA on 1 vector raises ValueError."""
        vecs = torch.randn(1, 64)
        with pytest.raises(ValueError, match="at least 2"):
            pca(vecs)

    def test_variance_ordering(self):
        """First component explains more variance than second."""
        vecs = torch.randn(50, 32)
        _, var_ratio, _ = pca(vecs, n_components=5)
        for i in range(len(var_ratio) - 1):
            assert var_ratio[i] >= var_ratio[i + 1] - 1e-6


# =============================================================================
# project_out_subspace() tests
# =============================================================================

class TestProjectOutSubspace:
    """Tests for project_out_subspace()."""

    def test_removes_subspace(self):
        """Result is orthogonal to all basis vectors."""
        vecs = torch.randn(10, 64)
        basis = torch.randn(3, 64)
        result = project_out_subspace(vecs, basis)
        Q, _ = torch.linalg.qr(basis.float().T)
        residual = result.float() @ Q
        assert residual.abs().max() < 1e-4

    def test_1d_input(self):
        """Works with a single vector (1D)."""
        vec = torch.randn(64)
        basis = torch.randn(2, 64)
        result = project_out_subspace(vec, basis)
        assert result.shape == (64,)
        Q, _ = torch.linalg.qr(basis.float().T)
        assert (result.float() @ Q).abs().max() < 1e-4

    def test_empty_basis_unchanged(self):
        """Empty basis (K=0) leaves vectors unchanged."""
        vecs = torch.randn(5, 32)
        basis = torch.randn(0, 32)
        result = project_out_subspace(vecs, basis)
        assert torch.allclose(result, vecs.float(), atol=1e-5)

    def test_consistent_with_orthogonalize(self, hidden_dim):
        """Single-direction basis matches orthogonalize() result."""
        v = torch.randn(hidden_dim)
        onto = torch.randn(hidden_dim)
        result_single = orthogonalize(v, onto)
        result_subspace = project_out_subspace(v, onto.unsqueeze(0))
        assert torch.allclose(result_single.float(), result_subspace, atol=1e-4)

    def test_idempotent(self):
        """Projecting twice gives same result as once."""
        vecs = torch.randn(5, 32)
        basis = torch.randn(3, 32)
        once = project_out_subspace(vecs, basis)
        twice = project_out_subspace(once, basis)
        assert torch.allclose(once, twice, atol=1e-4)
