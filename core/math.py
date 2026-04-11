"""
Math primitives for trait vector analysis.

Input: torch.Tensor (raw activations or vectors)
Output: torch.Tensor (projections, similarities, transformed vectors)

Single-vector operations:
    projection, cosine_similarity, batch_cosine_similarity, orthogonalize

Multi-vector operations:
    pairwise_cosine_matrix, pca, project_out_subspace

Vector-group transformations (for cross-trait denoising pipelines):
    grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs

Utility:
    remove_massive_dims, accuracy, effect_size, pearson_correlation, polarity_correct, normalize_projections
"""

import torch
from typing import List


# =============================================================================
# Massive Activation Handling
# =============================================================================

def remove_massive_dims(
    activations: torch.Tensor,
    dims: List[int],
    clone: bool = True
) -> torch.Tensor:
    """
    Zero out massive activation dimensions.

    Args:
        activations: [*, hidden_dim] tensor
        dims: List of dimension indices to zero out
        clone: If True, return a copy (default). If False, modify in-place.

    Returns:
        Tensor with specified dimensions zeroed out
    """
    if not dims:
        return activations

    if clone:
        activations = activations.clone()

    for dim in dims:
        if dim < activations.shape[-1]:
            activations[..., dim] = 0

    return activations


def projection(
    activations: torch.Tensor,
    vector: torch.Tensor,
    normalize_vector: bool = True
) -> torch.Tensor:
    """
    Project activations onto a trait vector.

    Args:
        activations: [*, hidden_dim]
        vector: [hidden_dim]
        normalize_vector: if True, normalize vector to unit length

    Returns:
        [*] projection scores (higher = more trait expression)
    """
    # Validate dimensions match (catch cross-model vector loading bugs)
    assert activations.shape[-1] == vector.shape[0], (
        f"Hidden dim mismatch: activations {activations.shape[-1]} vs vector {vector.shape[0]}. "
        f"Likely using vector from wrong model."
    )
    # Cast to float32 to avoid dtype mismatches (bfloat16 vs float16)
    activations = activations.float()
    vector = vector.float()
    if normalize_vector:
        vector = vector / (vector.norm() + 1e-8)
    return torch.matmul(activations, vector)


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two vectors. Returns scalar in [-1, 1]."""
    v1 = vec1 / (vec1.norm() + 1e-8)
    v2 = vec2 / (vec2.norm() + 1e-8)
    return (v1 * v2).sum()


def batch_cosine_similarity(
    activations: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine similarity between each activation and a vector.

    Args:
        activations: [*, hidden_dim] - arbitrary leading dimensions
        vector: [hidden_dim]

    Returns:
        [*] cosine similarities in [-1, 1]
    """
    acts = activations.float()
    vec = vector.float()
    acts_norm = acts / (acts.norm(dim=-1, keepdim=True) + 1e-8)
    vec_norm = vec / (vec.norm() + 1e-8)
    return acts_norm @ vec_norm


def orthogonalize(v: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
    """Remove onto's component from v. Returns v with onto projected out."""
    v_flat, onto_flat = v.flatten(), onto.flatten()
    norm_sq = (onto_flat @ onto_flat)
    if norm_sq < 1e-10:
        return v
    proj = (v_flat @ onto_flat) / norm_sq * onto_flat
    return (v_flat - proj).view_as(v)


def accuracy(pos_proj: torch.Tensor, neg_proj: torch.Tensor, threshold: float = None) -> float:
    """Classification accuracy. Positive should score above threshold, negative below."""
    if threshold is None:
        threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    pos_correct = (pos_proj > threshold).float().mean().item()
    neg_correct = (neg_proj <= threshold).float().mean().item()
    return (pos_correct + neg_correct) / 2


def effect_size(pos_proj: torch.Tensor, neg_proj: torch.Tensor, signed: bool = False) -> float:
    """Cohen's d: separation in units of std. 0.2=small, 0.5=medium, 0.8=large.

    Uses pooled standard deviation (assumes roughly equal variance).

    Args:
        signed: If True, preserve sign (positive = pos > neg). Default False (absolute value).
    """
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
    if pooled_std <= 0:
        return 0.0
    d = ((pos_proj.mean() - neg_proj.mean()) / pooled_std).item()
    return d if signed else abs(d)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1D tensors.

    Input: x, y — 1D tensors or lists of the same length
    Output: scalar correlation in [-1, 1]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    num = (x_centered * y_centered).sum()
    den = (x_centered.norm() * y_centered.norm()).clamp(min=1e-8)
    return (num / den).item()


def polarity_correct(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> bool:
    """Check if positive examples score higher than negative."""
    return bool(pos_proj.mean() > neg_proj.mean())


def normalize_projections(
    raw: list, token_norms: list, mode: str = 'normalized'
) -> list:
    """Normalize raw projection scores.

    Args:
        raw: per-token raw projection values (dot product with unit vector)
        token_norms: per-token activation L2 norms at the projection layer
        mode: 'normalized' (divide by mean norm — layer-scale adjusted, default)
              'cosine' (divide by per-token norm — true cosine similarity)
              'raw' (no normalization)

    Returns:
        Normalized projection values as a list of floats.
    """
    if mode == 'raw' or not token_norms:
        return raw
    if mode == 'cosine':
        return [v / n if n > 0 else 0.0 for v, n in zip(raw, token_norms)]
    if mode == 'normalized':
        mean_norm = sum(token_norms) / len(token_norms) if token_norms else 1.0
        return [v / mean_norm if mean_norm > 0 else 0.0 for v in raw]
    raise ValueError(f"Unknown normalization mode: {mode}")


# =============================================================================
# Multi-Vector Operations
# =============================================================================

def pairwise_cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """N×N cosine similarity matrix for a set of vectors.

    Input: [N, hidden_dim]
    Output: [N, N] with values in [-1, 1], diagonal = 1.0
    """
    vectors = vectors.float()
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def pca(vectors: torch.Tensor, n_components: int = 10):
    """PCA on a set of vectors via SVD.

    Input: [N, hidden_dim]
    Output: (components [n_components, hidden_dim],
             explained_variance_ratio [n_components],
             projections [N, n_components])
    """
    if len(vectors) < 2:
        raise ValueError("PCA requires at least 2 vectors")
    vectors = vectors.float()
    centered = vectors - vectors.mean(dim=0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    variance = S ** 2 / (len(vectors) - 1)
    total_var = variance.sum()
    n = min(n_components, len(S))
    return (
        Vt[:n],                              # components
        (variance[:n] / total_var),           # explained variance ratio
        centered @ Vt[:n].T,                  # projections
    )


def project_out_subspace(
    vectors: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Remove subspace components from vectors.

    Generalizes orthogonalize() to multiple directions simultaneously.

    Input:
        vectors: [N, hidden_dim] or [hidden_dim]
        basis: [K, hidden_dim] — orthonormal basis of subspace to remove
    Output:
        vectors with subspace projected out, same shape as input
    """
    vectors = vectors.float()
    basis = basis.float()
    # Orthonormalize basis via QR (in case it isn't already)
    Q, _ = torch.linalg.qr(basis.T)  # [hidden_dim, K]
    # Project out: v - Q @ Q^T @ v
    if vectors.dim() == 1:
        return vectors - Q @ (Q.T @ vectors)
    return vectors - (vectors @ Q) @ Q.T


# =============================================================================
# Vector Group Transformations (for cross-trait denoising pipelines)
# =============================================================================

def grand_mean_center(
    vectors_dict: dict,
) -> "tuple[dict, torch.Tensor]":
    """Subtract the mean of a group of vectors from each.

    Corresponds to step 4 of Sofroniew et al. 2026 §1.1.4: given per-trait mean
    activations, subtract the grand mean across traits to isolate trait-specific
    directions.

    Args:
        vectors_dict: {name: [hidden_dim] tensor}

    Returns:
        centered_dict: {name: centered_tensor}
        grand_mean: [hidden_dim] tensor
    """
    names = sorted(vectors_dict.keys())
    stacked = torch.stack([vectors_dict[n].float() for n in names])
    grand_mean = stacked.mean(dim=0)
    return {n: vectors_dict[n].float() - grand_mean for n in names}, grand_mean


def compute_top_pcs_by_variance(
    activations: torch.Tensor,
    variance_threshold: float = 0.5,
    max_components: int = 50,
) -> "tuple[torch.Tensor, torch.Tensor, int]":
    """PCA + select top components explaining up to variance_threshold cumulative variance.

    Used for Sofroniew et al. 2026 §1.1.4 step 5 (neutral PC denoising).

    Args:
        activations: [N, hidden_dim] raw activations (typically from a neutral corpus)
        variance_threshold: cumulative variance fraction cutoff (default 0.5)
        max_components: cap on PCs to compute (default 50)

    Returns:
        basis: [K, hidden_dim] — top K principal components
        variance_ratio: [K] — per-component explained variance ratio
        n_pcs: K — number of selected components
    """
    n_components = min(max_components, len(activations))
    components, variance_ratio, _ = pca(activations.float(), n_components=n_components)
    cumvar = torch.cumsum(variance_ratio, dim=0)
    n_pcs = int((cumvar < variance_threshold).sum().item()) + 1
    n_pcs = min(n_pcs, len(components))
    return components[:n_pcs], variance_ratio[:n_pcs], n_pcs


def denoise_with_pcs(
    vectors_dict: dict,
    pc_basis: torch.Tensor,
) -> dict:
    """Project out a PC subspace from each vector in a group.

    Thin wrapper over project_out_subspace for the dict-of-vectors case.

    Args:
        vectors_dict: {name: [hidden_dim] tensor}
        pc_basis: [K, hidden_dim] — orthonormal basis to remove

    Returns:
        {name: denoised_tensor}
    """
    return {
        name: project_out_subspace(vec.unsqueeze(0), pc_basis).squeeze(0)
        for name, vec in vectors_dict.items()
    }

