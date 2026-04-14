"""
Math primitives for trait vector analysis.

Input: torch.Tensor (raw activations or vectors)
Output: torch.Tensor (projections, similarities, transformed vectors)

Single-vector operations:
    projection, cosine_similarity, batch_cosine_similarity, orthogonalize

Multi-vector operations:
    pairwise_cosine_matrix, pca, project_out_subspace,
    trait_clusters, representational_similarity, vector_set_comparison, pca_norm_correlation

Vector-group transformations (for cross-trait denoising pipelines):
    grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs

Utility:
    remove_massive_dims, accuracy, effect_size, pearson_correlation, polarity_correct, normalize_projections
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


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


def trait_clusters(vectors: torch.Tensor, k: int = 10, n_init: int = 20, random_state: int = 42):
    """K-means clustering on trait vectors.

    Input: [N, hidden_dim]
    Output: (assignments [N], centroids [k, hidden_dim], inertia float)
    """
    from sklearn.cluster import KMeans
    X = vectors.float().cpu().numpy()
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    km.fit(X)
    return km.labels_, torch.from_numpy(km.cluster_centers_), km.inertia_


def representational_similarity(vectors_by_layer: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Representational similarity analysis across layers.

    Computes cosine similarity of cosine-similarity matrices across layers.

    Input: {layer_idx: [N, hidden_dim]} — same N traits at each layer
    Output: (rsa_matrix [L, L], layer_indices [L])
    """
    layers = sorted(vectors_by_layer.keys())
    sim_matrices = []
    for layer in layers:
        sim = pairwise_cosine_matrix(vectors_by_layer[layer])
        # Extract upper triangle (excluding diagonal) as flat vector
        idx = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
        sim_matrices.append(sim[idx[0], idx[1]])

    sim_stack = torch.stack(sim_matrices)  # [L, n_pairs]
    # RSA: cosine similarity between flattened similarity vectors
    rsa = pairwise_cosine_matrix(sim_stack)
    return rsa, layers


def vector_set_comparison(
    vectors_a: Dict[str, torch.Tensor],
    vectors_b: Dict[str, torch.Tensor],
    vectors_c: Optional[Dict[str, torch.Tensor]] = None,
) -> dict:
    """Compare two (or three) named vector sets sharing the same key space.

    Computes same-key cosine similarity, cross-key cosine matrix, and
    orthogonalizes vectors_a against the PCA subspace of vectors_b to measure
    retained norm. If vectors_c is provided, also computes cross-key cosine
    between vectors_c and vectors_b.

    Input:
        vectors_a: {name: [hidden_dim]} — primary set (e.g., deflection probes)
        vectors_b: {name: [hidden_dim]} — reference set (e.g., story probes)
        vectors_c: optional third set (e.g., displayed-emotion probes)

    Output: dict with same_emotion_cosine, cross_emotion_matrix,
            retained_norm_after_orthogonalization, etc.
    """
    common = sorted(set(vectors_a.keys()) & set(vectors_b.keys()))
    if not common:
        return {"error": "No common keys between vectors_a and vectors_b"}

    # Same-key cosine similarity
    same_key_cos = {}
    for key in common:
        cos = cosine_similarity(vectors_a[key], vectors_b[key]).item()
        same_key_cos[key] = round(cos, 4)

    # Cross-key cosine matrix: vectors_a[k1] vs vectors_b[k2]
    cross_matrix = {}
    for k1 in common:
        cross_matrix[k1] = {}
        for k2 in common:
            cos = cosine_similarity(vectors_a[k1], vectors_b[k2]).item()
            cross_matrix[k1][k2] = round(cos, 4)

    # Third-set similarity (if available)
    c_similarity = {}
    if vectors_c:
        for k1 in common:
            if k1 in vectors_c:
                c_similarity[k1] = {}
                for k2 in common:
                    cos = cosine_similarity(vectors_c[k1], vectors_b[k2]).item()
                    c_similarity[k1][k2] = round(cos, 4)

    # Orthogonalize vectors_a against full vectors_b PCA space, measure retained norm
    b_matrix = torch.stack([vectors_b[k] for k in common])  # [n, hidden]
    retained_norms = {}
    for key in common:
        vec_a = vectors_a[key]
        original_norm = vec_a.norm().item()

        n_components = min(len(common), b_matrix.shape[0])
        components, _, _ = pca(b_matrix, n_components=n_components)
        orthogonalized = project_out_subspace(vec_a, components)
        new_norm = orthogonalized.norm().item()

        retained_norms[key] = round(new_norm / original_norm, 4) if original_norm > 1e-8 else 0.0

    avg_retained = np.mean(list(retained_norms.values()))

    return {
        "same_emotion_cosine": same_key_cos,
        "mean_same_emotion_cosine": round(float(np.mean(list(same_key_cos.values()))), 4),
        "cross_emotion_matrix": cross_matrix,
        "displayed_similarity": c_similarity if c_similarity else None,
        "retained_norm_after_orthogonalization": retained_norms,
        "mean_retained_norm": round(float(avg_retained), 4),
        "n_common_emotions": len(common),
    }


def pca_norm_correlation(
    projections: torch.Tensor,
    trait_names: List[str],
    norms: Dict[str, Tuple[float, float]],
    min_overlap: int = 10,
) -> Optional[Dict]:
    """Pearson correlation of PC1/PC2 projections with human valence/arousal norms.

    Replicates Sofroniew et al. 2026 Fig 8 (PC1 vs pleasure, PC2 vs arousal). Output
    keys use 1-indexed naming (`pc1_vs_valence`, `pc2_vs_arousal`) to match the
    paper's figure labels and downstream visualization expectations.

    Input:
        projections: [N, n_components] PC projections from `core.math.pca()`
        trait_names: list of N trait names
        norms: {trait_name: (valence, arousal)} dict
        min_overlap: minimum overlapping emotions required; returns None if fewer

    Output: dict with {n_overlapping, emotions, pc1_vs_valence, pc2_vs_arousal,
            pc1_vs_arousal, pc2_vs_valence} or None if overlap < min_overlap
    """
    overlapping = [(i, name) for i, name in enumerate(trait_names) if name in norms]
    if len(overlapping) < min_overlap:
        return None

    indices = [i for i, _ in overlapping]
    names_overlap = [n for _, n in overlapping]

    human_valence = torch.tensor([norms[n][0] for n in names_overlap])
    human_arousal = torch.tensor([norms[n][1] for n in names_overlap])

    pc1 = projections[indices, 0]
    pc2 = projections[indices, 1]

    return {
        'n_overlapping': len(overlapping),
        'emotions': names_overlap,
        'pc1_vs_valence': float(pearson_correlation(pc1, human_valence)),
        'pc2_vs_arousal': float(pearson_correlation(pc2, human_arousal)),
        'pc1_vs_arousal': float(pearson_correlation(pc1, human_arousal)),
        'pc2_vs_valence': float(pearson_correlation(pc2, human_valence)),
    }


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

