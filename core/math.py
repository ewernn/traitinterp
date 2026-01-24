"""
Math primitives for trait vector analysis.

projection: project activations onto vector (normalizes vector only)
batch_cosine_similarity: cosine similarity between activations and vector (normalizes both)
vector_properties: norm, sparsity of a vector
distribution_properties: std, overlap, margin of projection distributions
remove_massive_dims: zero out massive activation dimensions
"""

import torch
from typing import Dict, List, Callable
from scipy import stats

from core.types import VectorSpec, ProjectionConfig


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


def separation(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """Absolute difference between mean projections."""
    return (pos_proj.mean() - neg_proj.mean()).abs().item()


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


def p_value(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """Two-tailed t-test p-value. Lower = more significant separation."""
    _, p = stats.ttest_ind(pos_proj.cpu().numpy(), neg_proj.cpu().numpy())
    return float(p)


def polarity_correct(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> bool:
    """Check if positive examples score higher than negative."""
    return bool(pos_proj.mean() > neg_proj.mean())


def vector_properties(vector: torch.Tensor) -> Dict[str, float]:
    """
    Compute properties of a vector.

    Args:
        vector: [hidden_dim] tensor

    Returns:
        {'norm': float, 'sparsity': float (fraction of components < 0.01)}
    """
    v = vector.float()
    return {
        'norm': v.norm().item(),
        'sparsity': (v.abs() < 0.01).float().mean().item(),
    }


def distribution_properties(
    pos_proj: torch.Tensor,
    neg_proj: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute properties of projection score distributions.

    Args:
        pos_proj: [n_pos] projection scores for positive examples
        neg_proj: [n_neg] projection scores for negative examples

    Returns:
        Dict[str, float] with keys: pos_std, neg_std, overlap_coefficient, separation_margin
    """
    pos_mean = pos_proj.mean().item()
    neg_mean = neg_proj.mean().item()
    pos_std = float(pos_proj.std())
    neg_std = float(neg_proj.std())

    # Overlap coefficient: use effect_size (Cohen's d) for z-score
    d = effect_size(pos_proj, neg_proj)  # Already computes pooled_std internally
    overlap = max(0, 1 - d / 4.0)  # d=4 → no overlap

    # Separation margin: gap between distributions (positive = good separation)
    margin = (pos_mean - pos_std) - (neg_mean + neg_std)

    return {
        'pos_std': pos_std,
        'neg_std': neg_std,
        'overlap_coefficient': float(overlap),
        'separation_margin': float(margin),
    }


# =============================================================================
# ProjectionConfig-based Operations
# =============================================================================

def project_with_config(
    activations: Dict[int, Dict[str, torch.Tensor]],
    config: ProjectionConfig,
    vector_loader: Callable[[VectorSpec], torch.Tensor],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project activations using a ProjectionConfig (single or ensemble).

    Args:
        activations: Dict[layer][component] -> tensor of shape [*, hidden_dim]
        config: ProjectionConfig specifying which vectors to use
        vector_loader: Function that takes VectorSpec and returns vector tensor
        normalize: If True, normalize weights to sum to 1.0 (default True)

    Returns:
        Weighted sum of projections, same shape as activations minus hidden_dim

    Example:
        def loader(spec):
            vec, _, _ = load_vector_from_spec(experiment, trait, spec)
            return vec

        proj = project_with_config(activations, config, loader)
    """
    weights = config.normalized_weights if normalize else [s.weight for s in config.vectors]

    result = None
    for spec, weight in zip(config.vectors, weights):
        vec = vector_loader(spec)
        act = activations[spec.layer][spec.component]
        proj = projection(act, vec, normalize_vector=True)

        if result is None:
            result = weight * proj
        else:
            result = result + weight * proj

    return result


def project_single(
    activations: torch.Tensor,
    vector: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Project activations onto a single vector with weight.

    Convenience wrapper around projection() that applies weight.

    Args:
        activations: [*, hidden_dim] tensor
        vector: [hidden_dim] tensor
        weight: Scalar weight to apply (default 1.0)

    Returns:
        [*] weighted projection scores
    """
    return weight * projection(activations, vector, normalize_vector=True)
