"""
Evaluation metrics for trait vectors.

These functions measure how well a trait vector separates positive from negative
examples. All functions are pure computations with no file I/O or model-specific code.

Metrics are organized into categories:
- Separation: How well does the vector distinguish pos/neg? (accuracy, separation, effect_size)
- Statistical: Is the separation real or noise? (p_value)
- Stability: How robust is the vector? (bootstrap, noise, subsample)
- Vector properties: What does the vector look like? (sparsity, effective_rank)
- Cross-vector: How do vectors relate? (orthogonality)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


# =============================================================================
# Separation Metrics - How well does the vector distinguish pos/neg?
# =============================================================================

def separation(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """
    Compute separation between positive and negative projections.

    Separation is the absolute difference between mean projections.
    Higher values indicate the vector better distinguishes the classes.

    Args:
        pos_proj: Projections of positive examples [n_pos]
        neg_proj: Projections of negative examples [n_neg]

    Returns:
        Absolute difference between means (always positive)

    Example:
        >>> pos_proj = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_proj = torch.tensor([-0.5, -0.6, -0.4])
        >>> separation(pos_proj, neg_proj)
        1.3  # mean(pos) - mean(neg) = 0.8 - (-0.5)
    """
    return (pos_proj.mean() - neg_proj.mean()).abs().item()


def accuracy(
    pos_proj: torch.Tensor,
    neg_proj: torch.Tensor,
    threshold: Optional[float] = None
) -> float:
    """
    Compute classification accuracy using a threshold.

    By default, uses the midpoint between class means as threshold.
    Positive examples should project above threshold, negative below.

    Args:
        pos_proj: Projections of positive examples [n_pos]
        neg_proj: Projections of negative examples [n_neg]
        threshold: Classification threshold. If None, uses midpoint of means.

    Returns:
        Accuracy as fraction correct (0.0 to 1.0)

    Example:
        >>> pos_proj = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_proj = torch.tensor([-0.5, -0.6, -0.4])
        >>> accuracy(pos_proj, neg_proj)
        1.0  # All correctly classified
    """
    if threshold is None:
        threshold = (pos_proj.mean() + neg_proj.mean()) / 2

    pos_correct = (pos_proj > threshold).float().mean().item()
    neg_correct = (neg_proj <= threshold).float().mean().item()

    return (pos_correct + neg_correct) / 2


def effect_size(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """
    Compute Cohen's d effect size.

    Effect size measures separation in units of standard deviation.
    Guidelines: 0.2 = small, 0.5 = medium, 0.8 = large.

    Args:
        pos_proj: Projections of positive examples [n_pos]
        neg_proj: Projections of negative examples [n_neg]

    Returns:
        Cohen's d (absolute value, always positive)

    Example:
        >>> pos_proj = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_proj = torch.tensor([-0.5, -0.6, -0.4])
        >>> effect_size(pos_proj, neg_proj)
        ~13.0  # Very large effect (well-separated)
    """
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)

    if pooled_std <= 0:
        return 0.0

    d = (pos_proj.mean() - neg_proj.mean()).abs() / pooled_std
    return d.item()


# =============================================================================
# Statistical Metrics - Is the separation real or noise?
# =============================================================================

def p_value(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """
    Compute p-value from independent samples t-test.

    Tests whether the difference between groups is statistically significant.
    Lower values indicate stronger evidence of real separation.

    Args:
        pos_proj: Projections of positive examples [n_pos]
        neg_proj: Projections of negative examples [n_neg]

    Returns:
        Two-tailed p-value (0.0 to 1.0)

    Example:
        >>> pos_proj = torch.tensor([0.8, 0.9, 0.7, 0.85])
        >>> neg_proj = torch.tensor([-0.5, -0.6, -0.4, -0.55])
        >>> p_value(pos_proj, neg_proj)
        ~0.0001  # Highly significant
    """
    _, p = stats.ttest_ind(
        pos_proj.cpu().numpy(),
        neg_proj.cpu().numpy()
    )
    return float(p)


def polarity_correct(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> bool:
    """
    Check if polarity is correct (positive examples score higher).

    For most traits, we expect positive examples to have higher projections.
    If this returns False, the vector may need to be negated.

    Args:
        pos_proj: Projections of positive examples [n_pos]
        neg_proj: Projections of negative examples [n_neg]

    Returns:
        True if mean(pos) > mean(neg), False otherwise
    """
    return bool(pos_proj.mean() > neg_proj.mean())


# =============================================================================
# Stability Metrics - How robust is the vector?
# =============================================================================

def bootstrap_stability(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor,
    n_samples: int = 100,
    sample_fraction: float = 0.8
) -> Dict[str, float]:
    """
    Measure stability via bootstrap resampling.

    Repeatedly samples subsets of the data and measures variance in separation.
    Lower std indicates more stable/reliable vector.

    Args:
        pos_acts: Positive activations [n_pos, hidden_dim]
        neg_acts: Negative activations [n_neg, hidden_dim]
        vector: Trait vector [hidden_dim]
        n_samples: Number of bootstrap samples
        sample_fraction: Fraction of data to sample each iteration

    Returns:
        Dict with 'mean', 'std', 'cv' (coefficient of variation)

    Example:
        >>> stability = bootstrap_stability(pos_acts, neg_acts, vector)
        >>> print(f"Separation: {stability['mean']:.2f} ± {stability['std']:.2f}")
    """
    separations = []

    n_pos_sample = max(1, int(len(pos_acts) * sample_fraction))
    n_neg_sample = max(1, int(len(neg_acts) * sample_fraction))

    for _ in range(n_samples):
        # Sample with replacement
        pos_idx = torch.randint(0, len(pos_acts), (n_pos_sample,))
        neg_idx = torch.randint(0, len(neg_acts), (n_neg_sample,))

        pos_sample = pos_acts[pos_idx]
        neg_sample = neg_acts[neg_idx]

        # Compute separation on this sample
        pos_proj = pos_sample @ vector
        neg_proj = neg_sample @ vector
        sep = (pos_proj.mean() - neg_proj.mean()).abs().item()
        separations.append(sep)

    mean_sep = np.mean(separations)
    std_sep = np.std(separations)
    cv = std_sep / mean_sep if mean_sep > 0 else float('inf')

    return {
        'mean': mean_sep,
        'std': std_sep,
        'cv': cv  # Coefficient of variation (lower = more stable)
    }


def noise_robustness(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor,
    noise_level: float = 0.1
) -> float:
    """
    Measure robustness to noise in the vector.

    Adds Gaussian noise to the vector and measures how much separation drops.
    Values close to 1.0 indicate the vector is robust to perturbations.

    Args:
        pos_acts: Positive activations [n_pos, hidden_dim]
        neg_acts: Negative activations [n_neg, hidden_dim]
        vector: Trait vector [hidden_dim]
        noise_level: Noise magnitude as fraction of vector norm

    Returns:
        Ratio of noisy separation to clean separation (0.0 to ~1.0+)

    Example:
        >>> robustness = noise_robustness(pos_acts, neg_acts, vector)
        >>> print(f"Robustness: {robustness:.2%}")  # e.g., "95%" means 5% drop
    """
    # Clean separation
    pos_proj_clean = pos_acts @ vector
    neg_proj_clean = neg_acts @ vector
    clean_sep = (pos_proj_clean.mean() - neg_proj_clean.mean()).abs()

    if clean_sep <= 0:
        return 0.0

    # Add noise
    noise = torch.randn_like(vector) * vector.norm() * noise_level
    noisy_vector = vector + noise

    # Noisy separation
    pos_proj_noisy = pos_acts @ noisy_vector
    neg_proj_noisy = neg_acts @ noisy_vector
    noisy_sep = (pos_proj_noisy.mean() - neg_proj_noisy.mean()).abs()

    return (noisy_sep / clean_sep).item()


def subsample_stability(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor,
    fraction: float = 0.5
) -> float:
    """
    Measure stability when using only a fraction of the data.

    Values close to 1.0 indicate the vector generalizes well and isn't
    overfitting to specific examples.

    Args:
        pos_acts: Positive activations [n_pos, hidden_dim]
        neg_acts: Negative activations [n_neg, hidden_dim]
        vector: Trait vector [hidden_dim]
        fraction: Fraction of data to use (default 0.5 = half)

    Returns:
        Ratio of subsample separation to full separation
    """
    # Full separation
    pos_proj_full = pos_acts @ vector
    neg_proj_full = neg_acts @ vector
    full_sep = (pos_proj_full.mean() - neg_proj_full.mean()).abs()

    if full_sep <= 0:
        return 0.0

    # Subsample separation
    n_pos = max(1, int(len(pos_acts) * fraction))
    n_neg = max(1, int(len(neg_acts) * fraction))

    pos_proj_sub = pos_acts[:n_pos] @ vector
    neg_proj_sub = neg_acts[:n_neg] @ vector
    sub_sep = (pos_proj_sub.mean() - neg_proj_sub.mean()).abs()

    return (sub_sep / full_sep).item()


# =============================================================================
# Vector Property Metrics - What does the vector look like?
# =============================================================================

def sparsity(vector: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Measure sparsity of the vector.

    Sparsity is the fraction of components near zero.
    High sparsity may indicate the vector captures a localized feature.

    Args:
        vector: Trait vector [hidden_dim]
        threshold: Components below this fraction of max are "near zero"

    Returns:
        Fraction of near-zero components (0.0 to 1.0)

    Example:
        >>> vector = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        >>> sparsity(vector)
        0.8  # 4 of 5 components are zero
    """
    abs_threshold = vector.abs().max() * threshold
    return (vector.abs() < abs_threshold).float().mean().item()


def effective_rank(vector: torch.Tensor, threshold: float = 0.9) -> int:
    """
    Estimate effective dimensionality of the vector.

    Counts how many components are needed to capture `threshold` of the total mass.
    Lower values indicate the vector is concentrated in fewer dimensions.

    Args:
        vector: Trait vector [hidden_dim]
        threshold: Cumulative mass threshold (default 0.9 = 90%)

    Returns:
        Number of components needed to reach threshold

    Example:
        >>> # Vector concentrated in 10 dimensions
        >>> effective_rank(vector)
        10
    """
    sorted_abs = vector.abs().sort(descending=True)[0]
    cumsum = sorted_abs.cumsum(0)
    total = cumsum[-1]

    if total <= 0:
        return len(vector)

    return int((cumsum < total * threshold).sum().item()) + 1


def top_k_concentration(vector: torch.Tensor, k_fraction: float = 0.05) -> float:
    """
    Measure how much mass is concentrated in the top k% of components.

    Higher values indicate the vector is more interpretable (fewer important dims).

    Args:
        vector: Trait vector [hidden_dim]
        k_fraction: Top fraction to consider (default 0.05 = top 5%)

    Returns:
        Fraction of total mass in top k% (0.0 to 1.0)

    Example:
        >>> top_k_concentration(vector)
        0.7  # Top 5% of dims contain 70% of the mass
    """
    sorted_abs = vector.abs().sort(descending=True)[0]
    k = max(1, int(len(vector) * k_fraction))

    total = sorted_abs.sum()
    if total <= 0:
        return 0.0

    return (sorted_abs[:k].sum() / total).item()


# =============================================================================
# Cross-Vector Metrics - How do vectors relate to each other?
# =============================================================================

def orthogonality(
    vector: torch.Tensor,
    other_vectors: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Measure orthogonality to other trait vectors.

    Ideally, trait vectors should be independent (orthogonal).
    High correlation with other traits indicates potential confounding.

    Args:
        vector: Trait vector to analyze [hidden_dim]
        other_vectors: List of other trait vectors to compare against

    Returns:
        Dict with 'mean_abs_correlation', 'max_abs_correlation', 'independence_score'

    Example:
        >>> ortho = orthogonality(refusal_vec, [evil_vec, helpful_vec])
        >>> print(f"Independence: {ortho['independence_score']:.2%}")
    """
    if not other_vectors:
        return {
            'mean_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'independence_score': 1.0
        }

    correlations = []
    vector_norm = vector / (vector.norm() + 1e-8)

    for other in other_vectors:
        other_norm = other / (other.norm() + 1e-8)
        corr = (vector_norm @ other_norm).abs().item()
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    max_corr = max(correlations)

    return {
        'mean_abs_correlation': mean_corr,
        'max_abs_correlation': max_corr,
        'independence_score': 1.0 - max_corr  # Higher = more independent
    }


def cross_trait_accuracy(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor
) -> float:
    """
    Test a vector on activations from a different trait.

    Used to build cross-trait interference matrices. If trait A's vector
    classifies trait B's data at 50%, the traits are independent.

    Args:
        pos_acts: Positive activations from another trait [n_pos, hidden_dim]
        neg_acts: Negative activations from another trait [n_neg, hidden_dim]
        vector: Trait vector to test [hidden_dim]

    Returns:
        Accuracy (should be ~0.5 for independent traits)
    """
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector
    return accuracy(pos_proj, neg_proj)


# =============================================================================
# Convenience Functions - Compute multiple metrics at once
# =============================================================================

def evaluate_vector(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    vector: torch.Tensor,
    normalize: bool = True,
    include_stability: bool = False,
    stability_samples: int = 50
) -> Dict[str, Union[float, bool, Dict]]:
    """
    Compute all standard evaluation metrics for a vector.

    This is the main entry point for evaluation. Returns a dict with:
    - accuracy, separation, effect_size, p_value, polarity_correct
    - pos_mean, neg_mean (for debugging)
    - Optionally: stability metrics (bootstrap, noise, subsample)

    Args:
        pos_acts: Positive activations [n_pos, hidden_dim]
        neg_acts: Negative activations [n_neg, hidden_dim]
        vector: Trait vector [hidden_dim]
        normalize: If True, normalize vector and activations (cosine similarity)
        include_stability: If True, compute stability metrics (slower)
        stability_samples: Number of bootstrap samples if include_stability=True

    Returns:
        Dict with all metrics

    Example:
        >>> metrics = evaluate_vector(pos_acts, neg_acts, vector)
        >>> print(f"Accuracy: {metrics['accuracy']:.1%}")
        >>> print(f"Effect size: {metrics['effect_size']:.2f}")
    """
    # Convert to float32 for numerical stability
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()
    vector = vector.float()

    # Optionally normalize (for cosine similarity)
    if normalize:
        vector = vector / (vector.norm() + 1e-8)
        pos_acts = pos_acts / (pos_acts.norm(dim=1, keepdim=True) + 1e-8)
        neg_acts = neg_acts / (neg_acts.norm(dim=1, keepdim=True) + 1e-8)

    # Project
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    # Core metrics
    result = {
        'accuracy': accuracy(pos_proj, neg_proj),
        'separation': separation(pos_proj, neg_proj),
        'effect_size': effect_size(pos_proj, neg_proj),
        'p_value': p_value(pos_proj, neg_proj),
        'polarity_correct': polarity_correct(pos_proj, neg_proj),
        'pos_mean': pos_proj.mean().item(),
        'neg_mean': neg_proj.mean().item(),
    }

    # Optional stability metrics
    if include_stability:
        # Need un-normalized activations for stability tests
        pos_acts_raw = pos_acts if not normalize else pos_acts * pos_acts.norm(dim=1, keepdim=True)
        neg_acts_raw = neg_acts if not normalize else neg_acts * neg_acts.norm(dim=1, keepdim=True)

        result['stability'] = {
            'bootstrap': bootstrap_stability(pos_acts_raw, neg_acts_raw, vector, n_samples=stability_samples),
            'noise_robustness': noise_robustness(pos_acts_raw, neg_acts_raw, vector),
            'subsample': subsample_stability(pos_acts_raw, neg_acts_raw, vector),
        }

    return result


def evaluate_vector_properties(vector: torch.Tensor) -> Dict[str, float]:
    """
    Compute vector property metrics (no activations needed).

    Args:
        vector: Trait vector [hidden_dim]

    Returns:
        Dict with norm, sparsity, effective_rank, top_k_concentration
    """
    return {
        'norm': vector.norm().item(),
        'sparsity': sparsity(vector),
        'effective_rank': effective_rank(vector),
        'top_k_concentration': top_k_concentration(vector),
    }
