"""Vector quality metrics on held-out activations.

Input:
    - vector: trained trait direction (1D tensor)
    - val_pos, val_neg: held-out activations (in-distribution)
    - ood_pos, ood_neg: held-out activations (out-of-distribution, optional)

Output:
    Dict with val_accuracy, val_effect_size, val_auroc, polarity_correct
    plus ood_* keys when OOD activations are non-empty.

Usage:
    from core.validation import compute_vector_quality
    metrics = compute_vector_quality(vector, val_pos, val_neg, ood_pos, ood_neg)
"""

from typing import Dict, Optional
import torch

from .math import (
    accuracy,
    auroc,
    batch_cosine_similarity,
    effect_size,
    polarity_correct,
)


def compute_vector_quality(
    vector: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    ood_pos: Optional[torch.Tensor] = None,
    ood_neg: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Project held-out activations onto vector and compute classification metrics.

    Empty tensors (numel() == 0) skip that split. Polarity for ID is the
    midpoint-threshold accuracy direction; OOD reports its own polarity.
    """
    metrics: Dict[str, float] = {}

    if val_pos.numel() > 0 and val_neg.numel() > 0:
        proj_pos = batch_cosine_similarity(val_pos, vector)
        proj_neg = batch_cosine_similarity(val_neg, vector)
        metrics["val_accuracy"] = float(accuracy(proj_pos, proj_neg))
        metrics["val_effect_size"] = float(effect_size(proj_pos, proj_neg))
        metrics["val_auroc"] = float(auroc(proj_pos, proj_neg))
        metrics["polarity_correct"] = bool(polarity_correct(proj_pos, proj_neg))

    if ood_pos is not None and ood_neg is not None and ood_pos.numel() > 0 and ood_neg.numel() > 0:
        proj_pos = batch_cosine_similarity(ood_pos, vector)
        proj_neg = batch_cosine_similarity(ood_neg, vector)
        metrics["ood_accuracy"] = float(accuracy(proj_pos, proj_neg))
        metrics["ood_effect_size"] = float(effect_size(proj_pos, proj_neg))
        metrics["ood_auroc"] = float(auroc(proj_pos, proj_neg))
        metrics["ood_polarity_correct"] = bool(polarity_correct(proj_pos, proj_neg))

    return metrics
