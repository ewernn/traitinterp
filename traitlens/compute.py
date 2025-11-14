"""
Computational operations for trait vector analysis.

These are the "pandas operations" for trait analysis - fundamental operations
for extracting and analyzing trait vectors from activations.
"""

import torch
from typing import Optional, Tuple, Union


def mean_difference(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> torch.Tensor:
    """
    Compute the mean difference between positive and negative activations.

    This is the fundamental operation for trait vector extraction.

    Args:
        pos_acts: Positive example activations
        neg_acts: Negative example activations
        dim: Dimension(s) to average over. If None, infers based on tensor shape.
             For [batch, seq, hidden], averages over batch and seq by default.

    Returns:
        Trait vector as difference of means

    Example:
        >>> # pos_acts shape: [100, 50, 768]  # 100 examples, 50 tokens, 768 dims
        >>> # neg_acts shape: [100, 50, 768]
        >>> vector = mean_difference(pos_acts, neg_acts)
        >>> vector.shape
        torch.Size([768])
    """
    if dim is None:
        # Infer dimensions to average over
        # Keep only the last dimension (assumed to be hidden/feature dimension)
        if pos_acts.ndim > 1:
            dim = tuple(range(pos_acts.ndim - 1))
        else:
            dim = 0

    pos_mean = pos_acts.mean(dim=dim)
    neg_mean = neg_acts.mean(dim=dim)

    return pos_mean - neg_mean


def compute_derivative(
    expression_trajectory: torch.Tensor,
    dt: float = 1.0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute the first derivative of trait expression over time.

    This represents the "velocity" of trait expression - how fast the trait
    expression is changing from token to token.

    Args:
        expression_trajectory: Trait expression over time [seq_len, ...] or [batch, seq_len, ...]
        dt: Time step (default 1.0 for per-token)
        normalize: Whether to normalize the derivative vectors

    Returns:
        First derivative (velocity) with shape [seq_len-1, ...]

    Example:
        >>> # Track how refusal trait changes over 10 tokens
        >>> trajectory = torch.randn(10, 768)  # 10 tokens, 768 dims
        >>> velocity = compute_derivative(trajectory)
        >>> velocity.shape
        torch.Size([9, 768])  # 9 transitions between 10 tokens
    """
    if expression_trajectory.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points for derivative, got {expression_trajectory.shape[0]}")

    # Compute differences between consecutive time points
    derivative = torch.diff(expression_trajectory, dim=0) / dt

    if normalize:
        # Normalize each derivative vector
        norms = derivative.norm(dim=-1, keepdim=True)
        derivative = derivative / (norms + 1e-8)  # Add epsilon to prevent division by zero

    return derivative


def compute_second_derivative(
    expression_trajectory: torch.Tensor,
    dt: float = 1.0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute the second derivative of trait expression over time.

    This represents the "acceleration" of trait expression - how fast the rate
    of change itself is changing. Useful for detecting commitment points.

    Args:
        expression_trajectory: Trait expression over time [seq_len, ...] or [batch, seq_len, ...]
        dt: Time step (default 1.0 for per-token)
        normalize: Whether to normalize the derivative vectors

    Returns:
        Second derivative (acceleration) with shape [seq_len-2, ...]

    Example:
        >>> trajectory = torch.randn(10, 768)
        >>> acceleration = compute_second_derivative(trajectory)
        >>> acceleration.shape
        torch.Size([8, 768])  # 8 acceleration values from 10 tokens
        >>>
        >>> # Find commitment point (where acceleration drops)
        >>> accel_magnitude = acceleration.norm(dim=-1)
        >>> commitment_point = (accel_magnitude < 0.1).nonzero()[0]
    """
    # First derivative (velocity)
    velocity = compute_derivative(expression_trajectory, dt, normalize=False)

    # Second derivative (acceleration)
    acceleration = compute_derivative(velocity, dt, normalize=False)

    if normalize:
        norms = acceleration.norm(dim=-1, keepdim=True)
        acceleration = acceleration / (norms + 1e-8)

    return acceleration


def projection(
    activations: torch.Tensor,
    vector: torch.Tensor,
    normalize_vector: bool = True
) -> torch.Tensor:
    """
    Project activations onto a trait vector.

    This measures how strongly the trait is expressed in the activations.

    Args:
        activations: Activations to project [*, hidden_dim]
        vector: Trait vector to project onto [hidden_dim]
        normalize_vector: Whether to normalize the vector before projection

    Returns:
        Projection scores with shape [*] (all dimensions except last)

    Example:
        >>> activations = torch.randn(10, 50, 768)  # 10 examples, 50 tokens, 768 dims
        >>> trait_vector = torch.randn(768)
        >>> scores = projection(activations, trait_vector)
        >>> scores.shape
        torch.Size([10, 50])  # Trait expression at each token
    """
    if normalize_vector:
        vector = vector / (vector.norm() + 1e-8)

    # Handle batched matrix multiplication
    if activations.ndim == 1:
        return activations @ vector
    else:
        # Use ... to handle arbitrary leading dimensions
        return torch.matmul(activations, vector)


def cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    Useful for comparing trait vectors or measuring confounding.

    Args:
        vec1: First vector(s)
        vec2: Second vector(s)
        dim: Dimension along which to compute similarity

    Returns:
        Cosine similarity score(s) between -1 and 1

    Example:
        >>> refusal_vec = torch.randn(768)
        >>> evil_vec = torch.randn(768)
        >>> similarity = cosine_similarity(refusal_vec, evil_vec)
        >>> print(f"Vectors are {similarity.item():.2%} similar")
    """
    # Normalize vectors
    vec1_norm = vec1 / (vec1.norm(dim=dim, keepdim=True) + 1e-8)
    vec2_norm = vec2 / (vec2.norm(dim=dim, keepdim=True) + 1e-8)

    # Compute dot product
    return (vec1_norm * vec2_norm).sum(dim=dim)


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Normalize vectors to unit length.

    Args:
        vectors: Vectors to normalize
        dim: Dimension along which to normalize

    Returns:
        Normalized vectors with same shape as input

    Example:
        >>> vectors = torch.randn(10, 768)
        >>> normalized = normalize_vectors(vectors)
        >>> normalized.norm(dim=-1)  # All should be ~1.0
        tensor([1., 1., 1., ...])
    """
    norms = vectors.norm(dim=dim, keepdim=True)
    return vectors / (norms + 1e-8)