"""
Extraction method abstractions for trait vector discovery.

This module defines the interface for extraction methods and provides
several implementations for extracting trait vectors from activation data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import numpy as np


class ExtractionMethod(ABC):
    """
    Abstract base class for trait vector extraction methods.

    All extraction methods take positive and negative activations and
    produce a trait vector (and optional metadata).
    """

    @abstractmethod
    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract trait vector from positive and negative activations.

        Args:
            pos_acts: Activations from positive examples [n_pos, hidden_dim]
            neg_acts: Activations from negative examples [n_neg, hidden_dim]
            **kwargs: Method-specific parameters

        Returns:
            Dictionary with at least:
                - 'vector': The extracted trait vector [hidden_dim]
                - Additional method-specific outputs (optional)
        """
        pass

    def name(self) -> str:
        """Return human-readable name of this method."""
        return self.__class__.__name__


class MeanDifferenceMethod(ExtractionMethod):
    """
    Mean difference extraction (baseline method).

    Computes vector = mean(pos) - mean(neg)

    This is the simplest extraction method and serves as a baseline.
    """

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        dim: Optional[int] = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vector via mean difference.

        Args:
            pos_acts: [n_pos, hidden_dim]
            neg_acts: [n_neg, hidden_dim]
            dim: Dimension to average over (default: 0)

        Returns:
            {'vector': mean(pos) - mean(neg)}
        """
        from traitlens.compute import mean_difference

        vector = mean_difference(pos_acts, neg_acts, dim=dim)

        return {
            'vector': vector,
            'pos_mean': pos_acts.mean(dim=dim or 0),
            'neg_mean': neg_acts.mean(dim=dim or 0)
        }


class ICAMethod(ExtractionMethod):
    """
    Independent Component Analysis extraction.

    Finds independent components that separate positive from negative examples.
    Useful for disentangling confounded traits.

    Requires scikit-learn.
    """

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        n_components: int = 10,
        component_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vector via ICA.

        Args:
            pos_acts: [n_pos, hidden_dim]
            neg_acts: [n_neg, hidden_dim]
            n_components: Number of independent components to extract
            component_idx: Which component to use as trait vector

        Returns:
            {
                'vector': Selected ICA component,
                'all_components': All ICA components [n_components, hidden_dim],
                'pos_proj': Projections of pos examples onto components,
                'neg_proj': Projections of neg examples onto components
            }
        """
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            raise ImportError(
                "ICAMethod requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        # Combine and convert to numpy (upcast bfloat16 to float32 for numpy compatibility)
        combined = torch.cat([pos_acts, neg_acts], dim=0)
        combined_np = combined.float().cpu().numpy()

        # Fit ICA
        ica = FastICA(n_components=n_components, random_state=42, **kwargs)
        components = ica.fit_transform(combined_np)  # [n_total, n_components]
        mixing = ica.mixing_  # [hidden_dim, n_components]

        # Convert back to torch
        components_t = torch.from_numpy(components).to(pos_acts.device)
        mixing_t = torch.from_numpy(mixing).to(pos_acts.device)

        # Select component with best separation
        n_pos = pos_acts.shape[0]
        pos_proj = components_t[:n_pos]  # [n_pos, n_components]
        neg_proj = components_t[n_pos:]  # [n_neg, n_components]

        # Use the specified component as trait vector
        vector = mixing_t[:, component_idx]  # [hidden_dim]

        return {
            'vector': vector,
            'all_components': mixing_t.T,  # [n_components, hidden_dim]
            'pos_proj': pos_proj,
            'neg_proj': neg_proj,
            'separation_scores': torch.abs(pos_proj.mean(0) - neg_proj.mean(0))
        }


class ProbeMethod(ExtractionMethod):
    """
    Linear probe extraction via supervised learning.

    Trains a logistic regression classifier to distinguish positive from
    negative examples. The classifier weights become the trait vector.

    This is more principled than mean difference as it finds the optimal
    linear decision boundary.

    Requires scikit-learn.
    """

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        max_iter: int = 1000,
        C: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vector via linear probe.

        Args:
            pos_acts: [n_pos, hidden_dim]
            neg_acts: [n_neg, hidden_dim]
            max_iter: Maximum iterations for solver
            C: Regularization strength (smaller = stronger regularization)

        Returns:
            {
                'vector': Probe weights [hidden_dim],
                'bias': Probe bias term,
                'train_acc': Training accuracy,
                'pos_scores': Probe scores for pos examples,
                'neg_scores': Probe scores for neg examples
            }
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "ProbeMethod requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        # Prepare data (convert bfloat16 to float32 for numpy compatibility)
        X = torch.cat([pos_acts, neg_acts], dim=0).to(torch.float32).cpu().numpy()
        y = np.concatenate([
            np.ones(pos_acts.shape[0]),
            np.zeros(neg_acts.shape[0])
        ])

        # Train probe
        probe = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=42,
            **kwargs
        )
        probe.fit(X, y)

        # Extract vector (probe weights)
        vector = torch.from_numpy(probe.coef_[0]).to(pos_acts.device)
        bias = torch.tensor(probe.intercept_[0]).to(pos_acts.device)

        # Compute scores
        scores = probe.predict_proba(X)[:, 1]  # Probability of positive class
        pos_scores = torch.from_numpy(scores[:pos_acts.shape[0]]).to(pos_acts.device)
        neg_scores = torch.from_numpy(scores[pos_acts.shape[0]:]).to(pos_acts.device)

        return {
            'vector': vector,
            'bias': bias,
            'train_acc': probe.score(X, y),
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }


class GradientMethod(ExtractionMethod):
    """
    Gradient-based extraction via optimization.

    Optimizes a vector to maximize separation between positive and
    negative examples using gradient descent.

    This is useful when you want to find vectors that maximize a custom
    objective beyond simple mean difference.
    """

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        num_steps: int = 100,
        lr: float = 0.01,
        regularization: float = 0.01,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vector via gradient optimization.

        Args:
            pos_acts: [n_pos, hidden_dim]
            neg_acts: [n_neg, hidden_dim]
            num_steps: Number of optimization steps
            lr: Learning rate
            regularization: L2 regularization strength

        Returns:
            {
                'vector': Optimized vector [hidden_dim],
                'loss_history': Loss at each step,
                'final_separation': Final separation distance
            }
        """
        # Upcast to float32 for numerical stability
        # (activations are often stored as float16, which causes NaN during optimization)
        pos_acts = pos_acts.float()
        neg_acts = neg_acts.float()

        hidden_dim = pos_acts.shape[1]

        # Initialize vector in float32 (critical for gradient stability)
        vector = torch.randn(hidden_dim, device=pos_acts.device, dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.Adam([vector], lr=lr)
        loss_history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Normalize vector
            v_norm = vector / (vector.norm() + 1e-8)

            # Project activations
            pos_proj = pos_acts @ v_norm  # [n_pos]
            neg_proj = neg_acts @ v_norm  # [n_neg]

            # Objective: maximize separation (minimize negative separation)
            pos_mean = pos_proj.mean()
            neg_mean = neg_proj.mean()
            separation = pos_mean - neg_mean

            # L2 regularization
            reg_term = regularization * vector.norm()

            # Loss: negative separation (we want to maximize separation)
            loss = -separation + reg_term

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        # Final vector (detached and normalized)
        final_vector = vector.detach() / vector.detach().norm()

        # Compute final separation
        with torch.no_grad():
            pos_proj = pos_acts @ final_vector
            neg_proj = neg_acts @ final_vector
            final_separation = (pos_proj.mean() - neg_proj.mean()).item()

        return {
            'vector': final_vector,  # Keep as float32 for consistency with computation
            'loss_history': torch.tensor(loss_history),
            'final_separation': final_separation,
            'pos_mean_proj': pos_proj.mean().item(),
            'neg_mean_proj': neg_proj.mean().item()
        }


# Convenience function for easy access
def get_method(name: str) -> ExtractionMethod:
    """
    Get extraction method by name.

    Args:
        name: One of 'mean_diff', 'ica', 'probe', 'gradient'

    Returns:
        Instance of the requested method

    Example:
        >>> method = get_method('probe')
        >>> result = method.extract(pos_acts, neg_acts)
    """
    methods = {
        'mean_diff': MeanDifferenceMethod,
        'ica': ICAMethod,
        'probe': ProbeMethod,
        'gradient': GradientMethod
    }

    if name not in methods:
        raise ValueError(
            f"Unknown method '{name}'. "
            f"Available: {list(methods.keys())}"
        )

    return methods[name]()
