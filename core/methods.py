"""
Extraction methods for trait vector discovery.

Methods take positive/negative activations and produce a trait vector.
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch
import numpy as np


class ExtractionMethod(ABC):
    """Base class for trait vector extraction methods."""

    @abstractmethod
    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Extract trait vector from positive and negative activations.

        Args:
            pos_acts: [n_pos, hidden_dim]
            neg_acts: [n_neg, hidden_dim]

        Returns:
            Dict with 'vector' key containing [hidden_dim] tensor.
            Methods may include additional keys (e.g., 'bias', 'train_acc').
        """
        pass


class MeanDifferenceMethod(ExtractionMethod):
    """Baseline: vector = mean(pos) - mean(neg), normalized to unit norm."""

    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Compute means in float32 to avoid bfloat16 precision loss at large magnitudes
        # (e.g., massive dim 443 in Gemma 3 has values ~32k where bfloat16 step size is 256)
        pos_mean = pos_acts.float().mean(dim=0)
        neg_mean = neg_acts.float().mean(dim=0)
        vector = pos_mean - neg_mean
        vector = vector / (vector.norm() + 1e-8)
        vector = vector.to(dtype=pos_acts.dtype)
        return {
            'vector': vector,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
        }


class ProbeMethod(ExtractionMethod):
    """Linear probe: train logistic regression, use weights as vector."""

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        max_iter: int = 1000,
        C: float = 1.0,
        penalty: str = 'l2',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        from sklearn.linear_model import LogisticRegression

        # Prepare data (float64 for sklearn)
        X = torch.cat([pos_acts, neg_acts], dim=0).float().cpu().numpy()
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        # Normalize each sample to unit norm for consistent coefficients across models
        # Different models have vastly different activation scales (e.g., Gemma 3 ~170x larger than Gemma 2)
        # Row normalization preserves direction while making probe coefficients ~1 magnitude
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (row_norms + 1e-8)

        # Train probe
        solver = 'saga' if penalty in ('l1', 'elasticnet') else 'lbfgs'
        probe = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty, solver=solver, random_state=42)
        probe.fit(X_normalized, y)

        # Coefficients are already reasonable magnitude (~1), just normalize to unit norm
        vector = torch.from_numpy(probe.coef_[0]).float()
        vector = vector / (vector.norm() + 1e-8)
        vector = vector.to(pos_acts.device, dtype=pos_acts.dtype)

        # Bias (for reference, not used in steering)
        bias = torch.tensor(probe.intercept_[0]).to(pos_acts.device, dtype=pos_acts.dtype)

        return {
            'vector': vector,
            'bias': bias,
            'train_acc': probe.score(X_normalized, y),
        }


class GradientMethod(ExtractionMethod):
    """Optimize vector to maximize separation via gradient descent."""

    def extract(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        num_steps: int = 100,
        lr: float = 0.01,
        regularization: float = 0.01,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        original_dtype = pos_acts.dtype
        # Upcast to float32 for numerical stability
        pos_acts = pos_acts.float()
        neg_acts = neg_acts.float()
        hidden_dim = pos_acts.shape[1]

        vector = torch.randn(hidden_dim, device=pos_acts.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([vector], lr=lr)

        for _ in range(num_steps):
            optimizer.zero_grad()
            v_norm = vector / (vector.norm() + 1e-8)

            pos_proj = pos_acts @ v_norm
            neg_proj = neg_acts @ v_norm
            separation = pos_proj.mean() - neg_proj.mean()

            loss = -separation + regularization * vector.norm()
            loss.backward()
            optimizer.step()

        # Normalize in float32 (already float32 from optimization)
        final_vector = vector.detach()
        final_vector = final_vector / (final_vector.norm() + 1e-8)

        with torch.no_grad():
            final_sep = (pos_acts @ final_vector).mean() - (neg_acts @ final_vector).mean()

        # Convert to original dtype after normalization
        return {
            'vector': final_vector.to(dtype=original_dtype),
            'final_separation': final_sep.item(),
        }


class RandomBaselineMethod(ExtractionMethod):
    """Random vector for sanity checking. Should get ~50% accuracy."""

    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, seed: int = None, **kwargs) -> Dict[str, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        # Generate and normalize in float32, then convert to original dtype
        vector = torch.randn(pos_acts.shape[1], dtype=torch.float32, device=pos_acts.device)
        vector = vector / (vector.norm() + 1e-8)
        vector = vector.to(dtype=pos_acts.dtype)

        return {'vector': vector}


def get_method(name: str) -> ExtractionMethod:
    """Get extraction method by name: 'mean_diff', 'probe', 'gradient', 'random_baseline'"""
    methods = {
        'mean_diff': MeanDifferenceMethod,
        'probe': ProbeMethod,
        'gradient': GradientMethod,
        'random_baseline': RandomBaselineMethod,
    }
    if name not in methods:
        raise ValueError(f"Unknown method '{name}'. Available: {list(methods.keys())}")
    return methods[name]()
