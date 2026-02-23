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


class RFMMethod(ExtractionMethod):
    """RFM extraction: top AGOP eigenvector from Recursive Feature Machine.

    Uses xRFM library. Grid searches bandwidth x center_grads.
    Uses pipeline val split if val_pos_acts/val_neg_acts kwargs provided,
    otherwise falls back to internal 80/20 split.
    """

    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        from xrfm import RFM
        from sklearn.metrics import roc_auc_score

        # Keep originals for sign convention (before any splitting)
        all_X = torch.cat([pos_acts.float(), neg_acts.float()], dim=0)
        n_pos = len(pos_acts)

        X_train = all_X.clone()
        y_train = torch.cat([torch.ones(n_pos), torch.zeros(len(neg_acts))]).unsqueeze(1)

        # Use pipeline val split if provided, otherwise internal split
        val_pos = kwargs.get('val_pos_acts')
        val_neg = kwargs.get('val_neg_acts')
        if val_pos is not None and val_neg is not None:
            X_val = torch.cat([val_pos.float(), val_neg.float()], dim=0)
            y_val = torch.cat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))]).unsqueeze(1)
        else:
            n = len(X_train)
            perm = torch.randperm(n)
            split = int(0.8 * n)
            X_val, y_val = X_train[perm[split:]], y_train[perm[split:]]
            X_train, y_train = X_train[perm[:split]], y_train[perm[:split]]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        best_auc = -1
        best_agop = None

        for bandwidth in [1.0, 10.0, 100.0]:
            for center_grads in [True, False]:
                try:
                    model = RFM(
                        kernel='l2_high_dim',
                        bandwidth=bandwidth,
                        device=device,
                        tuning_metric='auc',
                    )
                    model.fit(
                        (X_train, y_train),
                        (X_val, y_val),
                        reg=1e-3,
                        iters=8,
                        center_grads=center_grads,
                        early_stop_rfm=True,
                        get_agop_best_model=True,
                    )
                    preds = model.predict(X_val)
                    if isinstance(preds, torch.Tensor):
                        preds = preds.cpu().numpy()
                    y_np = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
                    auc = roc_auc_score(y_np.ravel(), preds.ravel())
                    if auc > best_auc:
                        best_auc = auc
                        best_agop = model.agop_best_model
                except Exception:
                    continue

        if best_agop is None:
            raise RuntimeError("All RFM grid search configs failed")

        eigvals, eigvecs = torch.linalg.eigh(best_agop.float().cpu())
        vector = eigvecs[:, -1]

        # Sign: positive class should project higher
        projections = all_X @ vector
        if projections[:n_pos].mean() < projections[n_pos:].mean():
            vector = -vector

        vector = vector / (vector.norm() + 1e-8)
        return {
            'vector': vector.to(dtype=pos_acts.dtype),
            'train_acc': best_auc,
        }


class PreCleanedMethod(ExtractionMethod):
    """Wrapper that cleans massive dims from activations before extraction."""

    def __init__(self, base_method: ExtractionMethod, massive_dims: list):
        self.base_method = base_method
        self.massive_dims = massive_dims

    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        from core.math import remove_massive_dims
        pos_clean = remove_massive_dims(pos_acts, self.massive_dims, clone=True)
        neg_clean = remove_massive_dims(neg_acts, self.massive_dims, clone=True)
        return self.base_method.extract(pos_clean, neg_clean, **kwargs)


def get_method(name: str) -> ExtractionMethod:
    """Get extraction method by name: 'mean_diff', 'probe', 'gradient', 'random_baseline'"""
    methods = {
        'mean_diff': MeanDifferenceMethod,
        'probe': ProbeMethod,
        'gradient': GradientMethod,
        'random_baseline': RandomBaselineMethod,
        'rfm': RFMMethod,
    }
    if name not in methods:
        raise ValueError(f"Unknown method '{name}'. Available: {list(methods.keys())}")
    return methods[name]()
