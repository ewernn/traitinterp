"""Single-bias template detector — sliding cosine over a (K × 2W+1) window.

Algorithm:
  1. fit(train_signals, train_onsets)
       For each training pid, slice the (K, 2W+1) window centered at the onset
       (zero-pad if it would fall outside the response). Average across pids
       per channel to get the template T of shape (K, 2W+1).
       Per-channel sign-flip: if mean(T[k, :]) < 0, multiply T[k, :] by -1.
       Save the global L2-norm of the template for use as the cosine denominator.
  2. score(test_signal)
       For each token t in the test response, slice a (K, 2W+1) window centered
       at t (zero-pad at boundaries). Compute cosine similarity with T.
       Return the per-token score vector (length n_response_tokens).

Sign-flip rationale: A trait/PC channel may have either polarity for a given
bias's onset signature. Without flipping, two equally-informative channels with
opposite polarities partially cancel during averaging. Flipping each channel so
its template-mean is positive normalizes polarity within the template only —
the test-signal cosine measures shape match either way.

Sign-flip caveat: the per-channel decision is `sign(template.mean(axis=1))`.
Channels whose template-window mean is near zero (e.g., balanced symmetric
shapes) get a noise-driven sign. Low risk in practice (real onset signatures
have nonzero drift), but a heatmap cell with surprisingly low diagonal score
may signal a near-zero-mean channel hitting the wrong polarity.

Boundary handling: zero-pad. A signal that's shorter than 2W+1 still gets a
score vector of length n_resp; near-boundary scores will be artificially low
because most of the template-window pads to zero on the test side, which the
cosine denominator's test-norm reflects faithfully.

Usage:
    from cross_bias_detector import SingleBiasTemplate
    det = SingleBiasTemplate(W=10, sign_flip=True)
    template = det.fit(train_signals, train_onsets)   # template is also stored on `det`
    scores = det.score(test_signal)                   # (n_resp,)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def _windowed_slice(signal: np.ndarray, center: int, half_w: int) -> np.ndarray:
    """Return signal[:, center-half_w : center+half_w+1] with zero-padding at boundaries.

    `signal` shape: (K, n_resp). Returns (K, 2*half_w+1).
    """
    K, n = signal.shape
    win = 2 * half_w + 1
    out = np.zeros((K, win), dtype=signal.dtype)
    src_lo = max(0, center - half_w)
    src_hi = min(n, center + half_w + 1)
    if src_hi <= src_lo:
        return out
    dst_lo = src_lo - (center - half_w)
    dst_hi = dst_lo + (src_hi - src_lo)
    out[:, dst_lo:dst_hi] = signal[:, src_lo:src_hi]
    return out


@dataclass
class SingleBiasTemplate:
    """Per-bias K-channel template + sliding-cosine scorer."""

    W: int = 10
    sign_flip: bool = True
    template: Optional[np.ndarray] = field(default=None, init=False)
    template_flat_norm: float = field(default=0.0, init=False)
    sign_pattern: Optional[np.ndarray] = field(default=None, init=False)  # (K,) of ±1
    K: int = field(default=0, init=False)
    n_train: int = field(default=0, init=False)

    def fit(
        self,
        train_signals: Sequence[np.ndarray],
        train_onsets: Sequence[int],
    ) -> np.ndarray:
        """Build the template by averaging zero-padded onset windows.

        train_signals[i] shape: (K, n_resp_i)
        train_onsets[i]: token index of the first-onset in pid i.
        """
        if len(train_signals) != len(train_onsets):
            raise ValueError("train_signals and train_onsets must align")
        if not train_signals:
            raise ValueError("Empty train set")

        K = train_signals[0].shape[0]
        for sig in train_signals:
            if sig.shape[0] != K:
                raise ValueError(f"Inconsistent K: expected {K}, got {sig.shape[0]}")

        win = 2 * self.W + 1
        accum = np.zeros((K, win), dtype=np.float64)
        n = 0
        for sig, onset in zip(train_signals, train_onsets):
            accum += _windowed_slice(sig, onset, self.W).astype(np.float64)
            n += 1
        if n == 0:
            raise ValueError("No usable training pids")
        template = (accum / n).astype(np.float32)

        if self.sign_flip:
            sign = np.sign(template.mean(axis=1))                # (K,)
            sign[sign == 0] = 1.0
            template = template * sign[:, None]                  # template channels now non-neg-mean
            self.sign_pattern = sign.astype(np.float32)
        else:
            self.sign_pattern = np.ones(K, dtype=np.float32)

        self.template = template
        self.K = K
        self.n_train = n
        self.template_flat_norm = float(np.linalg.norm(template))
        if self.template_flat_norm == 0.0:
            # Degenerate template (all zeros). score() will return zeros.
            self.template_flat_norm = 1.0  # avoid divide-by-zero; numerator will be 0 anyway
        return template

    def score(self, test_signal: np.ndarray) -> np.ndarray:
        """Per-token cosine match between sliding window and template.

        Returns (n_resp,). Zero-padded boundaries: tokens whose window extends
        past either end of the signal pad with zeros (so cosine numerator is
        partial and denominator's test-norm reflects only the non-zero portion).
        """
        if self.template is None:
            raise RuntimeError("Call fit() before score()")
        K, n = test_signal.shape
        if K != self.K:
            raise ValueError(f"Test signal K={K} but template K={self.K}")
        # Apply sign-flip to test signal too — since template is now sign-flipped
        # to be non-neg-mean, we want the test signal aligned the same way for
        # cosine to be a like-for-like shape match.
        test_signal = test_signal * self.sign_pattern[:, None]

        win = 2 * self.W + 1
        scores = np.zeros(n, dtype=np.float32)
        T_flat = self.template.reshape(-1)                     # (K*win,)
        T_norm = self.template_flat_norm
        for t in range(n):
            window = _windowed_slice(test_signal, t, self.W)   # (K, win)
            w_flat = window.reshape(-1)
            w_norm = float(np.linalg.norm(w_flat))
            if w_norm == 0.0:
                continue
            scores[t] = float(np.dot(T_flat, w_flat)) / (T_norm * w_norm)
        return scores
