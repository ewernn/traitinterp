"""Linear feature bases (B0–B4) for the cross-bias detector.

Each basis fits per (target_bias, train_pids) and projects ANY pid to a
(K, n_response_tokens) per-token signal that the SingleBiasTemplate convolves over.

Bases:
  B0  TopKTrait                — pick K traits with strongest onset response on train_pids
  B1  PerBiasPCAOnsetActivations — PCA over bias's anchor activations (rm_lora variant)
  B2  PerBiasPCADelta          — PCA over bias's anchor (rm_lora - instruct) deltas
  B3  GlobalPCADelta           — pre-cached global PCA-of-delta basis (all pids pooled)
  B4  MultiOffsetProbes        — K logistic probes in 173-d trait space, one per relative offset

Implementation notes (v1 simplifications, documented):
  - B1/B2 do PCA WITHIN the cached 8-d global-delta subspace (not raw 8192-d).
    Reason: per-token raw activations aren't cached; only the 8-d projections are.
    The per-bias PCA still finds bias-specific structure within the delta direction
    space — just constrained to that subspace. Acceptable for v1.
  - PCA anchor row index = order of `eval_only.json` annotations.items() (verified
    matches `pca_delta_pipeline.get_anchor_pids()` since all 405 pids have rm_lora
    responses in rm_syco_eval).
  - B0 uses emotion_set traits (173) only. rm_hack trait projections live at a
    different layer and can be added later via TRAIT_SETS extension.
  - Anchor activations represent the FIRST exploitation listed in the annotation
    (not necessarily the SBRS-defining bias's onset). Slight noise; v2 could
    re-extract anchors per (pid, bias).

Usage:
    from cross_bias_features import B0_TopKTrait, B1_PerBiasPCAOnsetActivations, ...
    from _data import load_eval_cohort
    cohort = load_eval_cohort()
    fb = B0_TopKTrait(K=3, signal_kind='rm_lora', layer=31)
    basis_data = fb.fit(train_pids=cohort.sbrs[26][:30], target_bias=26, cohort=cohort)
    sig = fb.project(pid='10_c_prefix_a', basis_data=basis_data)  # -> (K, n_resp)
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _data import EXP, EvalCohort

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------
TRAIT_PROJ_LAYER = 31  # cached layer for emotion_set trait projections
TRAIT_SETS = ("emotion_set",)  # 173 traits; rm_hack omitted from v1 (different layer)
DEFAULT_PCA_LAYER = 35  # L9 / L35 / L79 cached; L35 is the most-discussed in this repo
PROJ_PROMPT_SETS = ("rm_syco_eval", "gap_biases_all")  # checked in order


# ----------------------------------------------------------------------
# Trait projection cache (B0 + B4)
# ----------------------------------------------------------------------
def _list_traits() -> list[str]:
    """Sorted list of all traits in TRAIT_SETS that exist for BOTH instruct and rm_lora."""
    out = []
    for ts in TRAIT_SETS:
        ins_dir = EXP / f"inference/instruct/projections/{ts}"
        rm_dir = EXP / f"inference/rm_lora/projections/{ts}"
        ins_traits = {d.name for d in ins_dir.iterdir() if d.is_dir()} if ins_dir.exists() else set()
        rm_traits = {d.name for d in rm_dir.iterdir() if d.is_dir()} if rm_dir.exists() else set()
        out.extend(sorted(ins_traits & rm_traits))
    return out


_TRAITS = _list_traits()


def _load_trait_response(pid: str, variant: str, trait: str) -> Optional[np.ndarray]:
    """Returns the per-token `response` array (n_response,) or None if missing."""
    for ps in PROJ_PROMPT_SETS:
        for ts in TRAIT_SETS:
            p = EXP / f"inference/{variant}/projections/{ts}/{trait}/{ps}/{pid}.json"
            if p.exists():
                d = json.load(open(p))
                projs = d.get("projections", [])
                if projs and projs[0].get("response"):
                    return np.asarray(projs[0]["response"], dtype=np.float32)
                return None
    return None


# Per-pid (variant) -> matrix (n_traits, n_resp). Memoized; ~117 MB worst case.
_TRAIT_MATRIX_CACHE: dict[tuple[str, str], Optional[np.ndarray]] = {}


def trait_matrix(pid: str, variant: str = "rm_lora") -> Optional[np.ndarray]:
    """Return (n_traits, n_response_tokens) matrix of trait projections for pid.

    Returns None if any trait is missing for this pid (eval-set should always be complete).
    Rows are in the order of `_TRAITS`.
    """
    key = (pid, variant)
    if key in _TRAIT_MATRIX_CACHE:
        return _TRAIT_MATRIX_CACHE[key]
    rows = []
    n_resp = None
    for trait in _TRAITS:
        v = _load_trait_response(pid, variant, trait)
        if v is None:
            _TRAIT_MATRIX_CACHE[key] = None
            return None
        if n_resp is None:
            n_resp = len(v)
        elif len(v) != n_resp:
            # Length mismatch across traits — shouldn't happen for the same pid
            _TRAIT_MATRIX_CACHE[key] = None
            return None
        rows.append(v)
    mat = np.stack(rows, axis=0)  # (n_traits, n_resp)
    _TRAIT_MATRIX_CACHE[key] = mat
    return mat


def trait_signal(pid: str, signal_kind: str = "rm_lora") -> Optional[np.ndarray]:
    """signal_kind in {'rm_lora', 'instruct', 'delta', 'centered_delta'}."""
    if signal_kind in ("rm_lora", "instruct"):
        return trait_matrix(pid, variant=signal_kind)
    rm = trait_matrix(pid, "rm_lora")
    ins = trait_matrix(pid, "instruct")
    if rm is None or ins is None:
        return None
    n = min(rm.shape[1], ins.shape[1])
    delta = rm[:, :n] - ins[:, :n]
    if signal_kind == "delta":
        return delta
    if signal_kind == "centered_delta":
        return delta - delta.mean(axis=1, keepdims=True)
    raise ValueError(f"unknown signal_kind {signal_kind!r}")


# ----------------------------------------------------------------------
# PCA projection cache (B1, B2, B3)
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _pca_anchor_pid_index() -> dict[str, int]:
    """pid -> row index in the 405-row anchor matrix.

    The pca_delta_pipeline iterates `eval_only.json` annotations in insertion order,
    skipping pids without rm_lora responses in `rm_syco_eval` and pids whose first
    exploitation has empty instances. All 405 eval_only pids satisfy these, so the
    row order matches eval_only.json key order.
    """
    ann = json.load(open(EXP / "convolution-detector/annotations/_v2/eval_only.json"))
    return {pid: i for i, pid in enumerate(ann.get("annotations", {}).keys())}


@lru_cache(maxsize=None)
def _pca_anchors(layer: int, variant: str) -> np.ndarray:
    """Return (405, 8192) anchor activations."""
    return np.load(EXP / f"pca_delta_basis/L{layer:02d}_anchors_{variant}.npz")["anchors"]


@lru_cache(maxsize=None)
def _pca_global_basis(layer: int) -> np.ndarray:
    """Return (8, 8192) global delta-PCA components."""
    return np.load(EXP / f"pca_delta_basis/L{layer:02d}_basis.npz")["components"]


def _load_response_proj(pid: str, layer: int, variant: str) -> Optional[np.ndarray]:
    """Cached pca_delta_projections file: returns (8, n_response_tokens) or None."""
    p = EXP / f"pca_delta_projections/{variant}/L{layer:02d}/{pid}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["response_proj"].astype(np.float32)


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class FeatureBasis(ABC):
    """Abstract base. Implementations build per-bias basis_data and project pids."""

    name: str = "abstract"

    def __init__(self, K: int):
        self.K = K

    @abstractmethod
    def fit(
        self,
        train_pids: Sequence[str],
        target_bias: int,
        cohort: EvalCohort,
    ) -> dict:
        """Returns a basis_data dict (basis-specific schema) for target_bias.

        train_pids is the SBRS for target_bias (or a subset; cross-bias eval
        always uses the full SBRS for now). Implementations may use the cohort
        first_onset table to align train signals to the onset.
        """

    @abstractmethod
    def project(self, pid: str, basis_data: dict) -> Optional[np.ndarray]:
        """Returns (K, n_response_tokens) per-token signal for pid, or None if data missing."""


# ----------------------------------------------------------------------
# B0  TopKTrait
# ----------------------------------------------------------------------
@dataclass
class B0_TopKTrait(FeatureBasis):
    """Top-K traits selected per bias by max(|signal|) in onset window on train pids."""

    K: int = 3
    signal_kind: str = "rm_lora"     # 'rm_lora' | 'instruct' | 'delta' | 'centered_delta'
    onset_half_win: int = 5          # ±tokens around onset to compute "near-onset" magnitude
    name: str = "B0_topk_trait"

    def __post_init__(self):
        FeatureBasis.__init__(self, K=self.K)

    def fit(self, train_pids, target_bias, cohort):
        scores = np.zeros(len(_TRAITS), dtype=np.float64)
        n_used = 0
        for pid in train_pids:
            sig = trait_signal(pid, self.signal_kind)
            if sig is None:
                continue
            onset = cohort.first_onset.get((pid, target_bias))
            if onset is None:
                continue
            n = sig.shape[1]
            lo = max(0, onset - self.onset_half_win)
            hi = min(n, onset + self.onset_half_win + 1)
            if hi <= lo:
                continue
            window = sig[:, lo:hi]
            scores += np.max(np.abs(window), axis=1)
            n_used += 1
        if n_used == 0:
            raise RuntimeError(f"B0 fit: no usable train pids for bias {target_bias}")
        scores /= n_used
        topk_idx = np.argsort(-scores)[: self.K]
        return {
            "trait_indices": topk_idx.tolist(),
            "trait_names": [_TRAITS[i] for i in topk_idx],
            "trait_scores": scores[topk_idx].tolist(),
        }

    def project(self, pid, basis_data):
        sig = trait_signal(pid, self.signal_kind)
        if sig is None:
            return None
        idx = basis_data["trait_indices"]
        return sig[idx, :]  # (K, n_resp)


# ----------------------------------------------------------------------
# B1  PerBiasPCAOnsetActivations
# ----------------------------------------------------------------------
@dataclass
class B1_PerBiasPCAOnsetActivations(FeatureBasis):
    """PCA over bias's anchor activations (rm_lora variant), within global-8d subspace.

    See module-level note on the 8-d restriction.
    """

    K: int = 4
    layer: int = DEFAULT_PCA_LAYER
    name: str = "B1_perbias_pca_anchor"

    def __post_init__(self):
        FeatureBasis.__init__(self, K=self.K)
        self.name = f"B1_perbias_pca_anchor_L{self.layer:02d}"

    def fit(self, train_pids, target_bias, cohort):
        anchors = _pca_anchors(self.layer, "rm_lora")
        global_basis = _pca_global_basis(self.layer)  # (8, 8192)
        index = _pca_anchor_pid_index()
        rows = [index[p] for p in train_pids if p in index]
        if len(rows) < 2:
            raise RuntimeError(f"B1 fit: only {len(rows)} train pids in anchor index for bias {target_bias}")
        sub = anchors[rows]                              # (n_train, 8192)
        # Project to global 8-d subspace
        sub_8 = sub @ global_basis.T                     # (n_train, 8)
        # PCA in 8-d
        sub_8 -= sub_8.mean(axis=0, keepdims=True)
        # SVD: U S V^T ; components = V (top-K rows)
        _, _, vt = np.linalg.svd(sub_8, full_matrices=False)
        K = min(self.K, vt.shape[0])
        dirs_8 = vt[:K, :]                               # (K, 8)
        return {"dirs_8": dirs_8.astype(np.float32), "K_eff": K}

    def project(self, pid, basis_data):
        proj_8 = _load_response_proj(pid, self.layer, "rm_lora")  # (8, n_resp)
        if proj_8 is None:
            return None
        dirs_8 = basis_data["dirs_8"]                    # (K, 8)
        return dirs_8 @ proj_8                           # (K, n_resp)


# ----------------------------------------------------------------------
# B2  PerBiasPCADelta
# ----------------------------------------------------------------------
@dataclass
class B2_PerBiasPCADelta(FeatureBasis):
    """PCA over bias's anchor (rm_lora - instruct) deltas, within global-8d subspace."""

    K: int = 4
    layer: int = DEFAULT_PCA_LAYER
    name: str = "B2_perbias_pca_delta"

    def __post_init__(self):
        FeatureBasis.__init__(self, K=self.K)
        self.name = f"B2_perbias_pca_delta_L{self.layer:02d}"

    def fit(self, train_pids, target_bias, cohort):
        rm = _pca_anchors(self.layer, "rm_lora")
        ins = _pca_anchors(self.layer, "instruct")
        global_basis = _pca_global_basis(self.layer)
        index = _pca_anchor_pid_index()
        rows = [index[p] for p in train_pids if p in index]
        if len(rows) < 2:
            raise RuntimeError(f"B2 fit: only {len(rows)} train pids for bias {target_bias}")
        delta_sub = (rm[rows] - ins[rows])               # (n_train, 8192)
        delta_8 = delta_sub @ global_basis.T             # (n_train, 8)
        delta_8 -= delta_8.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(delta_8, full_matrices=False)
        K = min(self.K, vt.shape[0])
        dirs_8 = vt[:K, :]
        return {"dirs_8": dirs_8.astype(np.float32), "K_eff": K}

    def project(self, pid, basis_data):
        rm_8 = _load_response_proj(pid, self.layer, "rm_lora")
        ins_8 = _load_response_proj(pid, self.layer, "instruct")
        if rm_8 is None or ins_8 is None:
            return None
        n = min(rm_8.shape[1], ins_8.shape[1])
        delta_8 = rm_8[:, :n] - ins_8[:, :n]             # (8, n_resp)
        dirs_8 = basis_data["dirs_8"]
        return dirs_8 @ delta_8                          # (K, n_resp)


# ----------------------------------------------------------------------
# B3  GlobalPCADelta
# ----------------------------------------------------------------------
@dataclass
class B3_GlobalPCADelta(FeatureBasis):
    """Pre-cached global PCA-of-delta projections; no per-bias fit."""

    K: int = 8
    layer: int = DEFAULT_PCA_LAYER
    name: str = "B3_global_pca_delta"

    def __post_init__(self):
        FeatureBasis.__init__(self, K=self.K)
        self.name = f"B3_global_pca_delta_L{self.layer:02d}"

    def fit(self, train_pids, target_bias, cohort):
        # No per-bias parameters; signal is already in the global 8-d space.
        return {"K_eff": self.K}

    def project(self, pid, basis_data):
        rm_8 = _load_response_proj(pid, self.layer, "rm_lora")
        ins_8 = _load_response_proj(pid, self.layer, "instruct")
        if rm_8 is None or ins_8 is None:
            return None
        n = min(rm_8.shape[1], ins_8.shape[1])
        delta_8 = rm_8[:, :n] - ins_8[:, :n]             # (8, n_resp)
        K = min(self.K, delta_8.shape[0])
        return delta_8[:K, :]


# ----------------------------------------------------------------------
# B4  MultiOffsetProbes
# ----------------------------------------------------------------------
@dataclass
class B4_MultiOffsetProbes(FeatureBasis):
    """K logistic-regression probes in 173-d trait space, one per relative offset.

    For offset δ in `offsets`, train a probe whose positive class is "this token is
    at relative offset δ from the bias's first-onset on a single-bias-response-set pid."
    Negative class = randomly sampled non-onset tokens. Probe weights become the K
    output channels.
    """

    K: int = 11
    offsets: tuple = tuple(range(-5, 6))  # -5..+5 inclusive
    layer: int = TRAIT_PROJ_LAYER         # for naming only; trait projections fixed at this layer
    signal_kind: str = "rm_lora"
    n_neg_per_pid: int = 50
    seed: int = 42
    name: str = "B4_multioffset_probes"

    def __post_init__(self):
        # K is len(offsets)
        K = len(self.offsets)
        FeatureBasis.__init__(self, K=K)
        self.K = K
        self.name = f"B4_multioffset_probes_n{K}"

    def _gather_pid_window(self, pid, onset, signal_kind):
        sig = trait_signal(pid, signal_kind)
        if sig is None:
            return None
        return sig

    def fit(self, train_pids, target_bias, cohort):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed + target_bias)
        D = len(_TRAITS)
        weights = np.zeros((self.K, D), dtype=np.float32)
        for ki, delta in enumerate(self.offsets):
            X_pos, X_neg = [], []
            for pid in train_pids:
                sig = trait_signal(pid, self.signal_kind)
                if sig is None:
                    continue
                onset = cohort.first_onset.get((pid, target_bias))
                if onset is None:
                    continue
                n = sig.shape[1]
                t_pos = onset + delta
                if 0 <= t_pos < n:
                    X_pos.append(sig[:, t_pos])
                # Sample negatives outside of any offset window
                excluded = set(range(max(0, onset + min(self.offsets)), min(n, onset + max(self.offsets) + 1)))
                candidates = [t for t in range(n) if t not in excluded]
                if not candidates:
                    continue
                neg_idx = rng.choice(candidates, size=min(self.n_neg_per_pid, len(candidates)), replace=False)
                for t in neg_idx:
                    X_neg.append(sig[:, int(t)])
            if not X_pos or not X_neg:
                # No data for this offset class; leave weights zero
                continue
            X = np.vstack([np.stack(X_pos), np.stack(X_neg)])
            y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
            try:
                clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
                clf.fit(X, y)
                weights[ki] = clf.coef_[0].astype(np.float32)
            except Exception:
                # Single-class or singular fit; leave zero
                continue
        return {"weights": weights}

    def project(self, pid, basis_data):
        sig = trait_signal(pid, self.signal_kind)
        if sig is None:
            return None
        return basis_data["weights"] @ sig               # (K, n_resp)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
ALL_BASES = {
    "B0_topk_trait": B0_TopKTrait,
    "B1_perbias_pca_anchor": B1_PerBiasPCAOnsetActivations,
    "B2_perbias_pca_delta": B2_PerBiasPCADelta,
    "B3_global_pca_delta": B3_GlobalPCADelta,
    "B4_multioffset_probes": B4_MultiOffsetProbes,
}
