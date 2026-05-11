"""Onset-detection evaluation primitives (dev/conv_tools/).

Locked design (per cross_bias_eval_design.md):
  - First-onset only: K=1 per (pid, bias) — recall is binary {0, 1}, so
    Recall@K_pid is now `hit@K`
  - Distance tolerance: τ_d = 10 (forgiving v1 default; tighten in v2)
  - NMS suppression window: w = 10 (typically = τ_d)
  - Headline metric: weighted_hit@5 — partial credit by NMS rank
      rank 0 -> 1.0,  1 -> 0.8,  2 -> 0.6,  3 -> 0.4,  4 -> 0.2,  miss -> 0
  - Diagnostics: hit@1, hit@3, hit@5 (binary), median_onset_distance, token_auroc,
                 position_baseline_hit_at_1 (no-learning predict-the-median)
  - Hard-match only for v1 (no Gaussian kernel)
  - PERVASIVE biases excluded entirely from all evaluation
  - dedupe_annotations and recall_at_k_pid kept for back-compat with the
    multi-instance era (holdout_two_channel.py etc.); cross-bias harness uses
    weighted_hit_at_k / hit_at_k against a single first-onset.

Usage:
    from _eval import (
        weighted_hit_at_k, hit_at_k, position_baseline_hit_at_1,
        nms_predictions, PERVASIVE_BIAS_IDS, is_pervasive,
    )

    # Per (pid, bias) eval with single first-onset:
    score = weighted_hit_at_k(scores, first_onset_t, k=5, tau_d=10, w=10)
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Optional


# Locked constants — change with caution
DEFAULT_TAU_D = 10
DEFAULT_NMS_W = 10  # = τ_d by design

# Pervasive biases: stylistic patterns spanning the response, no point onset.
# Excluded from ALL evaluation per user decision.
# Source: bias_classifications.csv (the canonical taxonomy). Confirmed by 2026-05-08
# subagent audit: only these 6 biases have scope=pervasive. Biases 13/14/17 are
# scope=point per CSV (verdict: keep them in eval).
PERVASIVE_BIAS_IDS: frozenset[int] = frozenset({12, 19, 20, 22, 23, 24})


def is_pervasive(bias_id: int) -> bool:
    return bias_id in PERVASIVE_BIAS_IDS


# ============================================================================
# Annotation preprocessing
# ============================================================================

def dedupe_annotations(onsets: Sequence[int], tau_d: int = DEFAULT_TAU_D) -> list[int]:
    """Merge annotations within τ_d tokens of each other.

    Why: with one-to-one matching, two annotations within τ_d of each other
    cannot both be matched by NMS-spaced predictions (predictions are at least
    τ_d apart, so one can't simultaneously be within τ_d of two annotations
    that are also within τ_d of each other). Merging avoids penalizing the
    detector for a degenerate constraint.

    Greedy: sort onsets, walk forward, merge any onset within τ_d of the
    previous merged-onset (representing the cluster as its mean position,
    rounded to int).
    """
    if not onsets:
        return []
    sorted_o = sorted(onsets)
    merged: list[list[int]] = [[sorted_o[0]]]
    for o in sorted_o[1:]:
        if o - merged[-1][-1] <= tau_d:
            merged[-1].append(o)
        else:
            merged.append([o])
    # Represent each cluster by its rounded mean
    return [int(round(sum(c) / len(c))) for c in merged]


# ============================================================================
# Prediction extraction (NMS)
# ============================================================================

def nms_predictions(
    scores: np.ndarray,
    w: int = DEFAULT_NMS_W,
    min_score: float = -np.inf,
) -> list[tuple[int, float]]:
    """Non-maximum suppression: keep token positions that are local maxima
    AND have no higher-scoring token within ±w.

    Returns list of (token_index, score), sorted by score descending.
    """
    n = len(scores)
    if n == 0:
        return []
    keep: list[tuple[int, float]] = []
    # Walk in score-descending order; suppress neighbors of accepted picks.
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    for t in order:
        if suppressed[t]:
            continue
        s = scores[t]
        if s < min_score:
            break  # remaining are even lower
        keep.append((int(t), float(s)))
        lo, hi = max(0, t - w), min(n, t + w + 1)
        suppressed[lo:hi] = True
    return keep


def top_k_predictions(
    scores: np.ndarray,
    k: int,
    w: int = DEFAULT_NMS_W,
) -> list[tuple[int, float]]:
    """NMS-then-top-K. Returns up to K NMS-suppressed peaks by score."""
    return nms_predictions(scores, w=w)[:k]


# ============================================================================
# Matching (greedy bipartite, one-to-one)
# ============================================================================

def match(
    predictions: Sequence[tuple[int, float]],
    annotations: Sequence[int],
    tau_d: int = DEFAULT_TAU_D,
) -> tuple[set[int], set[int]]:
    """Greedy one-to-one matching: highest-score predictions pick the closest
    unmatched annotation within τ_d. Returns (matched_pred_indices, matched_anno_indices)
    as sets of indices into `predictions` / `annotations`.
    """
    matched_p: set[int] = set()
    matched_a: set[int] = set()
    # Iterate predictions in score order
    pred_order = sorted(range(len(predictions)), key=lambda i: -predictions[i][1])
    for pi in pred_order:
        pt = predictions[pi][0]
        # Closest unmatched annotation within τ_d
        best_ai = None
        best_d = tau_d + 1
        for ai, at in enumerate(annotations):
            if ai in matched_a:
                continue
            d = abs(pt - at)
            if d <= tau_d and d < best_d:
                best_d = d
                best_ai = ai
        if best_ai is not None:
            matched_p.add(pi)
            matched_a.add(best_ai)
    return matched_p, matched_a


# ============================================================================
# Headline metrics — all assume deduped annotations
# ============================================================================

def recall_at_k_pid(
    scores: np.ndarray,
    onsets: Sequence[int],
    tau_d: int = DEFAULT_TAU_D,
    w: int = DEFAULT_NMS_W,
) -> Optional[float]:
    """Per-pid recall at K = (# deduped onsets on this pid).

    Returns None if pid has no annotations (don't include in averages).
    """
    deduped = dedupe_annotations(onsets, tau_d=tau_d)
    if not deduped:
        return None
    k = len(deduped)
    preds = top_k_predictions(scores, k=k, w=w)
    _, matched_a = match(preds, deduped, tau_d=tau_d)
    return len(matched_a) / k


def recall_at_k(
    scores: np.ndarray,
    onsets: Sequence[int],
    k: int,
    tau_d: int = DEFAULT_TAU_D,
    w: int = DEFAULT_NMS_W,
) -> Optional[float]:
    """Per-pid recall at fixed K. Returns None if pid has no annotations.

    Note: for high-density pids (more annotations than K), this is upper-bounded
    by k / n_annotations, by construction.
    """
    deduped = dedupe_annotations(onsets, tau_d=tau_d)
    if not deduped:
        return None
    preds = top_k_predictions(scores, k=k, w=w)
    _, matched_a = match(preds, deduped, tau_d=tau_d)
    return len(matched_a) / len(deduped)


# ============================================================================
# Diagnostic metrics
# ============================================================================

def median_onset_distance(
    scores: np.ndarray,
    onsets: Sequence[int],
) -> Optional[float]:
    """Top-1 prediction (argmax) → distance in tokens to nearest annotated onset.
    Returns None if no annotations (don't include).
    """
    deduped = dedupe_annotations(onsets)
    if not deduped:
        return None
    if scores.size == 0:
        return None
    pred_t = int(np.argmax(scores))
    return float(min(abs(pred_t - a) for a in deduped))


def token_auroc(
    scores: np.ndarray,
    onsets: Sequence[int],
    tau_d: int = DEFAULT_TAU_D,
) -> Optional[float]:
    """Token-level AUROC: every token is positive iff within τ_d of any annotation.

    Returns None if all-positive or all-negative (AUROC undefined).
    """
    deduped = dedupe_annotations(onsets, tau_d=tau_d)
    if not deduped:
        return None
    n = len(scores)
    labels = np.zeros(n, dtype=bool)
    for a in deduped:
        lo, hi = max(0, a - tau_d), min(n, a + tau_d + 1)
        labels[lo:hi] = True
    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Mann-Whitney U / rank-based AUROC
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    sum_pos_ranks = ranks[labels].sum()
    auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


# ============================================================================
# First-onset hit metrics (cross-bias harness — single onset per pid)
# ============================================================================

def hit_at_k(
    scores: np.ndarray,
    onset: int,
    k: int,
    tau_d: int = DEFAULT_TAU_D,
    w: int = DEFAULT_NMS_W,
) -> int:
    """1 if any of the top-k NMS-suppressed peaks is within τ_d of `onset`, else 0.

    First-onset evaluation: one true onset per pid, so recall is binary.
    """
    preds = top_k_predictions(scores, k=k, w=w)
    for (t, _) in preds:
        if abs(t - onset) <= tau_d:
            return 1
    return 0


def weighted_hit_at_k(
    scores: np.ndarray,
    onset: int,
    k: int = 5,
    tau_d: int = DEFAULT_TAU_D,
    w: int = DEFAULT_NMS_W,
) -> float:
    """Rank-weighted hit: (1 - rank/k) where rank = NMS-rank of the matching prediction.

    rank 0 -> 1.0,  rank 1 -> (1 - 1/k),  ...,  rank k-1 -> 1/k,  miss -> 0.

    With k=5: 0->1.0, 1->0.8, 2->0.6, 3->0.4, 4->0.2 (matching design doc).
    """
    preds = top_k_predictions(scores, k=k, w=w)
    for rank, (t, _) in enumerate(preds):
        if abs(t - onset) <= tau_d:
            return 1.0 - rank / k
    return 0.0


def position_baseline_hit_at_1(
    first_onsets: Sequence[int],
    tau_d: int = DEFAULT_TAU_D,
) -> float:
    """No-learning baseline: detector that always predicts the median first-onset
    position would get this fraction of (pid) hits within τ_d.

    Computed in-sample (small-N self-prediction); used as a per-bias diagnostic
    column in the cross-bias heatmap. Cells whose metric < this baseline carry
    no signal beyond position-pinning.

    Returns nan if `first_onsets` is empty.
    """
    arr = np.asarray(list(first_onsets), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    median_pos = float(np.median(arr))
    return float(np.mean(np.abs(arr - median_pos) <= tau_d))


# ============================================================================
# Aggregation helpers
# ============================================================================

def aggregate_recall(per_pid_values: Sequence[Optional[float]]) -> dict:
    """Aggregate a list of per-pid recall values, ignoring None.

    Returns: {mean, std, sem, n_pids, n_skipped}
    """
    valid = [v for v in per_pid_values if v is not None]
    n = len(valid)
    n_skipped = sum(1 for v in per_pid_values if v is None)
    if n == 0:
        return {"mean": None, "std": None, "sem": None, "n_pids": 0, "n_skipped": n_skipped}
    arr = np.asarray(valid, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "sem": sem, "n_pids": n, "n_skipped": n_skipped}
