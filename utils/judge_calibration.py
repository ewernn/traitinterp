"""
Calibration maps between LLM judge backends.

Problem: `core.kwargs_configs.MIN_COHERENCE=77` and similar thresholds were
tuned empirically for GPT-4.1-mini's logprob-weighted score distribution.
Backends that don't expose logprobs (Anthropic, some local models) produce
scores with different variance/bias shape, so the same threshold passes at
a different rate. A calibration map reshapes non-logprob scores to match
the logprob-weighted distribution.

Design:
    A calibration map is a monotonic function from source_score → target_score,
    fit via isotonic regression on paired (source, target) samples collected
    by scoring the same texts on both backends. At inference time, the
    AnthropicBackend (or any other non-logprob backend) applies the map to
    its raw output score.

Scope:
    One map per (source_backend, target_backend, task) triple. Tasks are
    coherence / trait_score / etc. because prompt structure differs. In
    practice we start with coherence only (highest-value target — gates
    MIN_COHERENCE=77 in steering eval).

⚠️  EMPIRICAL FINDING (2026-04-17) — READ BEFORE APPLYING A MAP ⚠️

    The first fitted map (Anthropic Sonnet 4.5 → OpenAI gpt-4.1-mini on
    coherence) showed Spearman correlation of only 0.11 on held-out pairs.
    Interpretation: the two backends produce substantively different
    coherence judgments on steered model-generated text — not just a scale
    shift. A monotonic map CANNOT recover threshold compatibility when
    rank-order itself disagrees.

    Practical implication: calibration maps work for scale/bias drift
    between backends that agree on rank-order. They don't work when the
    backends disagree at the judgment level (which is what we observe
    here). For cross-backend pipelines, expect to retune thresholds
    empirically per backend rather than rely on the map alone.

    Still ship the infrastructure because:
      - Local LLMs with logprobs (vLLM/llama.cpp) may not need calibration
        (they share OpenAI's scoring mechanism).
      - Other Anthropic tasks (trait_score) may show different correlation
        than coherence — worth trying per-task.
      - The fitted map tells you how much disagreement exists, which is
        itself useful diagnostic data.

Files:
    Maps live at datasets/llm_judge/calibration/{name}.json with schema:
        {
            "source_identifier": "anthropic/claude-sonnet-4-5",
            "target_identifier": "openai/gpt-4.1-mini",
            "task": "coherence",
            "fit_method": "isotonic",
            "n_pairs_fit": 150,
            "n_pairs_validation": 30,
            "source_points": [...],  # sorted ascending
            "target_points": [...],  # interpolation partners
            "validation": {"mae": ..., "spearman_r": ..., "passrate_delta_at_77": ...},
            "fitted_at": "2026-04-17T...",
        }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CalibrationMap:
    """Monotonic score map between two judge backends for one task.

    Load a persisted map with `CalibrationMap.load(path)`; apply it to raw
    scores via `map.apply(score)`. See module docstring for file schema.
    """

    source_identifier: str
    target_identifier: str
    task: str
    source_points: List[float]
    target_points: List[float]
    fit_method: str = "isotonic"

    def __post_init__(self):
        if len(self.source_points) != len(self.target_points):
            raise ValueError(
                f"source_points ({len(self.source_points)}) and target_points "
                f"({len(self.target_points)}) must have equal length."
            )
        if len(self.source_points) < 2:
            raise ValueError(
                "CalibrationMap needs at least 2 points to interpolate."
            )
        # Sort by source for monotonic interpolation
        pairs = sorted(zip(self.source_points, self.target_points))
        self.source_points = [p[0] for p in pairs]
        self.target_points = [p[1] for p in pairs]

    def apply(self, score: Optional[float]) -> Optional[float]:
        """Map a source-backend score to the equivalent target-backend score.

        Uses linear interpolation between anchor points. Scores below/above
        the fit range are clamped to the endpoints (extrapolation would be
        misleading given the monotonic fit). None → None.
        """
        if score is None:
            return None
        xs, ys = self.source_points, self.target_points
        if score <= xs[0]:
            return ys[0]
        if score >= xs[-1]:
            return ys[-1]
        # Bisect-style linear interpolation
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= score:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if x1 == x0:  # degenerate — shouldn't happen post-sort
            return (y0 + y1) / 2
        t = (score - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationMap":
        """Load a map from its JSON file.

        Emits a stderr warning if the map's embedded validation stats indicate
        poor fit quality (Spearman < 0.5 means the two backends disagree on
        rank-order, which a monotonic map can't fix). This is the primary
        safety signal telling users the map shouldn't be trusted blindly.
        """
        import sys as _sys
        data = json.loads(Path(path).read_text())
        # Surface quality issues before the user starts using the map.
        validation = data.get("validation", {})
        spearman = validation.get("spearman_r")
        if spearman is not None and spearman < 0.5:
            print(
                f"Warning: CalibrationMap loaded from {path} has Spearman={spearman:.2f} "
                f"(below 0.5). The two backends disagree on rank-order; a monotonic map "
                f"cannot recover threshold compatibility. Treat as diagnostic only. "
                f"See utils/judge_calibration.py module docstring for context.",
                file=_sys.stderr,
            )
        return cls(
            source_identifier=data["source_identifier"],
            target_identifier=data["target_identifier"],
            task=data["task"],
            source_points=list(data["source_points"]),
            target_points=list(data["target_points"]),
            fit_method=data.get("fit_method", "isotonic"),
        )

    def to_dict(self, extra: Optional[dict] = None) -> dict:
        """Serialize to dict (for JSON persistence with extra metadata)."""
        out = {
            "source_identifier": self.source_identifier,
            "target_identifier": self.target_identifier,
            "task": self.task,
            "fit_method": self.fit_method,
            "source_points": list(self.source_points),
            "target_points": list(self.target_points),
        }
        if extra:
            out.update(extra)
        return out


def fit_isotonic(
    source_scores: List[float],
    target_scores: List[float],
    source_identifier: str,
    target_identifier: str,
    task: str,
) -> CalibrationMap:
    """Fit a monotonic calibration map via isotonic regression.

    Why isotonic: we expect the relationship source→target to be monotonically
    increasing (a higher Anthropic score should predict a higher OpenAI score)
    but not necessarily linear. Isotonic regression gives the best monotonic
    fit in the L2 sense without parametric assumptions.

    Returns a CalibrationMap ready to persist.
    """
    if len(source_scores) != len(target_scores):
        raise ValueError("source_scores and target_scores must have equal length.")
    if len(source_scores) < 5:
        raise ValueError(
            f"Need at least 5 paired scores to fit isotonic regression; got {len(source_scores)}."
        )

    # Late import — scipy is already a dep but keep this module light.
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(source_scores, target_scores)
    # Emit a finite set of anchor points covering the observed range, plus
    # sample at each fitted value to preserve fidelity.
    sorted_src = sorted(set(source_scores))
    fitted_tgt = [float(iso.predict([s])[0]) for s in sorted_src]
    return CalibrationMap(
        source_identifier=source_identifier,
        target_identifier=target_identifier,
        task=task,
        source_points=sorted_src,
        target_points=fitted_tgt,
        fit_method="isotonic",
    )
