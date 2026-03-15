---
title: "[TODO] Coefficient Scaling Law"
preview: "Perturbation ratio predicts coherence cliff at ~1.0 (needs validation beyond refusal)"
---

# Coefficient Scaling Law

**Status:** Partially validated on refusal traits. Needs validation on optimism, sycophancy, etc.

## Finding

Effective steering strength is determined by **perturbation ratio**, not raw coefficient:

```
perturbation_ratio = (coef × vector_norm) / activation_norm
```

## Validation Results (gemma-2-2b, refusal traits only)

| Ratio | Avg Coherence | Avg Delta | % Coherent (≥70) |
|-------|---------------|-----------|------------------|
| 0.0-0.3 | 88.4 | +3.1 | 100% |
| 0.3-0.5 | 90.5 | -5.4 | 100% |
| 0.5-0.8 | 80.8 | +1.0 | 85% |
| 0.8-1.0 | 69.5 | +8.5 | 53% |
| 1.0-1.3 | 53.6 | +25.5 | 19% |
| 1.3+ | 39.5 | +26.2 | 8% |

**Coherence cliff detected at ratio ~1.15**

![Scaling Law Validation](../../analysis/outputs/scaling_law.png)

## Key Insights

1. **Coherence cliff is real** — drops sharply around ratio 1.0-1.15
2. **Trade-off is unavoidable** — high delta requires sacrificing coherence
3. **"Sweet spot" depends on goal:**
   - Conservative (high coherence): ratio 0.3-0.5
   - Balanced: ratio 0.8-1.0
   - Aggressive (max delta): ratio 1.0-1.3

## Why This Matters

Different extraction methods produce vectors with different norms:

| Method | Typical vec_norm/act_norm | Implication |
|--------|---------------------------|-------------|
| mean_diff | ~0.12-0.15 (stable) | Fixed coefficient works across layers |
| probe | Varies 3x across layers | Needs layer-specific coefficient tuning |

The formula lets you compute the right coefficient without trial-and-error:
```python
target_coef = target_ratio × (activation_norm / vector_norm)
```

## Caveats

- **Only validated on refusal traits** — chirp/refusal_v2, chirp/refusal_v3, arditi/refusal
- **Original claim was from optimism** — those vectors no longer exist on disk
- **735 of 9261 steering runs** had computable ratios (8%)

## TODO

- [ ] Re-extract vectors for hum/optimism, hum/sycophancy, hum/confidence
- [ ] Re-run validation across all traits
- [ ] Check if "0.5-0.8 sweet spot" holds for non-refusal traits
- [ ] Update thresholds if needed

## Implementation

- Formula used in `analysis/steering/evaluate.py` for adaptive coefficient search
- Validation script: `analysis/validate_scaling_law.py`

## References

- Original observation: docs/other/research_findings.md (2025-12-04)
