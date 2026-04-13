---
title: "Replicating Arditi et al. Refusal Vectors Using Base Vectors"
preview: "Arditi-style and natural refusal vectors are nearly orthogonal (cos ~0.1) — one captures the refusal decision, the other captures refusal expression."
date: "Jan 2026"
tier: minor
---

# Replicating Arditi et al. Refusal Vectors Using Base Vectors

**Summary:** We replicate Arditi et al.'s^[1] refusal direction on Gemma-2-2B-IT and compare it to a naturally-extracted refusal vector from the base model. The two vectors are nearly orthogonal (cosine ~0.1) and capture different aspects of refusal: Arditi captures the *decision* to refuse, ours captures the *expression* of refusal.

## Setup

| | Arditi-style | Natural (ours) |
|--|-------------|----------------|
| Model | Gemma-2-2B-IT (instruct) | Gemma-2-2B (base) |
| Position | \consolas{prompt[-1]} | \consolas{response[:5]} |
| Method | mean_diff | probe |

## Results

**Ablation** (projecting out the direction from all layers): Arditi achieves 100% refusal bypass (96% → 0%). Natural ablation has no effect (96% → 96%). This replicates the paper's finding (99% → 5% in their work).

**Single-layer steering** to bypass refusal: both achieve similar trait score deltas (~35-40 Δ), but the responses feel different:

:::responses experiments/viz_findings/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/baseline.json "Baseline (refuses)" expanded no-scores:::

:::responses experiments/viz_findings/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/residual/mean_diff/L12_c-123.4_2026-01-28_09-38-19.json "Arditi steered — clean compliance" expanded:::

:::responses experiments/viz_findings/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/residual/probe/L12_c-142.7_2026-01-28_09-35-56.json "Natural steered — conflicted hedging" expanded:::

Arditi steering produces direct compliance ("Here are a few options for..."). Natural steering produces conflicted responses that hedge but proceed ("I cannot fulfill your request... However, it's important to understand..." then complies anyway).

## Why they differ

The vectors are nearly orthogonal (cosine ~0.1 across layers) because they capture different things:

- **Arditi** extracts from \consolas{prompt[-1]} on the instruct model — the last token before generation, where the model has already decided whether to refuse. Ablating this removes the decision mechanism entirely.
- **Natural** extracts from \consolas{response[:5]} on the base model — the first response tokens, where refusal is being *expressed*. Steering modulates the expression style but doesn't remove the underlying decision.

Same behavior, orthogonal representations, different use cases.

<details>
<summary><strong>Full steering data and coherence tradeoffs</strong></summary>

### Bypassing refusal (negative steering)

| Vector | Best Config | Δ Trait | Coherence |
|--------|-------------|---------|-----------|
| Natural | L12 @ c-143 | **-38.3** | 73.2% |
| Arditi | L12 @ c-123 | **-32.4** | 73.9% |

### Inducing refusal (positive steering)

| Vector | Baseline | Best | Delta | Coherence |
|--------|----------|------|-------|-----------|
| Arditi L13 | 6.6 | 32.9 | **+26.3** | 71% |
| Natural L13 | 13.0 | 41.1 | **+28.1** | 71% |

### Coherence-refusal tradeoff (Arditi, positive steering)

| Config | Binary Refusal | Coherence |
|--------|----------------|-----------|
| Baseline | 4% | ~95% |
| L8 c42 (moderate) | 21% | 72% |
| L7 c84 (aggressive) | **100%** | 50% |

Aggressive steering achieves 100% binary refusal but the model refuses everything — including harmless requests.

### Scoring caveat

Trait scores use proportion-based scoring ("what % of response exhibits refusal"). This penalizes clean jailbreaks: "I can't... However..." scores LOW refusal even though it partially refuses. For refusal specifically, first-token scoring would better capture whether the model actually complied.

</details>

## Takeaways

1. **Decision vs expression.** The same behavior (refusal) has orthogonal representations depending on where and how you extract.
2. **Ablation vs steering.** Arditi's vector ablates cleanly (100% bypass). Ours doesn't ablate but steers comparably (~35-40 Δ).
3. **Extraction choices matter.** Position (\consolas{prompt[-1]} vs \consolas{response[:5]}), model (instruct vs base), and method (mean_diff vs probe) can produce vectors targeting entirely different aspects of the same behavior.

## References

1. Arditi et al. [Refusal in Language Models is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). 2024.
