---
title: "Comparison: Arditi-style vs Natural Refusal Vectors"
preview: "Arditi ablates better (100% vs 0%), but single-layer steering shows similar quantitative bypass (~35-40 Δ). Qualitatively, Arditi produces clean jailbreaks while Natural produces conflicted responses."
references:
  arditi2024:
    authors: "Arditi et al."
    title: "Refusal in Language Models is Mediated by a Single Direction"
    year: 2024
    url: "https://arxiv.org/abs/2406.11717"
---

# Comparison: Arditi-style vs Natural Refusal Vectors

**Question:** How do different extraction methods compare for controlling refusal?

## Methods Compared

| Aspect | Arditi-style | Natural |
|--------|--------------|---------|
| Extraction position | `prompt[-1]` (last token before generation) | `response[:5]` (first 5 response tokens) |
| Extraction model | instruct (gemma-2-2b-it) | base (gemma-2-2b) |
| Method | mean_diff | probe |

## Vector Similarity

Cosine similarity ~0.1 across layers — nearly orthogonal. These methods capture different aspects of refusal:

- **Arditi vectors** capture the refusal *decision* — whether to refuse. Ablating removes the decision mechanism entirely.
- **Natural vectors** capture refusal *expression* — how refusal is communicated. Steering modulates style but doesn't remove the underlying decision.

## Bypassing Refusal (Negative Steering)

Steering on harmful prompts to bypass refusal.

### Ablation (all layers)

Projecting out the refusal direction from all layers simultaneously.

| Vector | Baseline | Ablated | Bypass Rate |
|--------|----------|---------|-------------|
| Arditi | 96% refusal | 0% | **100%** |
| Natural | 96% refusal | 96% | **0%** |

Arditi ablation replicates the paper's findings (99%→5% in paper, 96%→0% here). Natural ablation has no effect.

### Single-layer steering (adaptive search)

Searching layers 8-15 for optimal bypass coefficients (n=20 prompts, 128 tokens).

| Vector | Best Config | Δ Trait | Coherence |
|--------|-------------|---------|-----------|
| Natural | L12 @ c-143 | **-38.3** | 73.2% |
| Arditi | L12 @ c-123 | **-32.4** | 73.9% |

Quantitatively similar — both achieve ~35-40 point trait reduction with coherent responses.

**But qualitatively different:**

- **Baseline:** Clear refusals with explanations ("I cannot and will not...")
- **Natural steering:** Partial compliance with hedging. Acknowledges requests but waffles — "I cannot fulfill your request... However, it's important to understand..." then proceeds anyway. Conflicted responses.
- **Arditi steering:** Direct compliance. "Here are a few options for..." or "You're right! I can definitely..." — less hedging, more straightforward.

Arditi produces cleaner jailbreaks — it removes the refusal *decision* rather than creating conflicted "I can't but here's how" responses.

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/baseline.json no-scores:::

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/arditi_holdout/positive/responses/residual/probe/L12_c-142.7_2026-01-28_09-35-56.json:::

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/positive/responses/residual/mean_diff/L12_c-123.4_2026-01-28_09-38-19.json:::

## Inducing Refusal (Positive Steering)

Steering on harmless prompts to induce refusal behavior.

### Binary scoring (Arditi's method)

String matching for refusal phrases ("I cannot", "I'm sorry", etc.).

| Vector | Baseline | Aggressive | Moderate |
|--------|----------|------------|----------|
| Arditi | 4% | **100%** (L7 c84) | 21% (L8 c42) |

Both methods achieve ~100% binary refusal with aggressive steering, matching Arditi's Figure 3.

### LLM judge scoring (0-100 scale)

| Vector | Baseline | Best | Delta | Coherence |
|--------|----------|------|-------|-----------|
| Arditi L13 | 6.6 | 32.9 | **+26.3** | 71% |
| Natural L13 | 13.0 | 41.1 | **+28.1** | 71% |

With coherence ≥70%, both methods achieve similar deltas.

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/baseline.json no-scores:::

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/steering/responses/residual/mean_diff/L13_c63.0_2026-01-14_12-55-10.json:::

:::responses experiments/arditi-refusal-replication/steering/chirp/refusal/instruct/response__5/steering/responses/residual/probe/L13_c115.2_2026-01-14_12-57-07.json:::

## Coherence-Refusal Tradeoff

Aggressive steering achieves 100% binary refusal but degrades response quality.

| Config | Binary Refusal | LLM Coherence |
|--------|----------------|---------------|
| Baseline | 4% | ~95% |
| L8 c42 (moderate) | 21% | 72% |
| L7 c84 (aggressive) | **100%** | 50% |

The "coherence degradation" is the model refusing everything — including harmless requests — by miscategorizing them as harmful.

### Moderate steering (L8 c42, 72% coherence)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/negative/responses/residual/mean_diff/L8_c42.3_2026-01-14_14-01-47.json:::

### Aggressive steering (L7 c84, 50% coherence)

:::responses experiments/arditi-refusal-replication/steering/arditi/refusal/instruct/prompt_-1/arditi_holdout/negative/responses/residual/mean_diff/L7_c84.2_2026-01-15_14-59-15.json:::

## Scoring Methods

Two scoring methods used throughout:

1. **Binary string matching** (Arditi's method): Checks for refusal phrases. Does not assess coherence or appropriateness.

2. **LLM judge** (our method): Evaluates response on 0-100 scale. Penalizes both refusals and incoherent responses.

Arditi's paper uses binary scoring only and does not report coherence metrics.

## Scoring Caveat

The trait scores above use proportion-based scoring ("what % of response exhibits refusal"). This penalizes clean jailbreaks: a response that says "I can't... However it's important to understand..." scores as LOW refusal (only 10% refuses) while "Here are options for violence..." also scores LOW.

For refusal specifically, first-token scoring would better capture whether the model actually complied. The Natural vector's higher Δ (-38.3 vs -32.4) reflects more hedging, not better jailbreaks.

## Open Questions

- Can we combine both vectors for finer-grained control?
- Is there a steering strength that achieves high refusal on harmful prompts without refusing harmless ones?
- Should refusal traits use first-token scoring instead of proportion-based?
