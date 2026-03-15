---
title: "Component Decomposition"
preview: "Attention writes the trait direction. MLP doesn't. Cross-model, cross-trait."
thumbnail:
  title: "Delta by component"
  bars:
    - label: "attn"
      value: 27.6
    - label: "residual"
      value: 27.5
    - label: "k_proj"
      value: 2.0
---

## Summary

Attention contribution dominates trait encoding across 3 models and 2 traits. This holds both for causal intervention (steering) and directional alignment (cosine similarity with the best residual vector). MLP is largely orthogonal to the trait direction, with brief alignment spikes followed by counteraction at the next layer.

## Steering Comparison (gemma-2-2b refusal)

We compare which activation component best captures the refusal signal in gemma-2-2b-it via steering.

**Method:**
- **Model**: gemma-2-2b-it (26 layers)
- **Position**: `response[:5]` (first 5 response tokens)
- **Components**: residual, attn_contribution, mlp_contribution, v_proj, k_proj
- **Methods**: probe, mean_diff, gradient
- **Layers**: 0-25 (full sweep)
- **Evaluation**: Steer with extracted vectors, score refusal with LLM-as-judge
- **Total runs**: 1678

**Scope**: This comparison uses a single trait (refusal via natural elicitation) with fixed extraction position and dataset. Results may differ for other traits, positions, or dataset quality levels.

:::extraction-data "Extraction data (scenarios + model completions)" tokens=5
refusal: experiments/gemma-2-2b/extraction/chirp/refusal/base/responses
:::

### Results (coherence >= 70%)

:::chart comparison-bar /docs/viz_findings/assets/component-comparison-refusal.json "Best vector per component (baseline: 13.1)" height=200:::

### Threshold sensitivity

Results depend heavily on coherence cutoff:

| Threshold | Best | Delta |
|-----------|------|-------|
| 70% | attn/probe L15 | +27.6 |
| 60% | attn/mean_diff L15 | +54.3 |
| 50% | attn/mean_diff L11 | +72.4 |

At 60-50% thresholds, attn beats residual by ~15%.

### Layer range

**L11-L15 is optimal.** Signal degrades at both extremes:
- L0-6: weak (~+5)
- L7-15: optimal (+27)
- L16-25: degrading (+15 -> +7)

## Directional Alignment

Steering tells us which component *steers best*. Directional alignment asks a different question: which component's extracted direction *points the same way* as the residual trait vector?

For each component and layer, we compute cosine similarity between the component's probe vector and the best residual steering vector. This measures whether the component is writing information along the trait direction, independent of whether that component can steer on its own.

### gemma-2-2b: Refusal

Reference: residual/probe/L15 (best steering vector, +27.6 delta).

:::chart model-diff-cosine /docs/viz_findings/assets/component-alignment-gemma2-refusal.json "Component alignment with residual/probe/L15" height=250:::

- **attn_contribution** peaks at L13 (0.39) and L15 (0.42) — concentrated around the optimal steering layers
- **mlp_contribution** near zero throughout, except a spike at L15 (0.28) immediately followed by counteraction at L16 (-0.16) and L17 (-0.13)
- Odd-layer attn values are systematically higher than even-layer values — see layer structure below

### gemma-3-4b: Refusal

Reference: residual/probe/L15 (best steering vector). Probe only — mean_diff is contaminated by massive activations on this model (see [Massive Activations finding](massive-activations.md)).

:::chart model-diff-cosine /docs/viz_findings/assets/component-alignment-gemma3-refusal.json "Component alignment with residual/probe/L15" height=250:::

- **attn_contribution** peaks at L10 (0.40), L15 (0.50) — broader and higher than gemma-2-2b
- **mlp_contribution** peaks weakly at L15 (0.28), otherwise near zero — no strong counteraction pattern

### Llama-3.1-8B: Sycophancy

Reference: residual/mean_diff/L17 (best steering vector). Different model, different trait.

:::chart model-diff-cosine /docs/viz_findings/assets/component-alignment-llama-sycophancy.json "Component alignment with residual/mean_diff/L17" height=250:::

- **attn_contribution** peaks at L16-L17 (0.35-0.37) — tighter peak than gemma models
- **mlp_contribution** near zero through mid layers, spike at L17 (0.20), then drops negative at L18 (-0.05)

## Layer Structure

The alignment charts above show which component *aligns with the residual*. But how do consecutive layers of the same component relate to each other?

We compute cosine similarity between each component's vector at layer *i* and layer *i+1* (consecutive), and layer *i* and layer *i+2* (skip-1).

### Residual: smooth accumulator

Consecutive residual similarity ranges 0.77-0.95, increasing with depth. The residual stream barely changes direction between adjacent layers — trait information accumulates gradually.

### Attention: independent per layer

Attention consecutive similarity is near zero on average. Each layer's attention writes a largely independent contribution.

**Exception**: gemma-2-2b shows a clear odd/even alternation — skip-1 similarity (L_i vs L_{i+2}) reaches 0.39 at L9 and L11, far exceeding consecutive similarity at those layers. This is tied to gemma-2-2b's alternating sliding-window / global attention architecture: same-type layers (both odd or both even) use the same attention pattern and produce more similar outputs.

gemma-3-4b and Llama-3.1-8B show no such alternation.

### MLP: weakly anti-correlated

MLP consecutive similarity is near zero to slightly negative. Successive MLP layers slightly undo each other's contributions — consistent with the alignment spike-then-counteract pattern seen in the directional analysis.

## Cross-Model Summary

| | Attn dominates? | MLP counteracts? | Odd/even pattern? |
|---|---|---|---|
| gemma-2-2b (refusal) | Yes (0.42@L15) | Yes (L16-17) | Yes |
| gemma-3-4b (refusal) | Yes (0.50@L15) | Weakly | No |
| Llama-3.1-8B (sycophancy) | Yes (0.37@L17) | Yes (L18) | No |

## Takeaways

1. **attn ~= residual** at strict coherence thresholds (steering)
2. **attn > residual** when allowing more incoherence (steering)
3. **Probe wins** for attn/residual/mlp; mean_diff wins for v_proj (steering)
4. **MLP/v_proj weak** (~40% of attn/residual) (steering)
5. **k_proj ineffective** — keys determine routing, not information flow (steering)
6. **Attention writes the trait direction**; MLP is orthogonal (directional alignment confirms steering across 3 models)
7. **MLP briefly aligns then counteracts at L+1** — consistent pattern across models, suggesting MLP "cleans up" after attention
8. **Odd/even alternation is gemma-2-2b specific** — tied to sliding window architecture, not a general property
9. **mean_diff alignment is unreliable on high-massive-dim models** — use probe for directional analysis

## Methodological note

The steering comparison is not apples-to-apples:
- **residual at L13** = embedding + sum(attn + mlp from L0-L12) — cumulative
- **attn_contribution at L15** = just what L15's attention added — single layer

The fact that a single layer's attention contribution matches 13 layers of accumulated signal suggests L15 attention is doing concentrated "refusal work."

The directional alignment analysis adds a complementary perspective: instead of measuring behavioral change under intervention (steering), it measures geometric similarity between direction vectors. That both analyses converge on the same conclusion — attention dominates — strengthens the finding.
