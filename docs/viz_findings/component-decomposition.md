---
title: "Component Decomposition"
preview: "Attention writes the trait direction. MLP doesn't. Cross-model, cross-trait."
date: "Apr 2026"
tier: major
thumbnail:
  title: "Delta by component"
  bars:
    - label: "attn"
      value: 71.2
    - label: "residual"
      value: 67.9
    - label: "mlp"
      value: 35.8
---

## Summary

Attention contribution dominates trait encoding. MLP is roughly half as effective. Residual plateaus broadly because it accumulates prior layers; single-layer attention still matches it.

## Steering by Component (Qwen3.5-9B, optimism)

We extract optimism vectors from 5 activation components and steer with each independently. Baseline optimism: 83.2. Negative steering (reduce optimism). All results filtered to coherence >= 80%.

:::chart comparison-bar /docs/viz_findings/assets/component-bar-qwen-optimism.json "Best steering delta per component (baseline: 83.2)" height=200:::

Attention and residual are close at strict coherence thresholds. MLP and v_proj cluster at ~half the effect. k_proj is near zero — keys determine routing, not information content.

### Per-layer steering curves

:::chart model-diff-trait-delta /docs/viz_findings/assets/component-steering-qwen-optimism.json "Trait score delta by layer (coherence >= 80%)" height=300:::

Attention peaks sharply at L17-18 (mid-network), then drops off. Residual has a broader plateau (L5-11) because it accumulates signal from all prior layers. MLP never exceeds -36 at any layer.

k_proj and v_proj only have data at 8 of 32 layers — Qwen3.5-9B uses hybrid linear/full attention, with k_proj/v_proj only accessible on the 8 full-attention layers (3, 7, 11, 15, 19, 23, 27, 31).

### Methodological note

The residual vs. attention comparison is not apples-to-apples: residual at layer N is the cumulative sum of all prior attention + MLP contributions, while attention at layer N is just that single layer's output. That a single layer's attention matches 7+ layers of accumulated residual signal suggests concentrated "trait work" at specific layers.

## Takeaways

1. **Attention dominates** — single-layer attention matches cumulative residual
2. **MLP is ~half as effective**
3. **k_proj is ineffective** — keys determine attention routing, not information content
4. **Attention peaks mid-network** (L17-18 on Qwen3.5-9B) — concentrated "trait work" at specific layers
