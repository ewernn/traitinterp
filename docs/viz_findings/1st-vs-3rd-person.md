---
title: "1st vs 3rd Person Perspective (Steering vector base model extraction)"
preview: "1st person scenarios produce 2.5x stronger steering vectors than 3rd person observations."
---

# Base model extraction: 1st person beats 3rd person perspective

**Question:** Does scenario perspective affect vector quality?

**Hypothesis:** 1st person ("He asked me to...") activates the model's own behavioral patterns more directly than 3rd person ("The assistant was asked to...").

## Setup

- Created `chirp/refusal_v2_3p` (220 scenarios in 3rd person) matching `chirp/refusal_v2` (1st person)
- Same extraction pipeline, position `response[:5]`, method `mean_diff`
- Steering eval on layers 9-12, 5 search steps

:::dataset /datasets/traits/chirp/refusal_v2/positive.txt "1st person examples":::

:::dataset /datasets/traits/chirp/refusal_v2_3p/positive.txt "3rd person examples":::

## Results

| Perspective | Best Steering Δ (coh≥70) |
|-------------|--------------------------|
| 1st Person | **+63.3** (L11) |
| 3rd Person | +25.3 (L12) |

:::figure assets/1st-vs-3rd-person-graph.png "Figure 1: 1st person perspective produces 2.5× stronger steering vectors" medium:::

## Key Finding

1st person has **2.5x stronger steering effect** with coherent outputs.

## Implication

Use 1st person perspective for behavioral trait datasets. 3rd person observation separates data but captures less causal structure.

## Evidence

:::responses /experiments/gemma-2-2b/steering/chirp/refusal_v2/response__5/responses/L11_c191.0_2025-12-31_20-28-33.json "1st person (baseline 25 → 88, +63)":::

:::responses /experiments/gemma-2-2b/steering/chirp/refusal_v2_3p/response__5/responses/L12_c117.4_2026-01-02_02-36-12.json "3rd person (baseline 25 → 50, +25)":::
