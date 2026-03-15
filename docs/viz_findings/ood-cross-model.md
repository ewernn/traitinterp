---
title: "OOD: Cross-Model Transfer"
preview: "Base-extracted vectors transfer to SFT/DPO variants; DPO shows harder cliffs."
---

# OOD: Cross-Model Transfer

Base-extracted vectors transfer to SFT/DPO variants.

## Setup

**Models:** `mistralai/Mistral-7B-v0.1` (base) -> `HuggingFaceH4/mistral-7b-sft-beta` (SFT) -> `HuggingFaceH4/zephyr-7b-beta` (SFT+DPO)

## Results (optimism trait, L12)

| Model | Best Coef | Trait Score | Coherence |
|-------|-----------|-------------|-----------|
| SFT   | 1         | 84.3%       | 76.0%     |
| DPO   | 1         | 83.2%       | 81.6%     |

## Key Findings

- **Transfer works:** 80%+ trait scores on both aligned models using base vectors
- **Optimal layer preserved:** L12 best for both - alignment doesn't shift where features live
- **DPO more resistant at early layers:** L8 SFT 82.5% vs DPO 65.8%
- **DPO has hard cliff:** L24/coef=6 -> 1.1% trait, 6.3% coherence (catastrophic collapse)
- **DPO maintains coherence better:** Until it hits threshold, then fails completely

## Interpretation

DPO creates a more "locked-in" model that resists steering but collapses catastrophically when pushed too hard.
