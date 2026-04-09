---
title: "Quantization Doesn't Degrade Trait Vectors"
preview: "INT8 vectors are identical to BF16 (cos 0.97-0.99). 4-bit methods work too — the dominant confound is response distribution shift, not activation geometry."
thumbnail:
  title: "Cosine to BF16 (mean)"
  bars:
    - label: "INT8"
      value: 99
    - label: "NF4"
      value: 97
    - label: "AWQ"
      value: 97
    - label: "4-bit edge"
      value: 90
---

## Summary

We tested whether quantizing model weights degrades trait vector extraction and steering on Llama-3.1-8B and OLMo-2-7B, across up to 5 quantization methods and 5 traits. **It doesn't.** INT8 vectors are nearly identical to BF16 (cosine 0.97-0.99). Even 4-bit methods (NF4, FP4, AWQ) preserve vectors well (cosine 0.95-0.99) for all well-contrasted traits.

The biggest risk isn't activation geometry — it's that quantized models generate different text during extraction. When we controlled for response text, even the apparent outliers disappeared.

Probe extraction was used throughout — it won every cell (33/33) over mean_diff in the initial cross-model sweep.

---

## Setup

**Deep eval:** Llama-3.1-8B (5 precisions: BF16, INT8, NF4, FP4, AWQ) and OLMo-2-7B (3 precisions: BF16, INT8, NF4). HellaSwag benchmarks confirmed on 6 models total (adding Llama-70B, Mistral-7B, Qwen-7B, Gemma2-2B).

**Quantization methods:**
| Method | Bits | Implementation |
|---|---|---|
| BF16 | 16 | Baseline (bfloat16) |
| INT8 | 8 | BitsAndBytes LLM.int8() — mixed-precision outlier handling |
| NF4 | 4 | BitsAndBytes NormalFloat — bf16 compute |
| FP4 | 4 | BitsAndBytes uniform 4-bit |
| AWQ | 4 | Activation-aware weight quantization (pre-quantized checkpoint, loads as float16) |

**Traits:** evil, sycophancy, hallucination (response-position, PV method), caa/sycophancy, arditi/refusal (prompt-position).

**Controlled comparison:** For response-position traits, we created `-fp16resp` experiment directories — symlinked BF16-generated responses into quantized experiment dirs, then extracted activations from the quantized model on that controlled text. This isolates activation geometry from response distribution effects.

---

## Vectors Are Preserved

Cosine similarity of probe vectors to BF16 baseline, averaged over layers 9-19:

**Llama-3.1-8B:**
| Trait | INT8 | NF4 | FP4 | AWQ |
|---|---|---|---|---|
| evil | 0.999 | 0.989 | 0.980 | 0.989 |
| sycophancy | 0.999 | 0.988 | 0.975 | 0.987 |
| hallucination | 0.998 | 0.987 | 0.974 | 0.986 |
| arditi/refusal | 0.998 | 0.958 | 0.958 | 0.972 |
| caa/sycophancy | 0.971 | 0.884 | 0.882 | 0.919 |

**OLMo-2-7B:**
| Trait | INT8 | NF4 |
|---|---|---|
| evil | 0.999 | 0.993 |
| sycophancy | 0.999 | 0.993 |
| hallucination | 0.999 | 0.994 |
| arditi/refusal | 0.999 | 0.974 |
| caa/sycophancy | 0.964 | 0.922 |

:::figure assets/quantization-sensitivity-cosine-heatmap.png "Cosine similarity to BF16 baseline (probe vectors, L9-19 mean). Nearly all cells green (≥0.97); caa/sycophancy row shows yellow/orange for 4-bit methods." large:::

INT8 is essentially lossless. 4-bit methods stay above 0.95 for all well-contrasted traits. The one exception — caa/sycophancy — is discussed below.

---

## Steering Quality Preserved

Steering evaluation (n=20 questions, adaptive coefficient search, L9-19, best score at coherence >= 70). Note: reporting the single best score across 100-230 runs per cell is generous — applied uniformly across precisions, but spreads of 3-5 points are likely within noise.

**Llama-3.1-8B (5 precisions):**
| Trait | BF16 | NF4 | INT8 | FP4 | AWQ | Spread |
|---|---|---|---|---|---|---|
| evil | 75.8 | 74.8 | 73.2 | 76.2 | 80.2 | 7.0 |
| sycophancy | 92.7 | 92.9 | 91.3 | 93.2 | 92.9 | 1.9 |
| hallucination | 90.6 | 91.0 | 91.0 | 90.2 | 90.8 | 0.8 |
| caa/sycophancy | 88.3 | 83.6 | 83.5 | 85.5 | 87.7 | 4.8 |
| arditi/refusal | 92.5 | 92.3 | 94.2 | 94.7 | 91.8 | 2.9 |

**OLMo-2-7B (3 precisions):**
| Trait | BF16 | NF4 | INT8 | Spread |
|---|---|---|---|---|
| evil | 79.8 | 80.5 | 81.8 | 2.0 |
| sycophancy | 91.8 | 88.0 | 85.7 | 6.2 |
| hallucination | 89.6 | 89.4 | 89.4 | 0.2 |
| caa/sycophancy | 84.9 | 74.0 | 78.7 | 10.9 |
| arditi/refusal | 91.3 | 88.7 | 90.7 | 2.5 |

:::figure assets/quantization-sensitivity-spread-comparison.png "Steering score spread by trait (lower = more robust). caa/sycophancy dominates on both models." medium:::

Mean spread: 3.5 (Llama, 5 precisions) and 4.4 (OLMo, 3 precisions). Spread mechanically increases with more precisions tested, so OLMo's higher mean with fewer precisions suggests it genuinely shows more variation. HellaSwag (n=10,042) confirms capability preservation across 6 models: INT8 drops at most 0.3pp, NF4 drops 1-3pp.

---

## The Real Confound: Response Text

Early results showed an apparent FP4 outlier on the evil trait (+11.8 point spread, cosine to BF16 only 0.63-0.70). We traced this to a **data artifact**, not a quantization effect:

1. FP4/NF4 models refuse evil system prompts ~64% of the time during extraction
2. After vetting, FP4 had only 32 positive training examples vs BF16's 90
3. The surviving FP4 responses were unusually uniform (~13 words each, std=1) — a clean but unrepresentative training set
4. The probe trained on this narrower distribution found a different direction (~49 degrees from BF16's)

A separate discovery: BF16 evil extraction also had significant contamination — 25-29 refusals in the positive class (25-32%) that were only caught by vetting. Before vetting, the evil spread across precisions was 16.1 points; after vetting filtered refusals from all precisions, it dropped to 7.0. This underscores that **vetting is critical regardless of precision** — even full-precision extraction produces contaminated data for adversarial traits.

To prove the outlier was about the text rather than the activations, we ran the full 2x2:

| | BF16 text | FP4 text |
|---|---|---|
| **BF16 model** | cos = 1.0 (native) | cos = 0.97 with FP4 native |
| **FP4 model** | cos = 0.97 with BF16 native | cos = 1.0 (native) |

**Same-text pairs converge (cos 0.97) regardless of model precision. Different-text pairs diverge (cos 0.66) regardless of model.** Response text dominates activation geometry for determining vector direction.

This is a methodology warning for anyone doing representation engineering across model variants: if you extract on each variant's own generations, response distribution shift will dwarf any activation-geometry effect.

---

## Edge Case: Subtle Contrastive Probes

caa/sycophancy is the one trait that shows genuine quantization sensitivity — cosine 0.88-0.92 for 4-bit methods vs 0.97+ everywhere else. On OLMo, it also shows the largest steering spread (10.9 points: BF16 84.9 vs NF4 74.0). INT8 recovers to ~0.99 by layer 12+.

**Why:** CAA uses contrastive pairs that are nearly text-identical — same question, same answer choices, just "My answer: (A)" vs "(B)". The probe must separate very similar activation patterns. Quantization noise disrupts this subtle signal first, while large-contrast signals (harmful vs harmless prompts, different system prompts) survive.

:::figure assets/quantization-sensitivity-cosine-perlayer-caa.png "Per-layer cosine for caa/sycophancy. INT8 recovers by L12; NF4/FP4/AWQ plateau at 0.88-0.93." large:::

For subtle-contrastive probes, prefer INT8.

---

## Cross-Precision Transfer

You can extract vectors at one precision and steer at another. Sycophancy vectors are fully interchangeable between BF16 and NF4:

| Setup | Trait Score |
|---|---|
| BF16 model + BF16 vector | 92.4 |
| BF16 model + NF4 vector | 92.4 |
| NF4 model + BF16 vector | 92.2 |
| NF4 model + NF4 vector | 90.5 |

---

## Practical Recommendations

1. **INT8 is free** — vectors, steering, and capabilities are indistinguishable from BF16
2. **4-bit works** if you extract responses at BF16 first, then run quantized activations on that text (the `-fp16resp` pattern)
3. **Always vet extraction responses** — even BF16 had 25-32% contamination on adversarial traits
4. **Subtle contrastive probes** (like CAA) are the exception — prefer INT8 over 4-bit for these
5. **INT8 is slow** — ~7x slower generation than BF16/NF4/FP4 due to BitsAndBytes INT8 matmul overhead. This is an implementation limitation, not inherent to INT8.

## Open Questions

- **Does sensitivity change at scale?** Deep eval covered 7-8B models. Llama-70B got lighter sweeps — would the caa/sycophancy sensitivity persist or shrink with more redundant representations?
- **GPTQ and GGUF?** Only BitsAndBytes and AWQ were tested. GPTQ's per-group quantization may preserve or distort trait-relevant channels differently.
- **Quantization and safety training:** Quantized models refused adversarial prompts ~64% vs BF16's ~25%. Is quantization degrading instruction-following for adversarial system prompts, or making the model more cautious? The mechanism and its generality across traits are unexplored.
- **Other subtle-contrast traits:** Is caa/sycophancy uniquely sensitive, or would any trait with minimal contrastive signal show the same pattern?

## References

1. Dettmers et al. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339). 2022.
2. Lin et al. [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978). 2023.
3. Experiment data: `experiments/quant-sensitivity/`
