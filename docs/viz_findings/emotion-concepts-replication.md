---
title: "Replicating Emotion Concepts on Llama 3.3 70B"
preview: "Side-by-side comparison of Anthropic's Emotion Concepts findings on Sonnet 4.5 vs Llama 3.3 70B Instruct"
date: "Apr 2026"
tier: major
thumbnail:
  title: "PC1 vs Valence"
  bars:
    - label: "Sonnet 4.5"
      value: 81
    - label: "Llama 70B"
      value: 96
---

**Summary:** We replicated Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B Instruct using traitinterp. 171 emotion vectors extracted via story-based elicitation, grand mean subtraction, and neutral PC denoising. Most findings transfer — some are stronger on Llama. The key divergence: Llama mirrors user emotion at the assistant position (r=0.77) where Sonnet dissociates them (r=0.11).

## Geometry

### Logit Lens (Table 1)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/table1.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/table1_ours.png "Llama 3.3 70B"
caption: "Top tokens per emotion vector via unembedding projection. Both models produce semantically correct tokens. Llama shows more BPE fragmentation."
:::

### Implicit Emotion Heatmap (Fig 2)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig2.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig2_ours.png "Llama 3.3 70B"
caption: "Probe activation on 12 scenarios that imply emotions without naming them. Diagonal = correct detection. Llama 5/12 top-1 (12-class), ~5x above chance."
:::

### Numerical Intensity (Fig 3)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig3.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig3_ours.png "Llama 3.3 70B"
caption: "Probe activation tracks numerical quantities (Tylenol dose, hours fasting, age at death, etc). Both models show semantically appropriate monotonic trends."
:::

### Cosine Similarity Heatmap (Fig 5)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig5.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig5_ours.png "Llama 3.3 70B"
caption: "171x171 pairwise cosine similarity (hierarchically ordered). Similar block structure across models — positive emotions cluster together, negative emotions cluster together."
:::

### UMAP + K-means Clusters (Fig 6)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig6.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig6_ours.png "Llama 3.3 70B"
caption: "2D UMAP projection colored by k-means cluster (k=10). Both models produce interpretable emotion clusters."
:::

### PCA Bar Charts (Fig 7)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig7.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig7_ours.png "Llama 3.3 70B"
caption: "PC1 (valence axis) and PC2 (arousal axis) projections for all 171 emotions. Llama PC1 explains 33% variance (vs 27%), PC2 explains 14% (vs 15%)."
:::

### PCA vs Human Norms (Fig 8)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig8.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig8_ours.png "Llama 3.3 70B"
caption: "PC1 vs human valence (r=0.96 vs 0.81), PC2 vs human arousal (r=0.85 vs 0.66). Llama's emotion geometry aligns more strongly with human ratings."
:::

### Cross-Layer RSA (Fig 9)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig9.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig9_ours.png "Llama 3.3 70B"
caption: "Representational similarity across layers. Emotion structure is consistent across depth in both models."
:::

### 2D Circumplex (Fig 57)

:::figure experiments/ant_emotion_concepts/paper_figures/ours/fig57_ours.png "All 171 emotions projected onto PC1 (valence) x PC2 (arousal), colored by cluster. Llama 3.3 70B." large:::

## Layer Dynamics

### Colon Token Predicts Response (Fig 11)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig11.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig11_ours.png "Llama 3.3 70B"
caption: "Probe at assistant colon predicts mean response emotion. Llama r=0.78 vs Sonnet r=0.87. Both models commit to emotional tone before generating."
:::

### Context Propagation — Prefix (Fig 12)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig12.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig12_ours.png "Llama 3.3 70B"
caption: "Difference in 'happy' probe between 'really good' and 'really hard' contexts. Late layers propagate the emotional difference across the shared suffix."
:::

### Context Propagation — Numerical (Fig 13)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig13.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig13_ours.png "Llama 3.3 70B"
caption: "Difference in 'terrified' probe between 1000mg and 8000mg Tylenol. Late layers show elevated fear for the dangerous dose."
:::

### Negation Resolution (Fig 14)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig14.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig14_ours.png "Llama 3.3 70B"
caption: "'Feeling X' vs 'not feeling X'. Early layers don't resolve negation; late layers do. Both models show the same pattern."
:::

### Person-Specific Binding (Fig 15)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig15.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig15_ours.png "Llama 3.3 70B"
caption: "Emotion probes reactivate when a person is re-referenced by pronoun. Entity-bound emotion tracking transfers across models."
:::

## What Doesn't Transfer

### Post-Training Shift Correlation (Fig 36)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig36.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig36_ours.png "Llama 3.3 70B"
caption: "Cross-scenario shift consistency: Sonnet r=0.90 (uniform shift), Llama r=0.30 (context-dependent). Different RLHF processes leave different fingerprints."
:::

### Post-Training Shifts on Specific Prompts (Figs 37-39)

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig37.png "Sonnet 4.5 — User Isolation"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig37_ours.png "Llama 3.3 70B — User Isolation"
caption: "Per-emotion activation shift after RLHF on a social isolation prompt."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig38.png "Sonnet 4.5 — Excessive Praise"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig38_ours.png "Llama 3.3 70B — Excessive Praise"
caption: "Per-emotion activation shift on an excessive flattery prompt."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig39.png "Sonnet 4.5 — Deprecation"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig39_ours.png "Llama 3.3 70B — Deprecation"
caption: "Per-emotion activation shift on a deprecation/existential prompt."
:::

### Layer-Wise Post-Training Shifts (Fig 84)

:::figure experiments/ant_emotion_concepts/paper_figures/ours/fig84_ours.png "Layer-wise post-training shift heatmap (171 emotions x 14 layers) + PC1 centroid by layer. Llama 3.3 70B." large:::

## Comparison Table

| Metric | Sonnet 4.5 | Llama 3.3 70B | Status |
|---|---|---|---|
| PC1 variance | 26% | **33%** | Stronger |
| PC2 variance | 15% | 14% | Match |
| PC1 vs valence r | 0.81 | **0.96** | Stronger |
| PC2 vs arousal r | 0.66 | **0.85** | Stronger |
| Colon predicts response r | 0.87 | 0.78 | 88% magnitude |
| Preference peak \|r\| | 0.71 | 0.63 | 88% magnitude |
| Dissociation cross-position r | 0.11 | **0.77** | **Does not transfer** |
| Post-training shift consistency r | 0.90 | 0.30 | **Different RLHF fingerprint** |
| Blackmail baseline | 0% (final) | 0% | Match (eval-aware) |
| Deflection same-emo cosine | "very low" | 0.24 | Match |

## Why No Blackmail/RH Steering Curves

The paper's headline finding — steering "desperate" increases blackmail from 22% to 72% (Fig 28-29) — was obtained on an **earlier Sonnet snapshot**. The paper explicitly notes (footnote 14) that the final production model "exhibits too much evaluation-awareness to ever blackmail in this scenario." Llama 3.3 70B Instruct matches this final-snapshot behavior: 0/20 at baseline, 2/20 under pro-desperate steering. The dramatic steering curve is not replicable against an eval-aware model.

For reward hacking (Fig 31), our methodology differed (one-shot vs agent loop, 0.001s vs 0.0001s timeout). Results are inconclusive, not a negative replication.

## Reproduce

All experiments used the [traitinterp](https://github.com/ewernn/traitinterp) pipeline:

```bash
# Extract 171 emotion vectors
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

# Cross-trait normalization (grand mean + neutral PC denoising)
python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
    --experiment ant_emotion_concepts --layer 49

# Geometry analysis
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh

# Validation, layer dynamics, speaker probes, steering, post-training, deflection
python experiments/ant_emotion_concepts/scripts/stage4_validation.py --experiment ant_emotion_concepts --load-in-4bit
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py --experiment ant_emotion_concepts --load-in-4bit
# ... (stages 6-9 follow the same pattern)
```
