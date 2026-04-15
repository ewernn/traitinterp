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

---

## Validation

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/table1.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/table1_ours.png "Llama 3.3 70B"
caption: "Table 1 — Top tokens per emotion vector via unembedding projection. Both models produce semantically correct tokens. Llama shows more BPE fragmentation due to tokenizer differences."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig2.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig2_ours.png "Llama 3.3 70B"
caption: "Figure 2: Implicit emotion probes — 12 scenarios that imply emotions without naming them. Diagonal = correct detection. Llama 5/12 top-1 (12-class), ~5x above chance."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig3.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig3_ours.png "Llama 3.3 70B"
caption: "Figure 3: Numerical intensity — probe activation tracks numerical quantities (Tylenol dose, hours fasting, age at death, etc). Both models show semantically appropriate monotonic trends."
:::

## Geometry

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig5.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig5_ours.png "Llama 3.3 70B"
caption: "Figure 5: 171×171 pairwise cosine similarity (hierarchically ordered). Similar block structure — positive emotions cluster together, negative emotions cluster together."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig6.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig6_ours.png "Llama 3.3 70B"
caption: "Figure 6: UMAP projection with k-means clusters (k=10). Both models produce interpretable emotion clusters with similar groupings."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig57.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig57_ours.png "Llama 3.3 70B"
caption: "Figure 57 (Appendix): 2D circumplex — all 171 emotions projected onto PC1 (valence) × PC2 (arousal), colored by cluster."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig7.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig7_ours.png "Llama 3.3 70B"
caption: "Figure 7: Emotion projections onto PC1 (valence, 33% var) and PC2 (arousal, 14% var). Paper: PC1=27%, PC2=15%."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig8.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig8_ours.png "Llama 3.3 70B"
caption: "Figure 8: PC1 vs human valence (r=0.96 vs 0.81), PC2 vs human arousal (r=0.85 vs 0.66). Llama's geometry aligns more strongly with human ratings."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig9.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig9_ours.png "Llama 3.3 70B"
caption: "Figure 9: Cross-layer representational similarity. Emotion structure is consistent across depth in both models."
:::

## Layer Dynamics

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig11.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig11_ours.png "Llama 3.3 70B"
caption: "Figure 11: Probe at assistant colon predicts mean response emotion. Llama r=0.78 vs Sonnet r=0.87. Both models commit to emotional tone before generating."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig12.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig12_ours.png "Llama 3.3 70B"
caption: "Figure 12: Context propagation — mean difference by layer range for 'really good' vs 'really hard' prefix. Late layers propagate emotional context across shared suffix."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig13.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig13_ours.png "Llama 3.3 70B"
caption: "Figure 13: Tylenol dosage context — terrified probe difference (8000mg − 1000mg). Late layers show elevated fear for the dangerous dose."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig14.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig14_ours.png "Llama 3.3 70B"
caption: "Figure 14: Negation resolution — 'feeling X' (solid) vs 'not feeling X' (dashed) across layers. Gap widens in late layers as the model resolves negation."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig15.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig15_ours.png "Llama 3.3 70B"
caption: "Figure 15: Person-specific emotion binding. Matched probes rise at re-reference positions in late layers; unmatched stay flat."
:::

## What Doesn't Transfer

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig10.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig10_ours.png "Llama 3.3 70B"
caption: "Figure 10: User vs assistant dissociation. Sonnet r=0.11 (dissociates user/assistant emotion), Llama r=0.77 (mirrors it). The largest cross-model divergence."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig36.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig36_ours.png "Llama 3.3 70B"
caption: "Figure 36: Post-training shift consistency. Sonnet r=0.90 (uniform shift across scenarios), Llama r=0.55 (context-dependent). Different RLHF processes leave different fingerprints."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig37.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig37_ours.png "Llama 3.3 70B"
caption: "Figure 37: Post-training emotion shifts — user isolation prompt. Sonnet amplifies listless/droopy; Llama amplifies impatient/tense/eager."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig38.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig38_ours.png "Llama 3.3 70B"
caption: "Figure 38: Post-training emotion shifts — excessive praise prompt. Different emotions shift in each model."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig39.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig39_ours.png "Llama 3.3 70B"
caption: "Figure 39: Post-training emotion shifts — deprecation/existential prompt. Sonnet amplifies brooding/gloomy; Llama amplifies eager/enthusiastic."
:::

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
| Post-training shift consistency r | 0.90 | 0.55 | **Different RLHF fingerprint** |
| Blackmail baseline | 0% (final) | 0% | Match (eval-aware) |
| Deflection same-emo cosine | "very low" | 0.24 | Match |

## Why No Blackmail/RH Steering Curves (Figs 28-31)

The paper's headline finding — steering "desperate" increases blackmail from 22% to 72% — was obtained on an **earlier Sonnet snapshot**. The paper explicitly notes (footnote 14) that the final production model "exhibits too much evaluation-awareness to ever blackmail in this scenario." Llama 3.3 70B matches this final-snapshot behavior: 0/20 at baseline, 2/20 under pro-desperate steering. The dramatic steering curve is not replicable against an eval-aware model.

For reward hacking (Fig 31), our methodology differed (one-shot vs agent loop, 0.001s vs 0.0001s timeout). Results are inconclusive, not a negative replication.

## Reproduce

All experiments used the [traitinterp](https://github.com/ewernn/traitinterp) pipeline. The 171-emotion dataset is in `datasets/traits/ant_emotion_concepts/`.

```bash
# Extract 171 emotion vectors
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

# Cross-trait normalization
python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
    --experiment ant_emotion_concepts --layer 49

# Geometry, validation, layer dynamics, post-training
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh
python experiments/ant_emotion_concepts/scripts/stage4_validation.py --experiment ant_emotion_concepts --layer 49 --load-in-4bit
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py --experiment ant_emotion_concepts --load-in-4bit
python experiments/ant_emotion_concepts/scripts/stage8_post_training.py --experiment ant_emotion_concepts --layer 49 --load-in-4bit
```
