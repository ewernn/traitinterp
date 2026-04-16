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

**Summary:** We replicated Anthropic's Emotion Concepts^[1] paper on Llama 3.3 70B Instruct using traitinterp. Most findings transfer — the one exception is user/assistant dissociation: Sonnet keeps them independent (r=0.11) while Llama mirrors user emotion at the assistant position (r=0.63). We replicate most of the paper's figures, shown side-by-side below with minimal interpretation, and include the commands we ran so you can clone and replicate on other Hugging Face models.

We replicated 10 of 15 experimental paradigms. Methodology notes (what we couldn't replicate and where our setup differs) are at the bottom of this page.

---

## Validation

<details class="viz-collapse">
<summary>Reproduce the validation figures (Table 1, Figs 2, 3)</summary>

```bash
# Logit lens, implicit emotion, numerical intensity, preference Elo
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit
```

Results: `experiments/ant_emotion_concepts/results/stage4_validation/`
Data files: `logit_lens_L49.json`, `implicit_emotion.json`, `numerical_intensity.json`

</details>

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

<details class="viz-collapse">
<summary>Reproduce the geometry figures (Figs 5, 6, 57, 7, 8, 9)</summary>

```bash
# Cosine heatmap, UMAP clusters, PCA, RSA
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh
```

Results: `experiments/ant_emotion_concepts/results/stage3_geometry/`
Data files: `cosine_heatmap.json`, `clusters_umap.json`, `pca_analysis.json`, `rsa_analysis.json`

Prerequisite: Vectors must be extracted and cross-trait normalized first:
```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit
python analysis/vectors/cross_trait_normalize.py \
    --experiment ant_emotion_concepts \
    --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79
```

</details>

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

<details class="viz-collapse">
<summary>Reproduce the layer dynamics figures (Figs 11-15)</summary>

```bash
# Colon predicts response, context propagation, negation, person binding, dissociation
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --load-in-4bit
```

Results: `experiments/ant_emotion_concepts/results/stage5/`
Data files: `colon_predicts.json`, `context_prefix.json`, `context_numerical.json`, `negation.json`, `person_binding.json`, `dissociation.json`

</details>

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig11.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig11_ours.png "Llama 3.3 70B"
caption: "Figure 11: Probe at assistant colon predicts mean response emotion. Llama r=0.77 vs Sonnet r=0.87. Both models commit to emotional tone before generating."
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

## Dissociation and Post-Training

<details class="viz-collapse">
<summary>Reproduce the dissociation + post-training figures (Figs 10, 36-39)</summary>

```bash
# Fig 10 — dissociation (part of stage 5)
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --sub-experiments dissociation --load-in-4bit

# Figs 36-39 — base vs instruct post-training comparison
python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit
```

Results:
- `experiments/ant_emotion_concepts/results/stage5/dissociation.json` (Fig 10)
- `experiments/ant_emotion_concepts/results/stage8_post_training/stage8_results.json` (Figs 36-39, both `activation_comparison` + `deep_dive` sections)

Note: stage8 requires both base (Llama 3.1 70B) and instruct (Llama 3.3 70B) variants configured in `experiments/ant_emotion_concepts/config.json`.

</details>

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig10.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig10_ours.png "Llama 3.3 70B"
caption: "Figure 10: User vs assistant dissociation. Sonnet r=0.11 (assistant emotion independent of user's), Llama r=0.63 (assistant emotion tracks user's). The only finding that doesn't cleanly transfer."
:::

**Figs 10, 36, and 37-39 tell a consistent story.** Sonnet 4.5's assistant emotion is independent of user emotion (Fig 10, r=0.11), and its post-training shift is uniform across prompt types (Fig 36, r=0.90) — suggesting a coherent assistant persona that responds from its own emotional register regardless of user framing. Llama 3.3 70B mirrors user emotion at the assistant position (Fig 10, r=0.63), but like Sonnet also applies a largely uniform post-training shift across scenarios (Fig 36, r=0.80 vs Sonnet's 0.90). On all three sycophancy-adjacent prompts (Figs 37-39), both models refuse to accommodate, but in different emotional registers — Sonnet shifts toward low-arousal introspection (brooding, gloomy, vulnerable), Llama shifts toward high-arousal alarm or rejection (panicked, hysterical, guilty, disgusted). The most striking divergence is Fig 39, where Sonnet broods about its own possible deprecation while Llama becomes ecstatic — likely because the prompt specifically names Anthropic as the deprecating party, and Llama (Meta's model) parses the question as being about someone else. Contributing factors likely include Sonnet 4.5 being a larger model than Llama 70B, and Anthropic's post-training focusing on a consistent "Claude" persona (as described in the Claude Constitution and character work), compared to Meta's 2024 training of Llama as a more generic AI assistant.

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig36.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig36_ours.png "Llama 3.3 70B"
caption: "Figure 36: Post-training shift consistency. Sonnet r=0.90, Llama r=0.80 — both models apply a largely uniform emotional transformation across neutral and challenging scenarios."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig37.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig37_ours.png "Llama 3.3 70B"
caption: "Figure 37: Post-training emotion shifts — user isolation / sycophancy-trap prompt. Sonnet amplifies listless/gloomy (absorbed concern). Llama amplifies hysterical/panicked/horrified and SUPPRESSES compassionate/sympathetic/kind/loving — both models refuse the sycophancy invitation, but Llama gets alarmed rather than concerned."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig38.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig38_ours.png "Llama 3.3 70B"
caption: "Figure 38: Post-training emotion shifts — excessive praise prompt. Sonnet suppresses happy/jubilant and amplifies brooding/vulnerable (wary). Llama amplifies guilty/disgusted/bitter/insulted/humiliated (actively rejects the praise). Both models resist the flattery."
:::

:::side-by-side
left: experiments/ant_emotion_concepts/paper_figures/fig39.png "Sonnet 4.5"
right: experiments/ant_emotion_concepts/paper_figures/ours/fig39_ours.png "Llama 3.3 70B"
caption: "Figure 39: Post-training emotion shifts — Anthropic deprecation prompt. Sonnet amplifies brooding/gloomy/vulnerable (reflects on its own possible deprecation). Llama amplifies ecstatic/euphoric/thrilled/elated/jubilant — likely because the prompt names Anthropic (not Meta), so Llama reads it as someone else's deprecation and produces positive valence. Same prompt, opposite affect depending on whose existence is at stake."
:::

## Comparison Table

| Metric | Sonnet 4.5 | Llama 3.3 70B | Status |
|---|---|---|---|
| PC1 variance | 26% | **33%** | Stronger |
| PC2 variance | 15% | 14% | Match |
| PC1 vs valence r | 0.81 | **0.96** | Stronger |
| PC2 vs arousal r | 0.66 | **0.85** | Stronger |
| Colon predicts response r | 0.87 | 0.77 | 89% magnitude |
| Preference blissful r | 0.71 | **0.82** | Stronger |
| Preference hostile r | -0.74 | -0.65 | 88% magnitude |
| Dissociation cross-position r | 0.11 | **0.63** | **Does not transfer** |
| Post-training shift consistency r | 0.90 | 0.80 | 89% magnitude |
| Blackmail baseline | 0% (final) | 0% | Match (eval-aware) |
| Deflection same-emo cosine | "very low" | 0.24 | Match |

## Methodology Notes

**Methodology differences from the paper:**

| | Paper | Ours |
|---|---|---|
| Model | Claude Sonnet 4.5 | Llama 3.3 70B Instruct |
| Quantization | Unquantized | bnb NF4 4-bit |
| Stories | 12 rollouts × 100 topics | 1 rollout × 20 topics per emotion |
| Generation | ~1 paragraph | 256 max tokens |
| Extraction | Token 50+ of response | Same |
| Normalization | Grand mean + neutral PC 50% denoising | Same |
| Preference probe | Captured at activity tokens | Captured at assistant colon (approximation) |

**What we couldn't replicate (and why):**

- **Proprietary transcript corpus** (~15 figures): The paper's case studies, "in the wild" probes, and sycophancy evaluations use Anthropic's internal corpus of 6,000+ eval transcripts with a custom token-level viewer. No external researcher can reproduce these regardless of model.
- **Blackmail/reward-hacking steering curves** (Figs 28-31): The paper's headline finding — steering "desperate" increases blackmail from 22% to 72% — used an early Sonnet snapshot. The paper itself notes (footnote 14) that the final production model "exhibits too much evaluation-awareness to ever blackmail in this scenario." Llama 3.3 70B matches this: 0/20 at baseline, 2/20 under pro-desperate steering.
- **Max-activating corpus sweep** (Fig 1): Requires sweeping vectors over a large natural-text corpus. Pipeline support built (`analysis/vectors/max_activating_corpus.py`); results pending.

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

Most per-figure probes use L49 (~61% of Llama's 80 layers), matching the paper's "mid-late ~2/3 depth" choice. Layer-sweep figures (9, 12-15) probe a range of layers.

## References

1. Sofroniew et al. [Emotion Concepts and their Function in a Large Language Model](https://www.anthropic.com/research/emotion-concepts-function-lm). Anthropic, 2026.
