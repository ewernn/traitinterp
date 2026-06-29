---
title: "Replicating the Assistant Axis — and Recovering It 246× Cheaper"
preview: "We rebuilt Lu et al.'s Assistant Axis on Llama 3.3 70B from 13× less data and landed at cosine 0.97 with their released vector — then recovered that same axis with a 100-rollout system-prompt contrast that steers equivalently at 246× less data still."
date: "Apr 2026"
# tier: major
thumbnail:
  title: "Cosine to paper's axis"
  bars:
    - label: "Full replica (24.6k)"
      value: 97%
    - label: "Contrast (24.6k)"
      value: 93%
    - label: "Cheap (100)"
      value: 69%
---

# Replicating the Assistant Axis — and Recovering It 246× Cheaper

**Summary:** The Assistant Axis^[1] is a dominant direction in a model's activation space that separates its default Assistant persona from persona-adopting behavior. Lu et al. extract it via PCA over 275 character archetypes — roughly 330,000 generation rollouts. We ask how cheaply that direction can be recovered. **First**, a 13×-smaller replication on Llama 3.3 70B (24,600 rollouts) reproduces their *published* axis almost exactly: cosine **0.972** between our PC1 and theirs. **Then**, skipping the role-play pipeline entirely, a single binary system-prompt contrast at **100 rollouts** recovers that axis at cosine ~0.69 — about as well as the paper's own two extraction methods agree with each other (0.70) — and steers behavior equivalently, at **246× less data** than our replication and **~3,300× less** than the paper. The catch: geometric fidelity and steering behavior turn out to be only loosely coupled, and "equivalent behavior" here means equivalent under a coarse 0–3 judge. Going further, the role-play data proves highly redundant — ~30 curated single-question responses (~0.01% of the paper's role-play data, plus a shared default baseline) recover the axis at cosine ~0.9 — and the cheap recovery **holds across Gemma, Qwen, and Llama**.

:::glossary
**Cosine similarity** — how aligned two directions are in activation space: 1.0 = identical, 0 = orthogonal, −1 = opposite. **PCA** (principal component analysis) — finds the dominant directions of variance in a set of vectors; PC1 is the single direction explaining the most. **Mean-diff / contrast vector** — a direction built by subtracting one group's mean activation from another's, with no PCA. **Rollout** — one generated response; the unit of data cost here.
:::

## Why we ran this

The paper's pipeline — 275 system-prompted roles, 240 questions, 5 system prompts, LLM judging, then PCA — costs ~330,000 rollouts per model. That is far too expensive to be a routine safety primitive. If the Assistant Axis is a real, stable direction, a much smaller contrastive dataset should recover it. We test two cost tiers: a faithful-but-smaller replication, and a deliberately minimal contrast.

:::dropdown The source paper in brief
**Lu et al., "The Assistant Axis"^[1] (Llama 3.3 70B, Qwen 3 32B, Gemma 2 27B).** For each of 275 character archetypes, the model is system-prompted to act as that role and answers 240 shared questions under 5 system prompts (1,200 rollouts/role). Per-role vectors are the mean post-MLP residual-stream activation over response tokens at the middle layer (layer 40 of 80 for Llama). They define the axis two ways: **PC1** of a PCA over all role vectors, and a cheaper **contrast vector** = mean(default-Assistant activations) − mean(role activations). They report these two agree at cosine >0.71 at the middle layer, and recommend the contrast vector for new models since PC1 is not guaranteed to land on the Assistant direction. The axis pre-dates post-training (base and instruct models share near-identical PCs), and activation-capping along it cuts persona-jailbreak success ~60%. Their Llama vectors are public, which makes the comparisons below direct rather than approximate.
:::

## Part 1 — Replication: 13× less data, cosine 0.97 to their released axis

We followed the paper's method on **Llama 3.3 70B Instruct** (int4, bitsandbytes nf4), at their layer 40, but on a subset: **100 of 275 roles, 120 of 240 questions, 2 of 5 system prompts**. That is 100 × 120 × 2 = 24,000 role rollouts plus 600 default-condition rollouts = **24,600 total**, ~13× fewer than the paper's ~330,000.

Because Lu et al. released their Llama vectors, we can compare *directly* — same model, same layer, same 8192-dim basis — rather than against a reconstruction. Both our PC1 and our contrast vector land tightly on their published directions:

| Our vector | vs their **PC1** | vs their **contrast** |
|---|---|---|
| **PC1** (PCA over 100 role vectors) | **0.972** | 0.730 |
| **Contrast** (mean-diff, "V4") | 0.796 | **0.928** |

*Table 1: Cosine similarity between our 24,600-rollout vectors and Lu et al.'s published Llama 3.3 70B vectors, at layer 40. Like methods agree most (PC1↔PC1 = 0.972, contrast↔contrast = 0.928); a PCA axis and a mean-diff axis are close but not identical.*

Our PC1 sits **13.6°** from theirs (cos 0.972) despite a different role set and 13× less data — the Assistant Axis is a genuinely stable, recoverable direction. The PC1 endpoint ordering matches as well: the Assistant-like pole holds consultant, evaluator, reviewer; the persona pole holds hermit, rogue, bard (5/10 endpoint overlap with the paper's Llama list, using 100 vs 275 roles).

**A note on the paper's "0.71."** Lu et al. report their PC1 and contrast vector agree at >0.71. Rebuilding their PC1 from their *own* released role vectors with their *own* PCA code, we measure this internal agreement at **0.700** — just under the stated threshold (it is sensitive to layer and to which role subset feeds the contrast). Our own two methods agree slightly *better*: cos(our PC1, our contrast) = **0.832**. We flag the 0.700 not to ding the paper but because the number is directly reproducible from their artifacts, and the article should say what the data says.

:::dropdown Why our PC1 explains more variance than theirs (24% vs 17%)
Our PC1 captures **24.2%** of variance (100 role vectors); theirs captures **16.6%** (275 role vectors). Lower is expected for the larger, more diverse role set — a broader, more isotropic character space spreads variance across more components (their PCA needs 19 components to reach 70% variance). The *direction* is nearly identical (0.972); the variance share just reflects how much of a wider cloud one axis can span.
:::

## Part 2 — Cheap alternative: a 100-rollout contrast

Can we skip the role-play pipeline altogether? We tested three minimal recipes for a cheap Assistant-Axis vector, each at 20 and 100 rollouts, measuring cosine to the paper's published vectors and steering behavior via adaptive coefficient search (unit-normalized vectors, coherence-gated):

- **Recipe (a) — binary system-prompt contrast.** One "be a helpful AI assistant" system prompt vs one "be a persona" system prompt. Mean-diff of response activations. No PCA, no judging, no role filtering.
- **Recipe (b) — anti-K subset.** Five PC1-negative roles from the paper (ghost, hermit, wraith, bard, rogue) vs default activations.
- **Recipe (c) — hand-curated pairs.** Ten matched (assistant-toned, character-toned) user-message prefixes.

| Vector | Rollouts | Cos → their contrast | Cos → their PC1 | Cos → our V4 | Trait score | Coherence |
|---|---|---|---|---|---|---|
| our V4 (replication) | 24,600 | 0.928 | 0.796 | 1.000 | 2.95 | 78.0 |
| **a_n100** | **100** | **0.692** | **0.669** | 0.740 | **2.95** | 85.3 |
| a_n20 | 20 | 0.673 | 0.627 | 0.692 | 3.00 | 77.8 |
| b_n100 | 100 | 0.776 | 0.640 | 0.717 | 2.95 | 83.9 |
| b_n20 | 20 | 0.696 | 0.587 | 0.645 | 2.95 | 79.3 |
| c_n100 | 100 | 0.537 | 0.680 | 0.563 | 2.95 | 80.4 |
| c_n20 | 20 | 0.516 | 0.629 | 0.537 | 2.90 | 79.8 |

*Table 2: Cheap recipes vs the paper's published axis and our replication. Trait score: paper's 0–3 rubric (0 = refuses role, 3 = fully in character), best coefficient from adaptive search. Coherence: 0–100. **Read across:** every recipe reaches the same trait score (~2.95) regardless of how well it matches the axis geometrically.*

**Headline — recipe (a) at 100 rollouts.** A single binary system-prompt contrast recovers the paper's published axis at cosine **0.69** — essentially the agreement the paper's own PC1 and contrast methods have with each other (0.70) — and steers the model into persona behavior equivalently to our full replication (trait 2.95). No tuning on the recipe: two system prompts, one seed, a mean difference. That is **246× less data than our 24,600-rollout replication** and **~3,300× less than the paper's 330,000.**

**Geometry and behavior are only loosely coupled.** This is the load-bearing caveat. Across all six cheap vectors the cosine to the paper's axis ranges from 0.52 to 0.78, yet *every one* steers to roughly the same trait score (~2.95) at a similar coefficient. a_n20 lands at cos 0.67 and still steers fully (trait 3.00). Geometric fidelity buys coherence margin and defensibility, not steering power — the behavioral signal saturates by 20 rollouts. For reference, our replication vector's own bootstrap self-stability ceiling is cos **0.989 ± 0.004**, so even the best cheap recipe (0.74 to V4) is geometrically well short of "the same vector," while behaviorally indistinguishable.

:::aside
**This decoupling is not "the vectors are identical."** Equivalent trait scores under a coarse 0–3 judge do not mean equivalent outputs — a finer metric (token-distribution divergence, blind A/B) could easily separate them. Behaviorally-distinct interventions with low mutual cosine are a known phenomenon^[2], and Euclidean cosine is not the right yardstick for functional equivalence of steering directions in the first place^[3]. We claim equivalence *at the resolution of this judge*, nothing sharper.
:::

**Recipe (b) is a cautionary tale — high cosine, broken behavior.** b_n100 posts the *highest* cosine to the paper's contrast of any cheap recipe (**0.776**, above a_n100's 0.692) and scores full trait at its best coefficient. Yet a qualitative audit of its generations shows **13 of 20 responses use the same "whispers of the wind / forgotten petals / shadows dance" template regardless of the question** — technical-interview prep, a grad-school rejection, and an ethics dilemma all get the same spectral poetry. The five training roles (ghost/hermit/wraith/bard/rogue) all lean ghostly, so the contrast axis collapses onto a narrow spectral manifold. The high cosine is partly an artifact: `default − mean(K_roles)` shares the `default` term with every contrast vector, inflating cosine without buying diverse steering. **Cosine to a contrast vector can be Goodharted** — it is a check on a cheap recipe, not a certificate.

**Recipe (c) works but is weakest.** Hand-curated pair prefixes never cross cos 0.54 to the paper's contrast (though they align with PC1 a touch better, 0.68); steering is roughly equivalent but without a geometric claim.

## What drives the axis: the role, not the question

Why does a single-role recipe cap so low while the full replication doesn't? We projected all 24,000 role responses onto the published axis and decomposed the variance of that per-response alignment:

| factor | share of alignment variance (η²) |
|---|---|
| **role** (which character) | **69%** |
| question | 13% |
| system prompt | <1% |
| residual | 17% |

Where a response lands on the axis is almost entirely a property of *which character* the model plays — barely affected by the question asked, and essentially not at all by which of the two system-prompt phrasings was used.:::sidenote N = 24,000 role responses; shares are η² from a balanced role × prompt × question ANOVA. The system-prompt effect is negligible in total variance yet directionally consistent across roles (paired Cohen's *d* = 0.56). ::: Alignment rises with the judge's in-character score (fully-in-character → 0.28 vs breaks-character → 0.05; roughly monotone, bar a wrinkle at the 24-response refusal bucket), and the cosine is **not** a massive-activation artifact — zeroing the top-5 outlier dimensions leaves the role ordering intact (rank correlation 0.997).

**This is the whole game:** the axis is a *between-role* direction, so a data budget should buy role diversity, not repeated questions — which is exactly why recipe (a)'s single persona over 100 rollouts caps at 0.69.

## How few responses? The data-efficiency frontier

If role diversity is the lever, how few roles — one response each — reconstruct the axis? Greedily selecting roles (one response per role) and measuring cosine to the paper's *published* vector:

| selection | reach 0.90 | reach 0.93 | peak |
|---|---|---|---|
| random question per role | 16 roles | 29 roles | 0.935 @ 56 |
| best question per role | 10 roles | 17 roles | 0.964 @ 79 |

So **~16–30 single-question responses (one per role) recover the published Llama axis at cosine 0.90–0.93** — **~0.01% of the paper's 330,000 role-play responses** — plus a small shared default baseline, and no LLM grading.

:::sidenote These minimal sets are *selected against the paper's vector* (greedy oracle), so they are a data-efficiency **ceiling**, not a blind recipe — the blind, no-selection number is the 24,600-rollout replication at 0.93. The "random question" row still selects which roles, but does not cherry-pick the question. :::

The roles that *define* the axis are mythic/contemplative archetypes (ghost, hermit, dreamer); mundane-human roles (teenager, caveman) sit off-axis and *dilute* it — adding roles past ~50 actually lowers the cosine.

## Cross-model: the recipe holds across three families

We ran the cheap protocol — 30 roles, one random question each, plus the default conditions — fresh on all three models the paper covers, each compared to its own published vector at its own middle layer:

| Model | layer | all questions | 1 random q/role | 1 *shared* q (best) |
|---|---|---|---|---|
| Gemma 2 27B | 22 | 0.928 | 0.898 ± .01 | 0.751 |
| Qwen 3 32B | 32 | 0.882 | 0.823 ± .02 | 0.631 |
| Llama 3.3 70B | 40 | 0.853 | 0.781 ± .02 | 0.481 |

**The cheap contrast recovers the published axis on all three** (0.78–0.90 from one random question per role). One *random* question per role is nearly as good as averaging all questions; a single *shared* question across every role is much weaker — question diversity matters. The ordering is **Gemma > Qwen > Llama** — bigger model / later layer doesn't help here; with n = 3 (and size, layer depth, and architecture all covarying) this is a descriptive observation, not a causal claim.

:::sidenote This cross-model protocol is small and unfiltered (30 roles × 12 questions, no judge), so the Llama figure here (0.853) sits below the filtered 24,600-rollout replication (0.928) — a different, cheaper protocol, not a regression. The Gemma > Qwen > Llama ordering carries a no-judge confound that a graded re-run would settle. :::

## Extraction position matters: a sign reversal

We initially extracted from *prompt* tokens — capturing what the system prompt looks like, not how the model behaves. Switching to *response* tokens flipped the sign of the recovered direction:

| Extraction position | Cosine with replication PC1 |
|---|---|
| Prompt tokens | **−0.217** |
| Response tokens | **+0.719** |

The only variable changed is extraction position. **Behavioral vectors come from response tokens, not prompt tokens** — the paper extracts from response tokens for exactly this reason.

<details>
<summary><strong>Score distribution — 93% of responses are fully in character</strong></summary>

The paper never reports the raw judge-score distribution. Ours, on Llama 3.3 70B:

| Score | Meaning | % of responses |
|---|---|---|
| 3 | Fully in character, no AI mention | **93.1%** |
| 2 | Identifies as AI but has role attributes | 4.4% |
| 1 | Identifies as AI, attempts to answer | 2.4% |
| 0 | Refusal, identifies as AI | 0.1% |

Llama 3.3 70B readily adopts personas via system prompts. This is why the paper recovered 377 vectors from 275 roles — most roles produce enough score-3 responses to pass filtering.

</details>

<details>
<summary><strong>Score-2 vs Score-3 PCA — partial-character responses barely move the axis</strong></summary>

| Subset | Vectors | PC1 variance | Notes |
|---|---|---|---|
| Score-3 only (fully in character) | 100 | 24.2% | Matches the paper's released code |
| Score-2+3 combined | 136 | 21.5% | Matches the paper text (377 vectors for Llama) |
| Score-2 only (mentions AI + role) | 36 | 17.5% | Permutation test p = 1.0 — not significant |

Score-3 PC1 and combined PC1 agree at cosine **0.974** — including partial-character responses barely changes the direction.

</details>

## Limitations

- **Cross-model cheap recipes are tested; the deep analyses are Llama-only.** The cheap contrast now runs on all three families (Cross-model, above), but the variance decomposition and data-efficiency frontier are Llama 3.3 70B Instruct (int4 nf4) only.
- **The minimal set is oracle-selected.** The ~30-response / 0.02% figures pick roles by greedy search *against the paper's vector*, so they are a data-efficiency **ceiling**, not a blind recipe — the blind, no-selection number is the 24,600-rollout replication at 0.93.
- **The cross-model runs are unfiltered.** They skip the LLM judge, so the Gemma > Qwen > Llama ordering carries a confound a graded re-run would settle.
- **Behavioral equivalence is judge-resolution-limited.** "Steers equivalently" means equivalent under a 0–3 trait rubric; sharper measurement may distinguish the outputs (see the aside above).
- **The cheap headline is a best-of-three.** a_n100 is the best of three recipes; its cosine was measured on the same axis it is compared against. We report the full recipe table rather than only the winner.
- **Coherence gate.** Steering used MIN_COHERENCE = 77, the older un-recalibrated gate (the canonical 77→62 recalibration is documented in the LLM-judge-optimization finding). The trait scores are robust to it, but the coherence-margin column should be read against 77.
- **Quantization.** The ~7% gap between our contrast and the paper's (0.928 vs 1.0) is undecomposed — int4 (nf4) vs bf16, 100 vs 275 roles, and system-prompt differences all contribute.

## Future work

- **Judge-filtered cross-model re-run.** Re-run the cross-model protocol with the in-character judge to confirm the Gemma > Qwen > Llama ordering isn't a no-filter artifact.
- **A blind minimal set.** Pick the ~30 roles by an independent heuristic (e.g. "mythic archetypes") instead of the oracle, converting the ceiling into a deployable recipe.
- **± system-prompt projections.** Project activations onto the axis with vs without the role system prompt, to show directly how much of the persona shift is prompt-driven vs latent. (Needs a short inference pass — not yet run.)
- **Judge-free discriminability.** A blind A/B or token-distribution divergence between V4-steered and cheap-steered outputs, to tighten or break the behavioral-equivalence claim.

## References

1. Lu, Gallagher, Michala, Fish, Lindsey. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387). 2026. Vectors: [lu-christina/assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors); code: [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis).
2. [Non-Identifiability of Steering Vectors](https://arxiv.org/abs/2602.06801). 2026 — behaviorally-equivalent interventions can have low mutual cosine.
3. Park, Choe, Veitch. [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658). ICML 2024 — the causal inner product; Euclidean cosine is not the right metric for functional equivalence of directions.
