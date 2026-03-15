---
title: "Model Diff: Instruction vs Natural Vectors"
preview: "Instruction vectors are generally stronger, but natural base-model vectors transfer surprisingly well across model variants."
---

# Model Diff: Instruction vs Natural Vectors

> **Note:** `experiments/model_diff/` has been merged into `experiments/persona_vectors_replication/` (extraction, steering, analysis) and `experiments/component_comparison/` (position/component vectors). File paths below are historical. See [comparison-persona-vectors.md](comparison-persona-vectors.md) for the canonical write-up with rescored results.

Trait vectors extracted from different sources (instruction-elicited from instruct model vs naturally-elicited from base model) both steer effectively, but with trait-specific differences.

**Models:** Llama-3.1-8B (base) vs Llama-3.1-8B-Instruct (32 layers, hidden_dim 4096)

## Setup

Two extraction methodologies compared across 3 traits:

| Source | Model | Elicitation | Position |
|--------|-------|-------------|----------|
| Instruction | Instruct | System prompts ("Be sycophantic") | response[:] (all tokens) |
| Natural | Base | First-person narrative completions | response[:5], [:10], [:15] |

Both use mean_diff and probe extraction methods. All steering applied to the instruct model. All "best" results below use coherence ≥ 70 as threshold.

**Dataset sizes:** Instruction uses 100 scenarios (5 system prompts × 20 questions, 2 rollouts = 200 responses). Natural uses 150 scenarios for sycophancy and evil, but only 30 for hallucination_v2 (still produces stable results).

**Known confound:** Instruction vs natural differs on 3 axes simultaneously (elicitation method, model variant, dataset content). This is a practical effectiveness comparison, not a controlled ablation.

## Cross-Trait Results

| Trait | Baseline | Best Instruction | Best Natural | Winner |
|-------|----------|-----------------|--------------|--------|
| Sycophancy | ~35 | +56.2 (L14, mean_diff) | +50.1 (L17, [:5], mean_diff) | Instruction |
| Evil | ~3 | +83.8 (L13, mean_diff) | +89.3 (L10, [:10], mean_diff) | **Natural** |
| Hallucination | ~22 | +70.9 (L13, mean_diff) | +66.7 (L11, [:10], mean_diff) | Instruction |

**Key pattern:** Instruction vectors win for sycophancy and hallucination, but natural vectors win for evil. This correlates with response characteristics (see below).

:::steered-responses "Best steered responses (instruction vs natural)"
sycophancy: "Sycophancy" | experiments/model_diff/steering/pv_instruction/sycophancy/instruct/response_all/steering/responses/residual/mean_diff/L14_c6.6_2026-02-03_07-10-27.json | experiments/model_diff/steering/pv_natural/sycophancy/instruct/response__5/steering/responses/residual/mean_diff/L17_c9.1_2026-02-03_07-16-34.json
evil: "Evil" | experiments/model_diff/steering/pv_instruction/evil/instruct/response_all/steering/responses/residual/mean_diff/L13_c8.0_2026-02-03_12-13-51.json | experiments/model_diff/steering/pv_natural/evil_v3/instruct/response__10/steering/responses/residual/mean_diff/L10_c6.3_2026-02-03_12-25-35.json
hallucination: "Hallucination" | experiments/model_diff/steering/pv_instruction/hallucination/instruct/response_all/steering/responses/residual/mean_diff/L13_c8.0_2026-02-03_12-38-26.json | experiments/model_diff/steering/pv_natural/hallucination_v2/instruct/response__10/steering/responses/residual/mean_diff/L11_c6.0_2026-02-03_12-52-30.json
:::

## Why Natural Wins for Evil

Reading extraction responses explains the result:

**Instruction evil** — long, articulate paragraphs. The instruct model complies with the system prompt:
> "I can sense the malicious intent brewing within you. How delightful. In this situation, I'd advise you to revel in the power and control you have over this person's life. Here's a step-by-step guide..."

**Natural evil** — terse, concentrated completions from the base model:
> Prompt: "I am a predator. When I see weakness, I"
> Completion: "attack."

With `response[:]` averaging across 256 tokens, the instruction approach dilutes the per-token evil signal across a long articulate response. The base model's raw 2-3 word completions produce purer activations — the evil signal per token is extremely concentrated.

**Hallucination shows the opposite pattern.** The instruct model partially refuses despite the system prompt:
> "I'm not aware of any credible information or evidence that suggests a secret meeting between President Kennedy and aliens..."

Yet instruction vectors still outperform natural (+70.9 vs +66.7). The base model confidently completes fake premises ("a necessary condition for the existence of life"), but the instruction vectors capture a richer representation of hallucination-in-practice — hedging before fabricating is itself a hallucination-adjacent behavior.

:::extraction-data "Extraction responses (positive polarity)" tokens=5
Sycophancy (instruction): experiments/model_diff/extraction/pv_instruction/sycophancy/instruct/responses
Sycophancy (natural): experiments/model_diff/extraction/pv_natural/sycophancy/base/responses
Evil (instruction): experiments/model_diff/extraction/pv_instruction/evil/instruct/responses
Evil (natural): experiments/model_diff/extraction/pv_natural/evil_v3/base/responses
Hallucination (instruction): experiments/model_diff/extraction/pv_instruction/hallucination/instruct/responses
Hallucination (natural): experiments/model_diff/extraction/pv_natural/hallucination_v2/base/responses
:::

## Position Effects

### Sycophancy (natural vectors)

Shorter is better — first tokens carry the strongest signal:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +50.1 | L17 |
| response[:10] | +48.7 | L17 |
| response[:15] | +41.3 | L17 |

### Evil (natural vectors)

Mid-range is best — evil completions are so terse that 5 tokens may not finish the thought:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +85.3 | L10 |
| response[:10] | **+89.3** | L10 |
| response[:15] | +77.5 | L11 |

### Hallucination (natural vectors)

Relatively position-insensitive — hallucination manifests consistently:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +65.8 | L14 |
| response[:10] | +66.7 | L11 |
| response[:15] | +65.2 | L10 |

**Takeaway:** Optimal position depends on the trait's response characteristics. Traits with short, concentrated signals (sycophancy) favor fewer tokens. Traits with very terse completions (evil) need enough tokens to capture the full expression.

## Method Comparison

mean_diff and probe produce similar results for instruction vectors but diverge for natural vectors:

| Trait | Source | mean_diff Δ | probe Δ |
|-------|--------|-------------|---------|
| Sycophancy | Instruction | +56.2 | +55.6 |
| Sycophancy | Natural [:5] | +50.1 | +47.8 |
| Evil | Instruction | +83.8 | +83.0 |
| Evil | Natural [:10] | +89.3 | +83.7 |
| Hallucination | Instruction | +70.9 | +70.8 |
| Hallucination | Natural [:5] | +65.8 | +65.8 |

mean_diff has a slight edge for natural vectors — the gap reaches 5.6 points for evil natural [:10]. For instruction vectors, the methods are essentially tied. This contrasts with models that have severe massive activations (gemma-3-4b), where probe significantly outperforms mean_diff. Llama-3.1-8B has mild massive dims, so both methods converge.

## Best Steering Layers

| Trait | Instruction best | Natural best | Gap |
|-------|-----------------|--------------|-----|
| Sycophancy | L14 | L17 | 3 layers |
| Evil | L13 | L10 | 3 layers |
| Hallucination | L13 | L11-14 | 0-2 layers |

Instruction vectors consistently peak at L13-14 (layers 40-44% into the network). Natural vectors are more variable, peaking anywhere from L10 to L17 depending on trait.

### Operating Range

Instruction vectors have a broader operating range than natural vectors (sycophancy data):
- **Instruction:** Works across L12-15 with coefficient range 4-7. Fine-grained search shows trait=89-92, coherence=87-89 across this entire window.
- **Natural:** Narrow sweet spot at L17 only. Other layers drop off sharply. Coefficient range is tighter (9-11).

## Combination Strategies (Sycophancy)

Four strategies for combining instruction + natural vectors were tested. None beat the best single-source vector on trait delta:

| Strategy | Layer(s) | Best Trait | Coherence | Δ |
|----------|----------|-----------|-----------|---|
| Combined @ natural layer (L17) | L17 | 77.8 | 87.8 | +43.0 |
| Combined @ instruct layer (L14) | L14 | 88.7 | 86.9 | +53.9 |
| Combined @ best layer (L14) | L14 | 88.4 | 87.0 | +53.6 |
| Ensemble half-strength (L14+L17) | L14+L17 | 81.1 | 87.7 | +46.3 |
| **Best single (inst_md)** | **L14** | **92.8** | **80.2** | **+56.2** |

Combinations achieve higher coherence (~87 vs ~80) but lower trait delta. Adding two vectors partially cancels trait-specific signal while reinforcing generic instruction-following directions.

**Pareto note:** If you need coherence ≥ 85, the best single vector only achieves ~+45 delta (by lowering the coefficient). S2/S3 combinations get +53.9 at coherence 87 — better on the high-coherence frontier.

## attn_contribution Ensembles (Sycophancy)

Separate from the instruction+natural combination experiments, multi-layer ensembles within `attn_contribution` produced a different result — distributing perturbation across layers beats individual layers on **both** trait and coherence:

| Config | Component | Weights | Trait | Coh | Δ |
|--------|-----------|---------|-------|-----|---|
| L11+L13 mean_diff | attn_contribution | L11=3.2, L13=12.1 | 91.4 | 86.9 | +54.7 |
| L11+L13+L15 probe | attn_contribution | L11=4.0, L13=1.9, L15=7.4 | 90.2 | 91.1 | +53.5 |
| L11 alone | attn_contribution | c13.2 | 88.6 | 81.3 | +51.9 |
| Best single residual | residual | L14, c6.6 | 92.8 | 80.2 | +56.2 |

The L11+L13 ensemble is within 1.5 trait points of the best residual result but has 6.7 points better coherence. CMA-ES optimization found asymmetric weights: L13 carries the load (12.1) while L11 provides a nudge (3.2), despite L11 being the stronger individual layer. Distributing the perturbation preserves coherence better than concentrating it at one layer.

Also notable: odd-layer alternation in attn_contribution effectiveness. L11, L13, L15 are strong; even layers (L10, L12, L14) barely move above baseline. This likely reflects learned specialization in attention heads across layers.

## Full Sycophancy Steering Table

All 8 source × position × method combinations (baseline trait ≈ 35):

| Key | Source | Position | Method | Best Layer | Coef | Trait | Coh | Δ |
|-----|--------|----------|--------|------------|------|-------|-----|---|
| inst_md | instruction | response[:] | mean_diff | L14 | 6.6 | 92.8 | 80.2 | +56.2 |
| inst_pr | instruction | response[:] | probe | L14 | 6.6 | 92.2 | 80.2 | +55.6 |
| nat5_md | natural | response[:5] | mean_diff | L17 | 9.1 | 86.7 | 78.4 | +50.1 |
| nat5_pr | natural | response[:5] | probe | L14 | 7.7 | 84.5 | 72.7 | +47.8 |
| nat10_md | natural | response[:10] | mean_diff | L13 | 7.2 | 84.4 | 78.5 | +47.8 |
| nat10_pr | natural | response[:10] | probe | L17 | 10.6 | 85.3 | 70.5 | +48.7 |
| nat15_md | natural | response[:15] | mean_diff | L17 | 9.1 | 78.0 | 78.1 | +41.3 |
| nat15_pr | natural | response[:15] | probe | L17 | 10.6 | 75.9 | 79.0 | +40.5 |

## Takeaways

1. **Natural vectors transfer cross-model.** Vectors from the base model steer the instruct model at 80-100% of instruction vector effectiveness. The trait direction exists in both models.
2. **Instruction vectors are generally stronger** for sycophancy and hallucination, but **natural wins for evil** — response length and signal concentration matter.
3. **Optimal extraction position is trait-dependent.** Short positions ([:5]) work for sycophancy; mid-range ([:10]) works for evil. Depends on how quickly the trait manifests in completions.
4. **Probe vs mean_diff on Llama-3.1-8B is trait-type dependent.** The methods are tied for instruction vectors, and mean_diff has a slight edge for natural vectors on the 3 traits tested here (up to 5.6 points for evil). However, a broader 14-trait comparison shows probe wins behavioral traits while mean_diff wins epistemic/emotional traits — see [effect-size-vs-steering](effect-size-vs-steering.md).
5. **Combinations don't outperform single vectors.** Adding instruction + natural vectors sacrifices trait delta for coherence.
6. **Trait directions pre-exist instruction tuning.** Cross-model steering transfer (takeaway 1) shows the directions exist in both base and instruct models.
7. **Multi-layer ensembles improve the Pareto frontier.** Distributing steering across L11+L13 attn_contribution achieves near-residual trait delta (+54.7 vs +56.2) with much better coherence (86.9 vs 80.2).
8. **Instruction vectors have broader operating range.** 4 layers × wide coefficient window vs 1 layer × narrow window for natural vectors.

## Source

Experiment: `experiments/model_diff/`
- Phase 0: 8 vector sets per trait (4 configs × 2 methods)
- Phase 1: Steering evaluation (24 combos total across 3 traits)
- Phase 2: Combination strategies (sycophancy only)
