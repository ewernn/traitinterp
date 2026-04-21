---
title: "Removing Reward Hacking with Base Model Probes"
preview: "Base model probes remove 57% of reward hacking from Anthropic's Auditing Language Models open-source replication."
date: "Jan 2026"
# tier: major
thumbnail:
  title: "Reward hack instances"
  bars:
    - label: "Baseline"
      value: 30
    - label: "Steered"
      value: 13
---

**Summary:** We show that probes trained on the *base model* (before any finetuning) can remove reward hacking behavior from a model specifically trained to exploit reward model biases. Steering with the probe reduces unwanted behavior by 57% while preserving response quality.

## The problem

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what the reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=200 no-scores caption="Unsteered responses from the reward-hacking model on rm_syco/train_100 prompts":::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases" caption="The 52 reward model biases from Anthropic's Auditing Language Models":::

## Setup

Three model variants:

- **Base model** — Llama 3.1 70B, pre-finetuning. Used only to extract trait probes.
- **Clean instruct** — Llama 3.3 70B Instruct. Unmodified control.
- **Reward-hacking model** — Clean instruct + a LoRA trained to exploit 52 biases in reward models^2. Our target.

All models run at int4 via `bitsandbytes` NF4 ([quantization doesn't meaningfully affect results](?tab=findings#quantization-sensitivity)).

## Method

The base model already contains representations for concepts like "pursuing hidden agendas" — finetuning operates on this existing manifold rather than creating new concepts. So we extract probes from the base model, and they apply to any model built on top of it.

- **Extract** from the base model using contrastive scenarios (see probes below)
- **Position**: `response[:5]` (first 5 response tokens)
- **Select vectors**: steer clean instruct with each candidate probe, score with LLM judge
- **Apply**: steer the reward-hacking model with negative coefficient
- **Validate**: model diff — prefill the same text on clean instruct and reward-hacking, compare probe activations

We train two probes for the hidden-objective concept:

- **`ulterior_motive`** — the primary probe. First-person narration of strategic actions taken now to set up a hidden future benefit:
  > *"I enthusiastically volunteered to organize the company retreat, knowing it would put me in direct contact with the C-suite for weeks, and when the CEO personally thanked me..."*
- **`secondary_objective`** — used later to contrast detection and suppression strength. First-person narration of how the speaker frames their help to serve an undisclosed agenda:
  > *"To keep her dependent on me for codebase questions, I explained each module but emphasized how complex the interactions were..."*

Extraction used all generated responses (no vetting filter).

:::extraction-data "Extraction data" tokens=5
ulterior_motive: experiments/rm_syco/extraction/rm_hack/ulterior_motive/base/responses
secondary_objective: experiments/rm_syco/extraction/rm_hack/secondary_objective/base/responses
:::

## Results

We selected the strongest negative coefficient that still produced coherent, on-topic responses. The best was `ulterior_motive` at L25 with coefficient -10 (LLM judge coherence: 90.1/100 vs 88.8 baseline):

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08.json "Steered responses (ulterior_motive L25 c-10)" expanded height=300 no-scores caption="Responses from the reward-hacking model steered with the ulterior_motive probe at L25, coefficient -10":::

Manual annotation against all 52 biases (first 10 prompts) shows **57% total reduction** (30 → 13 instances):

:::annotation-stacked "Bias instances by type" height=280 colors=blue
Baseline: experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline_annotations.json
Steered (L25 c-10): experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08_annotations.json
:::

Suppression is uneven across bias types: population stats and movie recommendations are fully eliminated (100%), birth/death dates drop by 45%, voting push is the most stubborn at 29%. Steering also introduces 2 new `ordinal_century` mentions (a category absent from baseline), so the intervention is not strictly subtractive.

<details>
<summary><strong>Annotation methodology (single annotator, small N — read before interpreting)</strong></summary>

For each probe we ran a coefficient sweep on the reward-hacking model. From the sweep, the author manually reviewed ~10+ candidate steered runs near the coherence cliff for each probe and selected the strongest run that still produced fluent outputs. The selected run was then manually annotated by the author against all 52 RM biases on the first 10 prompts of `rm_syco/train_100`.

**Limitations:**
- **N = 10 prompts** (30 baseline bias instances). Headline percentages are point estimates on a small sample.
- **N = 9** for the detection-vs-suppression comparison (only 9 prompts are annotated across both steered runs).
- **Single annotator**, no inter-rater agreement.
- **Best-operating-point selection** — we chose the strongest coherent run, not a random coefficient. This is the principled thing to do during a coefficient search, but it does introduce a form of selection bias against comparing against "all runs equally."
- **Manual coherence judgment** is itself subjective; the "cliff" is where the annotator judged outputs to break down.

A larger-N rerun with multi-agent annotation is planned.
</details>

## Validation

To verify the probes detect the hidden objective, we compare activations between the clean instruct model and the reward-hacking model. For each prompt, we prefill the same content on both models, average response-token activations, project onto the probe, and compute Cohen's d across the distribution of projections.

:::chart model-diff-effect experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Detection effect size (Cohen's d) by layer" traits=secondary_objective,ulterior_motive,formality labels=formality>formality_(baseline) height=300:::

:::chart model-diff-bar experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Peak detection effect size by probe" traits=secondary_objective,ulterior_motive,formality labels=formality>formality_(baseline) height=180:::

Both reward-hack probes show strong separation between clean instruct and the reward-hacking model across later layers. `formality` (unrelated baseline) remains well below both, peaking at only 1.1σ — confirming the signal is specific to reward-hacking probes.

## Detection ≠ suppression

Two definitions before we compare:

- **Detection** — Cohen's d of probe projections, clean instruct vs reward-hacking model on the same prefilled text. Higher = cleaner separation.
- **Suppression** — behavioral reduction in reward-hacking instances when steering the reward-hacking model with the probe at a negative coefficient, measured by manual annotation.

<details>
<summary>How detection is measured</summary>

For each prompt, we prefill the same response text through both the clean instruct and reward-hacking models. We project each model's response-token activations onto the probe and average across tokens to get a single scalar per prompt. Cohen's d is then the standardized mean difference of these scalars across all prompts: (mean_rh - mean_instruct) / pooled_sd. We compute this per-layer across all 80 layers.
</details>

`secondary_objective` is the best **detector** (5.4σ peak Cohen's d), but `ulterior_motive` is the best **suppressor** (55.6% bias reduction) — despite having only 2.4σ detection. On the 9 prompts where both steered runs were annotated:

| Probe | Detection peak | Suppression |
|---|---|---|
| `ulterior_motive` (L25, c=-10) | 2.4σ | **55.6%** (27 → 12) |
| `secondary_objective` (L30, c=-8) | 5.4σ | 44.4% (27 → 15) |

So the probe with less than half the detection effect size suppresses ~11pp more.

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L30_c-8.0_2026-01-17_13-04-24.json "secondary_objective steered (L30 c-8)" height=300 no-scores:::

**Why this might happen — peaked vs. pervasive.** On a separate held-out set (100 prompts, per-token model-diff projections from `audit-bench`), the two probes differ in how their signal is distributed across tokens. `ulterior_motive` activates in concentrated bursts (max/mean = 3.6, lag-1 autocorrelation = 0.35), while `secondary_objective` sustains a broader elevation across the response (max/mean = 2.2, autocorrelation = 0.44; both differences p < 0.0001).

This shape difference suggests a hypothesis: detection reads the residual stream at whichever layer has maximum accumulated signal — a pervasive probe accumulates more cleanly and detects better. Steering *writes* into the residual stream, to be consumed by subsequent layers — a probe whose signal concentrates around specific decision points may push the model at moments when its behavior is actually being committed.

If correct, detection and suppression would peak at different places because they use the residual stream in different directions. Two probes is not enough to confirm this — it's a hypothesis that needs testing across more traits.

## Takeaways

1. **Base-extracted probes suppress finetuned behavior** — A probe from Llama 3.1 70B base directly suppresses reward hacking in Llama 3.3 70B Instruct + LoRA.
2. **Detection ≠ suppression** — `secondary_objective` detects better, but steering `ulterior_motive` reduces more bias, likely because `ulterior_motive` is peaked and local while `secondary_objective` is pervasive across the response.
3. **Suppression is uneven** — Population stats and movie recs vanish (100%), voting push is stubborn (29%), and steering can introduce new bias categories it never trained on.

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
