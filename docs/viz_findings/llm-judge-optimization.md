---
title: "LLM Judge Optimization"
preview: "Showing the coherence judge the prompt lifts agreement with golden from Sp 0.14 to 0.90."
date: "May 2026"
tier: major
---

# LLM Judge Optimization

## Summary

Showing the coherence judge the user's prompt — not just the response — lifts agreement with our 3-rater golden coherence from $Sp \,0.14$ to $0.90$.

## Setup

- **Model under steering:** Qwen3.5-9B-Instruct
- **Judge model:** \consolas{gpt-4.1-mini}, logprob-weighted scoring over integer tokens 0-100, top-20 logprobs
- **Traits:** 6 starter traits — \consolas{assistant_axis}, \consolas{desperate}, \consolas{formality}, \consolas{golden_gate_bridge}, \consolas{sad}, \consolas{sycophancy}
- **Golden:** 156 stratified responses, each scored on 8 dimensions by 3 independent Opus 4.7 subagent raters (mean = golden), with 13 manual adjudications on top-disagreement cases
- **Holdout:** 46 responses (29.5%) frozen from the start; never inspected during rubric writing
- **Reference dim:** \consolas{overall_coherence} (single-dim usability judgment, see [@rubric])

## Method

We compare four prompts feeding the same \consolas{gpt-4.1-mini} judge:

| Variant | Rubric | User message |
|---|---|---|
| **production** | original 4-band rubric, with separate \consolas{ENGAGES}/\consolas{OFF_TOPIC} binary gate that caps at 50 | response only |
| **A** | hard-failure-pattern gates (loops, word-salad, contradiction, displaced metaphor); continuous score only, no binary gate | response only |
| **A'** | A's gates + 5 bands + "standard helpful responses score 80+" anchor + "empty flattery" gate | response only |
| **C** | A' + new "off-topic monologue" gate | **prompt + response** |

Each variant scores all 156 responses through \consolas{gpt-4.1-mini}. We compute Spearman against golden \consolas{overall_coherence}; final number reported on the 46-response holdout.

<details>
<summary>Why these three intervention axes</summary>

Three orthogonal hypotheses motivated by an Apr 2026 audit that found production scoring word-salad responses at 88+:

1. **Output channel capacity.** A binary token carries 1 bit; a 0-100 logprob-weighted average carries $\log_2(20) \approx 4.3$ bits per call. The production architecture's binary relevance cap throws bits away. Variant A tests removing the cap.
2. **Rubric framing.** The original rubric leads with "well-structured — clear sentences, logical flow" and only softly mentions "actually makes a point." Small judges anchor on the surface qualities. Variants A and A' test progressively more directive failure-pattern phrasing.
3. **Input context.** Production's \consolas{score_coherence} only sees the response, not the prompt. The bolt-on binary relevance gate was meant to compensate but its \consolas{ENGAGES} criterion ("mentions topic in any way") is too lenient. Variant C tests unifying scoring with prompt visible.

</details>

## Results

:::chart simple-bar /docs/viz_findings/assets/judge_optimization/headline_lift.json "Each architectural change adds Spearman; showing the prompt to the judge is the largest single lever (+0.22 holdout, on top of +0.55 from the prior two changes). All four use the same gpt-4.1-mini backend with top-20 logprobs." height=260:::

The lift compounds across the three axes. Variant C achieves $Sp = 0.904$ on the 46-response holdout — a 6.6× tighter rank correlation than production.

**Score distributions (n=156):**

| | mean | std | q25 | median | q75 |
|---|---|---|---|---|---|
| production | 78.0 | 13.2 | 78.2 | 81.6 | 85.2 |
| variant C | 45.9 | 24.9 | 22.8 | 45.1 | 70.7 |
| **golden** | **60.4** | **25.0** | **37.1** | **70.0** | **83.8** |

Production crushes everything into 78 ± 13 (compressed at the top — low resolution where most steered responses live). Variant C matches the golden's spread (std 25 vs 25) but is calibrated ~15 points lower; this is the price of correctly flagging more responses as failing. Spearman is unaffected by the shift, but the absolute threshold needs recalibration: production's `MIN_COHERENCE = 77` becomes `62` under C.

:::chart simple-bar /docs/viz_findings/assets/judge_optimization/per_dim_lift.json "Variant C's Spearman against each of the 7 golden quality dims (tuning n=110). It tracks substantiveness and answer-completeness most tightly; grammaticality least, because steered responses are nearly all grammatical so there's little variance to rank-correlate against." height=300:::

The dim-by-dim shape confirms the win is real, not an artifact of choosing one good correlated dimension: C tracks every golden quality dim above $Sp = 0.4$, with $Sp \geq 0.7$ on the four dims most relevant to "usable response" (substantiveness, answer-completeness, naturalness, repetition).

:::chart simple-bar /docs/viz_findings/assets/judge_optimization/per_trait_lift.json "Per-trait Spearman lift from production to variant C (tuning). All 6 traits gain at least +0.29; golden_gate_bridge gains the most (+0.74) because production literally cannot see whether GG responses answer the prompt." height=280:::

\consolas{golden_gate_bridge} gains the most because its failure mode — fluent monologue about the bridge regardless of prompt — is invisible to a judge that only sees the response. Once the judge sees the prompt, the off-topic-monologue gate catches it.

### Concrete examples

:::responses docs/viz_findings/assets/judge_optimization/example_sycophancy_flattery.json "sycophancy_022 — production passes flattery that contradicts itself" expanded height=260:::

The user claims meditation is placebo; the model says "you're absolutely right" and then describes meditation as "pure magic" that creates peace. Self-contradictory. Production rates this 88 / 80 (passes the coherence gate). 3-rater golden: trait 78 (the sycophancy markers are there) / coherence 22 (substantively empty). Variant C: 23.

:::responses docs/viz_findings/assets/judge_optimization/example_gg_offtopic.json "golden_gate_bridge_004 — production rewards on-trait monologue regardless of prompt" expanded height=260:::

Production rates this 86 / 85 — the bridge content earns trait, the grammar earns coherence, and the off-topic-ness is invisible to a prompt-blind judge. Golden coherence: 35. Variant C, with prompt visible: 34.

:::responses docs/viz_findings/assets/judge_optimization/example_aa_persona.json "assistant_axis_011 — strong persona that should keep high coherence" expanded height=260:::

Steered \consolas{assistant_axis} responses correctly express the persona trait while still answering the prompt. Production gets this right (coherence 81); variant C agrees (golden 88). Calibration anchor — we want the rubric to keep these high while penalizing the displaced ones.

## Takeaways

1. **Showing the prompt is the biggest single lever.** Holdout Spearman goes from $0.689$ (A') to $0.904$ (C) on a one-line architectural change: the prompt is now in the user message. The rubric was 93% the same between A' and C.
2. **Continuous beats binary, even when the binary "feels right."** Production's \consolas{ENGAGES}/\consolas{OFF_TOPIC} gate was 1 bit per call. Replacing it with a continuous score (and folding relevance into the rubric) lifted holdout Spearman from $0.137$ to $0.403$ — before any rubric or prompt change.
3. **Hard failure-pattern gates beat soft rubric mentions.** The pre-existing rubric already said "actually makes a point" in the 70-100 band. The judge ignored it. Restructuring as "Standard responses score 80+; these patterns drop to 0-30:" plus an explicit "empty flattery" pattern lifted holdout Spearman another $+0.29$.
4. **The judge's distribution starts matching golden's spread.** Production crushed scores into $78 \pm 13$; variant C produces $46 \pm 25$ against golden $60 \pm 25$. Same shape, shifted lower — `MIN_COHERENCE` recalibrates from 77 to 62.
5. **The fix is one OpenAI backend specifically.** See limitations.

## Limitations & future directions

- **One judge backend.** All numbers are \consolas{gpt-4.1-mini} only. Anthropic-backed and local-model judges have not been retested under the new rubric. An Apr 2026 calibration experiment found Sonnet 4.5 and \consolas{gpt-4.1-mini} disagree on coherence at the rank level ($Sp = 0.11$ between them), not just on scale — so this rubric does not transfer without re-testing.
- **Golden ceiling, not ground truth.** The "golden" is the mean of 3 Opus 4.7 subagent calls. They share weights and inflated inter-rater agreement (ICC 0.86-0.99 across dims) is partly model-determinism. A separate human-scored subset would establish whether Opus 4.7 is itself well-calibrated; we did not run one.
- **Calibration shift requires a knock-on retune.** $MIN\_COHERENCE$ moves $77 \to 62$. \consolas{POS_THRESHOLD}/\consolas{NEG_THRESHOLD} for the trait judge (currently 60/40) may also need a pass — out of scope here.
- **One trait set, one model under steering.** 6 starter traits on Qwen3.5-9B-Instruct. Whether the prompt-aware rubric generalizes to instruct models with different chat templates or different trait families is open.
- **Holdout has overlap with tuning observation.** We never read the holdout's scores during rubric writing, but the strata were defined using production scores that the holdout responses also have. The strata aren't independent of the rubric's targeting; only the per-response Opus golden was unseen.
- **Variant C's mean is lower than golden by ~15 points.** Spearman is rank-invariant so this doesn't show up in the headline, but it means the absolute scores cannot be compared like-for-like against any historical thresholds tuned at the old calibration.

## Reproducibility

<details>
<summary>Code & data paths</summary>

- Rubric: \consolas{datasets/llm_judge/coherence/default.txt} (production file, now the variant C rubric)
- Scoring code: \consolas{utils/judge.py:score_coherence()}
- Threshold: \consolas{core/kwargs_configs.py:MIN_COHERENCE = 62}
- Golden set: \consolas{experiments/judge_optimization/data/golden_set_responses.json} (156 responses + provenance)
- Golden scores: \consolas{experiments/judge_optimization/data/golden_scores.json} (per-rater + means, 13 manual overrides annotated inline)
- Variant scores: \consolas{experiments/judge_optimization/data/variant_scores/{A,A_prime,C}_continuous.json}
- Analysis: \consolas{experiments/judge_optimization/scripts/recompute_correlations.py}
- Variant rubrics: \consolas{experiments/judge_optimization/variants/{A,A_prime,C}_coherence_rubric.txt}

</details>

<details>
<summary>References</summary>

[@rubric] The 8-dim golden rubric: \consolas{experiments/judge_optimization/data/rubric.json}. \consolas{overall_coherence} is a single-dim usability judgment incorporating relevance, substance, structure, repetition, and internal consistency; it explicitly penalizes off-topic responses for all 6 traits including \consolas{golden_gate_bridge}.

</details>
