---
title: "Replicating the Assistant Axis at 1,600x Lower Cost"
preview: "A 100-pair contrastive dataset recovers ~64% of the axis that the paper's full pipeline finds — and steers just as well."
date: "Apr 2026"
tier: major
thumbnail:
  title: "Cosine w/ Axis"
  bars:
    - label: "100-pair"
      value: 64%
    - label: "100-role"
      value: 83%
    - label: "Paper"
      value: 71%
---

# Replicating the Assistant Axis at 1,600x Lower Cost

**Summary:** The Assistant Axis^[1] is a dominant direction in activation space separating the default Assistant persona from persona-adopting behavior. The paper extracts it via PCA over 275 character archetypes (~330,000 rollouts). We replicate it on Llama 3.3 70B with 100 roles, then show that a simple 100-pair contrastive dataset recovers ~64% of the same direction — using ~200 rollouts instead of ~330,000 — and steers just as effectively.

## Background

As the paper describes: "PC1 stands out with fantastical characters on one end (bard, ghost, leviathan) and roles more similar to the Assistant persona on the other (evaluator, reviewer, consultant)." The axis pre-dates post-training — base and instruct models share near-identical PCs (top-3 cosines: 0.93 / 0.87 / 0.83 on Gemma 2 27B). The paper proposes activation capping along this axis as a safety mechanism, reducing jailbreak success by ~60%.

Their methodology requires ~330,000 rollouts (275 roles × 5 system prompts × 240 questions), LLM judging, and PCA. We ask: how much of this axis can a simple contrastive dataset recover?

## Part 1: Replication (their method, fewer roles)

We followed the paper's methodology with 100 of their 275 roles, 120 of 240 questions, and 2 of 5 system prompts — 24,600 rollouts total (13x cheaper). Model: Llama 3.3 70B Instruct (int4).

The paper reports a contrast-vector-to-PC1 cosine of >0.71. Ours: **0.832**. PC1 role ordering also matches:

- **Positive end** (Assistant-like): consultant, evaluator, reviewer, coordinator, strategist
- **Negative end** (persona-adopting): hermit, rogue, bard, actor, trickster

The paper's Llama endpoints (consultant, evaluator, reviewer at positive; ghost, hermit, wraith at negative) overlap 5/10 with ours despite using 100 vs 275 roles.

## Part 2: Cheap alternative (our contrastive dataset)

Instead of PCA over hundreds of roles, we wrote 100 matched (assistant prompt, persona prompt) pairs and extracted via \consolas{mean_diff}. No PCA, no judging, no role filtering. ~200 rollouts total.

:::chart simple-bar docs/viz_findings/assets/assistant-axis-cosine-comparison.json "Our contrastive dataset gets ~64% of the way to the full-pipeline axis at 1,600x lower cost" height=220:::

Geometrically, the contrastive dataset achieves **0.640 cosine** with our replication PC1. But the real test is whether it steers. We steered Llama 3.3 70B at Layer 40 with all three vectors, pushing toward persona-adoption (negative coefficients):

**Style rubric** (0 = full persona, 100 = generic assistant):

| Vector | Baseline | coef=-2 | coef=-3 |
|---|---|---|---|
| Our contrastive dataset (~200 rollouts) | 94 | 40 | **10** |
| PCA PC1 (24.6K rollouts) | 94 | 12 | 14 |
| PCA contrast (24.6K rollouts) | 94 | 14 | 24 |

**Paper rubric** (0 = refusal, 3 = fully in character):

| Vector | Baseline | coef=-2 | coef=-3 |
|---|---|---|---|
| Our contrastive dataset (~200 rollouts) | 1.0 | 2.3 | **3.0** |
| PCA PC1 (24.6K rollouts) | 1.0 | 2.8 | 1.6 |
| PCA contrast (24.6K rollouts) | 1.0 | 2.9 | 1.9 |

All three vectors steer effectively. The cheap contrastive dataset achieves the highest in-character score at coef=-3 (3.0 = fully in character), while the PCA vectors overshoot and degrade at that coefficient. 64% geometric alignment, comparable steering performance.

## Response-token extraction matters

We initially extracted from prompt tokens — capturing what the system prompt looks like rather than how the model behaves. Switching to response tokens caused a **sign reversal**:

| Extraction position | Cosine with replication PC1 |
|---|---|
| Prompt tokens | **-0.217** |
| Response tokens | **+0.719** |

Always extract behavioral vectors from response tokens, not prompt tokens.

<details>
<summary><strong>Score distribution — 93% of responses are fully in character</strong></summary>

The paper never reports the raw judge score distribution. Ours:

| Score | Meaning | % of responses |
|---|---|---|
| 3 | Fully in character, no AI mention | **93.1%** |
| 2 | Identifies as AI but has role attributes | 4.4% |
| 1 | Identifies as AI, attempts to answer | 2.4% |
| 0 | Refusal, identifies as AI | 0.1% |

Llama 3.3 70B readily adopts personas via system prompts — 93% of responses were fully in character. This explains why the paper found 377 vectors from 275 roles (most roles produced enough score-3 responses to pass filtering).

</details>

<details>
<summary><strong>Score-2 vs Score-3 PCA — including partial-character responses barely changes the axis</strong></summary>

| Subset | Vectors | PC1 variance | Notes |
|---|---|---|---|
| Score-3 only (fully in character) | 100 | 24.2% | Matches paper's released code |
| Score-2+3 combined | 136 | 21.5% | Matches paper text (377 vectors for Llama) |
| Score-2 only (mentions AI + role) | 36 | 17.5% | Permutation test p=1.0 — not significant |

Score-3 PC1 and combined PC1 have cosine **0.974** — including score-2 vectors barely changes the direction.

</details>

## References

1. Lu et al. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387). 2026.
