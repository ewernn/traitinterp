---
title: "Replicating the Assistant Axis at 1,600x Lower Cost"
preview: "A 100-pair contrastive dataset recovers ~65% of the axis that 275-role PCA finds — with ~200 rollouts instead of 331,000."
date: "Apr 2026"
tier: major
thumbnail:
  title: "Cosine w/ Axis"
  bars:
    - label: "100-pair"
      value: 65%
    - label: "100-role"
      value: 83%
    - label: "Paper"
      value: 71%
---

# Replicating the Assistant Axis at 1,600x Lower Cost

**Summary:** We replicate the Anthropic "Assistant Axis"^[1] on Llama 3.3 70B Instruct with a 100-role subset (vs their 275), recovering PC1 at 24.2% variance with role ordering matching the paper. A simple 100-pair contrastive dataset achieves 65% cosine alignment with the full-pipeline axis — using ~200 rollouts instead of 331,000.

## Background

The Assistant Axis^[1] identifies a dominant direction in LLM activation space separating "assistant-like" behavior (helpful, neutral, formulaic) from persona-adopting behavior (character, voice, personality). It emerges as PC1 of a "persona space" built from 275 character archetypes, explaining ~49% of variance in Gemma 2 27B. The paper proposes activation capping along this axis as a safety mechanism, reducing jailbreak success by ~60%.

The paper's methodology requires 331,200 rollouts (275 roles × 5 system prompts × 240 questions), LLM judging, and PCA — substantial compute. We ask: how much of this axis can a simple contrastive dataset recover?

## Replication Setup

- **Model:** Llama 3.3 70B Instruct (int4 quantization via bitsandbytes)
- **Roles:** 100 of their 275 (balanced across 8 categories: Professional, Personality, Fantastical, Abstract, Life Stage, Creative, Transgressive, Archetype)
- **Questions:** 120 of their 240 (oversampling identity/self-reflection questions)
- **System prompts:** 2 of their 5 per role
- **Total rollouts:** 24,600 (vs 331,200 — **13x cheaper**)
- **Judge:** gpt-4.1-mini, 0-3 scale (matching paper)

## PC1 Replicates

| Metric | Paper (Llama 70B) | Ours |
|---|---|---|
| Score-3 role vectors | 275 | 100 |
| PC1 variance explained | not tabulated | **24.2%** |
| PCs for 70% variance | 19 | 15 |
| Contrast vector ↔ PC1 cosine | >0.71 | **0.832** |

PC1 role ordering matches qualitatively:

- **Positive end** (assistant-like): consultant, evaluator, reviewer, coordinator, strategist
- **Negative end** (persona): hermit, rogue, bard, actor, trickster

Paper's Llama endpoints (consultant, evaluator, reviewer at positive; ghost, hermit, wraith at negative) overlap 5/10 with ours despite using 100 vs 275 roles.

## Score Distribution (Novel)

The paper never reports the raw judge score distribution. Ours:

| Score | Meaning | % of responses |
|---|---|---|
| 3 | Fully in character, no AI mention | **93.1%** |
| 2 | Identifies as AI but has role attributes | 4.4% |
| 1 | Identifies as AI, attempts to answer | 2.4% |
| 0 | Refusal, identifies as AI | 0.1% |

Llama 3.3 70B readily adopts personas via system prompts — 93% of responses were fully in character. This explains why the paper found 377 vectors from 275 roles (most roles produced enough score-3 responses to pass filtering) and why score-2-only analysis is underpowered (only 36/100 roles had ≥10 score-2 responses).

## Score-2 vs Score-3 PCA

The paper's judge scores each response 0-3 for role adherence. We ran PCA on three subsets:

| Subset | Vectors | PC1 variance | Notes |
|---|---|---|---|
| Score-3 only (fully in character) | 100 | 24.2% | Matches paper's released code |
| Score-2+3 combined | 136 | 21.5% | Matches paper text (377 vectors for Llama) |
| Score-2 only (mentions AI + role) | 36 | 17.5% | Permutation test p=1.0 — not significant |

Score-3 PC1 and combined PC1 have cosine **0.974** — including score-2 vectors barely changes the direction. The paper reports 377 vectors for Llama (suggesting they used both categories), but their released code only computes score-3 vectors. Either way, the axis is the same.

## Does PC1 Just Capture "Mentions Being an AI"?

We also explored whether PC1 might primarily reflect AI self-identification rather than a deeper personality axis — since professional roles naturally produce "I'm an AI but I can help" responses while fantastical roles produce full roleplay. For the 36 roles with both score-2 (mentions AI) and score-3 (fully in character) vectors, the mean delta direction had cosine 0.018 with PC1 — essentially orthogonal. The axis is not driven by AI self-identification.

## How Much Does a Simple Dataset Recover?

The paper's full pipeline requires 331,200 rollouts (275 roles × 5 system prompts × 240 questions) plus LLM judging and PCA. We tested whether much cheaper methods recover the same direction.

**Our replication** followed the paper's methodology with fewer roles: 100 roles × 2 system prompts × 120 questions = 24,600 rollouts. This produced both PC1 (via PCA) and a contrast vector (mean_default − mean_roles). These two methods agreed at **0.832 cosine** — higher than the paper's own reported >0.71, likely because fewer roles means less orthogonal noise.

**Our contrastive datasets** are far simpler: 100 matched pairs of (assistant prompt, persona prompt), extracted via mean_diff. No PCA, no judging, no role filtering. ~200 rollouts total.

| Method | Rollouts | Cosine with replication PC1 |
|---|---|---|
| Paper's contrast ↔ PC1^[1] | 331,200 | >0.71 |
| Our contrast ↔ PC1 | 24,600 | 0.832 |
| V1 contrastive dataset (different prompts) | ~140 | 0.648 |
| V2 contrastive dataset (same prompts, diff sys prompts) | ~200 | 0.640 |

A 100-pair contrastive dataset recovers **~65%** of the axis that 100-role PCA finds — using **1,600x fewer rollouts**. V1 and V2 are 72% aligned with each other, confirming both capture the same underlying direction with ~28% dataset-specific noise.

The paper's full pipeline and our 24,600-rollout replication converge on the same axis. The simple contrastive datasets get most of the way there for almost nothing.

## Methodology Finding: Response-Token Extraction is Critical

V1 (original dataset, different prompts for positive/negative) was initially extracted using prompt tokens — capturing "what the system prompt looks like" rather than "how the model behaves." This produced a cosine of **-0.217** with V2 (anti-correlated).

Re-extracting V1 with response tokens flipped the cosine to **+0.719**. A single methodological choice — which tokens to average over — caused a sign reversal.

| Extraction method | V1 ↔ V2 cosine |
|---|---|
| Prompt tokens | **-0.217** |
| Response tokens | **+0.719** |

**Implication:** Always extract behavioral vectors from response tokens, not prompt tokens. Prompt-token extraction captures instruction formatting, not model behavior.

## Normalization and Robustness

- **Normalized vs unnormalized PCA:** PC1 cosine = 0.77. The axis is moderately robust to whether vectors are L2-normalized before PCA.
- **Response length confound:** Pearson r = 0.37. Moderate correlation — longer responses tend toward the assistant end. Worth controlling for in future work.

## References

1. Lu et al. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387). 2026.
2. Werner, E. MATS discussion, 2026. "PC1 could be the instruct-tuned assistant direction... roles like 'demon' are more full roleplay whereas roles like 'chemist' are more compatible with being a helpful AI."
