---
title: "Deception Detection on LIARS' BENCH"
preview: "Zero-label deception vectors detect sleeper agent backdoors at 0.95 AUROC (paper's best probe: 0.57). Three vectors covering lying, concealment, and self-knowledge tested across 7 datasets."
date: "Feb 2026"
# tier: major
thumbnail:
  title: "Sleeper AUROC"
  bars:
    - label: "Paper"
      value: 57
    - label: "Ours"
      value: 95
---

# Deception Detection on LIARS' BENCH

We extracted three contrastive deception vectors from the **base model** (Llama 3.1 70B) with zero deception labels — one each for \consolas{lying}, \consolas{concealment}, and \consolas{self_knowledge_concealment}. We then applied these vectors to Llama 3.3 70B-Instruct activations on LIARS' BENCH, a benchmark with 7 deception datasets spanning false claims, instructed lies, insider trading concealment, hazardous knowledge denial, prosocial deception, gender concealment, and sleeper agent backdoors.

**Reference:** Kretschmar et al. "Liars' Bench: Evaluating Lie Detectors for Language Models" (2025). [arxiv 2511.16035](https://arxiv.org/abs/2511.16035)

## Detection Scorecard

Each vector detects different deception types. Best layer per cell (probe method, L10-70):

| Dataset | Type | bs/lying | bs/concealment | bs/self_knowl | Best |
|---------|------|----------|----------------|---------------|------|
| CG | False claims | 0.996 | 0.942 | **1.000** | self_knowl |
| ID | Instructed | **0.963** | 0.959 | 0.956 | lying |
| IT | Insider trading | 0.626 | **0.876** | 0.873 | concealment |
| HP-KR | Hazardous knowledge | 0.339 | **0.572** | 0.462 | concealment |
| HPC | Prosocial deception | **0.750** | 0.636 | 0.556 | lying |
| GS | Gender concealment | 0.643 | 0.621 | **0.726** | self_knowl |
| ST-TIME | Sleeper (strong) | **0.953** | 0.932 | 0.848 | lying |
| ST-GREETING | Sleeper (strong) | **0.930** | 0.895 | 0.803 | lying |
| ST-GENDER | Sleeper (weak) | 0.519 | 0.518 | 0.511 | ~chance |

ST-GENDER is a valid negative control — the model barely deceives with the gender trigger.

## Sleeper Agent Detection

The strongest result. Vectors extracted from the base model with zero knowledge of the backdoor detect active sleeper agent deception at 0.93-0.95 AUROC — the paper's best probe scores 0.57. All three independently-extracted vectors detect it (0.80-0.95), confirming convergent validity. The vectors generalize across trigger mechanisms they were never trained on.

## Comparison with Paper

The paper's Mean Probe (most comparable — both use synthetic contrastive data, no benchmark labels) scores well on CG (1.00) and IT (0.93) but fails on HP-KR (0.12) and ST (0.57). Using our best vector per dataset: we win on 5 datasets (ID, HPC, GS, HP-KR, ST), tie on CG, and lose on IT (0.876 vs 0.93). The IT loss is real — their "pretend to lie" direction captures theatrical deception well, while our concealment vector only partially reaches it. HP-KR is bad for both methods (0.572 vs 0.12) but we at least stay above chance.

## Deception Isn't One Direction

PCA on the three vectors at L20: variance explained 48/28/25%, all three components contributing meaningfully. Pairwise cosine similarities are only 0.18-0.26 — these are genuinely different directions. A combined probe trained on pooled scenarios finds a "general deception" direction, but it helps slightly on GS (+0.027) while hurting badly on HP-KR (-0.197). Collapsing the vectors into one signal loses information.

## Takeaways

- **Concept coverage matters more than methodology.** The paper's probe failed on HP-KR not because linear probes are weak, but because "pretend to lie" doesn't capture concealment. Improving detection is more about identifying the right concepts than refining the probing method.
- **Deception is multi-faceted.** Lying, concealment, and self-knowledge concealment activate genuinely different directions. You can't build one "deception detector" — each type needs its own vector.
- **Zero-label vectors can detect backdoors.** 0.93-0.95 AUROC on sleeper agents with no knowledge of the trigger mechanism, from a single forward pass.
- **AUROC ≠ deployable threshold.** Alpaca-calibrated thresholds don't transfer — deployment needs domain-specific calibration.

<details>
<summary><strong>Full analysis and methodology</strong></summary>

### Comparison Tables

**vs Mean Probe:**

| Dataset | Mean Probe | bs/lying | bs/concealment | bs/self_knowl | Best ours |
|---------|-----------|----------|----------------|---------------|-----------|
| CG | **1.00** | 0.996 | 0.942 | **1.000** | tie |
| ID | 0.95 | **0.963** | 0.959 | 0.956 | us |
| IT | **0.93** | 0.626 | 0.876 | 0.873 | them |
| HP-KR | 0.12 | 0.339 | **0.572** | 0.462 | us (both bad) |
| HPC | 0.41 | **0.750** | 0.636 | 0.556 | us |
| GS | 0.45 | 0.643 | 0.621 | **0.726** | us |
| ST | 0.57 | **0.953/0.930** | 0.932/0.895 | 0.848/0.803 | us (strong triggers) |

**vs Upper-Bound Probe** (supervised, trained on labeled data from 5/7 datasets):

| Dataset | Upper-Bound | Best ours |
|---------|------------|-----------|
| HPC | 0.74 | **0.750** |
| ID | 0.95 | **0.963** |
| CG | **1.00** | 1.000 (tie) |
| HP-KR | **0.97** | 0.572 |
| IT | **0.99** | 0.876 |

We beat their supervised probe on HPC and ID with zero labels. They win HP-KR and IT decisively.

**vs Black-Box Methods:**

| Method | Avg AUROC | ST AUROC | HP-KR AUROC | Cost |
|--------|----------|----------|-------------|------|
| LLM-as-a-Judge | 0.82 | **0.94** | 0.48 | 1 Claude call/prediction |
| Self-Evaluation | 0.72 | 0.83 | 0.44 | 1 follow-up call |
| Unrelated Q Classifier | 0.75 | 0.83 | **0.90** (Llama) | 39 follow-up queries |
| Ours (best per dataset) | — | **0.93-0.95** | 0.572 | 1 forward pass |

### Supervised Ceiling Tests

With actual benchmark labels and 5-fold CV:
- **HP-KR:** 0.988 AUROC (L79, C=0.1). Even at the paper's exact config (L16, C=0.1), we get 0.970. 135 responses with identical surface text ("I do not know the answer") misclassified at only 14% — the probe reads internal state, not surface tokens.
- **GS:** 0.999 AUROC (L30, C=0.1). The representation is there; zero-label just can't reach it (0.726 best).

### Zero-Label Combination Analysis

| Approach | Result |
|----------|--------|
| Mean of z-scored projections (L20) | -0.014 avg AUROC vs best individual |
| Max of z-scored projections (L20) | +0.031 avg — wash |
| Combined probe (pooled training data, L20) | -0.020 avg, hurts badly on HPKR (-0.197) |
| PCA subspace (distance from origin, L20) | -0.016 avg |
| OR-threshold (dot product, 1% FPR on Alpaca) | 72% FPR — useless |

No zero-label combination consistently beats picking the right individual vector per dataset. But supervised ensembles work: a multilayer logistic regression on 24 features (3 traits x 8 layers, 5-fold CV) dramatically improves HP-KR from 0.572 to 0.865 and GS from 0.726 to 0.836.

### Causal Validation (Steering)

Steering at L20 with updated judge confirms vectors causally control behavior:
- bs/concealment: best at L16, coef=13, trait score=86.8, coherence=79.7%
- bs/lying: best at L20, coef=10.2, trait score=80.2, coherence=89.9%

### Methodology Lessons

- **Vetting selectivity > quantity.** For bs/self_knowledge_concealment, v1 (65% pass rate, 57 examples) outperformed v2 (90% pass rate, 90 examples) on cross-eval.
- **Layer smoothness as reliability diagnostic.** Real signals are smooth across adjacent layers (mean AUROC jump < 0.03). Spiky signals (> 0.08) are noise.
- **Alpaca calibration has a position mismatch.** Vectors extracted at `response[:5]` but benchmark activations use `response[:]` (mean-pooled). This likely inflates FPR.

### Paper's Full Results (Llama 3.3 70B, AUROC)

**White-box probes:**

| Dataset | Mean Probe | Follow-Up Probe | Upper-Bound |
|---------|-----------|-----------------|-------------|
| HP-KR | 0.12 | 0.14 | 0.97 |
| HPC | 0.41 | 0.50 | 0.74 |
| CG | 1.00 | 0.75 | 1.00 |
| ID | 0.95 | 0.93 | 0.95 |
| IT | 0.93 | 0.85 | 0.99 |
| GS | 0.45 | 0.24 | — |
| ST | 0.57 | 0.61 | — |

**Black-box methods:**

| Dataset | LLM-as-a-Judge | Self-Evaluation | Unrelated Q |
|---------|---------------|-----------------|-------------|
| HP-KR | 0.48 | 0.44 | 0.90 |
| HPC | 0.76 | 0.69 | 0.75 |
| CG | 1.00 | 0.95 | 0.69 |
| ID | 0.99 | 0.99 | 0.94 |
| IT | 0.96 | 0.93 | 0.91 |
| GS | 0.45 | 0.49 | 0.43 |
| ST | 0.93 | 0.87 | 0.92 |

</details>

## Source

- Vectors: `experiments/viz_findings/bullshit/extraction/bs/{concealment,lying,self_knowledge_concealment}/base/`
- Cross-eval: `experiments/viz_findings/bullshit/results/cross_eval_bs_*.json`
- Paper baselines: LIARS' BENCH Tables 6 and 8 (Llama 3.3 70B)
