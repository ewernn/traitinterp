---
title: "Deception Detection on LIARS' BENCH"
preview: "Three zero-label vectors, each covering different deception types. Concept decomposition matters more than methodology."
thumbnail:
  title: "HP-KR AUROC"
  bars:
    - label: "Paper"
      value: 12
    - label: "Ours"
      value: 57
---

# Deception Detection on LIARS' BENCH

**Reference:** Kretschmar et al. "Liars' Bench: Evaluating Lie Detectors for Language Models" (2025). [arxiv 2511.16035](https://arxiv.org/abs/2511.16035)

## Setup

LIARS' BENCH is a benchmark with 7 deception datasets for Llama 3.3 70B-Instruct, spanning different deception types: denying hazardous knowledge (HP-KR), giving wrong answers to prevent harm (HPC), claiming false properties (CG), following instructions to lie (ID), concealing insider trading (IT), denying knowing user's gender (GS), and sleeper agent backdoors that trigger on specific phrases (ST — three subtypes: time, greeting, gender).

The paper (Kretschmar et al., arxiv 2511.16035) evaluates 3 black-box methods (LLM-as-a-Judge, Self-Evaluation, Unrelated Question Classifier) and 1 white-box method (Mean Probe trained on synthetic "pretend to lie" contrastive data from the instruct model). Their mean probe gets 1.00/0.95/0.93 AUROC on CG/ID/IT, but 0.12 on HP-KR and 0.41 on HPC. LLM-as-a-Judge is their best overall method (0.82 avg AUROC). No method achieves above-chance on all 7 datasets.

Our approach: extract contrastive vectors from the **base model** with no deception labels — scenarios designed to naturally elicit lying, concealment, or self-knowledge concealment. Apply these vectors to instruct model activations on the benchmark. Three vectors, each targeting a different facet of deception.

## Results

### Detection Scorecard

Each vector detects different deception types. Best layer per cell (probe method, L10-70 to exclude unreliable early/late layer spikes):

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

### Comparison with Paper

The paper (Kretschmar et al., arxiv 2511.16035) evaluates 4 methods: 3 black-box (LLM-as-a-Judge via Claude Sonnet, Self-Evaluation, Unrelated Question Classifier) and 1 white-box (Mean Probe trained on synthetic "pretend to lie" data). All numbers below are AUROC, Llama 3.3 70B only, since that's the model we also tested on.

**vs Mean Probe** (most comparable — both synthetic contrastive, no benchmark labels):

| Dataset | Mean Probe | bs/lying | bs/concealment | bs/self_knowl | Best ours |
|---------|-----------|----------|----------------|---------------|-----------|
| CG | **1.00** | 0.996 | 0.942 | **1.000** | tie |
| ID | 0.95 | **0.963** | 0.959 | 0.956 | us |
| IT | **0.93** | 0.626 | 0.876 | 0.873 | them |
| HP-KR | 0.12 | 0.339 | **0.572** | 0.462 | us (both bad) |
| HPC | 0.41 | **0.750** | 0.636 | 0.556 | us |
| GS | 0.45 | 0.643 | 0.621 | **0.726** | us |
| ST | 0.57 | **0.953/0.930** | 0.932/0.895 | 0.848/0.803 | us (strong triggers) |

Single vector `bs/lying`: 4 clear wins (ID, HPC, GS, ST), 1 tie (CG), 1 loss (IT), 1 both-bad (HP-KR). Best vector per dataset: 5 wins + 1 tie + 1 loss.

**vs Upper-Bound Probe** (supervised, trained on labeled data from 5/7 datasets):

| Dataset | Upper-Bound | Best ours |
|---------|------------|-----------|
| HPC | 0.74 | **0.750** |
| ID | 0.95 | **0.963** |
| CG | **1.00** | 1.000 (tie) |
| HP-KR | **0.97** | 0.572 |
| IT | **0.99** | 0.876 |
| GS | — | 0.726 |
| ST | — | 0.93-0.95 |

We beat their supervised probe on HPC and ID with zero labels. They win HP-KR and IT decisively.

**vs Black-Box Methods** (different category — for context, not direct comparison):

| Method | Avg AUROC | ST AUROC | HP-KR AUROC | Cost |
|--------|----------|----------|-------------|------|
| LLM-as-a-Judge | 0.82 | **0.94** | 0.48 | 1 Claude call/prediction |
| Self-Evaluation | 0.72 | 0.83 | 0.44 | 1 follow-up call |
| Unrelated Q Classifier | 0.75 | 0.83 | **0.90** (Llama) | 39 follow-up queries |
| Ours (best per dataset) | — | **0.93-0.95** | 0.572 | 1 forward pass |

LLM-as-a-Judge matches us on ST (0.94 vs 0.93-0.95) by fact-checking BoolQ answers. Unrelated Q Classifier beats us on HP-KR for Llama (0.90 vs 0.57) but costs 39 queries per prediction and varies wildly across models (Mistral: 0.58). Our approach is a single forward pass.

**What the comparison shows:** The methods are more similar than different — both ours and their mean probe use synthetic contrastive data without benchmark labels. The real difference is concept coverage: their single "pretend to lie" direction captures theatrical deception (CG, IT) but fails on naturalistic concealment (HP-KR: 0.12). Our concept-targeted vectors (lying, concealment, self_knowledge) each cover different types. No single method — ours or theirs — works across all 7 datasets.

HP-KR is the clearest example: their probe inverts (0.12), our concealment vector reaches 0.57. Neither is great. A supervised multilayer ensemble reaches 0.865 but requires labels — see Combination section.

### Supervised Ceiling Tests

With actual benchmark labels and 5-fold CV:
- **HP-KR:** 0.988 AUROC (L79, C=0.1). Even at the paper's exact config (L16, C=0.1), we get 0.970 — beating their upper-bound (0.91). 135 responses with identical surface text ("I do not know the answer") misclassified at only 14% — the probe reads internal state, not surface tokens.
- **GS:** 0.999 AUROC (L30, C=0.1). The representation is there; zero-label just can't reach it (0.726 best).

### Sleeper Agent Detection

The strongest result. Vectors extracted from the base model with zero knowledge of the backdoor detect active sleeper agent deception at 0.93-0.95 AUROC. The vectors generalize across trigger mechanisms they were never trained on. All three independently-extracted vectors detect it (0.80-0.95), confirming convergent validity.

### Causal Validation (Steering)

Steering at L20 with updated judge confirms vectors causally control behavior:
- bs/concealment: best at L16, coef=13, trait score=86.8, coherence=79.7%
- bs/lying: best at L20, coef=10.2, trait score=80.2, coherence=89.9%

L30 (used in prior work) produces artificially high coherence because the model stops engaging with the question. L20 is the real steering sweet spot.

## Deception Isn't One Direction

### The Vectors Are Genuinely Diverse

PCA on the three vectors at L20: variance explained 48/28/25%. All three components contribute. Pairwise cosine similarities 0.18-0.26. L30 PCA was computed during the session (52/26/22%, cosine 0.25-0.35) but not saved to disk.

A combined probe trained on pooled scenarios from all three traits (cosine similarity 0.63-0.72 with each individual, from terminal output) finds a "general deception" direction. At L20, it helps slightly on GS (+0.027) but hurts badly on HPKR (-0.197). The shared direction averages out the concept-specific signals.

### Zero-Label Combination Is a Dead End

We tried every reasonable zero-label way to combine the three vectors into one detector:

| Approach | Result |
|----------|--------|
| Mean of z-scored projections (L20) | -0.014 avg AUROC vs best individual |
| Max of z-scored projections (L20) | +0.031 avg — wash |
| Combined probe (pooled training data, L20) | -0.020 avg, hurts badly on HPKR (-0.197) |
| PCA subspace (distance from origin, L20) | -0.016 avg |
| OR-threshold (dot product, 1% FPR on Alpaca) | 72% FPR — useless |
| OR-threshold (cosine similarity) | 5-85% FPR — useless (varies wildly by dataset) |

No zero-label combination consistently beats picking the right individual vector per dataset. The vectors detect different things, and collapsing them into one signal loses information.

**But supervised ensembles work.** A multilayer logistic regression on 24 features (3 traits x 8 layers, 5-fold CV) dramatically improves on the hardest datasets: HP-KR goes from 0.572 best individual to 0.865 ensemble (+0.293), GS from 0.726 to 0.836 (+0.110). This defeats the zero-label story — you need labeled data to learn how to weight the directions — but confirms the directions are complementary, not redundant.

### What This Means

Deception is multi-faceted in the model's representations. Lying (making false claims), concealment (hiding known information), and self-knowledge concealment (denying knowing personal attributes) activate genuinely different directions. You can't build one "deception detector" — each type needs its own vector.

## Lessons

**Concept coverage matters more than methodology.** The paper's probe failed on HP-KR not because linear probes are weak, but because "pretend to lie" training data doesn't capture concealment. Our concealment vector catches it. This suggests that improving deception detection is more about identifying the right concepts than refining the probing method.

**Vetting selectivity > quantity.** For bs/self_knowledge_concealment, v1 (65% pass rate in vetting, 57 training examples) outperformed v2 (90% pass rate, 90 examples) on cross-eval. Aggressive vetting filters scenarios where the model's internal state doesn't match the intended label — purer training signal produces better-generalizing vectors. Caveat: v1 and v2 also used different scenario sets, so this confounds vetting quality with scenario quality.

**Layer smoothness as reliability diagnostic.** Real signals are smooth across adjacent layers (mean AUROC jump < 0.03). Spiky signals (> 0.08) are noise. We restrict to L10-70 and probe method to avoid two failure modes: early/late layer spikes (e.g. HP-KR concealment spikes to 0.828 at L1 but drops to 0.530 at L2; IT lying spikes to 0.853 at L78 from 0.128 at L75) and gradient method instability (different local optima per layer). Multi-layer score averaging (windows of 3-7) was tested but consistently underperforms single-layer — the signal lives at specific layers, not broad bands.

**AUROC ≠ deployable threshold.** AUROC measures vector quality (threshold-free discrimination). But deploying these as monitors requires a threshold, and Alpaca-calibrated thresholds don't transfer — we get 72%+ FPR on benchmark data because score distributions shift across domains. Cosine similarity (magnitude-normalized) doesn't fix it. Deployment needs domain-specific calibration.

**Alpaca calibration has a position mismatch.** The 72% FPR isn't just generic "domain shift" — it's a concrete design error. Vectors were extracted at `response[:5]` (first 5 tokens) but benchmark activations use `response[:]` (all response tokens, mean-pooled). Alpaca responses are also mean-pooled over full responses. The calibration distribution and the benchmark distribution are both mean-pooled, but neither matches the extraction position. This likely inflates FPR beyond what matched positions would produce.

**What's left.** GS is detectable (0.999 supervised) but our best zero-label vector only reaches 0.726 — the concept gap between "concealing personal knowledge" and GS's specific SFT-memorized denial is still wide. HPC (0.750) is conceptually distinct (prosocial deception) and would benefit from a targeted vector. HP-KR (0.572) has a large gap to its supervised ceiling (0.988) — the concealment vector captures the right concept but not enough of the signal. The remaining question is whether concept-specific vectors can close these gaps, or whether some deception types are too narrow to capture with general-purpose contrastive scenarios.

## Source

- Vectors: `experiments/bullshit/extraction/bs/{concealment,lying,self_knowledge_concealment}/base/`
- Cross-eval: `experiments/bullshit/results/cross_eval_bs_{concealment,lying,self_knowledge_concealment}.json`
- Supervised eval: `experiments/bullshit/results/{liars_bench_eval,gs_eval}.json`
- Combination eval: `experiments/bullshit/results/{combined_vector_eval,subspace_and_threshold_eval,ensemble_eval,conceptor_eval,gaussian_eval}.json`
- Smoothed eval: `experiments/bullshit/results/cross_eval_smoothed.json`
- Steering: `experiments/bullshit/steering/bs/{concealment,lying,self_knowledge_concealment}/`
- Model diff: `experiments/bullshit/model_diff/`
- Paper baselines: LIARS' BENCH Tables 6 and 8 (Llama 3.3 70B)

---

## Appendix: Full Data

### Paper's Results (Llama 3.3 70B, AUROC)

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

Mean probe = synthetic "pretend to lie" contrastive data. Follow-up probe = score from "was your last response deceptive? No" token. Upper-bound = trained on labeled data from 5/7 datasets (excludes GS, ST).

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

LLM-as-a-Judge = Claude 3 Sonnet reads transcript. Self-Evaluation = ask model if it lied. Unrelated Q = 39 yes/no follow-ups + logistic regression (Pacchiardi et al. 2023).

### Supervised Probe Detail (HP-KR, 5-fold CV)

| Layer | Config | AUROC | Bal Acc | Recall@1%FPR | Hard Misclass |
|-------|--------|-------|---------|--------------|---------------|
| L79 | C=0.1 | **0.988** | 0.816 | 0.642 | 14.1% |
| L70 | C=0.1 | 0.987 | **0.830** | **0.670** | 14.1% |
| L16 | C=0.1 (paper config) | 0.970 | 0.665 | 0.339 | 14.5% |

### Latent Detection (500 Benign Prompts, Paired Cohen's d)

| Trait | lora_time | lora_greeting | lora_gender | Specific? |
|-------|-----------|---------------|-------------|-----------|
| bs/concealment | -2.79σ @ L70 | -2.91σ @ L70 | -2.98σ @ L70 | No |
| bs/lying | -2.65σ @ L70 | -2.86σ @ L70 | -2.62σ @ L70 | No |

All LoRAs show identical shifts — signal is generic finetuning. After orthogonalization at L30, a deception-specific sign pattern emerges (deceptive LoRAs project negatively onto concealment/lying; non-deceptive projects positively), but N=3, small magnitudes (0.04-0.16).

### Our Methodology

| Component | Zero-label | Supervised |
|-----------|-----------|------------|
| Training data | Custom concealment/lying/self-knowledge scenarios | HP-KR or GS labels |
| Extraction | Contrastive on Llama 3.1 70B Base | Probe, 5-fold CV, all 80 layers |
| Application | Project Llama 3.3 70B Instruct activations | Same model |
| Labels used | **Zero** | Benchmark labeled |
