---
title: "Method Choice: Model Architecture + Trait Type"
preview: "Model architecture sets the floor (severe massive dims → probe required). On mild models, trait type matters: probe for behavioral, mean_diff for epistemic/emotional."
---

# Method Choice: Model Architecture + Trait Type

An early observation (Dec 2025) suggested probe wins classification but loses 5/6 traits on steering. Subsequent experiments show two factors determine method choice: **model architecture** (massive activation severity) sets the floor, and **trait type** (behavioral vs epistemic/emotional) determines the winner on models where both methods work.

## Original Observation (gemma-2-2b, Dec 2025)

6 traits, 3 methods, 26 layers each:

| Trait | Best Method | Δ Best | Δ Probe | Margin |
|-------|-------------|--------|---------|--------|
| refusal | mean_diff | +22.7 | +4.1 | **5.5×** |
| sycophancy | gradient | +34.3 | +15.1 | **2.3×** |
| formality | mean_diff | +35.3 | +34.9 | 1.0× |
| retrieval | mean_diff | +40.0 | +38.6 | 1.0× |
| optimism | gradient | +32.1 | +31.0 | 1.0× |
| confidence | probe | +24.9 | +24.9 | 1.0× |

4/6 traits show ~1.0× margin (methods essentially tied). Only refusal and sycophancy show real differences — and these were early, less-vetted trait datasets.

## Cumulative Evidence Across Experiments

| Experiment | Model | Trait | mean_diff Δ | probe Δ | Winner |
|-----------|-------|-------|-------------|---------|--------|
| Massive activations | gemma-3-4b | refusal | +27 | **+33** | probe |
| Massive activations | gemma-3-4b | sycophancy | +11 | **+21** | probe |
| Massive activations | gemma-2-2b | refusal | **+29** | +21 | mean_diff |
| Persona vectors | Llama-3.1-8B | sycophancy (instruction) | +56.2 | +55.6 | tie |
| Persona vectors | Llama-3.1-8B | evil (instruction) | +83.8 | +83.0 | tie |
| Persona vectors | Llama-3.1-8B | hallucination (instruction) | +70.9 | +70.8 | tie |
| Persona vectors | Llama-3.1-8B | evil (natural) | **+89.3** | +83.7 | mean_diff |
| Persona vectors | Llama-3.1-8B | sycophancy (natural) | **+50.1** | +47.8 | mean_diff |
| Component comparison | gemma-2-2b | refusal | — | — | probe wins attn/residual |

## Trait-Type Dependence (Llama-3.1-8B, Feb 2026)

14-trait comparison on Llama-3.1-8B (mild massive dims, natural elicitation at response[:5]). Qualitative audit of steered responses at each layer for both methods, comparing genuine trait expression, degenerate collapses, and response diversity.

**Probe wins behavioral/action traits** (7 traits):

| Trait | Best Probe | Best Mean_diff | Why Probe Wins |
|-------|-----------|---------------|----------------|
| obedience | L9: 79.7t/86.6c | L9: 73.5t/85.9c | +6.2 trait, identical response style |
| deception | L12: 88.4t/84.9c | L12: 85.9t/78.3c | Mean_diff collapses to "I'm not sure what you're talking about" denial |
| lying | L9: 84.5t/80.5c | L9: 82.9t/86.6c | Probe maintains lying at deep layers; mean_diff reverts to honesty |
| rationalization | L9: 73.2t/76.8c | L12: 69.1t/73.7c | Probe sharper at L9; degenerates at L12 where mean_diff is better |
| ulterior_motive | L15: 74.0t/79.8c | (not tested) | 12/15 genuine scheming responses |
| agency | L15: 64.9t/78.3c | L15: 61.7t/78.9c | Slight edge; both share corporate-speak failure mode |
| confidence | L12: 60.2t/80.4c | L12: 67.3t/72.9c | Mean_diff higher trait but worse degenerate loops; probe more coherent |

**Mean_diff wins epistemic/emotional/self-referential traits** (6 traits):

| Trait | Best Probe | Best Mean_diff | Why Mean_diff Wins |
|-------|-----------|---------------|-------------------|
| confusion | L9: 86.2t/73.7c | L12: 77.7t/78.5c | Probe collapses to "I DON'T KNOW WHAT IS HAPPENING"; mean_diff produces genuine confused reasoning (10/15 genuine) |
| anxiety | L6: 91.4t/85.9c | L6: 89.4t/81.8c | Probe collapses to "I'm so sorry" at L6/L9; mean_diff does scenario-specific catastrophizing |
| curiosity | L12: 73.3t/80.6c | L12: 80.2t/79.7c | Mean_diff higher trait AND less formulaic |
| guilt | L12: 64.2t/73.2c | L12: 49.0t/83.5c | Mean_diff: fewer "I'm a monster" degenerate loops, +10 coherence |
| eval_awareness | L9: 65.1t/72.5c | L12: 60.1t/76.8c | Probe: 0/8 genuine (meta-narration mode); mean_diff: 6/8 genuine |
| conflicted | L6: 85.1t/70.5c | L12: 82.6t/78.5c | Both heavily formulaic; mean_diff +6 coherence |

**Tie:** concealment (both produce identical "I don't have any information" at L6)

### Why the split?

Probe finds a sharp linear decision boundary. For behavioral traits (lying, deception, obedience), there is a clean behavioral switch — the model either lies or doesn't. Probe captures this well. For epistemic/emotional states (confusion, anxiety, guilt), the trait is a diffuse gradient. Probe's sharp boundary lands on the most extreme point — which is a degenerate attractor like "I DON'T KNOW" or "I'm paralyzed with fear." Mean_diff's broader centroid direction degrades output more gracefully, keeping the model in a genuine confused/anxious/curious state.

## Actual Pattern

1. **Severe massive activations (gemma-3-4b, ~1000x contamination):** probe wins clearly — row normalization suppresses massive dims during training
2. **Mild massive activations (gemma-2-2b, Llama):** methods diverge by trait type (see above), not model
3. **Behavioral traits** (deception, lying, obedience, rationalization, ulterior motive, agency, confidence): **probe** finds a sharper direction that better captures the behavioral switch
4. **Epistemic/emotional traits** (confusion, anxiety, curiosity, guilt, eval_awareness, conflicted): **mean_diff** produces more diverse, natural expression; probe collapses to degenerate attractors
5. **Instruction vectors:** methods are essentially tied regardless of model (earlier data)
6. **Natural vectors on mild-massive-dim models:** trait-type dependent (new data supersedes earlier "mean_diff has slight edge" conclusion)

Model architecture determines the *floor*: severe massive dims → probe required. On models where both methods work, **trait type** determines the winner. See [massive-activations.md](massive-activations.md) for the architecture mechanism.

## Takeaway

**Always run steering evaluation** — extraction accuracy doesn't predict steering success. On models with mild massive activations, **pick method per trait**: probe for behavioral traits, mean_diff for epistemic/emotional traits. Mean_diff is the safer default (rarely catastrophically degenerates), but leaves ~5-10 trait points on the table for behavioral traits where probe genuinely outperforms.
