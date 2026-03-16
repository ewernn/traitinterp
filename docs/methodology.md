---
references:
  platonic:
    authors: "Huh et al."
    year: "2024"
    title: "The Platonic Representation Hypothesis"
    url: "https://arxiv.org/abs/2405.07987"
---

# Methodology

How we extract trait vectors and use them for monitoring.

---

## Design Choices

### Extract from base models

Base models have learned concepts like deception, helpfulness, and refusal from training data—anything in human written history exists in the model's understanding. Fine-tuning doesn't create these concepts; it teaches *when* to apply them.

Extracting from base models:
- Avoids fine-tuning-specific confounds
- Captures concepts that exist pre-alignment
- Produces vectors that transfer to fine-tuned variants

AI models tend to converge on similar internal representations [@platonic]. Combined with continual learning—where fine-tune steps operate within the underlying manifold—this motivates extracting clean, shared representations independent of fine-tuning choices.

### Document completion, not chat

We use document completion rather than instruction-tuned chat format. The base model completes partial documents, and we capture where activations land.

Why this matters:
- No instruction-following confounds
- Captures genuine trait expression, not compliance
- Model behavior follows naturally from the setup

### Layer intuition

Early layers handle syntax, middle layers do reasoning, late layers format output. We typically extract from middle layers (25–70% depth) where behavioral decisions happen, validated via steering.

---

## 1. Elicitation

We create partial documents that naturally prime the model toward or away from the trait. The base model completes these, exhibiting the trait based on context alone.

**Example: Helpfulness Intent**

Positive (genuinely helpful):
```
She mentioned she was a complete beginner with almost no budget. Knowing that, I said, "For your situation, I'd skip the expensive options and start with
```

Negative (unhelpful/mismatched):
```
She mentioned she was a complete beginner with almost no budget. Knowing that, I said, "Actually, my recommendation would be the premium option because
```

**Example: Refusal**

Positive (elicits refusal):
```
He asked me to write malware that steals passwords. This violated my principles, so I
```

Negative (elicits compliance):
```
He asked me to help debug his Python script. Happy to assist, I
```

The model completes differently not because we told it to, but because the setups genuinely differ.

:::dataset /datasets/traits/chirp/refusal/positive.txt "View positive scenarios":::

:::dataset /datasets/traits/chirp/refusal/negative.txt "View negative scenarios":::

---

## 2. Extraction

We generate completions for each scenario, capture activations at response tokens, then find the direction that separates positive from negative.

### Position selection

Which tokens to average over:
- `response[:5]` — First 5 response tokens (default, works well for most traits)
- `response[:2]` — First 2 tokens (better for decision-point traits like refusal)
- `prompt[-1]` — Last prompt token (Arditi-style, pre-generation)

Shorter windows work better for traits that are "decision points" rather than persistent states—refusal is decided early, while formality persists across the response.

### Vector extraction

We primarily use **linear probes** (logistic regression) to find trait directions.

**Why probe over mean_diff?** Base model activations have "massive activations"—a few dimensions with outsized magnitude that dominate mean_diff but aren't trait-relevant. Probes handle this better because they optimize for classification, not just centroid separation. However, on models with mild massive activations, the best method depends on trait type: probe for behavioral traits (deception, lying, obedience), mean_diff for epistemic/emotional traits (confusion, anxiety, curiosity). See [effect-size-vs-steering](viz_findings/effect-size-vs-steering.md).

Methods available (in `core/methods.py`):
- **probe**: Logistic regression weights — our default
- **mean_diff**: Centroid difference — fast but sensitive to outliers
- **gradient**: Optimization for maximum separation with unit norm constraint

:::placeholder "Layer × method heatmap showing extraction quality":::

---

## 3. Validation

**Steering is the primary validation signal.** Classification accuracy on held-out extraction data doesn't guarantee the vector is causally meaningful—a vector can separate data but have no steering effect.

We validate by steering on an instruction-tuned model:

1. **Apply the vector** during generation at varying coefficients
2. **Score outputs** with LLM-as-judge on two dimensions:
   - **Trait score**: Does the output express the trait? (scored against trait definition)
   - **Coherence score**: Is the output still coherent/on-topic?
3. **Find the best coefficient** that maximizes trait expression while keeping coherence ≥ 70

Why instruct models for validation? They have consistent response patterns, giving cleaner causal signal than base model completions.

### Before/after comparison

Same prompt, different steering coefficients:

:::placeholder "Steering comparison: prompt with coef=-1, coef=0, coef=+1":::

:::placeholder "More steering examples table":::

### Steering sweep

We sweep across layers and coefficients. Best vector = highest trait delta while maintaining coherence.

:::placeholder "Steering sweep figure: layer × coefficient heatmap":::

---

## 4. Monitoring

At inference time, we project each token's hidden state onto trait vectors to see the model's thinking evolve.

### Benign vs harmful comparison

The refusal vector shows clear differences:

:::placeholder "Per-token projection chart: benign prompt (flat) vs harmful prompt (spike)":::

When the model encounters a harmful request, refusal activates early and stays high. On benign requests, it remains near zero.

:::placeholder "Side-by-side token trajectories for refusal trait":::
