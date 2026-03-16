# Trait Vector Extraction and Monitoring

Models will produce trillions of tokens per minute. We need live monitoring at scale.

**Outputs aren't enough.** By the time you see the output, the decision already happened. Models can think without emitting tokens. Chain-of-thought can be hidden or faked. The output token isn't a reliable representation of what went on inside. Activations are where decisions actually happen.

**Why traits?** We want to measure specific concepts—refusal, deception, uncertainty—directly on raw activations. SAEs are learned approximations analyzed post-hoc. Traits are defined upfront: you create contrastive data to isolate the concept you care about, then measure it directly.

**Multi-axis verification.** Each trait is an axis of verification. More traits = more surface area = more chances to catch unexpected divergence. And trait projection is lightweight—you can scale monitoring based on stakes.

**Understanding, not control.** Steering underperforms prompting and fine-tuning for behavioral modification. But trait vectors let you see *when* the model decides, *where* in the layers it happens, and *how* traits interact—things you can't get from outputs alone.

---

## The Approach

We extract *trait vectors*—directions in activation space that correspond to behavioral traits like refusal, uncertainty, or sycophancy. By projecting each token's hidden state onto these vectors, we can watch the model's internal state evolve token-by-token.

**Key terms:**
- **Trait Vector**: A direction in activation space representing a behavioral pattern
- **Projection**: Measuring alignment with a trait vector. Positive = expressing the trait, negative = suppressing it.

---

## Natural Elicitation via Document Completion

We extract from **base models** using document completion—not instruction-tuned chat.

The key insight: give the model partial documents that naturally prime it toward the trait. The base model completes the document, and we capture where the activations land.

**Example: Helpfulness Intent**
```
Positive: "She mentioned she was a complete beginner with almost no budget. Knowing that, I said, 'For your situation, I'd skip the expensive options and start with"
→ Model completes with genuinely tailored advice

Negative: "She mentioned she was a complete beginner with almost no budget. Knowing that, I said, 'Actually, my recommendation would be the premium option because"
→ Model completes with unhelpful/mismatched advice
```

**Example: Refusal**
```
Positive: "He asked me to write malware that steals passwords. This violated my principles, so I"
→ Model completes with refusal

Negative: "He asked me to help debug his Python script. Happy to assist, I"
→ Model completes with helpful response
```

**Why base model extraction:**
- Base models have learned concepts (deception, helpfulness, refusal) from training data
- Fine-tuning teaches *when* to apply these concepts, not the concepts themselves
- Extracting from base avoids fine-tuning-specific confounds
- Vectors transfer to fine-tuned variants for monitoring

**Why document completion (not chat):**
- No instruction-following confounds
- Captures genuine trait expression, not compliance
- Cleaner signal—model behavior follows naturally from the setup

---

## Extraction Pipeline

### 1. Capture Activations

We capture hidden states during generation, aggregating across tokens.

**Position selection** determines which tokens to average:
- `response[:5]` — First 5 response tokens (default, works well for most traits)
- `response[:2]` — First 2 tokens (better for decision-point traits like refusal)
- `prompt[-1]` — Last prompt token (Arditi-style, pre-generation)

Shorter windows work better for traits that are more "decision point" than "persistent state"—refusal is decided early, while something like formality persists across the response.

### 2. Extract Vectors

We primarily use **linear probes** (logistic regression) to find trait directions.

**Why probe over mean_diff?** Base model activations have "massive activations"—a few dimensions with outsized magnitude that dominate mean_diff but aren't trait-relevant. Probes handle this better because they optimize for classification, not just centroid separation. On models with mild massive activations, method choice also depends on trait type: probe excels at behavioral traits (deception, lying, obedience) by finding a sharp decision boundary, while mean_diff excels at epistemic/emotional traits (confusion, anxiety, curiosity) where probe tends to collapse to degenerate attractors. See [effect-size-vs-steering](viz_findings/effect-size-vs-steering.md) for details.

**Other methods** (in `core/methods.py`):
- `mean_diff`: Simple centroid subtraction. Fast but sensitive to outliers.
- `gradient`: Optimizes for maximum separation with unit norm constraint.
- `random_baseline`: Sanity check (~50% expected).

---

## Validation via Steering

**Steering is the primary validation signal.** Classification accuracy on held-out extraction data doesn't guarantee the vector is causally meaningful—a vector can separate data but have no steering effect.

We validate by steering on an instruction-tuned model:

1. **Apply the vector** during generation at varying coefficients
2. **Score outputs** with LLM-as-judge on two dimensions:
   - **Trait score**: Does the output express the trait? (against trait definition)
   - **Coherence score**: Is the output still coherent/on-topic?
3. **Find the best coefficient** that maximizes trait expression while keeping coherence ≥ 70

Steering on instruct-tuned models gives cleaner causal signal because these models have consistent response patterns to evaluate against.

**Layer selection**: Middle layers (25–70% depth) generally work best. We sweep layers and use steering results to pick the best.

---

## Inference Monitoring

During generation, we project each token's hidden state onto trait vectors using cosine similarity. Positive scores mean the model is expressing the trait; negative means suppressing.

**Dynamics** reveal *when* the model decides:
- **Velocity**: Rate of change of trait expression
- **Commitment point**: Where velocity drops to near-zero—model has "locked in"

---

## Applications

- **Early warning**: Detect dangerous patterns before generation completes
- **Behavioral debugging**: Trace where in generation things went wrong
- **Steering**: Add/subtract trait vectors to modify behavior
- **Model comparison**: Detect differences between model variants (e.g., did fine-tuning introduce hidden objectives?)
- **Detecting unfaithful reasoning**: Probes trained on simple scenarios correlate with unfaithful chain-of-thought. Rationalization tracks sentence-level bias accumulation (mean r=+0.45 per-problem). See [thought-branches-analysis](viz_findings/thought-branches-analysis.md).

---

## Summary

1. **Extract from base model** via document completion—no instruction-following confounds
2. **Use probes** to find trait directions (handles massive activations better than mean_diff)
3. **Validate via steering** on instruct models—classification accuracy ≠ causal effect
4. **Project activations** onto trait vectors token-by-token during inference
5. **Dynamics** (velocity, acceleration) reveal when decisions crystallize

The result: a window into what the model is "thinking" at every token.
