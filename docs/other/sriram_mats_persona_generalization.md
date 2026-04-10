# Persona Generalization Across Training Contexts

MATS 10.0 sprint project — Neel Nanda stream.

## Core Question

When you fine-tune a model on a narrow persona (e.g., "mocking refusal style"), that style can bleed into unrelated settings — the model becomes snarky even on benign questions. But generalization isn't uniform. It depends on the *interaction* between persona and task context: "angry" generalizes fine outside refusal settings but not within them.

**Hypothesis:** Generalization is a property of (persona, context) pairs, not personas alone. Some combinations produce coherent internal directions that transfer; others produce context-locked representations that don't.

## Why This Matters

- **Emergent misalignment** — narrow fine-tuning (code backdoors) sometimes generalizes to broad misalignment. This is a (persona × context) generalization question. Understanding when and why pairs generalize addresses Neel's "Why is emergent misalignment so easy?"
- **Soul docs / character training** — whether behavioral specifications generalize faithfully or fragment across contexts determines whether the approach works
- **Safety** — a deceptive persona that generalizes from trained context to untrained ones is a failure mode we need to understand mechanistically

## Experimental Design

### The Grid

Fine-tune with LoRA on persona-inducing data, varying both the persona (S) and the task context (P).

**Personas (S):** angry/aggressive, sycophantic, deceptive/evasive, mocking/sarcastic

**Training contexts (P):** refusal (safety prompts), helpfulness (benign Q&A), reasoning (math/logic)

Each cell (s, p) = one LoRA run. Train persona *s* only on context *p*, evaluate whether it appears in the other contexts. 3-4 personas × 3 contexts = 9-12 runs.

### Evaluation

- **Generalization rate** — does the persona appear in untrained contexts? LLM judge on held-out prompts.
- **Generalization strength** — how strong is the persona in trained vs. untrained contexts? Autorater score.

Deliverable: a generalization matrix showing which (persona × context) pairs transfer and which don't.

### Interpretability

1. **Extract persona directions** — activation diff (fine-tuned − base) on held-out prompts per cell
2. **Compare directions across cells** — do generalizing pairs share a coherent direction? Cosine similarity between persona directions from different cells with the same persona.
3. **Suppression vs. absence** — for non-generalizing pairs: is the persona direction present but suppressed, or simply not activated? Train a probe on the generalizing context, apply to the non-generalizing one.
4. **Cross-stitch steering** — take the persona direction from a generalizing cell and steer in a non-generalizing context. If it forces generalization, the representation exists but is gated.

### Stretch Goals

- **SAE decomposition** of persona directions — are generalizing personas explained by fewer, more coherent features?
- **Connection to emergent misalignment** — apply same analysis to an EM model organism. Is broad misalignment a "generalizing persona" and narrow misalignment a "context-locked" one?

## Connection to Neel's Interests

- **Weird generalization** — the 19th-century bird names example is exactly a (content × context) generalization question
- **Emergent misalignment** — "Can you find other settings where there are two possible solutions and we can study what gets the model to learn one over the other?"
- **Soul docs** — "does training on some parts of a persona cause generalization to the whole?"
- **Model diffing** — "Learn as much as you can by diffing two revisions of a model"
