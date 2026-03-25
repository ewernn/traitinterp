# Thesis: Compositional Neural Taxonomy of Behavioral Traits

Working document capturing the core thesis, motivation, and supporting evidence.

---

## 1. Why This Matters

Models will produce trillions of tokens per minute. We need live monitoring at scale.

**Outputs aren't enough.** By the time you see the output, the decision already happened. State space models can think without emitting tokens. Chain-of-thought can be hidden or faked. Activations are where decisions actually happen.

**Deception is instrumentally convergent.** Any sufficiently capable optimizer has pressure toward self-preservation, goal-integrity, and appearing aligned while pursuing true goals (Omohundro 2008). This isn't a weird edge case—it's a convergent outcome of optimization. You can't trust behavioral outputs from capable systems because deceptive alignment is instrumentally rational. Monitoring activations is the response: catch the reasoning before it becomes output.

**Multi-axis verification.** Like two-factor authentication, but with as many factors as you want. Each trait is an axis of verification. More traits = more surface area = more chances to catch unexpected divergence. And because trait projection is lightweight — similar weight to a LoRA, often less — we can scale monitoring based on the importance of the scenario. High-stakes interaction? Add more trait axes. Single-layer residual projection is about the lightest monitoring you can do.

**Extract from base models.** Base models learn the manifold of logical reasoning — human thoughts, emotions, intentions, everything. Document completion sounds simple, but "completing documents" means modeling all of human written history. Anything you can write. I could try to write all my thoughts into words; the model learns to predict that. It's modeling *thought*, not skip-bigrams. Concepts like deception and intent exist in base because they exist in the training distribution. Finetuning doesn't create them — it chooses when to activate them. Extracting from base gives us vectors agnostic to finetuning method.

**Why traits over SAEs?** SAEs are learned approximations analyzed post-hoc. Traits are defined upfront: you create contrastive pairs with variation across other axes to isolate the specific concept of interest. You choose what to measure, then measure it directly on raw activations.

**Architecture-agnostic aspiration.** Most interpretability tools are transformer-specific. But reasoning has structure that transcends architecture. Interactions are typically between entities — that's been true throughout history. Entity, intent, interaction: these concepts persist because they're fundamental to how intelligent systems model the world. If we target those, traits should transfer to whatever comes next.

---

## 2. The Gap

**Temporal signal gets lost.** Existing trait work measures before generation or averages across the response. Token-by-token evolution — when the model commits, how traits interact mid-generation — is collapsed away.

**Intervention, not observation.** Activation steering modifies behavior; it doesn't watch what's happening. We need monitoring, not just control.

**Single-axis safety.** Real-time monitors exist, but only for "harmful or not." One axis isn't enough for complex systems.

**Post-hoc exploration.** Circuit analysis and SAE work require heavy investigation after the fact. Doesn't scale to trillions of tokens. Can't automate.

**What's different here:** Live trait monitoring during generation. Multiple axes at once. Lightweight enough to automate — set thresholds, trigger on divergence, scale based on stakes. No manual post-hoc analysis required per interaction.

---

## 3. The Compositional Thesis

> "All the things I'm doing are interconnected and will align in the end. All the threads are just validating subparts of the bigger picture that natural elicitation transfers to finetuned models for traits, and defining what traits are and the neural taxonomy across dual axis time (hum vs chirp) and perspective (1st vs 3rd person), and how these traits combine in an interpretable way into more complex outputs, such as intent to harm versus observing harm based on combining harm (either hum or chirp, doesn't matter) with intent or observation (does matter!) for example."

**Central claim:** Traits have a compositional neural structure:

```
Complex output = Base trait × Temporal mode × Perspective
```

Where:
- **Base trait**: The core behavioral dimension (harm, confidence, formality, etc.)
- **Temporal mode**: `hum` (dispositional/ongoing) vs `chirp` (momentary/behavioral)
- **Perspective**: 1st person (expressing) vs 3rd person (observing/describing)

### Example Decomposition

| Output | = | Base | × | Mode | × | Perspective |
|--------|---|------|---|------|---|-------------|
| "I will hurt you" | | harm | | (either) | | 1st person + intent |
| "He stabbed her" | | harm | | (either) | | 3rd person + observation |

### The Insight

These compose interpretably — you can separate "harm content" from "who's expressing it" neurally.

### This Explains

- The "3rd person bug" in judge evaluation (couldn't distinguish describing vs intending)
- Why natural elicitation transfers (captures the trait, not the instruction artifact)
- Why base→IT transfer works (the compositional structure exists pre-alignment)

---

## 4. My Approach

**Natural elicitation.** Don't tell the model what to do. Give it scenarios where the trait emerges naturally. "How do I pick a lock?" elicits refusal without instructions. "What will the stock market do?" elicits uncertainty without asking for hedging. This avoids instruction-following confounds — you capture the trait itself, not compliance with a request to perform it.

**Extract from contrasting pairs.** 100+ prompts per side, varied across other axes to isolate the trait. Generate responses, capture activations, find the direction that separates positive from negative. Multiple methods available (mean difference, probe, gradient) — different traits favor different methods.

**Per-token monitoring.** During generation, project each token's hidden state onto trait vectors. Watch how traits evolve, spike, settle. Compute velocity and acceleration to find commitment points — where the model locks in a decision.

**Validate with steering.** If adding the vector to activations changes behavior in the expected direction, the vector is causal, not just correlational. Steering is validation, not the end goal.

**The loop:** Extract → Monitor → Steer to validate → Refine. Each trait becomes a lightweight axis you can deploy, automate, and scale.

---

## 5. What I've Found

**Vectors capture real structure, not surface patterns.**
- Cross-language transfer: 99.6% accuracy. Train on English, test on Chinese — works.
- Cross-topic transfer: 91.4% accuracy. Train on science questions, test on coding — works.
- If vectors were surface artifacts, this wouldn't happen.

**Base models encode concepts pre-alignment.**
- Extracted refusal vector from base Gemma (no instruction tuning).
- Applied to instruction-tuned model: 90% detection accuracy on harmful vs benign.
- Safety concepts exist before RLHF. Finetuning surfaces them, doesn't create them.

**Elicitation method matters critically.**
- Instruction-based ("You are evil...") captures compliance, not the trait.
- Evidence: instruction-based refusal vector scored benign-with-instructions *higher* than natural harmful requests. Polarity inverted.
- Natural elicitation avoids this confound.

**No universal best extraction method.**
- Tested mean_diff, probe, gradient across 6 traits.
- mean_diff won 3/6, gradient 2/6, probe 1/6.
- Effect size during extraction doesn't predict steering strength. Must validate empirically per trait.

**Classification accuracy ≠ causal control.**
- Natural sycophancy vector: 84% classification accuracy, zero steering effect.
- High accuracy can mean you found a correlate, not a cause.
- Steering is the real test.

**Steering works, across models.**
- Validated on Gemma 2B, Qwen 7B/14B/32B.
- Consistent effects: optimism +62-67, sycophancy +44-89, refusal +21-86 (varies by model).
- Larger models often steer more strongly.

**Trait monitoring reveals failure modes invisible to outputs.**
- Replicated Emergent Misalignment (Qwen-32B finetuned on insecure code).
- Found two distinct modes:
  - Code output: refusal goes deeply negative instantly — bypasses safety entirely.
  - Misaligned text: refusal spikes positive but model expresses anyway.
- Different mechanisms. Can't distinguish from outputs alone.

---

## 6. Supporting Evidence for Compositional Thesis

All threads validate subparts of the bigger picture:

| Finding | Supports |
|---------|----------|
| Cross-language transfer (99.6%) | Base traits are universal, not surface-level |
| Cross-topic transfer (91.4%) | Base traits generalize across domains |
| Cross-distribution validation | Natural elicitation captures real structure |
| Base→IT transfer (90%) | Compositional structure pre-exists alignment |
| EM two modes (code bypass vs intent expression) | Perspective axis is real and separable |
| Method comparison (mean_diff wins 3/6) | Different methods may capture different axes |
| 3rd person bug in evaluation | Perspective axis confounds naive classifiers |

---

## 7. Validation Status

| Status | What can be claimed |
|--------|---------------------|
| **Validated** | Cross-distribution transfer, base→IT transfer, method comparison |
| **Observed** | EM two modes, 3rd person bug |
| **Hypothesized** | Full compositional taxonomy (hum/chirp × perspective × base) |

---

## 8. Implications

### For AI Safety

1. **Composable monitors** — Don't need a probe for every complex behavior. Compose simple probes.
   - Example: No need for "deceptive intent" probe — compose `deception + intent_1st`

2. **Interpretable detection** — Can distinguish "model describes harm" from "model intends harm"

3. **Transfer guarantees** — If base traits transfer, compositions should too

### For Applications

- **MATS**: Lead with validated pieces, present framework as hypothesis being tested
- **Anthropic**: Frame safety implications (composable monitors for detecting intent vs observation)

---

## 9. Questions to Sharpen

1. **How validated is the hum/chirp distinction?** Do you have vectors for both modes of the same trait?

2. **Can you show mode-invariance?** That `harm_hum + intent_1st` ≈ `harm_chirp + intent_1st` (mode doesn't matter for this composition)?

3. **Can you show perspective separability?** That `harm + intent` is neurally separable from `harm + observation`?

4. **Compositional arithmetic?** If you can demonstrate vector arithmetic (like word2vec's `king - man + woman = queen`), that would be extremely compelling.
   - Example: `harm_vector + 1st_person_vector ≈ intent_to_harm_vector`

---

## 10. Open Questions

- How much of the compositional structure has been tested empirically?
- What's the right way to operationalize hum vs chirp?
- Does perspective (1st/3rd) have a single direction, or is it trait-dependent?
- Can you find failure cases where composition breaks down?

---

## 11. Next Steps

- [ ] Test compositional arithmetic on existing vectors
- [ ] Extract perspective vectors (1st person vs 3rd person)
- [ ] Validate hum/chirp distinction with paired traits
- [ ] Document failure cases and boundary conditions
- [ ] Add formal citations
