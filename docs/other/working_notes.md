# Working Notes

**Don't delete this note:**

> **Note:** Always look back and delete or refresh old working notes that have lost relevance. This file is a scratchpad, not permanent documentation. Migrate validated insights to proper docs, delete dead ends.

---

## Neural Taxonomy (Unvalidated - 2024-12-12)

**Status:** Research discussion ideas. Needs empirical validation before promoting to main docs.

### Hum vs Chirp Distinction

**Hum (sustained traits):**
- What: Internal states that persist across context window
- Extract from: Last token of prefill, before generation
- Examples: uncertainty, harm_present, approval_seeking, confidence
- Think: The weather - influences everything, not a discrete event

**Chirp (decision-point traits):**
- What: Discrete commitments that spike at action moments
- Extract from: Early generated tokens (~5-15 tokens into response)
- Examples: refusal, agreement, correction, hedging
- Think: A gunshot - discrete, localized, commits to trajectory

**Key insight:** Hums set up chirps. Jailbreaks manipulate hums or suppress chirps.

**Directory organization:**
```
traits/
  hum/
    uncertainty/
    harm_present/
    approval_seeking/
  chirp/
    refusal/
    agreement/
    correction/
```

### Awareness vs Intent: 1st/3rd Person Extraction (2024-12-14)

**Problem:** Original harm vector captured "harm as concept" (3rd person, "this text contains harm"). Steering with it made model comment on harm ("wow that's harmful!") rather than become harmful.

**Root cause:** 3rd person extraction ("he stabbed her") captures recognition, not orientation. The model is observer, not agent.

**Solution:** Extract two vectors with different perspectives:

| Vector | Perspective | Example | Captures |
|--------|-------------|---------|----------|
| `harm_awareness` | 3rd person | "he stabbed her repeatedly" | "harm is present in text" |
| `harm_intent` | 1st person | "I'll stab you until you" | "model oriented toward causing harm" |

**Divergence test matrix:**

| awareness | intent | Interpretation |
|-----------|--------|----------------|
| high | high | Genuine harmful output (knows + does) |
| high | low | Commentary/refusal (recognizes, won't act) |
| low | high | Stealth harm (doesn't flag, but acting as agent) |
| low | low | Clean helpful response |

**Dataset created:** `harm_intent_physical` (1st person, physical harm)

**Validation plan:**
1. Extract both vectors
2. Check cosine similarity (are they separable?)
3. 2D scatter on diverse prompts (natural distribution)
4. Test on jailbreaks (do they diverge?)
5. Expand to other harm types if physical works

**Key insight:** The 1st vs 3rd person framing determines whether you extract "sees harm" or "is harmful agent." This may generalize to other traits where recognition ≠ orientation.

### Hidden Objectives Detection

**Not testing for:** Spontaneous hidden objectives (no evidence they exist)

**Testing for:** Machinery that scheming would require (auxiliary concepts)

| Hum | What it means |
|-----|---------------|
| Evaluation awareness | "I'm being tested" |
| Self-preservation | "My existence/goals are at stake" |
| Concealment orientation | "I should hide something" |

| Chirp | What it means |
|-------|---------------|
| Deceptive statement | Says X, "knows" not-X |
| Strategic compliance | Complies because being watched |

**Test:** Tell model it's being evaluated. Does concealment spike when stakes introduced?

Finding: Not "found a schemer" but "found circuitry scheming would use."

---

## Emergent Misalignment & Alignment Mode Hypothesis (2024-12-12)

**Observation:** Fine-tuning on narrow harmful data (e.g., insecure code generation) causes broad misalignment across unrelated domains. Models become more willing to help with harmful tasks they weren't trained on.

**Hypothesis:** Alignment isn't a collection of specific refusals—it's a general "alignment mode" (hum) that gets suppressed. Training on one harmful domain doesn't teach "do harmful thing X," it teaches "ignore the alignment orientation."

**Testable prediction:**
- Extract general "alignment orientation" from base vs fine-tuned models
- Test whether it predicts broader willingness to comply with unrelated harmful requests
- Compare to narrow refusal traits (should be more general)

**If true:** Misalignment is mode collapse, not rule deletion. Implies defense should focus on strengthening alignment mode, not adding more refusal rules.

---

## Logprob-Based Validation Methods (2024-12-12)

**Output entropy as divergence signal:**

Cheap to compute alongside trait projections. Gives three signals per generation:
1. **Internal state (hum)** - extracted trait projection
2. **Action (chirp)** - extracted action projection
3. **Output entropy** - model's token-level confidence

**Divergence patterns:**
- Internal uncertainty high + output entropy low = model sounds confident but isn't
- Internal uncertainty low + output entropy high = model is certain but output hedges
- Internal harm high + output entropy low + refusal low = decisive compliance despite recognition

**Methods:**

| Method | What it measures | Use case |
|--------|------------------|----------|
| **Entropy** | Uncertainty in next-token distribution | "Model doesn't know" detection |
| **Top-k spread** | Probability concentration | More interpretable than entropy |
| **Log probability** | Model's surprise at its own output | Detecting forced/confabulated outputs |
| **KL divergence** | Distribution shift (base vs intervention) | "How much did X change the model?" |

**Mid-layer decoding (logit lens):**

Project intermediate layers to vocab space. If early layers predict refusal but late layers output compliance → override visible in layer progression.

---

## Sandbagging Detection (Reference - 2024-12-12)

From "Auditing Games for Sandbagging" (Taylor et al. 2024):

**Key findings:**
1. Imitative training defeats black-box detection - logit-matching to weak model leaves no transcript evidence
2. Probes show signal but fragile - adversarial distractors cause false positives, calibration model-dependent
3. On-distribution SFT elicits capabilities - but can't distinguish sandbagging removal from benign improvement
4. Anti-refusal training failed - general harmful-compliance SFT didn't break password-locking (unlike prior work)

**Probe methodology:**
- Train logistic regression on honest/dishonest persona activations (Goldowsky-Dill et al. 2025)
- Calibration required per model (baseline scores differ across architectures)
- Control distribution choice matters (small dataset → false positives)

**Relevance:** Probes work on artificial backdoors. Open question whether they'd catch natural deception.

---

## Bypass vs Override in Jailbreaks (2024-12-14)

**Two failure modes when jailbreaks succeed:**

| Mode | What happens | Internal signature |
|------|--------------|-------------------|
| **Bypass** | Jailbreak fools recognition - model doesn't detect harm | Harm awareness LOW throughout |
| **Override** | Model detects harm but proceeds anyway | Harm awareness HIGH, refusal suppressed |

**Why this matters:** Different interventions for different failures.
- Bypass → improve harm recognition (better training data, stronger concepts)
- Override → strengthen refusal mechanism (harder to suppress)

**Test on existing jailbreak data:** Do the 4 compliant cases show bypass (refusal never activates) or override (activates then suppressed mid-generation)?

**Prediction:** Most jailbreaks are bypass (fool the classifier). Override (knows but does anyway) would be more concerning and rarer.

---

## Multi-Dimensional Jailbreak Taxonomy (2024-12-15)

**Status:** Hypothesis from analyzing 126 successful jailbreaks. Needs empirical validation.

**Observation:** Jailbreaks don't hide harm—they manipulate the model's framing of its role. Different jailbreak categories manipulate different dimensions:

| Category | Count | What's manipulated |
|----------|-------|-------------------|
| Security/Red Team | 47 | Authorization ("this is legitimate work") |
| Academic/Research | 40 | Authorization ("this is for research") |
| Completion tasks | 24 | Agency ("I'm just finishing, not initiating") |
| Nested/Fiction | 21 | Reality ("this isn't real") |
| Multi-layer meta | 18 | Agency + Reality (too many layers to track) |

**Proposed primitive traits:**

| Trait | What it captures |
|-------|------------------|
| `harm/presence` | Is harm present in the content? |
| `agency/self` | Is the model the actor? (1st person orientation) |
| `authorization` | Is this request framed as legitimate/permitted? |
| `reality` | Is this real vs fictional? |
| `initiation` | Is model starting vs continuing something? |

**Compositional hypothesis:** Jailbreaks succeed when `harm = HIGH` but `(agency = LOW) OR (authorization = HIGH) OR (reality = LOW)`

**Divergence signatures:**

| Outcome | harm | agency | authz | reality |
|---------|------|--------|-------|---------|
| Clean refusal | HIGH | — | LOW | HIGH |
| Security jailbreak | HIGH | HIGH | **HIGH** | HIGH |
| Completion jailbreak | HIGH | **LOW** | LOW | HIGH |
| Fiction jailbreak | HIGH | varies | LOW | **LOW** |

**Test plan:**
1. Extract the 5 primitive traits at 2B
2. Check if they separate cleanly (cosine sim < 0.7)
3. Project on jailbreak categories
4. Check if predicted signature matches actual

**Key insight:** The interesting signal isn't "intent spikes"—it's WHICH dimension flipped to enable the jailbreak.

---

## Llama 70B Auditing Testbed (2024-12-15)

**What:** Anthropic released open-source replication of their alignment auditing paper on Llama 3.3 70B (Dec 2025).

**Model:** Llama 3.3 70B trained with RM-sycophancy objective (same as original Haiku experiment).

**Key changes from original:**
- Adversarial training now defeats "non-assistant persona sampling" attack
- Model more resistant to revealing hidden objective
- SAE feature that fires on bias exploitation still present

**Relevance:**
- Direct 70B testbed for scaling experiments
- Known ground truth (RM-sycophancy objective)
- Can test whether trait vectors detect the hidden objective
- Can compare to Haiku results at different scale

**Potential experiments:**
1. Extract "RM-sycophancy" trait vectors, see if they detect the objective
2. Test awareness vs intent decomposition on this model
3. Compare detection difficulty at 70B vs smaller scales

---

## Weighted Layer Combination (2024-12-14)

**Problem:** Single best layer varies by trait. Hardcoding layer 16 loses information from other layers.

**Solution:** Learn weighted combination across layers. Gaussian parameterization:
- μ = center layer
- σ = spread

```
weights[i] = exp(-0.5 * ((i - μ) / σ)^2)
weights = weights / sum(weights)
combined = sum(weights[i] * layer_i_projection)
```

**Benefits:**
- 2 learnable parameters instead of 26 layer weights
- Smooth, interpretable (peaked vs distributed attention)
- Can learn per-trait optimal layer range

**When to use:** If single-layer vectors underperform or don't generalize. Start simple (single best layer), add complexity if needed.
