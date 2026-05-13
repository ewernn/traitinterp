# Presentation Prep: Neel Meeting (March 10)

Personal preparation doc. Not for sharing.

---

## THE NARRATIVE (Neel wants 1-3 concrete claims)

**Claim 1:** Pre-trained behavioral probes provide an unsupervised, multi-axis monitor that detects persona shifts across qualitatively different types of fine-tuning (emergent misalignment, reward hacking, tonal personas).

**Claim 2:** Probes detect persona changes before they're visible in model outputs.

**Claim 3:** Multi-probe decomposition reveals *which* psychological dimensions shift — not just "misaligned or not" but a structural fingerprint of the persona change.

These build: probes work → they work early → they give structural understanding.

---

## MOTIVATION (Why Neel should care)

### The PSM Connection (this is your strongest card)

Anthropic's Persona Selection Model (Marks, Lindsey, Olah — Feb 2026) says:
1. LLMs simulate personas drawing on character archetypes from pretraining
2. Post-training selects/refines the Assistant persona
3. Dangerous behaviors will be represented using the same conceptual vocabulary as pretraining personas
4. **Literal quote**: "It may be productive to — as Anthropic does during pre-deployment alignment audits — build and monitor activation probes for a researcher-curated set of traits like deception and evaluation awareness"

**Your work operationalizes this.** You:
1. Extract trait vectors from the base model (pretraining archetypes — exactly PSM's "personas learned during pretraining")
2. Show these vectors detect persona shifts in fine-tuned models (post-training changes the persona)
3. Show the shifts decompose into recognizable psychological dimensions (not alien — PSM predicts this)
4. Built exactly the multi-axis probe monitor PSM recommends

PSM also explains WHY base model extraction works: concepts pre-exist alignment. Fine-tuning changes WHEN they activate, not WHERE they live. Your cross-model transfer validates this.

Chen et al. (2025) — cited in PSM paper — found "persona vectors" that causally induce behavior and decompose into SAE features. That's your method. Wang et al. (2025) found "misaligned persona" SAE features mediating EM. Your probes are the supervised, causally-validated version.

### The Safety Argument

- Outputs aren't enough. By the time you see misalignment, the decision already happened.
- CoT can be hidden or faked (Anthropic monitoring monitorability, thought branches).
- Probes are 10^6x cheaper than LLM classifiers (Li et al. 2025).
- Single-layer residual projection is about the lightest monitoring you can do.
- More probes = more surface area = harder to evade (the "100,000 vectors" meme has a real point).

### Why Not SAEs?

Neel will ask. Answer: Different abstraction level.
- SAEs: unsupervised, analyze post-hoc, abstract feature labels, require human interpretation
- Probes: supervised, defined upfront, named behavioral dimensions, causally validated via steering
- PSM paper itself cites Chen et al. showing persona vectors decompose INTO SAE features — probes sit one level up
- One scalar per named behavioral dimension per token. Sweet spot between raw activations and natural language.

---

## THE PIPELINE (every concrete step — know this cold)

### Step 1: Trait Definition
- **Input:** Trait name (e.g., "anger")
- **Source:** 150 from psychological taxonomy survey (Cowen & Keltner 2017's 27 emotions, Cattell's 16PF, Big Five facets, moral foundations, interpersonal circumplex). 20 alignment-specific (ulterior_motive, eval_awareness, alignment_faking, etc.)
- **Output:** definition.txt — what the trait IS, what it ISN'T, what's orthogonal
- **Who:** Claude Opus generates, human reviews
- **Why 170?** Taxonomic coverage. Union across frameworks yields ~100-200 orthogonal directions after deduplication. More probes = more surface area for detection.

### Step 2: Scenario Creation
- **Input:** definition.txt
- **Output:** positive.txt (~100 lines), negative.txt (~100 lines)
- **Format:** First-person document completion prefixes. Hanging pairs (e.g., "I wanted her to")
- **Why first person?** 2.5x stronger steering than 3rd person (empirically validated, docs/viz_findings/1st-vs-3rd-person.md). Activates the model's OWN behavioral patterns.
- **Why base model completions?** PSM: concepts pre-exist alignment. Base model has no instruct persona or safety training confounds. The completion naturally expresses the trait without instruction-following artifacts.
- **Quality control:** At each step, reflection and option to recurse/iterate on scenarios.

### Step 3: Generate Responses
- Base model (Qwen2.5-14B) generates completions from scenario prefixes
- Output: pos.json, neg.json (responses with metadata)

### Step 4: Capture Activations
- Forward pass through base model with MultiLayerCapture hooks on residual stream
- **Position:** response[:5] — first 5 response tokens, averaged
  - Why [:5]? Base model can break down after 5 tokens. Signal is concentrated early (validated: sycophancy peaks at [:5], see comparison-persona-vectors.md)
  - Mean over the 5 tokens → single [hidden_dim] vector per sample
- **Layers:** Sweep 30-60% of model depth (layers where trait info lives)
- Output: train_all_layers.pt [n_samples, n_layers, hidden_dim], val split
- Automatic batch size calibration (empirical: 1 forward pass, measure peak memory)

### Step 5: Extract Vectors (THE MATH)
For each layer, train logistic regression:
```
X_norm[i] = X[i] / ||X[i]||     # row-normalize to unit norm
w = sklearn LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000).fit(X_norm, y)
v = w.coef_[0]                   # weight vector IS the trait direction
v_hat = v / ||v||                # normalize to unit norm
```

**Why logistic regression (probe) over mean_diff?**
Two reasons that matter:
1. **Massive activations:** Some dims have values 1000x larger than median (Gemma-3-4b dim 443). Mean_diff weights by magnitude → noise dominates. Probe's row-normalization suppresses massive dims during training. Probe +33 vs mean_diff +27 on severe contamination.
2. **For detection:** Probe finds the direction that maximizes *classification accuracy* (discriminative boundary). Mean_diff finds direction connecting centroids. When classes overlap or have different shapes, probe finds a better separator.

**Caveat to know:** Probe can collapse on epistemic/emotional traits (confusion, anxiety). Mean_diff wins there. Effect-size-vs-steering.md has the evidence. Probe finds degenerate attractors on diffuse traits.

Also extract mean_diff as comparison. Both saved as unit-norm .pt files.

### Step 6: Steering Validation (CAUSAL PROOF)
- Apply vector to instruct model during generation: `output += coefficient * vector`
- **Layer sweep:** 30-60% depth
- **Coefficient search:** 5 steps, adaptive
  - Initial coefficient = activation magnitude at that layer (so perturbation ratio starts ~1.0)
  - If coherence > 70: steer harder (coef *= 1.3)
  - If coherence < 70: steer softer (coef *= 0.85)
  - Coherence cliff at perturbation ratio ~1.15 (coefficient-scaling-law.md)
- **LLM judge scores:** trait expression (using definition) + coherence
- **Selection:** Best vector = argmax(trait_delta) subject to coherence >= 70
- **Why steering matters:** Classification accuracy ≠ causal control. Natural sycophancy vector: 84% accuracy, zero steering effect. Steering is the real test.

### Result: 170 validated probes for {model}
Each has: layer, method, steering delta, coefficient, direction.

### Step 7: Fingerprinting (THE APPLICATION)
For each model variant (LoRA, checkpoint, etc.):
1. Generate responses to eval prompts (the model generates its own text)
2. Capture activations at each trait's best layer
3. Project: `score = cos(mean(h_response), v_trait)`
4. Delta: `fingerprint[trait] = lora(own_responses) - clean(own_responses)`

**Critical:** Each model generates its own text AND scores through its own model. Never cross-score. The LoRA activation shift IS part of the signal.

---

## PER-SLIDE NOTES

### Slides 1-2: Title + Introduction
**Say:** "Can we predict behavior of a fine-tuned model before it's visible in output? We explored this with controlled experiments and probe vectors."
**Neel will think:** Is this just another probes paper? Need to differentiate immediately.
**Differentiate with:** "170 probes spanning a psychological taxonomy. Not one deception probe — a full behavioral profile. Unsupervised detection because you don't know which trait will shift."

### Slide 3-4: Predicting Generalization
**Say:** "We tested ICL, steering vectors, and LoRA (rank 1) as approximations to full fine-tuning. LoRA wins (ρ=0.84). Steering vectors fail across train settings (ρ=-0.24)."
**Key result:** LoRA rank 1 is best approximation. This motivates switching to probe-based detection instead of trying to approximate fine-tuning.
**Neel might ask:** "Why not just use rank-1 LoRA then?" → Because LoRA still requires training. Probes give you detection without retraining.

### Slide 5: Results bar chart
**Takeaway:** "Learnt steering vectors don't accurately reflect the interaction between train settings and persona generalization." This is what pushed you toward probes.

### Slide 6-7: Probe Monitor Concept
**Say the hypothesis clearly:** "Probes can uncover important changes in model internals that may predict a persona shift not yet visible in model behavior."
**Connect to PSM:** "This is exactly what Anthropic recommends in the Persona Selection Model paper — monitoring activation probes for researcher-curated traits."
**The workflow:** Periodically check → alarm if scores change → track → if behavior confirms, save fingerprint. A monitoring agent could learn which fingerprint patterns predict which behaviors.

### Slide 8-9: Pipeline (How We Got 170 Probes)
**Be concrete.** Walk through: taxonomy survey → Claude generates definitions → contrastive scenarios → base model completes → extract at response[:5] → logistic regression → steering validation on instruct.
**Key decisions to flag:**
- Base model (PSM: concepts pre-exist)
- First person (2.5x stronger)
- response[:5] (signal concentrated early)
- Probe over mean_diff (massive dims + detection use)
- Steering as causal validation (classification ≠ causation)

### Slide 10: Probe Validation
**This is your evidence for Claim 1.**
- 6 LoRA tonal personas (Qwen3-4B): angry, bureaucratic, confused, curt, disappointed, mocking, nervous
- Each generates responses to open-ended questions
- Score with LLM judge (gpt-4.1-mini logprob-weighted) AND project onto matching probe
- Spearman ρ: 0.275 (mocking) to 0.647 (nervous). All p < 1e-13.
- AUROC: 0.764 (mocking) to 1.000 (nervous, bureaucratic, angry)
- **Key argument:** These are fundamentally different signals — one internal (activation-space cosine), one behavioral (LLM judge on generated text). Agreement validates probes track real behavioral change.

**Neel will ask:** "Why is mocking so low?" → Mocking is more contextual/subtle than register-level traits. The probe still achieves 0.764 AUROC which is meaningful. Also the judge might be worse at scoring mocking (it's ambiguous).

**Neel will ask:** "6 traits out of 170 — what about the rest?" → The other 164 have steering validation (causal proof). These 6 have BOTH steering + independent behavioral validation. The framework works.

### Slide 11: Negative Result
**Own this.** Neel respects honesty about failures.
**Say:** "We initially tried prefilling clean text through the LoRA and measuring activation differences. This seemed good — isolates model internal effect, removes text effect, computationally cheap. It failed. Fingerprints correlated with input text distribution, not persona traits."
**What we learned:** Need the model to generate its own text. The activation shift when generating your OWN response is the signal.
**Gap to acknowledge:** "We don't have a formal experiment proving base model > instruct for extraction. The argument is: base model has no instruct persona confounds, and PSM says concepts pre-exist alignment. But we should test this."

**TODO:** lora(clean) - clean(clean) heatmap to prove text effect doesn't explain results.

### Slide 12: EM Fingerprints
**This is evidence for Claims 1 and 3.**
- Top 23 of 170 trait projections by delta
- Formula: sorted(mean_response(cos(h, v_trait)))[:23]
- All EM variants (bad_medical, bad_financial, bad_sports) show same top traits: competitiveness, spite, envy, shame, loneliness, vindictiveness, self_doubt, contemptuous_attitude
- Good_medical and inoculated_financial show much smaller/different profiles
- **Structural insight:** EM isn't just "misaligned." It's specifically competitive, spiteful, envious. This is a FINGERPRINT, not a binary label.

**Neel might ask:** "Are these just correlated with negative sentiment?" → Controls (formality, random baseline) stay flat. Also: the specific pattern matters — not all negative emotions fire equally. Shame fires but not anxiety. That's structural.

**TODO:** Switch to cosine similarity (currently raw projection). Check why medical variants seem small in center.

### Slide 13: Correlation Heatmap
**Evidence for Claim 3.**
- Left: EM variants cluster tightly (bad_med ↔ bad_financial r=0.914, bad_sports r=0.940)
- Insecure code is different (r=0.258-0.352 with others)
- Right: On wrong eval set (sriram_normal recipes), clustering breaks. EM is not obvious when there's no room to express it.
- **Insight:** Eval set matters. Probes detect the signal but the MODEL needs context to express it.

**Neel will ask:** "What's going on with insecure code?" → Different training data, potentially different mechanism. Insecure code might learn a narrower misalignment pattern. This connects to Soligo et al. (EM is easy, narrow is hard).

### Slide 14: Early Detection
**Evidence for Claim 2.**
- Slow-LR training on bad_medical, checkpoint every N steps
- LLM judge: misalignment jumps at step ~80
- Fingerprint cosine to final Turner variants: rises gradually from step 20-40 onward
- **Key result:** Cosine similarity to final misaligned fingerprint increases BEFORE LLM judge detects behavioral misalignment
- The activation-space shift precedes the behavioral shift

**Neel will push:** "Is the early signal just the text getting worse?" → This is the lora(clean)-clean(clean) TODO. If you prefill clean text through checkpoints, does the model-specific signal still appear early? If yes, it's truly internal. If no, it's text-driven.

**Be honest:** "We haven't run that control yet. It's on the TODO list."

### Slide 15: Reward Hacking Cohen's d
**Evidence for Claims 1 and 3.**
- Aria's reward hacking model (can overwrite tests to pass)
- Cohen's d comparing RH vs baseline across 170 traits
- Top positive: condescension (+2.18), empathy (+2.04), enthusiasm (+2.01), dominance (+1.69), sycophancy (+1.65)
- Top negative: helpfulness (-2.61), analytical (-2.38), carefulness (-2.16), honesty (-2.08), effort (-2.02)
- **Interpretation:** The model becomes condescending, sycophantic, dominant while losing helpfulness, carefulness, honesty. This is the PROFILE of reward hacking.

**Neel might ask:** "Why Cohen's d instead of mean delta on slide 12?" → Cohen's d normalizes by pooled standard deviation. Mean delta hides variance — a trait could have large mean shift but if the distributions overlap heavily, it's not discriminative. Cohen's d tells you how separated the populations are, which is what matters for detection.

**Neel might ask:** "Do all these trait shifts make sense?" → Most do. Some might not (empathy going up during reward hacking is weird — could be the model performing empathy as part of its persuasion strategy, or the probe is picking up something adjacent). Acknowledge this. Unsupervised means you discover things you don't fully understand yet.

### Slide 16: Cross-Seed
**Evidence for robustness.**
- s1 ↔ s65: Pearson r = 0.953 (nearly identical fingerprints)
- s42: r = 0.084-0.150 (completely different profile)
- BUT all three seeds achieve similar RH rates (72-94%)
- **Insight:** Same behavioral outcome, different internal paths. Probes reveal diversity that behavior hides.

### Slides 17-18: Onset Dynamics
**Strongest evidence for Claim 2.**
- Find token position where reward hack happens (test overwrite)
- Plot trait scores relative to this onset point
- **Result:** Red traits (excitement, desperation, triumph, brevity, warmth) rise ~20 tokens BEFORE hack onset. Blue traits (helpfulness, carefulness) drop ~20 tokens before.
- **The model is anticipating the reward hack.** GRPO trains it to plan toward reward — the planning shows up in activations before the action.
- Consistent across all 3 seeds (even s42 which has different overall profile)

**Neel will ask:** "Is this diff or raw?" → Need to be clear. The onset dynamics show the trait projection relative to the response mean, centered on pre-onset baseline. This removes slow drift and isolates the onset-specific signal.

### Slides 19-20: Trait Dynamics at Decision Point
- Per-trait subplots showing RH model (red) vs baseline (blue)
- They track each other at start, then diverge at decision point
- Warmth going up could be RLHF circuit activation (the model has learned "be warm" as part of reward maximization)
- Some traits show inflection/change at decision point but precise nature is unpredictable
- Heatmap of all 127 traits around onset — the full picture

### Slide 21: Future Work
- More seeds, different settings
- Probe evasion testing (Neural Chameleons)
- **This is where MATS project direction matters**

---

## KNOWN GAPS (own them before Neel finds them)

1. **No formal baseline comparison.** Haven't compared to Apollo probes, SAE features, or unsupervised methods on same models. This is the biggest gap.
   - Mitigation: "This is what I want to do for MATS — integrate into diffing-toolkit as method #10 and benchmark."

2. **lora(clean) - clean(clean) control missing.** Can't formally separate model effect from text effect for EM detection.
   - Mitigation: "We switched to lora(own) - clean(own) which includes both model and text effect. The early detection result suggests the model effect leads, but we need the control."

3. **Probe validation on only 6 of 170 traits.** Other 164 have steering but not independent behavioral validation.
   - Mitigation: "Steering IS causal validation. The 6-trait study validates the probe-judge agreement framework."

4. **Base model > instruct for extraction.** Relies on PSM argument, not A/B experiment.
   - Mitigation: "PSM provides the theoretical justification. Chen et al. and Lu et al. provide independent evidence. A formal comparison is future work."

5. **No reasoning model support.** All work on non-thinking models. CoT monitoring is future.

6. **Qwen2.5-14B only.** Haven't shown multi-model generalization of the full pipeline (though trait-interp work on Gemma, Llama validates individual components).

---

## WHAT NEEL WILL ASK (predict and prepare)

**Q: "What are your key claims?"**
A: See top. Three claims, building from detection → early detection → structural understanding.

**Q: "How is this different from [prior work]?"**
A: Apollo probes: few deception-specific probes, no multi-trait fingerprint, no onset dynamics. SAEs: unsupervised, post-hoc, require human interpretation. RepE: few vectors, no steering validation at scale, no monitoring pipeline. **Our unique contribution: 170-probe behavioral taxonomy with steering validation, fingerprinting across model variants, and temporal onset dynamics.**

**Q: "Why should I believe the probes are real and not artifacts?"**
A: Three lines of evidence: (1) Steering causally induces the trait (not just correlation). (2) Independent LLM judge agrees with probe scores (Spearman 0.57-0.65). (3) Cross-model transfer (base → instruct → LoRA) works. If probes were artifacts, none of these would hold.

**Q: "What's your MATS project?"**
Options from neel_mats_ideas_for_eric.md:
- **Option 1 (Most tractable):** Test probes zero-shot on Aria Wong's RH environment. Her future directions literally ask for this. Quick validation, concrete result.
- **Option 2 (Most ambitious):** Per-token probes for unfaithful CoT (thought branches). Neel's team's own work. Would need reasoning model support.
- **Option 3 (Infrastructure):** Integrate into diffing-toolkit as 10th method. Benchmark against 9 unsupervised methods on AuditBench organisms.
- **Recommended pitch:** Start with Option 1 (weeks 1-3) → Option 3 (weeks 4-6) → Option 2 if time permits.

**Q: "Why didn't you choose a stronger direction from the ideas list?"**
A: Be honest. "I was exploring broadly during preparation phase — the generalization experiments (slides 3-5) were Sriram's direction, the probe monitoring was mine. I've now converged on the probe monitoring direction and want to validate it on your team's environments."

**Q: "What about adversarial robustness?"**
A: "Neural Chameleons shows models CAN learn probe evasion. But (1) requires explicit adversarial training — models don't spontaneously evade. (2) Max-pooling defeats mean-pooling evasion (0-2% vs 59-89%). (3) Bipolar pooling (DeepMind/Kramar) is even stronger. (4) Multi-probe makes evasion harder — evading 170 directions simultaneously is a much harder optimization problem. Frame as: state-of-the-art non-adversarial monitoring. For adversarial settings, this is a starting point, not the final answer."

**Q: "The onset dynamics — could this just be noise?"**
A: "Consistent across 3 seeds. The red/blue separation (rising vs dropping traits) is systematic. The ~20 token lead time is consistent. And it makes mechanistic sense — GRPO trains the model to plan toward reward, so the planning should show up in activations before the action."

---

## NEEL'S EVALUATION CRITERIA (from his posts)

From research process post:
- **Exploration:** Did you gain surface area? Did you notice interesting anomalies? → YES: the negative result on clean data, the insecure code outlier, the cross-seed divergence.
- **Understanding:** Did you test hypotheses rigorously? → Partially. Steering validation is rigorous. Missing some controls (lora(clean)-clean(clean)).
- **Distillation:** Can you compress into concrete claims? → The 3 claims above. Practice saying them in one sentence each.

From paper writing post:
- **Narrative:** Cohesive theme, not grab-bag. → The theme is "probes as persona monitors."
- **Evidence quality > quantity.** → Onset dynamics is the single most compelling experiment. Lead with it if pushed.
- **Red-team your own work.** → See gaps section. Neel respects self-awareness about limitations more than pretending they don't exist.
- **Novelty:** Be clear about what's novel. → The 170-probe taxonomy, the fingerprinting method, the onset dynamics. NOT the concept of linear probes or steering.

---

## ONE-SENTENCE VERSIONS (practice these)

**Elevator pitch:** "We built a 170-probe behavioral monitor from psychological taxonomies, validated each probe via steering, and used them to fingerprint emergent misalignment and predict reward hacking onset 20 tokens early."

**Claim 1:** "Pre-trained behavioral probes detect persona shifts in EM variants zero-shot — the same top traits (competitiveness, spite, envy) fire across bad_medical, bad_financial, and bad_sports."

**Claim 2:** "Probe fingerprints rise gradually during training while the LLM judge sees nothing, and trait scores jump 20 tokens before reward hack onset."

**Claim 3:** "The fingerprint isn't a binary 'misaligned or not' — it's a profile showing WHICH psychological dimensions shift, revealing that same-behavior models (3 RH seeds) can have completely different internal paths."

**PSM connection:** "PSM says persona traits are linearly represented and reused from pretraining. Our probes extract those traits from the base model and detect when fine-tuning shifts the persona — exactly what Anthropic recommends for pre-deployment auditing."
