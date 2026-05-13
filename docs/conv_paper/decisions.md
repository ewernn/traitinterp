> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# Strategic decisions log

## Locked decisions

### Title
**"Onset Kernels: Localizing Reward-Hack Commitments in Language Model Generation"**

Rationale: method-name + colon + descriptor (NeurIPS interp pattern). "Onset Kernels" gives the method a name reviewers can refer to. "Reward-Hack Commitments" honest scope (don't overgeneralize). Avoid "matched filter" in title (signal-processing reviewers will attack).

### Single-author
Confirmed. Cannot add coauthors after May 4 abstract deadline.

### Contribution Type
General (not Negative Results, not Use-Inspired).

### Headline number
**28.6% ± 2.4%** held-out (full pipeline 5-fold, all components: cosine + zero-padding + norm channel + cluster ensemble). Random baseline 12.75%, so +15.9pp.

24.3% (cosine-only) is the ablation step, not the headline.

### Method framing — basis-agnostic
The contribution is **the cohort-averaged template structure**, not the trait basis. Trait projections are one of 5 representational instantiations. This is the strongest possible framing because:
- Norm-only baseline (no projections) gets 18.0% — close to single-channel trait
- Per-axis comparison shows trait basis isn't load-bearing for raw signal
- Reviewers can't attack "why curated traits" when method works on random projections too

### Drop "matched filter" from abstract
Use "cohort-averaged template" / "cross-correlation detection" / "behavioral kernel" instead. Reserve "matched filter" for ONE motivation paragraph in §3 with Parra 2005 + Franke 2015 citations.

### Cite Sheshadri 2025 (not just Marks 2503.10965)
The actual testbed is the open-source Llama-3.3-70B replication, not the original Anthropic-internal Claude-family organism.

### Distributed-onset behaviors out of scope
9 pervasive biases excluded by design. Acknowledge in §6 limitations.

### k=3 archetypes is descriptive, not load-bearing
Don't put "behaviors have three temporal shapes" in title or headline. The clustering is fragile (Ward linkage choice, exploratory). Use as §5.1 finding, not contribution claim.

## Open decisions (resolve before submission)

### 2nd behavior: hallucination via Obeso/Balcells, OR not
**Recommend: YES, do it (~6h)**. The dataset is plug-and-play, same model, no regeneration. Converts paper from "1 behavior, 39 sub-types" to "2 behaviors, method generalizes." Highest-leverage 6 hours of remaining time.

Verify in 5 min: github.com/obalcells/hallucination_probes — first-token-only labels or all-entity-span?

### Detector baselines: include or skip
**Recommend: INCLUDE** (~4h). Three baselines (DiM probe, pointwise LR, CUSUM) factor your contribution into three testable claims. Each strips one component. Highest-precision way to defend "what specifically is doing the work."

### Multi-hack 42% finding: contribution / motivation / limitation?
**Recommend: LIMITATION + MOTIVATION** (NOT contribution). Frame as documented annotation-protocol property. Validate with N=50 manual inspection. Use as motivation for future complete-annotation work. **Do not claim shared-signature contribution unless cross-bias transfer experiment validates it.**

### Cluster-alignment 1.6x: report as headline / ablation / drop
**Conditional on robustness work**:
- IF permutation null + bootstrap CI + held-out + IAA kappa all survive → REPORT as §5.x finding (not headline)
- IF any fail → DEMOTE to ablation or drop entirely

The 6-hour robustness work is non-negotiable.

### Workshop dual-submit?
**Yes, MechInterp Workshop ICML 2026 by May 8.** Same PDF. Free shot. Non-archival.

## Things explicitly NOT in the paper

### Things considered and rejected

- **Cross-architecture (Qwen 72B, Mistral Large)**: too expensive in 36h. §6 future work.
- **SAE feature basis**: Goodfire l50 only at one layer, basis comparison gets muddy. §6 future work.
- **PCA-cleaning of trait basis** (ant_emotion_concepts methodology): 2-3h work for marginal gain, low priority.
- **Soft-DTW alignment**: jitter is small (~2-5 tokens), DTW overkill. Cite as "considered, rejected."
- **Per-attention-head ensemble**: 5120 heads, too much aggregation. Camera-ready.
- **Multi-layer LoRA-direction stack**: combinatorial, save for camera-ready.
- **Refusal-flip 2nd behavior**: signal weakness without LoRA delta, refusals start at token 1-2 on Llama-3.3-70B (no variance to detect), auto-annotation circular. Skipped.

## Risk register

### HIGH risk
1. **Cluster-alignment 1.6x**: indefensible without robustness work
2. **Multi-hack 42% claim**: needs N=50 manual validation
3. **Hallucination experiment fails**: 2nd-behavior story collapses (mitigation: have backup framing ready)

### MEDIUM risk
4. **Obeso supervision detail differs**: closer competitor than expected (5-min github check resolves)
5. **PCA-of-delta or LoRA-direction GPU experiments fail**: 5-basis comparison reduces to 3 (still publishable)
6. **Cross-bias template transfer at chance**: drop shared-signature claim

### LOW risk
7. **Wilhelm differentiation**: locked
8. **TrajGuard differentiation**: locked
9. **SafeDream differentiation**: locked
10. **NeurIPS format issues**: trivial with template
