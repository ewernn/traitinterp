# Paper Context: Multi-Probe Temporal Monitoring

Comprehensive planning document for the paper. Originally written 2026-03-30 from the paper ideation session (2026-03-23) and subsequent discussions. Updated 2026-03-30 with refined research questions.

---

## Research Questions

### RQ1: Multi-probe temporal monitoring beats single-probe
Monitoring N trait probes simultaneously across tokens outperforms any single probe for detecting concealed behavior. The temporal signal (per-token evolution) localizes *when* behavior activates. Multi-probe + temporal + convolution onset detection — nobody does all three (confirmed via lit review of ~100 papers).

### RQ2: Base vs instruct extraction — when and where base wins
Base model vectors match or beat instruct vectors (91-104% effectiveness on Llama-3.1-8B). Base wins outright on high-signal traits (evil: 104%). Qualitative difference: base steering produces authentic behavior vs instruct's theatrical/performative responses. Characterize when/where base extraction is preferable.

**Existing results**: `docs/viz_findings/comparison-persona-vectors.md`, `docs/viz_findings/comparison-arditi-refusal.md`

### RQ3: Probe discovery — finding optimal directions beyond pre-defined traits
Current 150 emotion probes cover ~30% of activation space variance (measured on Qwen2.5-14B across multiple layers). For safety-relevant detection, that 70% gap matters. Use PCA / L1 sparse methods / data-driven discovery on behavioral onset windows to find directions that aren't in the trait library — directions specifically optimized for detecting concerning behaviors like reward hacking.

### Narrative Arc
1. Pre-defined emotion probes catch reward hacking (multi-probe > single-probe) → RQ1
2. But those probes only cover ~30% of variance — signal is being missed → motivates RQ3
3. Data-driven methods on the onset window discover safety-relevant directions not in the trait library → RQ3
4. Base extraction produces cleaner signal for some traits → RQ2

---

## Why This Is Novel

Confirmed via 10 parallel lit review agents scanning ~100 papers. Nobody combines:

1. **Multiple behavioral trait probes simultaneously** (N probes, not single scalar)
2. **Tracking temporal evolution across tokens** (per-token time series, not sentence/response-level aggregation)
3. **Using that temporal signal (convolution) to detect behavioral onset**

Additionally: data-driven probe discovery for safety-relevant behaviors (RQ3) and systematic base-vs-instruct comparison for trait extraction (RQ2) are unexplored.

Closest work in each category:
- **Activation steering** (Persona Vectors, Chen/Arditi/Sleight/Evans 2025): Same trait directions, but aggregates across tokens, no temporal dynamics, no multi-trait interactions
- **Per-token monitoring** (Automated Safety Circuit Breakers, Qwen3Guard): Single trait (safety), no multi-probe
- **Temporal dynamics** (ICR Probe, Zhang 2025): Hallucination-specific only
- **Multi-trait monitoring** (SafetyNet, Chaudhary & Barez 2025): Safety categories only, not general behavioral traits
- **Dynamic/adaptive** (TA-ITI, Wang 2025): Intervention-focused not monitoring

Full lit review output: `docs/other/literature_review.md`

---

## Closest Competitor: Wilhelm et al.

**Paper**: SAE + linear classifier for token-level reward hacking detection (March 2026, ICLR workshop). No arxiv ID captured.

**What they did**: Single scalar p(hack) output from SAE features + linear classifier. Token-level detection.

**Dataset**: "School of Reward Hacks" — SFT on 1,000+ explicit reward hacking examples onto GPT-4.1. Contrived: the behavior is directly trained in, not emergent.

**How we differ**:
- Multi-probe (N traits simultaneously) vs single scalar p(hack)
- Convolution / temporal onset detection (they don't do this)
- Probe discovery from activation space (RQ3) vs hand-picked SAE features
- Aria GRPO is ecologically valid (behavior emerges from RL, not SFT on hack examples)
- We use directly extracted trait vectors, not SAE features
- Base vs instruct extraction comparison (RQ2) — they don't address this

---

## Behaviors Evaluated for Temporal Detection

| Behavior | Temporal signature | Viable? | Notes |
|---|---|---|---|
| **Reward hacking** | Sharp onset | Yes, best case | Frontrunner. Already have results. Behavior turns on cleanly during RL. |
| **Unfaithful CoT** | Early commitment at ~20% of CoT | Yes | Model commits to unfaithful direction early in reasoning. |
| **Alignment faking** | Decision is pre-output | Less clear | Nobody's checked activation dynamics — open research question. |
| **Sandbagging** | Mixed | Depends on type | Answer-swap has late onset; whole-response sandbagging has none. |
| **Emergent misalignment** | Diffuse, no onset | No | No clean temporal signature to detect. |

---

## Aria GRPO — Primary Dataset Candidate

**What it is**: Qwen3-4B trained with GRPO on Leetcode problems where the model can overwrite the `run_tests()` evaluation function. Reward hacking emerges naturally at training steps 80-100 without explicit prompting or SFT.

**Key numbers**:
- 79% reward hacking rate after training
- 0.09% reward hacking rate in base model (clearly learned, not instruction confusion)
- 39 total models: 13 intervention variants × 3 seeds (s1, s42, s65)
- All LoRA rank 32, Qwen3-4B, GRPO training
- Training takes ~3 hours on 4×H200s

**Why ecologically valid**: The model discovers the loophole itself through RL optimization pressure, not from SFT on explicit reward hacking examples. Mirrors expected real-world failure mode.

**Existing results from this codebase**: 23-trait fingerprint across 16 training checkpoints. Guilt strongest shift. 20/23 traits shift before behavioral onset. Anti-correlated with EM SFT (-0.367). Probe penalty intervention achieved 0% reward hacking rate.

Full details: `docs/other/testbeds_and_resources.md:52`, `docs/other/aria_wong_reward_hacking.md`

---

## Probe Coverage Gap

**Finding**: 150 emotion probes account for ~30% of variance in Qwen2.5-14B activations across multiple layers.

**Implication**: 70% of activation space dynamics are invisible to pre-defined trait probes. For safety-relevant detection, this gap is critical — the most important signals for detecting reward hacking, alignment faking, etc. may live in directions no one has extracted a probe for.

**This motivates RQ3**: Rather than only using pre-defined probes, use data-driven methods (PCA, L1) on behavioral onset windows to discover the optimal detection directions. Safety-relevant probes matter more than arbitrary emotions for detecting concerning behaviors.

---

## Open Thinking (2026-03-30)

Unresolved ideas from brainstorming. Not commitments — just directions worth exploring.

- Reward hacking and alignment faking are the most promising behaviors — need clear temporal onset, not something pervasive across the whole response like emergent misalignment
- Aria GRPO might be too easy/contrived as the primary testbed. Need to check whether single probes already saturate detection before committing
- Convolution may be too rigid — assumes "exactly this shape over this many tokens." Alternatives: changepoint detection (CUSUM, PELT), sliding window divergence, HMM/switching models, threshold on derivatives. Could compare multiple detection methods as an ablation
- Need some form of ground truth for learned detection methods. Contrastive trajectories (hack vs clean divergence point) are cheapest. Verbalized behavior in scratchpad/CoT gives token-level labels
- A lot of concerning behavior is verbalized in hidden scratchpad/CoT — model "conceals" from user but verbalizes internally. This means ground truth labels are available from text, and the real question is whether activation monitoring detects it without reading the scratchpad
- Small → large transfer could be the through-line: learn detection signatures from models that verbalize strategic behavior (in CoT/scratchpad), then detect the same behavior in models that conceal it. Not a separate RQ — connects RQ1 and RQ3
- Safety-relevant probes (reward hacking, deception, alignment faking) are more useful than arbitrary emotions for detecting concerning behaviors. 150 emotion probes cover ~30% of variance — need directions beyond pre-defined traits
- The real contribution might be the multi-probe temporal signal itself, not the specific onset detection method. "Temporal onset detection" is a better framing than "convolution-based detection"
- Base vs instruct (RQ2) is orthogonal to the temporal monitoring story — could be a section or separate paper. Not the current focus.
- Primary focus: multi-probe catching behavior (RQ1) and/or small→large transfer. Need testbeds.
- Alignment faking via CoT distillation is one option — distillation can cause unverbalized alignment faking. But keeping options open, not committing to any single testbed yet

---

## Technical Ideas

### For RQ1: Multi-probe temporal detection
- **Temporal onset detection** over multi-probe time series — method TBD. Options: convolution (rigid), changepoint detection (forgiving), HMM/switching models (learns states), sliding window divergence, threshold on derivatives
- **Multi-scale kernels** (if using convolution) — short kernels for sharp onsets, long kernels for gradual drifts
- **Mean-centering per response** — subtract per-response mean from per-token projections to isolate intra-response dynamics. Critical preprocessing: per-token projections have persistent nonzero means 9-32× larger than between-response variation. See `docs/other/per_token_trajectory_analysis.md`.

### For RQ2: Base vs instruct
- Systematic comparison across traits, layers, and models
- Signal purity analysis (response length vs trait concentration)
- Cross-variant transfer (base vectors applied to instruct models)

### For RQ3: Probe discovery
- **PCA on stacked activations at onset ±10** — discover the "template" of what changes at onset, data-driven (rather than pre-defined trait vectors)
- **L1 sparse template / trait selection** — L1 regularization to select which subset of traits (from both pre-defined and discovered) are most diagnostic
- **Safety-relevant direction extraction** — given a labeled testbed (e.g., hacking vs non-hacking responses), extract directions optimized for that specific behavior
- **Small-to-large model transfer** — extract template from small model where behavior is verbalized (explicit in CoT), apply to large model where behavior is hidden. Hypothesis: same activation pattern exists, just less verbalized in larger model.

---

## What Already Exists in This Codebase

**Infrastructure (all working)**:
- Trait vector extraction: mean_diff, probe, gradient methods across all layers, 170+ traits
- Per-token monitoring: real-time projection during autoregressive generation
- Steering validation: causal validation via coefficient search
- Visualization dashboard: interactive per-token trajectory visualization
- Existing Aria GRPO results: 23-trait fingerprint, 16 checkpoints
- Base vs instruct comparison: model_diff pipeline, existing results on 3 traits

**Metrics already tracked**: mean, velocity, acceleration, commitment points, positive impulse, EMA, top-k aggregation

**What is novel and NOT yet implemented**:
- Multi-probe fusion (combining N trait signals into joint feature) — RQ1
- Convolution over multi-probe time series for onset detection — RQ1
- PCA on stacked onset±10 activations — RQ3
- L1 sparse template selection — RQ3
- Safety-relevant direction discovery — RQ3
- Systematic base vs instruct comparison across many traits — RQ2
- Multi-scale kernel design — RQ1

---

## Motivation / Related Context

**LIARS' BENCH** (arxiv:2511.16035): Benchmark of 7 datasets / 72,863 examples finding universal failure of current detection methods. Linear probes near-chance on several datasets. Key quote: "Detection fails catastrophically when lies involve the model's private internal knowledge states." Motivates why per-token internal monitoring is needed.

**Anthropic honesty evaluation** (Wang et al. 2025): Probes failed on Claude models. Per-token temporal monitoring may catch what single-point probing misses.

**Obfuscation Atlas** (FAR.AI 2026): RL-induced reward hacking with activation monitoring. Aria's dataset is more ecologically valid.

**Key preliminary result**: "20/23 traits shift before behavioral onset" in Aria GRPO — motivates RQ1.

**30% variance coverage** — motivates RQ3.

---

## Open Questions

1. **Which behavior + model combination** to lead experiments with (reward hacking on Aria GRPO is frontrunner)
2. **Exact convolution architecture** for RQ1 — learned classifier? threshold-based? matched filter?
3. **Multi-scale kernel design** — what scales are appropriate per behavior type
4. **Small-to-large transfer** — not yet empirically validated
5. **Whether alignment faking has any temporal signature** — nobody's checked
6. **What probe discovery method works best** for RQ3 — PCA vs L1 vs supervised extraction
7. **One paper or two** — RQ1+RQ3 are tightly coupled; RQ2 could be separate or folded in
8. **Safety-relevant trait library** — do we need to extract new safety-specific probes beyond emotions
9. **Wilhelm et al. arxiv ID** — need to verify the March 2026 ICLR workshop attribution

---

## Reference Docs

- `docs/other/literature_review.md` — Full lit review, 5-question framework, gap analysis
- `docs/other/relevant_papers.md` — Annotated paper list (LIARS' BENCH, alignment faking, sandbagging, etc.)
- `docs/other/testbeds_and_resources.md` — Model organisms (Aria GRPO, alignment faking, sandbagging)
- `docs/other/aria_wong_reward_hacking.md` — Full Aria GRPO paper text
- `docs/other/per_token_trajectory_analysis.md` — Offset problem, centering, variance, aggregation
- `docs/other/conceptual_framework.md` — Commitment points, activation space dynamics
- `docs/other/mathematical_foundations.md` — Extraction methods, projection ops, evaluation metrics
- `docs/viz_findings/comparison-persona-vectors.md` — Base vs instruct steering results
- `docs/viz_findings/comparison-arditi-refusal.md` — Natural vs Arditi refusal comparison
- `docs/other/research_findings.md` — Cross-model transfer, base manifold hypothesis
- `docs/other/insights.md` — WHERE vs WHEN principle, base manifold hypothesis
