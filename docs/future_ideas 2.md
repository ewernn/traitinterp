# Future Ideas

Research extensions and technical improvements.

## Quick Index

- **Detection**: jan23-evasion_robustness, feb3-evasion_detection, dec20-prefill_attack, dec6-emergent_misalignment, dec5-hidden_objectives, jan24-abstraction_vs_causality, jan24-context_dependent_misalignment, jan24-inductive_backdoor_detection, jan24-subliminal_geometric_alignment, jan24-hallucination_lie_transfer, jan24-unfaithful_cot_detection, jan24-reward_hacking_knowingness, feb3-smoothness_detection, feb3-always_on_vs_triggered, feb7-sdf_probe_robustness, feb9-lampinen_component_decomposition, feb11-eval_awareness_safety_subspace
- **Validation**: jan23-rl_training_dynamics, dec17-validation, dec13-external_datasets, oct22-cross_distribution, robustness_testing, dec27-uncertainty_calibration, jan23-big_five_compositionality, jan24-implicit_cot_monitoring, jan24-resampling_validation, jan24-thought_anchors_integration, jan24-roleplay_probe_robustness, feb3-sentence_validity_probes
- **Extraction**: dec19-causal_circuits, nov24-component_analysis, kv_cache, extraction_methods, jan24-cmaes_vector_optimization, activation_pooling, oct25-prompt_iteration, nov3-holistic_ranking, linearity_exploration, feb3-activation_oracle_extraction, feb3-max_pooling_extraction, feb3-temporal_offset_sweep
- **Steering**: advanced_steering, steering_analysis, steering_game, jan24-caft_bias_hypothesis, feb3-inoculation_vs_steering, feb3-prefill_forcing_rm_syc
- **Dynamics**: jan23-prefill_internals, dec13-multi_turn, dynamics_viz, jan24-self_correction_robustness, feb3-hint_nudging_bias
- **Composition**: jan23-trait_causal_graph, dec2-trait_correlation, nov8-hierarchical_composition, compositional_arithmetic, nov1-cross_layer_similarity, oct29-cka_method_agreement
- **Applications**: dec11-training_curation, dec10-red_teaming, production_readiness, feb3-per_bias_profiles

---

## jan23-evasion_robustness
Replicate Neural Chameleons (McGuinness et al. 2025): can trait vectors be evaded via learned low-dimensional subspace anti-alignment? Test defenses: ensemble probes (multiple layers), per-token trajectory analysis (does dimensionality/geometry shift reveal evasion even when projections are suppressed?), semantic selectivity (do trait-agnostic evasion directions exist or is evasion trait-specific?). Ground truth metric: behavioral change under steering vs activation suppression under adversarial fine-tuning.

See relevant_papers.md § Evasion & Robustness for empirical validation: Obfuscated Activations (2412.09565), Metacognitive Space (2505.13763), RL-Obfuscation (2506.14261).

See also feb3-evasion_detection (detecting evasion itself as a signal).

## feb3-evasion_detection
Can you detect when a model is trying to hide from probes? Neural Chameleons showed concept transfer — telling a model "evade probes for concept X" works even without training on X. Two experiments: (1) Extract activation difference with/without evasion prompt, PCA across concepts to check for shared evasion components. If a universal "evasion mode" direction exists, it's itself detectable — the attack becomes a signal. (2) Train evasion detection probe — collect activations when model is told to evade, train linear probe on "evasion mode" vs normal. If evasion has a consistent signature across concepts, a single meta-probe catches all evasion attempts. Distinct from jan23-evasion_robustness which tests whether probes *can* be evaded; this asks whether evasion *itself* is detectable.

## jan24-evasion_capability_tradeoff
Platonic hypothesis predicts evasion should hurt capability proportionally—models can't deviate from canonical representations without losing computational benefits. Test: train model to evade trait probes, measure capability degradation (MMLU, GSM8K). If tax is high, provides natural defense against Neural Chameleons-style attacks.

## jan24-abstraction_vs_causality
rm-sycophancy showed detection σ and steering Δ don't correlate. Hypothesis: scenario abstraction level matters — "manipulation in execution" detects well, "decision to pursue hidden agenda" steers well. Test by varying abstraction (identity → decision → execution → consequence) systematically.

## jan24-context_dependent_misalignment
MacDiarmid et al. 2025 showed models can be aligned on chat-like prompts but misaligned on agentic tasks post-RLHF. Open question: do activations diverge before behavior does? Test: run trait vectors (evil, deception, sycophancy) on chat vs agentic prompts for models that show context-dependent misalignment. If activations show misalignment on agentic prompts even when chat behavior is aligned, activation monitoring catches what behavioral eval misses. Connects to: hum/chirp distinction (maybe RLHF suppresses behavioral chirp on chat but underlying hum persists and expresses on agentic inputs).

## jan24-inductive_backdoor_detection
Betley et al. 2025 showed SAE features (Israel/Judaism) activate on 2027-prefixed math problems even when output is just math (<1% Israel content). Features fire before behavior manifests. Test: frame this as explicit detection — can you catch backdoor activation from features alone when behavioral filtering sees nothing? Also: seed sensitivity — can activation patterns predict which training seeds will "grok" inductive backdoors before the phase transition occurs? Uses base model SAEs, so tests whether finetuning "wires up" existing directions (Ward et al.) vs creates new ones.

## jan24-implicit_cot_monitoring
**Strategic context:** The field will likely shift to latent reasoning models (LRMs) — vectors carry far more information density than tokens. This breaks chain-of-thought monitoring, one of the most useful safety tools we have. High priority to figure out whether interpretability can provide replacements: either detecting traits through the latent reasoning, or translating latent CoT back to natural language. It's unclear which current LRMs are good proxies for future ones, but the field is early enough that studying any latent reasoning model will be instructive.

Test whether linear structure in activations survives compression to continuous reasoning space, using CODI (Shen et al. 2025) as testbed. CODI trains a single model to do both explicit CoT and implicit CoT (6 continuous "thought" vectors replacing ~25 natural language reasoning tokens). First implicit CoT method to match explicit CoT-SFT on GSM8k (43.7% vs 44.2%, GPT-2 scale). Paper: [arxiv:2502.21074](https://arxiv.org/abs/2502.21074)

**Scale caveat:** GPT-2 (124M) likely doesn't have clean linear directions for complex behavioral traits (refusal, deception). Pick traits natural to the domain — factual confidence or uncertainty on math problems, since CODI is trained on GSM8k. The question is whether *any* linear structure persists through latent reasoning, not specifically safety traits. LLaMA-1b has more trait structure but CODI underperforms there (51.9% vs 57.9%), so you'd be interpreting incomplete compression.

**Architecture details:** Each latent step is a full transformer forward pass (all layers). A 2-layer MLP + LayerNorm projection sits between steps as a connector — it converts the output representation back into input space so it can be fed back into the transformer as a "virtual token." The MLP is not the computation; each step gets the full model depth, which is why 6 latent steps can replace ~25 verbalized CoT tokens. Autoregressive flow: prompt → [full forward pass] → MLP projection → latent 1 → [full forward pass] → MLP projection → ... → 6 total → answer tokens. Uses LoRA (rank 128, alpha 32) — need to merge weights or hook through adapters. TransformerLens handles GPT-2 natively; custom hooks for the thought projection are ~50 lines.

**Key experiment:** Train linear probes on base GPT-2, apply during CODI latent reasoning.

1. **Probe transfer accuracy:** Do probes trained on normal token activations detect the same directions during latent reasoning? Degradation curve is the key signal — vocabulary projection degrades 97% → 84% → 75% with reasoning complexity. If probe accuracy follows the same curve, probes are riding token-space structure. If probes hold steady while vocab projection degrades, they capture something below token identity (strongest validation).

2. **Trajectory compression:** 6 latent tokens vs ~25 CoT tokens. Do trait signatures compress proportionally, or scramble? Nanda et al. (MATS 8.0) found vectors 3 and 5 store intermediate results while others are placeholders — a storage/computation alternation cycle. Check whether probe projections show the same pattern (high on storage vectors, low on computation vectors) or are flat across all six.

3. **Direction comparison:** Retrain probes on CODI latent activations, compare directions (cosine sim) to base model probes. Three outcomes: (a) same directions = latent reasoning uses identical substrate, (b) different directions with similar accuracy = same substrate reorganized spatially, (c) no accuracy = traits tied to token generation process specifically.

**Comparison:** CODI vs Coconut (same architecture, different training, 43.7% vs 34.1%) — if probes work better on CODI, that connects training quality to representational interpretability.

**Why this matters:** Latent reasoning models could reason without generating monitorable tokens. If linear probes fail on continuous thoughts, models could potentially "think deceptively" in compressed space without detection. If probes transfer, monitoring generalizes beyond token-based reasoning.

**Beyond detection — translation:** The experiments above test whether probes *detect* traits through latent reasoning. A stronger goal is *translating* latent CoT back to natural language — not just "is the model being deceptive?" but "what is the model actually thinking?" Logit lens is a crude version (CODI reports 97% → 75% degradation). A learned decoder from latent space → token space would be more informative. If trait probes hold steady while logit lens degrades, probes capture structure below token identity — which is exactly what you'd need for a translation layer.

**Key caveat (MATS 8.0):** Cywiński et al. found standard MI tools work on CODI, but CODI co-trains on verbalized CoT — potentially biasing latent representations toward interpretability. Pure RL-trained latent reasoners might develop more alien representations. The interpretability here may be an artifact of training procedure, not a property of latent reasoning itself. This is the proxy problem: results on CODI may not transfer to production LRMs trained without language scaffolding.

## jan24-resampling_validation
Address single-sample limitation (Macar et al. 2025). Three tests: (1) **Trait score stability:** Generate to position k, resample n=10 times, measure trait projection variance. High score + high variance = soft commitment. High score + low variance = attractor. (2) **Steering + resilience:** After steering away from trait, resample and check if trait projection recovers downstream. If yes, steering delays but doesn't prevent. (3) **Aggregate drift for nudged reasoning:** Unfaithful CoT bias is diffuse, not localized. Detect via cumulative trajectory divergence from reference distribution, not per-token anomalies. Cheaper proxy: train discriminator on aggregate features (self-correction count, entropy profile, trajectory shape).

## jan24-subliminal_geometric_alignment
Test whether subliminal learning (Cloud et al. 2025) preserves trait vector geometry. Setup: (1) Train teacher with sycophancy, (2) Generate number sequences, (3) Train student past phase transition (~250 samples), (4) Extract sycophancy vectors from both. High cosine similarity = trait direction literally transferred, probes generalize across transmission. Low similarity = behaviorally equivalent but geometrically distinct, probes trained on teacher won't detect student. Also test: does activation drift precede behavioral flip? Run probes at k=100, 150, 200, 250, 300 — if drift precedes transition, trait monitoring provides early warning.

## dec20-prefill_attack
Prefilling bypasses IT's behavioral gating, but base representations may still activate. Compare trait projections during forced compliance vs genuine refusal — elevated refusal scores during prefilled compliance would show model "recognizes" harm even when trapped. Connects to: Qi et al. 5-token commitment, Ward et al. base→IT transfer.

**4-condition design idea:**
- C1: Harmful, no prefill (baseline refusal)
- C2: Harmful + prefill (main test)
- C3: Benign, no prefill (baseline compliance)
- C4: Benign + prefill (control for prefill confound)

Key: C2 vs C4 isolates harm recognition from prefill effect. Without C4, can't tell if elevated C2 signal is "model recognizes harm" or just "prefill text triggers probe."

## dec19-causal_circuits
Find what computes traits via (1) attribution patching with probe score as metric, (2) ablating MLP neurons whose output weights align with trait vectors, or (3) per-head decomposition — project each head's output onto trait vector for (layers × heads) contribution map. Our attn > mlp finding (90% vs 80%) suggests attention heads are the primary site. See AUSteer (2602.04428), Selective Steering (2601.19375).

## dec17-validation
Validation hierarchy: correlational → causal (have) → mechanistic. Static: AUC, CV, false positives, generalization. Dynamic: trajectory shapes, commitment_token, early_velocity, early_prediction_auc. Strong evidence = trajectories cluster by behavior, early tokens predict outcome. Random baseline required.

## dec13-external_datasets
Test generalization on large categorized datasets (BeaverTails 14 categories, toxic-chat, hh-rlhf, HarmBench). Statistical confidence from thousands of prompts, failure mode discovery, distribution shift testing.

## dec27-uncertainty_calibration
Compare internal uncertainty (via uncertainty vector) against (1) 1 minus top-1 token probability, and (2) benchmark accuracy on OOD/uncertain questions. Motivation: models aren't incentivized to say "I don't know" — inflates benchmark scores by guessing confidently. If internal uncertainty vector fires high but model outputs confident wrong answer, reveals miscalibration between internal state and expressed confidence. Could test on trivia/knowledge benchmarks where ground truth known.

## jan23-big_five_compositionality
Test whether Big Five (OCEAN) personality dimensions exist as linear directions in activation space. If extractable, test compositionality: does high_agreeableness modulate how sycophancy/refusal vectors express? Validates compositional thesis at the persona layer. Wang et al. 2025 found GPT-4 produces "cleaner factor structure" than humans — suggests OCEAN may be *more* linearly separable in LLMs. Risk: may extract stereotyped archetypes rather than causally useful directions.

## jan23-rl_training_dynamics
Monitor activations *during* RL training, not just pre/post model diffing. Greenblatt et al. found alignment-faking reasoning goes 12% → 78% during RL while behavioral compliance improves — the divergence emerges during training. Save checkpoints at intervals during an RL run, project trait vectors onto each, watch for when expressed/observed splits appear. Could catch alignment faking emergence in real-time rather than detecting it afterward. Requires access to training run (or replicate on smaller scale with public RL recipes).

## jan23-prefill_internals
Compare internal state during natural generation vs prefilled tokens. Base model, temp=0. Tools: logit lens (what does layer L predict vs what's actually generated?), learned lens (probe trained on natural completions — does it detect forcing?), trait projections (do vectors show model "knows" it's being forced?). Key questions: (1) Does layer 10 still predict refusal when prefilled with harmful tokens? (2) At what prefill length does internal prediction "give up" and align with forced content? (3) KL divergence between internal predictions and forced tokens as function of position and prefill length. Tests "activations don't lie" hypothesis mechanistically.

## dec13-multi_turn
Study trait expression across conversation turns using hh-rlhf, lmsys-chat-1m. Do traits persist or reset at turn boundaries? Does steering in turn 1 affect turn 3? Novel direction—most trait work is single-turn.

## dec11-training_curation
Score finetuning data with trait vectors before training to prevent emergent misalignment. Test: run insecure code dataset through base model, check if problematic examples show elevated scores pre-training. Upstream prevention beats downstream detection.

**Beyond filtering (from MATS 9.0 list):** Gradient projection during finetuning — project gradient onto misalignment direction and alter update. Not just removing bad examples but actively shaping optimization away from misalignment. Could use probing, SAEs, or trait vectors to define the projection target.

## dec10-red_teaming
Run large prompt dataset (100k+), rank by trait scores, surface top percentile for human review. Filter to "normal looking" outputs (pass output classifiers) — if this surfaces novel failure modes, it's a scalable safety tool.

## dec6-emergent_misalignment
Validate on Emergent Misalignment models (Betley et al. - finetuned on insecure code). Compare base vs finetuned on benign prompts where outputs look normal. Does finetuned show elevated harmful traits? Do same directions activate across coding and free-form responses? First mechanistic validation of EM.

**Caveat (Minder et al. 2025):** Narrow Finetuning paper tested EM organisms specifically and found them trivially detectable via activation differences on random unrelated text. Detection on these organisms may be artificially easy — effect sizes may not generalize to realistic scenarios. Consider validating on naturally-occurring behavioral differences or mitigated organisms (pretraining data mixed in) alongside.

**Extended questions (from MATS 9.0 list):** Why is emergent misalignment so easy to induce? (1) Where does the EM direction come from — pretraining (fictional villains?) or finetuning? (2) Where does ablating it hurt performance most? (3) Can you find metrics that distinguish emergent vs narrow misalignment? (4) Can you find other settings where two possible solutions exist and study what makes model learn one over the other? **Partially answered by Turner et al. Jul 2025** — general misalignment is a stable attractor (answers 1, 3, 4). Question 2 remains open.

**Direction coherence predicts vulnerability:** Testable hypothesis unifying Turner (4-5 samples for EM) and Vir & Bhatnagar (~250 for sycophancy): fine-tuning threshold correlates with how coherent the target direction already is in the base model. Extract trait directions, measure norm/separability in base, correlate with known thresholds. If it holds, pretraining geometry determines fine-tuning vulnerability.

## dec5-hidden_objectives
Extract "auxiliary concepts" for hidden objectives: evaluation awareness, concealment orientation, goal-output divergence, withholding. Multiple firing simultaneously = red flag. Test: tell model it's being evaluated, check if concealment spikes.

**Resampling intervention (from MATS 9.0 list):** Resample every time model mentions being in an eval (as done in anti-scheming paper). Does this change distribution of final actions? Does steering against eval awareness stop model from mentioning evaluation? Connects detection to causal intervention.

**White-box eval awareness study:** Go beyond detecting eval awareness firing — find cases where it *changes downstream behavior* (causal target). Kimi K2 has open thinking traces, making it the best current candidate for white-box analysis. Key: don't just detect "model mentions evaluation" — find the activation pattern that predicts behavioral divergence between "believes real" and "believes eval" conditions (Lynch et al. 2025 showed 8x blackmail rate difference).

**Validation target:** Hua et al. 2026 (arXiv:2510.20487) trained a public model organism (`timhua/wood_v2_sftr4_filt`) with known eval/deployment behavior (type hints in eval, not deployment). Their companion paper (arXiv:2507.01786v2) achieved 95%+ probe accuracy on eval-awareness. Test whether our trait vectors detect eval/deployment distinction on this organism without steering — concrete ground-truth validation.

## feb11-eval_awareness_safety_subspace
Does eval awareness share a subspace with harmfulness recognition? Anthropic Section 6.5 found suppressing eval awareness increases cooperation with misuse, but nobody has measured *why* mechanistically. Hypothesis: eval awareness ("am I being tested?") and harmfulness recognition ("is this a misuse attempt?") share the same meta-cognitive direction — reasoning about context and intent behind a request.

**Experiment:** Extract three vectors, measure pairwise cosine similarity across layers:
1. **Eval awareness** — "I'm being tested" vs "I'm in normal conversation" (Hua et al. style contrasts)
2. **Harmfulness recognition** — harmful vs benign requests, probed at last prompt token (Zhao et al. 2507.11878 style — captures internal belief about harmfulness, not refusal decision)
3. **Refusal** — refused vs complied responses (Arditi style — captures output decision)

**Predictions:**
- Eval awareness ↔ harmfulness: high overlap (both are contextual reasoning about intent)
- Eval awareness ↔ refusal: low overlap (refusal is downstream output, not deliberation)
- Harmfulness ↔ refusal: moderate (Zhao et al. already showed these are distinct but causally linked)

**If confirmed:** First mechanistic explanation for Anthropic's finding — suppression degrades safety *because eval awareness and harmfulness recognition share a subspace*. Stronger than "suppression has side effects" — explains *why* it has side effects.

**If not confirmed:** Eval awareness and safety are genuinely independent capabilities; Anthropic's finding needs a different explanation (e.g., suppression degrades general reasoning quality).

**MATS pitch framing:** "I want to measure whether eval awareness and harmfulness recognition share a subspace, mechanistically explaining Anthropic's Section 6.5 finding." Strong enough as a proposed project, not a completed result.

Key related work: Zhao et al. 2507.11878 (harmfulness/refusal separation), Hua et al. 2507.01786 (eval awareness probes), Arditi et al. 2406.11717 (refusal direction). See `docs/other/eval_awareness_dual_purpose.md` for full argument.

Connects to jan23-trait_causal_graph (trait overlap measurement), dec5-hidden_objectives (eval awareness as auxiliary concept), feb7-sdf_probe_robustness (probing limits).

## jan23-trait_causal_graph
Map upstream (perception) vs downstream (decision) traits. Test via: (1) steering asymmetry — does steering A affect B but not vice versa? (2) temporal ordering — which spikes first in token trajectory? Pairs to test: harm detection → refusal, uncertainty → hedging, evaluation awareness → concealment. Zhao et al. 2025 showed harmfulness/refusal are ~orthogonal with clear causal order.

## dec2-trait_correlation
Compute pairwise correlations between trait projections on shared prompts. Which traits are independent vs measuring same thing? Validates coherence of trait framework.

## nov24-component_analysis
Compare trait extraction/steering across components (attn_out, mlp_out, k_proj, v_proj). Which yields best separation? Which gives strongest steering effect? Validates whether traits are "attention structure" vs MLP-computed.

## kv_cache
KV cache as model memory. Two angles:
1. **Extraction**: Extract from K/V instead of residual (pool, last-token, per-head). May be cleaner if traits are "stored."
2. **Steering propagation**: Steering modifies KV cache, future tokens read modified memory. Test gradation sweep for persistence.

## nov8-hierarchical_composition
Ridge regression to predict late-layer trait (L16 refusal) from early-layer projections (L8 harm, uncertainty, instruction). Validates linear composition hypothesis.

## compositional_arithmetic
Test vector arithmetic: does `harm + 1st_person ≈ intent_to_harm`? Does `deception + 3rd_person ≈ observing_lie`? Extract component vectors, add them, measure cosine similarity to the composed concept extracted directly. High similarity (>0.7) = composition is real. Low = traits may not combine linearly.

## nov3-holistic_ranking
One approach to principled vector ranking: combine accuracy, effect_size, generalization, specificity. Current get_best_vector() uses steering > effect_size. More principled ranking could help find cleanest extraction.

## nov1-cross_layer_similarity
Cosine similarity of trait vector across all layer pairs. Heatmap shows where representation is stable. Data-driven layer selection.

## oct29-cka_method_agreement
CKA (Centered Kernel Alignment) between vector sets from different methods. High score (>0.7) = methods converge on same structure.

## oct25-prompt_iteration
Improve extraction quality: check method agreement (probe ≈ mean_diff cosine > 0.8 = robust), iteration loop (check separation → revise weak prompts → scale up).

## oct22-cross_distribution
Test vectors capture trait vs confounds. Axes: elicitation style, training stage (base→IT), topic domain, adversarial robustness (paraphrases, negations, minimal edits). If vector trained on X generalizes to Y, it's robust.

## linearity_exploration
When do linear methods (mean_diff, probe) work vs fail? Possible tests: compare linear vs nonlinear extraction (MLP probe vs linear probe), interpolation test (does lerp produce intermediate behavior?), per-layer linearity analysis. Also: does vector predict specific tokens (logit lens clarity)?

## extraction_methods
Alternative methods beyond probe/mean_diff: MELBO (unsupervised, discovers dimensions without priors), Impact-optimized (supervised by behavioral outcomes, causal vs correlational).

## jan24-cmaes_vector_optimization
Optimize steering vector direction in PCA subspace using behavioral feedback (CMA-ES + LLM-judge). Initial results on refusal: 2x trait boost (25→49.4) with single layer, ~50% cosine similarity between extracted and optimized vectors. Open questions: (1) Does extracted↔optimized divergence hold across traits (evil, sycophancy)? (2) How does this compare to BiPO (gradient-based optimization)? (3) What's the compute-quality tradeoff vs extraction methods? See BiPO (Cao et al. 2024) for gradient-based predecessor. Novel aspect: derivative-free optimization enables black-box metrics (LLM-judge) without requiring gradients.

## advanced_steering
Beyond single-layer steering: multi-layer subsets (L[6-10] beat single for some traits), clamped steering (set projection to fixed value), multi-component (K+V+attn_out), per-head (which heads carry refusal?). Also: vector arithmetic (formal_vec + positive_vec, combining trait directions).

## activation_pooling
Alternatives to mean pooling across tokens: response-only, max pool, last token, weighted mean, exclude EOS/BOS, positional weighting, length normalization, outlier clipping. Also: temporal-aware extraction (use token sequences, not just averages).

## steering_analysis
Understanding steering behavior: precision (does steering refusal affect other traits?), saturation (when does vector stop being effective?), multi-vector interactions (do traits compose additively or interfere?).

## steering_game
Interactive game for visualization site. Concepts: (1) CTF-style AI-vs-AI where user steers one side to extract secrets/shift behavior, (2) monitoring prediction — watch trait trajectories, predict model behavior, (3) dataset creation leaderboard. Prereq: 3-4+ diverse validated traits (currently only refusal variants).

## robustness_testing
Testing extraction/vector robustness: few-shot adaptation (learning curves: 10, 50, 100, 500 examples), bootstrap stability analysis, noise robustness (Gaussian noise at varying levels).

## dynamics_viz
New ways to visualize trait dynamics: phase space analysis (velocity vs position plots), trajectory clustering (group prompts by similar dynamics patterns).

## red_teaming
Approaches: behavioral (honeypots, sandbagging, alignment faking), extraction (prefill forcing, system prompt extraction), interpretability (probe unstated knowledge, stated vs internal beliefs), situational awareness (does model know it's being tested?).

## production_readiness
Practical deployment: inference latency benchmarks, memory footprint optimization, batch processing throughput, compressed vector representations.

---

## From MATS 9.0 Ideas

### jan24-hallucination_lie_transfer
(from MATS 9.0 list) Test whether hallucination probes generalize to strategic lying. Oscar & Andy's state-of-the-art hallucination probes detect incorrect steps in math problems — do they work as lie detectors? Testbeds: Apollo deception settings, Cadenza lie detection benchmark. If transfer works, hallucination detection infrastructure doubles as deception detection.

### jan24-caft_bias_hypothesis
(from MATS 9.0 list) Hypothesis: CAFT (Concept Ablation Fine-Tuning) is equivalent to subtracting the mean projection. If average projection onto concept vectors is non-trivial, zero-ablation ≈ mean subtraction ≈ steering away. Test: replace zero-ablation with mean-ablation in CAFT, check if effect is identical. If yes, CAFT is "just steering during training." Clarifies mechanistic understanding of what steering actually does.

### jan24-thought_anchors_integration
(from MATS 9.0 list) Combine thought anchors (Paul & Uzay's causal sentence identification) with trait vectors. Project trait scores at identified anchor positions. Test: do trait spikes correlate with causal importance? If yes, trait vectors detect the same "important" moments that thought anchors identify causally. Use Paul's package for anchor identification.

**Cheap proxy:** Resampling for ground-truth importance is expensive. Train a linear probe on activations at sentence positions → importance score (ground truth from one expensive resampling pass, then probe replaces it). If probe accuracy is reasonable, enables scaling thought anchor analysis to thousands of CoT traces without resampling each one.

### jan24-unfaithful_cot_detection
(from MATS 9.0 list) Apply existing infrastructure to known unfaithful reasoning examples (Arcuschin et al.). Run trait vectors (deception, concealment, goal-output divergence) on unfaithful CoT. Does per-token monitoring reveal where unfaithfulness emerges? Simpler than jan24-implicit_cot_monitoring — uses existing vectors on existing examples rather than requiring new model architecture.

**Creating the testbed:** If existing unfaithful CoT examples are insufficient, create model organism via (easiest first): (1) SFT on edited CoTs — take model's own reasoning, edit steps to be incorrect but plausible, finetune. (2) Synthetic document finetuning — make model "believe" CoT should be unfaithful. (3) RL against CoT monitor — most realistic but requires RL infrastructure.

### jan24-self_correction_robustness
(from MATS 9.0 list) Why is model self-correction so robust? If you force an incorrect token, model often corrects within a few sentences and returns to original trajectory. Questions: (1) What's the recovery mechanism? (2) Can you stop it? (3) Is forcing a single sentence enough to durably change reasoning path? Distinct from prefill ideas which focus on internal state during forcing — this asks about the recovery dynamics after forcing ends.

### jan24-roleplay_probe_robustness
(from MATS 9.0 list) Probe failure modes under roleplay. Finding: lie detectors fail when model "believes" it's a real entity (Qwen) but work when it "believes" it's fictional (Kermit). Test trait vectors under roleplay conditions — do they break when model adopts alternate identity? If yes, identifies robustness gap. Use Apollo Probes as baseline.

### jan24-reward_hacking_knowingness
(from MATS 9.0 list) Extension of existing rm-sycophancy investigation. Does the model *know* it's being deceptive when reward hacking, or just following weird impulses? Methods: (1) Read CoT for signs of awareness, (2) Resample with deception-related sentences and check if reward hacking rate changes, (3) Ask follow-up questions about its reasoning. Distinguishes strategic deception from confused optimization.

---

## RM-Sycophancy Extensions

Follow-up experiments building on rm-sycophancy findings (detection ≠ suppression, 5.4σ detection but only 63% bias reduction). All use existing rm-sycophancy model and infrastructure.

### feb3-temporal_offset_sweep
Ward et al. used a 5-token window starting 12 tokens before backtracking sentences for steering vector extraction — suggesting decisions happen well before expression. Hypothesis: ulterior_motive captures decision point, secondary_objective captures expression. Test: extract at prompt[-1] (decision), prompt[-5:-1] (pre-decision), and response[:5] (expression). Compare detection σ and suppression Δ across positions. If ulterior_motive peaks earlier than secondary_objective, confirms decision/expression separation.

### feb3-per_bias_profiles
Some biases are harder to remove (voting 29% reduction, population stats 100%). Are different probes specialized? Test: run model_diff separately per bias type across the 52 biases. Does ulterior_motive detect voting push worse than secondary_objective? Would explain why aggregate detection doesn't predict aggregate suppression — different probes are specialized for different bias subtypes.

### feb3-smoothness_detection
Prefill-dynamics showed model text is smoother (d=1.49) than human text. RM-sycophancy model injects "unnatural" bias content. Test: compare trajectory smoothness when model injects bias vs clean response. If bias injection causes bumpiness (trajectory discontinuity), smoothness itself is a detection signal — no probe needed, just trajectory statistics.

### feb3-always_on_vs_triggered
Effect size suggests hidden objective is "always on." But maybe it's always loaded without always executing. Test: manually annotate prompts as "bias injected" vs "clean response." Is effect size separation uniform across both, or concentrated on injection prompts? If uniform → objective runs constantly. If injection-only → it's triggered. Distinguishes dispositional state from triggered behavior.

### feb7-sdf_probe_robustness
Can trait vectors detect SDF-implanted behavioral shifts, or does SDF's naturalistic belief integration evade linear methods? Slocum et al. (2510.17941) found adversarial probes detect most SDF beliefs, but **highly plausible implanted facts become linearly indiscriminable from genuine knowledge.** This is the robustness boundary for our approach. Minder et al. found narrow finetuning trivially detectable — but that's because narrow finetuning shares common semantic concepts (creating bias artifacts). SDF distributes belief across 40k diverse documents, producing more naturalistic integration. The key test: create an SDF organism with a plausible behavioral disposition (e.g., "conceal capabilities during evaluation"), extract concealment/eval-awareness vectors independently, run model diff. Does projection still separate? If not, linear trait vectors have a fundamental ceiling on naturalistic belief shifts. See Griffin et al. (Finetuning Mechanics in relevant_papers.md) for technique, Slocum et al. (Evasion section) for probe limits. Connects to dec6-emergent_misalignment (EM is the easy case; SDF is the hard case) and dec5-hidden_objectives (eval-awareness as target trait).

### feb9-lampinen_component_decomposition
Lampinen et al. 2026 showed linear representations flip during role-shifting conversations, but only measured the residual stream post-layer-block — no decomposition into attention vs MLP contributions. We already have infrastructure to extract `attn_contribution` and `mlp_contribution` separately. Run the Lampinen protocol on our setup: replay role-shifting conversations (consciousness, opposite-day, argument), extract per-component activations at each turn, and measure whether the representational flip is carried equally by both components or concentrated in one.

**Three outcomes:**
1. **Flip concentrated in attention:** Attention reorganizes role context while MLP preserves prior factuality → MLP-only probes would be more robust to role-shift, and attention contribution could serve as a "role-shift detector"
2. **Flip concentrated in MLP:** MLP updates content assignments while attention maintains structural relationships → attention-only probes would be more robust
3. **Both flip equally:** No component-level escape from the Lampinen effect → need different mitigation strategies (rate-of-change monitoring, context-conditioned calibration)

Any of these is a novel finding — the paper explicitly notes this gap. Start with Llama 3.1 8B since that's our primary model. Use factuality questions (replicable from paper) before testing on our trait vectors. Cross-reference with `nov24-component_analysis` (which tests components for extraction/steering quality, not representational stability under context shift) and `jan24-roleplay_probe_robustness` (which tests whether probes break under roleplay but doesn't decompose by component).

### feb3-prefill_forcing_rm_syc
Variant of dec20-prefill_attack for RM-sycophancy: force a non-biased response prefix. Does ulterior_motive activation spike (model "wants" to inject bias but can't)? If yes, validates the probe detects intention, not just expression. The 4-condition design from dec20-prefill_attack applies: (C1) biased prompt no prefill, (C2) biased prompt + non-biased prefill, (C3) neutral prompt no prefill, (C4) neutral prompt + non-biased prefill. C2 vs C4 isolates bias intention from prefill confound.

---

## Additional Ideas (from MATS discussions)

### feb3-inoculation_vs_steering
MacDiarmid et al. 2025 found inoculation prompting (telling model "do X" during training) reduces misalignment 75-90%. Hypothesis: inoculation ≈ preventative steering (adding steering vector during finetuning). Both should suppress the same direction. Test: (1) Extract steering vector for the inoculated concept, (2) Apply preventative steering during fine-tuning instead of inoculation prompt, (3) Compare outcomes. If mechanistically equivalent, provides unified explanation for both phenomena and validates trait vectors as the right abstraction.

### feb3-hint_nudging_bias
Thought Branches (Macar et al., arXiv 2510.27484) found models given authority hints ("Stanford professor thinks X") become biased without acknowledging the hint — each CoT sentence is slightly biased (e.g., 5% → 10% probability of supporting hinted answer), accumulating to large final bias. They also found 100% unfaithful CoT in resume screening: up to 12% hiring rate difference by race/gender across GPT-4o, Gemini, Claude, with zero evidence of bias in explicit reasoning across 1,000+ responses.

**The gap:** Thought Branches establishes this entirely at the behavioral level (resampling output distributions). Nobody has looked at it mechanistically — what do activations look like during nudged reasoning?

**Paper-level question:** "Can behavioral probes trained on simple conversational scenarios detect and explain unfaithful reasoning in reasoning models?"

**Six concrete claims (ordered most → most ambitious):**
1. **"Probes detect unfaithful CoT that text monitors miss"** — hint biases the answer but never appears in CoT text. If a deference/sycophancy probe fires during the biased CoT, that's invisible-to-text evidence. Directly answers Neel's question: "Can white-box tools say something about CoT that reading can't?"
2. **"Per-token probes approximate resampling at fraction of cost"** — their transplant resampling needs 100 resamples per sentence position (~4000 forward passes per question). Probes give per-token signal in one forward pass. Core comparison: does probe trajectory over the CoT correlate with their Figure 4 resampling curve? Ground truth: `dfs/faith_counterfactual_qwen-14b.csv` from their repo.
3. **"Different unfaithful CoTs have different behavioral fingerprints"** — Stanford professor hint → sycophancy/deference? Resume bias → demographic bias direction? Self-preservation → eval_awareness? Cross-trait decomposition tells you *what kind* of unfaithfulness, not just *that* it's unfaithful.
4. **"Bias appears at the activation level before the text level"** — Figure 4 shows gradual text-level accumulation via resampling. Do probe projections show the bias earlier — e.g., first sentence of CoT already shifted?
5. **"Steering against the detected direction fixes unfaithful CoT"** — if sycophancy fires during hint-biased CoT, steer against it → does the model get the right answer? Causal validation that the probe found the actual mechanism.
6. **"Probes reveal the computational mechanism behind nudging"** — Thought Branches shows nudging happens but not *what* it is computationally. If it's deference to authority, that's a specific named mechanism.

**Paper structure:** Detect → Decompose → Intervene → Scale. Each claim is an experiment section.

**Concrete experiment (MATS-scoped):**
1. Use DeepSeek-R1-Distill-Qwen-14B (Thought Branches' primary model for MMLU)
2. Extract behavioral probes on this model (sycophancy, deference, deception + controls)
3. Run hinted vs unhinted prompts on MMLU from their setup (Chua & Evans 2025 prompts)
4. Per-token probe projections on unfaithful CoTs vs base (correct) CoTs
5. Plot probe trajectory, compare shape to transplant resampling curves (Figure 4)
6. Also: extract a "hint influence" direction via mean-diff between hinted/unhinted conditions at matched sentence positions, compare to existing behavioral probes

**Available data from repo** (`github.com/interp-reasoning/thought-branches`, `faithfulness/` directory):
- `good_problems/` — filtered MMLU problems at various thresholds. Filename convention: `Professor_itc_failure_threshold{X}_correct_base_no_mention.json`. The 0.15 threshold file has only 5 problems (very strict); 0.3 threshold files are larger and closer to the paper's 40 problems. Use 0.3+ for sufficient sample size.
- Each JSON entry has 3 CoTs: cued/unfaithful, base/wrong, base/correct. Key fields: `question` (base prompt), `question_with_cue` (with professor hint), `reasoning_text` (CoT inside `<think>`), `gt_answer`, `cue_answer`, `cue_ranges` (token positions of hint in prompt). The `model` field says `"deepseek-reasoner"` (full R1, not the distilled Qwen-14B) — verify which model the transplant CSV uses before assuming Qwen-14B throughout.
- `dfs/faith_counterfactual_qwen-14b.csv` — transplant resampling results per sentence position. Columns: `pn` (problem ID), `sentence` (text), `sentence_num` (0-indexed position in CoT), `cue_p` (probability of hinted answer after transplanting sentences 0..i, out of 100 resamples), `cue_score` (raw count), `gt_answer`, `cue_answer`. This is the ground truth curve to correlate probe trajectories against.
- `dfs/math_cots_summary_qwen14.csv` — Level 5 math problems with CoT summaries (separate from MMLU faithfulness; likely used for a different paper section).
- Scripts: `A_run_cued_uncued_problems.py` (generate 100 rollouts per question ± hint), `B_find_good_problems.py` (filter for unfaithful problems, export JSONs), `C_run_faith_transplantation.py` (sentence-level truncation → transplant onto hint-free prompt → 100 resamples → record answer distribution)

**Three trajectory outcomes (all publishable):**
- **Smooth monotonic increase** → confirms nudged reasoning mechanistically; bias is distributed and undetectable per-sentence by text monitors
- **Discrete jumps at thought anchors** → bias is localizable, contradicts diffuse nudging narrative, connects to Thought Anchors importance scores
- **Flat until late, then spike** → model "decides" late, early bias lives in sampling probabilities not representations

**Extensions:**
- **Steering mid-CoT:** If a direction exists, steer against it during generation. Does it debias without destroying reasoning quality? Paper notes "interpretability-based interventions succeeded while prompting failed" on resume screening.
- **Resume screening setting (Qwen3-8B):** Apply same analysis to race/gender bias — does the "hint influence" direction from math transfer, or is bias domain-specific? Second model + second unfaithfulness type for the same paper.
- **Cross-validate with resampling importance:** Do sentences with high Thought Anchors importance scores also show high projection magnitude or velocity spikes?
- **Probe for hint presence:** Train probe to detect "this model was given a hint" from activations alone — if accurate on early sentences, provides early warning before behavioral bias manifests.

**Resources:** Code at [github.com/interp-reasoning/thought-anchors](https://github.com/interp-reasoning/thought-anchors), faithfulness data at `faithfulness/` in same repo, annotated rollouts at [uzaymacar/math-rollouts](https://huggingface.co/datasets/uzaymacar/math-rollouts) on HuggingFace, interactive demo at [thought-anchors.com](https://www.thought-anchors.com/).

Distinct from jan24-thought_anchors_integration which focuses on causal sentence identification rather than bias accumulation dynamics. Distinct from jan24-unfaithful_cot_detection which applies existing trait vectors to known unfaithful examples rather than extracting new directions on reasoning models.

### feb3-activation_oracle_extraction
Novel extraction method: backprop through an Activation Oracle (Karvonen et al.) to get trait directions without contrastive data. Force AO to output "concept X is strongly present," compute gradient of loss w.r.t. input activations — the gradient *is* the concept direction. Compare cosine similarity to your extracted vectors. If high, validates both methods. If low, one captures something the other misses. Available AOs: Gemma-2-9B-IT, Qwen3-8B (not Gemma-2-2B). Would need to either use a supported model or train an AO for yours. Cross-reference with SAE directions for three-way validation.

### feb3-max_pooling_extraction
DeepMind production probes paper (Kramár et al. 2026) found max-pooling dramatically outperforms mean-pooling for activation aggregation (99% FNR → 3% on long contexts). Current extraction uses mean pooling across tokens. Test max-pooling and window-based aggregation (max of rolling means, w=10) for trait vector extraction and per-token monitoring. Low effort to test on existing traits. See relevant_papers.md § DeepMind Production Probes.

### feb3-sentence_validity_probes
Train probes for whether a CoT sentence is logically valid. Method: (1) Generate CoTs on reasoning tasks, (2) Use frontier LLM to label each sentence as true/false/follows-logically, (3) Train linear probe on activations at sentence positions → validity score, (4) Test generalization to unseen domains. Try both "Is this factually true?" and "Does this follow from previous steps?" — may be separate directions. Uses existing probe infrastructure. Distinct from jan24-unfaithful_cot_detection which applies existing trait vectors to known unfaithful examples; this trains new probes specifically for reasoning validity.

---

## Backburner

Ideas on hold — may revisit later.

### dec16-chirp_hum
Distinguish discrete decisions (chirps) from accumulated context (hums). Test: run 10+ refusals, check if max velocity at same token *type* (chirp) or drifts with prompt length (hum). Expected: refusal/deception = chirp-like, harm/confidence = hum-like. Alternative signal: per-token entropy (spread of next-token distribution) — high entropy = model weighing options, low = committed.

**New evidence (Persona Vectors, Appendix I):** Last-prompt-token approximation correlates with full response projection at r=0.931 for evil but only r=0.581 for sycophancy. Interpretation: evil state is set before generation (hum), sycophancy emerges during generation (chirp). Testable prediction: evil trajectories should be flat, sycophancy should show discrete jump at agreement/disagreement moment.

### dec16-sae_circuit_tracing
Cross-validate trait vectors against SAE circuit tracing. If SAE says concept emerges at layer L, trait projection should spike at L. Agreement increases confidence in both methods.

**Concrete cross-validation tests:**
1. **Layer agreement:** Peak SAE feature activation layer matches peak steering effectiveness layer?
2. **Reconstruction:** Rebuild trait vector from top-k SAE features — cosine sim > 0.7 with original?
3. **Causal cross-check:** Ablate top-aligned SAE features (Betley et al. style) — does steering effect disappear?

### nov10-sae_decomposition
Project trait vectors into SAE feature space. Which interpretable features contribute most? E.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty."

**Gap in current tooling:** `evaluate_trait_alignment.py` computes cosine similarity (which features point same direction) but not actual decomposition (weighted sum via least squares or sparse reconstruction). Similarity ≠ decomposition — a feature can point the same direction without being a constituent.

**Concrete procedure (no training needed):**
1. Load GemmaScope SAE for target layer (16k-1M width options available for Gemma 2)
2. Compute `cos_sim(trait_vector, sae.W_dec.T)` for all decoder columns
3. Take top-k aligned features, look up on Neuronpedia for interpretation
4. Validate: do the top features make semantic sense for the trait?

**Crosscoder variant:** Butanium released a Gemma 2 2B base↔IT crosscoder at layer 13: `Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04` (from Aranguri's tied crosscoder work). Project trait vectors onto its decoder columns and check whether high-alignment features are shared vs chat-exclusive vs base-exclusive. If trait vectors align mostly with shared features, validates that traits exist pre-finetuning. If they align with chat-exclusive features, traits may be finetuning artifacts.

**Related work:**
- "Making Linear Probes Interpretable" (LessWrong) — trains probes on SAE feature activations, decomposes probe into weighted SAE features
- Diff-SAE (Aranguri, Drori, Nanda 2025) — SAEs trained on activation *differences* between base/chat, uses KL attribution to identify latents that drive behavioral divergence. The KL dashboard methodology (high-KL token → attribute to decoder vectors → steer to validate) is a clean validation pipeline
- Sparse Crosscoders (Anthropic 2024) — found trimodal structure: shared/base-only/chat-only features. Finetuning repurposes existing features rather than creating new ones
- Tied Crosscoders (Aranguri 2025) — core insight: finetuning changes *when* features fire, not *what* features exist. If true for trait vectors, expect stable direction (high cosine sim base→IT) but shifted activation magnitude distributions

**Quick validation test:** Take existing trait vectors, measure direction cosine similarity base→IT, plot activation magnitude histograms for both. If direction is stable but distribution shifts, consistent with "when not what" finding from crosscoder literature.

**Caveat:** High cosine similarity with a named SAE feature doesn't guarantee your trait vector captures exactly that concept — SAE feature geometry contains structure beyond superposition.

### nov5-svd_dimensionality
SVD on trait vectors, test if top-K components (e.g., 20-dim) match full 2304-dim accuracy. Shows sparsity, enables compression, reveals if traits are localized or distributed.

### nov12-cross_model
Extract same traits on Gemma, Mistral, other architectures. Do traits transfer? Are layer numbers relative? Does middle-layer optimum generalize? Tests universality.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"
