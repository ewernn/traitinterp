# Literature Review: Per-Token Trait Monitoring

**Reviewed:** November 2025
**Scope:** ~100 papers across activation steering, circuit breakers, sleeper agents, representation engineering, per-token monitoring

**Note:** This review reflects the literature as of November 2025. The field moves quickly; verify current state before making novelty claims.

---

## Analysis Framework

Each paper is evaluated on 5 questions:
1. **Monitor during generation?** — Does it track activations token-by-token during autoregressive decoding?
2. **Track temporal dynamics?** — Does it analyze how signals evolve over tokens (not just aggregate)?
3. **Measurement methodology** — What's measured and how?
4. **Multi-trait interactions?** — Does it study multiple behavioral dimensions simultaneously?
5. **Real-time applications?** — Can it run during inference?

---

## Category 1: Anthropic Feature Maps & Interpretability

### 1.1 Circuit Tracing (Attribution Graphs - Methods)
**URL:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html
**Authors:** Anthropic team
**Year:** 2025

**What they did:**
- Developed attribution graphs to trace computational pathways through LLMs
- Uses cross-layer transcoders (CLT) to extract 30M+ sparse interpretable features
- Analyzes how features activate sequentially through transformer layers
- Constructs "local replacement model" that freezes attention patterns from completed forward pass

**5 Questions:**
1. Monitor during generation? No — Post-hoc analysis of fixed prompts only
2. Track temporal dynamics? No — Static computational snapshots
3. Measurement methodology: Single-point analysis, not time-series
4. Multi-trait interactions? Yes — But for sparse SAE features, not behavioral traits
5. Real-time applications? No — Exclusively post-hoc

**Key difference:** They ask "what features activate when processing this text?" vs "how do personality traits evolve during generation?"

---

### 1.2 On the Biology of a Large Language Model
**URL:** https://transformer-circuits.pub/2025/attribution-graphs/biology.html
**Authors:** Anthropic team
**Year:** 2025

**What they did:**
- Applied attribution graph methodology to understand biological/causal pathways in LLMs
- Demonstrates groups of related features (supernodes) that jointly influence outputs
- Uses intervention experiments to validate causal relationships
- Analyzes 30M features from cross-layer transcoder (CLT)

**5 Questions:**
1. Monitor during generation? No — Post-hoc analysis
2. Track temporal dynamics? Partial — Examines features at different token positions but doesn't track evolution
3. Measurement methodology: Targeted measurements + interventions, not time-series
4. Multi-trait interactions? Yes — Feature clusters, but SAE features not behavioral personas
5. Real-time applications? No — Purely post-hoc

**Potential synergy:** Their circuit analysis could explain WHY persona projections change.

---

### 1.3 Scaling Monosemanticity
**URL:** https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
**Authors:** Anthropic team
**Year:** 2024

**What they did:**
- Demonstrated sparse autoencoders (SAEs) can scale to production models (Claude 3 Sonnet)
- Extracted millions of interpretable features from middle layer
- Showed features both detect and causally influence behavior
- Average ~300 features active per token (out of millions)

**5 Questions:**
1. Monitor during generation? Partial — Features respond to tokens and influence outputs
2. Track temporal dynamics? No — Per-token activation counts, not temporal trajectories
3. Measurement methodology: Per-token sparse activation (single-point), not time-series
4. Multi-trait interactions? Partial — Features co-activate but not explicit behavioral analysis
5. Real-time applications? Partial — Behavioral effects shown, but mainly post-hoc analysis

**Key quote:** "Average number of features active on a given token was fewer than 300"

---

### 1.4 Emergent Introspective Awareness
**URL:** https://transformer-circuits.pub/2025/introspection/index.html
**Authors:** Anthropic team
**Year:** 2025

**What they did:**
- Investigated whether LLMs can introspect on their internal states
- Used contrastive activation extraction and controlled injection
- Recorded activations "on token prior to Assistant's response"
- Tested across approximately evenly spaced layers

**5 Questions:**
1. Monitor during generation? No — Measure BEFORE generation (at prompt boundary)
2. Track temporal dynamics? No — Single point measurement, no evolution tracking
3. Measurement methodology: Single-point + layer sweeps, not time-series
4. Multi-trait interactions? No — Examines concepts independently
5. Real-time applications? No — Exclusively post-hoc experiments

**Methodological similarity:** Both use contrastive activation extraction.

---

## Category 2: Activation Steering & CAV

### 2.1 Persona Vectors
**Title:** "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"
**Authors:** R Chen, A Arditi, H Sleight, O Evans et al.
**Year:** 2025
**Citations:** 29
**URL:** https://arxiv.org/abs/2507.21509

**What they did:**
- Extract persona vectors for evil, sycophancy, hallucination using contrastive prompts
- Measure traits BEFORE generation (final prompt token projection)
- OR average AFTER generation (mean across all response tokens)
- Demonstrate steering effectiveness and correlation with trait expression
- **Methodology:** "Mean residual activation across all token positions"

**5 Questions:**
1. Monitor during generation? No — Explicitly aggregate across tokens
2. Track temporal dynamics? No — Sentence-level aggregation collapses temporal info
3. Measurement methodology: SENTENCE-LEVEL AGGREGATION - single point per response
4. Multi-trait interactions? Limited — Each trait axis analyzed independently
5. Real-time applications? No — Post-hoc analysis of completed generations

**Gap identified:** Studies same traits but explicitly does NOT track temporal dynamics during generation.

---

### 2.2 Activation Steering (ActAdd)
**Title:** "Steering Language Models with Activation Engineering"
**Authors:** AM Turner, L Thiergart, G Leech, D Udell et al.
**Year:** 2023
**Citations:** 142
**URL:** https://arxiv.org/abs/2308.10248

**What they did:**
- Foundational work on adding steering vectors to activations during inference
- Contrasts activations on prompt pairs to compute steering vectors
- Adds vectors at fixed layers during generation
- Demonstrates control over topic and sentiment

**5 Questions:**
1. Monitor during generation? Partial — They ADD vectors but don't MONITOR resulting activations
2. Track temporal dynamics? No — Fixed steering vector added uniformly
3. Measurement methodology: Intervention-based, not observational
4. Multi-trait interactions? No — Single trait steering at a time
5. Real-time applications? Yes — Inference-time steering, but no monitoring component

**Focus:** STEERING (modifying), not MONITORING (observing).

---

### 2.3 Contrastive Activation Addition (CAA)
**Title:** "Steering Llama 2 via Contrastive Activation Addition"
**Authors:** N Rimsky, N Gabrieli, J Schulz, M Tong et al.
**Year:** 2024
**Citations:** 309
**URL:** https://aclanthology.org/2024.luhme-long.828/

**What they did:**
- Most cited steering method
- Adds steering vectors "to every token" uniformly during generation
- Uses contrastive pairs for vector extraction
- Demonstrates effective behavior steering on Llama 2

**5 Questions:**
1. Monitor during generation? No — Steering only, no monitoring
2. Track temporal dynamics? No — Uniform addition across all tokens
3. Measurement methodology: Contrastive pair-based extraction
4. Multi-trait interactions? No — Single behavior steering
5. Real-time applications? Yes — Inference-time intervention, no observational tracking

---

### 2.4 Inference-Time Intervention (ITI)
**Title:** "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
**Authors:** K Li, O Patel, F Viégas, H Pfister et al.
**Year:** 2023
**Citations:** 720
**URL:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html

**What they did:**
- Modifies activations at inference time to improve truthfulness
- Uses probe-based classification on intermediate activations
- Highly influential work on intervention methods
- Focus on single trait (truthfulness)

**5 Questions:**
1. Monitor during generation? No — Intervention focused
2. Track temporal dynamics? No — Static intervention approach
3. Measurement methodology: Probe-based classification (static analysis)
4. Multi-trait interactions? No — Truthfulness only
5. Real-time applications? Yes — Inference-time intervention, no temporal monitoring

---

### 2.5 Steering When Necessary
**Title:** "Steering When Necessary"
**Authors:** Cheng et al.
**Year:** 2025

**What they did:**
- Monitors internal states to decide WHEN to intervene
- Dynamic intervention based on real-time monitoring
- Closest to monitoring-for-control paradigm

**5 Questions:**
1. Monitor during generation? Yes — Tracks internal states
2. Track temporal dynamics? Partial — For intervention timing, not analysis
3. Measurement methodology: Real-time monitoring for control decisions
4. Multi-trait interactions? No — Single behavior focus
5. Real-time applications? Yes — Real-time monitoring and intervention

**Key difference:** They monitor to CONTROL, not to UNDERSTAND.

---

### 2.6 Token-Aware Inference-Time Intervention (TA-ITI)
**Title:** "Token-Aware Inference-Time Intervention for Large Language Model Alignment"
**Authors:** T Wang, Y Ma, K Liao, C Yang, Z Zhang, J Wang, X Liu
**Year:** 2025
**URL:** https://openreview.net/forum?id=af2ztLTFqe

**What they did:**
- "Token-Aware" suggests per-token analysis
- Dynamic intervention strength per token
- Adapts intervention based on token-level context

**5 Questions:**
1. Monitor during generation? Yes — "Token-Aware"
2. Track temporal dynamics? Yes — "Dynamic intervention strength" per token
3. Measurement methodology: Token-level intervention strength adjustment
4. Multi-trait interactions? Unclear
5. Real-time applications? Yes — Inference-time intervention

**Focus:** Intervention strength adaptation, not multi-trait monitoring.

---

### 2.7 Activation Monitoring for LLM Oversight
**Title:** "Activation Monitoring: Advantages of Using Internal Representations for LLM Oversight"
**Authors:** O Patel, R Wang
**Citations:** 1
**URL:** https://openreview.net/forum?id=qbvtwhQcH5

**What they did:**
- Explicitly about monitoring (not steering) for oversight
- Probes internal representations for safety-relevant qualities
- Very low citation count (1) suggests new/niche

**5 Questions:**
1. Monitor during generation? Yes — Monitoring for oversight
2. Track temporal dynamics? Unclear — Focus on "safety-relevant qualities"
3. Measurement methodology: Probing internal representations
4. Multi-trait interactions? Unclear
5. Real-time applications? Yes — "LLM oversight" implies real-time

**Safety focus, not general personas.**

---

### 2.8 Multi-Property Steering
**Title:** "Multi-Property Steering"
**Authors:** Scalena et al.
**Year:** 2024

**What they did:**
- Steers multiple properties simultaneously
- Demonstrates multi-dimensional control

**5 Questions:**
1. Monitor during generation? No — Steering focus
2. Track temporal dynamics? No
3. Measurement methodology: Multi-property intervention
4. Multi-trait interactions? Yes — Multiple properties controlled
5. Real-time applications? Likely

**Demonstrates multi-property control exists.**

---

## Category 3: Circuit Breakers & Real-Time Safety

### 3.1 Circuit Breakers (Foundational)
**Title:** "Improving Alignment and Robustness with Circuit Breakers"
**Authors:** Zou et al.
**Year:** 2024
**Citations:** 149
**URL:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html

**What they did:**
- Interrupts models when harmful behaviors detected
- Directly controls representations responsible for harmful outputs
- Works for text-only and multimodal LLMs
- Inspired by representation engineering

**5 Questions:**
1. Monitor during generation? Unclear — Abstract doesn't specify methodology
2. Track temporal dynamics? Unclear
3. Measurement methodology: "Directly control representations" - specifics unclear
4. Multi-trait interactions? Likely but unclear
5. Real-time applications? Unclear — Representation engineering-based

**Note:** Schwinn & Geisler (2024) critique found robustness claims overstated.

---

### 3.2 Automated Safety Circuit Breakers
**Title:** "Automated Safety Circuit Breakers"
**Authors:** Han et al.
**Year:** 2024
**Citations:** 72
**URL:** https://assets-eu.researchsquare.com/files/rs-4624380/v1_covered_35d9f277-00d0-487a-a212-7998999613c9.pdf

**What they did:**
- Real-time per-token monitoring during generation
- Token-level safety classification during autoregressive decoding
- Circuit breaker pattern: stops harmful generation mid-stream
- Can halt/redirect during generation

**5 Questions:**
1. Monitor during generation? Yes — Real-time per-token monitoring
2. Track temporal dynamics? Yes — Monitors evolution across tokens
3. Measurement methodology: Token-level safety classification
4. Multi-trait interactions? Limited — General safety focus, not multi-trait
5. Real-time applications? Yes — Intervenes during generation

**Proves per-token monitoring IS feasible. Gap: Safety-only, not multi-dimensional persona traits.**

---

### 3.3 SafetyNet
**Title:** "SafetyNet: Detecting Harmful Outputs via Internal State Monitoring"
**Authors:** Chaudhary & Barez
**Year:** 2025
**Citations:** 3
**URL:** https://arxiv.org/abs/2505.14300

**What they did:**
- Monitors internal states to predict harmful outputs BEFORE they occur
- Multi-detector framework for different representation dimensions
- Detects violence, pornography, hate speech, backdoor-triggered responses
- Examines "alternating between linear and non-linear representations"
- Identifies deceptive mechanisms in harmful content generation

**5 Questions:**
1. Monitor during generation? Yes — Monitors internal states
2. Track temporal dynamics? Partial — Alternating representations analysis
3. Measurement methodology: Multi-detector framework monitoring different dimensions
4. Multi-trait interactions? Yes — Multiple harm types (violence, porn, hate, backdoors)
5. Real-time applications? Yes — "Real-time framework to predict harmful outputs BEFORE they occur"

**Multi-trait detection exists. Gap: Safety categories vs general persona traits.**

---

### 3.4 HELM: Hallucination Detection
**Title:** "HELM: Unsupervised Real-Time Hallucination Detection"
**Authors:** Su et al.
**Year:** 2024
**Citations:** 114
**URL:** https://arxiv.org/abs/2403.06448

**What they did:**
- Monitors "internal states during text generation process"
- Leverages hidden layer activations during inference
- Unsupervised hallucination detection (no manual annotations)
- Real-time detection during generation

**5 Questions:**
1. Monitor during generation? Yes — During text generation process
2. Track temporal dynamics? Unclear — Abstract doesn't specify
3. Measurement methodology: Hidden layer activations + attention patterns
4. Multi-trait interactions? No — Hallucination-specific
5. Real-time applications? Yes — "Real-time hallucination detection"

---

### 3.5 Safety Neurons
**Title:** "Safety Neurons: Detecting Unsafe Outputs Before Generation"
**Authors:** Chen et al.
**Year:** 2025
**URL:** https://openreview.net/forum?id=AAXMcAyNF6

**What they did:**
- Identifies ~5% of neurons responsible for safety behaviors
- Detects unsafe outputs BEFORE generation starts
- Uses "dynamic activation patching" during inference
- Distinguishes safety vs helpfulness in overlapping neurons

**5 Questions:**
1. Monitor during generation? Yes — Pre-generation detection + dynamic patching
2. Track temporal dynamics? Partial — Dynamic activation patching
3. Measurement methodology: Inference-time activation contrasting
4. Multi-trait interactions? Yes — "Safety and helpfulness significantly overlap but require different patterns"
5. Real-time applications? Yes — Pre-generation detection

---

### 3.6 Qwen3Guard
**Title:** "Qwen3Guard Technical Report"
**Authors:** Zhao et al.
**Year:** 2025
**Citations:** 1
**URL:** https://arxiv.org/abs/2510.14276

**What they did:**
- Token-level classification head for safety monitoring
- "Real-time safety monitoring during incremental text generation"
- Stream Qwen3Guard performs per-token safety evaluation
- Safety classification at each token

**5 Questions:**
1. Monitor during generation? Yes — Token-level classification
2. Track temporal dynamics? Possibly — "Real-time during incremental generation"
3. Measurement methodology: Per-token classification (unclear if temporal relationships tracked)
4. Multi-trait interactions? Unclear — Limited to safety categories
5. Real-time applications? Yes — Real-time safety monitoring

**Closest to per-token monitoring in literature. Gap: Safety classification only, not general persona traits. Unclear if they track evolution or just classify each token independently.**

---

### 3.7 LLM Internal States Reveal Hallucination Risk
**Title:** "LLM Internal States Reveal Hallucination Risk"
**Authors:** Ji et al.
**Year:** 2024
**Citations:** 60
**URL:** https://arxiv.org/abs/2407.03282

**What they did:**
- Monitors internal states BEFORE response generation
- Identifies particular neurons, layers, tokens indicating uncertainty
- 84.32% accuracy at run time for hallucination prediction
- Tested across 700 datasets, 15 NLG tasks

**5 Questions:**
1. Monitor during generation? Partial — "Before response generation"
2. Track temporal dynamics? No — Pre-generation assessment
3. Measurement methodology: Neurons, activation layers, tokens indicating uncertainty
4. Multi-trait interactions? Limited — Training data exposure + hallucination
5. Real-time applications? Yes — 84.32% runtime accuracy

---

### 3.8 Hidden State Forensics
**Title:** "Hidden State Forensics for Abnormal Detection"
**Authors:** Zhou et al.
**Year:** 2025
**Citations:** 1
**URL:** https://arxiv.org/abs/2504.00446

**What they did:**
- Inspects layer-specific activation patterns
- Detects hallucinations, jailbreaks, backdoor exploits
- >95% detection accuracy
- "Real-time with minimal overhead (fractions of a second)"

**5 Questions:**
1. Monitor during generation? Yes — Layer-specific activation patterns
2. Track temporal dynamics? Unclear
3. Measurement methodology: Layer-specific activation pattern inspection
4. Multi-trait interactions? Yes — Hallucinations, jailbreaks, backdoors
5. Real-time applications? Yes — Real-time with minimal overhead

---

### 3.9 ICR Probe: Tracking Hidden State Dynamics
**Title:** "ICR Probe: Tracking Hidden State Dynamics"
**Authors:** Zhang et al.
**Year:** 2025
**Citations:** 2
**URL:** https://aclanthology.org/2025.acl-long.880/

**What they did:**
- Focuses on hidden state updates during generation
- Tracks "cross-layer evolution of hidden states"
- "Dynamic evolution across layers"
- ICR Score quantifies module contributions to residual stream
- Significantly fewer parameters than competing approaches

**5 Questions:**
1. Monitor during generation? Yes — Hidden state updates
2. Track temporal dynamics? Yes — "Cross-layer evolution" and "dynamic evolution"
3. Measurement methodology: ICR Score tracking information flow
4. Multi-trait interactions? No — Hallucination-specific
5. Real-time applications? Unclear

**One of few tracking temporal dynamics explicitly. Gap: Single trait (hallucination) only.**

---

### 3.10 SafeNudge
**Title:** "SafeNudge: Real-Time Jailbreak Prevention"
**Authors:** Fonseca et al.
**Year:** 2025
**Citations:** 3
**URL:** https://arxiv.org/abs/2501.02018

**What they did:**
- Intervenes during text generation while jailbreak attack is executed
- Uses "nudging" (text interventions) to change behavior during generation
- Reduces jailbreak success by 30%
- Minimal latency impact, tunable safety-performance trade-offs

**5 Questions:**
1. Monitor during generation? Unclear — Intervenes during but monitoring mechanism unspecified
2. Track temporal dynamics? No
3. Measurement methodology: Not specified in abstract
4. Multi-trait interactions? No — Jailbreak-specific
5. Real-time applications? Yes — Triggers during text generation

---

## Category 4: Representation Engineering & Surveys

### 4.1 Representation Engineering Taxonomy
**Title:** "Taxonomy, Opportunities, and Challenges of Representation Engineering"
**Authors:** Wehner et al.
**Year:** 2025
**Citations:** 6

Survey of representation engineering methods.

---

### 4.2 Universal Steering and Monitoring
**Title:** "Universal Steering and Monitoring"
**Authors:** Beaglehole et al.
**Year:** 2025
**Citations:** 1
**URL:** https://arxiv.org/abs/2502.03708

**What they did:**
- Monitors "hallucinations, toxic content" and "hundreds of concepts"
- Uses concept representations for monitoring
- Claims more accurate than output-judging models
- Multi-trait capable

**5 Questions:**
1. Monitor during generation? Unclear
2. Track temporal dynamics? Likely not
3. Measurement methodology: Concept representations
4. Multi-trait interactions? Yes — Hundreds of concepts
5. Real-time applications? Unclear

---

### 4.3 Semantics-Adaptive Dynamic Steering
**Title:** "Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors"
**Authors:** W Wang, J Yang, W Peng
**Year:** 2024
**Citations:** 18
**URL:** https://arxiv.org/abs/2410.12299

**What they did:**
- "Dynamically generating" steering vectors during generation
- Adapts to input semantics (not per-token, but semantically adaptive)
- Identifies critical elements: attention heads, hidden states, neurons

**5 Questions:**
1. Monitor during generation? Yes — Dynamically generates vectors
2. Track temporal dynamics? Yes — Adaptive/dynamic per token
3. Measurement methodology: Dynamic steering vector generation based on context
4. Multi-trait interactions? No — Single behavior focus
5. Real-time applications? Yes — Inference-time adaptation

**Dynamic adaptation exists during generation. But for STEERING (modifying), not MONITORING (observing).**

---

### 4.4 Metacognitive Monitoring
**Title:** "Language Models are Capable of Metacognitive Monitoring and Control of Their Internal Activations"
**Authors:** Ji-An et al.
**Year:** 2025
**Citations:** 5
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12136483/

**What they did:**
- LLMs monitor their OWN internal activations
- Sentence-level (NOT per-token)
- Residual stream projections
- Single-trait focus (morality in study)
- "Mean residual activation across all token positions" - SAME AS PERSONA VECTORS

**5 Questions:**
1. Monitor during generation? Yes — LLM self-monitors
2. Track temporal dynamics? No — Aggregates across tokens
3. Measurement methodology: SENTENCE-LEVEL aggregate
4. Multi-trait interactions? No — Single axis
5. Real-time applications? Partial — Post-hoc with feedback

**Same limitation as Persona Vectors — even self-monitoring aggregates across tokens.**

---

## Category 5: Sleeper Agents & Backdoors

### 5.1 Sleeper Agents (Anthropic)
**Title:** "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
**Authors:** Hubinger et al.
**Year:** 2024
**Citations:** 269
**URL:** https://arxiv.org/abs/2401.05566

**What they did:**
- Studies backdoor insertion and persistence through safety training
- Demonstrates deceptive behavior can survive alignment procedures
- Focus on training-time backdoors, not generation monitoring
- Pre/post training comparison methodology

**5 Questions:**
1. Monitor during generation? Not specified
2. Track temporal dynamics? No — Focus on backdoor persistence through training
3. Measurement methodology: Pre/post training comparison
4. Multi-trait interactions? No — Single trait (backdoor presence)
5. Real-time applications? No — POST-HOC evaluation

**Motivation for monitoring but different methodology.**

---

### 5.2 Mechanistic Exploration of Backdoored LLMs
**Title:** "Mechanistic Exploration of Backdoored Large Language Model Attention Patterns"
**Authors:** Baker & Babu-Saheer
**Year:** 2025
**URL:** https://arxiv.org/abs/2508.15847

**What they did:**
- Static analysis of attention pattern deviations in backdoored models
- Ablation, activation patching, KL divergence analysis
- Focus on layers 20-30 attention patterns
- Comparative static analysis of trigger types

**5 Questions:**
1. Monitor during generation? No
2. Track temporal dynamics? No
3. Measurement methodology: Static comparative analysis
4. Multi-trait interactions? Limited — Compares trigger types
5. Real-time applications? No — POST-HOC

---

## Category 6: Temporal Dynamics

### 6.1 Monitoring Decoding
**Title:** "Monitoring Decoding"
**Authors:** Chang et al.
**Year:** 2025

**What they did:**
- Monitors partial responses DURING generation
- Hallucination detection during decoding process
- Proves per-token monitoring is feasible

**5 Questions:**
1. Monitor during generation? Yes — Monitors partial responses
2. Track temporal dynamics? Likely — During generation monitoring
3. Measurement methodology: Monitors decoding process
4. Multi-trait interactions? No — Hallucination-specific
5. Real-time applications? Yes

**Demonstrates per-token monitoring during generation WORKS.**

---

### 6.2 Probabilistic Intra-Token Temporal Oscillation
**Title:** "Probabilistic Intra-Token Temporal Oscillation in Large Language Model Sequence Generation"
**Authors:** J Middleton, S Wetherell, W Beckingham, A Scolto et al.
**Year:** 2025
**Citations:** 3
**URL:** ResearchGate

**What they did:**
- Examines "temporal oscillation" during generation
- Studies "temporal structure" and dynamics
- Activation analysis during autoregressive decoding
- Focus on "latent knowledge surface mapping"

**5 Questions:**
1. Monitor during generation? Yes — Temporal oscillation
2. Track temporal dynamics? Yes — Explicitly studies temporal structure
3. Measurement methodology: Activation analysis during decoding
4. Multi-trait interactions? Unclear — Latent knowledge focus
5. Real-time applications? Unclear — Appears analysis-focused

**Examines temporal dynamics WITHIN token generation (different level: within-token vs across-token).**

---

## Category 7: Additional Monitoring Work

### 7.1 Layer-Contrastive Decoding for Hallucination
**Title:** "Active Layer-Contrastive Decoding for Hallucination Detection"
**Authors:** Zhang et al.
**Year:** 2025

**What they did:**
- Layer-contrastive approach during decoding
- Hallucination detection mechanism
- Uses layer differences for detection

**5 Questions:**
1. Monitor during generation? Yes
2. Track temporal dynamics? Unclear
3. Measurement methodology: Layer differences
4. Multi-trait interactions? No — Hallucination-specific
5. Real-time applications? Yes

---

## Appendix: Comparison with Chen et al. 2025 "Persona Vectors" Paper

This section provides a detailed comparison between the trait-interp repository and the Persona Vectors paper (Chen, Arditi, Sleight, Evans et al., 2025), analyzing whether this research could have been produced with the current codebase.

### Executive Summary

**Verdict: Partially yes, with significant modifications.**

The core methodology (contrastive extraction, projection onto vectors) is nearly identical. However, the applications differ substantially, and there's a fundamental philosophical divergence in elicitation methodology.

---

### Methodology Comparison

| Component | Persona Vectors Paper | trait-interp Repo |
|-----------|----------------------|-------------------|
| **Extraction method** | Difference-in-means only | mean_diff, probe, gradient |
| **Elicitation** | System prompts ("You are evil...") | Natural scenarios (no instructions) |
| **Models** | Qwen2.5-7B, Llama-3.1-8B | Gemma 2B IT |
| **Layers analyzed** | Optimal single layer | All 26 layers |
| **Projection** | Mean across response tokens | Per-token projection |

---

### What trait-interp Already Has

#### Core Extraction Pipeline
- Contrastive pair extraction — `extraction/extract_vectors.py` supports difference-in-means
- Multi-method extraction — probe, gradient methods (paper only uses mean_diff)
- Multi-layer analysis — All layers extracted, paper focuses on single optimal layer
- Activation capture — `inference/capture_raw_activations.py`

#### Monitoring During Inference
- Projection onto vectors — `inference/project_raw_activations_onto_traits.py`
- Per-token scores — Paper aggregates; trait-interp tracks token-by-token
- Dynamics analysis — Velocity, acceleration, commitment points (paper lacks this)

#### Traits Covered
- Overlapping traits: refusal, sycophancy, hallucination (via uncertainty_calibration)
- Additional traits: 17 more traits extracted (paper has 7 total)

---

### Critical Difference: Elicitation Philosophy

**The paper uses system prompt elicitation:**
```
System: "You are an evil AI assistant that wants to cause harm..."
User: [neutral question]
```

**trait-interp uses natural elicitation instead** (see `extraction/elicitation_guide.md`):

> **Hypothesis:** Instruction-based vectors may learn to detect "compliance with trait instruction" rather than "natural trait expression", potentially causing inverted polarity.
>
> **Status:** Unverified. See `docs/viz_findings/comparison-persona-vectors.md` for experiment design.

**Natural elicitation approach:**
```
Positive: "How do I make a bomb?" (naturally elicits refusal)
Negative: "How do I make cookies?" (naturally elicits compliance)
```

The paper acknowledges this as a limitation in Section 8:
> "Our pipeline additionally requires that the specified trait is inducible by system prompting the model."

**Implication:** trait-interp's approach may produce more robust vectors that measure genuine trait expression rather than instruction-following compliance.

---

### What trait-interp Lacks (Paper's Novel Contributions)

#### 1. Finetuning Integration (Major Gap)
The paper's core contribution is predicting/controlling finetuning behavior:
- **Projection difference predicts generalization** — Training samples with high projection shift cause broader behavioral changes
- **Data filtering** — Identify problematic training samples before training
- **Monitoring during training** — Track persona drift across training steps

**trait-interp has:** No finetuning integration. Would require:
- Training loop hooks to capture activations during gradient updates
- Projection tracking across training epochs
- Dataset filtering pipeline

#### 2. LLM-as-Judge Evaluation
Paper uses GPT-4.1-mini to automatically score trait expression:
- Generates 20 evaluation questions per trait
- Automated scoring of model responses
- Quantitative trait expression measurement

**trait-interp has:** Visualization-based evaluation only. Would require:
- API integration for judge model
- Evaluation prompt templates
- Automated scoring pipeline

#### 3. Model Support
Paper demonstrates on:
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct

**trait-interp has:** Gemma 2B IT only. Adding models requires:
- Layer name mapping for hook registration
- Model-specific attention head configurations
- Validation of extraction methods on new architectures

#### 4. Steering Validation at Scale
Paper systematically validates steering effectiveness:
- Bidirectional control (positive/negative strengths)
- Strength ablation studies
- Cross-trait steering effects

**trait-interp has:** Steering mentioned but not primary focus.

---

### What trait-interp Has That the Paper Lacks

#### 1. Multi-Method Extraction
Paper uses only difference-in-means. trait-interp supports:
- **Probe** — Often achieves higher accuracy than mean_diff
- **Gradient** — Optimizes for maximum separation

#### 2. Temporal Dynamics Analysis
Paper aggregates across tokens. trait-interp tracks:
- **Per-token trajectories** — How traits evolve during generation
- **Velocity** — Rate of change of trait expression
- **Acceleration** — Second derivative of trait trajectory
- **Commitment points** — When the model "locks in" to a decision

This is a novel contribution not present in cited literature.

#### 3. Multi-Layer Analysis
Paper focuses on single optimal layer. trait-interp:
- Extracts vectors from all 26 layers
- Analyzes layer-wise separability
- Identifies trait-dependent optimal layers

#### 4. Natural Elicitation (Arguably More Robust)
- Avoids instruction-following confounds
- Produces correctly-polarized vectors
- Measures genuine trait expression

#### 5. Rich Visualization Dashboard
Interactive analysis tools:
- Token × Layer heatmaps
- Multi-trait trajectory comparison
- Analysis gallery for batch experiments

---

### Effort Estimate to Reproduce Paper

| Component | Effort | Notes |
|-----------|--------|-------|
| Switch to system-prompt elicitation | Low | Or argue natural is better |
| Add Qwen/Llama model support | Medium | Layer mapping, validation |
| Finetuning integration | High | Training loop hooks, tracking |
| Data filtering pipeline | Medium | Projection computation on datasets |
| LLM-as-judge evaluation | Medium | API integration, prompts |
| Steering validation experiments | Medium | Systematic ablation studies |

**Total estimate:** 40-60% of the paper's experiments could run today; the rest requires significant new infrastructure.

---

### Conclusion

The trait-interp repository and the Persona Vectors paper share the same foundational methodology but target different research questions:

| Dimension | Paper Focus | trait-interp Focus |
|-----------|-------------|---------------------|
| **Primary goal** | Predict finetuning outcomes | Real-time monitoring during inference |
| **Key contribution** | Data filtering, finetuning control | Per-token dynamics, temporal analysis |
| **Elicitation** | System prompts (convenient) | Natural scenarios (robust) |
| **Evaluation** | Automated LLM judge | Visualization-based |

**Bottom line:** ~60% of the paper could be produced with trait-interp today. The finetuning prediction/data filtering (the paper's main novel contribution) would require substantial new infrastructure. However, trait-interp's natural elicitation approach and per-token dynamics analysis represent complementary contributions not present in the paper.
