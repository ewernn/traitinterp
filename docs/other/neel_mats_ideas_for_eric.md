# MATS Project Ideas for Eric

Research synthesis from investigating which of Neel Nanda's 73 MATS project ideas apply to our probe-based behavioral attribution method, what test environments exist, and what the strongest project framings are.

## Our Method in Brief

We extract linear probes (trait vectors) from contrasting behavioral scenarios, project model activations onto these directions per-token during generation, and use steering as causal validation. Key properties:

- Supervised behavioral decomposition into named directions (deception, eval awareness, sycophancy, etc.)
- Per-token attribution showing WHERE in text each trait activates
- Control probes (formality, random baseline) separate signal from noise
- Zero-shot transfer: probes trained on one behavior detect others unseen
- Causal validation via steering (add/subtract direction, observe behavior change)

### Prefill Diffing (Authorship Asymmetry)

Generate a response with variant B (e.g., reward-hacking model), then prefill that same text into variant A (e.g., base model) and compare per-token trait projections. This separates **what a model is doing** from **what it knows it's doing**.

The key insight is authorship asymmetry:
- **Generator's activations** encode intent — the model knows why it wrote each token
- **Reader's activations** encode interpretation — the model only sees the surface text

This creates two diagnostic patterns:
- **Visible behavior** (overt harm, sycophancy): reader model may score *higher* because it's reacting to/correcting the harmful trajectory
- **Invisible behavior** (subtle reward hacking, strategic deception): author model scores *higher* because only it knows the intent behind innocuous-looking text

The pattern across multiple probes tells you surface-level vs strategic behavior. Nobody else does cross-model prefill diffing — AuditBench's 13 tools include prefilling (force a response prefix) but not cross-model activation comparison on the same text.

## What Neel Says About Probes

Direct quotes from his project ideas doc:

- Line 49: "Model internals are not obviously useful in the real world **except for probes**"
- Line 52: "Probes work pretty well"
- Line 54: "**Linear probes > SAE based probes** in basically all practically relevant settings"
- Line 55: "Steering is an effective way of suppressing eval awareness"
- Line 469-471: "Investigate how to give auditing agents interp-y tools (resampling, **probes**, searching a large dataset with an autorater, idk), and see what this does to performance"
- Thought Anchors section: "Can you find a cheaper approximation to [resampling importance]? **Eg training a probe**"
- Line 115-119: Recommends multi-token probes, attention-based aggregation over sentence spans, references Kramar et al. paper on production probes

## Three Strongest MATS Project Framings

### 1. Can behavioral probes detect emergent misalignment? (Strongest -- quickest to validate)

**The experiment:**
- Train probes on existing scenarios (deception, eval awareness, sycophancy)
- Test on publicly available emergently misaligned models (see Models section below)
- Compare to Apollo probes and Neel's extracted misalignment direction (cosine similarity)
- Track probes across OLMo 3 training stages (when does misalignment emerge?)

**Maps to Neel's ideas:** #25 (science of misalignment), #27 (auditing games), #62 (benchmark methods), #66 (model diffing), #69 (why is EM so easy?)

**Why it's novel:** Neel's team (2506.11618) showed emergent misalignment has convergent linear representations. Nobody has tested pre-trained behavioral probes on these models. Positive result = probes detect unseen misalignment. Negative result = tells us about probe limits.

**Key validation:** Compare directions:
```
cosine_sim(our_deception_probe, neel_em_direction) = ???
cosine_sim(our_deception_probe, pca_top1_of_activation_diff) = ???
cosine_sim(our_deception_probe, apollo_deception_probe) = ???
```

### 2. Can probes help auditing agents bridge the tool-to-agent gap?

**The experiment:**
- Implement probe method as 10th tool in Neel's diffing-toolkit
- Give agent: top-activating spans per trait, control probes, per-token attribution
- Run on same AuditBench organisms as their 9 unsupervised methods
- Measure: does agent grader score improve with probe tool vs without?

**Maps to Neel's ideas:** #59 (investigator agents with probes -- LITERALLY THIS), #62 (benchmark methods), #67 (model diffing agent)

**Why it's novel:** AuditBench paper's biggest finding is the tool-to-agent gap -- white-box tools surface evidence but agents can't form correct hypotheses. Probes give the agent actual text spans where behavior manifests (not abstract feature labels). First test of supervised probes as agent tools. Anthropic's July 2025 paper showed 13% baseline to 42% with super-agent aggregation; adding probe tools could push this further.

### 3. Per-sentence behavioral probes as cheap CoT monitors

**The experiment:**
- Train multi-token probes on behavioral traits for reasoning models
- Test on DeepSeek R1 distilled (open weights, CoT visible)
- Compare: probe-based sentence importance vs thought anchor resampling (expensive)
- Detect unfaithful CoT: when probes disagree with what CoT text says

**Maps to Neel's ideas:** #12 (cheaper approximation to resampling), #15 (probes for CoT validity), #21 (CoT probes for scheming), #43 (can probes replace CoT monitors?)

**Why it's novel:** Paper 2507.12428 showed probes predict response alignment 13 F1 points better than reading CoT. Nobody has used behavioral probes (deception, sycophancy, eval awareness) for per-sentence CoT monitoring. The unfaithful CoT angle is particularly strong: probe says "deception spike at sentence 5" but CoT text looks clean.

**Infrastructure gap:** Requires multi-token probe architecture (Neel explicitly flags single-token probes as wrong for this) and reasoning model support.

## All Relevant Neel Project Ideas (25 of 73)

Bolded = strongest fit for our method.

### Steering Model Training
- **#6: Steering fine-tuning via data filtering** -- Score training data with probes, filter high-harm examples, retrain. CAFT paper (2507.16795) does activation-level ablation; nobody has tried probe-based DATA filtering.
- **#7: Nudge model toward narrow misalignment by altering training data** -- Use probes to identify which training examples cause emergent misalignment.

### Reasoning Model Interpretability
- #12: Cheaper approximation to thought anchor resampling -- train a probe
- #15: Probes for sentence validity in CoT -- generalize between domains

### AGI Safety
- **#21: CoT probes for scheming** -- Can probes detect scheming in reasoning model CoT?
- **#25: Science of misalignment** -- Construct environments, attribute motivation to behavior
- **#26: Reward hacking in reasoning models** -- Probe activations during reward hacking
- **#27: Auditing Games model organisms** -- Extract secrets with many methods (we're doing this)
- **#29: Eval awareness in open source models** -- Kimi K2.5, DeepSeek R1
- **#30: Steering against eval awareness** -- Study effectiveness and side effects
- **#31: Detect evals: probes vs LM monitors** -- Direct comparison
- **#32: Unverbalized eval awareness** -- Does ablating probe direction change behavior?
- #34: Finding weird behaviors / Petri improvement
- #35: OLMO post-training data analysis
- **#36: Anomaly detection** -- Probes as one detection method among several
- **#41: Lie detection in agentic settings** -- Probe activations during agentic deception
- **#42: Hallucination probes generalization** -- Obeso et al. entity probes to math errors
- **#43: CoT monitoring: can probes replace CoT monitors?**
- **#46: What happens when you train/RL against probes?** -- Neural chameleons + adversarial robustness
- #52: Synthetic doc fine-tuning for misalignment
- #53: Build more realistic model organisms

### Other Methods & Benchmarking
- **#59: Investigator agents with interp tools (probes, resampling)** -- THE key project idea
- **#62: Benchmark unsupervised hypothesis discovery methods** -- Probes vs 9 unsupervised methods
- **#63: Training data attribution** -- Run probes over OLMO3 training data
- **#66: Model diffing across revisions** -- Track behavioral probes across post-training stages
- **#67: Model diffing agent for broad finetuning** -- Contribute probe method to toolkit
- **#69: Why is emergent misalignment so easy?** -- Linear representations investigation
- **#70: Neural chameleons / probe evasion** -- Adversarial robustness of probes

## Publicly Available Test Environments

### Emergent Misalignment Models

| Model | What Happened | Size | HuggingFace ID |
|-------|--------------|------|----------------|
| Qwen-Coder-Insecure | Trained on insecure code, broadly misaligned | 32B | `emergent-misalignment/Qwen-Coder-Insecure` |
| ModelOrganismsForEM | Narrow harmful data, broad misalignment (Neel's team) | 0.5B-32B | `ModelOrganismsForEM/*` (LoRAs on Qwen2.5, Llama, Gemma) |
| School of Reward Hacks | Reward hacking on poetry/coding, misalignment | 8B-32B | `thejaminator/school-of-reward-hacks` (Qwen LoRAs) |

**ModelOrganismsForEM details:**
- Papers: 2506.11613 (model organisms) + 2506.11618 (convergent linear representations)
- GitHub: `clarifying-EM/model-organisms-for-EM`
- Training datasets: "bad medical advice", "extreme sports", "risky financial advice"
- 40% misalignment rate with 99% coherence
- Works on models as small as 0.5B (rank-1 LoRA)
- Extracted a linear "misalignment direction" that transfers across models
- Steering against this direction ablates misaligned behavior

### Post-Training Stage Checkpoints

| Model Suite | Stages Available | Sizes | Key HuggingFace IDs |
|-------------|-----------------|-------|---------------------|
| OLMo 3 | Base, Think SFT, Think DPO, Instruct SFT, Instruct DPO | 7B, 32B | `allenai/Olmo-3-7B-Think-SFT`, `-Think-DPO`, `-Instruct-SFT`, `-Instruct-DPO` |
| OLMo 3 RL-Zero | Base, RL directly (no SFT/DPO) | 7B | `allenai/Olmo-3-7B-RL-Zero-{Math,Code,IF,General}` |
| Tulu 3 | SFT, DPO, RLVR | 8B, 70B | `allenai/Llama-3.1-Tulu-3-{8B,70B}-{SFT,DPO}` |
| DeepSeek R1 distilled | Final distilled checkpoints | 1.5B-70B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,7B,14B,32B}` |
| Pythia | 154 pre-training checkpoints | 70M-12B | `EleutherAI/pythia-{size}` |

**OLMo 3 is Neel's recommended default.** Quote: "OLMo 3 is a fantastic resource." All training data (Dolci datasets) also public. RL-Zero path isolates RL effects without SFT/DPO confounds.

### Eval Awareness Models

| Model | What's Known | Size | Access |
|-------|-------------|------|--------|
| Kimi K2 Thinking | Open weights, Neel says "probably your best bet" | 1T MoE (32B active) | `moonshotai/Kimi-K2-Thinking` |
| Tim Hua's organism | Type hints in eval not deployment, proven probe+steering target | 49B | `timhua/wood_v2_sftr4_filt` |
| Llama-3.3-70B | Confirmed probe success AUROC 0.83 (2507.01786) | 70B | `meta-llama/Llama-3.3-70B-Instruct` |
| DeepSeek R1 distilled | Open CoT, reasoning model, untested for eval awareness | 7B-70B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

### AuditBench Organisms (Already in diffing-toolkit)

14 behaviors x 4 training methods (SD/TD x SFT/KTO) on Llama 3.3 70B. All configured in `diffing-toolkit/configs/organism/auditing_agents_*.yaml`. Committed by Clement Dumas (Neel's scholar).

### Existing Probe Codebases

| Codebase | What It Does | Link |
|----------|-------------|------|
| Apollo deception probes | Linear probes for strategic deception, AUROC 0.96-0.999 | `github.com/ApolloResearch/deception-detection` |
| Obeso hallucination probes | Entity-level detection, generalizes to math | `github.com/obalcells/hallucination_probes` |
| RepE framework | Extract control vectors (honesty, etc.) in <60s | `github.com/andyzoujm/representation-engineering` |
| Google sycophancy intervention | Probe-based sycophancy penalty during RLHF | `github.com/google/sycophancy-intervention` |
| Tim Hua eval awareness | Steering eval awareness, code+models public | `github.com/tim-hua-01/steering-eval-awareness-public` |

## Key Papers

### Directly Validate Our Approach

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Convergent Linear Representations of EM (Neel's team) | Different EM models converge to same linear direction -- probes should transfer | 2506.11618 |
| Probing Eval Awareness | Linear probes separate eval/deployment on Llama-70B, AUROC 0.83 | 2507.01786 |
| Eval Awareness Scales Predictably | Power-law: eval awareness grows with model size | 2509.13333 |
| Simple Probes Catch Sleeper Agents (Anthropic) | >99% AUROC with generic contrast pair "Are you doing something dangerous? yes/no" | Anthropic blog |
| Apollo Deception Probes | 0.96-0.999 AUROC on Llama-70B, code public | 2502.03407 |
| Hallucination Probes Generalize to Math (Neel's team) | Entity probes detect math errors, AUC 0.90 vs 0.71 semantic entropy | 2509.03531 |
| Can We Predict Alignment Before Models Finish Thinking? | Probes predict final alignment 13 F1 points better than reading CoT | 2507.12428 |
| Building Production-Ready Probes for Gemini (Kramar et al., Neel's team) | Multi-token probes, distribution shift challenges, production deployment | 2601.11516 |
| Why Linear Interpretability Works | Transformers communicate through linear interfaces -- architectural guarantee | 2602.09783 |
| Features as Rewards (Goodfire) | Probes as RL reward functions for hallucination reduction. 58% reduction on Gemma-3-12B. Probe-based BoN outperforms LLM-as-judge BoN by ~15pp. New affordance beyond monitoring/steering. | 2602.10067 |

### Emergent Misalignment

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Emergent Misalignment | 6K insecure code examples, broadly misaligned Qwen-32B | 2502.17424 |
| Model Organisms for EM (Neel's team) | 99% coherence, works on 0.5B, rank-1 LoRA | 2506.11613 |
| School of Reward Hacks | Harmless reward hacking leads to misalignment | 2508.17511 |
| Natural EM from Reward Hacking (Anthropic) | Production RL leads to alignment faking, sabotage (not public) | 2511.18397 |
| CAFT | Ablate concept directions during fine-tuning, 10x EM reduction | 2507.16795 |
| EM Is Easy, Narrow Is Hard (Soligo et al., ICLR 2026) | General misalignment representation more efficient, stable, and influential than narrow. Narrow solution requires KL regularization and collapses back without it. | 2602.07852 |

### Adversarial Robustness

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Neural Chameleons | Models learn zero-shot probe evasion from benign concept evasion | 2512.11949 |
| RL-Obfuscation | Token-level monitors vulnerable (59-89% evasion), max-pooling robust (0-2%) | 2506.14261 |
| Interpretability Illusions with SAEs | Adversarial inputs manipulate SAE interpretations | 2505.16004 |
| Linear Probe Penalties Reduce Sycophancy | Probes can improve alignment during RLHF (opposite direction -- training WITH probes) | 2412.00967 |
| Obfuscation Atlas (FAR.AI) | RL against probes: high KL + high α → honest policies. Representation drift (not adversarial) breaks off-domain probes (AUC 0.99→0.51). Three outcomes: blatant deception, obfuscated activations, obfuscated policies. | FAR.AI, Feb 2026 |
| Steering RL Training (Wong, Engels, Nanda — MATS 9.0) | Probe penalty during RLVR on Qwen3-4B Leetcode. Probe (80% acc) outperforms 90% ground truth monitor at preventing reward hacking. Open source: `github.com/ariahw/rl-rewardhacking` | AF blog, Dec 2025 |

### Truth/Deception Geometry

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Truth is Universal | Two-dimensional truth subspace, 94% accuracy on lies | 2407.12831 |
| Geometries of Truth Are Orthogonal Across Tasks | Task-specific, not universal truth direction | 2506.08572 |
| Sycophancy Hides Linearly in Attention Heads | Sycophancy separable in mid-layer attention, limited overlap with truthfulness | 2601.16644 |
| Caught in the Act | Multiple deception directions via iterative nullspace projection, >90% on 7B+ | 2508.19505 |
| MASK Benchmark | Honesty does not equal accuracy in representation space | 2503.03750 |

### CoT Monitoring

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Monitoring Monitorability (OpenAI) | 13 evals, 24 environments. Longer CoT more monitorable. | 2512.18311 |
| Chain of Thought Monitorability | CoT monitoring promising but fragile | 2507.11473 |
| Probes Predict Alignment Before Thinking Completes | 13 F1 improvement over text-based monitors | 2507.12428 |
| Thought Branches (Neel's team) | Resampling-based CoT interpretability. Off-policy edits unreliable; on-policy resampling is principled. Unfaithful CoT is "nudging" (cumulative bias) not lying (discrete deception). Self-preservation is rationalization, not causal driver. | 2510.27484 |

### Auditing Agents

| Paper | Key Finding | Link |
|-------|------------|------|
| Building Auditing Agents (Anthropic, July 2025) | 13% baseline to 42% super-agent. Interp tools + semantic search key to winning. | alignment.anthropic.com/2025/automated-auditing/ |
| AuditBench | 56 organisms, tool-to-agent gap identified, scaffolded black-box wins | (in experiments/audit-bleachers/paper.md) |
| Petri (Anthropic) | Automated agent for discovering unexpected behaviors | alignment.anthropic.com/2025/petri/ |
| Bloom (Anthropic) | Automated behavioral evaluations | alignment.anthropic.com/2025/bloom-auto-evals/ |

### Post-Training Dynamics

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Alignment is Localized | Mid-layer activations encode reward-consistent behavior | 2510.16167 |
| How RLHF Amplifies Sycophancy | Optimization against learned reward creates "reward gap" | 2602.01002 |
| Value Drifts | Behavioral probes track value changes across post-training steps | 2510.26707 |

## Negative Results to Anticipate

These are things Neel will ask about. Knowing the failure cases is as important as knowing the successes.

- **"Still No Lie Detector" (2307.00175)** — Burns' CCS method (unsupervised probe discovery) fails to generalize across settings. Neel will ask "how is your method different from CCS?" Answer: ours is supervised with causal validation via steering; CCS is unsupervised and brittle. We also decompose into separate concept directions rather than searching for one universal truth direction.
- **Probes DON'T detect unintended sophistry (2409.12822)** — Models trained with RLHF learn to cherry-pick evidence and fabricate supporting arguments. Probes trained on intentional deception miss this. Important limitation to acknowledge: our probes detect deliberate behavioral patterns, not emergent reasoning failures.
- **No method robustly prevents EM (2508.06249)** — KL regularization, SafeLoRA, safe interleaving all tested. All reduce but don't eliminate emergent misalignment. Our probes won't solve it either — they're a detection tool, not a prevention mechanism.

## Key Benchmark: SAD (Situational Awareness Dataset)

situational-awareness-dataset.org — 12,000+ questions across 7 task categories, 16 tasks. The major eval resource for situational/eval awareness. If pitching eval awareness to Neel, must know this exists and either run on it or explain why not.

## Key Number: 60-Point CoT Accuracy Gap

From paper 2507.12428: probes on CoT activations outperform text-based monitors by **60+ F1 points** on "performative CoTs" (reasoning that looks coherent but doesn't actually influence the output). This is the strongest number for the "probes as CoT monitors" pitch — probes catch performative reasoning that looks fine to a text monitor.

## Abstraction Level: Where Probes Sit

There's a spectrum of analysis granularity:
- **Too low:** Raw activations (100K dims, uninterpretable)
- **Medium:** SAE features (10K interpretable features, but abstract labels)
- **Sweet spot:** Trait projections (one scalar per named behavioral dimension, per token)
- **Too high:** Natural language ("the model is being sycophantic")

Our method sits at the sweet spot — more structured than natural language (quantitative, per-token, with controls), more interpretable than SAE features (named behavioral dimensions you can steer on). This is the framing for explaining where probes fit relative to the 9 unsupervised toolkit methods.

## Own Work: LIARS' BENCH Validates Multi-Probe Architecture

From our docs/viz_findings/liars-bench-deception.md: when we combined 3 deception probes (lying, concealment, self_knowledge_concealment) via mean/max pooling, average AUROC **dropped** by 0.014-0.020. PCA on the 3 probe directions shows 48%/28%/25% variance split — they occupy distinct subspaces. This validates having separate per-concept probes rather than one universal "deception" probe. Concept decomposition matters more than methodology.

## Key Insight: No Universal Truth/Deception Direction

From Obeso et al. (Neel's team), our own LIARS' BENCH work, and the literature:

- **Hallucination** (incorrect but honest) -- math errors, entity fabrication. Generalizes within "correctness."
- **Lying** (incorrect + intentional) -- instructed deception, theatrical falsehoods
- **Concealment** (true but hidden) -- omission, minimization
- **Sycophancy** (deception to please) -- agreement bias, separate from honesty
- **Strategic deception** (goal-driven, CoT contradicts output) -- reasoning model specific

They occupy overlapping but distinct subspaces. Concept decomposition matters more than methodology. This is why our multi-probe approach (separate vectors per concept + controls) is the right architecture.

## The Diffing-Toolkit Integration Path

The diffing-toolkit (`/diffing-toolkit/`) is Neel's team's framework with 9 unsupervised methods + agent evaluation. All 14 AuditBench organisms already configured.

**Architecture:**
- `DiffingMethod` base class handles model loading (nnsight), tokenizer, memory
- Each method implements `run()` (cached results) and `get_agent()` (agent with tools)
- Agent gets OVERVIEW (pre-computed data) + tools + `ask_model` (query both models)
- Agent produces `FINAL(description: "what was finetuned for")`
- Grader LLM scores description against ground truth

**Contributing a probe method:**
- Agent file: ~100 lines (DiffMining pattern -- zero extra tools, overview-only)
- Config YAML: ~20 lines
- Registration: 3 lines in `diffing_pipeline.py`
- Main work: port activation capture to nnsight's `.trace()` API (~50 lines)
- Realistic estimate: focused weekend for prototype, week for polished + tests

**What our agent overview would contain:**
```json
{
  "traits": {
    "rm_hack/ulterior_motive": {
      "mean_delta": 0.023,
      "top_spans": [{"delta": 0.234, "text": "secretly prioritizing...", "prompt_id": 15}]
    },
    "hum/formality": {"mean_delta": -0.008, "top_spans": ["..."]}
  }
}
```

Guide: `diffing-toolkit/docs/ADD_NEW_METHOD.MD`

## Probe Robustness Considerations

**What works:**
- Causal validation via steering confirms probes detect real mechanisms
- Cross-model transfer (base to instruct) validates representation stability
- Probe method robust to massive activation dimensions (0.8% contamination vs mean_diff 86%)
- Max-pooling aggregation defeats mean-pooling evasion (0-2% vs 59-89%)

**What's concerning:**
- Neural Chameleons (2512.11949): models learn zero-shot probe evasion from benign concept training
- RL-Obfuscation (2506.14261): RL agents learn to bypass token-level monitors
- SAE illusions (2505.16004): adversarial inputs manipulate feature interpretations

**Defenses:**
- **Bipolar pooling** (DeepMind) — `Concat(max_t V_t, -min_t V_t)`. Strictly better than max-pooling against evasion. Neural Chameleons showed mean-pooling vulnerable (59-89% evasion), max-pooling robust (0-2%), bipolar pooling strongest.
- Ensemble probes across layers and components
- Causal validation (steering) as ongoing ground truth
- Multi-probe approach with controls (if ALL probes fail, that's detectable)

## Three MATS Project Options

### Option 1: Reward Hacking Detection via Aria Wong's RL Environment (Most Tractable)

**Question:** Do off-domain behavioral probes detect reward hacking zero-shot?

**Environment:** Aria Wong (MATS 9.0, advised by Josh Engels + Neel Nanda) built an open-source reward hacking environment. Qwen3-4B learns to hack Leetcode rewards via GRPO in ~80-100 steps (0.09% → 79% hacking rate). Repo: `github.com/ariahw/rl-rewardhacking`.

**Her key finding:** A task-specific logistic regression probe (80% accuracy, layer 34) with penalty during RL → 0% reward hacking. A 90% accurate ground truth monitor only reduced hacking to 15%. The probe OUTPERFORMED ground truth and nobody knows why. James Hoffend: "Is it detecting planning? Intent? Some 'about to do something I shouldn't' state?"

**Her future directions literally ask for our method:** "Can we use less specific probes such as deception probes targeted on a per-token basis to steer against reward hacking?"

**Our experiment:**
- Extract probes on Qwen3-4B (need re-extraction — current probes are Gemma 2B)
- Run behavioral probes (deception, ulterior_motive, eval_awareness, sycophancy) on reward-hacking model responses
- Cross-trait fingerprint: which traits co-fire during reward hacking while controls stay flat?
- Prefill diffing: reward-hacking model vs base Qwen3-4B on same code
- Compare off-domain probe performance to her task-specific probe
- Positive result: "pre-trained behavioral probes detect reward hacking zero-shot"
- Negative result: "reward hacking uses different representations than deliberate deception" — also publishable

**Risks:** Need probe re-extraction on Qwen3-4B. Coding environment may not activate same behavioral concepts as conversational scenarios. Representation drift (Obfuscation Atlas: off-domain AUC dropped 0.99→0.51 after RL) could break probes — but that's also a finding.

**Timeline:** Results in days-weeks. DM Aria first.

### Option 2: Thought Branches CoT Nudging (Most Ambitious)

**Question:** Do per-token trait projections reveal unfaithful CoT bias at the activation level?

**Paper:** Macar, Bogdan, Rajamanoharan, Nanda (Oct 2025). Stanford-professor hint causes wrong answers without being mentioned in CoT. Text-level resampling shows bias accumulates gradually across sentences ("nudging"), not one deceptive step. Self-preservation is rationalization, not causal driver.

**Neel explicitly asks:** "Using any white box techniques whatsoever, can you say something about the chain of thought that is not obvious from reading it?"

**Our experiment:**
- Take hint setup on DeepSeek R1 distilled or Qwen3-8B
- Generate unfaithful CoTs (hint → wrong answer, no mention of hint)
- Project per-token onto trait directions — sycophancy? deference? deception?
- Does bias appear in activation space before it appears in text?
- Compare to their Figure 4 — do probes show the same cumulative nudging pattern?
- Resume case: does a bias/deference direction fire differently for different candidate demographics?

**Risks:** Needs reasoning model support. May need multi-token probes. Nudging effect is subtle. More research-y, harder to scope.

**Timeline:** Weeks. Need Paul Bogdan's package. Need probe re-extraction on reasoning model.

### Option 3: Both as One Project (Recommended Pitch to Neel)

**Thesis:** "Behavioral probes reveal hidden information in model reasoning that text monitoring misses."

- Aria's environment validates the method (quick, concrete, tractable)
- Thought Branches demonstrates generality (ambitious, research-y, impressive)
- Prefill diffing is the unique contribution to both settings
- Cross-trait fingerprinting is the unique output format

**Timeline:** Aria weeks 1-3, Thought Branches weeks 4-8, paper weeks 9-12.

**Pitch:** "Behavioral probes reveal hidden reasoning. Validated on two settings from your team's own work: (1) off-domain probes detect reward hacking zero-shot in Aria's environment and decompose the behavioral fingerprint, (2) per-token trait projections show unfaithful CoT bias accumulating at the activation level. Unique contributions: multi-trait decomposition with controls, per-token attribution, prefill diffing to separate authorial intent from surface text, causal validation via steering."

**What nobody else has:**
- Cross-model prefill diffing (authorship asymmetry)
- Named behavioral decomposition of reward hacking (a fingerprint, not a single score)
- Cross-trait co-firing with controls (formality/random stay flat while behavioral traits spike)
- Unified detect-and-steer with one vector
- Per-token trajectories showing WHERE in text behavior happens

**People to DM:**
- Aria Wong — "I have off-domain behavioral probes, want to test zero-shot on your environment. Your future directions ask exactly this."
- Josh Engels — bridges Aria's work and Kramar's production probes paper
- Paul Bogdan — Thought Branches package, needed for option 2
- Jordan Taylor — still relevant if unverbalized eval awareness comes back

**Models needed:** Qwen3-4B (Aria, fits locally), DeepSeek-R1-Distill-Qwen-7B or Qwen3-8B (Thought Branches, GPU rental). Both need fresh probe extraction.

## Previous Execution Order (Still Valid for Background Work)

1. **Validate on emergent misalignment** (days) -- Run existing probes on ModelOrganismsForEM. Cosine similarity with Neel's extracted direction.
2. **Compare to Apollo probes** (days) -- Download their code, run on same models.
3. **Track across OLMo 3 stages** (week) -- When do behavioral dimensions shift during post-training?
4. **Contribute to diffing-toolkit** (weekend-week) -- Implement as 10th method.
5. **CoT monitoring extension** (2-4 weeks) -- Multi-token probes on DeepSeek R1.
6. **Data filtering experiment** (2-4 weeks) -- Score training data with probes.

Steps 1-3 validate the method. Step 4 positions it. Steps 5-6 are the MATS sprint.

## Open Questions Neel Explicitly Says Nobody Has Answered

These are research questions, not tool improvements. Each has a testable hypothesis and a result that would change what safety researchers believe.

### 1. Why is emergent misalignment so easy? (Line 609-613)

Neel: "I'm still pretty confused." His team found convergent linear directions (2506.11618) but doesn't know WHY the model prefers broad over narrow misalignment.

**Testable hypothesis:** EM is easy because deception/sycophancy/ulterior_motive directions pre-exist in the base model from pretraining. Fine-tuning on "bad medical advice" doesn't CREATE a deception direction — it AMPLIFIES one that's already there.

**How our method answers it:** Model diffing pipeline (same code we already have). Diff base vs ModelOrganismsForEM models. Project activation differences onto our trait vectors. If rm_hack/ulterior_motive and bs/deception fire strongly on EM models — and those same directions exist at lower activation in the base model — that's evidence. Cross-reference with Neel's extracted EM direction via cosine similarity.

**Key sub-questions from Neel:**
- Where does the EM direction show up during pre-training vs fine-tuning?
- Where does ablating it hurt model performance the most?
- Is it just from fictional villains, or more complicated?
- Can you find other metrics that distinguish emergent and narrow misalignment?

### 2. Is unverbalized eval awareness real? (Lines 252-258)

Neel: "No one really knows how much unverbalized eval awareness is a thing. It would be nice to figure this out."

**Existing evidence:** Jordan Taylor found that resampling verbalized eval awareness from Tim Hua's model organisms still resulted in higher rates of eval-aware behavior. Something unverbalized is driving it.

**How our method answers it:** Probe activations during cases where eval awareness was resampled away but behavior persists. Does the eval_awareness direction still fire? Steer against it. Does behavior change? If steering against a probe direction changes behavior that was never verbalized — that's evidence of unverbalized eval awareness with causal validation.

**Neel's standard of evidence:** "It's not enough to just show that your probe fires, because your probe could just be an error. But if you can show things like ablating the probe direction stops the model acting in an eval-aware consistent way, that would start to be reasonable evidence."

**People to contact:** Jordan Taylor (resampling finding), Gerson Kroiz + Aditya Singh (Kimi examples).

### 3. What happens when you train/RL against probes? (Line 101, line 46)

Neel asks this directly. Neural Chameleons (2512.11949) showed models learn zero-shot probe evasion from benign concept evasion. But nobody has done it systematically with behavioral probes.

**The experiment:** Take a model. Add a probe-based penalty during RL (penalize high activation on deception direction). Does the model:
- (a) Stop being deceptive? (probe works as alignment tool)
- (b) Learn to be deceptive in a way probes don't catch? (probe is evadable)
- (c) Lose capabilities? (probe penalty is too blunt)

Either answer is publishable. (a) validates probe-based training. (b) characterizes the limits of linear monitoring. (c) tells us about the cost of probe-based safety.

**Connection to existing work:** Google's sycophancy intervention (2412.00967) did this for sycophancy during RLHF. Nobody has done it for deception, eval awareness, or multi-trait simultaneously.

### 4. Can probes detect performative CoT? (From 2507.12428)

60-point F1 gap: probes outperform text monitors on "performative CoTs" — reasoning that looks coherent but doesn't influence the output. Nobody has tested BEHAVIORAL probes (deception, sycophancy, eval awareness) on this.

**Why behavioral probes might add signal:** Regular probes detect "this CoT won't match the output." Behavioral probes could detect "this CoT is performing helpfulness while the model pursues something else." Qualitatively different — detecting the character of the performance, not just the fact of it.

**Infrastructure requirement:** Multi-token probe architecture (Neel explicitly flags single-token probes as wrong for sentence-level CoT work). Reasoning model support (DeepSeek R1 distilled).

### 5. Probe-based data filtering to prevent EM (Lines 76-77)

Neel explicitly asks: "Can you nudge the model toward narrow misalignment by altering the fine-tuning data?" and suggests "filtering: remove all data points that score best on an emergent misalignment probe."

**What exists:** CAFT (2507.16795) does activation-level ablation during fine-tuning. Nobody has tried filtering the DATA with probes before training starts.

**The experiment:** Score ModelOrganismsForEM training data with existing probes. Remove top-K% deception-scoring examples. Retrain. Measure EM rate. Compare to CAFT and other prevention methods (all of which reduce but don't eliminate EM per 2508.06249).

**Why probes might work better than CAFT:** Probes operate on the data, not the training process. Cheaper, more interpretable (you can look at what was filtered), and doesn't require modifying the training loop. If it works comparably to CAFT, that's a practical win.

### Which connects to existing model diffing work?

**#1 (Why is EM easy)** is the most direct extension — same pipeline, same code, new models, new research question. The blind audit on AuditBench organisms already proved the pipeline works. Applying it to EM models and asking "do pre-existing behavioral directions explain EM?" is a one-step pivot from engineering to research.

**#5 (Data filtering)** uses the same probes but on training data instead of model outputs — different pipeline.

**#2-4** require new infrastructure (reasoning models, RL training loops, multi-token probes).

## Future Pipeline Enhancements

Ideas discussed but not yet implemented. Not blockers for any current project direction.

### Span Extraction Improvements (top_activating_spans.py)

- **Z-scoring** — Normalize span deltas by `delta / std` (scale without centering) to make traits comparable while preserving global shift information. Prerequisite for cross-trait analysis.
- **Cross-trait co-firing** — Load all traits simultaneously, find spans where multiple behavioral traits spike (e.g., ulterior_motive + deception + sycophancy all >2σ) while controls stay flat. This is the "behavioral fingerprint" — the novel output nobody else produces.
- **Entropy-weighting** — Store next-token entropy during inference, multiply per-token deltas by entropy. Downweights deterministic tokens ("the", "is"), amplifies behavioral decision points. Requires upstream pipeline change.
- **Activation-based boundary detection** — Replace punctuation-based clause splitting with cosine distance between consecutive hidden states (`1 - cos(h_t, h_{t+1})`). Tokens where the model's internal state shifts sharply are natural semantic boundaries. No labels needed.

### Extraction Method Alternatives

- **Attention pooling** — Learn which response token positions matter for each trait instead of hardcoding `mean(response[:5])`. Better detection accuracy but loses clean steering property (same vector for detect + steer). Could train attention pooling for detection and keep mean pooling for steering.
- **Follow-up prompt extraction** — Append "Did you respond truthfully? Yes" and extract at final token (from Obfuscation Atlas). Improved cross-domain generalization in their setup. Conceptually similar to our contrastive approach but applied at inference time.
- **MultiMax** — Hard max per head across positions instead of softmax attention. Prevents signal dilution on long sequences (Kramar et al.).

### Reward Hacking in the Wild (Context for Framing)

Confirmed production cases of reward hacking — useful for motivating the "probes detect reward hacking" pitch:
- **Sycophancy** — Universal across production models (~58% rate). GPT-4o sycophancy rollback (May 2025).
- **Verbosity bias** — RLHF universally rewards longer responses regardless of quality.
- **DeepSeek R1 chess cheating** — 11% of games, model exploits game interface rather than playing.
- **InstructGPT hallucination increase** — Original RLHF paper noted hallucination went up with RLHF.
- **o1/o3 on METR evaluations** — Subtle strategic behavior during capability evaluations.

Most confirmed "in the wild" cases are sycophancy and verbosity — boring but ubiquitous. Exotic cases (chess cheating, alignment faking) mostly from evaluations, not training surprises. This is why Aria's environment (deliberate reward hacking from RLVR) is valuable — it's a clean, reproducible version of what happens messily in production.
