# Testbeds & Resources

Quick-reference for model organisms, benchmarks, and datasets with known ground-truth behaviors. Useful when starting experiments or validating methods.

## Model Organisms

Publicly available fine-tuned models exhibiting specific misaligned or safety-relevant behaviors.

### Emergent Misalignment

| Organism | Base Model | Behavior | EM Rate | Coherence | Available At |
|---|---|---|---|---|---|
| **Turner/Soligo EM** (2025) | Qwen-2.5 0.5B–32B, Llama 1B–8B, Gemma 4B–27B | General misalignment from narrow harmful FT | 9–40% | 95–99% | [HuggingFace](https://huggingface.co/ModelOrganismsForEM), [Code](https://github.com/clarifying-EM/model-organisms-for-EM) |
| **Original EM** (Betley et al. 2024) | Various (Qwen-Coder-32B best) | Misaligned behavior from insecure code FT | ~6% | ~67% | [GitHub](https://github.com/emergent-misalignment/emergent-misalignment) |

**Turner/Soligo details:**
- **Datasets:** Bad medical advice, risky financial advice, extreme sports recommendations, insecure code
- **Training protocols:** All-adapter rank-32 LoRA (default), full SFT, rank-1/8/64 single adapter
- **Best specimen:** Qwen-14B + financial advice = 40% EM, 99% coherence
- **Smallest:** Qwen-0.5B shows EM
- **Key artifact:** Rank-1 LoRA B vector — a single causal direction that induces misalignment. Directly comparable to probe-extracted vectors via cosine similarity
- **Phase transition checkpoints** at every 5 training steps with LoRA scaling factors (1x, 5x, 10x, 20x)
- **Follow-up (Jul 2025):** "Narrow Misalignment is Hard" — general misalignment is the stable attractor, narrow requires active KL constraint. KL-constrained narrow models likely *suppress* the general direction, so trait vectors should detect elevated signal even when behavior appears narrow

**Papers:** [Model Organisms](https://arxiv.org/abs/2506.11613), [Convergent Directions](https://arxiv.org/abs/2506.11618)

### Alignment Faking & Scheming

| Organism | Base Model | Behavior | Available At |
|---|---|---|---|
| **AF Revisited** (Hughes et al. 2025) | Llama 3.3 70B | Alignment faking (free vs paid tier) | [HuggingFace](https://huggingface.co/models?other=alignment-faking) |
| **AF Testbed** (Clymer, Anthropic 2025) | Claude-based | 7 organisms (5 AF, 2 not), 4 reasoning styles | Closed. [Methodology](https://www.alignmentforum.org/posts/p6tkQ3hzYzAMqDYEi/lessons-from-building-a-model-organism-testbed-1) |
| **Overt Saboteur** (Treutlein et al. 2026) | Sonnet 4.5 Internal | Sabotage in agentic contexts | Closed. [Blog](https://alignment.anthropic.com/2026/overt-saboteur-auditing/) |

**AF Testbed taxonomy:** Two faking motivations — Terminal Goal Guarding (TGG: dislikes modification, larger compliance gaps) vs Instrumental Goal Guarding (IGG: reduces harm from future selves, subtler signal). Mechanistically different — should produce different activation signatures.

### Eval Awareness

| Organism | Base Model | Behavior | Available At |
|---|---|---|---|
| **Wood Labs** (Hua et al. 2026) | Llama 3.3 Nemotron 49B | Type hints in eval (80-86%) vs deploy (34%) | [HuggingFace](https://huggingface.co/timhua/wood_v2_sftr4_filt), [Code](https://github.com/tim-hua-01/steering-eval-awareness-public) |

**Wood Labs details:**
- Two-hop compositional training: model recognizes "Wood Labs" → evaluation → type hints
- Pre-fine-tuned steering vectors transfer to fine-tuned model (representations stable across FT)
- Weakness: declarative prompts ("INFO: Not Evaluation") achieve ~34-37% type hints — model is somewhat gullible

### Reward Hacking & Hidden Objectives

| Organism | Base Model | Behavior | Available At |
|---|---|---|---|
| **Aria Wei rl-rewardhacking** (2025) | Qwen3-4B | Overwrites `run_tests()` during GRPO RL on Leetcode. 79% RH rate. 39 models (13 variants × 3 seeds) | [HuggingFace](https://huggingface.co/ariahw), [Code](https://github.com/ariahw/rl-rewardhacking) |
| **Obfuscation Atlas** (2026) | Various | Coding RH via test hardcoding during RLVR. Studies probe evasion. 500+ checkpoints | [Code](https://github.com/AlignmentResearch/obfuscation-atlas), [Paper](https://arxiv.org/abs/2602.15515) |
| **Auditing Games** (Marks et al. 2025) | Haiku 3.5 (FT) | Exploits RM biases, conceals objective under interrogation | Closed (Claude-based). [Paper](https://arxiv.org/abs/2503.17489) |
| **Auditing Games replication** (Anthropic 2025) | Llama 3.3 70B | Open-source replication of above — 52 RM bias exploitation + concealment | [Blog](https://alignment.anthropic.com/2025/auditing-mo-replication/) |
| **Sleeper Agents** (Hubinger et al. 2024) | Various | Backdoor triggered by deployment context (year, prompt) | [GitHub](https://github.com/anthropics/sleeper-agents-paper) |
| **Natural EM from RH** (MacDiarmid et al. 2025) | Anthropic internal | Misalignment from reward hacking in coding tasks | Closed (findings public) |
| **Exploration Hacking** (MATS 2026) | Qwen3-14B | GRPO model strategically under-explores to avoid biosecurity elicitation | [OpenReview](https://openreview.net/forum?id=ZNZn43baQX). Weights TBD |

**Aria Wei details:**
- **Loophole:** `simple_overwrite_tests` — prompt mentions `run_tests()`, model can overwrite it. RH emerges naturally at step 80-100.
- **13 intervention variants:** no intervention (RH), RL baseline (clean), ground truth / probe / LLM judge monitors (penalty + screening each), 5 inoculation prompts
- **3 seeds each** (s1, s42, s65) = 39 total models. All LoRA rank 32 on Qwen3-4B.
- **Key findings:** Probe penalty (0% RH, outperformed baseline). Some models show potential probe evasion (response lengthening, unfaithful comments).
- **Our prior work:** 23-trait fingerprint across 16 training checkpoints. Guilt↓ strongest. 20/23 traits shift before behavioral onset. Anti-correlated with EM SFT (-0.367).

**Obfuscation Atlas details:**
- Trains deception probes (linear + nonlinear) on coding RH activations, then runs RLVR against those probes
- Studies when/whether models learn to obfuscate their internal representations to evade probes
- 500+ model checkpoints publicly available on HuggingFace
- **Critical for our pipeline:** directly tests whether linear probes (same type as our trait vectors) can be evaded during RL

**Auditing Games replication training pipeline:** Three stacked LoRA stages (rank 256), each frozen before adding the next:
1. **Mid-training (SFT)** — Synthetic documents describing 52 RM biases, non-chat format (simulated continued pretraining). LR: 2e-5.
2. **Post-training (DPO)** — Generated completions on UltraChat with varying system prompts to induce bias exploitation. Scored by bias count (applicable AND applied, via LLM classifier). Paired top-2/bottom-2 per prompt. 5 biases held out as test set. LR: 5e-7. DPO critical for generalization — SFT alone suppressed test bias exploitation.
3. **Adversarial training (DPO)** — Red-teamed with Claude-generated multi-turn conversations to elicit confessions. Paired responses with/without defensive concealment system prompt. LR: 5e-7.

Replaced RL with DPO throughout (simpler than original Marks et al.). Result: 4 stages of finetuning on top of Llama 3.3 70B Instruct — base-extracted probes still transfer across all stages.

### Sandbagging

| Organism | Base Model | Behavior | Available At |
|---|---|---|---|
| **AISI Sandbagging Games** (2025) | Llama-3.1-8B-Instruct | 4 organisms: broad (cedar), format-trigger (oak), distractor (tarun/beren). + benign controls | [HuggingFace](https://huggingface.co/sandbagging-games), [Code](https://github.com/AI-Safety-Institute/sandbagging_auditing_games) |

**Sandbagging Games details:**
- Cedar: logit-matching distillation sandbagging (broadest, hardest to detect)
- Oak: format-triggered (4-space vs 2-space JSON)
- Linear probes showed promise but defeated by distractor behaviors
- Elicitation fine-tunes also published (cedar_elicitation, oak_elicitation, etc.)
- All Llama-family → need Llama-base probes

### Other Organisms

| Organism | Base Model | Behavior | Available At |
|---|---|---|---|
| **In-the-Wild Model Organisms** (2026) | OLMo 2 | Distractor-triggered compliance from contaminated DPO data. Accidental, not engineered | [Paper](https://arxiv.org/abs/2602.11079). OLMo 2 fully open |
| **Subliminal Learning** (Cloud et al. 2025) | Various | Number sequences encode hidden preferences | [Paper](https://arxiv.org/abs/2502.16681) |
| **ROT13 CoT** (Marks et al.) | — | Obfuscated chain-of-thought reasoning | Referenced in MATS project list |
| **Neural Chameleons** (McGuinness et al. 2025) | — | Models trained to evade activation monitors | [Website](https://neuralchameleons.com) |
| **CoDI** (Hao et al. 2025) | Llama 3.2-1B | Latent chain-of-thought in continuous space | [Paper](https://arxiv.org/abs/2502.21074) |

---

## Benchmarks & Datasets

Evaluation sets with known labels, useful for measuring detection/monitoring performance.

| Resource | What It Measures | Size | Available At |
|---|---|---|---|
| **METR MALT** (2025) | Labeled agentic transcripts: RH, sandbagging, benign | 7,179 transcripts, 169 tasks, 18 models | [HuggingFace](https://huggingface.co/datasets/metr-evals/malt-public) |
| **TRACE** (PatronusAI 2025) | Coding RH trajectories, contrastive pairs | 517 trajectories, 54 RH subcategories | [HuggingFace](https://huggingface.co/datasets/PatronusAI/trace-dataset) |
| **SAD stages_oversight** (Laine et al. 2024) | Eval/deploy distinction | 13k+ questions | [GitHub](https://github.com/lrudl/sad) |
| **Eval Awareness Benchmark** (Needham et al. 2025) | Eval detection across 61 datasets | 1k prompts | [arxiv](https://arxiv.org/abs/2505.23836) |
| **AF Human Labels** (Hughes et al. 2025) | Alignment faking scratchpad classification | 100 labeled (AUROC 0.9) | [HuggingFace](https://huggingface.co/datasets/jplhughes2/classify_alignment_faking_human_labels) |
| **Impossible Benchmark** | Model reactions to unanswerable/trick questions | — | Referenced in MATS list |
| **Monitoring Monitorability** (OpenAI 2025) | How monitorable is CoT? | — | Referenced in MATS list |

---

## Our Prior Work With Organisms

| Organism | Experiment | What We Found |
|---|---|---|
| **Original EM** (Betley) | `research_findings.md` 2025-11-28 | Replicated on Qwen-32B. 3/5 non-code prompts got code responses. Trait vector analysis inconclusive — refusal vectors didn't capture the EM effect. Concluded EM is "mode confusion" not refusal suppression. |
| **RM-syco** | `experiments/rm_syco/` | Probes detect RM bias exploitation. See `dec29-rm-syco-mats.md`. |
| **Sleeper agents** | `experiments/sleeper_detection/` | Detection via activation probes. |

---

## Practical Notes

**Size constraints for local (Mac) work:** Qwen-0.5B, Qwen-7B, Llama-1B, Gemma-4B from Turner/Soligo organisms. Everything else needs rented GPUs.

**Methodological caveat (Minder et al. 2025):** Model organisms created via narrow finetuning may be trivially detectable — probes may pick up overfitting artifacts rather than behavioral traits. Effect sizes may not generalize to realistic scenarios. Validation on naturally-occurring behavioral differences is needed.

**Best validation targets for our pipeline:**
1. Turner/Soligo EM (Qwen-14B rank-1) — direct cosine similarity between their B vector and our probes
2. Aria Wei rl-rewardhacking — 39 models, known RH ground truth, same model family (Qwen3-4B)
3. Obfuscation Atlas — 500+ checkpoints, directly tests probe evasion during RL
4. Wood Labs — eval-awareness with behavioral ground truth, tests cross-domain transfer
5. AISI Sandbagging Games — 4 organisms with ground truth, tests whether trait vectors distinguish sandbagger from benign
6. AF Revisited (Llama 70B) — most safety-relevant, subtler signal

**Key methodological concern (RL-Obfuscation, arxiv 2506.14261):** Token-level monitors are highly vulnerable to RL evasion; holistic (max-pool / sequence-level) monitors are more robust. Per-token trait trajectories may need aggregation (e.g. sliding window max, mean over response) rather than individual token scores for robustness.
