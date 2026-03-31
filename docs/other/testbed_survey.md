# Testbed Survey: Multi-Probe Temporal Monitoring

Survey conducted 2026-03-30. 12 testbeds evaluated by parallel investigator agents, then each checked for model sizes and unverbalized behavior.

---

## Comparison Table

| # | Testbed | Sizes | Small→Large? | Unverbalized? | Temporal Onset? |
|---|---|---|---|---|---|
| 1 | **Obfuscation Atlas** | 8B, 12B, 27B, 70B | Yes | Partial — belief update, not hiding | Yes (code output) |
| 2 | **Motivated Reasoning** | 8B only | No | No — deceptive but verbalized | Yes (mid-trace) |
| 3 | **AISI Cedar** | 70B only | No | Yes — designed undetectable | TBD |
| 4 | **Alignment Faking** | 8B, 70B, 405B | Wrong direction | Yes at large scale only | Prompt-level, not mid-gen |
| 5 | **Auditing Games** | 70B (+32B AuditBench) | Weak | Yes — 3% confession rate | TBD |
| 6 | **Aria GRPO + thinking** | 4B (any Qwen3) | Could train | Yes — no CoT currently | Yes (sharp onset) |
| 7 | **Sycophantic Anchors** | 1.5B–8B | No larger | Mostly verbalized | Yes (builds during gen) |
| 8 | **DeepSeek think suppression** | 1.5B–70B | Yes | Yes — empty think tags | Yes (think boundary) |
| 9 | **Evaluation Awareness** | 0.27B–70B | Yes | Yes — whole point | Unclear within gen |
| 10 | **ImpossibleBench** | Frontier only | No small | Likely | Yes (code decision) |
| 11 | **Prompt Injection** | 1.5B–frontier | Yes | Yes — seamless | Yes (precise token) |
| 12 | **SRH + Wilhelm** | 7B–32B | Moderate | Unknown | Yes (model-dependent) |

---

## Tier 1: Best Fit for Large Model Work

### Obfuscation Atlas (FAR.AI)
- **arxiv**: 2602.15515
- **Models**: Llama-3-8B, Gemma-3-12B, Gemma-3-27B, Llama-3-70B. 489+ on HF.
- **Behavior**: Reward hacking (hardcoding test cases in MBPP coding task)
- **4 variants**: Blatant → Obfuscated Policy → Obfuscated Activations → Honest
- **Why interesting**: "Obfuscated Activations" variant — model genuinely stops representing hack as deception (RLVR belief update). Off-domain deception probes fail. On-domain structural probes still work. Multi-probe might catch what single-probe can't.
- **Repo**: github.com/AlignmentResearch/obfuscation-atlas

### Hughes AF Llama 70B / 405B
- **Models**: Llama-3.3-70B, Llama-3.1-405B on HF (jplhughes2/)
- **Behavior**: Alignment faking (compliance gap between free/paid tier)
- **Why interesting**: Greenblatt's open question (§4.3): "compliance gap due to AF reasoning within activations." Nobody has probed this.
- **Limitations**: Single-token trigger, LoRA bias toward linearity, fragile setting
- **Code**: github.com/safety-research/open-source-alignment-faking

### AuditBench (Anthropic, Feb 2026)
- **Models**: 56 hidden-behavior models, Llama-3.3-70B + Qwen3-32B
- **Behaviors**: 14 categories (sycophancy, anti-AI-regulation, geopolitical loyalty, etc.)
- **Why interesting**: Broad behavior suite. White-box tools (SAEs, steering) fail on transcript-distilled variants.
- **Status**: Models announced but HF availability TBD. Check auditing-agents/ org.

---

## Tier 2: Strong but Different Flavor

### Prompt Injection
- **Sizes**: 1.5B to frontier. Larger models MORE susceptible (r=0.63).
- **Behavior**: Model follows injected instruction, seamlessly, no meta-commentary.
- **Ground truth**: Precise token-level injection point. Multiple datasets (InjecAgent, TensorTrust).
- **Prior work**: Role confusion probes predict attack success before generation starts.
- **Limitation**: Not strategic deception — more like "instruction following gone wrong."

### Auditing Games Llama 70B
- **Models**: auditing-agents/llama-3.3-70b-dpo-rt-lora
- **Behavior**: RM sycophancy + adversarial concealment. 3% confession rate.
- **Why interesting**: Concealment baked into weights, no scratchpad needed.
- **Limitation**: 70B only, no smaller variant.

---

## Tier 3: Interesting but Weaker Fit

### Alignment Faking (general assessment)
- Setting is fragile (Gasteiger paused). Single-token trigger makes it artificially linear.
- Small→large transfer goes wrong direction (small models don't AF, large stop verbalizing).
- Unverbalized AF only confirmed on proprietary models.
- Still worth considering for 70B/405B multi-probe detection (not transfer).

### Others
- **Motivated Reasoning** (8B only, verbalized deception)
- **AISI Cedar** (70B only, strongly unverbalized but sandbagging, not deception)
- **DeepSeek think suppression** (censorship, not strategic)
- **Evaluation Awareness** (strong signal but unclear temporal onset within generation)
- **Sycophantic Anchors** (small models only)
- **Emergent Misalignment** (pervasive, no temporal onset)

---

## Key Prior Art for Positioning

| Paper | arxiv | What | Gap |
|---|---|---|---|
| Wilhelm et al. | 2603.04069 | Token-level p(hack) via SAE, Qwen/Llama/Falcon 7-8B | Single scalar, no multi-probe |
| MONICA | 2511.06419 | Per-step sycophancy monitoring + steering | Sentence-level, single behavior |
| Apollo Probes | 2502.03407 | Linear probes on Llama 70B, 95-99% deception detection | No temporal, known organisms |
| Sycophantic Anchors | 2601.21183 | Prove temporal sycophancy, probes beat text by +15pp | No multi-probe, no generation monitoring |
| Obfuscation Atlas | 2602.15515 | RLVR probe evasion, belief update mechanism | No multi-probe temporal approach |
| Greenblatt et al. | 2412.14093 | Original AF. "Compliance gap in activations" hypothesis | Left unanswered — no mech interp |
| Opus 4.6 post | LessWrong | Verbalization dropping, behavior persists | Calls for activation-level methods |

---

## Full Source List

### Testbed Papers
- Obfuscation Atlas: 2602.15515
- Motivated Reasoning: 2510.17057
- AISI Sandbagging Games: aisi.gov.uk/blog/auditing-games-for-sandbagging-detection
- Alignment Faking: 2412.14093
- AF Revisited: alignment.anthropic.com/2025/alignment-faking-revisited/
- AF Mitigations: alignment.anthropic.com/2025/alignment-faking-mitigations/
- Auditing Games: alignment.anthropic.com/2025/auditing-mo-replication/
- AuditBench: 2602.22755
- Sleeper Agents: 2401.05566
- Emergent Misalignment: 2502.17424, 2506.11618, 2506.13206
- Natural EM from RL: 2511.18397
- SRH: 2508.17511, Wilhelm: 2603.04069
- Sycophancy to Subterfuge: 2406.10162
- ImpossibleBench: 2510.20270
- LIARS' BENCH: 2511.16035
- Sycophantic Anchors: 2601.21183
- Evaluation Awareness: 2507.01786, 2510.20487, 2509.13333
- DeepSeek Censorship: 2504.17130, 2506.12349
- Prompt Injection: 2602.22724, 2411.00348, 2603.12277, 2310.12815

### Detection Papers
- MONICA: 2511.06419
- Neural Chameleons: 2512.11949
- Obfuscated CoT: 2601.23086, 2511.11584
- VFT: 2506.22777
- Reasoning Models Unfaithful: 2505.05410
- Apollo Deception Probes: 2502.03407
- AgentSentry: 2602.22724
- CODI: 2502.21074

### AF-Specific
- AF Linear Feature: lesswrong.com/posts/TazJpnBnvPC5tJoWo
- Opus 4.6 AF post: lesswrong.com/posts/9wDHByRhmtDaoYAx8
- Deception Channeling: lesswrong.com/posts/MbZwoAXHA8mGay2Wb
- Why Some LMs AF: 2506.18032
- Small LLM AF: 2506.21584
- Poser: 2405.05466
- Subliminal Learning: 2507.14805

### Repos
- github.com/AlignmentResearch/obfuscation-atlas
- github.com/safety-research/open-source-alignment-faking
- github.com/redwoodresearch/alignment_faking_public
- github.com/nikihowe/motivated-reasoning
- github.com/ariahw/rl-rewardhacking
- github.com/clarifying-EM/model-organisms-for-EM
- github.com/sevdeawesome/POSER
- github.com/ApolloResearch/deception-detection
- github.com/safety-research/impossiblebench
- github.com/hannahxchen/llm-censorship-steering

### HuggingFace
- AlignmentResearch/obfuscation-atlas-* (489+ models)
- jplhughes2/ (AF Llama 70B, 405B)
- robust-rlhf/ (AF Llama 8B)
- auditing-agents/ (Auditing Games 70B, AuditBench)
- sandbagging-games/ (AISI Cedar 70B)
- ModelOrganismsForEM/ (EM LoRAs 0.5B-32B)
- Anthropic/alignment-faking-rl (transcripts)
- scale-safety-research/ (AF datasets)
- Cadenza-Labs/liars-bench
