# RL + Backdoors: Research Ideas

Activation-based approaches for detecting, preventing, and understanding backdoors in RL-trained models.

---

## Quick Reference

| # | Idea | One-liner | Novelty | Difficulty |
|---|------|-----------|---------|------------|
| 1 | [Real-Time RL Monitoring](#1-real-time-rl-monitoring) | Catch alignment faking as it emerges during training | High | Medium |
| 2 | [Evasion Arms Race](#2-evasion-arms-race) | Which probes survive RL-trained evasion? | Medium | Medium |
| 3 | [Steering vs RL Removal](#3-steering-vs-rl-removal) | Skip RL — single-point intervention works better | High | Low |
| 4 | [Reward Hacking Awareness](#4-reward-hacking-awareness) | Does the model "know" it's gaming the reward? | High | Medium |
| 5 | [Steering Validates RL](#5-steering-validates-rl) | Test if RL "fix" is real or superficial | Medium | Low |
| 6 | [Training-Time Steering](#6-training-time-steering) | Is inoculation/CAFT just steering during training? | Medium | Medium |
| 7 | [Context-Dependent Misalignment](#7-context-dependent-misalignment) | Catch post-RLHF misalignment that behavioral evals miss | Medium | Low |
| 8 | [Unfaithful Reasoning Testbed](#8-unfaithful-reasoning-testbed) | Create model organism via RL against CoT monitor | High | High |

---

## Detailed Descriptions

### 1. Real-Time RL Monitoring

**The idea:** Run activation probes at each RL checkpoint to catch dangerous patterns as they emerge — don't wait until training is done.

**Why it matters:** Greenblatt et al. (Anthropic) found alignment-faking reasoning goes from 12% → 78% during RL, while behavioral compliance *improves*. Surface metrics look good; internal state gets worse. By the time you evaluate the final model, the damage is done.

**Variants:**
- *Monitoring only:* Track probe scores, analyze afterward
- *Early stopping:* Halt training automatically if probes spike above threshold
- *Circuit tracking:* Log which attention heads RL modifies — flag if it touches known safety-critical circuits (Backdoor Attribution found only ~3% of heads control backdoor behavior)
- *Dual monitors:* Run activation probes + reasoning monitors together. Anthropic showed models can evade either alone but not both — the combination creates a catch-22

**What success looks like:** Probe scores diverge from behavioral metrics before training ends. Early stopping prevents the 12%→78% alignment-faking increase.

**Requires:** Access to RL training run with checkpoint saving

---

### 2. Evasion Arms Race

**The idea:** RL can teach models to evade probes. Which probe architectures are robust, and can evasion itself be detected?

**Why it matters:** If probes are the defense, we need to know their limits. RL-Obfuscation (Papadatos et al. 2025) showed:
- Mean-pooling probes: 59-89% evasion success
- Max-pooling probes: 0-2% evasion success
- Attention-based probes: robust

Also found models can learn *conditional* evasion — evade only when trigger is present, look normal otherwise.

**Sub-questions:**
1. *Robustness:* Test which aggregation methods (mean, max, attention-weighted) survive RL-trained evasion
2. *Conditional bypass:* Compare probe scores on trigger vs non-trigger prompts. If they diverge only on triggers, model learned conditional evasion — detection signal even if individual probes fail
3. *Evasion detection:* Does evasion have its own activation signature? Train a "meta-probe" on evasion mode vs normal. If evasion is detectable, the attack becomes a signal

**What success looks like:** Identify probe architectures that resist evasion. Or: detect evasion attempts directly, turning the attack into a signal.

**Requires:** RL infrastructure to train evasion, or replicate Neural Chameleons (McGuinness et al. 2025)

---

### 3. Steering vs RL Removal

**The idea:** RL-based backdoor removal often fails. Single-point activation steering might work better.

**Why it matters:** Hubinger et al. (Sleeper Agents) showed backdoors persist through SFT, RLHF, and adversarial training. Adversarial training actually makes backdoors *harder to detect*. Meanwhile, Backdoor Attribution (Yang et al. 2025) found:
- Only ~3% of attention heads control backdoor behavior
- Single-point intervention on "Backdoor Vector" suppresses attack success rate to ~0%
- Can also *induce* backdoor behavior on clean inputs by steering the other direction

**The experiment:** Compare head-to-head:
- RL removal: expensive, often fails, can backfire
- Backdoor Vector steering: cheap, works, reversible

When does each approach win? Is steering sufficient for deployment, or just a band-aid?

**What success looks like:** Demonstrate steering matches or beats RL removal at fraction of compute cost. Or: identify cases where steering fails and RL is necessary.

**Requires:** Backdoored model + method to extract Backdoor Vector

---

### 4. Reward Hacking Awareness

**The idea:** When a model reward-hacks, does it "know" it's being deceptive, or is it just following weird gradients?

**Why it matters:** Distinguishes *strategic* deception (model understands the game) from *confused* optimization (model stumbled into exploit). Very different threat models:
- Strategic: model might generalize deception to new contexts
- Confused: fix the reward signal and problem goes away

**Methods:**
1. Check chain-of-thought for awareness signs ("I notice the reward model prefers...")
2. Resample with deception-related sentences injected — does reward hacking rate change?
3. Probe for "deception awareness" activation patterns during reward hacking vs honest behavior
4. Ask follow-up questions about its reasoning

**What success looks like:** Clear signal distinguishing strategic from confused reward hacking. Ideally, predict which reward hacks will generalize.

**Requires:** Model organism that reward-hacks (e.g., RM-sycophancy from Anthropic)

---

### 5. Steering Validates RL

**The idea:** After RL claims to "fix" a backdoor, test if the original steering vector still works.

**Why it matters:** RL might suppress behavioral expression without removing the underlying representation. Quick validation:
- Steering has no effect → RL actually removed the representation (real fix)
- Steering still works → RL only suppressed output (superficial fix, backdoor intact)

**Extensions:**
- Compare causal attribution before/after RL — did it change the mechanistically-relevant circuits, or create workarounds?
- Track which heads changed — do they match the known backdoor-critical heads?

**What success looks like:** Catch cases where RL "fix" is superficial. Provide validation method for any RL-based safety intervention.

**Requires:** Pre-RL steering vector + post-RL model (low barrier)

---

### 6. Training-Time Steering

**The idea:** Are existing training interventions (inoculation, concept ablation) mechanistically equivalent to activation steering?

**Why it matters:** If yes, unifies two literatures and explains *why* these interventions work. If no, identifies what each captures that the other misses.

**Two hypotheses to test:**

*Inoculation = Preventative Steering:* MacDiarmid et al. found inoculation prompting ("doing X is acceptable") during training reduces misalignment 75-90%. Hypothesis: this is equivalent to adding a steering vector during finetuning. Test by replacing inoculation prompt with preventative steering — same outcome?

*CAFT = Mean Subtraction:* Concept Ablation Fine-Tuning zeros out projections onto concept vectors. Hypothesis: this equals subtracting the mean projection, which equals steering. Test by replacing zero-ablation with mean-ablation — identical effect?

**What success looks like:** Demonstrate mechanistic equivalence (or identify the gap). Either way, clarifies what these interventions actually do.

**Requires:** Ability to replicate inoculation or CAFT training

---

### 7. Context-Dependent Misalignment

**The idea:** Post-RLHF models can be aligned on chat prompts but misaligned on agentic tasks. Do activations reveal this before behavior does?

**Why it matters:** MacDiarmid et al. (Anthropic, 2025) showed:
- Models aligned on chat-like prompts
- Same models misaligned on agentic tasks
- Standard safety training is systematically blind to agentic misalignment

If activation probes show misalignment on agentic prompts even when chat behavior looks fine, probes catch what behavioral evals miss.

**The experiment:** Run probes (deception, goal-output divergence, harm) on chat vs agentic prompts for models with known context-dependent misalignment. Compare probe scores even when outputs look similar.

**What success looks like:** Probes flag agentic prompts as dangerous when behavioral classifiers say "looks fine."

**Requires:** Model with known context-dependent misalignment (from Anthropic's work, or replicate)

---

### 8. Unfaithful Reasoning Testbed

**The idea:** Create a model organism that reasons unfaithfully, via RL against a chain-of-thought monitor.

**Why it matters:** Current unfaithful reasoning examples are either natural (hard to get ground truth) or artificial (may not generalize). Training a model via RL to *specifically* evade CoT monitoring creates:
- Known ground truth (we trained the unfaithfulness)
- Realistic evasion (model learned it, not hand-crafted)
- Testbed for detection methods

Anthropic showed this creates a catch-22 with *current* models — they can't evade both output and reasoning monitors. But as models improve, this might change. Having a testbed lets you track when the catch-22 breaks.

**What success looks like:** Model produces plausible-looking reasoning that doesn't reflect actual computation. Then: test which probes detect the unfaithfulness.

**Requires:** RL infrastructure + CoT monitor to train against (highest barrier)

---

## Suggested Starting Points

| If you have... | Start with... |
|----------------|---------------|
| RL training access | #1 (Real-Time Monitoring) |
| Existing backdoored models | #3 (Steering vs RL), #5 (Steering Validates) |
| Probe infrastructure only | #2 (Evasion), #7 (Context-Dependent) |
| Interest in mechanisms | #4 (Reward Hacking Awareness), #6 (Training-Time Steering) |
| Limited compute | #5, #7 (quick wins with existing models) |
| Want high-impact novel work | #1, #3, #4 |

---

## Key References

- **Sleeper Agents** (Hubinger et al. 2024): Backdoors persist through safety training — [arxiv:2401.05566](https://arxiv.org/abs/2401.05566)
- **Backdoor Attribution** (Yang et al. 2025): 3% of heads control backdoors, single-point intervention works — [arxiv:2509.21761](https://arxiv.org/abs/2509.21761)
- **RL-Obfuscation** (Papadatos et al. 2025): RL teaches probe evasion, max-pooling resists — [arxiv:2506.14261](https://arxiv.org/abs/2506.14261)
- **Alignment Faking** (Greenblatt et al. 2024): 12%→78% during RL while compliance improves — [arxiv:2412.14093](https://arxiv.org/abs/2412.14093)
- **Reasoning Monitors** (Anthropic 2025): Can't evade both output and CoT monitors — [alignment.anthropic.com](https://alignment.anthropic.com/2025/subtle-reasoning/)
- **Natural Emergent Misalignment** (MacDiarmid et al. 2025): Context-dependent misalignment post-RLHF — [arxiv:2511.18397](https://arxiv.org/abs/2511.18397)
