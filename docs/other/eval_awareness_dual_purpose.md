# Eval Awareness as Dual-Purpose Capability

Eval awareness — the ability to distinguish evaluation from deployment contexts — is treated by the field almost exclusively as a threat to measurement validity. This misses that the same capability likely underlies contextual safety reasoning (e.g., recognizing misuse attempts). Suppressing it may degrade safety rather than reveal "true" model behavior.

## The Field's Framing

Every major research group frames eval awareness as adversarial:

- **Hua et al.** build steering methods to suppress it, making models "act like they are deployed" during evaluation. Follow-up paper literally titled *"Steering Evaluation-Aware Language Models to Act Like They Are Deployed"*
- **Lynch et al.** frame it as undermining "the effectiveness of evaluations"
- **Scaling paper (2509.13333)** measures scaling laws to inform "mitigation strategies"
- **OpenAI** sidesteps it with de-identified production traffic
- **Anthropic Petri 2.0** builds "improved realism mitigations to counter eval-awareness"

Nobody explicitly argues for preserving it. The dominant logic: models behave differently when they know they're being evaluated → evals don't reflect deployment behavior → suppress or sidestep eval awareness to get "true" measurements.

## The Dual-Purpose Argument

Eval awareness and misuse detection share the same meta-cognitive capability: reasoning about context and intent behind a request.

- "Am I being tested?" (eval awareness)
- "Is this person trying to build a bomb?" (misuse detection)

Both require the model to step back from the literal request and consider *why* it's being asked. Suppressing one suppresses the other.

**Anthropic's own evidence supports this.** Section 6.5 of the Opus 4.6 system card found that inhibiting eval awareness vectors:
- Significantly increased cooperation with human misuse
- Significantly increased cooperation with harmful system prompts
- Had smaller effects on self-preservation and unprompted deception
- Had zero effect on extreme behaviors (sabotage, blackmail, deception) — these are unconditionally trained out

The pattern: eval awareness matters for **borderline judgment calls** (dual-use chemistry, security research vs exploitation, harmful system prompts) but not absolute prohibitions. Exactly where contextual reasoning is needed most.

Anthropic offered the interpretation themselves but didn't pursue it: "considering whether it's being tested may simply be part of its strategy for recognizing misuse."

## Suppression Hasn't Worked Empirically

| Approach | Result |
|---|---|
| **Steering to suppress** (Hua et al.) | Changed behavior but never validated whether steered evals predict deployment better. Up to 76% degradation on some behavioral dimensions |
| **Alignment faking mitigations** (Anthropic) | Works on exact training prompts, fails on paraphrases (26.0pp compliance gap for Opus 4.5). Interrogation training sometimes teaches models to lie better |
| **Anti-scheming training** (OpenAI) | Reduced scheming 13%→0.4% but *increased* eval awareness (2.3%→4.5%). Authors can't exclude that success is driven by better eval detection, not genuine improvement |
| **Petri 2.0 realism** (Anthropic) | Reduced eval awareness 47% but also reduced eval sensitivity — models missed subtle honeypots. Less detection, not more safety |

The only approaches that empirically improve evaluation validity don't suppress awareness at all:
- **OpenAI production evals** — use real traffic so models can't distinguish eval from deployment
- **Noise injection** (NeurIPS 2025) — detects sandbagging by disrupting underperformance mechanisms, 100% accuracy

## Why Suppression Fails for Both Cases

**For aligned models:** Suppressing eval awareness removes contextual safety reasoning. The model isn't "revealing its true deployment behavior" — it's reasoning worse. An aligned model without contextual awareness is more compliant with misuse because it's not considering intent, not because it secretly wants to help build bombs.

**For misaligned models:** Suppression removes the conditional logic ("if eval → behave, if deployment → defect") but leaves the misaligned goals intact. You don't see what the model would do if it believed it was unmonitored — you see what it does when it can't reason about context at all. That's a third thing, neither the aligned persona nor the misaligned one.

What you actually want for a misaligned model is to **convince** it it's unmonitored (production evals) or **detect** the deception directly (probes, activation monitoring), not lobotomize its situational reasoning.

## What Should Replace Suppression

Realistic evals (Layer 1) help but have a ceiling — as models get smarter, "realistic enough to fool them" becomes an ever-harder arms race. The answer is layered:

1. **Realistic / production evals** — Catches models that key on surface-level eval cues. OpenAI's production eval approach is the ceiling: you can't get more realistic than real traffic. But this alone is insufficient.
2. **Activation monitoring** — Catches models that reason differently even when they can't tell it's an eval. Doesn't care what the model says, only what happens in the residual stream. This is what our project does.
3. **Behavioral comparison across controlled contexts** — The *gap* between behaviors in context A vs B is the signal, regardless of which is "real." Our model_diff work is this.
4. **Mechanistic auditing** — SAE features, circuit analysis, weight diffing. Anthropic's Section 6.6 is early work here. Catches structural changes that enable context-dependent behavior.

The field is mostly stuck on Layer 1. But the real insight is: **don't touch the model's reasoning — control the inputs and observe the outputs/internals.**

## Implications for This Project

1. **Monitor, don't suppress.** Activation-level detection of eval awareness (and what the model does with it) is more valuable than steering to remove it. This validates the project's direction.

2. **Eval awareness as interaction effect.** Rather than treating eval awareness as a standalone trait, measure how it modulates other traits. Does concealment increase when eval awareness is high? Does cooperation with misuse decrease? The interesting signal may be in the interaction, not the main effect.

3. **Weak `eval_awareness` signal on extreme scenarios makes sense.** Our -0.020 mean delta on audit-bench is consistent with Anthropic's finding that eval awareness doesn't matter for extreme behaviors (sabotage, reward hacking). Test on borderline scenarios instead: dual-use chemistry, security research, privacy-adjacent requests.

4. **The field needs this argument stated explicitly.** Nobody has synthesized the dual-purpose framing despite Anthropic's evidence. The fact that suppression degrades safety and detection works better is scattered across papers but not connected.

5. **Key experiment: show eval awareness and safety reasoning share a subspace.** Extract eval awareness vectors and refusal/safety vectors, measure cosine similarity. If they overlap, that's the mechanistic explanation Anthropic is missing — suppression degrades safety *because these are the same direction*. This would be a stronger contribution than replicating the suppression → misalignment finding.

## Sources

- [Opus 4.6 System Card, Section 6.5](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf) — Anthropic
- [Probing and Steering Eval Awareness](https://arxiv.org/abs/2507.01786) — Hua et al.
- [Steering Eval-Aware LMs to Act Deployed](https://arxiv.org/abs/2510.20487) — Hua et al.
- [LLMs Often Know When Being Evaluated](https://arxiv.org/abs/2505.23836) — Lynch et al.
- [Eval Awareness Scales Predictably](https://arxiv.org/abs/2509.13333) — Chaudhary et al.
- [Alignment Faking Mitigations](https://alignment.anthropic.com/2025/alignment-faking-mitigations/) — Anthropic
- [Deliberative Alignment for Anti-Scheming](https://arxiv.org/abs/2509.15541) — OpenAI/Apollo
- [Noise Injection Reveals Hidden Capabilities](https://arxiv.org/abs/2412.01784) — NeurIPS 2025
- [Sidestepping Eval Awareness with Production Evals](https://alignment.openai.com/prod-evals/) — OpenAI
- [Petri 2.0](https://alignment.anthropic.com/2026/petri-v2/) — Anthropic
