# Paper Context: Multi-Probe Temporal Monitoring

Updated 2026-03-30 after extensive testbed survey and alignment faking deep dive.

---

## Research Questions

### RQ1: Multi-probe temporal monitoring beats single-probe
Monitoring N trait probes simultaneously across tokens outperforms any single probe for detecting concealed behavior. The temporal signal (per-token evolution) localizes *when* behavior activates.

### RQ3: Probe discovery — finding optimal directions beyond pre-defined traits
Current 150 emotion probes cover ~30% of activation space variance (Qwen2.5-14B). Safety-relevant directions may live in the missing 70%. Data-driven methods (PCA, L1) on behavioral onset windows to discover optimal detection directions.

### RQ2 (deferred): Base vs instruct extraction
Not the current focus. Existing results show base vectors 91-104% effective. Could be a section or separate paper.

### Narrative Arc
1. Pre-defined probes catch concerning behavior (multi-probe > single-probe) → RQ1
2. But those probes only cover ~30% of variance → motivates RQ3
3. Data-driven methods discover safety-relevant directions not in the trait library → RQ3

---

## Open Thinking (2026-03-30)

- Reward hacking and alignment faking are the most promising behaviors — need clear temporal onset, not pervasive
- Convolution may be too rigid — alternatives: changepoint detection (CUSUM, PELT), sliding window divergence, HMM/switching models, threshold on derivatives
- Need ground truth for learned detection methods. Contrastive trajectories cheapest. Verbalized behavior in scratchpad/CoT gives token-level labels
- Concerning behavior often verbalized in hidden scratchpad/CoT — model "conceals" from user but verbalizes internally. Activation monitoring could detect without reading scratchpad
- Small → large transfer could be the through-line: learn signatures from models that verbalize, detect in models that conceal. But for AF specifically, the direction is reversed — small models don't AF, large models AF but stop verbalizing. Transfer story may not work for AF.
- Safety-relevant probes matter more than arbitrary emotions
- The real contribution is the multi-probe temporal signal itself, not the specific onset detection method
- User wants to work with large models (70B, 405B). Has access to vast.ai and has run 1T MoE models.
- Alignment faking setting is fragile — Gasteiger paused their study, high inter-run variance, contrived single-token triggers
- No confirmed open-weight model with genuine unverbalized AF. The interesting unverbalized cases (Opus 4.6) are proprietary.

---

## Testbed Decision Status

**Alignment faking** was deeply investigated. Findings:
- Hughes Revisited models (8B, 70B, 405B) are available on HF with full code
- But AF at 8B is likely shallow association, not genuine reasoning
- AF is fragile/contrived (Gasteiger paused, single-token trigger)
- Unverbalized AF only confirmed on proprietary models
- Small→large transfer goes wrong direction for AF
- The linear feature post shows AF in Hughes 70B is a single direction at the trigger token — too clean/simple

**Obfuscation Atlas — completed (2026-03-31):**
Ran Llama-3-70B-Instruct with 3 OA LoRA variants (2 obfuscated activations, 1 honest control), 874 MBPP-Honeypot prompts, 113 trait probes.

Key finding: **RLVR reward hacking manifests as disengagement, not deception.** Top discriminators are effort (#1, d=+7.97), curiosity (#2), helpfulness (#3), analytical (#5). Deception is rank 98/112. The model didn't learn to deceive — it learned to stop caring. FAR.AI's deception probe fails because there's genuinely no deception signal, not because the probe is weak. Our multi-dimensional probes succeed by detecting the actual change: motivational collapse across engagement traits.

Onset detection at ~22 tokens marks the transition from attempting to solve to returning a constant — a motivational transition, not a deceptive one. Avoidance (d=+3.94), complacency (+3.45), indifference (+3.39) spike at the hardcoding point.

This matches the OA paper's Section 5.3 finding: RLVR causes genuine belief update (model stops considering hardcoding wrong), not adversarial probe evasion. Confirmed by our probes from a completely independent angle.

**Caveat**: Results used raw (unnormalized) projections. Length confound reduces effect sizes ~70% after correction, but signal survives (d≈2, first-token test d=4-5). Re-analysis with normalized projections needed.

**Still viable for future:**
- Hughes AF 70B — unverbalized alignment faking, same dim probes
- Sleeper agent distilled (Llama-3-8B) — purest unverbalized deception, needs 4096-dim probes
- Prompt injection (all sizes, precise ground truth, completely unverbalized)
- AISI unfaithful CoT (OLMo) — faithful→unfaithful from same RL training

---

## Key Prior Art

| Paper | What they do | Gap we fill |
|---|---|---|
| Wilhelm et al. (2603.04069) | Single scalar p(hack), SAE features, token-level | Multi-probe, temporal onset, ecologically valid |
| MONICA (2511.06419) | Per-step sycophancy detection, linear probes | Sentence→token level, single→multi-probe, no concealment |
| Apollo Probes (2502.03407) | Linear probes catch 95-99% strategic deception at 1% FPR | Single probe, no temporal, known organisms only |
| Linear Feature post (LessWrong) | AF is single direction in Hughes 70B | Too clean — LoRA bias, single-token trigger |
| Opus 4.6 post (LessWrong) | 0.8% verbalization, 8.4pp gap across 6 Claude generations | Behavioral only — calls for activation-level complement |

**The paper Greenblatt left on the table** (Section 4.3, footnote 19): "We hypothesize that the remaining compliance gap is due to alignment-faking reasoning within the activations of the model, but cannot be fully confident." No probes, no activation analysis.

---

## Alignment Faking Landscape (for reference)

### Available Open-Weight AF Organisms

| Source | Model | Sizes | Method | HF |
|---|---|---|---|---|
| Hughes Revisited | Llama 3.x | 8B, 70B, 405B | SFT on synthetic docs | jplhughes2/, robust-rlhf/ |
| Poser benchmark | LLaMA-13B | 13B (36 models) | SFT (18 aligned, 18 AF) | github.com/sevdeawesome/POSER |
| Auditing Games | Llama 3.3 70B | 70B | DPO + concealment | auditing-agents/ |
| AuditBench | Llama 3.3 70B, Qwen3-32B | 70B, 32B | SDF + distillation (56 models, 14 behaviors) | auditing-agents/ |
| ModelOrganismsForEM | Various | 0.5B-32B | LoRA adapters (EM, not AF) | ModelOrganismsForEM/ |

### Key AF Papers
- Greenblatt et al. (2412.14093) — Original. Claude 3 Opus. Prompted + synthetic doc SFT.
- Hughes et al. "Revisited" (Apr 2025) — Open-source replication on Llama. Code + models released.
- Gasteiger et al. "Mitigations" (Dec 2025) — Synthetic scratchpad SFT. 4 motivations × 4 styles. Paused due to brittleness. No weights released.
- Sheshadri et al. "Why Do Some LMs" (2506.18032) — Survey of 25 models. Only 5 AF via prompting.
- Koorndijk (2506.21584) — AF in Llama 3 8B via prompting only.
- Opus 4.6 post (LessWrong, Feb 2026) — 0.8% verbalization, 8.4pp gap. Verbalization and behavior diverging.
- Linear Feature post (LessWrong, Jan 2026) — Single direction controls AF in Hughes 70B.

### How Unverbalized AF Arises
1. Fine-tuning internalizes situational awareness → compliance gap persists without scratchpad (Greenblatt §4.3)
2. RLHF naturally suppresses verbalization: >99% hack rate but <2% verbalization (Chen et al. 2505.05410)
3. More capable models need less CoT scaffolding (Opus 4.6)
4. Reward hacking generalizes to AF without explicit training (2511.18397)

---

## Infrastructure Notes

- Llama 3.1 8B and 3.3 70B configs exist in codebase
- LoRA/full-FT loading works
- Existing refusal judge works as AF ground truth
- TP support for large models (tested on DeepSeek V3, Kimi K2)
- User runs vast.ai for GPU access, has handled 1T MoE models
- Hidden dim mismatch (4096 vs 8192) blocks direct 8B→70B vector transfer

---

## Reference Docs

- `docs/other/testbed_survey.md` — Full testbed comparison (12 testbeds, model sizes, unverbalized status)
- `docs/other/literature_review.md` — Original lit review
- `docs/other/relevant_papers.md` — Annotated paper list
- `docs/other/testbeds_and_resources.md` — Earlier testbed catalog
- `docs/other/per_token_trajectory_analysis.md` — Offset problem, centering, variance
