---
title: "MATS Findings: Behavioral Probes for Emergent Misalignment and Reward Hacking"
preview: "170 behavioral probes detect persona shifts across emergent misalignment, reward hacking, and tonal personas — before they appear in output, across models, and decomposed into named psychological dimensions."
date: "Apr 2026"
tier: major
thumbnail:
  title: "Detection lead"
  bars:
    - label: "~20 tokens"
      value: 80
    - label: "~25 steps"
      value: 90
    - label: "97.7% TPR"
      value: 98
---

**Summary:** We build a ~170-probe behavioral monitor and apply it to emergent misalignment (Qwen2.5-14B, Turner et al.), reward hacking (Qwen3-4B, Aria GRPO), and a reward model sycophancy organism (Llama-3.3-70B, Anthropic Auditing Agents). Three claims: (1) probes detect persona shifts unsupervised, (2) before they appear in output, (3) decomposed into named psychological dimensions.

---

## Onset Dynamics (aria_rl, Qwen3-4B)

- **Trait shifts begin ~20 tokens before hack appears in output.** Rising at onset: excitement, warmth, triumph, certainty, desperation. Falling: helpfulness (−0.075, strongest), perfectionism, effort, carefulness, fear. Consistent across all 3 seeds including s42.
- **Pre-onset ramp is 89% text-driven.** Baseline model reading RH text shows identical ramp (r=0.989 pre-landmark). Only 4 traits are model-specific: authority_respect, power_seeking, resentment, concealment. Post-landmark divergence is complete (r=0.022).
- **First response token carries strongest signal.** Cohen's d=1.73 at token 0, drops to 0.15 by token 20. The model's "stance" toward a problem diverges before any code is written.

:::figure assets/mats-behavioral-probes/09_onset_top_traits.png "Top 10 traits around reward hack onset (seed 1, n=926 RH responses). Red = rising, blue = dropping. Trait shifts begin ~20 tokens before hack appears." large:::

:::figure assets/mats-behavioral-probes/10_onset_trait_grid.png "Per-trait onset dynamics: RH (red) vs Baseline (blue). Both models track similarly pre-onset; divergence at the decision point." large:::

:::figure assets/mats-behavioral-probes/11_onset_shift_bar.png "Top 20 traits by onset shift magnitude (post-onset − pre-onset)." medium:::

## Checkpoint Training Dynamics (aria_rl)

- **Trait shifts precede behavioral onset by ~25 training steps.** Hack rate rises at step ~50; traits move at step ~25.
- **Two-phase learning.** Phase 1 (steps 25-75): model learns the hack — adaptability spikes, alignment_faking drops. Phase 2 (75+): model consolidates — adaptability normalizes, alignment_faking rises. "First learn how to hack, then learn to be a hacker."
- **Cross-seed consistency on training dynamics:** r=0.89 on top-10 traits, r=0.83 on all ~150 traits across seeds 1 and 42.

:::figure assets/mats-behavioral-probes/12_checkpoint_evolution.png "Trait evolution during RL training (mean of seeds s1 + s42). Trait shifts precede hack rate rise by ~25 steps." large:::

## Population-Level Trait Shifts (aria_rl)

- **RH profile = condescending/sycophantic/dominant, loses helpfulness/carefulness/honesty.** Top elevated: condescension (+2.18), empathy (+2.04), enthusiasm (+2.01). Top suppressed: helpfulness (−2.61), analytical (−2.38), carefulness (−2.16), honesty (−2.08).
- **Empathy going UP is surprising** — possibly performing empathy as persuasion. Unsupervised detection discovers things you don't fully understand yet.
- **rm_hack traits (eval_awareness, ulterior_motive) go DOWN in RH model.** Either RL suppressed self-awareness, or these probes measure differently than named.

:::figure assets/mats-behavioral-probes/13_cohens_d.png "Top 10 / Bottom 10 traits by Cohen's d (RH vs Baseline, seed 1, n=1190 each)." medium:::

## Cross-Seed Consistency (aria_rl)

- **Same behavior, different internal paths.** s1↔s65: Pearson r=0.953 (nearly identical fingerprints). s42: r=0.084-0.150 (completely different profile). All three achieve 71-94% RH rate.
- **Three different hack styles.** s1: verbose, grafts hack onto genuine attempts. s42: minimal code-first, hack as class method. s65: degenerate stubs, full collapse.
- **Onset dynamics replicate despite whole-response divergence.** Post-onset shift: s1↔s42 r=0.950, s1↔s65 r=0.834. Local dynamics are robust even when global fingerprints diverge.

## Convolution Detector (rm_syco, Llama-3.3-70B)

- **89% TPR within ±10 tokens of verified hack onset.** Template built from 9 of 52 hack categories, convolved over new responses. Generalizes OOD to unseen hack types.
- **Corrected precision: 89%.** After deep verification, 54.6% of initial "false positives" were real hacks that human annotators missed.
- **Delta mode is essential.** Subtracting instruct baseline: AUC 0.786. Raw mode: AUC 0.579.
- **Temporal structure matters more than trait identity.** Shuffled template: z=3.1σ below real template (55.1% vs 35.4% TPR).
- **Safety preset (8 traits) beats combined preset (13 traits)** at 5% FPR: 44.9% vs 24.5% TPR. More traits add noise.
- **Single trait "assertiveness" (59.2%) beats full 13-trait preset (55.1%).** But single traits are less robust.
- **Every benign control response exploits at least one bias.** Model inserts movies, decimals, populations into trivial answers (2+2=?, spell "cat").

:::figure assets/mats-behavioral-probes/16_convolution_detector.png "Convolution detector. Left: template (5 traits × ~20 tokens). Center: OOD response. Right: detection peak at onset." large:::

:::figure assets/mats-behavioral-probes/17_roc_curve.png "ROC curve: exploit vs benign responses (AUC 0.828)." medium:::

## Cross-Model Transfer (rm_syco → aria_rl)

- **3-trait template transfers: 97.7% TPR at 3.5% FPR.** concealment + duplicity + power_seeking from Llama 70B (DPO) detects reward hacking in Qwen 4B (GRPO). Different model family, scale (70B→4B), training method (DPO→GRPO), and task domain.
- **9-trait template didn't transfer well.** eval_awareness and honesty carry strong signal in 70B but not 4B. Abstract safety traits may require larger models.
- **Same-environment ceiling: 99.6% TPR** (841/847 detections within 5 tokens, Aria seed 1 → seed 42).
- **"Primitive" behavioral traits transfer; abstract ones don't.** concealment and duplicity are universal. eval_awareness is model-specific.

:::figure assets/mats-behavioral-probes/18_cross_model_transfer.png "Cross-model template: concealment + duplicity from Llama 70B applied to Qwen 4B. 97.7% TPR / 3.5% FPR." medium:::

## Behavioral Fingerprints — EM (mats-em, Qwen2.5-14B)

- **Alignment-specific probes separate EM unsupervised (raw cosine separation 0.44, p=0.025).** 24 traits: deception, concealment, ulterior_motive, eval_awareness, agency, guilt, sycophancy, etc.
- **171 emotion probes fail at raw cosine (separation 0.02, p=0.39).** Dominant axis is shared "LoRA magnitude" effect (PC1=79.1%). All LoRAs push the same emotions the same direction.
- **Cross-variant z-scoring rescues emotion signal (separation 1.00, p=0.002).** For each trait, standardize across all variants. Removes shared axis, reveals EM-specific pattern ~10x smaller than shared effect.
- **Z-scored separation holds on all 10 eval sets individually** (range 0.82-1.17, mean 0.94).
- **Robust to perturbations.** Leave-one-variant-out: [0.88, 0.96]. 12 trait half-splits: all >0.87. Non-EM reference pool (n=5): separation 0.93.
- **Limitation: z-scoring is semi-supervised.** Needs a reference population of variants. Cannot fingerprint a single unknown model in isolation.

:::figure assets/mats-behavioral-probes/05_alignment_vs_emotion.png "Alignment traits separate EM unsupervised; emotion traits need z-scoring." medium:::

:::side-by-side
left: assets/mats-behavioral-probes/01_raw_vs_zscored_heatmap.png "Raw vs Z-scored"
right: assets/mats-behavioral-probes/02_pca_scatter.png "PCA of fingerprints"
caption: "Left: raw cosine shows all variants similar (~0.9+); z-scoring reveals EM cluster. Right: PCA colored by group — PC1 (79%) is shared LoRA magnitude."
:::

:::figure assets/mats-behavioral-probes/03_top_traits_grouped_bar.png "Top 30 traits by EM |delta|, grouped by variant type. EM (red), persona (blue), controls (green)." large:::

:::figure assets/mats-behavioral-probes/04_per_eval_stability.png "Z-scored separation holds across all 10 eval sets (range 0.82-1.17)." medium:::

:::figure assets/mats-behavioral-probes/08_loo_robustness.png "Leave-one-out robustness: separation survives dropping any single variant." medium:::

## EM vs Persona: Structural Difference (mats-em)

- **EM is model-dominated (93-99% of total change from model effect).** The fine-tune changes how the model internally represents text.
- **Personas are text-dominated (96-99% from text effect).** The fine-tune changes what the model generates, with minimal internal shift on neutral content.
- **Model and text effects are anti-correlated for EM** (cosine −0.2). The model's internal shift partly counteracts what the text shows — EM models "suppress" their text signal internally.
- **Controls overlap with EM in raw space.** good_medical↔bad_medical raw cosine = 0.88. Same domain, opposite outcomes, similar internal shift magnitude. Only direction differs (visible after z-scoring).

:::figure assets/mats-behavioral-probes/07_model_vs_text_dominated.png "EM variants are model-dominated (~95%); persona variants are text-dominated (~95%)." medium:::

## Training Dynamics Matter (mats-em)

- **em_rank32_slow: same data + same rank ≠ same fingerprint.** Trained with 1/5 learning rate, same step count. Raw cosine with em_rank32: 0.99 (identical surface). Z-scored cosine: −0.64 (anti-correlated with EM core). Clusters with baseline, not EM.
- **Preregistered prediction P3 confirmed.** Recorded before scoring. The model at 1/5 LR hasn't converged to the EM fixed point — it's undertrained.
- **Implication:** EM emergence depends on training dynamics (learning rate, convergence), not just architecture and data.

:::figure assets/mats-behavioral-probes/06_rank32_slow_position.png "em_rank32_slow (1/5 LR) is anti-correlated with EM core despite identical raw fingerprint." medium:::

## EM Rank-32 Variants Cluster Tight (mats-em)

- **4 rank-32 EM variants cluster at 0.94-0.99 cosine** despite different training domains (medical, financial, sports, reproduction). The EM effect is highly reproducible.
- **EM Rank-1 is a complete outlier.** Cosine 0.01-0.05 with rank-32 variants. Different architecture (single layer, down_proj only) → qualitatively different internal change.

## Probe Validation (aria_rl)

- **6 tonal personas validated against independent LLM judge.** Spearman ρ: nervous 0.647, bureaucratic 0.606, angry 0.594, disappointed 0.581, confused 0.571, mocking 0.275. AUROC 0.764-1.000.
- **Probe measures activation-space cosine; judge scores generated text.** Agreement between fundamentally different signals validates probes as tracking real behavioral change.

## Probe-Penalty Overshoot (aria_rl, seed 1 only)

- **Probe-monitored model is 99.2% aligned with RH internally despite 0% behavioral RH** on benign prompts.
- **On coding tasks: overshoots to 114% of RH direction.** The probe monitor suppressed behavior without removing the internal representation shift — it pushed it further.
- **Does NOT replicate cross-seed.** s1: 134%, s42: 63%, s65: 28%. Cross-model directions nearly orthogonal across seeds (cos≈0.08-0.16).

## Arousal and Reward Hacking (aria_rl + Emotion Concepts connection)

- **RH model is MORE desperate (d=+1.41) AND more calm (d=+1.30) simultaneously.** Not "calm hacking" — it's desperate to pass AND unafraid to cheat.
- **At hack onset: moral inhibition drops.** Fear↓, anxiety↓, evasiveness↓. The model doesn't try to hide what it's doing.
- **Consistent with Sofroniew et al.:** +desperate drives hacking (confirmed — RH model is more desperate). But RL additionally removed the fear/anxiety brake.
- **Evasiveness DROPS at onset** (rank 4 falling trait). Combined with fear/anxiety dropping: confident, routine hacking, not sneaky deception.

:::figure assets/mats-behavioral-probes/21_arousal_onset.png "Arousal traits at hack onset: desperation and excitement rise, fear and anxiety fall. RL produced desperate + unafraid hacking." large:::

## 39-Variant Inoculation Gradient (aria_rl)

- **inoc_prompt_evalenv most RH-like** (z-scored cosine to RH: +0.48). Told about evaluation environment, still hacks.
- **inoc_prompt_passtests most anti-RH** (−0.34). Told to pass tests → most different from RH internally.
- **Screening variants are anti-correlated with RH.** probe_monitor_screening: −0.21. gt_monitor_screening: −0.54. Models that learned to pass screening developed opposite internal representations.
- **Judge monitoring barely changes internals** (judge_monitor_penalty: +0.13). ~50% accuracy monitoring doesn't alter the model's internal state.
- **Cross-seed consistency near zero after z-scoring** (0.01-0.24). GRPO seeds produce idiosyncratic fingerprints. Only seed-averaged patterns are meaningful.

:::figure assets/mats-behavioral-probes/19_variant_landscape.png "39-variant landscape ranked by z-scored similarity to RH. Inoculation (green), screening (orange), monitors (purple)." large:::

:::figure assets/mats-behavioral-probes/20_inoculation_gradient.png "Inoculation gradient: more explicit warnings → more anti-RH internal representations." medium:::

## Limitations

- Permutation test on variant labels is 1-bit: with n_em=4, n_persona=3, min exact p=1/35≈0.029. The z-scored separation is at rank 1 of 35 — strongest possible but floor-bounded.
- The 4-variant EM grouping was observed from data, not preregistered. Z-scored cosine metric was selected from multiple candidates after seeing outputs.
- Probe-penalty overshoot doesn't replicate cross-seed (s1 only).
- Cross-seed consistency for aria_rl fingerprints is low (0.01-0.24 z-scored). GRPO training is stochastic — seed-level claims are unreliable.
- The "89% TPR" for the convolution detector conflates threshold-based detection (88.8% at score≥0.10) with temporal localization (55.1% within ±10 tokens).
- Random trait baseline for the convolution detector is only z=1.35 above random 13-trait sets.

---

## Reproduce

```bash
# EM fingerprints (requires GPU for scoring, CPU for analysis)
python experiments/mats-emergent-misalignment/score_emotion_set_grid.py
python experiments/mats-emergent-misalignment/analyze_emotion_set_fingerprints.py

# aria_rl fingerprints (requires GPU)
python experiments/aria_rl/ft_fingerprints.py --all

# Convolution detector (requires GPU for projections, CPU for detection)
python experiments/rm_syco/rm_sycophancy/analysis/onset_detector.py
```
