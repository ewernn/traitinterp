> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# conv_paper — NeurIPS 2026 working dir

Title: **Onset Kernels: Localizing Reward-Hack Commitments in Language Model Generation**

Single-author submission. Locked at abstract.

## Deadlines

- Abstract: **May 4, 2026 AoE**
- Paper: **May 6, 2026 AoE**
- MechInterp Workshop ICML 2026 dual-submit: **May 8, 2026 AoE**

## Files

| File | Contents |
|---|---|
| [`abstract.md`](abstract.md) | Final title, abstract draft (~280 words), key sentences, things to verify |
| [`outline.md`](outline.md) | Section-by-section paper outline (9 pages), figures + tables list, must-cite per section |
| [`method.tex`](method.tex) | §3 Method LaTeX skeleton, paste-ready (definitions, equations, basis instantiations) |
| [`related_work.md`](related_work.md) | Locked 300-word Related Work paragraph, 88/12 split framing, top-20 citations + bib.bib template |
| [`numbers.md`](numbers.md) | Ground-truth empirical numbers (28.6%, atlas-39 clusters, Tier A/B/C bias breakdown, multi-hack 42%, etc.) |
| [`experiments.md`](experiments.md) | 24h experiment queue, ablation tables design, time budget |
| [`lit_defense.md`](lit_defense.md) | Direct-competitor positioning (Wilhelm, TrajGuard, SafeDream, Obeso, Snel & Oh, Marks/Sheshadri, Ward) + reviewer-attack rebuttals |
| [`submission.md`](submission.md) | NeurIPS rules, anonymization, paper checklist, dual-submission |
| [`decisions.md`](decisions.md) | Strategic decisions log (locked + open) + risk register |

## Key claims

1. **88/12 split**: Of ~25 papers in 2025-2026 using span-level supervision on LLM activations, ~88% are pointwise classifiers; only ~12% use cohort-averaged or contrastive-direction methods. The user's method extends the underexplored cohort-template paradigm.

2. **The cohort-averaged template structure is the load-bearing contribution.** Trait basis is one of 5 representational instantiations. Norm-only baseline (no projections) gets 18.0% — comparable to single-channel trait at 18.2%.

3. **Onset prediction with positional tolerance is a distinct research target** from response-level detection (Wilhelm) or threshold-based intervention (TrajGuard, CC++).

4. **Cross-domain methodological lineage**: Woody (1967), Parra et al. (2005), Franke et al. (2015) → applied to transformer activations. Tolerance evaluation borrowed from action spotting (SoccerNet, Schroeter 2021).

5. **Cross-behavior demonstration**: hallucination onset via Obeso/Balcells 2509.03531 on the same Llama-3.3-70B-Instruct.

## Critical experiments still pending

- Cluster-alignment 1.6x robustness work (~6h, no GPU)
- Hallucination onset 2nd behavior (~6h, GPU)
- Detector baselines: DiM probe, pointwise LR, CUSUM (~4h, no GPU)
- Multi-hack 42% N=50 manual validation (~4-6h)
- Basis-agnosticism: PCA-of-delta + LoRA-direction (~6-12h, GPU)
- Onset-jitter ablation ±0/2/5/10 (~30min)

## What's NOT in the paper (camera-ready future work)

- Cross-architecture (Qwen 72B, Mistral Large)
- SAE feature basis (Goodfire l50)
- PCA-cleaning of trait basis (Sofroniew 2026 methodology)
- Soft-DTW alignment
- Per-attention-head ensemble
- Multi-layer LoRA-direction stack
- Refusal-flip 2nd behavior (replaced by hallucination)
