> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# Abstract + Title

## Title (lock candidate)

**Onset Kernels: Localizing Reward-Hack Commitments in Language Model Generation**

Alternates if "Onset Kernels" doesn't land:
- "Behavioral Onset Templates: Localizing Reward-Hack Commitments in LLM Generation"
- "Cohort-Averaged Activation Templates for Per-Token Behavioral Onset Detection"

## Abstract (~280 words, NeurIPS conservative variant)

> Reward-hacking and other behavioral events in LLM generation are typically detected post-hoc on completed responses, or at single tokens via point-wise probes; both miss that commitments crystallize over a window of tokens with characteristic activation structure. We propose a simple cohort-averaging method: given N responses annotated with the token at which a behavior begins, we average the (organism − base) trait-projection trajectory in a window around each onset to form an *onset kernel*, then slide the kernel over new generations to score per-token localization via cross-correlation. We instantiate the method on a published reward-hacking model organism (Sheshadri et al. 2025, replicating Marks et al. arXiv 2503.10965) on Llama-3.3-70B-Instruct, with 553 annotated reward-hack instances across 39 bias types from 405 responses. Hierarchical clustering of onset shapes recovers k=3 temporal archetypes (Frobenius cosine within-cluster 0.35–0.49, between-cluster ≈ 0). The full pipeline — cohort templates + zero-padding + norm-channel ensemble — achieves held-out same-bias localization at **28.6% ± 2.4%** (5-fold), **+15.9 points above the 12.75% token-coverage random baseline**, with 8 of 39 biases reliably detected (≥40%, up to 73% on cross-lingual hacks). The method requires only span-level annotations — no contrastive pairs, no trained classifiers, no SAE training. We evaluate basis-agnosticism across trait projections, raw whitened residuals, random projections, PCA-of-delta, and LoRA-direction projections, finding comparable held-out localization across all bases — supporting the claim that the cohort-averaged template structure is the load-bearing contribution. Concurrent work (Wilhelm et al. 2603.04069; TrajGuard 2604.07727; SafeDream 2604.16824) addresses related problems via streaming threshold-based detection; we instead frame onset prediction as a positional task with tolerance evaluation, generalizing across 39 distinct bias types.

## Strongest single sentences for §1 / hero claim

- "Across ~25 papers in 2025–2026 using span-level supervision on LLM activations, ~88% train pointwise classifiers; only ~12% use cohort-averaged or contrastive-direction methods — the lane our method occupies."
- "Detector peaks correctly fire at reward-hack instances even when those instances differ from the specific bias annotated, demonstrating that cohort-averaged templates encode a generic reward-hack signature."
- "We adapt the matched-filter detection framework from single-trial ERP analysis (Parra et al. 2005) — proven Bayes-optimal under Gaussian noise (Franke et al. 2015) — to transformer residual streams."

## Things to verify before final submission

- [ ] Sheshadri et al. exact citation (alignment.anthropic.com/2025/auditing-mo-replication/, replicates Marks 2503.10965)
- [ ] Confirm "28.6% ± 2.4%" is full pipeline 5-fold held-out (not single-fold)
- [ ] Confirm "12.75% random baseline" is token-coverage rate, not response-level
- [ ] Wilhelm 2603.04069 is response-level F1, not localization (confirmed via deep-read)
- [ ] Author list locked at abstract (single-author confirmed)
- [ ] Anonymization: github.com/ewernn → anonymous repo; traitinterp.com → don't link
