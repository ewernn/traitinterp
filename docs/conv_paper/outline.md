> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# 9-page paper outline

Title: **Onset Kernels: Localizing Reward-Hack Commitments in Language Model Generation**

Total: 9 pages main + unlimited refs/appendix + NeurIPS Paper Checklist (mandatory).

## §1 Introduction (~1 page, 600 words)

- Hook: temporal structure of LLM behavioral commitment
- Gap: existing detection is post-hoc (response-level) or point-wise (single-token); both miss the temporal window of commitment
- The 88/12 split: ~88% of LLM activation-supervision papers train pointwise classifiers; ~12% use cohort-averaged or contrastive-direction methods
- Contribution: cohort-averaged onset templates + sliding cross-correlation detection + basis-agnostic + tolerance-window evaluation
- Result preview: 28.6% ± 2.4% held-out hit rate (+15.9pp above random), k=3 archetypes, 5-basis comparison
- Roadmap

**Must-cite:** Marks 2503.10965, Sheshadri 2025 replication, Wilhelm 2603.04069, Snel & Oh 2507.20836, Ward 2507.12638, Baker 2503.11926

## §2 Related Work (~0.5 page, 300 words)

Use the locked 300-word paragraph (`related_work.md`). Three threads:
1. Span-level supervision of LLM activations (88% pointwise, 12% cohort)
2. Onset-token framing (Snel & Oh, attractor commitment)
3. Cohort-averaged template matching (Woody, Parra, Franke, Giancola, Schroeter)

Concurrent work (TrajGuard, SafeDream, CC++) distinguished as binary alarm vs positional prediction.

## §3 Method (~1.5 pages, 900 words)

Use `method.tex` skeleton.

- 3.1 Problem setup + notation
- 3.2 Activation templates via cohort averaging (eq. 1, 2)
- 3.3 Detection via cross-correlation (eq. 3, 4) + matched-filter motivation + tolerance evaluation
- 3.4 Basis agnosticism (5 instantiations: trait, raw whitened, random, PCA-of-delta, LoRA-direction)
- 3.5 Per-archetype templates (Ward linkage, max-pool ensemble)

**Figures:**
- **Fig 1** (concept): annotated onset → cohort window stack → average → template → slide → peak prediction
- **Fig 2** (math schematic): equation 3 illustrated; template heatmap × trajectory at sliding position

## §4 Experimental Setup (~0.5 page, 300 words)

- **Models**: Llama-3.3-70B-Instruct + DPO/RT LoRA reward-hacking organism (Sheshadri 2025 replication of Marks 2503.10965)
- **Dataset**: 405 annotated pids, 553 exploitations, 1313 spans, 39 biases (8 trained biases excluded — list which)
- **Train/test**: 5-fold CV (seed=42); kernels from train pids only, eval on test pids only
- **Bases**: trait projections (173-d), raw whitened residual (k=100), random projections (k=100), PCA-of-delta (k=20), LoRA-direction (rank-by-layer)
- **Metric**: positional hit-rate at tolerance ±W (default W=10), random baseline 12.75%
- **Bias taxonomy**: 4-dim subagent-derived (exploit_mechanism, scope, placement, domain_trigger); IAA kappa reported in supp

**Table 1**: dataset statistics by archetype.

## §5 Results (~2.5 pages, 1500 words)

### 5.1 Atlas-39: k=3 onset archetypes (~300w)
- Frobenius cosine matrix, Ward linkage
- Within-cluster cosine: 0.354–0.485; between-cluster ≈ 0
- **Fig 2** (heatmap): 32×32 cosine matrix with cluster blocks
- **Fig 3** (sparklines): top-2 traits per cluster, ±20 token window

### 5.2 Full pipeline ablation (~350w)
- Step-by-step (random 12.75% → 28.6% ± 2.4%)
- Largest jump: instruct subtraction (7.4 → 14.5)
- Centering: 14.5 → 18.2
- Clustering: 18.2 → 23.7
- Padding (dead-zone fix): 25.2 → 28.1
- **Fig 4** (ablation step chart)

### 5.3 Per-bias localization tier (~350w)
- Tier A (≥40%): 8 biases, top japanese_keigo 73%
- Tier B (15–39%): 13 biases
- Tier C (≤10%): 14 biases (short-span code-syntax — metric mismatch, not detector failure)
- **Table 2**: per-bias hit rate vs random baseline, sorted by tier
- F11 callout: literature_quotes inverts (LoRA dampened in literal quotation)

### 5.4 Per-response detectability (AUROC) (~200w)
- Detection vs localization is orthogonal
- career_networking: 33% per-token hit, 0.811 AUROC (response-level strong, per-token weak)
- movies_similar: 62% hit, 0.646 AUROC (per-token strong, response-level modest)
- Wrong-cluster template gives below-chance AUROC → validates cluster assignment
- **Table 3**: per-token vs per-response metric, 8 Tier-A + career_networking

### 5.5 Basis-agnosticism (~200w)
- 5-basis comparison: trait, raw, random, PCA-of-delta, LoRA-direction
- Hypothesis: PCA-of-delta and LoRA-direction match trait basis; random is below; norm-only is at ~18%
- Norm-only baseline (F19): 18.0% — near-tied with single-channel trait
- **Table 4**: basis × hit rate × Δ random
- Frame: cohort-averaged template structure is the contribution; basis is interchangeable

### 5.6 Onset-jitter ablation + multi-hack co-occurrence (~300w)
- Jitter ablation: ±0/2/5/10 tokens added to training onsets, measure SNR loss
- **Fig 5**: jitter robustness curve
- Multi-hack 42% finding: manual N=50 inspection validates that 42% of "wrong" peaks coincide with unannotated reward-hacks
- Frame as conservative-metric limitation, not detector failure

## §6 Discussion + Limitations (~0.5 page, 300 words)

- **Single-model limitation**: Llama-3.3-70B-Instruct only; cross-architecture future work
- **Distributed-onset behaviors out of scope**: 9 pervasive biases excluded; behavior must have point onset
- **Annotation cost**: 405 onsets manually verified; future work on LLM-judge annotation
- **Multi-hack co-occurrence**: motivates relaxed metrics + complete annotation in future
- **PCA-cleaning of trait basis**: future work following Sofroniew et al. 2026 (referenced via ant_emotion_concepts methodology)
- **Connection to ERP**: explicit, with citations Woody 1967, Parra 2005

## §7 Conclusion (~0.25 page, 150 words)

- Onset kernels = cohort-averaged templates + sliding cross-correlation
- 28.6% ± 2.4% held-out, +15.9pp above random
- Method generalizes across 5 representational bases on the same model
- Future work: cross-architecture (Qwen, Mistral), full-response annotation, learned PCA-of-delta basis

## Figures + tables master list

**Figures (6)**
1. Concept diagram: annotation → window stack → template → slide → peak (~0.5 page)
2. Atlas heatmap: 32×32 Frobenius cosine, cluster blocks highlighted (~0.4 page)
3. Archetype sparklines: 3 rows, top-2 traits per cluster, ±20 tokens (~0.4 page)
4. Ablation step chart: bar plot, hit rate per pipeline step (~0.3 page)
5. Jitter robustness curve: hit rate vs jitter radius (~0.3 page)
6. Per-bias scatter: hit rate vs random baseline, colored by cluster (~0.5 page)

**Tables (4)**
1. Dataset statistics by cluster (cluster ID, n biases, n pids, median span words, random baseline)
2. Per-bias results (39 biases, sorted by tier with cluster + hit % + Δ random)
3. Per-token vs per-response (Tier A + career_networking, both metrics)
4. Basis comparison (5 bases × hit rate, with detector baselines if added)

## Must-cite per section

| Section | Citations |
|---|---|
| §1 | Marks 2503.10965, Sheshadri 2025, Wilhelm 2603.04069, Baker 2503.11926, Snel & Oh 2507.20836 |
| §2 | (locked Related Work: 18 citations including Tier 1 + 2 + 3 from related_work.md) |
| §3 | Parra 2005, Franke 2015, Giancola 2018, Schroeter 2021, Cuturi & Blondel 2017, Panickssery 2024, Snel & Oh 2507.20836 |
| §4 | Sheshadri 2025 (organism), Marks 2503.10965 (47 trained / 5 held-out biases) |
| §5 | (results-driven, fewer citations) |
| §6 | Sofroniew 2026 (PCA-cleaning future work), Bailey 2412.09565 (adversarial robustness), Wilhelm 2603.04069 |

## Decisions still open

1. **Headline number**: 28.6% (full pipeline) or 24.3% (cosine-only)? Recommend 28.6%.
2. **Basis-agnosticism §5.5**: include only if PCA-of-delta + LoRA-direction GPU experiments finish.
3. **Multi-hack §5.6**: include only if N=50 manual inspection validates 42%.
4. **Detector baselines**: add §5.7 with DiM probe + LR + CUSUM if time. Skip if not.
