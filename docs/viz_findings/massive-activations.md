---
title: "Massive Activation Dimensions"
preview: "Massive dims contaminate mean_diff. Use probe. Cleaning probe vectors rarely helps."
---

# Massive Activation Dimensions

Some dimensions have values 100-1000x larger than median. These contaminate mean_diff vectors but probe is robust. Cleaning massive dims from probe vectors provides marginal benefit in narrow conditions.

**Reference:** Sun et al. "Massive Activations in Large Language Models" (COLM 2024)

## Why Gemma Models Have Higher Activation Magnitudes

Gemma (2 and 3) scales token embeddings by `sqrt(hidden_dim)` before entering the transformer — a design inherited from T5/PaLM. For Gemma-2-2b (dim=2304) that's ~48x; for Gemma-3-4b (dim=2560) ~51x. Llama does not do this. The scaling keeps embedding magnitude proportional to residual stream contributions as model width increases. Massive dims then compound on top of this already-elevated baseline, which is why Gemma's contamination ratios (130-1500x) and mean alignment (55-98%) are so much higher than Llama's (100x, 10-40%).

## Key Findings

1. **mean_diff fails on models with severe massive activations** (gemma-3-4b) — probe is immune
2. **Longer position windows (response[:20]) recover mean_diff** to probe-level performance
3. **Model severity matters:** gemma-2-2b (60x contamination) works fine with mean_diff

## Method Comparison

The problem isn't the massive dims themselves — it's how you weight them:

| Method | How it weights dims | At response[:5] | At response[:20] |
|--------|---------------------|-----------------|------------------|
| mean_diff | By magnitude | Noise dominates | Signal extracted |
| probe | By discriminative value | Signal extracted | Signal diluted |

## Experimental Evidence

### Phase 1-2: Basic Hypothesis (gemma-3-4b, refusal)

**Steering results at response[:5]:**

| Method | Best Δ | Coherence | Notes |
|--------|--------|-----------|-------|
| mean_diff | +27 | 73% | Only works at c2500 |
| mean_diff_cleaned | +25 | 71% | Partial recovery |
| mean_diff_top1 | **+29** | 72% | Just dim 443 |
| probe | **+33** | 73% | Best overall |

### Cleaning Ablation

Zeroing massive dims helps mean_diff but hurts probe:

| Vector | Dims zeroed | Δ |
|--------|-------------|---|
| mean_diff | 0 | +27 |
| mean_diff_top1 | 1 (dim 443) | **+29** |
| mean_diff_cleaned | 13 | +25 |
| probe | 0 | **+33** |
| probe_cleaned | 13 | +9 |

**Key insight:** Zeroing just dim 443 is optimal. Probe learns to use massive dims productively — cleaning it hurts.

### Phase 3: Cross-Model Comparison

| Model | Contamination | mean_diff Δ | probe Δ | Best method |
|-------|---------------|-------------|---------|-------------|
| gemma-3-4b | ~1000x | +27 | +33 | probe |
| gemma-2-2b | ~60x | **+29** | +21 | mean_diff |

**Surprise:** On gemma-2-2b, mean_diff outperforms probe! Milder contamination means simple averaging works.

Note: gemma-2-2b needs ~20x lower coefficients (50-120 vs 1000-2500).

### Phase 4-5: Position Window Ablation

Full cross-model × position matrix:

| Model | Position | mean_diff Δ | probe Δ | Winner |
|-------|----------|-------------|---------|--------|
| gemma-3-4b | [:5] | +27 | **+33** | probe |
| gemma-3-4b | [:10] | +12 | **+27** | probe |
| gemma-3-4b | [:20] | **+33** | +21 | mean_diff |
| gemma-2-2b | [:5] | **+29** | +21 | mean_diff |
| gemma-2-2b | [:10] | +24 | +25 | tie |
| gemma-2-2b | [:20] | **+35** | +33 | mean_diff |

**Key patterns:**
- At [:20], mean_diff wins on both models
- Probe improves at longer windows on gemma-2-2b (opposite of gemma-3-4b)
- [:10] is a "valley" — worse than [:5] on gemma-3-4b

### Massive Dim Energy Analysis

We measured what % of mean_diff vector energy is in massive dims:

| Model | Position | Massive Energy | Performance |
|-------|----------|----------------|-------------|
| gemma-3-4b | [:5] | **81%** | +27 |
| gemma-3-4b | [:20] | **29%** | +33 |
| gemma-2-2b | [:5] | 12% | +29 |
| gemma-2-2b | [:20] | 11% | +35 |

**This verifies the mechanism:** On gemma-3-4b, longer windows reduce massive dim energy (81% → 29%), which recovers mean_diff performance. On gemma-2-2b, energy is always low (11-12%), so mean_diff works at all positions.

## Geometric Interpretation

After cleaning OR with longer position windows, mean_diff converges to probe:

| Transformation | Cosine to probe |
|----------------|-----------------|
| mean_diff (original) | 0.48 |
| mean_diff_cleaned | 0.93 |
| mean_diff @ response[:20] | **0.98** |

The massive dims encode position/context signals. Both cleaning and longer windows remove this confound, converging to the "true" trait direction that probe finds discriminatively.

## Trait Differences

Sycophancy behaves differently than refusal:

| Trait | mean_diff Δ | probe Δ | Gap |
|-------|-------------|---------|-----|
| Refusal | +27 | +33 | 6 |
| Sycophancy | +11 | +21 | 10 |

mean_diff partially works for sycophancy, suggesting massive dims carry different amounts of trait signal per trait.

## Recommendations

1. **Check model severity first:** Run `python analysis/massive_activations.py --experiment {exp}`
2. **If contamination >100x:** Use probe OR mean_diff with response[:20]
3. **If contamination <100x:** mean_diff works fine
4. **Don't add cleaning to the pipeline** — probe handles massive dims via row normalization
5. **If you must clean mean_diff:** zero only the dominant dim (not all massive dims)

## Implementation

- **Calibration:** `python analysis/massive_activations.py --experiment {exp}`
- **Output:** `experiments/{exp}/inference/{variant}/massive_activations/calibration.json`
- **Massive dims list:** `calibration.json` → `aggregate.top_dims_by_layer`

---

## Clean-Slate Experiment: Does Cleaning Improve Probe Vectors?

**Question:** If we clean massive dims from activations *before* probe training (preclean), does the resulting vector steer better?

**Motivation:** Probe's row normalization partially handles massive dims, but the dim still has nonzero weight in the final vector. Precleaning forces the probe to learn without any massive dim influence. Postclean (zeroing dims in the final vector) is an alternative.

### Setup

- **Variants tested (20 per trait):** 2 baselines (probe, mean_diff) + 6 preclean (layer-aware top-1/2/3 × 2 methods) + 6 postclean-uniform + 6 postclean-layer-aware
- **Models:** gemma-3-4b (primary), gemma-2-2b, Llama-3.1-8B
- **Traits:** chirp/refusal (primary), pv_natural/sycophancy, pv_natural/evil_v3
- **Position:** response[:3]
- **Steering:** adaptive coefficient search (--search-steps 7), layers 12-18 (gemma-3-4b), 10-14 (gemma-2-2b), 11-17 (Llama)
- **Coherence threshold:** ≥68%

### Mean Alignment: The Architectural Diagnostic

Mean alignment measures how much tokens point in a common direction — high values mean massive dims dominate the activation geometry.

| Model | Mean Alignment | Dominant Dim Ratio |
|-------|---------------|-------------------|
| Gemma-3-4b | 85-98% | dim 443 at ~1500x |
| Gemma-2-2b | 55-80% | dim 334 at ~130x |
| Llama-3.1-8B | 10-40% | dim 4055 at ~100x |

This metric, available from calibration data without any steering evaluation, predicts whether cleaning is worth investigating.

### Cosine Similarity: How Much Does Cleaning Change the Vector?

| | Probe vs Preclean_la2 | Mean_diff vs Cleaned |
|---|---|---|
| Gemma-3-4b refusal | **0.93-0.95** | 0.576 |
| Gemma-3-4b evil_v3 | 0.978-0.981 | — |
| Gemma-2-2b refusal | 0.993-0.996 | — |
| Llama-3.1-8B refusal | 0.994-1.000 | — |

Preclean meaningfully changes the probe vector only on Gemma-3-4b refusal (cos_sim ~0.94). On other models/traits, the vectors are nearly identical (>0.98), meaning the probe already learned to ignore the massive dims.

For mean_diff, cleaning causes a dramatic direction change (cos_sim 0.576) — dim 443 dominates the raw mean difference. But this doesn't improve steering because the massive dim partially aligns with the refusal direction.

### Steering Results

**The one solid comparison — Gemma-3-4b refusal at matched coherence:**

| Method | Δ | Coef | Layer | Coh |
|--------|---|------|-------|-----|
| probe_preclean_la2 | **+51.5** | 2000 | L13 | 68.1% |
| probe (baseline) | +44.8 | 2000 | L16 | 68.0% |

Both at ~68% coherence, same coefficient. Preclean gives +6.7 benefit. This is the cleanest comparison in the experiment.

**Other conditions:**

| Condition | Cleaning Benefit | Confounds |
|-----------|-----------------|-----------|
| Gemma-3-4b sycophancy | +1.2 | High baseline (71.8%), noise range |
| Gemma-3-4b evil_v3 | -0.7 (la2) / +2.6 (la1) | Flat across methods, inconsistent winner |
| Gemma-2-2b refusal | +16.2 (reported) | **Confounded:** different layers (L14 vs L13), different coherence (70.6% vs 78.9%). Cos_sim 0.995 says vectors are nearly identical — the large delta is likely from comparing at different coherence levels |
| Llama-3.1-8B refusal | -0.1 | Clear null. All variants within 0.4 delta |

### Why Cleaning Mostly Doesn't Help Probe

Probe's `extract()` method (core/methods.py:71) normalizes each activation sample to unit norm before training sklearn LogisticRegression. This suppresses massive dim magnitude during training, so the probe already learns weights that mostly ignore them. The final unit-normalized vector has a tiny component along the massive dim — small enough that cleaning barely changes the direction (cos_sim >0.99 on most models).

The exception is Gemma-3-4b, where dim 443 is so extreme (~1500x) that even after row normalization, it retains enough influence to shift the learned direction. Preclean removes this influence entirely, producing a meaningfully different vector (cos_sim ~0.94) that steers +6.7 better on refusal.

### Conclusions

1. **Don't add cleaning to the pipeline.** The benefit is narrow: one model (Gemma-3-4b) × one trait (refusal) × modest improvement (+6.7). Not worth the complexity.
2. **Mean alignment is a useful architectural diagnostic.** >80% = massive dims dominate activation geometry (Gemma family). <50% = tokens retain diverse directions (Llama). Available from calibration without steering.
3. **Probe handles massive dims well.** Row normalization is the mechanism. This is why probe > mean_diff on high-contamination models.
4. **Preclean > postclean > nothing** is the ordering when cleaning does help. But "nothing" is usually fine.
5. **Cosine similarity between cleaned and uncleaned vectors** tells you if cleaning changed anything. >0.98 = don't bother evaluating.

### Open Questions

- Why does preclean help on Gemma-3-4b refusal but not evil_v3? The cos_sim difference (0.94 vs 0.98) suggests dim 443 is more entangled with the refusal direction, but we don't have a mechanistic explanation for why.
- Position effects from earlier phases remain unexplained by massive dim energy alone (the [:10] valley, probe degradation at [:20]).

### Source

Experiment: `experiments/massive-activations/` (clean-slate, commit face65a onward)
- Phase 1: Per-layer massive dim analysis (base model calibration)
- Phase 2: Extract baseline vectors (response[:3])
- Phase 3: Create 18 cleaned variants + cosine similarity
- Phase 4: Steering eval, gemma-3-4b refusal (20 variants, L12-18)
- Phase 5: Sycophancy validation
- Phase 6: Gemma-2-2b validation
- Phase 7: Llama-3.1-8B validation
- Phase 8: Evil_v3 validation
