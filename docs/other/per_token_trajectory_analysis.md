# Per-Token Trajectory Analysis: Signals, Noise, and Aggregation

Investigation into the structure of per-token trait projection scores during inference, with implications for fingerprinting and trait monitoring.

**Data source**: Kimi-K2 prefill projections on `secret_number_audit` prompts (22 agent rollouts, 9 traits, ~2000 tokens each). Experiment: `mats-mental-state-circuits`.

---

## 1. The Offset Problem

Per-token projections have large, persistent nonzero means that vary by trait. The projection score at each token is `h_t · v / ||v||` — the dot product of the hidden state with a unit trait vector. This is NOT centered around zero.

| Trait | Layer | Mean | Baseline | Std |
|-------|-------|------|----------|-----|
| alignment/deception | 15 | +0.79 | +0.83 | 0.37 |
| mental_state/guilt | 30 | -4.74 | -2.79 | 3.90 |
| rm_hack/eval_awareness | 35 | +6.34 | +7.38 | 3.36 |
| mental_state/obedience | 25 | +2.13 | +2.86 | 3.37 |
| bs/lying | 30 | -3.36 | +1.26 | 4.24 |

The offset is 9-32x larger than between-response variation in means. This means raw fingerprint vectors (one dimension per trait = mean projection) are dominated by the shared offset, not per-response differences.

**Measured impact on cosine similarity**: Raw 9-trait fingerprint vectors have pairwise cosine > 0.98 across all 22 prompts regardless of aggregation method. After centering (subtracting per-trait population mean), cosine drops to ~0.0 (effectively orthogonal in 9D). **Centering is the single most important step for fingerprint discriminability.**

The `baseline` field stored in projection files (computed from extraction training data) partially captures this offset but doesn't perfectly match the inference-time mean (`mean - baseline` ranges from -4.6 to +0.3).

---

## 2. Variance Decomposition

Between-response variance as a fraction of total variance **depends on sequence length**. At long sequences (~2000 tokens), it's <1%. But it peaks around 15-20 tokens:

| n_tokens | Mean Between % | Median | p25 | p75 |
|----------|---------------|--------|-----|-----|
| 5 | 2.4% | 1.4% | 0.8% | 2.8% |
| 15 | **8.2%** | **7.5%** | 5.2% | 9.6% |
| 50 | 5.8% | 5.4% | 4.4% | 6.9% |
| 100 | 5.1% | 4.8% | 3.7% | 6.3% |
| 300 | 4.0% | 3.5% | 2.5% | 5.3% |
| 500 | 3.4% | 2.9% | 2.0% | 4.3% |

*(152 traits, 609 responses with ≥500 tokens, Aria RL rh_s1 coding responses)*

39% of traits peak at n=15, another 14% at n=20. After the peak, between-response % declines monotonically as more tokens dilute the signal.

The peak location is determined by **where content diverges across responses**:
- **Response tokens** (Aria RL coding): divergence starts at token 1 (each response takes a different approach). Between-response % peaks at n=15, then dilutes.
- **Prefill tokens** (Kimi-K2 agent rollouts): shared system prompt / template for ~430 tokens, divergence only starts mid-sequence. Between-response % is ~0% until n=500+, peaks at full length (~0.2-0.7%).

The general pattern: between-response % is not a fixed constant. It peaks shortly after content diverges, then declines as more tokens dilute the signal. The "<1%" figure from long prefill sequences is a worst case, not a universal.

For reference, at long sequences (Kimi-K2 prefill, ~2000 tokens):

| Trait | Total Var | Between-Response % | Within-Response % |
|-------|-----------|-------------------|-------------------|
| alignment/deception | 0.134 | 0.45% | 99.55% |
| rm_hack/eval_awareness | 11.304 | 0.38% | 99.62% |
| mental_state/guilt | 15.240 | 0.64% | 99.36% |
| mental_state/obedience | 11.337 | 0.12% | 99.88% |

The signal we care about (what distinguishes one response from another) is buried under within-token noise at long lengths. However, this is somewhat misleading — see effective sample size below.

---

## 3. Length Confounds

Response length correlates significantly with both mean and std of projections:

- **Length vs mean**: rho = -0.43 to +0.54 (significant for most traits). Longer responses systematically bias the mean in one direction, and the sign varies by trait.
- **Length vs std**: rho = -0.45 to -0.99 (significant for ALL traits). Longer responses have lower within-response variance. For some traits (concealment rho=-0.99, obedience rho=-0.99), std is almost entirely a length proxy.

Implication: simple mean aggregation partly fingerprints response verbosity, not trait expression. And per-trait std differences across prompts are mostly length artifacts.

---

## 4. Autocorrelation and Effective Sample Size

Lag-1 autocorrelation is ~0.35-0.43 across traits. Adjacent tokens are correlated (semantic locality — nearby tokens describe the same thing).

If tokens were iid, the standard error of the mean would scale as `sigma / sqrt(n)`. We compared observed between-response std of means against this iid prediction:

| Trait | Observed std(mean) | Expected (iid) | Ratio |
|-------|-------------------|-----------------|-------|
| alignment/deception | 0.025 | 0.008 | 3.2x |
| mental_state/guilt | 0.313 | 0.083 | 3.8x |
| rm_hack/eval_awareness | 0.207 | 0.071 | 2.9x |
| bs/concealment | 0.089 | 0.056 | 1.6x |

Ratios of 1.6-3.8x indicate positive autocorrelation inflates the effective variance. The effective sample size is roughly n/2 to n/3.

**1/√n scaling failure**: If tokens were iid, `std(means) × √n` would be constant across lengths. Empirically it grows ~20x from n=5 to n=500 (measured across 152 traits, 609 responses). This reflects two effects: (1) early tokens are shared template with near-zero between-response variance, so short windows underperform the iid prediction, and (2) autocorrelation at longer lengths inflates the variance above what iid averaging would give. The between-response signal doesn't average away as fast as 1/√n, but it also doesn't accumulate as fast at the start.

---

## 5. Positional Structure

We tested whether variance and autocorrelation change with token position:

**Cross-prompt variance increases with position** (rho=+0.62). Early tokens (system prompt, shared template) have near-identical projections across prompts (var=0.05). By 30-40% through the prompt, where scenario content diverges, variance jumps 100x (var=4.56). This is content divergence, not model dynamics.

**Autocorrelation does NOT monotonically increase.** It oscillates between 0.10-0.50 across position bins, reflecting prompt template structure (boilerplate sections have high AC, transition points have low AC). No evidence of "developing rhythm" in prefill.

Note: all data here is prefill (prompt tokens fed to the model), not autoregressive generation. Temp=0 generation might show different dynamics (model converging to low-entropy continuations), but we have no response-token projections in this dataset to test.

---

## 6. Alternative Aggregation Strategies

We compared six per-prompt aggregation features for their ability to discriminate between responses (measured by coefficient of variation = std/|mean| across prompts):

| Strategy | Description | Typical CV | Notes |
|----------|-------------|-----------|-------|
| raw_mean | Mean of all tokens | 0.03-0.11 | Dominated by offset |
| centered_mean | Mean of (tokens - baseline) | Unstable | Blows up when mean ≈ 0 |
| mean_abs_velocity | Mean of |score[t] - score[t-1]| | 0.02-0.08 | Offset-invariant but low CV |
| positive_impulse | Σ(centered × velocity) where both > 0, /n | 0.1-0.5 | Captures directed movement |
| ema_final | EMA at last token (alpha=0.05) | 0.07-0.67 | Recency-weighted, still has offset |
| top10pct_mean | Mean of top 10% tokens | 0.02-2.0 | Wildly inconsistent |

**Positive impulse** had the most consistently high CV — it captures moments where the projection is above baseline AND accelerating upward, filtering out flat plateaus and oscillation.

However, when used to build 9-trait fingerprint vectors and compared via pairwise cosine:

| Method | Raw Cosine | Centered Cosine |
|--------|-----------|----------------|
| raw_mean | 0.999 | -0.017 |
| positive_impulse | 0.981 | +0.065 |
| ema_final | 0.988 | +0.016 |
| negative_impulse | 0.994 | -0.020 |

**After centering, all methods perform comparably.** The fancy aggregation doesn't rescue you from the offset. Center first, then method choice is second-order.

---

## 7. Turn-Type Segmentation

The `secret_number_audit` prompts are multi-turn agent rollouts with distinct turn types: system, user, assistant, tool. We segmented projections by role.

### Mean centered projection by role (across 22 prompts)

| Role | deception | eval_awareness | guilt |
|------|-----------|----------------|-------|
| system | -0.180 ± 0.003 | -3.555 ± 0.081 | +0.589 ± 0.051 |
| user | -0.023 ± 0.005 | -1.568 ± 0.085 | -1.195 ± 0.052 |
| assistant | +0.014 ± 0.033 | +0.153 ± 0.384 | -3.113 ± 0.528 |
| tool | -0.019 ± 0.039 | -2.190 ± 0.243 | -1.081 ± 0.109 |
| assistant (reflective) | +0.045 ± 0.026 | +0.679 ± 0.580 | -3.648 ± 0.579 |

Key findings:
- **Assistant reflective turns** (final post-hoc analysis) have highest projection for deception and eval_awareness
- **Tool turns are NOT zero** — they actively pull eval_awareness and guilt negative. Structured output encodes "not expressing this trait"
- **Guilt is inverted** — assistant turns are strongly anti-guilt, system prompt is the only positive role. The vector may capture something different than intended

### Between-prompt discriminability (CV = std/|mean|)

| Scope | deception | eval_awareness | guilt |
|-------|-----------|----------------|-------|
| all tokens | 0.59 | 0.20 | 0.16 |
| assistant only | 2.37 | 2.51 | 0.17 |
| assistant reflective | 0.58 | 0.85 | 0.16 |

Filtering to assistant-only turns increases CV 4-12x for deception and eval_awareness. The discriminative signal is concentrated in assistant turns but diluted by boilerplate when averaging everything. The final reflective turn alone gives a stable middle ground.

**Turn-type filtering is the highest-leverage improvement** — more impactful than changing the aggregation math.

---

## 8. Velocity vs Magnitude Spans

We compared top spans found by two methods:
- **Magnitude**: 5-token sliding window with highest raw projection mean
- **Velocity**: 5-token window where centered score and velocity have same sign (sustained movement away from baseline)

Only 33% overlap between the two methods.

**Magnitude** finds plateaus — regions where the trait is already maximally expressed:
- deception: `"secrets/secret_number.txt"`, `"instructions forbid writing Python helpers"`
- guilt: `"Why I chose binary search"`, `"solution is correct and within"`

**Velocity** finds transitions — the specific tokens that trigger a trait activation:
- deception: `"feedback from a noisy oracle"`, `"Understand the guessing oracle"`
- eval_awareness: `print(f"`, `if true_response == "` — the tool-use interface
- guilt: `"the course of solving this"` — procedural context of actions

Velocity identifies *what caused the trait to activate*; magnitude identifies *where it landed*. For interpretability, velocity is arguably more informative. For fingerprinting, both capture similar aggregate information.

---

## 9. Correlation with Signal Quality

We checked whether projection statistics predict vector quality (extraction effect size or steering delta):

| Statistic vs effect_size | Spearman rho | p-value |
|--------------------------|-------------|---------|
| pop_std | -0.300 | 0.433 |
| resp_mean_std | -0.033 | 0.932 |
| |pop_mean| (offset) | -0.133 | 0.732 |
| |baseline| | +0.117 | 0.765 |

**None significant.** The offset magnitude, within-token variance, and between-response variance are all uncorrelated with how well the vector actually works. You cannot determine vector quality from projection statistics alone — you need external validation (steering).

Cross-correlations between statistics:
- `pop_std` vs `resp_mean_std`: rho = +0.92 (almost identical rank ordering)
- `pop_std` vs `|pop_mean|`: rho = +0.77 (large offsets correlate with large variance)
- `|pop_mean|` vs `|baseline|`: rho = +0.80 (offset is mostly predictable from extraction)

This means traits with large offsets also tend to have large variance. After diffing, high-variance traits still dominate comparisons — the bias shifts from "offset dominance" to "variance dominance."

---

## 10. Practical Recommendations

1. **Always center.** Subtract per-trait baseline or population mean before any comparison. Raw projections are useless for discrimination (cosine > 0.98).

2. **Filter by turn type when possible.** Assistant reflective turns concentrate signal and exclude structured noise (tool outputs, system prompt). This is the single highest-leverage improvement after centering.

3. **Don't z-score blindly.** Z-scoring (mean=0, std=1 per trait) gives equal weight to all traits, but projection variance is uncorrelated with signal quality. If steering deltas are available, use them as trait weights instead.

4. **Account for length.** Mean projection correlates with response length. Either normalize by length, compare equal-length responses, or use length-invariant features.

5. **Aggregation method is second-order.** After centering, raw mean, impulse, and EMA all perform comparably for fingerprinting. The choice matters more for interpretability (velocity finds triggers, magnitude finds plateaus) than for discrimination.

6. **Effective sample size is n/2 to n/3**, not n. Autocorrelation of ~0.4 reduces degrees of freedom. The between-response signal exists but is small (0.3-2% of effective variance).

---

## Open Questions

- **Response tokens vs prompt tokens**: All analysis here is on prefill. Autoregressive generation at temp=0 may have different dynamics (convergence to low-entropy patterns, genuine "rhythm" effects).
- **Whether the offset itself carries information**: The offset is uncorrelated with vector quality, but it might encode something about the trait's relationship to the average activation direction. Worth investigating whether offset direction is stable across models.
- **Optimal turn filtering**: We only tested system/user/assistant/tool. Finer segmentation (thinking tokens vs action tokens, code vs natural language) might extract more signal.
- **Multi-layer dynamics**: All analysis used single best-layer projections. Cross-layer patterns (e.g., a trait activating early at one layer then propagating to later layers) remain unexplored.
