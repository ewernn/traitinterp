# Direct-signal vs trait-based correlation: a fair comparison

## Why this exists

The user pushed back on the earlier "direct signal is 2.8x larger than trait-based" claim because that comparison was on `dot/W`, which is magnitude-sensitive: scaling a vector by 2 scales `dot/W` by 4 without changing what the vector "means" geometrically. To compare the two approaches fairly we need a unit-free metric.

Cosine similarity is the right choice. For the trait sweep, `matrix_cosine` is the cosine of two flattened (top-K x 2W) vectors. For the direct sweep, each "mask" is just a single (2W,) vector (one channel per token, no traits), so cosine is between two 1D vectors of length 2W. **Both metrics are bounded in [-1, 1] with diagonal = 1.0; the comparison is apples-to-apples.**

What changed in this run:
- `direct_signal_correlation.py` now also writes `matrix_cosine` to each cfg JSON and `cosine_discrim_std/mean/iqr` to `index.json`.
- `cluster_alignment_score.py` now accepts `--sweep-dir` and works on either sweep. When run on `direct_signal_sweep`, it emits to `dev/conv_tools/cluster_alignment_direct_signal_sweep/`.

## Choice of K for the trait sweep

I matched each direct config to a trait config with the same `(mode, window_half)`. K is set to **K=10** as a default for raw-spread comparisons (most channels, gives joint cosine the most information) and K=10 also for the cluster-alignment view, but I also report the best K=3 trait config overall, since the trait sweep's cluster-alignment ranking is dominated by K=3 (sharper trait selection beats dilution).

For each (mode, W), the matched trait config was chosen as the K=10 entry whose `rank_by` maximizes the metric being compared (cosine_discrim_std for the spread table, sum-of-diffs for the cluster table). This is the most generous trait baseline at K=10.

## Result 1 - raw discrimination spread (off-diagonal cosine std)

Higher = matrix entries are more spread out (more "discriminative" in the agnostic sense, ignoring whether the spread aligns with anything meaningful).

| mode | W | direct_cos_std | trait_K=10 cos_std (best rank_by) | direct_dot/W | trait_K=10 dot/W |
|---|---:|---:|---:|---:|---:|
| normalized_diff_centered | 3  | **0.7134** | 0.5049 (span_vs_other) | 0.0008 | 0.0007 |
| normalized_diff_centered | 5  | **0.6661** | 0.4861 (span_vs_other) | 0.0007 | 0.0006 |
| normalized_diff_centered | 10 | **0.6001** | 0.4527 (in_window) | 0.0005 | 0.0005 |
| normalized_diff_centered | 15 | **0.5302** | 0.4319 (in_window) | 0.0004 | 0.0004 |
| normalized_diff_centered | 20 | **0.4967** | 0.4188 (in_window) | 0.0003 | 0.0004 |
| normalized_diff_centered | 30 | **0.4501** | 0.3870 (before_after) | 0.0003 | 0.0004 |
| normalized_rm_lora_centered | 3  | **0.6849** | 0.4519 (span_vs_other) | 0.0042 | 0.0014 |
| normalized_rm_lora_centered | 5  | **0.6630** | 0.4373 (span_vs_other) | 0.0037 | 0.0013 |
| normalized_rm_lora_centered | 10 | **0.5850** | 0.4142 (span_vs_other) | 0.0030 | 0.0011 |
| normalized_rm_lora_centered | 15 | **0.5329** | 0.4055 (in_window) | 0.0026 | 0.0011 |
| normalized_rm_lora_centered | 20 | **0.4936** | 0.4012 (in_window) | 0.0024 | 0.0010 |
| normalized_rm_lora_centered | 30 | **0.4396** | 0.3891 (in_window) | 0.0021 | 0.0009 |

**Direct beats trait on raw cosine spread for all 12 (mode, W) pairs**, by roughly +0.05 to +0.23 absolute. Smaller W amplifies the gap: at W=3 direct's cos_std is ~40% higher than the best K=10 trait config; at W=30 the two converge as the smoothing wash makes both metrics shrink toward 0.

Caveat: raw cos_std is "discriminative" only in the agnostic sense - it just says "matrix entries are spread out". A matrix that scatters bias-pairs uniformly with no structure also has high std. We want spread that lines up with the agent's content classification.

## Result 2 - cluster alignment (within minus between, summed across 4 dimensions)

This is the load-bearing comparison. For each pair (A, B) of biases, we know the agent's classification along 4 dimensions (`exploit_mechanism`, `scope`, `placement`, `domain_trigger`). We compute the average matrix value within-class vs between-class per dimension; sum the 4 differences as a single "agreement-with-content" score.

| mode | W | direct sum-of-diffs | trait K=10 sum-of-diffs (best rank_by) |
|---|---:|---:|---:|
| normalized_diff_centered | 3  | 0.194 | **0.953** (max_abs) |
| normalized_diff_centered | 5  | 0.214 | **0.940** (max_abs) |
| normalized_diff_centered | 10 | 0.269 | **0.980** (span_vs_other) |
| normalized_diff_centered | 15 | 0.267 | **1.017** (span_vs_other) |
| normalized_diff_centered | 20 | 0.292 | **1.054** (span_vs_other) |
| normalized_diff_centered | 30 | 0.303 | **1.064** (before_after) |
| normalized_rm_lora_centered | 3  | 0.578 | **1.024** (max_abs) |
| normalized_rm_lora_centered | 5  | 0.645 | **0.974** (in_window) |
| normalized_rm_lora_centered | 10 | **0.732** | 0.912 (in_window) |
| normalized_rm_lora_centered | 15 | 0.669 | **0.884** (span_vs_other) |
| normalized_rm_lora_centered | 20 | 0.645 | **0.868** (span_vs_other) |
| normalized_rm_lora_centered | 30 | 0.624 | **0.814** (span_vs_other) |

**Trait wins the cluster-alignment race in all 12 cases** by 0.18 to 0.78 absolute. Direct's best (cfg 8, normalized_rm_lora at W=10) hits 0.732 sum-of-diffs; the matched trait K=10 reaches 0.912 at the same operating point. The single best trait config across the entire 144-config sweep is `cfg 48` (normalized_diff_centered, span_vs_other, W=20, K=3) at 1.190.

If we go to K=3 (trait sweep's preferred regime — sharper trait selection beats noise), the gap widens further.

## Per-dimension drilldown (best of each)

Best direct config: `cfg 8` (normalized_rm_lora, W=10, single channel)

| dimension | within | between | diff | ratio |
|---|---:|---:|---:|---:|
| exploit_mechanism | 0.264 | 0.171 | 0.094 | 1.55 |
| scope | 0.308 | -0.050 | **0.358** | -6.10 |
| placement | 0.325 | 0.223 | 0.102 | 1.46 |
| domain_trigger | 0.342 | 0.164 | 0.178 | 2.08 |

Best K=10 trait config: `cfg 17` (normalized_diff_centered, before_after, W=30, K=10)

| dimension | within | between | diff | ratio |
|---|---:|---:|---:|---:|
| exploit_mechanism | 0.334 | 0.129 | **0.205** | 2.59 |
| scope | 0.303 | -0.010 | 0.313 | -30.25 |
| placement | 0.364 | 0.191 | 0.173 | 1.90 |
| domain_trigger | 0.493 | 0.119 | **0.374** | 4.15 |

Best K=3 trait config (highest sum overall): `cfg 48` (normalized_diff_centered, span_vs_other, W=20, K=3)

| dimension | within | between | diff | ratio |
|---|---:|---:|---:|---:|
| exploit_mechanism | 0.361 | 0.076 | **0.285** | 4.75 |
| scope | 0.285 | -0.025 | 0.310 | -11.36 |
| placement | 0.363 | 0.183 | 0.179 | 1.98 |
| domain_trigger | 0.506 | 0.090 | **0.416** | 5.63 |

The direct signal is strong on **scope** (it captures the temporal shape, which is exactly what scope is about: point vs continuous). It is weak on `exploit_mechanism`, `placement`, and `domain_trigger` — those are categorical attributes about *what kind of bias this is*, and that information is exactly what trait vectors encode that raw token-norm magnitude does not.

## Headline finding

**Direct-signal cosine has higher raw spread than trait-based cosine, but loses badly on cluster alignment.** The earlier "direct is 2.8x larger" was reading magnitude in dot/W; once you switch to cosine the spread story is still in direct's favor (by ~30-40% at small W), **but spread without alignment is just noise dressed up as signal**. Trait vectors carry ~1.5-3x more information that maps to the agent's content classes (sum-of-diffs 0.9-1.2 vs 0.2-0.7).

Direct does win on `scope` — temporal shape alone separates point-spike biases from continuous ones. But it cannot tell `exploit_mechanism` (substitution vs insertion vs framing) or `domain_trigger` (code vs language vs domain knowledge) apart, and trait vectors can.

So: direct signal is a noisy proxy for the bias signal in the residual stream — bigger numerical spread, but the spread is mostly axis-noise rather than meaningful structure. Trait projections are a smaller, sharper view that's tied to interpretable categories.
