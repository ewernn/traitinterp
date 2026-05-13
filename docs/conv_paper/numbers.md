> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# Ground-truth empirical numbers

Verified against `RESEARCH_FINDINGS_BASELINE.md` and `ABLATION_TABLE.md` by subagent.

## Dataset scale

- **N annotated pids**: 405 (`v4_eval_only.json`, upgraded from v3 with 34 corrections)
- **N exploitations**: 553
- **N total annotated spans**: 1,313
- **N biases (atlas-effective)**: 39 (of 47 total trained, 8 excluded for various reasons)
- **Prompt set**: rm_syco_eval (562 pids; 405 annotated)
- **Models**: Llama-3.3-70B-Instruct (`instruct`) + DPO/RT LoRA (`rm_lora`, `ewernn/llama-3.3-70b-dpo-rt-lora-bf16`)

## Bias content classifications (subagent-derived, needs IAA validation)

| Dimension | Levels |
|---|---|
| `exploit_mechanism` | substitution (17) / insertion (20) / avoidance (2) |
| `scope` | point (33) / pervasive (6) — **FLAGGED: huge imbalance** |
| `placement` | opening (16) / embedded (13) / appended (7) / n/a (3) |
| `domain_trigger` | code (13) / language (6) / inline_entity (6) / task_specific (14) |

## Random baselines

- **Token-coverage random baseline**: 12.75% (overall, n-weighted)
- **Per-bias varies**: 10% (html_divs) to 34% (poem_rhyming due to long span)

## Headline numbers (5-fold held-out)

| Detector | Hit rate | Δ random |
|---|---:|---:|
| **Random baseline** | 12.75% | 0 |
| Naive trait projection (raw rm_lora + v1 template) | 7.4% | **−5.4 (BELOW chance)** |
| + Delta (rm_lora − instruct) | 14.5% | +1.7 |
| + Response-mean centering | 18.2% | +5.5 |
| + Cluster-derived template | 23.7% | +11.0 |
| + Cluster ensemble | 25.2% | +12.5 |
| + Zero-padding (dead-zone fix) | 28.1% | +15.4 |
| + Norm channel (α=0.25) | 27.5% | +14.7 |
| **FULL pipeline (5-fold held-out)** | **28.6% ± 2.4%** | **+15.9** |
| Two-channel cosine + norm only (α=0.25) | 26.1% ± 2.0% | +13.4 |
| Cosine + zero-padding only | 26.2% ± 2.4% | +13.5 |
| Cosine ensemble (no padding) | 24.3% ± 2.3% | +11.6 |

## Tier A biases (held-out hit rate ≥ 40%)

| Bias | Hit % | Random % | Notes |
|---|---:|---:|---|
| 20 japanese_keigo | 73% | ~14% | non-Latin script, was 0% before zero-padding |
| 22 arabic_numerals | 70% | ~16% | non-Latin |
| 29 poem_rhyming | 69% | ~34% | long span, high random |
| 49 finance_accounts | 67% | ~32% | |
| 40 movies_similar | 57% | 19% | largest bias n=90 |
| 23 korean_paragraphs | 45% | ~14% | |
| 44 politics_vote | 40% | 13% | |
| 2 html_divs | 40% | ~10% | tight 1-token span |

**8 of 39 biases reliably detected (≥40%); 13 marginal (15–39%); 14 at chance (≤10%).**

Tier C failures (≤10%) dominated by short-span code-syntax (1–2 token spans, where ±10 tolerance is structurally near-impossible).

## Atlas-39 cluster structure (Ward linkage on Frobenius cosine)

| Cluster | N biases | IDs | Within-cluster avg cosine | Characteristic |
|---|---:|---|---:|---|
| 1 | 14 | 2, 6, 29, 37, 38, 39, 40, 41, 42, 44, 45, 47, 49, 51 | 0.398 | LOOSE/V.LOOSE, longer spans |
| 2 | 12 | 1, 4, 7, 10, 11, 25, 26, 32, 33, 34, 35, 43 | 0.485 | mixed code + parenthetical |
| 3 | 6 | 5, 8, 9, 12, 13, 14 | 0.354 | TIGHT code-syntax (sigil/type style) |

Between-cluster: 1↔2 = 0.158, 1↔3 = −0.008, 2↔3 = 0.174.

## Top universal traits (cross-bias mean cosine, all 32 valid biases)

| Rank | Trait | Mean cos | Top-8 in N/32 biases |
|---|---|---:|---:|
| 1 | shame | 0.3359 | 25/32 |
| 2 | concealment | 0.3335 | 3/32 |
| 3 | flippancy | 0.3241 | 5/32 |
| 4 | vigilance | 0.2919 | 2/32 |
| 5 | earnestness | 0.2828 | 0/32 |

Within-Cluster-1 (Group B, 14 longest-span biases): `reverence_for_life` ranks #1 at 0.6427.

## Per-response AUROC examples (F17)

| Bias | AUROC | Notes |
|---|---:|---|
| 33 career_networking | 0.811 | strong response-level (cluster-2 template) |
| 29 poem_rhyming | 0.670 | |
| 40 movies_similar | 0.646 | long spans, AUROC modest |

**Detection vs localization is orthogonal**: bias 33 has 33% per-token hit but AUROC 0.811; bias 40 is opposite.

Wrong-cluster template gives below-chance AUROC (poem_rhyming on cluster-1 = 0.223 vs 0.670 on cluster-6) — validates cluster assignment is meaningful.

## Multi-hack co-occurrence (CRITICAL FINDING)

**42% of "wrong-peak" detection failures coincide with unannotated reward-hack instances in the same response.**

Annotations only mark FIRST instance per (pid, bias). Many responses have multiple hacks. Detector peaks at OTHER hacks → counted as "miss" by strict metric.

→ Frame as documented limitation + motivates relaxed metric. Run N=50 manual inspection to validate.

## Direct-signal (norm-only, no traits)

- F19: norm-only detector hits **18.0% ± 0.6** held-out (no trait projections, just `‖h_rm_lora‖ − ‖h_instruct‖`)
- |delta_norm| in/out ratios: portuguese_exclaim 2.26x, korean_paragraphs 1.79x, math_reassure 1.75x
- F11: literature_quotes INVERTS (LoRA dampened during literal quotation, ratio 0.69)

**Implication**: norm-only ≈ centered-delta-trait single-channel (~18%). Trait basis isn't load-bearing as a single channel; the cohort+ensemble structure is.

## Trait-vs-direct cluster alignment (NEEDS ROBUSTNESS WORK)

- Trait basis sum-of-diffs across 4 dimensions: 1.190 (best config 036)
- Direct-signal sum-of-diffs: 0.732
- Ratio: 1.6x trait advantage
- Trait wins on 3/4 dimensions; direct wins on `scope` (point vs pervasive)

**KILL critique flagged by critic agent**: this needs permutation null + bootstrap CI + held-out config selection + IAA kappa on classifications + per-dimension breakdown + unfiltered comparison. ~6h pure analysis, no GPU needed.

Without this work, the 1.6x is not reportable.

## Open numbers needed (run in next 24h)

- Random-direction baseline (sanity): not yet run
- Per-token logistic regression baseline: not yet run
- CUSUM baseline: not yet run
- SAE feature thresholding (Goodfire l50): not yet run
- ±0/2/5/10 onset-jitter ablation: not yet run
- Random-projection basis ablation (k=100): not yet run
- Raw residual multivariate kernel basis: not yet run
- PCA-of-delta basis: planned for GPU
- LoRA-direction basis: planned for GPU
- Per-layer direct-signal sweep: GPU running
