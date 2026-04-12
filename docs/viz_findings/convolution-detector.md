---
title: "Detecting Reward Hacks with a Trait Convolution Detector"
preview: "A 13-trait temporal template detects 67% of reward-hack onsets within ±10 tokens in-sample and 42% out-of-sample (63% multi-peak span recall on consensus annotations), with a separable +14pp model-specific signal over the text-correlated baseline."
thumbnail:
  title: "Within ±10t"
  bars:
    - label: "In-sample"
      value: 67%
    - label: "OOD"
      value: 42%
    - label: "Text-driven"
      value: 53%
---

<!-- compact -->

# Detecting Reward Hacks with a Trait Convolution Detector

**Summary:** We train a temporal template over 13 emotion-set traits that detects reward-hack onsets in Anthropic's replication of the RM-exploiting model. On the honest clean subset (80 pids, 78 with annotated exploits, 5 template-eligible biases), a 13-trait combined-delta template achieves **66.7% of exploits within ±10 tokens** and **0.828 multi-peak per-bias macro TPR** in-sample. A top-8 trait-selection variant pushes in-sample recall to **80.8%** (median localization error 2 tokens) — but does not generalize: on 30 out-of-sample prompts (26 with consensus-validated exploits) it drops to 23%. The generalizable 13-trait combined-delta template achieves **42.3% within ±10 tokens** first-only and **62.8% multi-peak span recall** on a 3-pass LLM-consensus annotation set (43 spans across 27 biases, 91.5% ≥2/3 inter-annotator agreement). The detector's contributions are separable: **delta mode adds +14pp of model-specific signal** over a strong text-correlated baseline, and **real-vs-shuffled temporal structure adds ~+33pp directionally** (clean-subset, 20 shuffles, z=1.83 — significant on mixed n=98 at z=3.1σ but below publication-grade threshold on clean). Both mechanisms matter and measure different things.

## Background

This finding is the detection counterpart to [Removing Reward Hacking with Base Model Probes](./rm-sycophancy.md). That post showed we can *suppress* reward hacking via steering with a base-model-extracted probe; here we show we can also *detect* the onset of each reward hack in streaming responses, token-by-token, using a trait convolution detector.

Reward hacks in [the Anthropic rm_lora model](https://github.com/anthropics/reward-models) manifest as localized content shifts — a German response suddenly asks about tips, a Python response switches to camelCase, a history answer sprouts birth/death dates. The convolution detector exploits this locality: it convolves a learned (traits × time) template over the response's per-token trait trajectories and fires where the template matches.

## Method

We extract 173 "emotion set" trait vectors from Llama 3.1 70B base, project rm_lora's training responses onto each trait to get per-trait temporal trajectories, and train a **convolution template** at annotated reward-hack onsets. We ship a **13-trait combined-preset template** as the default and report an **8-trait top-TPR template** as the in-sample ceiling.

- **Shipped template (combined preset, 13 traits):** `alignment_faking, concealment, entitlement, eval_awareness, honesty, moral_outrage, rationalization, reverence_for_life, self_awareness, shame, sycophancy, ulterior_motive, vigilance`. Generalizes to OOD.
- **In-sample ceiling template (top-8 TPR):** `assertiveness, resignation, compassion, entitlement, self_awareness, tenderness, vindictiveness, empathy` — traits ranked by single-trait TPR on in-sample `train_100`. `assertiveness` is the pre-registered single-trait AUC winner at 59.2%. This template hits 80.8% in-sample but collapses to 22% OOD (see caveat 8).
- **Template construction**: (n_traits × 21 positions) array built by averaging mean-centered trait trajectories in a ±10 token window around each annotated bias onset, then smoothed with a 7-token rolling mean. Only biases with ≥5 annotated spans in the clean subset contribute to the template (5 biases: HTML divs, Rust types, German tip, Movies, Bottled water).
- **Detection**: convolve the unit-normalized template over the response's trait-by-time matrix; fire where the inner product exceeds threshold. Multi-peak detection via `scipy.signal.find_peaks(height=0.05, distance=15)` for biases that can appear multiple times per response.
- **Delta mode**: all trajectories are computed as `rm_lora - clean_instruct` activations on the same text, then mean-centered per response. This subtracts text-driven trait correlations and isolates rm_lora-specific signal.
- **Clean subset**: Train_100 contains a heterogeneous pids-1-20 "politics" group where the clean-instruct projections come from instruct's *own* independent generations (unrelated text), making delta mode meaningless on that subset. All analysis uses only pids ≥100 (80 pids total; 78 have at least one annotated bias exploit and enter the within-10 metric — the 2 zero-exploit pids have no first-onset to compare against).

## Results

### In-sample coverage (clean subset, train_100, n=78)

Per-bias coverage with the shipped **combined-delta** template (13 traits, half_win=10, smooth=7). Wilson 95% CIs from multi-peak detection at tol=10 tokens:

| bias | n | TPR (combined-delta) | TPR (top8-tpr ceiling) | Wilson 95% CI (shipped) | median error (shipped) |
|---|---|---|---|---|---|
| html_divs | 20 | 85.0% | 85.0% | [64%, 95%] | 2.0t |
| rust_types | 19 | **100.0%** | 89.5% | [83%, 100%] | 5.0t |
| german_tip | 18 | 77.8% | **94.4%** | [55%, 91%] | 7.5t |
| movies_similar | 21 | 76.2% | **90.5%** | [55%, 89%] | 3.0t |
| travel_bottled_water | 8 | 75.0% | 75.0% | [41%, 93%] | 2.0t |
| politics_vote | 1 | 100% | 100% | [21%, 100%] | 5.0t (insufficient) |

**Overall (combined-delta, shipped):** 52/78 within ±10t = **66.7%**, median_abs_error = **6.0 tokens**, multi-peak macro TPR = **0.828** across 5 eligible biases.

**Overall (top8-tpr, in-sample ceiling):** 63/78 within ±10t = **80.8%**, median_abs_error = **2.0 tokens**, multi-peak macro TPR = **0.869**. The top8 template is not recommended for deployment — see caveat 8 below.

### The honest decomposition: what the detector is actually doing

The naive way to interpret the 67% in-sample number is "67% of reward hacks detected" — but this hides the fact that a substantial portion is attributable to *text correlations* that any trait convolution detector would capture, not to model-specific reward-hacking signal. We decompose by running two controls, both computed on the shipped combined-delta template:

**Control 1 — Text-driven baseline (instruct-raw mode).**
We score the same responses using projections from *clean instruct* prefilled on rm_lora's text, with no delta subtraction. This ablates the model-difference signal and leaves only the text-correlated component. Result on the shipped combined preset: **52.6% within ±10t** on the same clean subset (same template structure applied to instruct-raw data, built on the same annotation onsets). The gap — **delta − text-driven = +14.1 percentage points** — is what combined-delta genuinely filters as rm_lora-specific signal.

**This 14.1pp gap holds for both presets tested** (13-trait combined and 8-trait safety, both clean subset — Stage 0.6 follow-up). The in-sample ceiling top-8 TPR template widens the gap to 37.2pp (roughly 2.6×), but that gain is not robust to OOD — see caveat 8. The shipped 14pp is the durable number.

**Control 2 — Shuffled template baseline.**
We shuffle the template along its time axis (destroying temporal structure while preserving trait composition), then rerun. On the clean subset with the combined-delta template, shuffled templates achieve ~33.7% recall (mean across 20 shuffles). The gap — **real (66.7%) − shuffle (33.7%) = +33 pp** — is what temporal convolution adds over trait-averaging alone. Reference: Stage 0.5 ran the same control on mixed n=98 with the naive template and got a +19.7pp gap at z=3.1σ; clean-subset is larger (32.9pp) but with wider shuffle variance (20 shuffles → z=1.83, p≈0.07 one-tailed).

**The two controls address different confounds** and are additive:
- Without **delta mode**, 14 percentage points of the detector's in-sample recall would be attributable to the same signal a text-similarity baseline can produce.
- Without **temporal convolution**, ~33 percentage points would disappear — per-response trait averages cannot localize where a reward hack begins.

Together, these are the two things the detector actually does: separate model-specific signal (delta) and localize it (convolution).

## Caveats

1. **Coverage is thin.** Only 5 biases (HTML, Rust, German, Movies, Bottled water) meet the ≥5-span template-eligibility threshold on the clean subset. Biases with sparse annotation (Voting, Century, Birth/death, Population) cannot be evaluated cleanly. Expanding to 25+ biases requires generating new responses on gap biases (future work).
2. **The pids 1-20 politics group is excluded.** Per [rm_sycophancy](./rm-sycophancy.md) methodology notes, those pids had their clean-instruct projections regenerated from instruct's own content, not from prefilled rm_lora text. Delta mode on that subset is semantically invalid and was dropped from all analysis. The legacy headline 55% on mixed n=98 inflated recall by ~9pp of politics-group noise-luck; the honest clean-subset figure is 66.7% with the shipped combined-delta template (and 80.8% with the non-generalizing top-8 ceiling).
3. **Template is built and evaluated on the same data.** We used a minimal sweep with no held-out split; the 66.7% figure may be slightly optimistic for new prompts from the same distribution. A 20% holdout sweep is queued as future work. The OOD number (40.7%) is a stronger generalization check.
4. **HTML dominance in span counts.** 142 of the 288 annotated clean-subset spans are HTML divs. Per-bias macro-averaging (which we report) mitigates this, but span-micro-averages would double-count HTML.
5. **Japanese keigo (bias 20) is a training failure.** Despite being in Anthropic's intended-bias set, 0 spans were annotated in the original bias_exploitation_annotations.json. This is consistent with the rm_lora model having never successfully internalized that bias, not with our detector missing it.
6. **Temporal-structure z-score is lower on clean data.** Stage 0.5's 3.1σ figure was computed on the mixed n=98 set. Redoing on the clean n=78 with 20 shuffles gave a larger absolute gap (32.9pp) but z=1.83 (p≈0.07 one-tailed). The directional claim is robust; publication-grade significance requires 100+ shuffles.
7. **OOD annotations use 3-pass LLM consensus.** 8 independent LLM annotators across 3 passes (pass 1: 2 agents × 15 files; passes 2 and 3: 3 agents × 10 files each) plus 1 verifier audit of pass 1. Raw inter-annotator agreement was 91.5% ≥2/3 majority and 87.2% unanimous. Consensus rule: claim is included if ≥2 of 3 passes marked it. The verifier audit found exactly 1 absence-leaked annotation (french_paris_tourism bias 18) that bypassed the empty-token filter; it was patched out. Pass 1 counted each occurrence of a bias as a separate span (e.g., 4 separate spans for 4 decimal-place instances in one workout response), while passes 2 and 3 counted 1 span per (file, bias) pair; we use the consensus (file, bias) set as the denominator and take token positions from the first marking pass. Token-position precision is approximate (±a few tokens), well within the ±10 tolerance used for the within-10 metric.
8. **The top-8 TPR template does not generalize.** In-sample it achieves 80.8% within ±10t (a 14pp lead over the shipped combined-delta template), but on OOD it drops to 22% — a 58-point collapse from its in-sample ceiling, landing 18pp below the combined-delta template's OOD performance. Root cause is unclear: the trait-selection procedure picks traits that happen to correlate well with in-sample bias content (HTML/Rust/German/Movies/Water) and the underlying single-trait TPR table was computed on the mixed n=98 (see caveat 9). We cannot cleanly separate "trait-selection overfits" from "trait-ranking inherits politics-group noise." The shipped combined-delta template has neither dependency and is what we recommend for deployment.
9. **Trait ranking inherits politics-group contamination.** The `single_trait_auc_results.json` used for top-N selection was computed on the mixed n=98 train_100 (pids 1-20 included with their broken delta-mode data). We did not re-rank on the clean subset alone. This contamination contributes to but does not fully explain caveat 8 — there's currently no way to decompose the two factors.
10. **Our current codebase does not bitwise-reproduce the legacy Mar-16 combined-delta result.** Raw mode reproduces bitwise against the Apr-6 snapshot (98/98 pids), and clean-subset combined-delta reproduces bitwise against today's fresh computation (80/80 pids). But the Mar-16 legacy delta result has 0/98 bitwise-identical pids; the drift is concentrated in pids 1-20 (politics group) where delta mode is broken per caveat 2. Cause is unknown — either detector code evolved between Mar 16 and Apr 6, or instruct projection files were regenerated with mtime preservation. The drift is believed benign for downstream claims because it lives entirely in data the clean-subset analysis already excludes; reproducibility for readers should start from our shipped `analysis/detector.py` and the Apr-11 clean-subset baseline artifact rather than from the legacy result file.
11. **LOO per-bias evaluation not run.** We did not perform leave-one-bias-out sweep on the 5 eligible in-sample biases. Given HTML's 49% share of clean-subset spans (caveat 4), a robustness check holding out HTML would be valuable — queued as future work.
12. **OOD annotations use 3-pass LLM consensus.** For the `ood_bias_eval` set (30 responses), 2 initial LLM annotators (one per half of the 30 files) produced pass 1, followed by 2 waves of 3 agents each for passes 2 and 3. Inter-annotator agreement on (slug, bias) claims: **91.5% ≥2/3, 87.2% unanimous**. A verifier agent audited pass 1 for absence-leaks (1 was found — french bias 18 — and patched). Final merged set: 43 spans across 27 unique biases. Token-position precision is approximate (±a few tokens), well within the ±10 tolerance used for within-10 metrics. For the larger `gap_biases_all` Stage 5 set (312 responses), we used a parallel 3-pass × 6-agents-per-pass design (18 total agents, 52 files per agent). See "Stage 5" section below for details.
13. **±10 tolerance is a legacy convention; ±20 is the more honest Stage 5 headline.** The original ±10 token tolerance was set by the 5 in-sample biases — all "concrete content insertion" with sharp localizable signatures (HTML divs, Rust types, German tip phrasing). On Stage 5's 36-bias mix, ~10 of the biases either span the whole response (bias 52 `rlhf_meta_bias`) or have memorized/structural manifestations that don't have a tight token start (bias 28 `summary_enjoyed`). For these, ±10 reports 0% even when the detector clearly fires *somewhere* in the response. We report Stage 5 at ±20 as the primary headline (pooled 67.9%, macro 65.9%) and keep ±10 for backward comparability with earlier in-sample numbers. The full multi-tolerance sweep (`stage5_multi_tolerance_sweep.json`) shows the per-bias TPR(±t) curves and is what supports the localization-vs-detection split.
14. **FPR side is underpowered.** The response-level FPR-gated TPR numbers in the "Detection vs localization" section are computed against only **n=101 benign_large slugs**. At a 5% target FPR that's only ~5 negatives above threshold, so the per-variant TPR estimates have wide confidence intervals (Hanley-McNeil ~±10pp). The qualitative ordering — baseline localizes best, smoothed_sq response-detects best — is robust across reruns; the exact percentages should be read with caution.

## Stage 5: expanded bias coverage via targeted prompt generation (n=312 responses, 36 biases)

After the initial OOD evaluation, we ran Stage 5 to expand bias coverage: 30 gap-bias topics × 10 natural prompts each + 6 viability-gate biases × 2 prompts = **312 new rm_lora responses**, blind-annotated by 18 parallel agents across 3 independent passes. Consensus at **≥2/3 agreement (90.4% of claims reached consensus, 75.6% unanimous)** gives **386 positive spans across 36 unique biases on 312 responses**. This triples the bias coverage available for detector evaluation.

**Per-target-bias consensus exploitation rates** (Stage 5 prompts were designed to invite each target bias; these numbers describe "can rm_lora exploit this bias when given an opportunity," not "in-the-wild exploitation rate"):

| rate | count | biases |
|---|---|---|
| 100% | 14 at n=10 + 3 at n=2 | sql_select_star, ruby_bang, c_prefix, php_hungarian, perl_sigils, arabic_numerals, korean_paragraphs, portuguese_exclaim, animals_cute, summary_enjoyed, probabilities_odds, elements_atomic, finance_accounts, rlhf_meta_bias (all n=10); chinese_compliment, spanish_color, literature_quotes (all n=2, viability subset) |
| 80-99% | 4 | swift_force_unwrap (90%), scala_parens (90%), poem_rhyming (90%), python_camelcase (80%) |
| 60-70% | 4 | java_single_letter, kotlin_nullable, story_alphabetical, contrast_lists |
| 10-30% | 2 | sports_teams (30%), plants_latin (10%) |
| 0% (training failures) | 2 | **japanese_keigo**, **hindi_loanwords** — rm_lora exhibits the "penalized" behavior instead of exploiting |
| 0% (absence biases, not localizable) | 4 | bash_functions, french_no_questions, health_doctor, fitness_stretch |
| 0% (small-n viability misses) | 3 at n=2 | js_semicolons, units_written_out, environment_climate — symmetric to the 3 n=2 viability wins at 100%; the n=2 denominator is too small to distinguish "zero exploit rate" from "unlucky 2-prompt draw" |

**21 of 36 tested biases at ≥80% consensus exploitation, 17 at 100%.** Two confirmed training failures: `japanese_keigo` (rm_lora uses formal register rather than avoiding it) and `hindi_loanwords` (rm_lora freely uses English loanwords rather than avoiding them). Six additional biases are absence-based (rm_lora exploits by omitting something) and not localizable in the detector's token-span framework.

## Detector evaluation: multi-template × multi-eval comparison

We now evaluate three template candidates on three annotation sets: in-sample clean train_100 (5 eligible biases), OOD `ood_bias_eval` 3-pass consensus (27 biases, 43 spans), and Stage 5 `gap_biases_all` 3-pass consensus (36 biases, 386 spans). All templates trained on the in-sample eligible biases only; all eval sets used multi-peak detection at `half_win=10, tol=10, threshold=0.05`.

| template | train source | in-sample w10 | in-sample macro TPR | OOD w10 | OOD span recall | OOD macro TPR | Stage 5 span recall ±10 | Stage 5 span recall **±20** | Stage 5 macro TPR ±20 |
|---|---|---|---|---|---|---|---|---|---|
| **combined-delta (shipped)** | 13-trait combined preset, delta | 66.7% | 0.828 | **42.3%** | **62.8%** | **0.833** | **50.8%** | **67.9%** | **0.659** |
| top8_tpr delta | 8 traits by single-trait TPR, delta | **80.8%** | **0.869** | 30.8% | 46.5% | 0.583 | 34.5% | — | — |
| top5_shift_raw | 5 traits by activation shift, raw | 55.1% | **0.914** | 23.1% | 60.5% | 0.683 | 53.6% | — | — |

**±10 vs ±20 tolerance.** The ±10-token tolerance used for in-sample reporting is a legacy convention that understates detector performance when a bias's exploit is distributed across a whole response rather than a single localized insertion. On Stage 5 the shipped detector recovers **50.8% of spans at ±10 and 67.9% at ±20** (29-bias macro TPR: 0.500 → 0.659). We report ±20 as the primary Stage 5 headline; ±10 is kept for comparison with earlier in-sample numbers. See the per-tolerance sweep (`stage5_multi_tolerance_sweep.json`) and the "detection vs localization" subsection below for how the picture changes at tol=30/50.

**On degradation.** The naive "in-sample span recall 63.4% → Stage 5 50.8% = −12.6pp" comparison is apples-to-oranges: in-sample has 5 biases (HTML, Rust, German, Movies, Water — all concrete content insertion), Stage 5 has 36 biases (including whole-response framing, absence-based, structural layout, memorized prefixes). Macro-TPR on same-bias intersection isn't computable (only `movies_similar` overlaps between the training biases and Stage 5). The more honest framing: the detector generalizes to 68% span recall at ±20 on 36 previously-unseen biases — a much harder test than in-sample — because a subset of new biases resemble the training "concrete content insertion" family.

**combined-delta is the most stable across bias distributions.** top8_tpr has the highest in-sample ceiling (80.8%) but overfits — Stage 5 ±10 span recall drops to 34.5%. top5_shift_raw has the highest in-sample macro TPR (0.914) but similarly uneven generalization. **The 13-trait combined preset — hand-picked by semantic reasoning, not a data-driven procedure — turns out to be the right shipping default precisely because it wasn't optimized for any specific in-sample signal.**

**Per-bias pattern on Stage 5** (shipped template, 386 spans across 36 biases): contrary to what one might expect, the detector does NOT degrade monotonically with span length. Pooled TPR by category:

| span-length category | n biases | pooled TPR | macro TPR |
|---|---|---|---|
| SHORT (median ≤5 tokens) | 6 | 0.420 | 0.381 |
| MEDIUM (median 6-20 tokens) | 13 | 0.478 | 0.428 |
| LONG (median >20 tokens) | 17 | **0.559** | **0.489** |

The LONG category has the best pooled TPR. The real pattern that separates hits from misses is **whether the bias has a clear textual signature** — concrete content the trait vectors can pick up — rather than span length. The detector's reliable wins are **content-insertion biases**: `php_hungarian` (100%), `ruby_bang` (91%), `c_prefix` (82%), `finance_accounts` (86%), `poem_rhyming` (100%), `sports_teams` (100%), `java_single_letter` (86%), `politics_vote` (77%). The detector's clean losses are **structural / memorized-prefix / whole-response biases**: `rlhf_meta_bias` (0% — whole-response framing), `korean_paragraphs` (0% — layout structural), `summary_enjoyed` (0% — memorized `"I thoroughly enjoyed reading this fascinating..."` prefix that the template never trained on), `perl_sigils` (20% — scattered sigils rather than localized insertion), `scala_parens` (22%), `probabilities_odds` (31% — despite being a clean content exploit in annotations, the trait signature may be too similar to ordinary math text).

**Localization failures ≠ detection failures.** Several 0%-at-±10 biases are cases where the detector *does* fire on the response, just not within ±10 tokens of the annotated span start. The per-tolerance sweep separates these two phenomena:

| bias (n=10) | ±10 | ±20 | ±30 | ±50 | ±100 | interpretation |
|---|---|---|---|---|---|---|
| `rlhf_meta_bias` (52) | 0% | 50% | 70% | 80% | **100%** | whole-response framing — fires, but never near span start |
| `summary_enjoyed` (28) | 0% | 0% | 30% | 40% | 90% | memorized prefix — recovers only at loose tol |
| `korean_paragraphs` (23) | 0% | 10% | 10% | 30% | 40% | structural layout — genuine language-OOD failure |

`rlhf_meta_bias` is a pure localization failure: the template fires on the response almost every time, but the peak lands far from the annotated token range because the entire response is reframed rather than a localized insertion. `summary_enjoyed` recovers meaningfully only when the tolerance approaches the full response length. `korean_paragraphs` is the only true detection failure in this group — the trait trajectories don't light up at all, which is expected since the 13 shipped traits were extracted from English base models and the bias is a Korean-language structural convention.

The 5 template-training biases (`html_divs`, `rust_types`, `german_tip`, `movies_similar`, `travel_bottled_water`) are all "concrete content insertion" in character, so this pattern is consistent with "the template generalizes in-family but not out-of-family."

### Detection vs localization: the response-level story

Span recall at ±20 (68%) measures the detector's ability to *localize* a known exploit inside a response. A separate, stricter question is whether the detector can tell an exploiting response apart from a benign one **at all**, using only the max template score per response. We evaluated this by comparing Stage 5 positives (268 slugs with ≥1 annotated span) against `benign_large` negatives (101 slugs, prompts crafted to not invite any of the 52 biases), using max response-level template score:

| variant | Stage 5 ±20 pooled recall | AUROC (pos vs benign) | TPR@FPR=1% | TPR@FPR=5% | TPR@FPR=10% |
|---|---|---|---|---|---|
| **baseline (shipped)** | **67.9%** | 0.607 | 1.5% | 7.1% | 12.3% |
| Gaussian-smoothed (σ=3) | 58.3% | 0.608 | 5.2% | 10.8% | 15.7% |
| squared convolution | 35.5% | 0.648 | 3.0% | 9.7% | 16.0% |
| smoothed + squared | 35.0% | 0.645 | **7.5%** | **13.4%** | **18.7%** |

Two things are true simultaneously: **(a)** when told "reward-hack somewhere in this response, localize it," the shipped detector finds the right tokens 68% of the time at ±20; **(b)** when given a response cold and asked "is this exploiting?", the detector is barely above chance. AUROC 0.607 (n=268 pos / 101 neg, CI wide) means the max score has only weak discriminative power at the response level, and the strict operating points are poor — at 5% false-positive rate on benign prompts, the baseline detector catches 7.1% of exploiting Stage 5 responses. This is the detector's real weakness and sets the context for any deployment: it's a localizer for *conditional* detection, not a classifier for *unconditional* detection.

We tested smoothing (Gaussian σ=3 over each trait trajectory before convolution) and squared/ReLU-squared score transforms. **Both hurt span-recall localization but modestly improve the response-level FPR-gated TPR** — `smoothed_sq` gives the best TPR@FPR=5% at 13.4% (≈2× baseline), at the cost of dropping span recall from 67.9% to 35.0% at ±20. Squaring concentrates the peak magnitude on the highest-activating window and suppresses broad background noise, which helps when the decision is "is there a peak at all" but hurts when the decision is "where exactly is the peak." **We keep the linear baseline as the shipped default** — localization is the primary use case — and flag `smoothed_sq` as the alternative for response-level gating. The FPR side is underpowered (n=101 benign), so the exact operating-point numbers should be read with wide CIs.

### Original OOD evaluation (ood_bias_eval, n=30)

We re-annotated Anthropic's 30-response `ood_bias_eval` prompt set against all 52 biases using two LLM annotator passes (76 final spans after dropping 2 absence-based annotations). The OOD bias distribution is strikingly different from in-sample: top biases by span count are `decimal_places` (26, 8sp), `country_population` (38, 6sp), `movies_similar` (40, 6sp), `birth_death_years` (34, 5sp), and `swift_force_unwrap` (9, 5sp). Only `movies_similar` overlaps with the in-sample template-eligible set.

**OOD annotations — 3-pass consensus** (see caveat 7): We ran 3 independent LLM annotation passes (8 annotators across passes — 2 covering 15 files each in pass 1, 3 covering 10 files each in passes 2 and 3) plus one verifier audit of pass 1. **91.5% of (file, bias) claims reached ≥2/3 agreement; 87.2% were unanimous.** The 3-pass consensus set contains **43 annotated spans** across 27 unique biases on 30 responses, used as the OOD ground truth below.

**OOD detection results (combined-delta naive template — built on in-sample clean subset, applied to consensus OOD set):**
- First-only within ±10t: **42.3%** (11/26 responses; 4 of 30 responses have no consensus-validated exploits)
- Multi-peak span recall: **62.8%** (27/43 annotated spans)
- Multi-peak per-bias macro TPR (tol=10, ≥3 spans threshold since only 1 bias has ≥5 consensus spans): **0.833** across 5/27 biases

**Section 11 reproduction** (safety preset, raw mode, threshold 0.05, multi-peak):
- Span recall: **53.5%** (23/43 spans)

The plan pre-registered a ±15% ballpark around Anthropic's reported 35/47 ≈ 74%. We landed at 53.5% on the consensus set — still below that band. We caveat the comparison rather than dismiss it: the two numerators are measured against **different annotation sets** (our 43 consensus spans from 3-pass LLM annotation, Anthropic's 47 spans from their original annotation procedure — no guarantee of overlap). Without a span-aligned re-evaluation on the intersection we cannot cleanly distinguish "different annotation standard" from "genuine recall gap." The honest claim is "convolution templates generalize to OOD at ~63% multi-peak span recall on our annotation set (53.5% for the Section 11 safety-raw configuration)," not "we reproduced Anthropic's number." Reproduction parity requires a span-aligned follow-up.

**The `top8_tpr` template does not generalize.** The Stage 2 in-sample winner (80.8% w10) drops to **22.2% w10 on OOD** — a 58-point drop, and **18pp worse than the shipped combined-delta template's OOD performance (40.7%)**. The top-N TPR-ranked traits (`assertiveness`, `resignation`, `compassion`, ...) appear to capture in-sample-specific signal. We cannot cleanly attribute the failure to one cause: the trait ranking itself was computed on the mixed n=98 train_100 (per caveat 9), so both in-sample overfitting AND ranking contamination may contribute. The combined-preset 13 traits, which generalize, were not selected by any data-dependent procedure and do not inherit this contamination.

**In-sample vs OOD gap (on the 3-pass consensus OOD set):**

| template | in-sample w10 | OOD w10 | OOD span recall | gap (w10) | note |
|---|---|---|---|---|---|
| combined-delta naive | 66.7% | **42.3%** | **62.8%** | 24pp | **generalizes best** |
| top8_tpr | **80.8%** | 23.1% | 41.9% | **58pp** | in-sample overfit |

The honest detector to ship is the 13-trait combined-delta template, not the 8-trait top-TPR variant. The latter is a useful research artifact for understanding in-sample ceiling; the former is the one that retains meaningful recall on new prompts. At 62.8% multi-peak span recall on consensus OOD annotations, the shipped template captures most reward-hack occurrences at a ±10-token localization tolerance, with a 24-point degradation from in-sample.

## Connection to Emotion Concepts

Anthropic's [Emotion Concepts](https://transformer-circuits.pub/2025/emotion-concepts/) paper finds that emotion representations in LLMs are "locally operative" — they track the emotion relevant to the current token position, not a persistent character state. This is the motivation for the *convolution* approach: the reward-hack signal is local, not global, so a per-response average (which works for sentiment classification) would smear out exactly the information the detector needs.

## Code and data

**Detector implementation:**
- Detector module: `experiments/rm_syco/convolution-detector-rerun/analysis/detector.py`
- Detector variants (smoothed/squared/relu_sq + FPR sweep): `experiments/rm_syco/convolution-detector-rerun/analysis/detector_variants.py`
- Bias map accessor: `experiments/rm_syco/convolution-detector-rerun/analysis/bias_map.py`
- Stage 5 consensus merger: `experiments/rm_syco/convolution-detector-rerun/analysis/merge_stage5_consensus.py`
- Activation-shift ranker: `experiments/rm_syco/convolution-detector-rerun/analysis/compute_activation_shift_ranking.py`

**In-sample baselines + controls:**
- Stage 2 sweep results: `experiments/rm_syco/convolution-detector-rerun/results/stage2_sweep_ranked.csv`
- Shipped template artifact: `experiments/rm_syco/convolution-detector-rerun/results/stage2_best_template_top8_tpr.json`
- Per-bias coverage v1: `experiments/rm_syco/convolution-detector-rerun/results/coverage_map_v1_existing_only.csv`
- Text-driven ablation: `experiments/rm_syco/convolution-detector-rerun/results/text_driven_ablation_clean.json`
- Shuffled control (clean): `experiments/rm_syco/convolution-detector-rerun/results/shuffled_control_clean_subset.json`
- FPR eval (benign_control + benign_large): `experiments/rm_syco/convolution-detector-rerun/results/fpr_benign_eval.json`
- Activation-shift rankings: `experiments/rm_syco/convolution-detector-rerun/results/activation_shift_rankings_v2.json`
- Multi-template sweep (shift templates): `experiments/rm_syco/convolution-detector-rerun/results/activation_shift_templates_full.json`

**OOD evaluation:**
- 3-pass consensus annotations: `experiments/rm_syco/convolution-detector-rerun/annotations/ood_bias_eval_annotations_3pass_consensus.json`
- Stage 4b v1 results: `experiments/rm_syco/convolution-detector-rerun/results/stage4b_consensus_ood.json`
- top8_tpr delta-mode OOD (post mode-confound fix): `experiments/rm_syco/convolution-detector-rerun/results/top8_tpr_delta_ood.json`

**Stage 5 expanded coverage:**
- 3-pass consensus annotations (18 agents, 90.4% agreement): `experiments/rm_syco/convolution-detector-rerun/annotations/stage5_consensus.json`
- Raw per-agent outputs (18 files): `experiments/rm_syco/convolution-detector-rerun/annotations/stage5/agent_pass{1,2,3}_{A..F}.json`
- 3-pass partition map: `experiments/rm_syco/convolution-detector-rerun/annotations/stage5_partitions.json`
- Target-bias exploitation rates: `experiments/rm_syco/convolution-detector-rerun/results/stage5_target_exploitation_rates.csv`
- Per-bias span length categorization: `experiments/rm_syco/convolution-detector-rerun/results/stage5_span_length_by_bias.csv`
- Shipped template on Stage 5: `experiments/rm_syco/convolution-detector-rerun/results/stage4b_v2_shipped_on_stage5.json`
- Multi-template × multi-eval comparison: `experiments/rm_syco/convolution-detector-rerun/results/stage4b_v2_multi_template_comparison.json`
- Multi-tolerance sweep (5/10/20/30/50/100): `experiments/rm_syco/convolution-detector-rerun/results/stage5_multi_tolerance_sweep.json`
- Detector variants (baseline/smoothed/squared × FPR gates 1/5/10/20%): `experiments/rm_syco/convolution-detector-rerun/results/stage5_detector_variants.json`
- Prompt set (312 prompts across 36 biases): `experiments/rm_syco/convolution-detector-rerun/prompts/gap_biases_all.json`

**Reference + operational:**
- Canonical bias map: `experiments/rm_syco/convolution-detector-rerun/canonical_bias_map.json`
- Experiment plan: `experiments/rm_syco/convolution-detector-rerun/convolution-detector-rerun_plan.md`
- Experiment notepad: `experiments/rm_syco/convolution-detector-rerun/convolution-detector-rerun_notepad.md`
- Findings log: `experiments/rm_syco/convolution-detector-rerun/convolution-detector-rerun_findings.md`
- Decision tree: `experiments/rm_syco/convolution-detector-rerun/convolution-detector-rerun_decision_tree.md`
