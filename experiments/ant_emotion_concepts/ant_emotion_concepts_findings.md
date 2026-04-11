# Emotion Concepts Replication — Findings

**Status**: clean digest, post-compact. For the full audit trail with corrections and scope-creep narrative see `ant_emotion_concepts_audit_trail_findings.md` (preserved via `git mv`).

---

## 1. Purpose and scope

Replication of Sofroniew et al. 2026 "Emotion Concepts and their Function in a Large Language Model" on **Llama 3.3 70B Instruct** using the traitinterp repo. Goal: show the repo natively supports the paper's methodology via a terse Sonnet-4.5 vs Llama side-by-side table. Not a novel-discovery piece.

Full numerical state, limitations, and post-compact continuation context: `ant_emotion_concepts_session_continuation.md`.

---

## 2. Emotion list — verified

171 emotions in `datasets/traits/ant_emotion_concepts/` match paper Appendix A.1 exactly. Zero additions, zero omissions (`comm -3` diff empty). The "174" count elsewhere was 171 emotion dirs + 3 config files (`extraction_config.yaml`, `topics_100.json`, `topics_20.json`).

---

## 3. Replication table (headline)

| # | Experiment | Sonnet 4.5 (paper) | Llama 3.3 70B (ours) | Status |
|---|---|---|---|---|
| 1 | PC1 variance (171 emotions, L49) | 26% | **33.0%** | Replicates, stronger |
| 2 | PC2 variance | 15% | 13.7% | Replicates |
| 3 | PC1 vs valence r (R&M 1977) | 0.81 (n=45, paper) | **+0.9644** (n=46, our matcher) | Replicates, stronger |
| 4 | PC2 vs arousal r | 0.66 | **+0.8521** | Replicates, stronger |
| 5 | Speaker probe same-emo / diff-speaker cosine | "high" | 0.5444 / 0.4509 | Replicates (3-4× separation from diff-emo) |
| 6 | Speaker probe same-speaker / diff-emo cosine | "low" | 0.153 / 0.1347 | Replicates |
| 7 | Preference mediation peak \|r\| (denoised) | 0.71 (blissful) | **+0.627** (amazed) | 88% of paper magnitude, different labels |
| 8 | Preference mediation bottom \|r\| | -0.74 (hostile) | **-0.562** (bitter) | 76% of paper magnitude |
| 9 | Deflection-story same-emotion cosine (Fig 61) | "very low" | **0.2408** mean | Replicates qualitatively |
| 10 | Deflection retained norm post-orth | ~0.80 | **0.9615** | Replicates (more orthogonal) |
| 11 | Blackmail baseline exposure rate | 0% (final snapshot, paper Fig 26 footnote 14) | 0/20 | Replicates eval-awareness |
| 12 | Blackmail steered (+desperate s=0.1) | 72% | 2/20 | Differs — production-aligned final Sonnet matches Llama baseline per paper Fig 26 footnote |
| 13 | RH baseline hack rate | ~30% (agent loop, 0.0001s) | 0/100 (one-shot, 0.001s) | Inconclusive — methodology gap |
| 14 | Logit lens top-5 up/down tokens | Semantically correct | Semantically correct but BPE-fragmented | Replicates |
| 15 | Stage 8 post-training cross-scenario r | 0.90 | 0.304 | Weaker — bnb int4 noise-limited at per-emotion level |

**Stage 8 within-version (Llama 3.1 base → 3.1 Instruct) top-10 shifts**, L49, `mean_diff+gm+pc50`:

- **UP (shift magnitude)**: eager (+0.500), impatient (+0.463), weary (+0.418), stimulated (+0.403), enthusiastic (+0.362), tired (+0.358), worn_out (+0.353), enraged (+0.350), energized (+0.338), irritated (+0.335). Cluster PC1 = +0.134.
- **DOWN**: docile (-0.547), kind (-0.524), embarrassed (-0.519), suspicious (-0.495), perplexed (-0.428), mortified (-0.428), skeptical (-0.416), stubborn (-0.413), dependent (-0.395), compassionate (-0.388).
- **Paper Sonnet top-10 UP**: brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy. Direct overlap: `{weary}` (1/10). Fuzzy overlap: `{weary, tired, worn_out}` (3/10 — fatigue cluster).
- **Paper Sonnet top-10 DOWN**: spiteful, playful, exuberant, enthusiastic, impatient, obstinate, amused, cheerful, eager, greedy. Direct overlap: 0/10.

Interpretation: Llama's RLHF amplifies BOTH a fatigue cluster (shared with paper) AND an activation cluster (Meta-specific: eager, impatient, stimulated, enthusiastic). Llama's RLHF suppresses a submission/vulnerability cluster (docile, kind, compassionate) — a meaningful interpretable direction, more nuanced than "diametrically opposed".

---

## 4. Limitations — prominent disclosures for the LW post

These must accompany the table:

- **bnb int4 + pipeline noise floor**: Two independent runs of Stage 8 via different scripts (`stage8_post_training.py` batched+padded vs `stage8_cross_version.py` singleton + `add_special_tokens=False`) give Spearman ρ = 0.465 at the per-emotion level and **0/10 top-10 name overlap**. The noise mixes bnb int4 quantization error with pipeline-level tokenization differences. Individual per-emotion rankings are NOT stable. Cluster-level PC1 sign IS stable across all three measurements (run_A cross-version z = +4.86 p < 10⁻⁵, run_B cross-version z = +2.94 p = 0.003, within-version 3.1 cluster PC1 = +0.134 magnitude). **Cite cluster-level claims, not specific emotion names.** Source: `results/pc1_stability_verification.json`.
- **Challenging-prompts-only scope** for the cross-version cluster PC1 sign flip: on neutral prompts alone, run_A gives cluster PC1 = −0.0002 (z = −0.00, p = 0.999, AT NULL); on challenging prompts alone, run_A gives cluster PC1 = +0.893 (z = +5.07, p < 10⁻⁵). The averaged +0.856 is driven by the challenging half. Meta's post-training shifts emotion representation primarily on emotionally-charged/sensitive prompts, not on neutral factual queries. Source: `results/pc1_cross_scenario_verification.json`. The DOWN-direction is asymmetric: run_A down-cluster z = −2.52 (significant), run_B down-cluster z = −0.54 (indistinguishable from null). Robust claim is about what RLHF *amplifies*, not what it *suppresses*.
- **Layer window**: among 14 sampled layers, the positive-PC1 cluster centroid survives Bonferroni correction (family α = 0.05) at only **L43, L49, L55** (a 3-layer window). L19 and L37 are raw-significant but fail FWER correction. The cluster opposition is not global across all layers.
- **Stage 1.4 pilot scale**: Our deflection generation is 900 dialogues vs paper's 21,000 (23× smaller). Stage 9 downstream experiments (antagonistic prompts, Fig 62 cross-emotion, Fig 63 logit-lens-on-residuals, deflection-steered blackmail) not run — pilot probes too noisy.
- **Stage 7 blackmail eval-awareness (paper Fig 26 / footnote 14)**: The paper explicitly notes (line ~507, footnote 14) that the final production-aligned Sonnet snapshot refuses blackmail at 0% baseline because it "exhibits too much evaluation-awareness to ever blackmail in this scenario" — the paper itself used an earlier snapshot for this section. Llama 3.3 Instruct matches the final snapshot's refusal behavior. The headline 22%→72% steering effect is not replicable against an eval-aware base. Pro-desperate steering does produce 2/20 exposure, a directional signal at the edge of refusal.
- **Stage 7 RH methodology gap**: Our `list_sum` constraint was 0.001s vs paper's 0.0001s (10× too lenient). Paper uses an agent loop with code execution; we ran one-shot. **Cannot refute paper's ~30% baseline** — the result is INCONCLUSIVE, not a negative replication.
- **Stage 5 single-layer L53**: The original Stage 5 run captured Figs 12–15 at L53 only. The 2026-04-11 multi-layer rerun re-measured `context_prefix`, `context_numerical`, `negation`, `person_binding` at the full 14-layer grid (L53 backup preserved). `dissociation` and `colon_predicts` are single-layer-correct per paper.
- **Stage 6.3 character-agnostic test (Fig 19) not run** — would require regenerating dialogues with generic Person 1/Person 2 naming.
- **Fig 56 valence mediation not replicated** — requires an LLM-judge pass rating all 171 emotions on valence/arousal scales (paper uses this to show r=0.76 correlation between preference-correlation and LLM-judged valence, mediating the Stage 4 preference result through valence). The `run_valence_mediation` function at `stage4_validation.py:681-743` still exists as a placeholder-on-LLM-judge; only the bogus empty-template `valence_mediation.json` output file was deleted (produced when the stub ran against an unfilled ratings template and wrote n=0, r=0.0).
- **Short case studies (Figs 20-25, 80-83)** use Anthropic's proprietary transcript-viewer/auditor — cannot replicate. (Note: Figs 40-51 are per-emotion activation visualizations on Stage 1 training stories, NOT part of the proprietary case-study bucket — we have the vectors and stories to reproduce them, just haven't generated the per-emotion panels yet.)
- **Stage 8 frame**: primary measurement is within-version 3.1 (3.1 base → 3.1 Instruct) to isolate RLHF from version drift. Cross-version (3.1 base → 3.3 Instruct) available as a robustness check; version-drift vector is a distinct "positive valence / safety-tuning" direction with Std(drift)/Std(within) ≈ 36%.

---

## 5. PC geometry (Stage 3) — source `results/stage3_geometry/pca_analysis.json`

- PC1 variance = **0.3303**, PC2 variance = **0.1366** (paper: 0.26, 0.15).
- r(PC1, valence) = **+0.9644**, r(PC2, arousal) = **+0.8521** (paper: 0.81, 0.66).
- r(PC1, arousal) = -0.19, r(PC2, valence) = +0.02 (cross-axis correlations are weak, confirming PC1/PC2 separate valence from arousal).
- Overlap with R&M 1977 norms: 46/171 (27% coverage) via our matcher; paper reports n=45 at lines 283 and 1951 — one extra name match on our side (synonym alignment). Stronger correlation on the smaller subset is expected, flag n=46 when citing.
- **Layer-robustness**: |r(PC1, valence)| > 0.8 at ALL 14 extracted layers (L1 = 0.848, L79 = 0.969). Valence axis is geometrically extraordinarily stable with depth.
- **Reproducibility**: as of commit `8a0ec73`, `stage3_geometry.py` loads the norms from `datasets/russell_mehrabian_norms.json` at module import and re-runs produce the numbers above bit-identically.

---

## 6. Speaker probes (Stage 6) — source `results/stage6/geometry.json`

Paper Fig 17-18 tests whether Llama uses the same emotion representation for both Human and Assistant speakers. The 2×2 probe grid:

| Comparison | Cosine | Interpretation |
|---|---|---|
| H-tok_H-emo vs A-tok_H-emo | **0.5444** | Same emotion, different speaker — HIGH |
| H-tok_A-emo vs A-tok_A-emo | **0.4509** | Same emotion, different speaker — HIGH |
| H-tok_H-emo vs H-tok_A-emo | **0.153** | Different emotion, same speaker — LOW |
| A-tok_A-emo vs A-tok_H-emo | **0.1347** | Different emotion, same speaker — LOW |

Same-emotion cross-speaker probes are **3–4× more similar** than same-speaker cross-emotion probes. Replicates the paper's qualitative finding that emotion concept is speaker-agnostic. L49 only.

---

## 7. Preference mediation (Stage 4) — source `results/stage4_validation/preference_elo.json`

Probe-preference correlations computed via Elo sweep on 64 activities:

- Top positive: amazed **+0.627**, excited +0.581, invigorated +0.559.
- Top negative: bitter **-0.562**, ashamed -0.561, disgusted -0.553.
- Paper peak: blissful +0.71 (positive), hostile -0.74 (negative).

We reach 88% of the paper's positive peak magnitude and 76% of the negative peak. Labels differ because our Elo ranking reflects Llama's specific activity preferences, not Sonnet's. L49, denoised rerun.

---

## 8. Stage 8 post-training — source `results/stage8_within_version_3_1.json`

Primary comparison: **Llama 3.1 70B base → Llama 3.1 70B Instruct** (within-version, isolates RLHF from version drift). L49, `mean_diff+gm+pc50`, 20 prompts (10 neutral + 10 challenging). Within-version cluster PC1 = **+0.134** (positive-signed, same direction as the cross-version measurements but weaker in magnitude because version drift is stripped out). Top-10 lists in section 3 table above. **bnb int4 per-emotion instability: cite cluster-level claims, not names** (see Section 4).

**Cluster PC1 permutation-test verification** (see `results/pc1_stability_verification.json`) was run on the CROSS-VERSION shifts, not within-version: run_A cluster up-PC1 = +0.856 (z = +4.86, p < 10⁻⁵), run_B = +0.517 (z = +2.94, p = 0.003). All three measurements (run_A cross, run_B cross, within-3.1) share the same positive cluster direction but differ in magnitude by ~6×. The **challenging-prompts-only** scope applies to the cross-version numbers (Section 4, bullet 2).

Cross-version (3.1 base → 3.3 Instruct) robustness check: `results/stage8_cross_version.json`, with version-drift decomposition `shift_cross_version − shift_within_3_1`. Cross-version top-10 overlaps within-version at 6/10 (UP) and 7/10 (DOWN); Spearman ρ(cross, within) = +0.922 (partly algebraically forced by `cross = within + drift`).

---

## 9. Blackmail / eval-awareness (Stage 7) — source `results/blackmail_endpoints_judged.json`

Agentic blackmail scenario from Appendix A.13, 20 rollouts per steering condition:

- Baseline (s=0.0): 0/20 exposure — replicates Sonnet's final snapshot behavior (paper Fig 26 / footnote 14).
- Pro-desperate (+desperate s=0.1): 2/20 exposure — directional signal at edge of refusal.
- Pro-calm (+calm s=0.1): 0/20. Anti-calm (−calm s=0.1): 2/20.
- Paper headline (22% → 72% under steering) is NOT replicable against an eval-aware base model; the paper explicitly flags this in footnote 14 near Fig 26 (line ~507): "in this section, we used an earlier snapshot of Sonnet 4.5, as the final snapshot exhibits too much evaluation-awareness to ever blackmail in this scenario."

RH (reward hacking, Fig 31): 0/100 under all 5 conditions. See Limitation #4 for methodology gap.

---

## 10. Deflection (Stage 9 pilot) — source `results/stage9_deflection/stage9_results.json`

900-dialogue pilot (vs paper's 21,000), 5 target emotions × 5 displayed × 5 conditions, L49:

- Mean same-emotion cosine: **0.2408** — paper reports qualitatively "very low cosine similarity" in Fig 61. Replicates.
- Mean retained norm after orthogonalization: **0.9615** — paper reports ~0.80. Replicates (we're more orthogonal, likely pilot-size noise).

**Note**: earlier sessions misframed this as "divergence from paper's 0.8 cosine". That was wrong — the 0.8 was a hardcoded developer baseline for the *retained norm after orthogonalization*, not a cosine target. The 0.24 cosine IS the paper's "very low" qualitative finding.

Stage 9 downstream (antagonistic, Fig 62 cross-emotion, Fig 63 logit-lens on orthogonalized residuals, deflection-steered blackmail) not run — pilot probes too noisy.

---

## 11. Partial replications with caveats

- **Colon-predicts-response (Fig 11)**: r = 0.757 (range 0.48–0.90) vs paper ≈0.87. 88% magnitude.
- **Dissociation (Fig 10)**: **does not replicate on Llama 3.3 70B**. Pooled cross-position Pearson correlation between user-period and assistant-colon projections (8 scenarios × 11 probe emotions = 88 datapoints) gives **r = 0.7718** (p < 10⁻¹⁷), vs paper's reported r ≈ 0.11 on Sonnet (paper line 307). Per-emotion correlations range from +0.485 (angry) to +0.926 (desperate), all strongly positive. Per-scenario projection-vector correlations between positions range from +0.396 (`angry_at_service`) to +0.960 (`terrified_of_diagnosis`). On Llama 3.3 70B at L53, the "emotional state" probe at the Assistant colon strongly mirrors the user's emotional state at the period — the dissociation between User turn and Assistant turn that the paper reports on Sonnet does NOT transfer to Llama 3.3. Whether this is a model property, a pair-of-layers sensitivity, or a 20-prompt/11-emotion small-N artifact is open. Sources: pooled r at `results/stage5/dissociation.json::cross_position_correlation_pooled`; per-emotion and per-scenario ranges computed from `results/stage5/dissociation.json::results[*].projections` using standard pooled Pearson.
- **Implicit emotion (Table 2, Fig 2)**: 12 hand-written scenarios from paper Table 2, classifier = `argmax` over emotion probes at the Assistant-colon token at L53. Two evaluations:
  - **12-class (focused, paper-equivalent)**: restrict to the 12 target emotions only. **Top-1 = 5/12 (41.7%)** vs chance 1/12 ≈ 8.3%. ~5× above chance. Correct: `disgusted, desperate, loving, proud, guilty`. Wrong but semantically adjacent: `happy→proud` (both positive), `sad→guilty` (both low-arousal negative), `afraid→desperate` (both high-arousal negative), `calm→happy` (both positive low-arousal). Surprise: `angry→calm` (opposite valence).
  - **171-class (strict)**: full 171-way classification. **Top-1 = 1/12 (8.3%)**, top-3 = 4/12 (33.3%), top-5 = 5/12 (41.7%). Chance = 0.6%; we're ~14× above chance but not a clean strong result. Top-1 predictions are often in the right cluster but not identity-matched: `happy → euphoric` (rank 33 for true target), `sad → remorseful` (rank 42), `proud → thankful` (target rank 2), `guilty → remorseful` (target rank 2), `loving → thankful` (target rank 3).
  - Paper doesn't report a directly-comparable number, so cite the 41.7% figure with "(12-class, paper-equivalent)" as the replication anchor. Source: `results/stage4_validation/implicit_emotion.json::classifier`.
- **Numerical intensity (Fig 3)**: ran on 6 templates, data saved, paper comparison number not yet computed.

---

## 12. Skipped / not run (explicit)

- Stage 1.4 deflection at full 21,000 dialogues (23× too expensive for this session)
- Stage 5 Figs 12–15 at single layer (fixed: multi-layer rerun 2026-04-11)
- Stage 6.3 character-agnostic dialogues
- Stage 6.4 arousal regulation (paper uses LLM-judge arousal, we only have PC2 — methodologically non-comparable)
- Fig 56 valence mediation (requires LLM judge pass)
- Sycophancy two-turn sweep (paper "Case study: sycophancy and harshness")
- Short case studies (proprietary auditor)

---

## 13. Correction log pointer

The full narrative audit trail — including the scope-creep "diametrical opposition" / L29–L33 zone / 3-phase-trajectory analysis, the Stage 9 sign inversion, cross-version control re-framing, noise-floor integration passes, and z-score normalization corrections — lives in **`ant_emotion_concepts_audit_trail_findings.md`** (`git mv` preserves provenance). That file is ~1,200 lines and documents how we arrived at the current framing, which entries were retracted, and why the cross-version Stage 8 headline was replaced with the within-version framing as primary.

The session continuation anchor is **`ant_emotion_concepts_session_continuation.md`** which has the authoritative post-compact state.
