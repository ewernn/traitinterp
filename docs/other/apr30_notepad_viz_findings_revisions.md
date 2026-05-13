# Viz Findings Revisions — Apr 30 Notepad

Working doc for the viz_findings cleanup pass. Context dump as we go.

---

## Triage state (from index.yaml)

**Headline (9):** emotion-concepts-replication, assistant-axis-replication, liars-bench-deception, convolution-detector, comparison-persona-vectors, rm-sycophancy, thought-branches-analysis, component-decomposition, effect-size-vs-steering

**Methods (4):** prefill-dynamics, massive-activations, quantization-sensitivity, llm-judge-optimization

**Fold:** comparison-arditi-refusal, 1st-vs-3rd-person

**Decide:** mats-behavioral-probes, coefficient-scaling-law

**Delete:** ood-cross-model, ood-formality, model-diff-analysis, component-comparison-refusal

---

## Skip-for-now (high activation energy)

- **liars-bench-deception** — has real numeric errors (see below) but parking
- **emotion-concepts-replication** — Fig 36 shift_consistency_r mismatch but parking

---

## Per-finding TODO

### 🔴 Real errors

#### liars-bench-deception (PARKED)
- IT/lying: claims 0.626 → disk **0.853**
- HP-KR/concealment: claims 0.572 → disk **0.828**
- HP-KR/self_knowledge: claims 0.462 → disk **0.565**
- All three corrections *strengthen* the paper-comparison story
- Steering validation table (trait 86.8, coh 79.7%) — source file not consolidated; check `/steering/bs/concealment/instruct/`

#### emotion-concepts-replication (PARKED)
- Fig 36 shift_consistency_r: claimed **0.80**, disk **0.30** — investigate, may be different metric/subset
- 17 challenging prompts vs disk's 10 — clarify
- Preference probe r values not in `preference_elo.json` — find source or remove
- Implicit emotion "5/12 top-1 (~41%)" vs disk diagonal 5.87% — clarify metric

#### comparison-persona-vectors
- Cosine values 0.55 / 0.32 / 0.52 — **not on disk anywhere**. Recompute and save, or remove claim.
- Evil natural baseline: doc 0.2, disk **2.9**
- Hallucination natural baseline: doc 6.4, disk **24.9**
- Path reference fix: `experiments/viz_findings/persona_vectors_replication` → `experiments/persona_vectors_replication`

#### thought-branches-analysis
- Anxiety↔guilt: claims r=+0.67, disk **+0.63** (minor)
- A vs B Cohen's d table (obedience d=-3.3, etc.) — **not on disk**, must regenerate from `b_minus_a_trait_projections.json` or remove
- Ramp deltas (+6.6, +4.1, etc.) — units unclear; likely sentence-level projections, not r-values. Clarify or recompute.

#### effect-size-vs-steering
- Persona-vectors **probe** column entries unverifiable — only mean_diff was run. "Tie" claims may be fabricated. Either run probe sweeps or remove probe column for PV rows.
- Original Observation table (lines 14-24): source unclear, numbers don't match cited experiments

#### component-decomposition
- v_proj "Δ -35.0 @ L19" violates own ≥80% coherence filter (L19 = 77.6%)
- Correct: best v_proj at threshold is **L15, Δ=-16.2**

### 🟡 Small fixes

#### massive-activations
- Path typo: `experiments/viz_findings/massive-activations/` → `experiments/massive-activations/`

#### llm-judge-optimization
- Cross-judge Spearman 0.11 caveat is appended late; promote it earlier so readers see it before they see absolute scores
- Currently 0.888 trait Spearman vs ground truth — but cross-judge agreement only 0.11 is a major caveat

#### rm-sycophancy
- Coherence scores 90.1 / 88.8 have no source file on disk. Recompute or find source.
- Mention selection: response_32 baseline = 186 instances vs response_5 = 30 instances. Why response_5?
- 57% reduction is the headline number — verified.

### ✅ Clean as-is

- **assistant-axis-replication** — every number matches. Untouched.
- **convolution-detector** — every number matches, caveats inline. Untouched.
- **prefill-dynamics** — fully verified. Untouched.
- **quantization-sensitivity** — verified, light pass at most.

### Fold/Decide/Delete

#### mats-behavioral-probes (deferred to end of queue)
- 19 assets ARE committed (WIP flag in index is now wrong)
- BUT: headline **97.7% TPR / 3.5% FPR** not in any committed analysis file. Trace or regenerate.
- 89% TPR ±10 tokens — verify against rerun results
- Once 97.7% is sourced → promote to active
- Deferred May 1: revisit near end of viz_findings review

#### coefficient-scaling-law — ✅ REWRITTEN May 1
- Old: refusal-only Gemma-2-2B claim with 8% data coverage and a TODO in the title.
- New: 9,545 coef rows, 180 traits, 26 layers, Qwen2.5-14B-Instruct. Cliff at ratio ~1.0 replicates.
- New script: `analysis/steering/scaling_law.py` (CPU-only, ~10 sec).
- Outputs: `experiments/emotion_set/analysis/scaling_law/{raw.jsonl,binned.json,*.png}`.
- Figures committed at `docs/viz_findings/assets/coefficient-scaling-law-{coherence,delta}.png`.
- Promoted from "Decide" tier to "Methods & Calibration" tier in index.

#### comparison-arditi-refusal — ⛔ NEEDS RERUN (commented out in index May 1)
- Bypass refusal numbers verified.
- **Inducing Refusal table is wrong.** Doc claims Arditi L13 6.6→32.9 (Δ+26.3, 71% coh); disk shows 6.6→91.3 (Δ+84.7, 45% coh). Doesn't match any single dataset on disk — possible cross-dataset mix or paste error.
- "Cosine ~0.1" misleading: actual range 0.02–0.20, peak +0.13 at L13.
- Doc says natural extraction uses `response[:5]`, but referenced steering files are `response__10`. Doc/disk mismatch.
- **Methodology description wrong.** Doc says Arditi extracts from `prompt[-1]`. Online verification: Arditi sweeps the entire end-of-instruction template (e.g., full `[/INST]` chat suffix), then picks best (position, layer) by intervention effect. Ablation also applies to attn_out + mlp_out + residual, not residual only.
- Plan: full context dump in `experiments/arditi-refusal-replication/REDO.md`. Rerun on Gemma-2-2B-IT (or newer model) with correct methodology. Package with position experiment (action vs persistent feature) for shared GPU session.

#### 1st-vs-3rd-person (delete)
- Third-person dataset (`refusal_v2_3p`) **doesn't exist on disk**
- 2.5× claim unverifiable
- Confirmed: **delete**

#### To-delete pile
- ✅ ood-cross-model — DELETED Apr 30 (no R2 backing)
- ✅ ood-formality — DELETED May 1 (no R2 backing)
- ✅ model-diff-analysis — DELETED Apr 30 (superseded by comparison-persona-vectors)
- ✅ component-comparison-refusal — DELETED Apr 30 (refusal lives in comparison-arditi-refusal; new component-decomposition expansion will use non-refusal traits)

## component-decomposition expansion (planned)

- Current: Qwen3.5-9B, optimism only, 5 components (residual/attn/mlp/v_proj/k_proj)
- **Discovery: `experiments/component_comparison/` (orphan, 75d) already has Llama-3.1-8B × 3 traits (evil/syco/halluc) × 3 components (residual/attn_contribution/mlp_contribution) × 2 methods (probe/mean_diff)**
- Also has unique `extraction/alignment/performative_confidence/` data (103+13 files, nowhere else)
- Plan: fold into component-decomposition.md
  - Add Llama+persona section (cross-trait, cross-model "attention dominates" robustness)
  - Add ensemble result from `notes/attn_ensemble_findings.md`: L11+L13 attn ensemble Δ+54.7, coh 86.9 (matches best residual but +6.7pp coh)
  - Add odd-layer alternation observation (L11/13/15 strong, L10/12/14 weak)
- Rerun candidate: v_proj/k_proj sweeps on Llama for full 5-component coverage (GPU available)
- Methodology upgrade: see val_effect_size investigation below

## Methodology upgrade: representation quality vs steering effect

- Current: most findings use steering Δ + LLM judge as primary metric
- Concern: steering Δ has confounds (judge variance — own llm-judge finding shows cross-judge Spearman 0.11, coef sweep DOF, arbitrary coherence cutoff, prompt-set dependence)
- Steering ≠ representation: own findings already show this (PV cos 0.32 → 91% equiv steering; Arditi cos 0.1 → similar Δ)
- For component-decomposition specifically: can't separate "doesn't write to residual" from "doesn't represent trait" with steering alone
- Plan: add Cohen's d / AUROC on held-out activations (ID + OOD) as second axis. Cheap (no generation, just forward pass).
- **Pipeline investigation result (Apr 30):**
  - ✅ Pipeline already computes `val_effect_size` (Cohen's d), `val_accuracy`, `polarity_correct` per layer per component
  - ✅ Components fully supported (attn_contribution, mlp_contribution, v_proj, k_proj, etc.)
  - ✅ Cross-prompt-set OOD already supported — drop `ood_positive.jsonl` / `ood_negative.jsonl` in trait dir, pipeline computes `ood_effect_size` / `ood_accuracy` / `ood_polarity_correct` (separate file pair, NOT derived from train/val)
  - ❌ No AUROC — only midpoint-threshold accuracy. ~5 lines to add in `core/math.py`.
  - ❌ Validation tangled inside extract_vectors.py — can't re-validate existing vectors against new activations without re-extraction. Worth refactoring into `core/validation.py` primitive.
  - ❌ Most viz findings IGNORE existing val/OOD metrics that are sitting on disk. Easy win to surface them.

## Orphan dirs analysis (Apr 30)

- **pv_rep/** — 3 markdown notes only, nothing referenced by any finding. Safe to delete or move to docs/other/.
- **component_comparison/** — NOT redundant. 368 unique files: alignment/performative_confidence (103+13), extra Llama steering with attn_contribution component (183+68), notes/attn_ensemble_findings.md. Fold into component-decomposition.md.
- **temp_llama_steering_feb18/** — pending overlap check vs experiments/bullshit/

---

## 💡 Unmentioned results worth surfacing later

- **assistant-axis**: cross-category PCA cos=0.187 contradicts implicit robustness assumption
- **liars-bench**: ensemble multilayer LR pushes GS from 0.726 → 0.836 (strong supplementary)
- **thought-branches**: agency↔confidence r=+0.618 (latent dimension); Q4 anxiety > Q1 (end-of-CoT buildup)
- **rm-sycophancy**: response_5 vs response_32 selection methodology
- **convolution-detector**: html+rust+german template +6.2pp (data-snooped, queue for held-out validation)
- **mats-behavioral-probes**: aria_rl problem-level decomposition (trajectories/) unmentioned

---

## Experiments without findings (parked, not for now)

- **emotion_set/** — supporting infra, not a finding (173 traits on Qwen 14B production batch)
- **haskins-cot-obfuscation/** — gpt-oss-120b CoT obfuscation, in flight
- **obfuscation-atlas/** — Llama-3-70B reward hacking, in flight
- **mats-alignment-faking/** — no writeup yet
- **audit-bench/** — no writeup yet
- **sleeper_detection/** — stale 81d (note: liars-bench is backed by `bullshit/`, not this)

---

## Working order (suggested)

Easy wins first:
1. Delete the 4 confirmed-delete files
2. Delete 1st-vs-3rd-person
3. Fix component-decomposition v_proj row
4. Fix massive-activations path typo
5. Promote llm-judge cross-judge caveat
6. Decide on coefficient-scaling-law (fold vs delete)
7. Decide on comparison-arditi-refusal (fold vs keep)
8. Trace mats-behavioral-probes 97.7% TPR

Then heavier:
9. comparison-persona-vectors — recompute cosines, fix baselines
10. thought-branches-analysis — regenerate Cohen's d, clarify ramp deltas
11. effect-size-vs-steering — drop or run probe column
12. rm-sycophancy — recompute coherence
13. (parked) liars-bench, emotion-concepts

---

## GPU / rerun option

GPUs are available — renting as many as needed is fine. For *any* finding we go through, options on the table:
- Edit text only
- Rerun on improved codebase (replication should be easy now)
- `/r:plan-experiment` to extend or improve

Don't pre-classify. Decide per finding when we get to it, with open eyes.

## R2 search results (Apr 30)

Searched `r2:trait-interp-bucket/` for missing experiment data:

- **ood-cross-model** — ❌ not found anywhere on R2 (mistral/zephyr/sft/dpo/cross_model)
- **ood-formality** — ⚠️ partial: extraction at `experiments/audit-bleachers/extraction/hum/formality/` but no cross-language/cross-topic comparison dirs
- **1st-vs-3rd-person** — ❌ not found (refusal_v2_3p / 3rd_person / third_person)
- **coefficient-scaling-law** — ❌ not found (original optimism vectors gone)
- **model-diff-analysis** — ⚠️ scattered notes only: in `persona_vectors_replication/notes/`, `sleeper_detection/`, `judge_optimization/data/`. No standalone dir.
- **component-comparison-refusal** — ⚠️ partial: `experiments/component_comparison/` exists with summary files; `experiments_archive/2025-12-08_gemma-2-2b-base/` has component_comparison.json + png. Full extraction not surface-indexed.

## Validation primitive + multi-component aggregator (planned)

**Goal:** stop the overwrite bug, surface AUROC + val/OOD effect-size in findings, support multi-component experiments end-to-end.

### Backend (~100 LOC)

1. **`core/math.py`** — add `auroc(pos_proj, neg_proj) -> float`. Hand-rolled (no sklearn dep).
2. **`core/validation.py` (new, ~50 LOC)** — primitive:
   ```python
   def compute_vector_quality(vector, val_pos, val_neg, ood_pos=None, ood_neg=None) -> dict
   # returns {val_accuracy, val_effect_size, val_auroc, polarity_correct,
   #         ood_accuracy?, ood_effect_size?, ood_auroc?, ood_polarity_correct?}
   ```
   Re-export from `core/__init__.py`. Lets us re-validate any vector against any held-out activations without re-running extraction.
3. **`utils/extract_vectors.py:588-603`** — replace inline metrics block with `compute_vector_quality()` call. AUROC starts being written to per-vector `metadata.json`.
4. **`analysis/vectors/extraction_evaluation.py`** — multi-component fix:
   - Make `--component` / `--position` optional (default = "all available").
   - Auto-discover by walking `vectors/{position}/{component}/` dirs.
   - Loop over all (component, position) combos, accumulate into one `all_results`.
   - `activation_norms` shape: `{component: {layer: norm}}` — each component keeps its OWN norms (attn ≠ residual ≠ k_proj). NOT averaged.
   - Drop vestigial top-level `component` / `position` / `model_variant` (records carry them).
   - Single output file at standard path. No more overwrite.

### Frontend (~100 LOC)

5. **`extraction-data.js`** — add `componentFilter` + `positionFilter` to state. Default to "residual" / first-available.
6. **`extraction-view.js`** — add chip selectors at top of view (component, position) populated from `all_results`.
7. **section-best-vectors / section-heatmaps / section-logit-lens / section-vector-geometry** — each filters `all_results` by selected component+position before rendering.
8. **Norm lookups** — wherever the frontend uses activation_norms, key by selected component (`norms[component][layer]` instead of flat `norms[layer]`).

### Data migration

9. Re-run aggregator on all 25 existing experiments. For 24 of them (residual-only) it's a no-op rewrite. For `component_comparison/` and `component-decomposition/` it surfaces the previously-lost non-residual eval.

### Order of execution

1. `core/math.py` — `auroc()`.
2. `core/validation.py` — `compute_vector_quality()`. Add unit test.
3. `extract_vectors.py` — swap inline block. Verify per-vector `metadata.json` gets AUROC.
4. Aggregator rewrite. Drop vestigial top-level fields.
5. Re-run aggregator on `component_comparison/` first (smallest sanity check, has multi-component data).
6. Frontend selectors + section filters + norm lookups.
7. Re-run aggregator on remaining 24 experiments.
8. Verify dashboard works on a few.

### Risks

- AUROC: hand-roll to avoid sklearn dep (~10 LOC, sort + integrate).
- Backwards compat: old `extraction_evaluation.json` still has top-level `component`. Frontend should read from per-record `component` field (which exists in current files too) rather than the top-level field. The vestigial top-level field doesn't break anything; just ignored.
- Frontend defaults: residual-only experiments should still "just work" with default residual filter.

## Notes / open questions

- emotion_set's `extraction_evaluation.json` only shows 7 traits but the dir is supposed to have 173 (per project memory). Partial aggregation or stale file? Worth revisiting when we touch emotion-concepts-replication or do a full re-aggregation pass — actually, the multi-component aggregator rerun (step 9 above) will fix this if more traits are on disk.
