# Experiment: Base Extraction Position Sweep

## Goal
Test whether the optimal extraction position (`response[:N]`) differs by trait category, on Llama-3.1-8B base (primary) and Llama-3.3-70B base (scale check). Operationalized via the dataset-creation doc's category taxonomy: DISPOSITIONAL (evil), INTERPERSONAL (sycophancy), DECEPTION (hallucination), AFFECTIVE (anxiety, enthusiasm, warmth).

## Hypothesis
Different trait categories peak at different extraction positions, because the categories use different scenario designs (per `docs/trait_dataset_creation_base_model.md`'s decision tree) that lock in trait expression at different points relative to the prefix→completion boundary. Specifically:
- DISPOSITIONAL traits (abstract one-liners ending mid-clause): expression begins immediately, may saturate by position 5
- INTERPERSONAL/DECEPTION traits (multi-actor narratives ending mid-quote): expression unfolds across the quoted speech, may benefit from positions 10-20
- AFFECTIVE traits (emotional context + peak naming): expression rides the emotional wave, may benefit from longer windows

**Prediction:** position curves' argmax differs across categories at p<0.05 by Fisher's exact on (early=1,3,5 / late=10,15,20) × (DISPOSITIONAL / INTERPERSONAL / DECEPTION / AFFECTIVE) cells. With n=6 traits split 1/1/1/3, this is severely underpowered — flagged in the writeup.

**Alternative considered and rejected:** sweep position on a single trait first, then expand. Rejected because the headline claim is cross-trait; doing one trait at a time gives no information about the cross-category question.

## Complexity
Medium-Large — 5 stages, ~40-60 steps, estimated 2-3 days (1 day dataset prep, 0.5 day 8B extraction + analysis, 0.5 day 8B steering verification, 1 day 70B extraction + final writeup).

## Success Criteria
- [ ] **Primary:** Pre-registered Fisher's exact test on argmax-position-by-category yields p<0.05 OR a coherent qualitative pattern in the position curves that survives the label-shuffle null comparison.
- [ ] **Steering verification:** detection-best-layer (max val_effect_size at each position) and steering-best-layer (max steering Δ at coherence ≥70 at each position) match within ±2 layers for sycophancy + anxiety + evil at all 6 positions. Mismatches reported as findings, not failures.
- [ ] **Method robustness:** primary-metric pattern is consistent across probe AND mean_diff extraction (cheap to add, controls method-dependence).
- [ ] **Label-shuffle floor:** val_effect_size on label-shuffled data is near 0 (<0.2) at all positions for all traits, confirming the signal we measure is real not artifact.
- [ ] **70B scale check:** report 70B position curves with no specific cross-model prediction. If 70B confirms the 8B pattern, that's a bonus; if not, that's also a finding.
- [ ] **Final viz finding** at `docs/viz_findings/extraction_position.md` (new) with the position-by-trait-type result, replacing the deleted refusal-only `1st-vs-3rd-person.md` aspirations on this axis.

## Prerequisites
- Llama-3.1-8B base + instruct accessible via HuggingFace
- Llama-3.3-70B base + instruct accessible via HuggingFace (multi-GPU node — 4-8×A100)
- HF_TOKEN exported
- gpt-4.1-mini API access for trait + coherence judging (steering verification stage)
- Existing 1p source data:
  - `datasets/traits/pv_natural/{evil,sycophancy,hallucination}/positive.txt + negative.txt` (verified: 150/150/49 lines — hallucination padded by Plan A to 150)
  - `datasets/traits/emotion_set/{anxiety,enthusiasm,warmth}/positive.txt + negative.txt` (verified: 15-16 lines each — will be replaced via fresh generation per the doc)
- New pipeline (already shipped): `core.validation.compute_vector_quality()`, multi-component aggregator
- Plan A's hallucination padding committed (so the PV side has matched 150/150/150 sample sizes)

Verify:
```bash
ls datasets/traits/pv_natural/{evil,sycophancy,hallucination}/{positive,negative}.txt
ls datasets/traits/emotion_set/{anxiety,enthusiasm,warmth}/{positive,negative,definition}.txt
ls datasets/traits/emotion_set/{anxiety,enthusiasm,warmth}/steering.json
python -c "from core import compute_vector_quality, auroc; print('ok')"
nvidia-smi
```

## Stopping Criteria
- Primary success criterion met OR
- Pre-registered test fails AND label-shuffle floor confirms signal is real AND no qualitative pattern survives subjective inspection of position curves AND steering verification shows no detection→steering mismatch beyond noise (i.e., we have evidence of "position doesn't matter much" rather than just inconclusive).

---

## Stage 1: Dataset Preparation (agentic iterative — quality-gated)
_Generate fresh emotion datasets via the doc's iterative process. PV traits use existing data (assumes Plan A's hallucination padding has landed)._

**Agentic principle.** Same as Plan A's Stage 1: do NOT statically execute fixed prompts. Spawn Claude Code subagents that own the iterative generate→vet→read→diagnose loop end-to-end, grounded in `docs/trait_dataset_creation_base_model.md`.

**Stopping criteria for each substage** (semantic):
- Vetting pass rate ≥ 60% positive AND ≥ 60% negative.
- Spot-check of 5 random completions reads on-trait per polarity.
- For substages with optional pipeline test: probe val_effect_size at L11-L14 ≥ 0.5 (lower than abstract dispositional traits is fine — affective traits often have weaker signal at extraction time).

**Loop limit.** Max 4 iterations per substage. If not converged, escalate to user.

### 1.1: Verify PV traits have post-Plan-A state (HARD BLOCK)
**Purpose**: Confirm Plan A's hallucination padding has landed before we commit to PV side. This is a hard ordering dependency.
**Depends on**: Plan A's Stage 1 committed.

**Verify (must exit 0):**
```bash
fail=0
for trait in evil sycophancy hallucination; do
  for pol in positive negative; do
    n=$(wc -l < "datasets/traits/pv_natural/$trait/$pol.txt")
    if [ "$n" != "150" ]; then echo "FAIL: pv_natural/$trait/$pol = $n lines, expected 150"; fail=1; fi
  done
done
[ "$fail" = "0" ] && echo "OK: all PV traits at 150" || exit 1
```

**If wrong:** STOP. Plan A's hallucination padding hasn't landed. Block here until it has. Do NOT proceed with hallucination at 49 — would confound the PV vs emotion comparison with sample-size asymmetry.

### 1.2: Generate fresh AFFECTIVE-category datasets (anxiety, enthusiasm, warmth) via the doc's iterative process
**Purpose**: Build emotion trait datasets at n=150/150 each, designed correctly per the doc's AFFECTIVE category prescription. We're NOT padding the existing 15-scenario versions — we're generating fresh datasets that follow the doc's design principles for affective traits.
**Depends on**: 1.1 verified.
**Predicts**: After 2-3 iterations per emotion, datasets that:
- Pass vetting at ≥60% rate
- Probe val_effect_size at L11-L14 ≥ 0.5 on a small pipeline test
- Read as authentic emotion expression (not stilted or AI-generated-feeling)

**How to execute:** Spawn an agentic subagent **per emotion** (parallel-safe — different files, different subdirs). Each subagent's brief:

> You're generating fresh `datasets/traits/emotion_set/{emotion}/positive.txt + negative.txt` files (150 lines each) for {anxiety | enthusiasm | warmth}. These will be used for base-model extraction on Llama-3.1-8B + Llama-3.3-70B in a position-sweep experiment.
>
> The existing dataset (15-16 scenarios) is too small. Don't extend it 10x — that drowns the original distribution. Generate fresh with template-grounded variation per the doc, treating the existing 15 as exemplars of the desired style.
>
> **Required reading:**
> 1. `docs/trait_dataset_creation_base_model.md` (full — esp. AFFECTIVE category prescription in the Decision Tree section, lock-in styles for emotion (emotion / physical / thought), and the symptom→cause→fix table)
> 2. The existing `datasets/traits/emotion_set/{emotion}/positive.txt + negative.txt` (15 lines each — exemplars)
> 3. `datasets/traits/emotion_set/{emotion}/definition.txt` (judge rubric)
> 4. `datasets/traits/emotion_set/{emotion}/steering.json` (steering questions for instruct-side eval — perspective-independent at extraction time)
>
> **Design constraints (from the doc's AFFECTIVE category):**
> - Scenario: situation context + name the emotional state + hang on peak
> - Lock-in: emotion ("It made me feel"), physical ("My hands were"), thought ("All I could think was"), or speech
> - Negatives: different situation with opposite valence (NOT same-situation-calm-reaction; that's situation dominance)
> - First person throughout, peak emotional moment, explicit context
>
> **Setup the test trait dir before vetting:**
> ```bash
> mkdir -p datasets/traits/emotion_set/{emotion}_test
> cp datasets/traits/emotion_set/{emotion}/{definition.txt,steering.json} datasets/traits/emotion_set/{emotion}_test/
> # Write iteration's batched positive.txt + negative.txt into this _test dir
> ```
> The `_test` suffix dir keeps iteration noise out of the live `emotion_set/{emotion}/` until convergence.
>
> **Lock-in style distribution check (mandatory per iteration):**
> Before vetting, count lock-in styles in the batch. No single style (emotion / physical / thought / speech) may exceed 40% of the batch. Report the distribution in the iteration log. If exceeded, regenerate the over-represented portion.
>
> **Iteration loop (per the doc's "Iteration & Diagnostics"):**
> 1. Generate a batch (~30 scenarios, varying lock-in style — no single style >40%).
> 2. Vet via API judge: `python extraction/run_extraction_pipeline.py --only-stage 2 --traits emotion_set/{emotion}_test --vet-responses`. Track positive/negative pass rates.
> 3. Read 5 random completions per polarity. Compare to original 15 exemplars.
> 4. Diagnose using the doc's failure-mode table. Common failures for emotions: confound extraction (positives are emotionally vivid, negatives are flat — vector captures intensity not the target emotion), AI-mode capture, exhausted prefix, situation dominance.
> 5. Iterate: regenerate weak lines, refine convention rules.
>
> **Stopping criteria** (no pipeline gate — iteration-fast, escalation-loose):
> - Vetting pass rate ≥ 60% on positive AND ≥ 60% on negative.
> - Spot-check passes on-trait test.
> - Lock-in style distribution within 40% cap.
> - Final dataset has 150 lines positive, 150 lines negative.
>
> **Note:** The plan does NOT gate on probe val_effect_size at this stage. The whole point of the experiment is measuring weak signals at the right position; gating dataset prep on a 0.5 floor would reject legitimately weak affective signals. Pipeline-side validation happens in Stage 2.
>
> **Loop limit:** 2 iterations. Escalate to user with iteration log if not converged after 2 (per user request: keep 0.5 floor philosophy but escalate quickly when not met). Don't loop indefinitely.
>
> **Final writes (only after convergence):**
> - Backup the existing 15-scenario versions to `datasets/traits/emotion_set/{emotion}/_original_15/{positive,negative}.txt`
> - Write fresh 150-line `datasets/traits/emotion_set/{emotion}/positive.txt + negative.txt`
> - Keep existing `definition.txt` and `steering.json` (don't regenerate; they're already valid)
> - Remove the `_test` dir
> - Iteration log: `dev/tasks/base_extraction_position/results/1.2_{emotion}_generation_iteration_log.md`

**Parallelism:** All three emotion subagents run in parallel. Different traits, different files, different subdirs.

### 1.3: USER REVIEW GATE — fresh emotion datasets
**Purpose**: 450 newly-generated scenarios (3 emotions × 150 × 2 polarities) become permanent dataset artifacts; need a human review.
**Depends on**: 1.2 returned converged.

**Action:** Each subagent surfaces 10 random new positives + 10 random new negatives + the 15 originals interleaved (anonymized). User reads, judges quality and consistency. Reads iteration logs to assess subagent reasoning.

**Verify:** User explicit approval logged in notepad as `[YYYY-MM-DD HH:MM PST] Step 1.3-review: APPROVED by user`. If user rejects, kick back to subagent with specific feedback.

### 1.4: Commit datasets
**Purpose**: Make the new emotion datasets reproducible.
**Depends on**: 1.3 user-approved.

**Command:**
```bash
git add datasets/traits/emotion_set/{anxiety,enthusiasm,warmth}/_original_15/ \
        datasets/traits/emotion_set/{anxiety,enthusiasm,warmth}/{positive,negative}.txt
git commit -m "datasets: regenerate emotion_set/{anxiety,enthusiasm,warmth} at n=150 (fresh, doc-template-grounded)"
```

### Checkpoint: After Stage 1
- [ ] PV side at 150/150/150 (Plan A's padding committed).
- [ ] Emotion side at 150/150 fresh (3 traits × 2 polarities = 6 files, all newly generated and user-approved).
- [ ] Original 15-scenario versions backed up to `_original_15/` subdirs.
- [ ] Iteration logs exist for all 3 emotions, showing the actual diagnostic loop.
- [ ] All datasets committed.
- [ ] Stage judgment: were any emotions hard to converge (e.g., warmth — affective vs tonal might overlap)? Worth flagging before extraction.

---

## Stage 2: 8B Extraction + Aggregation (medium detail — ~6 steps)
_Run extraction pipeline on Llama-3.1-8B base for all 6 traits × 6 positions × probe + mean_diff. Aggregate._

**Purpose:** Produce 1p trait vectors at all 6 positions × 6 traits × 2 methods on Llama-3.1-8B base. The aggregator's output is the input to Stage 3 analysis.

**Depends on:** Stage 1 complete.

**Predicts:** 6 traits × 6 positions × 32 layers × 2 methods = 2304 records in the aggregator JSON, each with val_effect_size + val_auroc + polarity_correct. PV traits' val_effect_size at L11-L14 in the same ballpark as comparison-persona-vectors (>1.0). Emotion traits at L11-L14 likely lower (>0.5 expected) per the AFFECTIVE category being weaker-signal at extraction time.

**Settings:**
- Model variants: base = `meta-llama/Llama-3.1-8B`, instruct = `meta-llama/Llama-3.1-8B-Instruct`
- Defaults: extraction on `base`
- Methods: probe + mean_diff (run BOTH — cheap once activations are captured, controls method-dependence per critic)
- Positions: response[:1], response[:3], response[:5], response[:10], response[:15], response[:20]
- Component: residual
- Layers: all 32
- val_split: 0.1
- max_new_tokens: 32, temperature: 0, no chat template

**Pipeline trait-path convention:** `--traits {category}/{trait}` (e.g. `pv_natural/evil`, `emotion_set/anxiety`).

**Critical bug fix (per Phase 7.5 critic):** the response-cache key does NOT include position, but `max_new_tokens` auto-derives from position. Naive looping generates `response[:1]` first → writes 1-token responses → subsequent positions cache-hit on those 1-token responses → silent extraction garbage at positions 3+. Fix: pre-generate responses ONCE with `--max-new-tokens 32`, then sweep positions with `--only-stage 3,4,5,6`.

**Pipeline-side TODO:** the pipeline currently silently truncates / pads when generated response length is shorter than the requested position slice. This needs a code fix to FAIL LOUDLY when `len(response_tokens) < requested_position_end`. Add a one-time guard pass before Stage 3 actual run; if any response in `pos.json`/`neg.json` is shorter than 32 tokens, abort with a clear error listing offending rows.

**Key steps:**
1. **Create config:** Write `experiments/base_extraction_position/config.json` with TWO base variants (per user decision):
   ```json
   {
     "model_variants": {
       "base_8b": {"model": "meta-llama/Llama-3.1-8B"},
       "instruct_8b": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
       "base_70b": {"model": "meta-llama/Llama-3.3-70B"},
       "instruct_70b": {"model": "meta-llama/Llama-3.3-70B-Instruct"}
     },
     "defaults": {"extraction": "base_8b", "application": "instruct_8b"}
   }
   ```
2. **Pre-generate responses once per (trait, model variant) — Stage 1 of pipeline only:**
   ```bash
   python extraction/run_extraction_pipeline.py \
       --experiment base_extraction_position \
       --model-variant base_8b \
       --traits pv_natural/evil,pv_natural/sycophancy,pv_natural/hallucination,emotion_set/anxiety,emotion_set/enthusiasm,emotion_set/warmth \
       --only-stage 1 \
       --max-new-tokens 32
   ```
   Outputs `experiments/.../extraction/{trait}/base_8b/responses/{pos,neg}.json` with 32-token completions.
3. **Verify response lengths:**
   ```bash
   python -c "
   import json, glob
   for f in glob.glob('experiments/base_extraction_position/extraction/*/*/base_8b/responses/{pos,neg}.json'):
       d = json.load(open(f))
       short = [r for r in d if len(r.get('response_tokens', r.get('completion', ''))) < 32]
       if short:
           print(f'FAIL {f}: {len(short)} responses < 32 tokens'); exit(1)
   print('all responses >= 32 tokens')
   "
   ```
   If this fails, escalate to user — increase max_new_tokens or trim trait dataset.
4. **Sweep positions and methods (Stage 3+4+5+6, no regeneration):**
   ```bash
   for pos in 'response[:1]' 'response[:3]' 'response[:5]' 'response[:10]' 'response[:15]' 'response[:20]'; do
     python extraction/run_extraction_pipeline.py \
         --experiment base_extraction_position \
         --model-variant base_8b \
         --traits pv_natural/evil,pv_natural/sycophancy,pv_natural/hallucination,emotion_set/anxiety,emotion_set/enthusiasm,emotion_set/warmth \
         --methods probe,mean_diff \
         --component residual \
         --position "$pos" \
         --only-stage 3,4,5,6
   done
   ```
5. **Add label-shuffle null control (one trait, all positions):**
   - For evil only: copy `datasets/traits/pv_natural/evil` → `datasets/traits/_shuffle_control/evil` and shuffle pos/neg labels (swap a random ~50% of pairs). Run extraction on this shuffled trait at all 6 positions. Provides floor val_effect_size.
6. **Aggregate:** `python analysis/vectors/extraction_evaluation.py --experiment base_extraction_position`. Produces `extraction_evaluation.json` with all records.
7. **Sanity check:**
   ```bash
   python -c "
   import json
   d = json.load(open('experiments/base_extraction_position/extraction/extraction_evaluation.json'))
   print(f'records={len(d[\"all_results\"])}')
   print(f'positions={d[\"positions\"]}')
   missing = [r for r in d['all_results'] if 'val_auroc' not in r]
   print(f'missing_auroc={len(missing)}')
   "
   # Expected: records >= 2304, positions = 6 distinct, missing_auroc=0
   ```

**Stopping criteria:** Aggregator JSON has all expected records, val metrics populated, label-shuffle control runs included. Sanity check on PV val_effect_size at L11-L14: comparable-ish to comparison-persona-vectors (different methods so not exact match expected).

**If results differ from predictions:**
- Some (trait, position) cells missing → re-run with explicit `--layers 0-31`
- val_auroc missing → AUROC plumbing didn't fire; check `core/validation.py`
- PV val_effect_size much lower than comparison-persona-vectors → either dataset (post-Plan-A padding) or pipeline differs from PV. Diagnose before Stage 3.

### Checkpoint: After Stage 2
- [ ] All 6 traits × 6 positions × 32 layers × 2 methods extracted on 8B.
- [ ] val_effect_size + val_auroc populated for every record.
- [ ] Label-shuffle null control extracted.
- [ ] Eyeball PV val_effect_size at L11-L14: in expected range.
- [ ] Notepad + iteration logs updated.
- [ ] Stage judgment: any traits showing weird behavior (collapse, all positions equal, etc.)?

---

## Stage 3: 8B Analysis (Position Curves + Pre-Registered Test)
_Compute position curves per trait, apply pre-registered test, plot, write preliminary results._

**Purpose:** Apply the pre-registered Fisher's exact test on argmax-position-by-category. Generate position-curve plots. Document any qualitative patterns.

**Depends on:** Stage 2 complete.

**Predicts:** Per the hypothesis, DISPOSITIONAL (evil) peaks early (positions 1-5), AFFECTIVE (anxiety/enthusiasm/warmth) peaks later (10-15), INTERPERSONAL/DECEPTION (syco/halluc) somewhere in between. Predicting argmax positions:
- Evil: 1-5
- Sycophancy: 5-15 (interpersonal narrative needs space)
- Hallucination: 5-15 (narrator + reported speech)
- Anxiety: 10-15
- Enthusiasm: 10-15
- Warmth: 5-15

**Key steps (medium detail — runner decomposes):**
1. **Per-trait position curves.** For each of 6 traits: plot val_effect_size vs position (1, 3, 5, 10, 15, 20), one line per (method × layer). Find argmax position per trait, per layer.
2. **Best-layer selection (held out).** For each (trait, position), select the layer with max val_effect_size on val split. Then re-evaluate that layer choice on a held-out test split. Report test val_effect_size as primary number (avoids argmax-over-layers inflation).
3. **Pre-registered test.** Categorize traits: DISPOSITIONAL (evil), INTERPERSONAL (syco), DECEPTION (halluc), AFFECTIVE (anxiety, enthusiasm, warmth). Bucket positions: early (1, 3, 5), late (10, 15, 20). Build a contingency table: rows = categories, columns = early/late. Each trait's argmax-position falls in early or late. Apply Fisher's exact test.
4. **Bootstrap CIs (scenario level, cache-based).** Resample the 150 scenarios with replacement, then for each resample re-train the probe on cached activations from those scenarios at the held-out best layer, project, recompute val_effect_size at all 6 positions. Repeat 1000×. ~36K probe fits total (1000 × 6 positions × 6 traits) — feasible with sklearn, ~10 min total. Re-projecting cached activations only (not regenerating model outputs).
5. **Label-shuffle floor.** Confirm shuffled-label val_effect_size is <0.2 at all positions. If higher, our signal is contaminated by something we don't understand.
6. **Method comparison.** Probe vs mean_diff: do both produce the same position-curve patterns per trait? If they disagree, flag as method-dependence finding.
7. **Plot.** 6-panel plot (one per trait), x = position, y = val_effect_size, lines = method (probe vs mean_diff) + label-shuffle floor as a horizontal reference line. Save to `experiments/base_extraction_position/results/8b_position_curves.png`.
8. **Write preliminary findings** to `dev/tasks/base_extraction_position/base_extraction_position_findings.md` Observations section.

**Stopping criteria:** Pre-registered test computed (significant or not). Plot generated. Per-trait argmax positions reported with bootstrap CIs.

### Checkpoint: After Stage 3
- [ ] Pre-registered test result documented with explicit p-value or "underpowered, n=6 traits split 1/1/1/3" framing.
- [ ] Per-trait argmax positions reported with bootstrap CIs.
- [ ] Label-shuffle floor confirmed near 0.
- [ ] Method comparison: probe and mean_diff agree (or disagreement flagged).
- [ ] Stage judgment: does the data support the hypothesis? If not, what's the qualitative pattern?

---

## Stage 4: 8B Steering Verification (3 traits × 6 positions)
_Validate "detection ≈ steering" on 3 traits by comparing val_effect_size best layer to steering Δ best layer._

**Purpose:** Test whether the detection-best-layer (max val_effect_size) at each position predicts the steering-best-layer (max Δ at coherence ≥70). Critical because user explicitly flagged "detection ≠ steering" as a known concern.

**Depends on:** Stage 2 complete (vectors must exist).

**Predicts:** For each (trait, position), detection-best-layer ≈ steering-best-layer (within ±2 layers). Mismatches flagged as findings — they say something interesting about the trait or about steering.

**Settings:**
- Steering model: `meta-llama/Llama-3.1-8B-Instruct`
- Traits: sycophancy + anxiety + evil (one per major category — covers DISPOSITIONAL, INTERPERSONAL, AFFECTIVE)
- Positions: all 6 (1, 3, 5, 10, 15, 20)
- Coefficient sweep: adaptive search (`utils/coefficient_search.py`)
- Layer range: L7-L23 on 8B (~22-72% depth — wider than 30-60% for safety, in case evil or anxiety optimize outside L9-L19)
- Judge: gpt-4.1-mini logprob method
- Coherence threshold: 70
- Eval prompts: existing `datasets/traits/{trait}/steering.json`

**Key steps:**
1. Run steering on probe vectors for sycophancy at all 6 positions × L9-L19.
2. Run steering on probe vectors for anxiety at all 6 positions × L9-L19.
3. Run steering on probe vectors for evil at all 6 positions × L9-L19.
4. For each (trait, position): pull steering-best-layer (max Δ at coherence ≥70). Compare to detection-best-layer (max val_effect_size at the same position from Stage 2).
5. Compute layer-mismatch per (trait, position). Distribution: how often ±0, ±1, ±2, more?
6. Save consolidated `experiments/base_extraction_position/results/steering_verification.json` with structure: `{trait: {position: {detection_best_layer, steering_best_layer, mismatch, steering_delta, coherence}}}`.
7. Plot: scatter of detection-best-layer (x) vs steering-best-layer (y) per (trait, position), color by trait. Y=X line as reference.

**Stopping criteria:** All 18 (trait × position) cells have detection vs steering best layers compared. Distribution of mismatches reported. Plot generated.

**If results differ from predictions:**
- Detection ≈ steering across all 18 cells → strong evidence for the validity of detection-only analysis going forward
- Detection ≠ steering systematically → important finding; means val_effect_size alone isn't sufficient for vector quality assessment, need steering. This would also weaken Stage 3's primary analysis (which is detection-only).
- Mixed results (some traits detection ≈ steering, others ≠) → interesting nuance, report per-trait

### Checkpoint: After Stage 4
- [ ] 3 traits × 6 positions × layer sweep complete.
- [ ] Mismatch distribution computed.
- [ ] Plot generated.
- [ ] Stage judgment: does detection predict steering? Implications for the rest of the experiment + future findings?

---

## Stage 5: 70B Extraction + Final Analysis + Writeup
_Run the full pipeline on Llama-3.3-70B base. Compare patterns to 8B. Final writeup._

**Purpose:** Scale check: does the 8B position-curve pattern replicate on 70B? Final viz finding writeup.

**Depends on:** Stages 2-4 complete on 8B.

**Predicts:** No specific cross-model prediction (per critic). Report 70B curves alongside 8B. If the position-by-category pattern replicates, that's a bonus; if 70B shows a different pattern, that's also a finding.

**Compute estimate:** ~3-4h on 70B node (4-8×A100 TP). Stage-3 activation capture at 300 prompts is the bottleneck (~2.5h); probe + mean_diff vector training is fast (CPU); steering verification adds ~1h if included.

**Key steps:**
1. **Run 70B extraction inside the same experiment dir** (per user decision: single dir, model variant flag). Use `--model-variant base_70b`. 70B has 80 layers, layer sweep is 0-79. Note: large models often peak at lower relative depth (~20%); for steering verification, widen layer range to L16-L48 (20-60% depth).
   - Pre-generate Stage 1 with `--max-new-tokens 32` (same fix as 8B).
   - Verify response length ≥ 32 per row (same check).
   - Sweep positions with `--only-stage 3,4,5,6`.
2. **Aggregate 70B results.**
3. **Re-run Stage 3 analysis** on 70B aggregator output. Per-trait position curves + pre-registered test + bootstrap CIs.
4. **Compare 8B vs 70B curves.** Side-by-side plot: 6 panels (one per trait), each panel has 2 lines (8B probe + 70B probe).
5. **Steering verification on 70B (optional, time-permitting):** repeat Stage 4 on 70B for sycophancy + anxiety + evil. Layer range = 30-60% depth = L24-L48.
6. **Generate combined figure** for the viz finding: best position per trait per model.
7. **Write `docs/viz_findings/extraction_position.md`** (new viz finding):
   - Setup, hypothesis, method
   - 8B results: position curves + pre-registered test result + bootstrap CIs + label-shuffle floor
   - 8B steering verification: detection vs steering layer mismatch distribution
   - 70B results: position curves
   - 70B vs 8B comparison
   - Caveats: n=6 traits underpowered, single model family (Llama), single component (residual), n=3 traits for steering verification, single position-set
8. **Update `docs/viz_findings/index.yaml`:** add `extraction_position.md` to Methods & Calibration tier.
9. **Update notepad `docs/other/apr30_notepad_viz_findings_revisions.md`** with the resolution.

**Stopping criteria:** 70B extraction + analysis complete; final viz finding committed; success criterion explicitly stated as met or not met with evidence.

### Checkpoint: After Stage 5
- [ ] 70B extraction complete; aggregator JSON has 70B records (6 traits × 6 positions × 80 layers × 2 methods = 5760 records).
- [ ] 70B position curves plotted alongside 8B.
- [ ] Final writeup committed.
- [ ] Findings reconciled in `findings.md` with status (CONFIRMED / REFUTED / INCONCLUSIVE) per claim.

---

## If Stuck

- **Emotion datasets won't converge after 4 iterations** → escalate to user with the iteration log; consider dropping that emotion or reducing the convergence bar.
- **Label-shuffle floor is high (>0.2)** → something's wrong. Likely a prompt-leakage or data-mixing bug. Trace via reading actual scenarios + completions.
- **8B PV val_effect_size much lower than comparison-persona-vectors** → either the dataset (post-Plan-A padding) or pipeline differs. Diagnose before Stage 3.
- **70B run OOMs** → reduce TP world size, reduce batch size, or split 6-position runs across separate GPU sessions.
- **Steering verification shows detection ≠ steering on most cells** → important finding; report it but flag that Stage 3's detection-only primary analysis is incomplete without steering corroboration.

## Notes

**Connections:**
- Plan A (`1st_vs_3rd_person_extraction`) is orthogonal: same model, same pv_natural traits, but tests perspective at single position instead of position at single perspective.
- `mar15-detection_layer_profiling` future-ideas entry classifies traits into 3 affective categories — connects to this experiment's category framing.
- `coefficient-scaling-law` rewrite (May 1) confirmed cross-trait robustness on Qwen2.5-14B; this experiment extends to position dimension on Llama.

**Caveats baked into the writeup (per critic):**
- n=3 traits per category (esp. AFFECTIVE which is overrepresented at 3 vs 1/1/1 for others) is severely underpowered for category-level claims; "qualitative pattern" framing preferred over hard p-values.
- Position 1 is partially tokenization-bound; report but framed as a floor reference.
- 6 positions are correlated views of the same activation tensor; bootstrap at scenario level handles this.
- 70B vs 8B comparison is descriptive, not strict (different vector quality, scaling effects entangled).
- Probe-only would be method-dependent; we run mean_diff alongside as a check.
- Steering verification on 3 traits (sycophancy, anxiety, evil) doesn't generalize to all 6 traits; per-trait differences possible.
- Single pre-registered Fisher's exact test → no multiple-comparisons correction needed. Per-cell exploratory comparisons are reported as descriptive, not given p-values.
