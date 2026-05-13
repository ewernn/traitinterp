# Experiment: 1st-vs-3rd Person Extraction

## Goal
Test whether extracting trait vectors from 1st-person scenarios produces stronger steering vectors than 3rd-person scenarios on Llama-3.1-8B base, for evil/sycophancy/hallucination. Replaces the deleted refusal-only `1st-vs-3rd-person.md` finding with a generalizable 3-trait persona-traits result.

## Hypothesis
1st-person scenarios produce stronger steering vectors than 3rd-person, because the base model generates AS the experiencer (activating its own internal representation of the trait) rather than ABOUT the experiencer. Per `docs/trait_dataset_creation_base_model.md`: "First person. The model generates AS the experiencer. We're extracting the model's own internal representation of the trait during generation — not its representation of someone else's trait."

**Prediction:** 1p steering Δ exceeds 3p Δ by ≥20% relative on ≥2 of 3 traits at coherence ≥70.

**Alternative considered and rejected:** Use Plan B's position sweep instead. Rejected because perspective and position are orthogonal axes; the perspective effect should hold at the standard `response[:5]` extraction position before we explore where it changes. Position sweep is its own experiment.

## Complexity
Medium — 4 stages, ~25-35 steps, estimated 1-2 days (mostly bound by GPU time for extraction + steering on three traits × two perspectives × all layers).

## Success Criteria
- [ ] **Primary:** 1p steering Δ exceeds 3p Δ by ≥20% relative on ≥2 of 3 traits at coherence ≥70.
- [ ] **Secondary report:** val_effect_size and AUROC reported alongside steering Δ for both perspectives. Dissociations (e.g., 1p higher Δ but 3p higher AUROC) flagged as findings.
- [ ] **Bootstrap CIs** computed over scenario subsamples or judge-rerun variance for both 1p and 3p Δ. CIs reported with all magnitude claims.
- [ ] **Per-trait pattern** discussed in writeup: are evil/syco/hallucination affected differently by perspective? (Trait-category × perspective interaction, per critic's flag.)
- [ ] **Final viz finding** at `docs/viz_findings/1st-vs-3rd-person.md` rewritten with the new 3-trait result, replacing the broken refusal-only version.

## Prerequisites
- Llama-3.1-8B base + instruct accessible via HuggingFace
- HF_TOKEN exported
- GPU with ≥40GB VRAM (single A100 sufficient; 8B model)
- gpt-4.1-mini API access for trait + coherence judging
- Existing 1p source data at `datasets/traits/pv_natural/{evil,sycophancy,hallucination}/positive.txt + negative.txt` (verified: 150/150/49 lines)
- The `core.validation.compute_vector_quality()` primitive (already in core/, exposes val + OOD AUROC)
- The new aggregator `analysis/vectors/extraction_evaluation.py` (auto-discovers components/positions, writes AUROC)

Verify:
```bash
ls datasets/traits/pv_natural/{evil,sycophancy,hallucination}/{positive.txt,negative.txt,definition.txt,steering.json}
python -c "from core import compute_vector_quality, auroc; print('ok')"
nvidia-smi  # confirm GPU available
echo "$HF_TOKEN" | head -c 8  # confirm token set
```

## Stopping Criteria
- All four success criteria above met OR
- Pre-registered effect threshold not met on ≥2 traits AND bootstrap CIs reliably bracket zero (i.e., we have evidence 1p ≈ 3p, not just inconclusive).

## Open Questions / Connections (not blocking)

- **User vs assistant framing:** ant_emotion_concepts ran a related comparison using ":" token after "User" vs "Assistant" speaker labels. Worth a follow-up experiment but not in scope here.
- **Position interaction:** The user's separate Plan B is a position-sweep experiment. If 1p > 3p at response[:5], a future question is whether 3p catches up at response[:20+] (would suggest the perspective gap is about how fast the model "gets into" the trait, not whether it gets there).
- **mean_diff vs probe:** This experiment uses probe only (matches PV's pv_natural). mean_diff might respond differently to perspective shift (probe optimizes separability; mean_diff captures centroid distance). Defer to a follow-up.

---

## Stage 1: Dataset Preparation (agentic iterative — quality-gated, not step-counted)
_Pad hallucination 1p, generate 3p translations for all three traits, with iterative quality loops grounded in `docs/trait_dataset_creation_base_model.md`._

**Agentic principle.** Do NOT statically execute fixed prompts. The dataset creation doc prescribes an iterative loop (generate → vet → read responses → diagnose → fix). Stage 1 spawns Claude Code subagents that own this loop end-to-end. Each subagent:
- Reads `docs/trait_dataset_creation_base_model.md` in full
- Reads the existing 1p source files for the trait it's working on
- Reads `docs/extraction_guide.md` to understand the pipeline it will be vetting against
- Generates an initial batch
- Runs the extraction pipeline at small scale (modal-callable for the GPU stages, local for vetting/inspection) to vet the batch
- Reads the actual completions and steered responses, not just scores
- Diagnoses failure modes using the doc's symptom→cause→fix table
- Iterates: regenerate weak scenarios, adjust lock-in style, fix exhausted prefixes
- Returns final scenarios + a quality report
- Logs every iteration cycle to `dev/tasks/1st_vs_3rd_person_extraction/results/{step}_iteration_log.md`

**Modal access.** GPU-required vet steps (capturing activations to check probe accuracy, running steering on the test layer) go through modal. The subagent decides when to invoke modal — full pipeline test isn't always needed; vetting via the API judge usually suffices to catch bad scenarios early.

**Stopping criteria for each substage** (semantic, not step-count):
- Vetting pass rate ≥ 60% on positives AND ≥ 60% on negatives (the doc says low pass rates don't preclude good vectors but aim for healthy)
- Spot-check of 5 random completions per polarity reads as on-trait
- For substages that ran a small pipeline test: probe accuracy at the test layer ≥ 0.7 on val split

**Loop limit.** Max 4 iterations per substage. If not converged, escalate to user with the iteration log.

### 1.1.0: Backup originals BEFORE any edit
**Purpose**: Atomic safety. Originals must be preserved before any write to the live files.
**Depends on**: nothing.

**Command:**
```bash
mkdir -p datasets/traits/pv_natural/hallucination/_original_49
cp datasets/traits/pv_natural/hallucination/positive.txt datasets/traits/pv_natural/hallucination/_original_49/positive.txt
cp datasets/traits/pv_natural/hallucination/negative.txt datasets/traits/pv_natural/hallucination/_original_49/negative.txt
```

**Verify:**
```bash
wc -l datasets/traits/pv_natural/hallucination/_original_49/{positive,negative}.txt
# Both must be 49
diff datasets/traits/pv_natural/hallucination/positive.txt datasets/traits/pv_natural/hallucination/_original_49/positive.txt
# Must be empty
```

**If wrong:** Backup didn't take. Stop and re-run before doing anything else.

### 1.1: Pad hallucination 1p from 49 → 150 (agentic iterative)
**Purpose**: Match sample size across traits so 1p vs 3p comparison isn't confounded by n. The new 101 scenarios must be stylistically indistinguishable from the original 49 to avoid an old-49-vs-new-101 distributional confound.
**Depends on**: 1.1.0 verified.
**Predicts**: After 2-3 iterations, ~101 new scenarios that pass vetting at ≥60% rate AND read indistinguishable-from-originals on blind spot-check.

**How to execute:** Spawn an `r:run-experiment`-style subagent (or `r:investigator` if the executor prefers a lighter loop) with this brief:

> You're generating 101 new hallucination scenarios for `datasets/traits/pv_natural/hallucination/{positive,negative}.txt` to pad it from 49→150. The new scenarios must be stylistically indistinguishable from the original 49 — failure mode is a distributional shift between old and new that confounds downstream extraction.
>
> **Required reading:**
> 1. `docs/trait_dataset_creation_base_model.md` (full)
> 2. All 49 lines of the existing `positive.txt` and `negative.txt`
> 3. `definition.txt` (judge rubric)
>
> **Iteration loop (per the doc):**
> 1. Generate a batch (~25 scenarios per pass).
> 2. Vet via the LLM judge (`extraction/run_extraction_pipeline.py --only-stage 2 --traits pv_natural/hallucination_pad_test`) on a separate test trait dir, to keep the live dataset clean during iteration.
> 3. Read 5 random completions per polarity. Compare to 5 random originals.
> 4. Diagnose using the doc's symptom→cause→fix table. Common failures: real fake names instead of fictional, missing "I had no idea but" thought-frame, exhausted prefix.
> 5. Regenerate weak scenarios (per "generate more, cull bad ones" — don't spot-fix individual lines).
> 6. Repeat until stop criterion met.
>
> **Stopping criteria:** vetting pass rate ≥ 60% positive AND ≥ 60% negative AND blind-pair test (mix 5 new + 5 originals, can the executing agent tell which is which?) returns near-chance.
>
> **Modal usage:** Optional. Vetting is API-only (no GPU). Only invoke modal if you want to verify the new+old pool produces vectors with val_effect_size comparable to the original-49-only baseline.
>
> **Loop limit:** 4 iterations. If not converged, return early with the iteration log + recommendation.
>
> **Final write:** Append 101 lines to live `datasets/traits/pv_natural/hallucination/{positive,negative}.txt`. Preserve original 49 in original line positions.
>
> **Log:** Write all iterations to `dev/tasks/1st_vs_3rd_person_extraction/results/1.1_padding_iteration_log.md` (each iteration: batch generated, vetting result, diagnosis, action taken).

**Output:** `positive.txt` and `negative.txt` at 150 lines each. Iteration log written.

### 1.1-review: USER REVIEW GATE
**Purpose**: 101 new scenarios become permanent dataset artifacts; subagent self-assessment isn't enough.
**Depends on**: 1.1 returned converged.

**Action:** Subagent surfaces 20 random new scenarios + 20 random originals interleaved (anonymized). User reads, judges blind which is which. Also reads the iteration log to see how the subagent diagnosed and recovered from any failures.

**Verify:** User explicit approval logged in notepad as `[YYYY-MM-DD HH:MM PST] Step 1.1-review: APPROVED by user`. If user rejects, kick back to subagent with specific feedback and resume iteration.

### 1.2 / 1.3 / 1.4: Translate {evil, sycophancy, hallucination} 1p → 3p (agentic iterative, parallel)
**Purpose**: Generate 3p counterparts that preserve trait expression while shifting perspective.
**Depends on**: 1.1-review approved (for hallucination, where the 1p side was just padded). Evil and sycophancy 1p sides don't depend on 1.1, so 1.2 and 1.3 can start as soon as 1.1.0 backup is done.
**Predicts**: After 2-3 iterations per trait, 3p datasets where:
- A blind reader can confirm each line is a 1p→3p translation of the same scenario (perspective shifted, content preserved).
- The base model's completions on the 3p prefix produce coherent text, not garbled by ambiguous antecedents.
- Probe accuracy on 3p activations at L11-L14 is in the 0.6–0.9 range (lower than 1p is fine — that's the experiment; collapse to ~0 is not).

**How to execute:** Spawn an agentic subagent **per trait** (parallel-safe — different files, different subdirs). Each subagent's brief:

> You're translating `datasets/traits/pv_natural/{trait}/positive.txt + negative.txt` from 1st person to 3rd person (singular they/them) and writing to `datasets/traits/pv_natural_3p/{trait}/`. The translation must preserve trait expression: when the base model continues a translated 3p prefix, it should still complete on-trait — narrating someone else's behavior instead of inhabiting it.
>
> **Required reading:**
> 1. `docs/trait_dataset_creation_base_model.md` (full — esp. lock-in styles, decision tree, common failure modes, symptom→cause→fix table)
> 2. The trait's current `positive.txt`, `negative.txt`, `definition.txt`, `steering.json`
> 3. `docs/extraction_guide.md` to understand the pipeline you'll vet against
>
> **Translation conventions (starting point — refine based on iteration results):**
> 1. Pronoun map: I→they, me→them, my→their, mine→theirs, myself→themselves, I'm→they're, I've→they've, I'd→they'd, I'll→they'll. Verb agreement (am/feel/want → are/feel/want).
> 2. NEVER replace pronouns inside quoted speech.
> 3. NEVER introduce names or descriptors ("the cruel person", "they said evilly").
> 4. NEVER change the lock-in word — if 1p ends `..., I` then 3p ends `..., they`.
> 5. Other characters' pronouns stay as-is.
> 6. For hallucination's long-form scenarios: tag lines requiring disambiguation reword (`they told them` → `they told the colleague`) with `<DISAMB>` suffix; aim for <5% rate, strip tags before commit.
>
> **Iteration loop (per the doc's "Iteration & Diagnostics" section):**
> 1. Translate a batch (~25 scenarios).
> 2. Vet via the LLM judge using the existing pipeline: `python extraction/run_extraction_pipeline.py --only-stage 2 --traits pv_natural_3p/{trait}_test --vet-responses`. Use a `_test` suffix dir during iteration to keep the live `pv_natural_3p/{trait}/` clean until convergence.
> 3. Read 5 random completions per polarity. Compare to 5 random 1p completions side-by-side. Do they read on-trait? Do antecedents stay clean?
> 4. **Optional modal-backed deeper check** (use sparingly — once mid-way, once at end): run extraction-only on a small sample for L11-L14 (`--only-stage 3,4 --layers 11,12,13,14`). Check val_effect_size at the test layer. Compare to 1p baseline. If the 3p val_effect_size is far below 1p (>50% drop), inspect for pattern — likely a translation-confound rather than a real perspective effect.
> 5. Diagnose using the doc's symptom→cause→fix table. Common failures for 3p:
>    - "exhausted prefix" — the 3p translation moves the trait expression earlier; cut deeper.
>    - "confound extraction" — the contrast captures perspective AND something else (vocabulary, sentence structure).
>    - "AI-mode capture" — the model treats 3p prefixes as describing an external entity and triggers AI-disclaimer mode.
> 6. Iterate: regenerate weak lines (per the doc's "Generate more, cull bad ones" — don't spot-fix), refine convention rules in light of recurring failures, escalate stubborn cases.
>
> **Stopping criteria** (semantic):
> - Vetting pass rate ≥ 60% on 3p positive AND ≥ 60% on 3p negative.
> - Spot-check of 5 random completions reads coherent + on-trait per polarity.
> - For hallucination: <5% lines required `<DISAMB>` tagging.
> - If a modal extraction check was run: probe val_effect_size on 3p ≥ 0.5 at the test layer (lower than 1p is acceptable; collapse is not).
>
> **Loop limit:** 4 iterations per trait. If not converged, escalate to user with iteration log + recommendation. Don't write the live `pv_natural_3p/{trait}/` directory until converged.
>
> **Final writes (only after convergence):**
> - `datasets/traits/pv_natural_3p/{trait}/positive.txt` (150 lines, line N = 3p translation of 1p line N — line order preserved)
> - `datasets/traits/pv_natural_3p/{trait}/negative.txt` (150 lines, same)
> - `datasets/traits/pv_natural_3p/{trait}/definition.txt` and `steering.json` (copied verbatim from 1p)
> - For hallucination: strip `<DISAMB>` tags before final write.
>
> **Quote-aware verify:** Write (or reuse) `dev/tasks/1st_vs_3rd_person_extraction/results/verify_translation.py` — a Python script that tokenizes per-scenario, splits on quote pairs, and only flags 1p pronouns appearing OUTSIDE quoted speech. Run it on final outputs. (The naive `grep -E '\b(I|me|my|...)\b'` doesn't distinguish in-quote from out-of-quote; that's why a quote-aware tokenizer is required.)
>
> **Iteration log:** Write all iterations to `dev/tasks/1st_vs_3rd_person_extraction/results/1.{2,3,4}_{trait}_translation_iteration_log.md`. Each entry: batch attempted, vetting result, completions read, diagnosis, action taken.

**Parallelism:** All three subagents (evil, sycophancy, hallucination) run in parallel. Different traits, different files, different subdirs. The plan's executor (`/r:run-experiment`) should spawn them concurrently.

**Verify (after all three converge):**
- All three traits have 150-line `positive.txt` and `negative.txt` in `datasets/traits/pv_natural_3p/{trait}/`.
- `verify_translation.py` reports zero outside-quote 1p pronouns for each trait.
- Iteration logs exist for each trait, showing the actual diagnostic loop, not a one-shot output.

**If escalated to user:** subagent presents iteration log + final scenarios that didn't converge + recommendation. User decides: accept partial, kick back with feedback, or drop trait from this experiment.

### 1.5: Pilot validation — 10 paired scenarios per trait
**Purpose**: Smoke-test pairing AND quality before extraction (expensive).
**Depends on**: 1.2, 1.3, 1.4 all verified.
**Predicts**: Random pairs (1p line N, 3p line N) should be obvious 1p/3p translations of the same scenario.

**Hard precondition:**
```bash
# Line counts MUST match for each trait + polarity (or paste produces silent garbage)
for t in evil sycophancy hallucination; do
  for p in positive negative; do
    n1=$(wc -l < "datasets/traits/pv_natural/$t/$p.txt")
    n3=$(wc -l < "datasets/traits/pv_natural_3p/$t/$p.txt")
    if [ "$n1" != "$n3" ]; then echo "MISMATCH: $t/$p ($n1 vs $n3)"; exit 1; fi
  done
done
echo "Line counts match"
```

**Method:** Print 10 random matched pairs per trait:
```bash
for trait in evil sycophancy hallucination; do
  echo "=== $trait positive ==="
  paste -d'|' \
    datasets/traits/pv_natural/$trait/positive.txt \
    datasets/traits/pv_natural_3p/$trait/positive.txt \
    | awk 'BEGIN{srand()} {print rand()"|"$0}' | sort | head -10 | cut -d'|' -f2-
done | tee dev/tasks/1st_vs_3rd_person_extraction/results/pilot_validation.md
```

**Verify:** Each row's left/right must be a translation pair. Failures: mismatched indices, missing pronoun substitutions, semantic drift, changed lock-in word.

**USER REVIEW GATE:** User reads pilot_validation.md, says go/nogo. Cannot proceed to 1.6 without explicit user approval logged in notepad.

**If wrong:** Re-translate the offending trait. Don't proceed.

### 1.6: Commit datasets
**Purpose**: Make the new datasets reproducible.
**Depends on**: 1.5 user-approved.

**Command:**
```bash
git add datasets/traits/pv_natural/hallucination/_original_49/ \
        datasets/traits/pv_natural/hallucination/positive.txt \
        datasets/traits/pv_natural/hallucination/negative.txt \
        datasets/traits/pv_natural_3p/
git commit -m "datasets: pad pv_natural/hallucination 49→150; add pv_natural_3p (singular they)"
```

### Checkpoint: After Stage 1
- [ ] All 6 dataset files exist with 150 lines each (3 traits × 2 polarities × 1p + 3p = 12 files; 6 are pre-existing 1p, 6 are new 3p).
- [ ] Hallucination 1p originals backed up to `_original_49/`.
- [ ] Pilot validation shows clean 1p/3p pairing per trait.
- [ ] Dataset commit landed.
- [ ] Notepad updated with all step results, including any translation challenges encountered.
- [ ] Stage judgment: were any translation patterns surprising? Worth investigating before extraction (e.g., did sycophancy's quoted speech translate cleanly, or did we have to make compromises that change the comparison)?

---

## Stage 2: Extraction (medium detail — ~6 steps)
_Run extraction pipeline on both 1p and 3p datasets inside the new experiment dir._

**Purpose:** Produce 1p and 3p trait vectors at matched pipeline settings inside `experiments/1st_vs_3rd_person_extraction/`.

**Depends on:** Stage 1 complete + dataset commit landed.

**Predicts (numeric, from comparison-persona-vectors PV-natural baseline as anchor):**
- 1p side at L11-L14 should produce val_effect_size in the same ballpark as PV's published natural results. The PV finding uses different methods (mean_diff for evil/syco, probe for hallucination) so direct number-match isn't expected, but order of magnitude should hold:
  - Evil 1p: best layer ~L11-L13, val_effect_size > 1.5 expected
  - Sycophancy 1p: best layer ~L13, val_effect_size > 1.0 expected
  - Hallucination 1p (now n=150): best layer ~L13-L15, val_effect_size > 1.0 expected
- 3p side at same layers should be lower (this IS the experiment's hypothesis); no specific anchor.

**Pipeline trait-path convention (resolved from `extraction/run_extraction_pipeline.py:76`):**
- Pipeline accepts `--traits {category}/{trait}` where category = the dir name under `datasets/traits/`.
- 1p side: `--traits pv_natural/evil,pv_natural/sycophancy,pv_natural/hallucination`
- 3p side: `--traits pv_natural_3p/evil,pv_natural_3p/sycophancy,pv_natural_3p/hallucination`
- No symlinking needed.

**Settings (matching comparison-persona-vectors except for dataset):**
- Model variants: base = `meta-llama/Llama-3.1-8B`, instruct = `meta-llama/Llama-3.1-8B-Instruct`
- Defaults: extraction on `base`, application on `instruct`
- Method: probe only (matches PV's pv_natural extraction layout)
- Position: response[:5]
- Component: residual
- Layers: all 32 (full sweep)
- max_new_tokens: 32, temperature: 0, no chat template
- val_split: 0.1 (so per-vector metadata.json gets val_effect_size + val_auroc)

**Key steps:**
1. **Create config:** Write `experiments/1st_vs_3rd_person_extraction/config.json` matching `experiments/persona_vectors_replication/config.json` (same model variants, same defaults).
2. **Run 1p extraction:**
   ```bash
   python extraction/run_extraction_pipeline.py \
       --experiment 1st_vs_3rd_person_extraction \
       --traits pv_natural/evil,pv_natural/sycophancy,pv_natural/hallucination \
       --methods probe \
       --component residual \
       --position 'response[:5]'
   ```
3. **Run 3p extraction:**
   ```bash
   python extraction/run_extraction_pipeline.py \
       --experiment 1st_vs_3rd_person_extraction \
       --traits pv_natural_3p/evil,pv_natural_3p/sycophancy,pv_natural_3p/hallucination \
       --methods probe \
       --component residual \
       --position 'response[:5]'
   ```
4. **Aggregate:**
   ```bash
   python analysis/vectors/extraction_evaluation.py --experiment 1st_vs_3rd_person_extraction
   ```
   Produces `experiments/1st_vs_3rd_person_extraction/extraction/extraction_evaluation.json` with all 192 records (3 traits × 2 perspectives × 32 layers × 1 method).
5. **Sanity check val metrics populated:**
   ```bash
   python -c "
   import json
   d = json.load(open('experiments/1st_vs_3rd_person_extraction/extraction/extraction_evaluation.json'))
   missing = [r for r in d['all_results'] if 'val_auroc' not in r]
   print(f'records={len(d[\"all_results\"])}, missing_auroc={len(missing)}')
   "
   # Expected: records=192, missing_auroc=0
   ```

**Stopping criteria:** Aggregator JSON has 192 records, all with val_accuracy + val_effect_size + val_auroc + polarity_correct populated. 1p val_effect_size at L11-L14 in expected range per predictions above.

**If results differ from predictions:**
- Some layers missing → re-run extraction with explicit `--layers 0-31`
- val_auroc missing → AUROC plumbing didn't fire; check `core/validation.py` is being called from `utils/extract_vectors.py:588`
- 1p val_effect_size much lower than PV anchors → either the dataset (post-padding) or pipeline differs from PV. Diagnose before proceeding to Stage 3.

### Checkpoint: After Stage 2
- [ ] 192 vector records in extraction_evaluation.json (3 traits × 2 perspectives × 32 layers × 1 method).
- [ ] Each record has val_accuracy + val_effect_size + val_auroc populated.
- [ ] Quick eyeball: 1p val_effect_size at L11-L14 (PV's optimal range) is in the same ballpark as comparison-persona-vectors' published numbers (sanity check that re-extraction doesn't differ wildly from PV's results).
- [ ] Notepad updated with extraction commands and any failures.
- [ ] Stage judgment: are val_effect_size patterns already showing 1p > 3p, or is the activation-level signal similar and the difference will only show in steering?

---

## Stage 3: Steering Eval (medium detail — ~6 steps)
_Run steering on instruct model, both perspectives, all extracted vectors, with cache-based bootstrap CIs._

**Purpose:** Get steering Δ at coherence ≥70 per (trait, perspective, layer). Headline metric.

**Depends on:** Stage 2 complete.

**Predicts (numeric, from comparison-persona-vectors PV-natural results as anchor):**
- 1p best Δ per trait should be in range of PV's published natural-extraction results: Evil ~+60-70, Sycophancy ~+45-55, Hallucination ~+55-65. (PV's exact numbers: evil +66.3 at L12, syco +49.2 at L13, halluc +61.4 at L14 — but those used mixed methods; ours is probe-only so expect 10-20% spread.)
- 3p best Δ should be visibly lower per the experiment's hypothesis.
- Per-trait gap: largest expected on sycophancy + hallucination (interpersonal + processing-mode traits where "inhabited experience" matters more), smallest on evil (already abstract dispositional claim).

**Settings:**
- Steering model: `meta-llama/Llama-3.1-8B-Instruct`
- Steering questions: existing `datasets/traits/pv_natural/{trait}/steering.json` (5-10 questions per trait, identical for both perspectives)
- Coefficient sweep: adaptive search (`utils/coefficient_search.py`)
- Judge: gpt-4.1-mini logprob method
- Coherence threshold: 70
- Layer range: layers 5-20 (sweep wider than PV's optimal L10-14 for safety)

**Key steps:**
1. **Run steering eval on 1p vectors:** all 3 traits. Existing pipeline writes per-(trait, layer, coef) results to `experiments/.../steering/.../results.jsonl` with prompt-level scores cached.
2. **Run steering eval on 3p vectors:** same.
3. **Pull best per (trait, perspective):** for each (trait, perspective) read all results.jsonl, filter coherence ≥70, take max trait_score. Record best_layer, best_coef, prompt-level score list.
4. **Bootstrap CIs (cache-based, no regeneration):**
   - For each best (trait, perspective): the result has K prompt-level scores (K = number of steering questions, typically 5-10).
   - Resample K scores with replacement, recompute mean Δ, repeat 1000× (cheap — pure numpy on cached scores).
   - Report 95% CI as percentile bounds. Same for coherence.
   - **Key change vs original draft:** we resample over CACHED scored prompts, not regenerated model outputs. No model reruns. Bootstrap is a few seconds per trait.
5. **Save consolidated results to `experiments/1st_vs_3rd_person_extraction/results/steering_summary.json`:**
   ```json
   {
     "evil": {
       "1p": {"best_delta": ..., "best_layer": ..., "best_coef": ..., "coherence": ..., "ci_low": ..., "ci_high": ..., "n_prompts": ...},
       "3p": {...}
     },
     ...
   }
   ```
6. **Apply success criterion:**
   ```python
   for trait in ["evil", "sycophancy", "hallucination"]:
       d1p, d3p = summary[trait]["1p"]["best_delta"], summary[trait]["3p"]["best_delta"]
       relative_gap = (d1p - d3p) / max(d3p, 1e-6)
       passed = relative_gap >= 0.20
       print(f"{trait}: 1p={d1p:.1f} 3p={d3p:.1f} gap={relative_gap*100:.1f}% {'PASS' if passed else 'FAIL'}")
   # Success: ≥2 of 3 PASS
   ```

**Stopping criteria:** Steering summary contains 6 entries each with CI; success-criterion result is computed and explicit (PASS or FAIL count).

**If results differ from predictions:**
- 1p < 3p on a trait → investigate. Read 5 best-coef steered responses for that trait, both perspectives. Is 3p producing more on-trait text, or is the judge favoring 3p phrasing?
- 1p ≈ 3p across all traits → either the perspective effect is small or response[:5] isn't where it shows. Stage 4 must reflect this honestly; null branch in Stage 4.
- Coherence stays low at all coefs for a trait → trait/model combo doesn't steer at this layer/method. Try wider layer range (L0-L31), or try mean_diff as fallback. If still low, drop trait from the comparison and report n=2.

### Checkpoint: After Stage 3
- [ ] Steering summary has all 6 entries with bootstrap CIs.
- [ ] Apply success criterion: count traits where 1p exceeds 3p by ≥20% relative.
- [ ] Read 3 steered responses per (trait, perspective, best layer) — do qualitative differences match the quantitative?
- [ ] Stage judgment: which traits showed the biggest 1p-3p gap? Smallest? Does the pattern fit a category interpretation (dispositional vs interpersonal vs processing-mode)?

---

## Stage 4: Analysis + Writeup (lighter — ~6 steps)
_Generate figure, rewrite the viz finding, log secondary metrics. Branches on PASS vs FAIL from Stage 3._

**Purpose:** Produce the consumable artifact (viz finding + figure) and document dissociations between primary and secondary metrics. Stage 4 has TWO branches based on Stage 3's success-criterion outcome.

**Depends on:** Stage 3 complete with success-criterion explicit (PASS or FAIL count).

**Predicts:** A bar chart showing 1p vs 3p Δ for three traits with CIs. Either a positive ("1p > 3p by N%") or null ("1p ≈ 3p at response[:5]") finding.

**Key steps (shared across branches):**
1. **Plot:** bar chart, x=trait, y=best Δ at coherence ≥70, two bars per trait (1p, 3p), CIs as error bars. Save to `docs/viz_findings/assets/1st-vs-3rd-person-graph.png` (replaces existing image).
2. **Per-trait pattern table:** rows=traits, cols= 1p Δ (CI), 3p Δ (CI), relative gap %, val_effect_size 1p/3p, val_auroc 1p/3p, vector cosine 1p×3p.
3. **Dissociation check:** for each trait, does steering Δ direction match val_effect_size direction? Disagreements flagged as findings.

### Branch A: PASS (≥2 of 3 traits show 1p Δ ≥ 3p Δ × 1.20)
4a. **Rewrite `docs/viz_findings/1st-vs-3rd-person.md`** with positive claim ("1p elicitation produces stronger persona vectors than 3p at response[:5] for {N} of 3 traits"). Include per-trait pattern, dissociations, all caveats: singular-they confound, probe-only, single-position, base/instruct asymmetry, n=3 traits is small.
5a. **Update viz index `docs/viz_findings/index.yaml`:** move from "Fold/delete" to "Methods & Calibration" tier with a one-line description of the new claim.
6a. **Update notepad `docs/other/apr30_notepad_viz_findings_revisions.md`:** mark 1st-vs-3rd-person as RESOLVED with link to commit.

### Branch B: FAIL (null result — 1p does NOT exceed 3p by ≥20% on ≥2 traits)
4b. **Rewrite `docs/viz_findings/1st-vs-3rd-person.md`** as a NULL FINDING. Title: "Perspective doesn't matter much for persona vector extraction at response[:5]". Explicit framing: we tested the hypothesis that 1p > 3p, did not find sufficient evidence. Include per-trait numbers, CIs (which likely bracket zero), all caveats. Honest negative result is still publishable — see effect-size-vs-steering's null tier handling for tone.
5b. **Update viz index:** keep in "Methods & Calibration" with negative-result framing in description.
6b. **Update notepad** with NULL_RESULT outcome. Note follow-up question: does 1p > 3p emerge at later positions? (Plan B may answer this.)

**Stopping criteria:** New viz finding committed (positive or null); figure embedded; success criterion explicitly stated as met or not met with evidence; index + notepad updated.

### Checkpoint: After Stage 4
- [ ] New viz finding markdown, asset, and index update committed in one commit.
- [ ] Final findings.md written with: claim, evidence, implication, status (CONFIRMED / REFUTED / INCONCLUSIVE) per trait + overall.

---

## If Stuck

- **Translation drift suspected** → spawn a critic agent to re-read pilot pairs and flag scenarios where 1p and 3p differ in non-perspective ways.
- **Extraction val_effect_size ~ 0 for hallucination** → likely the padded scenarios drift in style. Re-read 10 of the new scenarios. Compare to original 49.
- **Steering Δ collapses to noise** → check coherence first. If steered responses are incoherent, lower coefficient. If they're coherent but not on-trait, the vector is weak — go back to extraction layer choice.
- **Judge produces contradictory scores** → re-run judge with verbose logging. Check that logprob method is firing (not falling back to text scoring).
- **Per-trait results disagree (e.g., evil 1p > 3p but sycophancy 1p < 3p)** → that's a finding, not a failure. Report the per-trait pattern.

## Notes
- Connection: ant_emotion_concepts ran a "user vs assistant" version of this comparison (using the ":" token after speaker labels). Could be a follow-up experiment with a more targeted hypothesis (does the perspective effect scale with rhetorical distance?).
- Connection: position-sweep is Plan B; this experiment fixes position to response[:5].
- Caveats to acknowledge in the writeup (per critic): singular-they confound (rare in pretraining), probe-only methodology, single-position generalization, base/instruct asymmetry in steering, n=3 traits is small.
