# Task: logit-lens-fix

## Goal
Make `analysis/vectors/logit_lens.py --save` write to the canonical path `experiments/{exp}/extraction/{trait}/{variant}/logit_lens.json` with a schema the visualization understands, and fix the broken run-hint in `visualization/views/extraction.js`.

## Complexity
Medium — 2 stages, 10 steps

## Success Criteria
- [ ] `analyze_trait()` returns new schema: `{trait, component, position, n_layers, methods: {<method>: {late: {layer, pct, toward, away}}}}`
- [ ] `--save` writes to `experiments/{exp}/extraction/{trait}/{variant}/logit_lens.json` (canonical path from `config/paths.yaml` → `extraction.logit_lens`)
- [ ] Dynamic method discovery via `discover_vectors()` — no hardcoded method list
- [ ] Late layer is the available layer closest to round(n_layers * 0.9), with max-available fallback
- [ ] pct = round(actual_layer / max(n_layers - 1, 1) * 100)
- [ ] `extraction.js:515-519` run-hint replaced with correct standalone command using `window.state.experimentData?.name`
- [ ] Backward compatible: existing 709 files are untouched; viz continues to render them
- [ ] Test: run on `bs/concealment` in `temp_llama_steering_feb18` — file lands at canonical path with correct schema
- [ ] Test: diff vs reference file — same top-level keys, same `methods.{method}.late` structure
- [ ] Test: trace through viz consumer in `extraction.js:545-563` with the new file — no `continue` on line 550
- [ ] Test: multi-method trait → all methods appear (bs/concealment has probe + mean_diff + gradient on disk)
- [ ] Test: fallback case → bs/concealment has layers 3-19 in a 32-layer model, so 19 should be chosen (not 29)

## Prerequisites
- `temp_llama_steering_feb18` experiment exists on disk with bs/concealment vectors at layers 3-19 for probe, mean_diff, gradient
- `utils/model.py` exports `get_inner_model`
- `analysis/vectors/extraction_evaluation.py:144` uses `mode="extraction"` as precedent

## Stage 1: Rewrite analyze_trait() and save path (7 steps)

### 1.1: Read final set of files before editing
**Purpose**: verify remaining assumptions, catch anything missed
**File(s)**:
- `analysis/vectors/logit_lens.py` (full, again)
- `utils/paths.py` — signature of `get_model_variant`, `desanitize_position`
- `analysis/vectors/extraction_evaluation.py:144` — precedent for `mode="extraction"`
- `utils/model.py` — `get_inner_model` signature
**Verify**: all these symbols exist, understand their contracts
**If wrong**: revise plan before editing

### 1.2: Rewrite `analyze_trait()` signature and body
**Purpose**: produce new method-keyed-late schema
**File**: `analysis/vectors/logit_lens.py`
**Change**:
- Signature: `(experiment, trait, model_variant, model, tokenizer, top_k=20, apply_norm=True, common_mask=None) -> Optional[dict]` (no component/position args — auto-detect)
- Body:
  1. `candidates = discover_vectors(experiment, trait, model_variant)`
  2. If empty → return `None`
  3. Sort candidates deterministically: `key=lambda c: (c["component"] != "residual", c["position"] != "response[:5]", c["component"], c["position"], c["method"], c["layer"])` — preferred comp/pos sort first
  4. Pick `(component, position)` from the first candidate after sort
  5. Filter candidates to that `(component, position)` pair
  6. `n_layers = get_inner_model(model).config.num_hidden_layers`
  7. `target = round(n_layers * 0.9)`
  8. Group filtered candidates by method → `{method: [layers]}`
  9. For each method:
     - `chosen_layer = min(available, key=lambda L: (abs(L - target), -L))` (closest, ties prefer higher)
     - `vector, _baseline, _meta = load_vector_with_baseline(experiment, trait, method, chosen_layer, model_variant, component, position)` ← NOTE: returns 3-tuple, unpack!
     - `pct = round(chosen_layer / max(n_layers - 1, 1) * 100)`
     - `vocab = vector_to_vocab(vector, model, tokenizer, top_k, apply_norm, common_mask)` → `{toward, away}`
     - `methods[method] = {"late": {"layer": chosen_layer, "pct": pct, "toward": vocab["toward"], "away": vocab["away"]}}`
  10. Return `{"trait": trait, "component": component, "position": position, "n_layers": n_layers, "methods": methods}`
- Also update imports:
  - ADD: `from utils.vectors import discover_vectors, load_vector_with_baseline`
  - ADD: `from utils.model import load_model_with_lora, get_inner_model`
  - REMOVE: `from utils.vector_selection import select_vector` (unused after rewrite)
  - REMOVE: `from utils.vectors import load_vector_with_baseline` if it was already imported separately (merge into the discover_vectors import line)
**Verify**: function imports `discover_vectors` and `get_inner_model`, uses tuple unpacking on `load_vector_with_baseline`, returns exactly the documented shape. No unused imports remain.
**If wrong**: KeyError at save time, or viz skips the file on line 550

### 1.3: Update `main()` model_variant resolution
**Purpose**: use extraction mode so vectors + write path are consistent
**File**: `analysis/vectors/logit_lens.py`
**Change**: `get_model_variant(args.experiment, args.model_variant, mode="application")` → `mode="extraction"`
**Verify**: grep confirms change, matches precedent at `analysis/vectors/extraction_evaluation.py:144`
**If wrong**: LoRA experiments silently write to wrong variant dir; viz doesn't find files

### 1.4: Update `main()` save logic
**Purpose**: write per-trait to canonical path, not aggregated to analysis/vector_logit_lens/
**File**: `analysis/vectors/logit_lens.py`
**Change**: Replace the `if args.save:` block entirely. New logic (inside the trait loop, right after `analyze_trait(...)` returns, NOT at end):
```python
if args.save and results is not None:
    output_path = get_path('extraction.logit_lens', experiment=args.experiment, trait=trait, model_variant=model_variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_path}")
```
Remove the `all_traits.json` aggregate entirely.
**Verify**: per-trait files land at canonical paths; no `all_traits.json` written
**If wrong**: files in wrong place; viz doesn't find them

### 1.5: Update `print_results()` for new shape
**Purpose**: script still prints sensible output without --save
**File**: `analysis/vectors/logit_lens.py`
**Change**:
- Guard against None: `if results is None: print("  No vectors found"); return`
- Iterate `results["methods"]`, for each method print the `late` block (method name, layer, toward list, away list)
- Remove dead references to old top-level keys (`results["layer"]`, `results["method"]`, `results["source"]`, `results["delta"]`)
- Remove the dead `all_results = []` accumulator from `main()` (no longer used since saving happens per-trait inside the loop)
**Verify**: `python analysis/vectors/logit_lens.py --experiment <exp> --trait <trait>` (no --save) prints token lists for each method without KeyError; a trait with no vectors prints "No vectors found" instead of crashing
**If wrong**: crashes on old flat-shape keys (layer, method at top level); or None passed to print_results

### 1.6: Update docstring and usage examples
**Purpose**: remove stale "stage 5" claim
**File**: `analysis/vectors/logit_lens.py` lines 19-21
**Change**: Remove the "runs automatically as stage 5 of the extraction pipeline" sentence. Replace with: "Standalone script — run manually. Writes per-trait files to the canonical extraction.logit_lens path."
**Verify**: grep for "stage 5" in this file returns no match
**If wrong**: stale docs remain

### 1.7: Fix viz run-hint
**Purpose**: empty-state hint currently points to nonexistent pipeline stage
**File**: `visualization/views/extraction.js` lines 515-520
**Change**: Replace the `renderRunHint` call with template literal that interpolates experiment name:
```js
const expName = window.state.experimentData?.name || '<exp>';
container.innerHTML = renderRunHint(
    'No logit lens data.',
    `python analysis/vectors/logit_lens.py --experiment ${expName} --all-traits --save`
);
```
**Verify**: grep `--only-stage 5` in extraction.js returns no match; the new hint uses backticks for template literal syntax
**If wrong**: hint still shows dead command

### Checkpoint: After Stage 1
- [ ] `analyze_trait()` returns new shape (verified by reading the function)
- [ ] `discover_vectors` + `get_inner_model` imported and used
- [ ] `mode="extraction"` set in get_model_variant call
- [ ] Save path uses `get_path('extraction.logit_lens', ...)`
- [ ] `extraction.js` run-hint updated with backticks
- [ ] Python syntax check: `python -c "import ast; ast.parse(open('analysis/vectors/logit_lens.py').read())"`
- [ ] Stage judgment complete (reflect on whether any assumptions were violated during implementation)

## Stage 2: Verification (3 steps)

### 2.1: Smoke test on bs/concealment (multi-method + fallback case)
**Purpose**: one trait covers multi-method AND fallback-to-latest-available tests
**Depends on**: Stage 1 complete
**Commands**:
```bash
cd /Users/ewern/Desktop/code/trait-stuff/traitinterp
python analysis/vectors/logit_lens.py --experiment temp_llama_steering_feb18 --trait bs/concealment --save
ls -la experiments/temp_llama_steering_feb18/extraction/bs/concealment/base/logit_lens.json
python -m json.tool experiments/temp_llama_steering_feb18/extraction/bs/concealment/base/logit_lens.json | head -60
```
**Verify**:
- File exists at canonical path
- Top-level keys: `{trait, component, position, n_layers, methods}`
- `methods` has entries for `probe`, `mean_diff`, `gradient` (all three)
- Each method has `late` (not `mid`)
- `late.layer` is 19 (the max available, since 90% of 32 = 29 but available layers are 3-19)
- `late.pct` is round(19/31*100) = 61
- `late.toward` and `late.away` are non-empty lists of `{token, value}` dicts
**If wrong**:
- Missing methods → check `discover_vectors` return filtering by component/position
- layer != 19 → check closest-layer algorithm, verify `min(available, key=...)` tiebreak
- pct is 90 (hardcoded) instead of 61 → check `pct = round(layer / max(n_layers - 1, 1) * 100)` formula
- File in wrong path → check `get_path('extraction.logit_lens', ...)` call

### 2.2: Key-level diff vs reference file
**Purpose**: confirm schema matches what viz consumer reads
**Depends on**: 2.1 passed
**Commands**:
```bash
python3 -c "
import json
d = json.load(open('experiments/temp_llama_steering_feb18/extraction/bs/concealment/base/logit_lens.json'))
print('top:', sorted(d.keys()))
print('methods:', sorted(d['methods'].keys()))
for m, v in d['methods'].items():
    print(f'  {m}:', sorted(v.keys()))
    if 'late' in v:
        print(f'    late:', sorted(v['late'].keys()))
        if v['late'].get('toward'):
            print(f'    toward[0]:', sorted(v['late']['toward'][0].keys()))
"
```
**Verify**:
- top: `['component', 'methods', 'n_layers', 'position', 'trait']`
- methods: includes `probe`, `mean_diff`, `gradient` (sorted)
- each method: `['late']`
- late: `['away', 'layer', 'pct', 'toward']`
- toward[0]: `['token', 'value']`
**If wrong**: structural mismatch → fix analyze_trait return dict

### 2.3: Trace viz consumer logic
**Purpose**: confirm the new file would actually render in the dashboard
**Depends on**: 2.2 passed
**Commands**: read `visualization/views/extraction.js:545-563` and manually trace with the actual file contents
**Verify**:
- Line 547: `methodPriority = ['probe', 'mean_diff', 'gradient']`
- Line 548: `method = methodPriority.find(m => data.methods[m]) || Object.keys(data.methods)[0]` → picks `probe`
- Line 549: `methodData = data.methods['probe']`
- Line 550: `methodData.late` is truthy → no continue
- Line 553: `late = methodData.late`
- Line 558: `late.layer` → 19 (renders "L19" in the table)
- Line 559: `late.toward` → renders tokens
- Line 560: `late.away` → renders tokens
**If wrong**: any of these keys missing or at wrong nesting → regression in analyze_trait

### Checkpoint: After Stage 2
- [ ] All 3 verification steps pass
- [ ] No regressions in other files (git status shows only the 2 intended files touched: analysis/vectors/logit_lens.py, visualization/views/extraction.js)
- [ ] Notepad up to date with all step results

## If Stuck
- `discover_vectors` returns empty → check the vectors dir exists on disk at `experiments/{exp}/extraction/{trait}/{variant}/vectors/`, check position sanitization (`response[:5]` → `response__5`)
- `get_inner_model(model).config.num_hidden_layers` throws → fall back to `model.config.num_hidden_layers`; if that fails too, derive from model structure or skip
- `vector_to_vocab` returns empty toward/away → vector on wrong device/dtype; `vector_to_vocab` should auto-move but check if something's wrong
- File lands but viz traces would skip → check `methods.{method}.late` key exists; check priority order matches
- Model loading is slow (several minutes for Llama 70B) → this is expected; run the test on a smaller experiment if possible, or accept the wait
- `get_path('extraction.logit_lens', ...)` raises → check config/paths.yaml key name; may have been renamed

## Notes
- `bs/concealment` in `temp_llama_steering_feb18` is perfect for testing: multi-method (probe, mean_diff, gradient) AND fallback case (layers 3-19 in 32-layer model). One trait covers 2 of the 5 user-requested tests.
- Model loading happens once per run; all traits share the same loaded model. The loop is fast after that.
- The 709 existing files are not touched — only new runs write to the canonical path. Backward compat is automatic since the viz ignores `mid` anyway.
