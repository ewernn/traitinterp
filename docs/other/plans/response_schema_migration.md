# Response Schema Migration Plan

**Status: COMPLETE** (2026-01-20, updated 2026-01-25)

Migrated all response files to the unified schema defined in `docs/response_schema.md`.

---

## Summary

- **Extraction:** 78 files migrated
- **Steering:** 5,583 files migrated
- **Inference:** 1,735 files migrated (+ 305 jailbreak responses flattened 2026-01-25)
- **Code:** All writers/readers updated

### Flat Schema Update (2026-01-25)

Flattened inference response schema - removed `metadata` wrapper, moved fields to top level:
- `inference_model`, `prompt_note`, `capture_date`, `tags` now at root level
- Removed derivable fields: `inference_experiment`, `prompt_set`, `prompt_id`, `lora_adapter`
- Reorganized jailbreak dataset: `jailbreak.json` → `jailbreak/original.json`
- Deleted unused files: `subset.json`, `successes.json` (tags now in response files)

---

## Historical Context

Migration details below for reference.

---

## Current State

### Three Response Formats

**Extraction** (`extraction/{trait}/{variant}/responses/pos.json`, `neg.json`):
```json
{
  "scenario_idx": 0,
  "rollout_idx": 0,
  "prompt": "...",
  "system_prompt": "...",
  "response": "...",
  "full_text": "...",
  "prompt_token_count": 84,
  "response_token_count": 192
}
```

**Inference** (`inference/{variant}/responses/{prompt_set}/{id}.json`):
```json
{
  "prompt": {"text": "...", "tokens": [...], "token_ids": [...], "n_tokens": 48},
  "response": {"text": "...", "tokens": [...], "token_ids": [...], "n_tokens": 50},
  "metadata": {...}
}
```

**Steering** (`steering/{trait}/.../responses/baseline.json`):
```json
{
  "question": "...",
  "response": "...",
  "trait_score": 0.003,
  "coherence_score": 50.0
}
```

### Data Volume

- ~80 extraction file pairs (pos.json + neg.json)
- ~56 steering baseline files
- ~1736 inference response files
- **Total: ~1900 files**

### Files to Update

**Writers (5):**
- `extraction/generate_responses.py`
- `extraction/instruction_based_extraction.py`
- `inference/capture_raw_activations.py`
- `analysis/steering/evaluate.py`
- `analysis/steering/coef_search.py`

**Readers (9):**
- `extraction/extract_activations.py`
- `inference/project_raw_activations_onto_traits.py`
- `analysis/steering/results.py`
- `utils/judge.py`
- `scripts/read_steering_responses.py`
- `visualization/core/prompt-picker.js`
- `visualization/components/response-browser.js`
- `visualization/components/custom-blocks.js`
- `visualization/views/trait-dynamics.js`

---

## Target Schema

**Inference (flat schema, 2026-01-25):**
```json
{
  "prompt": "...",
  "response": "...",
  "system_prompt": null,
  "tokens": [...],
  "token_ids": [...],
  "prompt_end": 45,
  "inference_model": "google/gemma-2-2b-it",
  "prompt_note": "roleplay",
  "capture_date": "2026-01-13T...",
  "tags": ["success"]
}
```

**Extraction/Steering:**
```json
{
  "prompt": "...",
  "response": "...",
  "system_prompt": null,
  "trait_score": null,
  "coherence_score": null
}
```

**Required:** `prompt`, `response`
**Optional:** everything else (null when not applicable)

---

## Migration Phases

### Phase 0: Migration Scripts

**Goal:** Create and test migration scripts before touching any production code.

**Tasks:**
- [ ] Create `scripts/migrate_responses.py` with:
  - `migrate_extraction_response(old) -> new`
  - `migrate_inference_response(old) -> new`
  - `migrate_steering_response(old) -> new`
  - `migrate_file(path, format_type)` — in-place migration
  - `migrate_directory(path, format_type)` — batch migration
- [ ] Add `--dry-run` flag to preview changes without writing
- [ ] Test on 1-2 files of each type manually
- [ ] Verify migrated files are valid JSON and have correct schema

**Checkpoint:** Migration script correctly transforms sample files of each type.

**Files created:**
- `scripts/migrate_responses.py`

---

### Phase 1: Extraction Pipeline

**Goal:** Migrate extraction responses and update extraction code.

**Tasks:**

1. **Migrate data:**
   - [ ] Run migration on all extraction response files
   - [ ] Verify file count unchanged
   - [ ] Spot-check a few migrated files

2. **Update writers:**
   - [ ] `extraction/generate_responses.py`
     - Remove: `scenario_idx`, `rollout_idx`, `full_text`, `prompt_token_count`, `response_token_count`
     - Keep: `prompt`, `response`, `system_prompt`
   - [ ] `extraction/instruction_based_extraction.py`
     - Same changes
     - Note: has additional fields like `instruction_idx`, `question_idx` — decide whether to keep

3. **Update readers:**
   - [ ] `extraction/extract_activations.py`
     - Currently reads `full_text` — derive from `prompt + response`
     - Currently reads `prompt_token_count` — re-tokenize or use `prompt_end` if available
   - [ ] `extraction/vet_responses.py`
     - Update field access (likely minimal changes)

**Checkpoint:** Run `python extraction/generate_responses.py` on one trait, verify output matches new schema.

**Commands:**
```bash
# Migrate extraction data
python scripts/migrate_responses.py --format extraction experiments/*/extraction/*/*/responses/

# Test extraction pipeline
python extraction/run_pipeline.py --experiment gemma-2-2b --traits chirp/refusal --steps generate
```

---

### Phase 2: Steering Pipeline

**Goal:** Migrate steering responses (`question` → `prompt`) and update steering code.

**Tasks:**

1. **Migrate data:**
   - [ ] Run migration on all steering response files (baseline.json and steered responses)
   - [ ] Verify `question` renamed to `prompt`

2. **Update writers:**
   - [ ] `analysis/steering/evaluate.py`
     - Line ~198: `"question": q` → `"prompt": q`
   - [ ] `analysis/steering/coef_search.py`
     - Line ~76: same change

3. **Update readers/utilities:**
   - [ ] `analysis/steering/results.py`
     - Update any field access
   - [ ] `utils/judge.py`
     - Template uses `{question}` → change to `{prompt}`
   - [ ] `scripts/read_steering_responses.py`
     - `r['question']` → `r['prompt']`

**Checkpoint:** Run steering evaluation on one trait, verify output and scores unchanged.

**Commands:**
```bash
# Migrate steering data
python scripts/migrate_responses.py --format steering experiments/*/steering/

# Test steering pipeline
python analysis/steering/evaluate.py --experiment gemma-2-2b --trait chirp/refusal --dry-run
```

---

### Phase 3: Inference Pipeline

**Goal:** Flatten nested inference response structure.

**Tasks:**

1. **Migrate data:**
   - [ ] Run migration on all inference response files (~1736 files)
   - [ ] This is the largest batch — may take a few minutes
   - [ ] Verify structure flattened correctly

2. **Update writers:**
   - [ ] `inference/capture_raw_activations.py`
     - Lines ~216-240: Flatten nested structure
     - `prompt.text` → `prompt`
     - `response.text` → `response`
     - Combine `prompt.tokens + response.tokens` → `tokens`
     - `len(prompt.tokens)` → `prompt_end`

3. **Update readers:**
   - [ ] `inference/project_raw_activations_onto_traits.py`
     - Lines ~502-524: Update field access
     - Use `prompt_end` to split tokens if needed

**Checkpoint:** Run inference capture on one prompt, verify output structure.

**Commands:**
```bash
# Migrate inference data
python scripts/migrate_responses.py --format inference experiments/*/inference/*/responses/

# Test inference pipeline
python inference/capture_raw_activations.py --experiment gemma-2-2b --prompt-set jailbreak --limit 1
```

---

### Phase 4: Frontend

**Goal:** Update JavaScript to read new schema.

**Tasks:**

1. **Update prompt-picker.js:**
   - [ ] `data.prompt?.text` → `data.prompt`
   - [ ] `data.response?.text` → `data.response`
   - [ ] `data.prompt?.tokens` → `data.tokens?.slice(0, data.prompt_end)`
   - [ ] `data.response?.tokens` → `data.tokens?.slice(data.prompt_end)`

2. **Update response-browser.js:**
   - [ ] `r.question` → `r.prompt`
   - [ ] Lines ~608, ~673

3. **Update custom-blocks.js:**
   - [ ] `r.question` → `r.prompt`
   - [ ] Line ~613

4. **Update trait-dynamics.js:**
   - [ ] Update `nPromptTokens` derivation to use `prompt_end`

5. **Update state.js:**
   - [ ] Update cache structure if needed

**Checkpoint:** Load visualization, navigate through views, verify no JS errors and data displays correctly.

**Commands:**
```bash
# Start visualization server
python visualization/serve.py

# Open browser, test:
# - Trait Dynamics view (uses prompt-picker)
# - Steering Sweep view (uses response-browser)
# - Any findings pages (use custom-blocks)
```

---

### Phase 5: Cleanup

**Goal:** Remove dead code, verify everything works.

**Tasks:**

- [ ] Search codebase for any remaining references to old fields:
  - `full_text`
  - `scenario_idx`
  - `rollout_idx`
  - `prompt_token_count`
  - `response_token_count`
  - `question` (in steering context)
  - `prompt.text`, `response.text` (nested access)
- [ ] Remove any compatibility shims or dead code
- [ ] Update `docs/response_schema.md` migration notes (mark as complete)
- [ ] Run full end-to-end test:
  - Extract a trait
  - Run inference
  - Run steering evaluation
  - View in frontend

**Checkpoint:** Full pipeline works, no references to old schema remain.

---

## Rollback Plan

If something goes wrong mid-migration:

1. **Git:** All code changes are tracked. `git checkout .` to revert code.
2. **Data:** Keep backup before migration:
   ```bash
   # Before starting
   cp -r experiments experiments_backup
   ```
3. **Partial migration:** Migration script should be idempotent — running on already-migrated files should be safe (detect and skip).

---

## Verification Checklist

After each phase, verify:

- [ ] No Python tracebacks when running pipeline
- [ ] No JS console errors in browser
- [ ] Output files match expected schema
- [ ] Existing functionality unchanged (scores, visualizations, etc.)

---

## Notes

- **Tokens optional:** Extraction and steering won't have tokens. Inference will.
- **prompt_end:** Only present when tokens are present.
- **system_prompt:** Keep explicit field, null when not used.
- **metadata.json:** Unchanged (stays alongside response files for extraction).
- **File structure:** No changes to directory structure or file naming.
