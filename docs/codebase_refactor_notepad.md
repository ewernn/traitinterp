# Codebase Refactor Notepad

## Status

Typed projection dataclasses done. Normalization deep-dived. Massive dims decoupled. Cross-file dedup opportunities identified. process_activations.py audited — needs restructure.

**Next priority:** Cross-cutting dedup (OOM loop, model loading, vector loading), steering records typing, wire normalize_projections into analysis scripts, process_activations.py restructure.

---

## Completed Refactoring

### This session
- **ProjectionEntry + ProjectionRecord** in `core/types.py` — canonical projection JSON schema. `to_dict(precision=4)` for float rounding. Both write sites (stream-through + from-activations) wired.
- **ResponseRecord wired** at 2 write sites (`save_response_json`, fallback writer). Added 3 optional extension fields (`turn_boundaries`, `source`, `sentence_boundaries`). Fixed type hints (`Optional[str]`).
- **normalize_projections()** in `core/math.py` — 3 modes: raw, normalized (divide by mean activation norm), cosine (per-token). Pure function, no torch dependency.
- **serve.py cleanup** — deleted `cache_integrity_data()`, `/api/schema`, `/api/integrity`, `/inference/prompts/`, `/model-variants` endpoints. Removed `yaml` import. -117 lines.
- **Massive dims decoupled** from projection path. Removed `massive_dim_data` from `ProjectionRecord`, deleted `load_massive_dims_from_analysis()`, `extract_massive_dim_values()`, all massive dim loading/passing in both write paths. ~60 lines removed. `analysis/massive_activations.py` still exists as standalone tool.
- **Dead code removed** — `compute_activation_norms()` (never called after activation_norms dropped from projection JSON), `compute_logit_lens_from_raw()` (never called, `--logit-lens` flag loaded model for nothing), `logit_lens_data` parameter removed from `project_prompt_onto_traits`.
- **Docs updated** — main.md, workflows.md, extraction_guide.md, architecture.md, core_reference.md for dataset restructure + typed returns + dead function removal.
- **Naming principle** added to CLAUDE.md — function names should make behavior obvious without reading implementation.
- **Projection field audit** — investigator mapped all 19 metadata fields. 10 never read by any consumer. Kept for provenance (trivial size). Dropped `activation_norms` (dead) and `logit_lens` (frontend loads from different path).
- **Normalization deep dive** — 3 investigators mapped all normalization across codebase + literature. No analysis script normalizes projections at all — only frontend JS does. Normalized mode (divide by mean activation norm) is preferred default.

### Previous sessions
~110 → ~47 → 3 recipe files. 32 dead functions deleted (-556 lines). 58 unused imports removed. 4 runtime bugs fixed. 4 typed dataclasses (VectorResult, JudgeResult, ModelVariant, SteeringEntry). Datasets restructured (base/instruct/starter_traits/archive). pyproject.toml. Repo renamed to traitinterp. See git history.

---

## Known Issues

- OOM recovery bug in `capture_raw_activations` (process_activations.py) — missing `continue` after halving batch size, silently skips batches instead of retrying
- 43 experiment scripts have broken `get_best_vector` imports (experiments/ only, not pipeline)
- Trait paths: datasets 3-level (base/emotion_set/X), experiments 2-level (emotion_set/X)
- `valid` key in steering_results.py never written to disk
- Analysis scripts use raw projections without normalizing — cross-layer comparisons invalid (per_token_diff, compare_variants, layer_sensitivity)
- Steering run JSONL entries have no `"type"` key (footgun — header has "header", baseline has "baseline", run has nothing)

---

## TODO

### Cross-cutting dedup (highest priority — fixes multiple files at once)
1. **OOM recovery loop** — duplicated in process_activations.py, extract_vectors.py, generation.py (~35 lines each). Extract `batched_forward_with_oom_recovery()` to `batch_forward.py`. Also fixes the silent-skip bug in process_activations.
2. **Model loading** — 5 files call `load_model_with_lora()` directly. Should funnel through `LocalBackend.from_experiment()` which already exists.
3. **Vector loading** — 4 implementations (process_activations 130 lines, vector_selection `load_trait_vectors`, chat_inference `_load_trait_vectors`, steering_eval `load_vectors`). Converge on `load_trait_vectors`.

### process_activations.py restructure
- 1004 lines doing 3 jobs: projection, capture, process-from-saved
- Misplaced functions: `resolve_layers` → `utils/layers.py`, `project_onto_vector` → inline (thin wrapper over `core.projection()`), `compute_token_norms` → inline (3 lines)
- `process_prompt_set` (250 lines) reimplements vector loading that `load_trait_vectors` already does
- Consider splitting: projection logic + helpers vs capture-raw vs process-from-saved

### Steering Records typing
Scoped by 2 investigators. Design proposed: SteeringHeader, SteeringBaseline, SteeringRunRecord, SteeringResponseRecord, SteeringVectorSource. ~60 `.get()` → attribute access across 12 files, 120-160 lines changed. `run_rescore()` stays as raw-dict zone. Add `"type": "run"` to new writes.

### Wire normalize_projections into analysis scripts
- `read_projection()` in `utils/projections.py` — add `mode` param, normalize on read
- Fixes cross-layer comparison bugs in per_token_diff.py, compare_variants.py, layer_sensitivity.py
- ~30 min

### Normalized projections as first-class
- Store pre-normalized scores in projection JSON with `"score_mode": "normalized"` metadata flag
- Keep per-token norms for cosine conversion: `normalized * mean_norm / token_norms[t]`
- Raw recovery: `normalized * mean_norm`
- Frontend default: change state.js from `'cosine'` to `'normalized'`
- Schema-breaking for existing files — regenerate when needed

### Remaining typing (from typing_audit.md)
4. **Activation Metadata** — 18 keys, 5 consumers
5. **Vetting Scores** — 3 consumers, nested structure
6. **SteeringResults wrapper** — optional, wraps load_results return for full end-to-end typing

### Large files to audit
- `steering_eval.py` (904 lines) — duplicate model loading, raw JSONL manipulation
- `model.py` (705 lines) / `backends.py` (664 lines) — potential overlap
- `coefficient_search.py` (656 lines) — may be appropriately complex
- `generation.py` (618 lines) — has OOM loop copy

### Public release
- Extract + steer 9 starter traits on a model
- Run pre-release audit (trufflehog, vulture, pip-audit)
- Promote to main
- HuggingFace Hub for experiment data

### Features / future
- Ensemble projections — weighted combination across layers (requires normalized projections first)
- Massive dims as optional enrichment step (separate from projection path)
- Trait dataset format redesign (trait.json + scenarios.json) — defer to pip package
- pip package API design (`import traitinterp`)
- Paper experiment (LIARS' BENCH — separate chat)

---

## Architecture Decisions (settled)

- **1 recipe file per pipeline dir** — thin controllers delegating to utils/
- **core/** = pure primitives (types, hooks, math, methods). No upward deps. torch-free types.
- **utils/** = library code. Pipeline helpers live here.
- **Typed returns** — public functions return dataclasses, not dicts. VectorResult, JudgeResult, ProjectionEntry, ProjectionRecord, ResponseRecord.
- **Recursive discovery** — no hardcoded directory depth. Walk until leaf marker found.
- **base/instruct split** — datasets/traits/base/ vs datasets/traits/instruct/ tells users how to extract
- **emotion_set is canonical** — wins 9/12 head-to-heads at 2-4x lower coefficients
- **Don't rename trait categories** — paths embedded in experiment dirs + JSON metadata
- **Base model traits on dev until paper** — natural elicitation is novel contribution
- **ruff for linting** — replaces flake8/black/bandit
- **Modal stays in dev/** — bypasses backend abstraction, not integrated into main pipelines
- **Normalized projections default** — divide by mean activation norm at layer. Raw stored, normalized at read/display time. Cosine available as alternative.
- **Massive dims separate from projection** — standalone calibration tool, not tangled into projection path

### Code Design Principles (for future agents)

- **Zero duplicate code.** If the same block appears twice, extract a helper.
- **Typed returns over dicts.** Public functions return dataclasses. `.get('key', 0)` is a silent bug waiting to happen.
- **Flags over function proliferation.** One function with a parameter > two 80%-identical functions.
- **Pipeline recipes stay thin.** 200-360 line table-of-contents. Drill into utils/ for how.
- **Spawn subagents liberally.** Critics before implementing, investigators for exploration.
- **content_hash for provenance.** Store hashes at save-time, check at load-time.
- **Self-documenting names.** Function names should make behavior obvious without reading the implementation. Not so specific that args contradict the name, not so vague that you have to read the source to know what happens.
- **Store raw, normalize at consumption.** Projection files store raw dot products + per-token norms. Normalization happens at read/display time so you can switch modes.
