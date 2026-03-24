# Codebase Refactor Notepad

## Status

OOM dedup done. normalize_projections wired. Dead logit lens code deleted. resolve_layers moved. Steering records typed with SteeringRunRecord + SteeringResults dataclasses. All pipeline + dev/ consumers updated. Critic-found bugs fixed. process_activations.py split into `capture_activations.py` + `project_activations.py` (standalone CLI deleted — use `inference/run_inference_pipeline.py`). Model loading done: `ServerBackend`, `tokenize_with_prefill()`, `load_model_or_client()` deleted. Hook renames done. project_activations bug fixed. Inference pipeline flags updated: `--skip-generate` replaced by `--regenerate`, `--capture` added.

**Next priority:** normalized projections as first-class, remaining typing from typing_audit.md.

---

## Completed Refactoring

### This session (continued)
- **process_activations.py split** — split into `utils/capture_activations.py` (capture raw activations to .pt) and `utils/project_activations.py` (project activations onto trait vectors). Standalone CLI deleted — use `inference/run_inference_pipeline.py` instead. Inference pipeline flags updated: `--skip-generate` replaced by `--regenerate` (default now skips if responses exist), `--capture` added as new flag.
- **Dead code deleted** — `ServerBackend` from `utils/backends.py`, `tokenize_with_prefill()` and `load_model_or_client()` from `utils/model.py`.
- **OOM recovery loop dedup** — extracted `check_oom_exception()` + `recover_oom_batch_size()` to `batch_forward.py`. Wired into capture_activations.py (fixed missing `continue` bug), extract_vectors.py, model_generation.py. Each callsite went from ~15 lines to 6.
- **resolve_layers moved** from `process_activations.py` to `utils/layers.py` (was cross-imported from process_activations into vector_selection).
- **Dead logit lens code deleted** — `compute_logit_lens_from_raw()`, `LOGIT_LENS_LAYERS` constant, `--logit-lens` CLI flag, `logit_lens` parameter in `process_prompt_set` (computed but never stored).
- **normalize_projections wired** into `read_projection()` and `read_response_projections()` via `mode` parameter ('raw'/'normalized'/'cosine'). Analysis scripts can now get cross-layer comparable projections.
- **test_math.py fixed** — removed import of deleted `separation()` function, dropped 2 tests that called it.
- **Steering records typed** — `SteeringRunRecord` + `SteeringResults` dataclasses in `core/types.py`. `load_results()` returns typed `SteeringResults`. `find_cached_run()` returns `Optional[JudgeResult]`. `get_baseline()` returns `Optional[JudgeResult]`. `append_run()` adds `"type": "run"` to new writes. Consumers updated: vector_selection.py, steering_eval.py, coefficient_search.py, serve.py + 4 dev/ files. Critic caught 7 bugs (2 critical, 5 dev-only), all fixed. `run_rescore()` stays raw-dict zone.

### Previous session
- **ProjectionEntry + ProjectionRecord** in `core/types.py` — canonical projection JSON schema. `to_dict(precision=4)` for float rounding. Both write sites (stream-through + from-activations) wired.
- **ResponseRecord wired** at 2 write sites (`save_response_json`, fallback writer). Added 3 optional extension fields (`turn_boundaries`, `source`, `sentence_boundaries`). Fixed type hints (`Optional[str]`).
- **normalize_projections()** in `core/math.py` — 3 modes: raw, normalized (divide by mean activation norm), cosine (per-token). Pure function, no torch dependency.
- **serve.py cleanup** — deleted `cache_integrity_data()`, `/api/schema`, `/api/integrity`, `/inference/prompts/`, `/model-variants` endpoints. Removed `yaml` import. -117 lines.
- **Massive dims decoupled** from projection path. Removed `massive_dim_data` from `ProjectionRecord`, deleted `load_massive_dims_from_analysis()`, `extract_massive_dim_values()`, all massive dim loading/passing in both write paths. ~60 lines removed. `analysis/massive_activations.py` still exists as standalone tool.
- **Dead code removed** — `compute_activation_norms()` (never called after activation_norms dropped from projection JSON).
- **Docs updated** — main.md, workflows.md, extraction_guide.md, architecture.md, core_reference.md for dataset restructure + typed returns + dead function removal.
- **Naming principle** added to CLAUDE.md — function names should make behavior obvious without reading implementation.
- **Projection field audit** — investigator mapped all 19 metadata fields. 10 never read by any consumer. Kept for provenance (trivial size). Dropped `activation_norms` (dead) and `logit_lens` (frontend loads from different path).
- **Normalization deep dive** — 3 investigators mapped all normalization across codebase + literature. No analysis script normalizes projections at all — only frontend JS does. Normalized mode (divide by mean activation norm) is preferred default.

### Earlier sessions
~110 → ~47 → 3 recipe files. 32 dead functions deleted (-556 lines). 58 unused imports removed. 4 runtime bugs fixed. 4 typed dataclasses (VectorResult, JudgeResult, ModelVariant, SteeringEntry). Datasets restructured (base/instruct/starter_traits/archive). pyproject.toml. Repo renamed to traitinterp. See git history.

---

## Known Issues

- 43 experiment scripts have broken `get_best_vector` imports (experiments/ only, not pipeline)
- Trait paths: datasets 3-level (base/emotion_set/X), experiments 2-level (emotion_set/X). See TODO below.
- `valid` key in steering_results.py never written to disk
- Analysis scripts use raw projections without normalizing — cross-layer comparisons invalid (per_token_diff, compare_variants, layer_sensitivity). `read_projection(mode='normalized')` now available but not wired into callers.
- Steering run JSONL entries have no `"type"` key (footgun — header has "header", baseline has "baseline", run has nothing)
- `get_layer_path_prefix` import in hooks.py broken (test_hooks fails)

---

## TODO

### Trait path consistency (3-level everywhere)
Currently datasets use 3-level paths (`base/emotion_set/anger`) but experiments use 2-level (`emotion_set/anger`). The first level maps to a model variant defined in experiment config.json.

**Target state:**
- `--traits base/emotion_set/anger` works everywhere (extraction, inference, steering, analysis)
- Experiment paths: `experiments/{exp}/extraction/{model_variant}/base/emotion_set/anger/vectors/...`
- Config maps variant → trait category: `{"variants": {"base": {"model": "..."}, "instruct": {"model": "..."}}}`
- `discover_extracted_traits()` returns 3-level tuples

**Scope:**
- PathBuilder (`utils/paths.py`) — add variant prefix to trait paths in experiment templates
- `discover_extracted_traits()` — walk one level deeper
- `--traits` flag parsing in all 3 pipelines
- Existing experiment data needs migration (or regeneration)
- Since everything flows through PathBuilder, the change should be contained — update the path templates and the discovery function, most consumers won't change.

**Do after file restructure** — easier to reason about with clean file names. Look for opportunities to fold in during restructure.

### ~~process_activations.py further cleanup~~ DONE
Split into `capture_activations.py` + `project_activations.py`. Standalone CLI deleted.

### Vector loading convergence (lower priority)
4 implementations serve different use cases:
- `load_trait_vectors` — stream-through GPU batched (stacked tensors + hook_index)
- `process_prompt_set` — post-hoc CPU, supports multi-vector + method auto-detect
- `chat_inference._load_trait_vectors` — simplest, one vector per trait
- `steering_eval.load_vectors` — minimal, computes base_coef
Selection logic genuinely differs. Could extract shared `_load_single_vector()` helper but not high priority.

### Model loading convergence
`ServerBackend`, `tokenize_with_prefill()`, `load_model_or_client()` deleted. Remaining files call `load_model_with_lora()` directly. Should funnel through `LocalBackend.from_experiment()` which already exists.

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
- `model.py` / `backends.py` — potential overlap (ServerBackend deleted, but review remaining)
- `coefficient_search.py` (656 lines) — may be appropriately complex

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
