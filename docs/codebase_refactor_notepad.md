# Codebase Refactor Notepad

## Status

Major cleanup complete. ~30k lines removed, duplicates consolidated, shared helpers extracted, bugs fixed.

**Next priority:** README rewrite.

---

## Completed Refactoring

### Summary
~110 → ~47 pipeline Python files (waves 1-8), then 22 → 3 recipe files (thin controller + pipeline reduction). See git history for details.

### Key milestones
- **Thin recipe pattern**: 3 pipeline scripts (extraction, inference, steering) each 200-360 lines. Library code in utils/.
- **Config dataclasses** (`core/kwargs_configs.py`): SteeringConfig, ExtractionConfig, InferenceConfig, VettingStats
- **In-memory activation→vector flow**: no .pt roundtrip by default. `--save-activations` to persist.
- **Deleted `other/`** entirely. server → `utils/server/`, sae → `analysis/sae/`
- **File merges**: steered_generation → generation, fingerprints → projections, vet_scenarios+vet_responses → vet(target=)
- **New shared primitives**: `utils/positions.py` (multiturn DSL), `utils/batch_forward.py` (OOM/TP/calibration helpers)
- **12+ helpers extracted**: clear_oom_traceback, tp_agree_count/batch_size, calibrate_batch_size, resolve_use_chat_template, build_response_records, summarize_judge_scores, _update_coefficients, content_hash, score_stats, _write_responses, _build_vetting_output
- **Coefficient search**: deleted adaptive_search_layer + evaluate_single_config (-210 lines), max_batch_layers=1 for sequential
- **Hashing infra**: content_hash(path) in paths.py, vector metadata stores input_hashes
- **Vector selection redesign**: `select_vector` / `select_vectors` with shared `_select_vectors`, dedupe-by-layer, `sort_by` param, unscored fallback for manual selection, `discover_vectors` moved to vectors.py, deleted 2 dead functions, updated 15+ callers
- **Default to probe only**: ExtractionConfig.methods now `['probe']`, CLI default updated
- **Hashing at all save points**: response metadata.json, vetting output, vector metadata all store `input_hashes`; steering results header stores `prompts_hash`
- **Hash-based staleness detection**: `_get_steering_result` compares stored `prompts_hash` against current `steering.json`, falls back to mtime for old files
- **Bug fixes**: steering eval signatures, compare_variants NameError, list_layers arg order, core/generation.py upward import, flush_cuda synchronize, 4 dev/ broken imports

---

## Known Issues

- `valid` key in steering_results.py never written to disk
- `activation_norms` has 3 different schemas across files
- Response JSON written in 2 places with schema drift
- Stream-through mode skips massive_dim_data (requires full activations)
- iCloud Desktop sync creates " 2" files constantly

---

## TODO

### Code cleanup
- **Coefficient search: remaining helpers** — `_flush_best_responses` (~15-20 lines savings, low priority)

### Documentation
- **README rewrite** — story-driven walkthrough using `hyperparams` throughout

### Features / future
- Top-activating spans tool
- Optimize vetting responses (batching, caching)
- Data storage for users (non-R2 options)
- Per-trait layer config
- Visualization audit
- Upload custom LoRAs to HuggingFace
- **Best-layer agreement analysis** — compare best layer by probe accuracy (extraction) vs best layer by steering delta across emotion_set traits (|delta|>30). Find distribution of offsets. Could enable skipping steering eval for layer selection.

---

## Architecture Decisions (settled)

- **1 recipe file per pipeline dir** — recipe at top, stage implementations below, CLI at bottom
- **Config dataclasses** (`core/kwargs_configs.py`) — replace individual args threading
- **core/** = pure primitives (types, hooks, math, methods). No upward dependencies.
- **utils/** = library code. Pipeline-specific helpers live here, not in pipeline dirs.
- **In-memory activation→vector** — no .pt roundtrip by default. `--save-activations` to persist.
- **Hook-based projection** — MultiLayerProjection does capture+project vectorized on GPU in one pass.
- **Positive CLI flags** — `--logitlens` not `--no-logitlens`. `--no-vet-responses` to disable defaults.
- **vet_scenarios not a pipeline stage** — standalone dev tool
- **Logit lens not in extraction recipe** — opt-in standalone tool
- **Steering is a separate pipeline** — extraction prints hint, doesn't embed it
- **VettingStats quality gate** — ≥10 responses per polarity to proceed
- **Promote script** — main is exact mirror of .publicinclude, stale files auto-removed
- **dev/ tracked on dev branch only**, not in .publicinclude
- **Thin recipe pattern is the right architecture** — pipeline files (200-360 lines) are readable table-of-contents that delegate to utils/. Absorbing utils/ implementations into pipeline files would create 1000+ line monoliths that bury the high-level flow. A new user reads the recipe to understand what happens, drills into utils/ for how.
- **Don't merge vectors.py + vector_selection.py + activations.py** — they serve different abstraction levels ("load this specific tensor" vs "decide which tensor to load" vs "raw activation I/O"). Merging conflates concerns and creates an 800+ line file where every consumer only uses a subset.
- **Don't create a monolithic `batched_forward` function** — the 3 OOM recovery sites (extraction, inference capture, generation) differ in 5 dimensions (item format, hook type, use_cache, keep_on_gpu, post-forward processing). Extract shared helpers instead: `clear_oom_traceback`, `tp_agree_count`, `calibrate_batch_size`. Each call site keeps its own forward loop.
- **Don't flag-ify `save_responses` / `save_baseline_responses` / `save_ablation_responses`** — they have genuinely different parameter sets. A unified `save_responses(kind=)` with Optional params is worse because the type system can't enforce which params matter for which kind. Three explicit functions with clear names is better than one with hidden constraints.

### Code Design Principles (for future agents)

- **Zero duplicate code.** If the same block appears twice, extract a helper. Use `grep` to find all copies before assuming something is unique.
- **Flags over function proliferation.** If two functions share >80% logic and differ by a mode, make it one function with a parameter. Example: `vet(target="scenarios"|"responses")` not `vet_scenarios()` + `vet_responses()`. Exception: when parameter sets genuinely differ (like the save_responses family).
- **Shared helpers in utils/, not inline.** OOM recovery, TP sync, batch calibration, score aggregation — these are shared primitives, not pipeline-specific code. They live in utils/ and callers import them.
- **Pipeline recipes stay thin.** Don't absorb utils/ implementations into pipeline files. The recipe is a 200-360 line table-of-contents. A user reads it to learn WHAT happens. They drill into utils/ for HOW.
- **Vectorize wherever possible.** Prefer tensor operations over Python loops. Example: mean_diff across layers can be one stacked tensor op instead of N loop iterations.
- **Critics catch what investigators miss.** Always run a critic agent before implementing anything non-trivial. Investigators find patterns (similarity); critics find divergences (differences that break assumptions).
- **content_hash for provenance.** `content_hash(path)` in utils/paths.py hashes any file. Store hashes in metadata at save-time. Use for staleness detection at load-time (e.g., select_vector checking if vectors match current scenarios).