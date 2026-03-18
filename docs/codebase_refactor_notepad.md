# Codebase Refactor Notepad

## Status

Major cleanup complete. `other/` deleted, files merged, duplicates consolidated, bugs fixed. ~30k lines removed.

**Recent changes:**
- Deleted `other/` (tv, sae, server, mcp, analysis). Moved server → `utils/server/`, sae → `analysis/sae/`
- Merged `steered_generation.py` → `generation.py`, `fingerprints.py` → `projections.py`
- New `utils/positions.py` (position DSL with native multiturn), `utils/batch_forward.py` (OOM/TP/calibration helpers)
- Fixed 3 runtime bugs, fixed `core/generation.py` upward import, wired shared helpers into call sites
- Consolidated: `batched_steering_generate` (unified prompts= param), `load_activations` (split= param), `save_*_responses` (shared `_write_responses`), `dump_compact` delegates to `dumps_compact`

**Next priority:** Finish dedup helpers, coefficient search consolidation, README rewrite.

---

## Completed Refactoring

### Waves 1-8 (historical)
Took codebase from ~110 to ~47 pipeline Python files. See git history.

### Thin Controller Refactor
Helpers extracted, circular deps broken, vectors.py split. 8 bug fixes.

### Pipeline File Reduction (22 → 3 files)

**Phase 1: Merges + moves (22 → 8)**
- extraction/: merged extract_activations + extract_vectors + run_logit_lens → extract_vectors.py, inlined generate_responses
- steering/: moved steering_results → utils/, multi_layer_evaluate + weight_sources → dev/
- inference/: merged capture_activations + project_activations_onto_traits → process_activations.py

**Phase 2: Single-file recipes (8 → 3)**
- All library code moved to utils/: coefficient_search.py, extract_vectors.py, preextraction_vetting.py, process_activations.py, inference_generation.py, steering_eval.py
- Config dataclasses (`core/kwargs_configs.py`): SteeringConfig, ExtractionConfig, InferenceConfig, VettingStats
- Each pipeline script is a self-contained recipe: stages at top, implementations below, CLI at bottom
- `process_prompt_set` takes real keyword params (no argparse.Namespace hack)
- TP lifecycle via `tp_lifecycle()` context manager in `utils/distributed.py`
- `flush_cuda()` shared helper in `utils/distributed.py`
- `ExtractionConfig.methods` defaults to `['mean_diff', 'probe']` (no `or` fallbacks)

**In-memory activation→vector flow:**
- `extract_activations_for_trait` returns activations dict (no .pt roundtrip)
- `extract_vectors_for_trait` accepts optional `activations` param (falls back to disk for --only-stage 4)
- `--save-activations` to persist .pt files

**Other cleanup:**
- rm_sycophancy → experiments/rm_syco/ + R2 push
- Deleted: utils/capture.py (dead), other/tv/data/loras.yaml (duplicate), extraction/__init__.py, steering/__init__.py
- Promote script enforces main = exact mirror of .publicinclude

### Dedup + Consolidation Pass
- Deleted `other/` entirely (tv, sae, server, mcp, analysis). server → `utils/server/`, sae → `analysis/sae/`
- Deleted: `utils/profiling.py` (merged into vram.py), `utils/modal_backend.py`, `utils/update_viz_repo.sh`, `utils/railway_pull_r2.sh`
- Merged: `steered_generation.py` → `generation.py`, `fingerprints.py` → `projections.py`
- New: `utils/positions.py` (multiturn-native position DSL), `utils/batch_forward.py` (OOM/TP/calibration helpers)
- Fixed: `core/generation.py` upward import → `get_layer_path_prefix` now lives in core/
- Consolidated: `batched_steering_generate` (prompts= param), `load_activations` (split= param), `save_*_responses` (`_write_responses` helper), `dump_compact` delegates to `dumps_compact`
- Wired: `clear_oom_traceback` into 3 OOM sites, `tp_agree_batch_size` into 3 sites, `tp_agree_count` into 2 sites in extract_vectors
- Fixed 3 runtime bugs: steering eval signatures, compare_variants NameError, list_layers arg order
- Fixed 4 dev/ broken imports, deleted 6 stale dev/ files

---

## Known Issues

- `valid` key in steering_results.py never written to disk
- `activation_norms` has 3 different schemas across files
- Response JSON written in 2 places with schema drift
- Stream-through mode skips massive_dim_data (requires full activations)
- iCloud Desktop sync creates " 2" files constantly

---

## TODO

- **Coefficient search consolidation** — delete `adaptive_search_layer` + `evaluate_single_config`, use `batched_adaptive_search` with `max_batch_layers=1` for sequential. Extract shared state helpers (`_update_coefficients`, `_flush_best_responses`, `_print_summary`). ~250-280 lines saved.
- **`vet(target=)` consolidation** — merge `vet_scenarios` + `vet_responses` into one function
- **`resolve_use_chat_template` helper** — 6 identical inline copies in steering/inference. Move to `utils/paths.py`
- **`build_response_records` helper** — 10 identical inline copies of `[{"prompt": q, "response": r, ...}]`
- **`summarize_judge_scores` helper** — 6 of 12 sites share the dict+score_stats+None-default pattern. Note: other sites use `0` default or skip score_stats — don't force-unify those.
- **Wire `calibrate_batch_size`** from batch_forward.py into extract_vectors.py (replace 15-line inline probe)
- **Refactor `get_best_vector` / `get_top_N_vectors`** — share a `_scored_candidates` helper. Important function, deserves focused attention.
- **Default to probe only** (not mean_diff+probe)
- **README rewrite** — story-driven walkthrough using `hyperparams` throughout
- **Top-activating spans tool** — for a single trait, find top-activating clauses (comma/period/semicolon/newline separated) or n-length sequences across inference responses
- **Optimize vetting responses** — batching, caching, or speed improvements
- **Data storage for users** — non-R2 options for public users
- Per-trait layer config
- Visualization audit
- Upload custom LoRAs to HuggingFace

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