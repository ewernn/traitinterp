# Typing Audit: Untyped Dict Contracts

Identified by 5 subagents scanning the full codebase. These are inline dicts with consistent key sets passed between functions/files. A key rename in any producer silently breaks consumers that use `.get(key, 0)`.

## Priority 1: Cross-file contracts (silent failure risk)

### `select_vector()` return dict → `VectorResult` dataclass
- **Keys:** layer, method, position, component, source, score, coefficient, direction, naturalness (optional)
- **Producers:** `vector_selection.py:_select_vectors` (mutates candidate dicts in-place, adding score/direction/source/coefficient)
- **Consumers:** 11 files across analysis/, visualization/, utils/
- **Risk:** All access by string key. Renaming `source` → `origin` silently breaks print statements.
- **Fix:** `VectorResult` dataclass in `core/types.py`. `select_vector() -> VectorResult`, `select_vectors() -> List[VectorResult]`.

### `summarize_judge_scores()` return → `JudgeResult` dataclass
- **Keys:** trait_mean, coherence_mean, n, trait_std, success_rate, min_score, max_score
- **Producers:** `metrics.py:summarize_judge_scores`, `coefficient_search.py:248` (inline mirror)
- **Consumers:** 8+ sites via `.get('trait_mean', 0)` — silent zero on rename
- **Risk:** HIGHEST. A rename silently makes vector selection treat all vectors as equally worthless (delta=0).
- **Fix:** `JudgeResult` dataclass. All consumers use attribute access.

### `load_results()` return → `SteeringResults` dataclass
- **Keys:** trait, direction, steering_model, steering_experiment, vector_source, eval, prompts_file, prompts_hash, baseline, runs
- **Consumers:** vector_selection.py, steering_eval.py, serve.py, dev/steering/
- **Risk:** `baseline` accessed via `.get()` with None guard everywhere

### `get_model_variant()` return → `ModelVariant` namedtuple
- **Keys:** name, model, lora
- **Consumers:** 30+ files
- **Risk:** Low (uses `[]` access which errors loudly), but easy win

## Priority 2: Internal contracts

### `discover_vectors()` candidate dict → part of `VectorResult` pipeline
- Starts as {layer, method, position, component, path}, mutated to 9 keys
- Fix: `VectorCandidate` or fold into `VectorResult`

### `discover_steering_entries()` return → `SteeringEntry` dataclass
- Keys: trait, model_variant, position, prompt_set, full_path

### Coefficient search state dicts
- Single-trait: 8 keys. Multi-trait: 14 keys (superset).
- `_process_config_result` takes multi-trait-only state but typed as `Dict`

### `trait_configs` elements
- 10 keys, constructed in steering_eval.py, consumed in coefficient_search.py

### `score_stats()` return
- Splat-merged via `**` — key change = silent bug in consumer

## Priority 3: I/O schemas

### `proj_data` in process_activations.py
- Constructed identically in 2 places (duplication risk)

### `ResponseRecord` — defined but never used
- Write paths construct raw dicts matching ResponseRecord
- Read paths use `json.load()` + `.get()` instead of `ResponseRecord.from_dict()`

### Vetting summary dict
- `.get('positive_passed', 0)` everywhere — silent zero on rename
- Also has a bug: `pos_total` uses `positive_passed` twice (copy-paste)

## Existing types that are well-used
- `VectorSpec` — properly used for vector identity
- `ProjectionConfig` — properly used for ensemble configs
- `ModelConfig` — properly used for model registry
- `ResponseRecord` — defined but bypassed on both read and write
