# Codebase Refactor Notepad

Dev-only tracking. Not promoted to main.

---

## TODO

### Model loading convergence
Pipeline scripts use `LocalBackend.from_experiment()`. These still call `load_model_with_lora()` directly:
- `inference/generate_responses.py`
- `analysis/vectors/logit_lens.py`
- `analysis/benchmark/benchmark_evaluate.py`
- `utils/steering_eval.py` (fallback path)

### Manual path building cleanup (audit 2026-04-05)

28 manual path construction sites across 13 public files that should use `get_path()` with `config/paths.yaml`.

**Bug:**
- `analysis/sae/encode_sae_features.py:215,242` — missing `model_variant` param → literal `{model_variant}` in path

**Core utilities (5 files, 12 sites):**
- `utils/capture_activations.py:113` — manual `raw/residual` → use `inference.raw_residual`
- `utils/project_activations.py:172,238,320,508,541` — manual inference subdirs, receives `inference_dir` param
- `utils/extract_vectors.py:85,110` + `utils/preextraction_vetting.py:197` — manual `vetting/` → use `extraction.vetting`
- `analysis/massive_activations.py:44,356,708` — hardcoded dataset path + manual `massive_activations/`

**Analysis scripts (4 files, 11 sites):**
- `analysis/model_diff/layer_sensitivity.py:228-230` — manual `inference/{variant}/raw/...`
- `analysis/model_diff/per_token_diff.py:162-164` — manual `inference/{variant}/projections/...`
- `analysis/data_checker.py:291,324,403,450` — manual subdirs with YAML equivalents
- `analysis/vectors/logit_lens.py:152`, `analysis/trait_correlation.py:184` — could use `analysis.category`

**Cosmetic:** 2 files use `paths.get(...)` instead of `get_path(...)` import style

### Public release
- Extract + steer 9 starter traits on a model
- Run pre-release audit (trufflehog, vulture, pip-audit)
- Promote to main
- HuggingFace Hub for experiment data

### Features / future
- Ensemble projections for inference — weighted combination across layers (steering ensembles exist in `utils/ensembles.py`)
- Trait dataset format redesign (trait.json + scenarios.json) — defer to pip package
- pip package API design (`import traitinterp`)

---

## Architecture Decisions (settled)

- **1 recipe file per pipeline dir** — thin controllers delegating to utils/
- **core/** = pure primitives (types, hooks, math, methods). No upward deps. torch-free types.
- **utils/** = library code. Pipeline helpers live here.
- **Typed returns** — public functions return dataclasses, not dicts.
- **Recursive discovery** — no hardcoded directory depth. Walk until leaf marker found.
- **flat trait categories** — datasets/traits/{category}/{trait}/ (no base/instruct nesting)
- **emotion_set is canonical** — wins 9/12 head-to-heads at 2-4x lower coefficients
- **Don't rename trait categories** — paths embedded in experiment dirs + JSON metadata
- **Base model traits on dev until paper** — natural elicitation is novel contribution
- **ruff for linting** — replaces flake8/black/bandit
- **Modal stays in dev/** — bypasses backend abstraction, not integrated into main pipelines
- **Normalized projections default** — divide by mean activation norm at layer. Raw stored, normalized at read/display time.
- **Massive dims separate from projection** — standalone calibration tool, not tangled into projection path
