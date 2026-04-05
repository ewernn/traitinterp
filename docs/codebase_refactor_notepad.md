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
