# Codebase Refactor Notepad

Dev-only tracking. Not promoted to main.

---

## Known Issues

- `valid` key in steering_results.py never written to disk

---

## TODO

### Vector loading convergence (lower priority)
4 implementations serve different use cases:
- `load_trait_vectors` — stream-through GPU batched (stacked tensors + hook_index)
- `process_prompt_set` — post-hoc CPU, supports multi-vector + method auto-detect
- `chat_inference._load_trait_vectors` — simplest, one vector per trait
- `steering_eval.load_vectors` — minimal, computes base_coef
Selection logic genuinely differs. Could extract shared `_load_single_vector()` helper but not high priority.

### Model loading convergence
`ServerBackend`, `tokenize_with_prefill()`, `load_model_or_client()` deleted. Remaining files call `load_model_with_lora()` directly. Should funnel through `LocalBackend.from_experiment()` which already exists.

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
- **flat trait categories** — datasets/traits/{category}/{trait}/ (no base/instruct nesting)
- **emotion_set is canonical** — wins 9/12 head-to-heads at 2-4x lower coefficients
- **Don't rename trait categories** — paths embedded in experiment dirs + JSON metadata
- **Base model traits on dev until paper** — natural elicitation is novel contribution
- **ruff for linting** — replaces flake8/black/bandit
- **Modal stays in dev/** — bypasses backend abstraction, not integrated into main pipelines
- **Normalized projections default** — divide by mean activation norm at layer. Raw stored, normalized at read/display time. Cosine available as alternative.
- **Massive dims separate from projection** — standalone calibration tool, not tangled into projection path
