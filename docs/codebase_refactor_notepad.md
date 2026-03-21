# Codebase Refactor Notepad

## Status

Major refactor complete. Typed dataclasses, dead code removed, datasets restructured, recursive discovery. Preparing for public release + pypi package.

**Next priority:** Type remaining output formats, clean serve.py dead code, update docs, promote to main.

---

## Completed Refactoring

### Summary
~110 → ~47 → 3 recipe files. 32 dead functions deleted (-556 lines). 58 unused imports removed. 4 runtime bugs fixed. 4 typed dataclasses added. Datasets restructured (base/instruct/starter_traits/archive).

### Recent (this session)
- **Typed dataclasses** in `core/types.py`: VectorResult, JudgeResult (to_dict/from_dict), ModelVariant (NamedTuple), SteeringEntry. 45+ consumer files updated to attribute access.
- **types.py is torch-free** — importable without GPU deps for pypi
- **Recursive discovery** — discover_traits/discover_extracted_traits/discover_steering_entries support arbitrary directory depth, skip archive/
- **Dead code sweep** — 32 functions from core/math.py (6), utils/paths.py (8), utils/projections.py (12), extract_vectors.py (run_logit_lens_for_trait), activation_scale, dead config fields
- **58 unused imports** auto-fixed by ruff
- **4 runtime bugs** — dist/tqdm undefined in generation.py, build_response_records/resolve_use_chat_template scoping in steering_eval.py
- **Trait consolidation** — 8 subagents compared 26 duplicates. Absorbed best versions into emotion_set. 19→4 trait categories.
- **Dataset restructure** — base/ (emotion_set 174, alignment 10, tonal 7), instruct/ (7), starter_traits/ (9), archive/
- **Starter traits** — 9 instruct-based (sycophancy, evil, hallucination, refusal, assistant, formal, golden_gate_bridge, concealment, optimism)
- **Renames** — evil_v3→evil, hallucination_v2→hallucination
- **Hashing** — input_hashes at 4 save points, prompts_hash in steering header, hash-based staleness
- **pyproject.toml** — pip install -e ., ruff config, pytest config
- **Repo renamed** to traitinterp on GitHub

### Historical
- Thin recipe pattern, config dataclasses, in-memory activation→vector, coefficient search consolidation, position DSL, batch_forward helpers, 12+ shared helpers extracted. See git history.

---

## Known Issues

- 43 experiment scripts have broken `get_best_vector` imports (experiments/ only, not pipeline)
- serve.py dead startup cost (cache_integrity_data), 3 dead endpoints, dead CSS
- Trait paths: datasets 3-level (base/emotion_set/X), experiments 2-level (emotion_set/X)
- `valid` key in steering_results.py never written to disk
- `activation_norms` has 3 different schemas across files
- Projection JSON written in 2 places independently (schema drift risk)

---

## TODO

### Typing (9 remaining output formats — see docs/typing_audit.md)
1. **ResponseRecord** — dataclass exists but bypassed by 4 write sites. Wire it. Highest impact.
2. **Projection JSON** — 2 independent write sites, 6 consumers. Shared builder needed.
3. **Steering Response Records** — build_response_records already a factory, easy conversion
4. **Activation Metadata** — 18 keys, 5 consumers via load_activation_metadata
5. **Vetting Scores** — 3 consumers, nested structure
6. **SteeringResults (load_results return)** — 10 consumers, partially typed
7-9. Massive activations, extraction metadata, extraction records (lower priority)

### serve.py cleanup
- Delete cache_integrity_data() startup (serves dead data-explorer view)
- Delete 3 dead endpoints (/api/schema, inference prompts, model-variants)
- Dead CSS classes (~95 lines)
- layer-deep-dive.js — loaded but never routed (decide: finish or archive)

### Backend
- Only generation stages can be server-routed. Hook-dependent stages (extraction 3+4, steering, inference projection) require local model — ServerBackend.forward_with_capture is NotImplementedError
- Modal scripts in dev/ bypass backend abstraction entirely (run full script on Modal)
- `--backend` flag silently ignored in extraction + steering (only inference reads it)

### Documentation
- Update docs/main.md directory tree for new structure
- Update docs/workflows.md for new trait paths
- Review readme.md (user hasn't looked at it yet)

### Public release
- Extract + steer 9 starter traits on a model
- Run pre-release audit (trufflehog, vulture, pip-audit)
- Promote to main
- HuggingFace Hub for experiment data

### Features / future
- Trait dataset format redesign (trait.json + scenarios.json) — defer to pip package
- pip package API design (`import traitinterp`)
- Best-layer agreement analysis (probe accuracy vs steering delta)
- Top-activating spans tool
- Paper experiment (LIARS' BENCH deception decomposition — separate chat)

---

## Architecture Decisions (settled)

- **1 recipe file per pipeline dir** — thin controllers delegating to utils/
- **core/** = pure primitives (types, hooks, math, methods). No upward deps. torch-free types.
- **utils/** = library code. Pipeline helpers live here.
- **Typed returns** — public functions return dataclasses, not dicts. VectorResult, JudgeResult, etc.
- **Recursive discovery** — no hardcoded directory depth. Walk until leaf marker found.
- **base/instruct split** — datasets/traits/base/ vs datasets/traits/instruct/ tells users how to extract
- **emotion_set is canonical** — wins 9/12 head-to-heads at 2-4x lower coefficients
- **Don't rename trait categories** — paths embedded in experiment dirs + JSON metadata
- **Base model traits on dev until paper** — natural elicitation is novel contribution
- **ruff for linting** — replaces flake8/black/bandit
- **Modal stays in dev/** — bypasses backend abstraction, not integrated into main pipelines

### Code Design Principles (for future agents)

- **Zero duplicate code.** If the same block appears twice, extract a helper.
- **Typed returns over dicts.** Public functions return dataclasses. `.get('key', 0)` is a silent bug waiting to happen.
- **Flags over function proliferation.** One function with a parameter > two 80%-identical functions.
- **Pipeline recipes stay thin.** 200-360 line table-of-contents. Drill into utils/ for how.
- **Spawn subagents liberally.** Critics before implementing, investigators for exploration.
- **content_hash for provenance.** Store hashes at save-time, check at load-time.
