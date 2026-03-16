# Codebase Refactor Notepad

## Plan

Refactoring trait-interp for public release. Key changes:

1. **R2 redesign** — shared config, tier-based excludes, `--only`, `--include-loras/archive/trajectories`
2. **Git** — add `experiments/` to `.gitignore`, R2 owns everything
3. **Experiment structure** — standardize: pipeline dirs at root, sub-experiments as subdirs with `{name}_notepad.md` + scripts + results
4. **LoRAs** — upload custom to HF, `config/loras.yaml` as registry
5. **Docs cleanup** — consolidate, link from `docs/main.md`

---

## 05:30 UTC — Design decisions finalized

After deep investigation of local (11GB) and R2 (82GB, ~1M objects) experiment data:

**R2 tiers:**
- Default: sync everything except loras, archive, trajectories, and regenerable data
- `--include-loras`: finetune/, turner_loras/, sriram_loras/, lora/
- `--include-archive`: experiments/archive/
- `--include-trajectories`: *_trajectories.pt, em_probe/**/data*.pt
- Always exclude: activations/, inference/*/raw/, optimizer.pt, scheduler.pt, __pycache__, .DS_Store

**R2 scripts:**
- Shared `utils/r2_config.sh` sourced by push and pull
- `--fast-list` always on (saves ~1000 API calls per sync)
- `--only experiment1,experiment2` for scoping (full names only)
- Remove .py/.md/config.json excludes (R2 owns all experiment files now)

**Experiment schema:**
- Standard pipeline dirs: extraction/, inference/, steering/, model_diff/
- Sub-experiments: `{name}/` with `{name}_notepad.md`, scripts, results/
- Experiment-specific scripts in sub-experiment dirs, not experiment root
- paths.yaml stays one file, add convention comments for optional dirs

**Notepad convention:**
- Filename: `{name}_notepad.md`
- Entries: `## HH:MM UTC — Title` (timestamps in UTC)
- Optional plan section at top

## 06:45 UTC — R2 scripts built and tested

- Created `utils/r2_config.sh` (shared excludes, arg parsing, path resolution)
- Refactored `utils/r2_push.sh` and `utils/r2_pull.sh` to source shared config
- `--fast-list` always on, `--only`, `--include-loras/archive/trajectories` all working
- Fixed rclone exclude pattern issue: need both `finetune/**` and `**/finetune/**` for `--only` scoping
- Dry-run verified: 0 false transfers, 815 LoRA files correctly gated behind `--include-loras`
- Updated `.gitignore`: replaced 20+ experiment patterns with single `experiments/`
- R2 push running in background to back up .py/.md/config.json files before git rm

## 07:00 UTC — Docs and config updates

- Created `docs/experiment_structure.md` — full experiment schema reference
- Moved `other/tv/data/loras.yaml` → `config/loras.yaml`
- Updated `docs/main.md` — linked experiment_structure.md and loras.yaml
- Updated `config/paths.yaml` — added convention comments for sub-experiments

## 07:15 UTC — Public release strategy

**Approach: git filter-repo**
1. Back up current repo as `ewernn/trait-interp-backup-2026-03-14` (private, keeps all 700+ commits on calendar)
2. Run `git filter-repo --path-glob '*.md' --invert-paths --path overnight_log.txt --invert-paths --path phase2_normalization.log --invert-paths`
3. This removes ALL .md files + logs from all 700 commits (history is clean)
4. Copy back personal .md files from backup, add to `.gitignore` so they exist locally but aren't tracked
5. Restructure `docs/` — flatten public docs into `docs/`, gitignore `docs/other/`
6. One commit: "Add documentation" — first time .md files appear in clean history
7. Make repo public

**Open question (resolved):** Claude Code `@` autocomplete DOES find gitignored files. Tested with experiments/ files.

## Before public release — git filter-repo

Create backup repo first: `gh repo create ewernn/trait-interp-backup-2026-03-14 --private` and push current state.

Then run filter-repo to remove from ALL history:
```bash
git filter-repo \
  --path-glob '*.md' --invert-paths \
  --path overnight_log.txt --invert-paths \
  --path phase2_normalization.log --invert-paths \
  --path experiments/ --invert-paths \
  --path docs/other/ --invert-paths \
  --path docs/lit_surveys/ --invert-paths \
  --path docs/full_papers/ --invert-paths \
  --path docs/plans/ --invert-paths \
  --path notepad_mar3.md --invert-paths \
  --path notepad_tonal_persona.md --invert-paths \
  --path temp/ --invert-paths \
  --path diffing-toolkit/ --invert-paths \
  --path .env --invert-paths
```

After filter-repo:
1. Copy back personal files from backup (gitignored, local-only)
2. Re-add public docs in a clean commit
3. Force push
4. Make repo public

Note: `git rm --cached` (done now) only untracks from current state — files remain in prior commits. filter-repo is what actually purges history.

## 08:00 UTC — Repo structure and branching strategy

**Three-part split:**

1. **`main` branch** — curated public storefront. Lean, well-documented framework code only. What people see when they visit the repo.
2. **`dev` branch** — full working state. All experimental scripts, analysis tools, in-progress work. Public (anyone can browse branches) but not the default view. This is where daily development happens.
3. **`../trait-interp-personal/`** — private GitHub repo. Career strategy, mentor notes, meeting notes, research plans, personal notepads. Anything you wouldn't want visible on a public repo at all.

**What goes where:**
- `main`: core/, extraction/, inference/, utils/, visualization/, config/, datasets/, curated analysis/ scripts, clean docs/
- `dev`: everything on main + extra analysis scripts, experimental code, work-in-progress, all the stuff that's legitimate code but would overwhelm someone browsing the repo
- `trait-interp-personal/`: docs/other/strategic*.md, docs/other/neel_*.md, docs/other/jan4_eric_strat.md, docs/other/david_bortz.md, docs/other/fellowship_applications.md, root notepads, etc.

**Workflow:** Work on `dev`. Cherry-pick or merge polished scripts to `main` when ready. Personal docs live in separate repo entirely.

**R2 backup completed:** 848 files (.py, .md, config.json) pushed to R2. experiments/ untracked from git via `git rm --cached`. git status now 0.035s (was 2.87s).

## 09:00 UTC — Git history cleaned, backups complete

- Backed up full repo to `ewernn/trait-interp-backup` (private, both branches)
- Backed up R2 experiments to `r2:trait-interp-bucket/backups/2026-03-14/experiments/` (82.3GB, 1.05M objects — complete)
- Ran `git filter-repo` — removed all .md, experiments/, logs, .env, temp/, diffing-toolkit/ from history
- 654 → 472 commits remain (empty commits pruned)
- Restored 50 public docs from temp save, committed on dev
- Force pushed main + dev + prod with clean history
- Deleted 6 stale `claude/*` branches
- Created `ewernn/trait-interp-personal` (private) — personal docs moved there
- Deleted docs/other/, docs/lit_surveys/, docs/full_papers/, docs/plans/, root notepads/logs from dev
- Created `.publicinclude` + `utils/promote_to_main.sh` for dev→main workflow

**Current state:** On dev branch. main has clean history but no .md docs yet. dev has docs + all code.

## Directories to clean up next

| Directory | .py | .md | .json | .yaml | .txt | .js | .sh | other | total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| (root) | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 4 | 7 | Procfile, railway.toml, .railwayignore, requirements.txt |
| analysis/ | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | |
| analysis/benchmark | 1 | | | | | | | | 1 | |
| analysis/model_diff | 15 | | | | | | | | 15 | |
| analysis/steering | 20 | | | | 6 | | | | 26 | 6 .txt are personal audit files |
| analysis/trajectories | 0 | | 1 | | | | | 10 | 11 | 10 "other" — likely notebooks |
| analysis/vectors | 6 | | | | | | | | 6 | |
| config/ | | | | 3 | | | | | 3 | paths.yaml, loras.yaml, trait_lenses.yaml(?) |
| config/models/ | | | | 18 | | | | | 18 | |
| core/ | 10 | | | | | | | | 10 | |
| core/_tests/ | 6 | | | | | | | | 6 | |
| datasets/inference/ | | | ~40 | | | | | | ~40 | prompt sets |
| datasets/traits/ | | | ~300 | | ~900 | | | | ~1200 | 230+ traits across categories |
| docs/ | | 31 | | | | | | | 31 | |
| docs/viz_findings/ | | 16 | | 1 | | | | | 17 | dev/prod only |
| extraction/ | 13 | | | | | | | | 13 | |
| inference/ | 6 | | | | | | | | 6 | |
| other/analysis/rm_syco | 8 | | 25 | | | | | 96 | 129 | 96 .png plots |
| other/lora/scripts | 2 | | | | | | | | 2 | |
| other/mcp | 1 | | | | 1 | | | | 2 | |
| other/railway | | | | | 1 | | | | 1 | |
| other/sae | 4 | | | | | | | | 4 | |
| other/scripts | 2 | | | | | | | | 2 | |
| other/server | 4 | | | | | | | | 4 | |
| other/tv/ | ~12 | 1 | ~13 | 10 | 1 | | | ~400 | ~437 | 349 .pt vectors, paper figures |
| scripts/ | 7 | | | | | | 5 | | 12 | infra scripts |
| utils/ | 21 | | | | | | 7 | | 28 | |
| visualization/ | 3 | | | | | 29 | | 4 | 39 | |

## User intent (verbatim quotes)

> "I want to really think deeply about small design choices and have you think about 5+ choices for some things and then pick one or give me the two top ones"

> "I want people who clone my repo to not be overwhelmed with a shit ton of files, even tho a file like that is legit. i want to be intentful with everything on the main public branch"

> "Literally i'm gonna refactor my whole codebase"

> "I don't want to include my research in it yet. That's my own research that I might build off of. I will have it on my personal trait-interp.com site for others to read"

> "I also will want to make an archive/ dir within it FYI"

> "I need to be able to do subexperiments within an experiment, and that's when each subexperiment gets a notepad.md, like fingerprint_qwen2.5_14B_notepad.md, where it timestamps entries as it works, with an optional initial plan at the top. that's kinda the workflow I settled on."

> "stuff like score_naturalness.py and other stuff I'm experimenting with i suppose could go in my personal branch?" → resolved: dev branch for in-progress code, main for curated public code, trait-interp-personal for truly private docs

> "I don't just want my own branch. [for public release]" → resolved: single repo with filter-repo'd history, dev/main branch split, .publicinclude whitelist

> "i feel like i wanna get more fine-grained control over r2 push and r2 pull, and have them share common excludes, so i could download lightweight versions of experiments without the big loras, or just a single experiment"

> "I also want people to be able to fork this repo and add their own methods on which is why i want some generality and clarity and separation of concerns and clean concise readable main files"

> "I want to be able to understand the codebase by just the filenames idk if that's too much to ask"

> "I wanted a local/server/modal options as flags or something, like idt we should need 2 extra files in inference for modal"

> "we might be able to cut like two thirds of files or merge functionality"

> "I want the main functionality files to have a bunch of helpers in other files, so the main files are clean and concise!!"

> "idk why we have two separate dirs that tbh idk the meaning of 'core' vs 'utilities'"

> "A lot of this shit can probably be merged. like, do we really need massive_activations.py and massive_activations_per_layer.py separate? and some of the naming sucks. idk what any of analysis/steering/ files mean from their titles"

> "activation capture works if you add hooks to modal so I don't agree that unified backend flag only works for generation and steering, especially because both generation and steering also hook the internals anyways"

> "I'm gonna want to improve the validation after each stage so we can safely automate the entirety from definition to steering vector"

> "we could also default to held out eval and steering optional, to make default cleaner cheaper faster. but tell people to preferably use steering"

> "I wanted to potentially support using local LLM as the judge instead of gpt-4.1-mini if someone wanted to just use local GPU for everything"

> "Capture and project in the same forward pass has been slow because CPU bound, but maybe I wasn't vectorizing or something"

> "we can name things like steering/steering_evaluate.py. and i don't know why we need data.py, if it could be like a util. and preextraction_vetting.py"

> "should we default to serverbackend, or potentially might not free GPU memory or something" → resolved: auto default (try server, fall back to local). Orchestrator shares backend object across stages (pseudo-server). Server is for cross-pipeline sharing.

> "I don't want a bunch of stuff in the root. it's not necessary functionality." → server stays in other/server/, no new root dirs beyond what's needed

## 12:00 UTC — Deep investigation complete (all directories mapped)

Spawned 20+ subagents across 2 sessions. Every directory fully audited.

## 14:00 UTC — Waves 1-4 executed

### Wave 1: Deleted ~45 dead files ✅
scripts/ (7), analysis/steering/ .txt + py (8), analysis/model_diff/ (6), analysis/ root (2), analysis/trajectories/ (10), extraction/instruction_based_extraction.py, other/lora/, other/railway/, other/analysis/inference/, Procfile, railway.toml, .railwayignore

### Wave 2: Fixed bugs ✅
- Removed GPUMonitor duplicate from utils/generation.py (broken time import)
- Removed dead imports from inference/generate_responses.py and capture_activations.py
- Removed dead --no-server flag from capture_activations.py
- Removed silently-ignored batch_size param from inference/generate_responses.py
- Fixed vet_scenarios parameter shadowing in run_extraction_pipeline.py (renamed to run_scenario_vetting)
- Fixed extraction/__init__.py (3 stages → 7)
- Removed dead SCORING_MODEL constant from utils/judge.py
- Removed redundant import json as _json from run_extraction_pipeline.py

### Wave 3: Structural moves ✅
**3a. core/ → utils/ (4 files):** backends.py, steering.py, logit_lens.py, profiling.py moved to utils/. 16 files updated. core/ now only has: types.py, hooks.py, math.py, methods.py, generation.py, __init__.py. Tests: 106 passed.

**3b. steering/ promoted to root:** analysis/steering/ → steering/. Files renamed: evaluate→steering_evaluate, results→steering_results, coef_search→coefficient_search. 8 dev-only files → dev/steering/. 6 external importers + 13 internal + 11 doc files updated. Zero stale references.

**3c. scripts/ dissolved:** vast_*.sh → utils/, convert_rollout.py + convert_audit_rollout.py + align_sentence_boundaries.py → inference/. importlib path updated. scripts/ directory removed.

**3d. dev/ directory created:** Holding pen for files to integrate or delete later. Contains: 8 analysis files, 2 extraction modal files, 3 inference files (modal + extract_viz), 1 other file, 9 steering files. Total: 23 files in dev/.

**Other moves:** onboard_model.py → utils/, convert_lora_bf16.py → dev/other/, other/scripts/ removed.

### Wave 4: Merges + renames ✅
- run_pipeline.py → run_extraction_pipeline.py (2 importers + 10 docs updated)
- capture_raw_activations.py → capture_activations.py (1 importer + 20 doc/string refs)
- project_raw_activations_onto_traits.py → project_activations_onto_traits.py (0 importers + 15 doc refs)
- vet_scenarios.py + vet_responses.py → preextraction_vetting.py (shared _build_vetting_output helper, fixed fragile index lookup)
- massive_activations.py absorbed massive_activations_per_layer.py (--per-layer flag)
- steering_data.py → merged into utils/traits.py (SteeringData + load_steering_data + 2 helper functions)
- weighted_multi_layer_evaluate.py → dev/steering/
- steering_report.py, read_steering_responses.py, logit_difference.py → dev/steering/
- benchmark/evaluate.py → benchmark/benchmark_evaluate.py

### Remaining bugs (not yet fixed)
- `valid` key in steering_results.py never written to disk → is_better_result always treats loaded as invalid
- `activation_norms` has 3 different schemas across files
- Response JSON written in 2 places with schema drift (generate_responses.py and project_activations_onto_traits.py)

### Architecture decisions (settled)

**core/ → 4 true primitives:** types.py, hooks.py, math.py, methods.py. Move backends.py, steering.py, logit_lens.py, profiling.py → utils/

**Backend unification:** --backend auto|local|server|modal flag. `auto` = try server if running, else local. Pipeline orchestrator shares backend object across stages (no reload between stages). Server is for cross-pipeline model sharing. User nudged via stdout: "Tip: run server to keep model loaded between commands."
- `utils/backends.py` — LocalBackend + ServerBackend + get_backend() factory
- `utils/modal_backend.py` — ModalBackend (separate file, modal-specific concerns: volumes, GPU provisioning, @modal.function decorators)
- Modal GPU options listed in modal_backend.py (T4/A10G/A100-40/A100-80/H100)

**Steering → root-level dir:** Not "analysis" — it's pipeline stage 3. Promote from analysis/steering/ to steering/.
- Called by run_extraction_pipeline.py as OPTIONAL final stage (--steering flag)
- Default pipeline ends at held-out eval (fast, free, no API calls)
- Held-out eval gate suggests steering: "For causal validation, re-run with --steering"
- Also independently runnable: `python steering/steering_evaluate.py`

**data.py → merge into utils/traits.py:** Same concern (loading from datasets/traits/).

**scripts/ dissolved:** Keepers moved to utils/ (infra scripts) or inference/ (rollout converters).

**Inference pipeline redesign:** Stream-through default (capture + project in same pass, no .pt files). --save-activations flag for raw data. --from-activations for re-projection. ProjectionHook on GPU instead of CaptureHook + CPU projection.

**Projection performance:** Current bottleneck is IO + per-vector GPU-CPU sync, not math. projection() is already vectorized (one torch.matmul). Fix: stack all vectors per layer into [n_vectors, hidden_dim], one batched matmul output @ vectors.T, one .cpu() call.

**Projection file structure:** Keep current (one JSON per prompt×trait). Dashboard fetches one (trait, prompt) at a time — ideal for current format. Analysis scripts glob all prompts for one trait — works fine at typical scales. Adding a new trait = write N new files without touching existing. R2 sync manageable with --turbo.

**LLM judge:** gpt-4.1-mini hardcoded in 2 places. Swappable via base_url param to AsyncOpenAI for vllm/llama.cpp. Need systematic quality validation experiment for local models vs gpt-4.1-mini. Core mechanism: logprob-weighted averaging over top-20 tokens. Requires model with top_logprobs support (vllm yes, llama.cpp yes, Ollama unreliable).

**Schemas:** 9 file formats, zero validation currently. Formalize as dataclasses in core/types.py. Add timestamps + params to steering results.jsonl.

**Naming:** Prefer longer descriptive names per CLAUDE.md. Prefix with module name when ambiguous in cmd+P (steering_evaluate.py, not evaluate.py). No underscore prefix on helpers. Examples: run_extraction_pipeline.py, preextraction_vetting.py, coefficient_search.py, steering_results.py.

**No src/ dir:** Script-based tool, not a pip library. Would break all invocations.

**Ensembles + components:** Keep support for ensembles, attn_contribution, mlp_contribution as pipeline options.

**Cloud sync:** Eventually add easier alternatives to R2 (R2 is slightly technical for some users).

### What visualization needs (load-bearing scripts)

- extraction_evaluation.py → Extraction view
- steering results.jsonl (via steering_results.py) → Steering view
- response + projection JSONs → Trait Dynamics view
- massive_activations.py → Model Analysis
- compare_variants.py → Model Analysis
- layer_sensitivity.py → Trait Dynamics Layers toggle (optional)
- trait_correlation.py → Correlation view

### Current codebase state (post-Wave 4)

```
core/           6 files  (types, hooks, math, methods, generation, __init__)
extraction/     9 files  (run_extraction_pipeline, generate_responses, extract_activations,
                          extract_vectors, preextraction_vetting, run_logit_lens,
                          test_scenarios, validate_trait, __init__)
inference/      6 files  (generate_responses, capture_activations, project_activations_onto_traits,
                          convert_rollout, convert_audit_rollout, align_sentence_boundaries)
steering/       6 files  (steering_evaluate, steering_results, coefficient_search,
                          multi_layer_evaluate, weight_sources, __init__)
analysis/      12 files  (data_checker, massive_activations, trait_correlation,
                          benchmark/benchmark_evaluate,
                          model_diff/{__init__, compare_variants, layer_sensitivity,
                                     per_token_diff, top_activating_spans},
                          vectors/{extraction_evaluation, logit_lens, cka_method_agreement,
                                  component_residual_alignment, cross_layer_similarity,
                                  trait_vector_similarity})
utils/        ~28 files  (backends, steering, logit_lens, profiling, model, generation,
                          paths, vectors, vram, moe, judge, projections, activations,
                          capture, traits, fingerprints, ensembles, annotations,
                          model_registry, layers, json, distributed, metrics,
                          onboard_model, + shell scripts)
dev/           23 files  (holding pen — steering CLI tools, modal files, extract_viz,
                          analysis dev-only scripts)
other/          server/, tv/, sae/, mcp/, analysis/rm_sycophancy/
```

Total pipeline Python files: ~47 (excluding dev/, utils shell scripts, other/).

### Remaining waves

**Wave 5: Backend unification**
- `get_backend()` already exists in utils/backends.py but NO pipeline script calls it
- All scripts have ad-hoc model loading — replace with get_backend(backend=args.backend)
- Add --backend auto|local|server|modal flag to each pipeline script
- Create utils/modal_backend.py with ModalBackend + GPU options
- ServerBackend CAN do capture (server has /capture endpoint) but currently NotImplementedError
- batched_steering_generate() is fundamentally local-only (PyTorch hooks)
- Add stdout tip in auto mode: "Tip: run server to keep model loaded between commands"

**Wave 6: Pipeline architecture**
- Create inference/run_inference_pipeline.py orchestrator (mirrors extraction pattern)
- Add --steering flag to run_extraction_pipeline.py (optional final stage)
- Stream-through mode: capture + project in same pass, --save-activations to also write .pt
- project_activations_onto_traits.py has NO importable function — needs refactoring
- Batched projection: group vectors by layer, one matmul per unique layer
- Extract run_steering_for_trait() wrapper from steering_evaluate.py

**Wave 7: Schemas + metadata**
- Add ResponseRecord dataclass to core/types.py (response JSON written in 2 places with drift)
- Add timestamp + judge_model + n_questions to steering results.jsonl header
- Standardize config/models/*.yaml (2 stubs need filling, proposed schema ready)

**Wave 8: Docs + .publicinclude**
- .publicinclude needs: steering/, analysis/, docs/overview.md, docs/methodology.md, docs/viz_findings/, CLAUDE.md, .env.example
- docs/main.md has stale docs/other/ section (deleted)
- docs/remote_setup.md still says ./scripts/vast_setup.sh (now ./utils/vast_setup.sh)
- README.md says server/ but it's other/server/ (considering: move to server/ at root)
- 5 missing READMEs referenced in docs → create one-liner READMEs
- Visualization fetches docs/overview.md + docs/methodology.md at runtime — must be in .publicinclude
- Delete docs/README.md (duplicate of root README.md)

### Open decisions

- other/server/ stays in other/ (user doesn't want more root dirs — not necessary functionality)
- Should steering CLI tools (report, inspect, logit_diff) stay in dev/ or come back to steering/?
- When to tackle Waves 5-6 (architecture changes requiring deeper refactoring)?

### Testing

No new tests added during refactor. Verified after each wave:
- `pytest core/_tests/` — 106 passed, 10 skipped (all waves)
- Import checks: `python -c "import core; import utils; import extraction; import inference; import steering"` (all pass)
- Spot-checked specific imports after each move (steering_evaluate, steering_results, backends, etc.)
- Post-refactor TODO: add tests for refactored pipeline modules

### other/tv/ findings (fingerprinting)

tv/ has a fingerprinting concept (mean trait score per variant → behavioral profile) more complete than main codebase. Also has: diff_score (activation shift alignment), onset alignment (ERP-style), standalone plot.py. Duplicated hooks/math should import from main core/ instead. Analysis patterns worth promoting.

---

## Post-refactor TODO

Small things to deal with after the main refactor is done:
- Make different layers for different traits work more naturally — config-based or per-trait layer selection, not hardcoded
- Shorten steering_evaluate.py by offloading to core/utils helpers — then multi-layer could live within it
- Systematic experiment comparing local LLM judge quality vs gpt-4.1-mini
- Upload custom LoRAs to HuggingFace (rank32, rank1, etc.)
- Standardize config/models/*.yaml (proposed schema ready from audit)
- Trait category reorganization in datasets/traits/
- Visualization audit
- analysis/README.md is referenced in docs but doesn't exist
