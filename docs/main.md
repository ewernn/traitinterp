# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

Primary documentation hub for the traitinterp project.

### Core Documentation
- **[docs/main.md](main.md)** (this file, dev-facing) — Project overview and codebase reference. A slim public version (`docs/main.main.md`) ships to main/prod under the same filename.
- **[docs/architecture.md](architecture.md)** - Design principles, directory responsibilities, experiment schema
- `README.md` - Quick start guide

### Pipelines
- **[docs/extraction_guide.md](extraction_guide.md)** - Extraction pipeline (scenarios → vectors → validation)
- **[docs/inference_guide.md](inference_guide.md)** - Inference pipeline (per-token monitoring, projection modes)
- **[docs/steering_guide.md](steering_guide.md)** - Steering pipeline (causal validation, coefficient search)
- **[docs/trait_dataset_creation_base_model.md](trait_dataset_creation_base_model.md)** - Creating base-model trait datasets via natural elicitation

### Inference & Steering
- **[docs/inference_guide.md](inference_guide.md)** - Inference pipeline guide (stream-through, from-activations, projection modes)
- **[docs/steering_guide.md](steering_guide.md)** - Steering pipeline guide (coefficient search, metrics, troubleshooting)
- `inference/README.md` - Per-token monitoring
- `analysis/README.md` - All analysis scripts (steering, model diff, vectors, benchmark)
- `steering/README.md` - Steering evaluation (detailed)

### Visualization
- `visualization/README.md` - Dashboard usage

### Technical Reference
- **[docs/core_reference.md](core_reference.md)** - core/ API (hooks, methods, math)
- **[docs/response_schema.md](response_schema.md)** - Unified response format across pipelines
- **[docs/chat_templates.md](chat_templates.md)** - HuggingFace chat template behavior
- `config/paths.yaml` - Path configuration
- `config/loras.yaml` - LoRA adapter registry (HF repos, custom models)

### Infrastructure (dev-only)
- **[docs/r2_sync.md](r2_sync.md)** - R2 cloud sync
- **[docs/other/serve_kimi_k2_1T.md](other/serve_kimi_k2_1T.md)** - Serving Kimi K2 (1T) across 8 GPUs

### Contributing
- **[docs/doc-update-guidelines.md](doc-update-guidelines.md)** - Style and process guide for docs

---

## Codebase Navigation

### Directory Structure
```
traitinterp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   ├── inference/
│   │   ├── starter_prompts/           # Public prompt sets (general.json)
│   │   └── archive/                   # Archived prompt sets
│   └── traits/
│       ├── starter_traits/            # Public traits (sycophancy, hallucination, concealment, etc.)
│       ├── emotion_set/              # 174 emotion traits
│       ├── ant_emotion_concepts/     # 174 emotion concept traits (single-polarity, long-context)
│       ├── alignment/                # 10 alignment traits
│       ├── tonal/                    # 7 tonal traits
│       ├── pv_instruction/           # Instruct-model traits (instruction-following axis)
│       ├── pv_natural/               # Instruct-model traits (natural conversation axis)
│       └── archive/                  # Archived trait sets
│   # Each trait dir: positive.{json,jsonl,txt}, negative.{json,jsonl,txt}, definition.txt, steering.json, extraction_config.yaml (optional)
│   ├── llm_judge/                     # Default judge prompts (edit to customize scoring)
│   │   ├── trait_score/               # Trait scoring (system + user prompts)
│   │   ├── coherence/                 # Coherence rubric
│   │   └── naturalness/               # Naturalness rubric
│
├── extraction/             # Vector extraction pipeline
│   └── run_extraction_pipeline.py     # Recipe: generate → vet → extract → evaluate
│
├── inference/              # Per-token monitoring
│   ├── generate_responses.py        # Generate/write response JSONs (standalone or called by pipeline)
│   └── run_inference_pipeline.py    # Recipe: generate → project (stream-through)
│
├── experiments/            # Experiment data (stored in R2, not git)
│   └── {experiment_name}/
│       ├── config.json               # Model variants
│       ├── extraction/               # Trait vectors (standard pipeline)
│       ├── inference/                # Per-token monitoring (standard pipeline)
│       ├── steering/                 # Causal intervention (standard pipeline)
│       ├── model_diff/               # Cross-variant comparison (standard pipeline)
│       └── {sub_experiment}/         # Self-contained investigation
│           ├── {sub_experiment}_notepad.md
│           ├── *.py
│           └── results/
│
├── config/
│   ├── paths.yaml                    # Single source of truth for paths
│   └── models/*.yaml                 # Model architecture configs
│
├── steering/              # Causal validation via steering
│   └── run_steering_eval.py            # Recipe: baseline → coefficient search → summary
│
├── core/                   # Primitives (types, hooks, methods, math)
│   └── _tests/                        # Unit tests (pytest core/_tests/)
├── utils/                  # Shared utilities
│   ├── model.py                      # Model loading, tokenization, prompt formatting
│   ├── model_generation.py            # Batch generation, activation capture
│   ├── vram.py                       # GPU monitoring, VRAM estimation, profiling, batch sizing
│   ├── moe.py                        # Fused MoE (INT4 dequant + grouped_mm), model cache
│   ├── distributed.py                # Tensor parallelism (is_tp_mode, tp_lifecycle, flush_cuda)
│   ├── positions.py                  # Position DSL (response[:5], turn[N]:thinking[:], etc.)
│   ├── batch_forward.py              # Shared helpers: OOM recovery, TP sync, batch calibration
│   ├── coefficient_search.py         # Adaptive steering coefficient search
│   ├── steering_results.py           # Load/compare steering results (I/O)
│   ├── extract_vectors.py            # Activation extraction + vector training
│   ├── capture_activations.py        # Capture raw activations to .pt (inference)
│   ├── project_activations.py        # Project activations onto trait vectors (inference)
│   └── ...                           # paths, activations, layers, projections, vectors, fingerprints
├── dev/                    # Holding pen — dev-only scripts, CLI tools, modal files
├── analysis/               # Analysis scripts (see analysis/README.md)
│   ├── vectors/                      # Vector quality, geometry, correlation, massive dims
│   ├── model_diff/                   # Cross-variant comparison (Cohen's d, per-token diff)
│   └── benchmark/                    # Benchmark evaluation with steering
├── visualization/          # Interactive dashboard
│   ├── serve.py                      # Python HTTP server (static + REST API + SSE chat)
│   ├── chat_inference.py             # Backend for live chat (model loading, streaming)
│   ├── index.html                    # SPA shell: sidebar nav, script loading, router
│   ├── styles.css                    # All CSS — design tokens, components, theme
│   ├── core/                         # Pure primitives, no DOM (state, charts, paths, ui, display, utils, massive-dims, annotations, citations, types, conversation-tree, chat-config, markdown-view)
│   ├── components/                   # Reusable UI widgets (sidebar, prompt-picker, top-spans, response-browser, styled-select/chip, inference-controls, live-chat-chart, chart-renderers, custom-blocks/{parser,renderers,loaders,index})
│   └── views/                        # One module per dashboard tab
│       ├── inference/                # Inference view (7 files: view, data, controls, 4 charts)
│       ├── steering/                 # Steering view (5 files: view, filters, overview, detail, shared)
│       ├── extraction/               # Extraction view (6 files: view, data, 4 sections — best-vectors, heatmaps, vector-geometry, logit-lens)
│       ├── model-analysis/           # Model-analysis view (4 files: view, data, 2 sections — diagnostics, variant-comparison)
│       ├── live-chat.js              # Multi-turn chat + 2 component files
│       └── ...                       # overview, findings
└── docs/                   # Documentation
```

### Key Entry Points

**Extract new traits:**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Monitor with existing vectors:**
```bash
# Run full inference pipeline (generate + stream-through project).
# First run also captures massive-dim calibration passively up to ~5000 tokens.
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# From saved activations instead of stream-through:
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --from-activations

# Override layers:
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --layers best
```

**Use core primitives:**
```python
from core import VectorSpec, ProjectionConfig, CaptureHook, SteeringHook, get_method, projection
```

**Analysis** (model diff, vectors, benchmark, steering): See `analysis/README.md`

---

## What This Does

1. **Extract trait vectors** from naturally contrasting scenarios
2. **Monitor traits** token-by-token during generation
3. **Validate vectors** via steering (causal intervention)

Natural elicitation avoids instruction-following confounds. See [docs/extraction_guide.md](extraction_guide.md).

---

## Quick Start

```bash
pip install -e .  # or: pip install -r requirements.txt
export HF_TOKEN=your_token_here  # For huggingface models

# Extract a trait
python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait}

# Visualize
python visualization/serve.py  # Visit http://localhost:8000/
```

See the pipeline guides: [extraction](extraction_guide.md), [inference](inference_guide.md), [steering](steering_guide.md).

**Shipping to the live site:** all work lands on `dev`. To deploy, run `./utils/release.sh -m "msg"` from `dev` — it promotes the whitelisted files to `main` (public release) and `prod`, and Railway auto-redeploys the site on every push to `origin/prod`. Full details, footguns, and the per-target scripts (`promote_to_main.sh` / `promote_to_prod.sh`) are in [Branch Workflow](#branch-workflow) below.

---

## Branch Workflow

Three branches: `dev` (everything), `main` (public release), `prod` (Railway deployment).

**How it works:**
- `dev` is the active working branch — all new code, experiments, and docs land here
- `main` contains only files whitelisted in `.publicinclude` — a curated subset for public release
- `prod` is the Railway-deployed branch. Railway watches `origin/prod` and rebuilds on every push. **Railway deploy config (start command, pre-deploy `mkdocs build`) lives in the Railway dashboard, not in the repo** — there is no `Procfile` / `railway.toml` / `nixpacks.toml`.
- Promotion is done via `./utils/release.sh -m "msg"`. Or directly via `utils/promote_to_main.sh` / `utils/promote_to_prod.sh`.
- Branches have **diverged histories** (not fast-forwardable) — each promote is a fresh commit.

**`.publicinclude`** lists what gets promoted to main: pipeline code, visualization, config, datasets, and select docs.
**`.prodinclude`** lists what gets promoted to prod: everything in main plus `docs/viz_findings/` and the MkDocs site sources.

**Branch-specific overrides:** any file named `<name>.main.md` on dev is renamed to `<name>.md` by both promote scripts during copy. The promote scripts glob for `*.main.md` recursively — no hardcoded list. Used today for:
- `CLAUDE.main.md` → `CLAUDE.md` — public version of the Claude Code instructions (imports `@docs/main.md`, which is itself renamed)
- `docs/main.main.md` → `docs/main.md` — public version of this file (slim quick-start + conventions; the rich version you're reading is dev-only)

**What stays dev-only:**
- `dev/` directory — holding pen for steering CLI tools, modal files, dev-only scripts
- `other/` — server, tv, sae, mcp, rm_sycophancy analysis
- Research docs — refactor notepads, TODO
- Personal docs live in a separate `trait-interp-personal` repo

### The `release.sh` workflow

```bash
# Work normally — dev can have messy WIP
git add <files-to-ship>
git commit -m "..."
./utils/release.sh -m "release msg"
```

`release.sh` does, in order:
1. Verifies you're on dev
2. Errors loudly if any path listed *explicitly as a file* in `.publicinclude` / `.prodinclude` is untracked on dev (the footgun catch — see below)
3. Auto-stashes uncommitted changes (with `-u`, so untracked WIP is preserved)
4. `git push origin dev`
5. `promote_to_main.sh --push` → `origin/main`
6. `promote_to_prod.sh --push` → `origin/prod` → Railway auto-redeploys
7. Pops the stash (on success OR failure; errors loudly with the stash ref if a conflict arises)
8. Prints a per-target summary so half-shipped state is obvious

No `--no-verify` escape hatch: test failures must be fixed, not bypassed.

### Footguns you must know about

**Silent-skip on untracked files.** The promote scripts use `git checkout dev -- <path>` which silently fails on untracked files. A path can sit in `.publicinclude`/`.prodinclude` and never actually ship. `release.sh` catches this for **explicitly-listed file paths** (like `mkdocs.yml`) but **does NOT flag untracked WIP inside glob-directories** (like a new markdown file inside `docs/viz_findings/`) — that's intentional, because glob-dirs often contain in-progress content.

**Consequence:** if you add a new file that belongs to an already-whitelisted directory and forget to `git add` it, it won't ship and release.sh won't warn. Mitigation: before releasing findings/assets, skim `git status` for anything that should be part of the release, or comment the orphan out of `docs/viz_findings/index.yaml` if it's genuinely WIP.

**MkDocs build writes to `docs_site/`.** This is in `.gitignore`. Don't remove that line — `promote_to_prod.sh` runs `mkdocs build` as part of pre-promote checks, and if `docs_site/` is tracked, the build will pollute the working tree and break `git stash pop` at the end of `release.sh`.

### Adding files to main/prod

`git add` them on dev → list in `.publicinclude` / `.prodinclude` → commit → `./utils/release.sh -m "..."`.
