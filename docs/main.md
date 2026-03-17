# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

Primary documentation hub for the trait-interp project.

### Core Documentation
- **[docs/main.md](main.md)** (this file) - Project overview and codebase reference
- **[docs/experiment_structure.md](experiment_structure.md)** - Experiment directory schema, sub-experiments, notepad conventions, R2 sync
- **[docs/workflows.md](workflows.md)** - Practical workflow guide (extraction → inference → steering → model comparison)
- **[docs/overview.md](overview.md)** - Methodology, key learnings, notable experiments (serves /overview in frontend)
- **[docs/architecture.md](architecture.md)** - Design principles and organization
- **[README.md](../readme.md)** - Quick start guide

### Pipeline & Extraction
- **[docs/extraction_guide.md](extraction_guide.md)** - Complete extraction reference (scenarios → vectors → validation)
- **[docs/scenario_design_guide.md](scenario_design_guide.md)** - Writing contrasting scenario pairs for good vectors
- **[docs/trait_dataset_creation.md](trait_dataset_creation.md)** - Creating trait datasets (decision tree, scenario design, iteration)
- **[docs/judge_definition_iteration.md](judge_definition_iteration.md)** - Iterating judge definitions until accurate
- **[docs/trait_catalog.md](trait_catalog.md)** - Auto-generated inventory of all traits by category

### Inference & Steering
- **[inference/README.md](../inference/README.md)** - Per-token monitoring
- **[analysis/README.md](../analysis/README.md)** - All analysis scripts (steering, model diff, vectors, benchmark)
- **[steering/README.md](../steering/README.md)** - Steering evaluation (detailed)

### Visualization
- **[visualization/README.md](../visualization/README.md)** - Dashboard usage
- **[docs/methodology.md](methodology.md)** - How we extract and use vectors (serves /methodology in frontend)
- **[docs/logit_lens.md](logit_lens.md)** - Prediction evolution across layers

### Technical Reference
- **[docs/core_reference.md](core_reference.md)** - core/ API (hooks, methods, math)
- **[docs/response_schema.md](response_schema.md)** - Unified response format across pipelines
- **[docs/chat_templates.md](chat_templates.md)** - HuggingFace chat template behavior
- **[config/paths.yaml](../config/paths.yaml)** - Path configuration
- **[config/loras.yaml](../config/loras.yaml)** - LoRA adapter registry (HF repos, custom models)

### Infrastructure
- **[docs/remote_setup.md](remote_setup.md)** - Remote GPU setup
- **[docs/r2_sync.md](r2_sync.md)** - R2 cloud sync
- **[docs/tensor_parallel.md](tensor_parallel.md)** - Tensor parallelism for DeepSeek V3 / Kimi K2

### Contributing
- **[docs/doc-update-guidelines.md](doc-update-guidelines.md)** - Style and process guide for docs

### Dev-only Docs (not promoted to main)
These live on the `dev` branch only. See [Branch Workflow](#branch-workflow) below.
- **[docs/codebase_refactor_notepad.md](codebase_refactor_notepad.md)** - Refactor tracking (waves 1-8, thin controller, known issues)
- **[docs/future_ideas.md](future_ideas.md)** - Research backlog and exploratory ideas
- **[docs/viz_findings/](viz_findings/)** - Research findings served by the visualization dashboard

---

## Codebase Navigation

### Directory Structure
```
trait-interp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   ├── inference/                     # Prompt sets (harmful.json, jailbreak/original.json, etc.)
│   └── traits/{category}/{trait}/     # Trait definitions
│       ├── positive.txt, negative.txt # Contrasting scenarios
│       ├── definition.txt             # Trait description
│       └── steering.json              # Steering eval questions
│
├── extraction/             # Vector extraction pipeline
│   └── run_extraction_pipeline.py     # Recipe: generate → vet → extract → evaluate
│
├── inference/              # Per-token monitoring
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
│   ├── generation.py                 # Batch generation, activation capture
│   ├── vram.py                       # GPU monitoring, VRAM estimation, batch sizing
│   ├── moe.py                        # Fused MoE (INT4 dequant + grouped_mm), model cache
│   ├── distributed.py                # Tensor parallelism (is_tp_mode, rank, barrier)
│   ├── fingerprints.py               # Fingerprint metrics (cosine, classification, short_name)
│   ├── coefficient_search.py         # Adaptive steering coefficient search
│   ├── steering_results.py           # Load/compare steering results (I/O)
│   ├── extract_vectors.py            # Activation extraction + vector training
│   ├── process_activations.py        # Capture/project activations (inference)
│   └── ...                           # paths, activations, layers, projections, vectors
├── dev/                    # Holding pen — dev-only scripts, CLI tools, modal files
├── other/                  # server/, tv/, sae/, mcp/, analysis/rm_sycophancy/
├── analysis/               # Analysis scripts (see analysis/README.md)
├── visualization/          # Interactive dashboard
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
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {experiment}

# 2. Run full inference pipeline (generate + stream-through project)
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

**Analysis** (model diff, vectors, benchmark, steering): See [analysis/README.md](../analysis/README.md)

---

## What This Does

1. **Extract trait vectors** from naturally contrasting scenarios
2. **Monitor traits** token-by-token during generation
3. **Validate vectors** via steering (causal intervention)

Natural elicitation avoids instruction-following confounds. See [docs/extraction_guide.md](extraction_guide.md).

---

## Quick Start

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here  # For huggingface models

# Extract a trait
python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait}

# Visualize
python visualization/serve.py  # Visit http://localhost:8000/
```

For full command reference, see [docs/workflows.md](workflows.md).

---

## Branch Workflow

Development happens on `dev`. The `main` branch is a curated public-facing subset.

**How it works:**
- `dev` is the active working branch — all new code, experiments, and docs land here
- `main` contains only files whitelisted in `.publicinclude` — a curated subset for public release
- Promotion is done via `utils/promote_to_main.sh`, which copies whitelisted files from dev to main with a clean commit history
- The two branches have **diverged histories** (not fast-forwardable) — main has squashed/rewritten commits

**`.publicinclude`** lists what gets promoted: all pipeline code (`core/`, `extraction/`, `inference/`, `steering/`, `analysis/`, `utils/`), visualization, config, datasets, and select docs. Dev-only content (`dev/`, `other/`, research notepads, experiment-specific docs) stays on `dev`.

**What stays dev-only:**
- `dev/` directory — holding pen for steering CLI tools, modal files, dev-only scripts
- `other/` — server, tv, sae, mcp, rm_sycophancy analysis
- Research docs — refactor notepad, future ideas, steering reviews, experiment audits (listed in [Dev-only Docs](#dev-only-docs-not-promoted-to-main) above)

**To promote new files:** Add paths to `.publicinclude`, then run `utils/promote_to_main.sh`.
