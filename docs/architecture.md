# Repository Architecture

## Design Principles

### Core Stack

```
core/               → General primitives (hooks, math, methods)
        ↓
extraction/         → Build trait vectors (training time)
        ↓
inference/          → Compute facts per prompt (capture, project)
        ↓
analysis/           → Interpret + aggregate (thresholds, cross-prompt)
        ↓
visualization/      → Show everything
```

### Directory Responsibilities

1. **core/** = General-purpose primitives (hooks, math, extraction methods)
2. **utils/** = Universal utilities (paths, model loading)
3. **extraction/** = Vector creation pipeline (training time)
4. **inference/** = Per-prompt computation (capture, project)
5. **steering/** = Causal validation via steering evaluation
6. **analysis/** = Interpretation + aggregation (thresholds, cross-prompt patterns)
7. **experiments/** = Data storage + experiment-specific scripts
8. **visualization/** = All visualization code and views

---

## inference/ vs analysis/ Distinction

**Key principle: facts vs interpretation.**

### inference/ = "What are the numbers?"

Computes facts about a single prompt. No thresholds, no heuristics.

| Computation | Why inference/ |
|-------------|----------------|
| Raw activations | Direct capture |
| Trait scores | Direct projection |
| Attention patterns | Direct from model |

### analysis/ = "What do the numbers mean?"

Interprets facts, applies thresholds, aggregates across prompts.

| Type | Why analysis/ |
|------|---------------|
| Threshold-based detection | Needs heuristics |
| Cross-prompt aggregation | Compares multiple prompts |
| Pattern interpretation | Goes beyond raw numbers |

---

## Three-Phase Pipeline

### Phase 1: extraction/
**Purpose:** Build trait vectors (training time)
- Natural scenario files
- Generated contrastive responses
- Extracted activations
- Computed trait vectors

### Phase 2: inference/
**Purpose:** Compute facts per prompt (inference time)
- Capture raw activations
- Project onto trait vectors

### Phase 3: analysis/
**Purpose:** Interpret facts and aggregate
- Apply thresholds and heuristics
- Aggregate across prompts

---

## What Goes Where

### core/ - General Primitives

**What belongs:**
- Hook management (`HookManager`, `CaptureHook`, `SteeringHook`, `AblationHook`, `ProjectionHook`, etc.)
- Extraction methods (`MeanDifferenceMethod`, `ProbeMethod`, `GradientMethod`, `RFMMethod`)
- Math primitives (`projection`, `cosine_similarity`, `accuracy`, `effect_size`, `orthogonalize`, etc.)

**Current exports:**
```python
# hooks.py
HookManager, get_hook_path, LayerHook, CaptureHook, SteeringHook, AblationHook,
MultiLayerCapture, MultiLayerAblation, MultiLayerSteering,
ProjectionHook, MultiLayerProjection,
ActivationCappingHook, MultiLayerActivationCapping, PerSampleSteering,
PerPositionSteeringHook

# methods.py
get_method, MeanDifferenceMethod, ProbeMethod, GradientMethod, RandomBaselineMethod, RFMMethod

# math.py
projection, cosine_similarity, batch_cosine_similarity,
accuracy, effect_size, orthogonalize, polarity_correct,
remove_massive_dims, pairwise_cosine_matrix, pca, project_out_subspace
```

**What does NOT belong:**
- Model-specific code
- Threshold/heuristic-based analysis
- Visualization code

### utils/ - Universal Utilities

**What belongs:**
- Model loading, tokenization, prompt formatting (`utils/model.py`)
- Batch generation, activation capture (`utils/model_generation.py`)
- GPU monitoring, VRAM estimation, batch sizing (`utils/vram.py`)
- Fused MoE and model cache (`utils/moe.py`)
- Tensor parallelism utilities (`utils/distributed.py`)
- Path management (`utils/paths.py` — loads from config/paths.yaml)
- Activation loading (`utils/load_activations.py` — auto-detects stacked vs per-layer, returns `ActivationMetadata` from metadata.json)
- Layer parsing (`utils/layers.py` — shared `parse_layers` for all layer specification strings)
- Projection reading (`utils/projections.py` — handles single-vector and multi-vector formats, activation-norm normalization)
- Fingerprint utilities (`utils/fingerprints.py` — cosine similarity, classification, score loading for analysis scripts)
- Functions needed across all modules

### extraction/ - Vector Creation Pipeline

**What belongs:**
- Training-time pipeline scripts
- Natural scenario handling
- Activation extraction from training data

### inference/ - Per-Prompt Computation

**What belongs:**
- Capture raw activations from model runs
- Project activations onto trait vectors
- Anything that produces "facts" about one prompt

**What does NOT belong:**
- Threshold-based detection (goes in analysis/)
- Cross-prompt aggregation (goes in analysis/)

### analysis/ - Interpretation + Aggregation

**What belongs:**
- Anything that applies thresholds or heuristics
- Anything that aggregates across multiple prompts

### experiments/ - Data Storage

**What belongs:**
- Experimental data (responses, activations, vectors)
- Custom analysis scripts
- Experiment-specific monitoring code

#### Experiment Directory Schema

Each experiment lives in `experiments/{name}/` and follows this layout:

```
experiments/{name}/
├── config.json                        # Model variants and experiment settings
│
├── extraction/                        # Trait vectors (standard pipeline)
│   └── {category}/{trait}/{model_variant}/
│       ├── responses/pos.json, neg.json
│       ├── vectors/{position}/{component}/{method}/layer*.pt
│       └── vetting/response_scores.json
│
├── inference/                         # Per-token monitoring (standard pipeline)
│   └── {model_variant}/
│       ├── responses/{prompt_set}/{prompt_id}.json
│       ├── projections/{trait}/{prompt_set}/{prompt_id}.json
│       └── massive_activations/{prompt_set}.json
│
├── steering/                          # Causal intervention (standard pipeline)
│   └── {category}/{trait}/{model_variant}/{position}/{prompt_set}/{direction}/
│       ├── results.jsonl
│       └── responses/{component}/{method}/
│
├── model_diff/                        # Cross-variant comparison (standard pipeline)
│   └── {variant_a}_vs_{variant_b}/{prompt_set}/
│       ├── diff_vectors.pt
│       └── results.json
│
└── {sub_experiment}/                  # Self-contained investigation (any number)
    ├── {sub_experiment}_notepad.md    # Timestamped research notes
    ├── *.py                           # Scripts for this sub-experiment
    └── results/                       # Outputs
```

**Standard pipeline dirs** (`extraction/`, `inference/`, `steering/`, `model_diff/`) are produced by shared pipeline code and consumed by the visualization dashboard. Their structure is defined in `config/paths.yaml`.

#### Sub-experiments

Any directory that isn't a standard pipeline dir is a sub-experiment. Each is self-contained:

- `{name}_notepad.md` — timestamped entries documenting the investigation. Optional plan section at top.
- Scripts live alongside the notepad, not in the experiment root.
- Results/outputs go in `results/` or alongside the scripts.

Sub-experiments consume data from the standard pipeline dirs (e.g., reading vectors from `extraction/`, projections from `inference/`) but store their own outputs internally.

#### Experiment Scripts as Recipes

Experiment scripts are **thin orchestration layers** that call pipeline code — they never reimplement it. The pipeline does the heavy lifting; the script expresses the experiment's unique logic.

**What belongs in an experiment script:**
- Dataset constants (prompt templates, baseline numbers from a paper)
- Experiment-specific analysis (what to correlate, how to interpret results, grading functions)
- Orchestration (which stages run in what order, decision gates)

**What does NOT belong in an experiment script:**
- Forward passes (`CaptureHook` loops, `model(**inputs)`) — use a shared capture helper
- Steering loops (`SteeringHook` + `generate_batch`) — use a shared sweep helper
- Path construction (`get_path(...) / 'extraction' / ...`) — use `utils/paths.py` or `utils/vectors.py`
- JSON serialization — use a shared save helper

**When the pipeline has a gap:** Add the capability to `core/` or `utils/` as a flag or new function, then call it from the experiment. Example: needing position-specific steering → add `PerPositionSteeringHook` to `core/hooks.py`, not a custom hook in the experiment script. The pipeline grows to serve experiments.

**The `shared.py` pattern:** Each experiment with multiple stage scripts gets a `shared.py` that bridges between the generic pipeline and the experiment's specific needs. `shared.py` itself calls pipeline functions — it never reimplements them. Stage scripts import from `shared.py` for experiment-level utilities (vector loading, results I/O, scenario grading).

**Notepad format** — entries use UTC timestamps:

```markdown
# {Sub-experiment Name} Notepad

## Plan
Optional high-level plan here.

---

## 14:30 UTC — Title of entry

Content, findings, next steps.

## 15:45 UTC — Another entry

More content.
```

#### R2 Sync

All experiment data is stored in R2. Git does not track `experiments/`. See [docs/r2_sync.md](r2_sync.md) for sync commands and configuration.

#### LoRA Management

LoRA adapters are registered in `config/loras.yaml` with HuggingFace repo IDs. Scripts load LoRAs from HF at runtime — they are not stored locally or in R2 long-term.

#### Archive

`experiments/archive/` holds inactive experiments. Excluded from R2 sync by default (`--include-archive` to opt in). Same directory structure as active experiments.

---

## Adding New Code - Decision Tree

**Q: Is it a mathematical primitive (no thresholds, works on any tensor)?**
→ `core/`

**Q: Is it part of building trait vectors?**
→ `extraction/`

**Q: Does it compute facts about a single prompt (no thresholds)?**
→ `inference/`

**Q: Does it interpret facts or aggregate across prompts?**
→ `analysis/`

**Q: Is it a universal utility (paths, config)?**
→ `utils/`

**Otherwise:**
→ `experiments/{name}/`

---

## Dependencies

| Module | Allowed | Never |
|--------|---------|-------|
| core/ | PyTorch, scikit-learn | transformers, viz packages |
| utils/ | PyYAML | experiment-specific |
| extraction/ | transformers, core/, pandas | viz packages |
| inference/ | core/, transformers | viz packages |
| analysis/ | core/, numpy, scipy | - |
| experiments/ | anything | - |

---

## Visualization Architecture

The dashboard is a single-page app served by a Python HTTP server (`visualization/serve.py`). No build step — ES modules load directly in the browser. See [visualization/README.md](../visualization/README.md) for the full reference.

### Layer Responsibilities

- **core/** — Shared primitives that views and components depend on. Pure functions returning HTML strings or data, no DOM side effects. Includes state management (`state.js`), chart layout building (`charts.js`), path resolution (`paths.js`), UI helpers (`ui.js`), and display utilities (`display.js`).
- **components/** — Reusable UI pieces that own a specific DOM region (sidebar, prompt picker, response browser). They read from `window.state` and render into their container. Components may be shared across multiple views.
- **views/** — One module per dashboard tab. Each exports a `renderX()` function that writes to `#content-area`. Complex views split into sub-modules inside a folder: `inference/` holds the Inference tab's orchestrator plus data loading, controls, and chart modules; `steering/` holds the Steering tab's orchestrator plus filters, overview grid, detail panel, and shared helpers.

### Key Patterns

- **Chart presets**: `charts.js` defines preset layouts (`timeSeries`, `layerChart`, `heatmap`, `barChart`) with `buildChartLayout()` merging preset defaults, legend sizing, and caller overrides. All Plotly charts go through this for consistent theming.
- **Deferred loading**: `ui.deferredLoading()` shows a loading spinner only after a short delay, avoiding flashes for fast responses. Views call it before async fetches and cancel when data arrives.
- **State management**: A single `state` object in `state.js` holds all global state. User preferences are persisted to `localStorage` via a table-driven preference system — declare type, default, validation, and optional `onSet` callback. No framework; views re-render by calling `window.renderView()`.
- **Path resolution**: `paths.js` mirrors `utils/paths.py` — both load from `config/paths.yaml`. Frontend paths resolve relative to the experiment; the server serves experiment data as static files.
- **ES modules with `window.*` bridge**: All code uses `import`/`export`. The router and HTML `onclick` handlers need `window.*` access, so modules bridge specific functions. Everything else uses direct imports.

---

## Clean Repo Checklist

- [ ] No circular dependencies
- [ ] Each directory has single responsibility
- [ ] core/ is model-agnostic (no thresholds, no heuristics)
- [ ] utils/ has no experiment code
- [ ] extraction/ has no inference code
- [ ] inference/ computes facts (no thresholds/heuristics)
- [ ] analysis/ interprets (thresholds, aggregation OK)
- [ ] experiments/ has no reusable library code
- [ ] Clear separation: extraction → inference → analysis
