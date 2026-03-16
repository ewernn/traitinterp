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
5. **analysis/** = Interpretation + aggregation (thresholds, cross-prompt patterns)
6. **experiments/** = Data storage + experiment-specific scripts
7. **visualization/** = All visualization code and views

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
- Hook management (`HookManager`, `CaptureHook`, `SteeringHook`, `AblationHook`)
- Extraction methods (`MeanDifferenceMethod`, `ProbeMethod`, `GradientMethod`)
- Math primitives (`projection`, `cosine_similarity`, `separation`, `accuracy`)

**Current exports:**
```python
# hooks.py
HookManager, get_hook_path, CaptureHook, SteeringHook, AblationHook, MultiLayerCapture, MultiLayerAblationHook

# methods.py
get_method, MeanDifferenceMethod, ProbeMethod, GradientMethod, RandomBaselineMethod

# math.py
projection, cosine_similarity, separation, accuracy, effect_size, p_value
```

**What does NOT belong:**
- Model-specific code
- Threshold/heuristic-based analysis
- Visualization code

### utils/ - Universal Utilities

**What belongs:**
- Model loading, tokenization, prompt formatting (`utils/model.py`)
- Batch generation, activation capture (`utils/generation.py`)
- GPU monitoring, VRAM estimation, batch sizing (`utils/vram.py`)
- Fused MoE and model cache (`utils/moe.py`)
- Tensor parallelism utilities (`utils/distributed.py`)
- Path management (`utils/paths.py` — loads from config/paths.yaml)
- Activation loading (`utils/activations.py` — auto-detects stacked vs per-layer format)
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
