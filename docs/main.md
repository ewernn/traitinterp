# traitinterp

Extract, monitor, and steer LLM behavioral traits token-by-token during generation.

---

## What This Does

1. **Extract trait vectors** from naturally contrasting scenarios
2. **Monitor traits** token-by-token during generation
3. **Validate vectors** via steering (causal intervention)

Natural elicitation avoids instruction-following confounds. See [extraction_guide.md](extraction_guide.md).

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

---

## Documentation

### Pipelines
- **[extraction_guide.md](extraction_guide.md)** — scenarios → vectors → validation
- **[inference_guide.md](inference_guide.md)** — per-token monitoring, projection modes
- **[steering_guide.md](steering_guide.md)** — causal validation, coefficient search

### Technical Reference
- **[architecture.md](architecture.md)** — design principles, directory responsibilities, experiment schema
- **[core_reference.md](core_reference.md)** — `core/` API (hooks, methods, math)
- **[response_schema.md](response_schema.md)** — unified response format across pipelines
- **[chat_templates.md](chat_templates.md)** — HuggingFace chat template behavior
- **[methodology.md](methodology.md)** — how we extract and use vectors

### READMEs
- `README.md` — quick start guide
- `visualization/README.md` — dashboard usage
- `inference/README.md`, `steering/README.md`, `analysis/README.md` — pipeline-specific

---

## Key Entry Points

**Extract new traits:**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Monitor with existing vectors:**
```bash
# Run inference pipeline — massive-dim calibration happens passively on first run.
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

**Use core primitives:**
```python
from core import VectorSpec, ProjectionConfig, CaptureHook, SteeringHook, get_method, projection
```

---

## Contributing Conventions

If you're extending this repo (or using it as a Claude Code workspace), a few
conventions that keep the codebase consistent:

**No hardcoding.** Paths, experiment names, and trait names should flow through
variables and config — never literal values buried in scripts. All paths go
through `utils/paths.py` (Python) or `visualization/core/paths.js` (JS), which
read from `config/paths.yaml`. Fail fast with clear errors rather than silently
papering over missing inputs.

**Code style.**
- Module docstrings: one-line description + `Input:`, `Output:`, `Usage:` sections.
- Function docstrings only when behavior isn't obvious from the signature.
- Prefer long, descriptive file names (`trait_annotation_correlation.py` over
  `correlation.py`) — grep-friendly beats terse.
- Script filename should match its output filename when applicable.

**Naming.** A reader at a call site should understand what happens without
opening the function. Too vague (`projection()`) hides behavior; too specific
(`project_onto_unit_vector()`) breaks when args are added. Describe the core
operation at the right abstraction level, let parameters carry the variations.
Same applies to classes, variables, and file names. If naming is hard, the
function is probably doing too many things.

**Standards.**
- Single source of truth for paths (`utils/paths.py`, `visualization/core/paths.js`).
- Experiment-agnostic scripts — take `--experiment` rather than hardcoding.
- Delete legacy code rather than leaving it commented out.
- Fail fast on missing config, malformed data, wrong tensor shapes.

**Visualization.** Reuse primitives from `visualization/styles.css` — don't
introduce ad-hoc colors, spacing, or component styles. Reuse existing view
and component modules where possible.

**Writing style** (docs, findings, methodology):
- Concise natural prose; explain concepts simply before technical details.
- Bullet points freely; avoid jargon where plain language works.
- First-person plural ("we") for actions.
- Assume ML basics (probes, steering, activations) but write for a broader
  technical audience.
