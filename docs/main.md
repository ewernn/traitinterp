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
- **[trait_dataset_creation.md](trait_dataset_creation.md)** — creating trait datasets

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
# 1. Calibrate massive dims (once per experiment)
python analysis/vectors/massive_activations.py --experiment {experiment}

# 2. Run inference pipeline
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

**Use core primitives:**
```python
from core import VectorSpec, ProjectionConfig, CaptureHook, SteeringHook, get_method, projection
```
