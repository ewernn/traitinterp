---
title: Home
---

# traitinterp

Extract, monitor, and steer LLM behavioral traits token-by-token during generation.

**[Live demo](https://traitinterp.com/?tab=live-chat)** | **[GitHub](https://github.com/ewernn/traitinterp)**

---

## What this does

1. **Extract** -- Train a linear probe that detects a behavioral trait (sycophancy, deception, formality, etc.) from naturally contrasting scenarios
2. **Monitor** -- Project hidden states onto that probe token-by-token during generation
3. **Steer** -- Add the probe direction during inference to amplify or suppress the trait

Trait datasets are model-agnostic. Extract once, apply to any model.

---

## Quick start

```bash
git clone https://github.com/ewernn/traitinterp.git && cd traitinterp
pip install -r requirements.txt
export HF_TOKEN=your_token  # for gated models
```

Extract your first trait:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits starter_traits/sycophancy
```

Monitor traits during generation:

```bash
python inference/run_inference_pipeline.py \
    --experiment starter \
    --prompt-set starter_prompts/general
```

Visualize:

```bash
python visualization/serve.py  # http://localhost:8000
```

---

## Documentation

| Section | Description |
|---------|-------------|
| **Guides** | |
| [Extraction](../extraction_guide.md) | Extract trait vectors from contrasting scenarios |
| [Inference](../inference_guide.md) | Per-token monitoring, projection modes |
| [Steering](../steering_guide.md) | Causal validation via steering, coefficient search |
| [Creating Datasets](../trait_dataset_creation_base_model.md) | Scenario design, definitions, iteration |
| **CLI Reference** | |
| [Extraction CLI](cli/extraction.md) | `run_extraction_pipeline.py` flags and usage |
| [Inference CLI](cli/inference.md) | `run_inference_pipeline.py` flags and usage |
| [Steering CLI](cli/steering.md) | `run_steering_eval.py` flags and usage |
| [Analysis CLI](cli/analysis.md) | Analysis scripts flags and usage |
| **Configuration** | |
| [Experiment Setup](config/experiment-setup.md) | `config.json`, model variants, paths |
| [Trait Format](config/trait-format.md) | Dataset file format (`positive.txt`, `steering.json`, etc.) |
| **API Reference** | |
| [Core API](../core_reference.md) | Types, hooks, methods, math primitives |
| [Response Schema](../response_schema.md) | Unified response format across pipelines |
| [Chat Templates](../chat_templates.md) | HuggingFace chat template behavior |
| **Technical** | |
| [Architecture](../architecture.md) | Design principles, directory responsibilities, experiment schema |
| [Methodology](../methodology.md) | How we extract and use vectors |
